#!/usr/bin/env python3
"""Tests for the dense_ann extension. Run: make -C ext/dense test"""
import json
import os
import sqlite3
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXT = os.path.join(HERE, "..", "build", "dense_ann")
COUNTVFS = os.path.join(HERE, "..", "build", "countvfs")


def data(n=3000, nq=40, dim=384, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((40, dim)).astype(np.float32)
    A = rng.standard_normal((32, dim)).astype(np.float32) / np.sqrt(dim)

    def gen(m):
        x = centers[rng.integers(0, 40, m)] / np.sqrt(dim) + rng.standard_normal((m, 32)).astype(np.float32) @ A * 0.8
        return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)
    return gen(n), gen(nq)


def exact(X, q, k):
    return list(np.argsort(-(X @ q))[:k] + 1)


class DenseAnnTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, cls.Q = data()
        cls.tmp = tempfile.mkdtemp()

    def open(self, name, page_size=4096, params="", build=True, finalize=True, n=None):
        path = os.path.join(self.tmp, name)
        if os.path.exists(path):
            os.remove(path)
        db = self.connect(path)
        db.execute(f"PRAGMA page_size={page_size}")
        db.execute(f"CREATE VIRTUAL TABLE v USING dense_ann(dim=384, M=12, ef_construction=100{', ' + params if params else ''})")
        X = self.X[:n] if n else self.X
        db.execute("BEGIN")
        db.executemany("INSERT INTO v(rowid, embedding) VALUES (?, ?)", ((i + 1, x.tobytes()) for i, x in enumerate(X)))
        db.execute("COMMIT")
        if build:
            db.execute("INSERT INTO v(v) VALUES ('build')")
        if finalize:
            db.execute("INSERT INTO v(v) VALUES ('finalize')")
        return db, path

    @staticmethod
    def connect(path):
        db = sqlite3.connect(path, isolation_level=None)
        db.enable_load_extension(True)
        db.load_extension(EXT)
        return db

    def search(self, db, q, extra="", k=10):
        rows = db.execute(f"SELECT rowid, distance, stats FROM v WHERE embedding MATCH ? AND k = {k} {extra}",
                          (q.tobytes(),)).fetchall()
        return [r[0] for r in rows], [r[1] for r in rows], (json.loads(rows[0][2]) if rows else {})

    def recall(self, db, extra="", X=None):
        X = self.X if X is None else X
        return np.mean([len(set(self.search(db, q, extra)[0]) & set(exact(X, q, 10))) / 10 for q in self.Q])

    def test_exact_matches_numpy(self):
        db, _ = self.open("exact.db", params="store_vectors=f32")
        for q in self.Q[:10]:
            ids, dist, _ = self.search(db, q, "AND exact = 1")
            self.assertEqual(ids, exact(self.X, q, 10))
            self.assertAlmostEqual(dist[0], 1 - float(self.X[ids[0] - 1] @ q), places=5)

    def test_recall_and_hints(self):
        db, path = self.open("basic.db", params="vectors=inline")
        r = self.recall(db, "AND ef = 64 AND rerank = 2")
        self.assertGreater(r, 0.9)
        _, _, st = self.search(db, self.Q[0], "AND rerank = 2")
        self.assertEqual(st["raw"], 1)           # page hints in use
        self.assertEqual(st["fallback"], 0)      # and all valid
        self.assertGreater(st["rounds"], 0)
        # Same results through a fresh connection, and through the SQL path.
        db2 = self.connect(path)
        a = self.search(db2, self.Q[1], "AND rerank = 2")[0]
        db2.execute("BEGIN")                     # explicit txn disables direct reads
        b, _, st2 = self.search(db2, self.Q[1], "AND rerank = 2")
        db2.execute("COMMIT")
        self.assertEqual(st2["raw"], 0)
        self.assertEqual(a, b)

    def test_rerank_improves(self):
        db, _ = self.open("rerank.db")
        self.assertGreater(self.recall(db, "AND ef = 64 AND rerank = 1"), self.recall(db, "AND ef = 64 AND rerank = 0"))

    def test_beam_reduces_rounds(self):
        db, _ = self.open("beam.db")
        r1 = self.search(db, self.Q[0], "AND ef = 64 AND beam = 1")[2]["rounds"]
        r8 = self.search(db, self.Q[0], "AND ef = 64 AND beam = 8")[2]["rounds"]
        self.assertLess(r8, r1 / 3)

    def test_separate_layout(self):
        db, _ = self.open("sep.db", params="layout=separate")
        self.assertGreater(self.recall(db, "AND ef = 64 AND rerank = 1"), 0.85)
        st = self.search(db, self.Q[0], "AND ef = 64 AND beam = 4")[2]
        self.assertEqual(st["fallback"], 0)
        # Up to two dependent rounds per step (node rows, then neighbour codes;
        # the second is skipped when all code pages are already cached), and
        # extra pages for the codes.
        self.assertGreater(st["rounds"], st["expanded"] // 4 + 2)
        self.assertGreater(st["pages"], st["expanded"])

    def test_vector_types(self):
        for vt in ("none", "int8", "f16", "f32"):
            db, _ = self.open(f"vt-{vt}.db", params=f"store_vectors={vt}", n=1000)
            ids = self.search(db, self.X[5], "AND rerank = 1")[0]
            self.assertEqual(ids[0], 6, vt)
            emb = db.execute("SELECT embedding FROM v WHERE rowid = 6").fetchone()[0]
            if vt == "none":
                self.assertIsNone(emb)
            else:
                e = np.frombuffer(emb, np.float32)
                self.assertGreater(float(e @ self.X[5]) / np.linalg.norm(e), 0.99)

    def test_json_and_scan(self):
        db, _ = self.open("json.db", n=500)
        ids = self.search(db, self.X[3])[0]
        q = json.dumps([float(v) for v in self.X[3]])
        ids2 = [r[0] for r in db.execute("SELECT rowid FROM v WHERE embedding MATCH ? AND k = 10", (q,))]
        self.assertEqual(ids, ids2)
        self.assertEqual(db.execute("SELECT count(*) FROM v").fetchone()[0], 500)
        # LIMIT works as k.
        ids3 = [r[0] for r in db.execute("SELECT rowid FROM v WHERE embedding MATCH ? LIMIT 3", (self.X[3].tobytes(),))]
        self.assertEqual(ids3, ids[:3])

    def test_autobuild_on_query(self):
        db, _ = self.open("auto.db", build=False, finalize=False, n=800)
        ids = self.search(db, self.X[10])[0]
        self.assertEqual(ids[0], 11)
        self.assertEqual(db.execute("SELECT value FROM v_config WHERE key='built'").fetchone()[0], 1)

    def test_incremental_insert_and_delete(self):
        db, _ = self.open("incr.db", n=2000, params="vectors=inline")
        new = self.X[2000:2300]
        db.execute("BEGIN")
        for i, x in enumerate(new):
            db.execute("INSERT INTO v(rowid, embedding) VALUES (?, ?)", (100000 + i, x.tobytes()))
        db.execute("COMMIT")
        hits = sum(self.search(db, x, "AND rerank = 2")[0][:1] == [100000 + i] for i, x in enumerate(new))
        self.assertGreater(hits, 290)
        # Recall over old and new points together.
        allX = np.vstack([self.X[:2000], new])
        ids_map = list(range(1, 2001)) + list(range(100000, 100300))
        rec = np.mean([len(set(self.search(db, q, "AND ef = 64 AND rerank = 2")[0])
                           & {ids_map[j] for j in np.argsort(-(allX @ q))[:10]}) / 10 for q in self.Q])
        self.assertGreater(rec, 0.9)
        # Delete: gone from results and from the table.
        db.execute("DELETE FROM v WHERE rowid = 100007")
        self.assertNotIn(100007, self.search(db, new[7], "AND rerank = 2")[0])
        self.assertIsNone(db.execute("SELECT rowid FROM v WHERE rowid = 100007").fetchone())
        with self.assertRaises(sqlite3.Error):
            db.execute("INSERT INTO v(rowid, embedding) VALUES (5, ?)", (new[0].tobytes(),))

    def test_page_size_64k_packing(self):
        db, _ = self.open("p64k.db", page_size=65536, params="vectors=inline")
        st = self.search(db, self.Q[0], "AND ef = 64 AND rerank = 2")[2]
        self.assertEqual(st["fallback"], 0)
        self.assertLess(st["pages"], st["expanded"])   # several expanded nodes per page
        self.assertGreater(self.recall(db, "AND ef = 64 AND rerank = 2"), 0.9)

    def test_finalize_after_vacuum(self):
        db, path = self.open("vac.db", finalize=False, n=1500)
        db.execute("VACUUM")
        db.execute("INSERT INTO v(v) VALUES ('finalize')")
        st = self.search(db, self.X[0])[2]
        self.assertEqual((st["raw"], st["fallback"]), (1, 0))

    def test_cold_query_reads(self):
        _, path = self.open("cold.db", n=1500, params="vectors=inline")
        boot = sqlite3.connect(":memory:")
        boot.enable_load_extension(True)
        boot.load_extension(COUNTVFS)
        db = sqlite3.connect(f"file:{path}?vfs=countvfs", uri=True, isolation_level=None)
        db.enable_load_extension(True)
        db.load_extension(EXT)
        db.load_extension(COUNTVFS)
        db.execute("SELECT countvfs_reset()")
        _, _, st = self.search(db, self.Q[0], "AND ef = 32 AND beam = 4 AND rerank = 2")
        c = json.loads(db.execute("SELECT countvfs_stats()").fetchone()[0])
        # Every search round and the setup round were announced as prefetches.
        self.assertEqual(c["prefetch_calls"], st["rounds"] + st["setup_rounds"])
        self.assertGreaterEqual(c["distinct"], st["pages"] + st["setup_pages"])

    # ------------------------------------------------------------ IVF layout

    def test_ivf_two_rounds(self):
        db, _ = self.open("ivf.db", params="layout=ivf, nlist=50")
        r = self.recall(db, "AND nprobe = 25 AND rerank_k = 64")
        self.assertGreater(r, 0.95)
        _, _, st = self.search(db, self.Q[0], "AND nprobe = 8 AND rerank_k = 32")
        self.assertEqual((st["raw"], st["fallback"], st["rounds"]), (1, 0, 2))   # lists, then vectors
        self.assertEqual(st["lists"], 8)
        _, _, st = self.search(db, self.Q[0], "AND nprobe = 8 AND rerank = 0")
        self.assertEqual(st["rounds"], 1)
        # exact PQ over all lists equals a full probe without rerank.
        a = self.search(db, self.Q[1], "AND exact = 2")[0]
        b = self.search(db, self.Q[1], "AND nprobe = 50 AND rerank = 0")[0]
        self.assertEqual(a, b)
        self.assertEqual(self.search(db, self.Q[1], "AND exact = 1")[0], exact(self.X, self.Q[1], 10))

    def test_ivf_variants(self):
        for params in ("ivf_residual=0", "ivf_centroids=int8", "ivf_centroids=pq", "store_vectors=int8",
                       "store_vectors=none"):
            db, _ = self.open("ivfv.db", params="layout=ivf, nlist=30, " + params, n=1500)
            ids, _, st = self.search(db, self.X[7], "AND nprobe = 30 AND rerank_k = 32")
            self.assertEqual(ids[0], 8, params)
            self.assertEqual(st["fallback"], 0, params)

    def test_ivf_insert_delete(self):
        db, _ = self.open("ivfi.db", params="layout=ivf, nlist=40", n=2000)
        new = self.X[2000:2100]
        db.execute("BEGIN")
        for i, x in enumerate(new):
            db.execute("INSERT INTO v(rowid, embedding) VALUES (?, ?)", (50000 + i, x.tobytes()))
        db.execute("COMMIT")
        hits = sum(self.search(db, x, "AND nprobe = 8")[0][:1] == [50000 + i] for i, x in enumerate(new))
        self.assertGreater(hits, 95)
        db.execute("DELETE FROM v WHERE rowid = 50003")
        db.execute("DELETE FROM v WHERE rowid = 17")
        self.assertNotIn(50003, self.search(db, new[3], "AND nprobe = 8")[0])
        self.assertNotIn(17, self.search(db, self.X[16], "AND nprobe = 8")[0])
        db.execute("INSERT INTO v(v) VALUES ('finalize')")
        ids, _, st = self.search(db, new[5], "AND nprobe = 8")
        self.assertEqual(ids[0], 50005)
        self.assertEqual(st["rounds"], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
