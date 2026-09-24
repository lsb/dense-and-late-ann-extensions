#!/usr/bin/env python3
"""Tests for the late_plaid extension (run: make -C ext/late test).

Uses words-100 embeddings if present, otherwise synthetic clustered data.
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import apsw
import numpy as np

HERE = Path(__file__).resolve().parent.parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
import pyref  # noqa: E402

EXT = str(HERE / "build" / "late.so")


def data():
    v = REPO / "data" / "emb" / "words-100.lateon.vectors.npy"
    if v.exists():
        V = np.load(v).astype(np.float32)
        O = np.load(REPO / "data" / "emb" / "words-100.lateon.offsets.npy")
        return V, O
    rng = np.random.default_rng(0)
    cent = rng.standard_normal((64, 48)).astype(np.float32)
    lens = rng.integers(20, 60, 100)
    O = np.concatenate([[0], np.cumsum(lens)])
    V = cent[rng.integers(0, 64, O[-1])] + 0.3 * rng.standard_normal((O[-1], 48)).astype(np.float32)
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    return V, O


def connect(path):
    con = apsw.Connection(path)
    con.enable_load_extension(True)
    con.load_extension(EXT, "sqlite3_late_init")
    return con


def query(con, q, opts, k=10):
    rows = list(con.execute("SELECT rowid, score, stats FROM t WHERE t MATCH ? AND k = ? AND opts = ?",
                            (q.astype(np.float32).tobytes(), k, opts)))
    return [r[0] for r in rows], [r[1] for r in rows], json.loads(rows[0][2]) if rows else {}


class LatePlaidTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.V, cls.O = data()
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dbs = {}
        for nbits in (1, 2, 4):
            path = os.path.join(cls.tmp.name, f"n{nbits}.db")
            con = connect(path)
            con.execute(f"CREATE VIRTUAL TABLE t USING late_plaid(dim=48, nbits={nbits}, centroids=256)")
            with con:
                for i in range(len(cls.O) - 1):   # non-identity rowids
                    con.execute("INSERT INTO t(rowid, vectors) VALUES (?, ?)",
                                (1000 + 3 * i, cls.V[cls.O[i]:cls.O[i + 1]].tobytes()))
            con.execute("INSERT INTO t(t) VALUES ('build')")
            con.execute("INSERT INTO t(t) VALUES ('finalize')")
            con.close()
            cls.dbs[nbits] = path
        cls.q = cls.V[cls.O[7] + 2:cls.O[7] + 6]          # four tokens of document 7

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_exact_matches_python_reference(self):
        for nbits, path in self.dbs.items():
            con = connect(path)
            ids, scores, st = query(con, self.q, "exact=1", k=20)
            ref = pyref.Index(__import__("sqlite3").connect(path))
            docs = ref.docs(__import__("sqlite3").connect(path))
            sc = pyref.maxsim_all(docs, self.q)
            top = sorted(sc, key=lambda d: -sc[d])[:20]
            self.assertEqual([1000 + 3 * d for d in top[:5]], ids[:5], f"nbits={nbits}")
            np.testing.assert_allclose(scores[:5], [sc[d] for d in top[:5]], rtol=1e-4)

    def test_postings_match_docs(self):
        import sqlite3
        con = sqlite3.connect(self.dbs[2])
        ref = pyref.Index(con)
        docs = ref.docs(con)
        n = 0
        for c, ids, vecs in ref.postings(con):
            for d, v in zip(ids, vecs):
                self.assertTrue(np.any(np.all(np.abs(docs[int(d)] - v) < 1e-6, axis=1)))
                n += 1
        self.assertEqual(n, ref.T)

    def test_layouts_find_source_document(self):
        con = connect(self.dbs[4])
        for opts in ("layout=warp nprobe=8", "layout=plaid approx=codes nprobe=4",
                     "layout=plaid approx=ivf nprobe=4 ndocs=32", "layout=warp nprobe=4 rerank=16"):
            ids, _, st = query(con, self.q, opts)
            self.assertIn(1000 + 3 * 7, ids[:3], opts)
            self.assertEqual(st["hint_miss"], 0)
            self.assertEqual(st["fallbacks"], 0)

    def test_rounds(self):
        con = connect(self.dbs[2])
        _, _, st = query(con, self.q, "layout=warp")
        self.assertGreaterEqual(st["static_rounds"], 1)
        # rowids here are not 0..N-1, so every query ends with a rowid-map lookup
        _, _, st = query(con, self.q, "layout=warp")
        self.assertEqual(st["rounds"], 1 + 1)             # hinted postings, rowid map leaf
        _, _, st = query(con, self.q, "layout=plaid approx=ivf")
        self.assertLessEqual(st["rounds"], 2 + 2)         # IVF + docs (+ b-tree levels, cold)
        _, _, st = query(con, self.q, "layout=plaid approx=ivf")
        self.assertEqual(st["rounds"], 2 + 1)             # IVF, docs leaves, rowid map

    def test_vacuum_keeps_results(self):
        path = os.path.join(self.tmp.name, "vac.db")
        import shutil
        shutil.copy(self.dbs[2], path)
        con = connect(path)
        before = query(con, self.q, "layout=warp")[0]
        con.execute("CREATE TABLE junk(x)")
        con.execute("DROP TABLE junk")
        con.execute("DELETE FROM t_buffer")
        con.execute("VACUUM")
        con.close()
        con = connect(path)
        after, _, st = query(con, self.q, "layout=warp")
        self.assertEqual(before, after)
        self.assertGreater(st["hint_miss"], 0)            # pages moved: hints are stale, walk used

    def test_two_level(self):
        path = os.path.join(self.tmp.name, "two.db")
        con = connect(path)
        con.execute("CREATE VIRTUAL TABLE t USING late_plaid(dim=48, nbits=2, centroids=256, coarse=16, layout=warp)")
        vec, off = os.path.join(self.tmp.name, "v.npy"), os.path.join(self.tmp.name, "o.npy")
        np.save(vec, self.V.astype(np.float16))
        np.save(off, self.O.astype(np.int64))
        con.execute("INSERT INTO t(t) VALUES (?)", (f"build_npy {vec} {off}",))
        con.execute("INSERT INTO t(t) VALUES ('finalize')")
        con = connect(path)
        ids, _, st = query(con, self.q, "nprobe=8 cprobe=4")
        self.assertIn(7, ids[:3])
        self.assertGreater(st["cells"], 0)
        exact = query(con, self.q, "exact=1", k=5)[0]
        import sqlite3
        ref = pyref.Index(sqlite3.connect(path))
        acc = {}
        for c, ids_, vecs in ref.postings(sqlite3.connect(path)):
            s = vecs @ self.q.T
            for d, row in zip(ids_, s):
                acc[int(d)] = np.maximum(acc.get(int(d), -9), row)
        top = sorted(acc, key=lambda d: -acc[d].sum())[:5]
        self.assertEqual(top, exact)


    def _build(self, name, params):
        path = os.path.join(self.tmp.name, name)
        con = connect(path)
        con.execute(f"CREATE VIRTUAL TABLE t USING late_plaid(dim=48, centroids=256, {params})")
        vec, off = os.path.join(self.tmp.name, "v.npy"), os.path.join(self.tmp.name, "o.npy")
        np.save(vec, self.V.astype(np.float16))
        np.save(off, self.O.astype(np.int64))
        con.execute("INSERT INTO t(t) VALUES (?)", (f"build_npy {vec} {off}",))
        con.execute("INSERT INTO t(t) VALUES ('finalize')")
        con.close()
        return path

    @staticmethod
    def _meta(path):
        import sqlite3
        return pyref._stream(sqlite3.connect(path), "t_meta")

    def test_centroid_types(self):
        """int8 / int4 centroids (format 3): smaller static data, exact scores
        still match the Python decoder, finalize is idempotent."""
        sizes = {}
        for cq in ("f16", "int8", "int4"):
            path = self._build(f"cq-{cq}.db", f"nbits=2, centroid_type={cq}")
            import sqlite3
            ref = pyref.Index(sqlite3.connect(path))
            self.assertEqual(ref.cq, {"f16": 0, "int8": 1, "int4": 2}[cq])
            con = connect(path)
            ids, scores, st = query(con, self.q, "exact=1", k=20)
            sizes[cq] = st["static_bytes"]
            docs = ref.docs(sqlite3.connect(path))
            sc = pyref.maxsim_all(docs, self.q)
            top = sorted(sc, key=lambda d: -sc[d])[:5]
            self.assertEqual(top, ids[:5], cq)
            np.testing.assert_allclose(scores[:5], [sc[d] for d in top], rtol=1e-4)
            self.assertIn(7, query(con, self.q, "layout=warp nprobe=8")[0][:3], cq)
            m1 = self._meta(path)
            con.execute("INSERT INTO t(t) VALUES ('finalize')")
            self.assertEqual(m1, self._meta(path), f"{cq}: finalize changed the static data")
            if cq != "f16":   # stored values reproduce the in-memory centroids
                c = ref.centroids
                self.assertTrue(np.all(np.isfinite(c)))
                self.assertLess(abs(np.linalg.norm(c, axis=1).mean() - 1), 0.1)
        self.assertLess(sizes["int8"], 0.62 * sizes["f16"])
        self.assertLess(sizes["int4"], sizes["int8"])

    def test_format_2_still_reads(self):
        """A version-2 static blob (float16 centroids, u32 list lengths) gives
        the same answers as the version-3 blob it was converted from."""
        import sqlite3
        import struct
        path = self._build("v2.db", "nbits=2, centroid_type=f16")
        con = connect(path)
        want = [query(con, self.q, o, k=10)[:2] for o in ("layout=warp nprobe=8", "layout=plaid approx=ivf nprobe=4")]
        con.close()
        ref = pyref.Index(sqlite3.connect(path))
        blob = self._meta(path)
        K, D = ref.K, ref.dim
        head = bytearray(blob[:64 + 132 + 8 + 8])
        struct.pack_into("<I", head, 4, 2)
        p = len(head) + 4                                    # skip the v3 centroid type
        body = blob[p:p + 2 * K * D]
        p += 2 * K * D
        rest = pyref._varints(blob, p, K)
        ivf, p = rest
        post, p = pyref._varints(blob, p, K)
        v2 = bytes(head) + body + ivf.astype("<u4").tobytes() + post.astype("<u4").tobytes() + blob[p:]
        chunk = ref.chunk
        db = sqlite3.connect(path)
        db.execute("DELETE FROM t_meta")
        for i in range(0, len(v2), chunk):
            db.execute("INSERT INTO t_meta(id, data) VALUES (?, ?)", (i // chunk + 1, v2[i:i + chunk]))
        db.commit(); db.close()
        self.assertEqual(struct.unpack_from("<I", self._meta(path), 4)[0], 2)
        con = connect(path)
        got = [query(con, self.q, o, k=10)[:2] for o in ("layout=warp nprobe=8", "layout=plaid approx=ivf nprobe=4")]
        self.assertEqual(got, want)

    def test_unknown_version_is_a_clear_error(self):
        import sqlite3
        import struct
        path = self._build("v99.db", "nbits=2")
        blob = bytearray(self._meta(path))
        struct.pack_into("<I", blob, 4, 99)
        db = sqlite3.connect(path)
        first = db.execute("SELECT id, data FROM t_meta ORDER BY id LIMIT 1").fetchone()
        db.execute("UPDATE t_meta SET data = ? WHERE id = ?", (bytes(blob[:len(first[1])]), first[0]))
        db.commit(); db.close()
        con = connect(path)
        with self.assertRaises(apsw.Error) as cm:
            query(con, self.q, "layout=warp")
        self.assertIn("format version 99", str(cm.exception))

    def test_warm(self):
        """MATCH 'warm' loads the static data and the document table's interior
        pages; the next query then needs no static round."""
        for path in (self.dbs[2], self._build("warm.db", "nbits=2, centroid_type=int8")):
            con = connect(path)
            rows = list(con.execute("SELECT rowid FROM t WHERE t MATCH 'warm'"))
            self.assertEqual(rows, [])
            _, _, st = query(con, self.q, "layout=plaid approx=ivf nprobe=4")
            self.assertEqual(st["static_rounds"], 0)
            con2 = connect(path)
            query(con2, self.q, "layout=plaid approx=ivf nprobe=4")
            _, _, warm_state = query(con2, self.q, "layout=plaid approx=ivf nprobe=4")
            self.assertEqual(st["rounds"], warm_state["rounds"])   # no b-tree level left to read
            self.assertEqual(list(con.execute("SELECT count(*) FROM t WHERE t MATCH 'warm'")), [(0,)])


if __name__ == "__main__":
    unittest.main(verbosity=2)
