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


if __name__ == "__main__":
    unittest.main(verbosity=2)
