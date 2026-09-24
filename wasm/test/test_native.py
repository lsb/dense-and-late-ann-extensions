"""Native tests: Python's sqlite3 on the project's SQLite build, with httpvfs.so
and demo_ext.so loaded as extensions.

    LD_LIBRARY_PATH=build/native python3 wasm/test/test_native.py
(make test-native sets LD_LIBRARY_PATH itself; without it the system SQLite is
used, which also works for these tests.)
"""
import json
import math
import os
import sqlite3
import subprocess
import sys
import time
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
NATIVE = os.path.join(ROOT, "build", "native")
DB = os.path.join(ROOT, "build", "test", "fts-test.db")


def ensure_db():
    if not os.path.exists(DB):
        subprocess.run(["node", "-e",
                        "import('./wasm/test/helpers.mjs').then(m => m.makeTestDb())"],
                       cwd=ROOT, check=True)


def load(conn, name):
    conn.enable_load_extension(True)
    conn.load_extension(os.path.join(NATIVE, name))
    conn.enable_load_extension(False)


class NativeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ensure_db()
        boot = sqlite3.connect(":memory:")
        load(boot, "httpvfs")   # registers the VFS process-wide

    def open(self, **params):
        q = "&".join(f"{k}={v}" for k, v in {"vfs": "httpvfs", **params}.items())
        conn = sqlite3.connect(f"file:{DB}?{q}", uri=True)
        load(conn, "demo_ext")
        return conn

    def stats(self, conn):
        return json.loads(conn.execute("SELECT httpvfs_stats()").fetchone()[0])

    def test_fts_matches_default_vfs(self):
        plain = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
        conn = self.open()
        for q in ["SELECT rowid FROM docs_fts WHERE docs_fts MATCH 'bab* OR bac*' ORDER BY rowid",
                  "SELECT count(*) FROM docs_fts WHERE docs_fts MATCH 'a*'"]:
            self.assertEqual(conn.execute(q).fetchall(), plain.execute(q).fetchall())
        s = self.stats(conn)
        self.assertGreater(s["requests"], 0)
        self.assertGreater(s["rounds"], 0)
        self.assertEqual(conn.execute("SELECT hello('native')").fetchone()[0], "hello, native")

    def test_readonly(self):
        conn = self.open()
        with self.assertRaises(sqlite3.OperationalError):
            conn.execute("CREATE TABLE t(x)")

    def test_speculative_batching_and_latency(self):
        ids = [17, 311, 555, 901, 1234, 1500, 1777, 2020, 2345, 2600, 2900, 3100, 3333, 3600, 3800, 3999]
        conn = self.open(latency_ms=10, readahead_kb=0)
        conn.execute("SELECT id FROM docs LIMIT 1").fetchall()
        conn.execute("PRAGMA httpvfs_reset=cache")
        conn.execute("PRAGMA cache_size=0")   # keep SQLite's own cache out of the way
        t0 = time.perf_counter()
        seq = [conn.execute("SELECT body FROM docs WHERE id=?", (i,)).fetchone() for i in ids]
        t_seq = time.perf_counter() - t0
        r_seq = self.stats(conn)["rounds"]
        self.assertGreaterEqual(r_seq, len(ids))

        conn.execute("PRAGMA httpvfs_reset=cache")
        t0 = time.perf_counter()
        passes = conn.execute("SELECT httpvfs_warm('SELECT body FROM docs WHERE id=?', ?)",
                              (",".join(map(str, ids)),)).fetchone()[0]
        bat = [conn.execute("SELECT body FROM docs WHERE id=?", (i,)).fetchone() for i in ids]
        t_bat = time.perf_counter() - t0
        r_bat = self.stats(conn)["rounds"]
        self.assertEqual(seq, bat)
        self.assertLessEqual(r_bat, 4)
        # Simulated latency: 10 ms per round.
        self.assertGreater(t_seq, 0.010 * len(ids))
        self.assertLess(t_bat, t_seq / 2)
        print(f"\n  native, latency 10 ms: {len(ids)} lookups sequential {r_seq} rounds "
              f"{t_seq*1e3:.0f} ms; batched ({passes} passes) {r_bat} rounds {t_bat*1e3:.0f} ms",
              file=sys.stderr)

    def test_pragma_stats(self):
        conn = self.open()
        conn.execute("SELECT count(*) FROM docs").fetchall()
        txt = conn.execute("PRAGMA httpvfs_stats").fetchone()[0]
        self.assertIn("rounds=", txt)

    def pragma_stats(self, conn):
        txt = conn.execute("PRAGMA httpvfs_stats").fetchone()[0]
        return {k: float(v) for k, v in (kv.split("=") for kv in txt.split())}

    def test_request_budget(self):
        """max_req=6: a round of 16 scattered lookups is merged into at most
        6 requests per round; the results are unchanged and the blocks
        fetched in between are counted as over-fetch (and cached)."""
        ids = [17, 311, 555, 901, 1234, 1500, 1777, 2020, 2345, 2600, 2900, 3100, 3333, 3600, 3800, 3999]
        want = sqlite3.connect(f"file:{DB}?mode=ro", uri=True).execute(
            f"SELECT id, body FROM docs WHERE id IN ({','.join(map(str, ids))}) ORDER BY id").fetchall()
        for extra, label in [({}, "unlimited"), ({"max_req": 6, "rtt_ms": 2000, "bw_kbps": 8100, "net_auto": 0}, "coalesce"),
                             ({"max_req": 6, "multipart": 1}, "multipart")]:
            conn = self.open(readahead_kb=0, **extra)
            conn.execute("SELECT id FROM docs LIMIT 1").fetchall()
            conn.execute("PRAGMA httpvfs_reset=cache")
            conn.execute("PRAGMA cache_size=0")
            conn.execute("SELECT httpvfs_warm('SELECT body FROM docs WHERE id=?', ?)",
                         (",".join(map(str, ids)),)).fetchone()
            s = self.pragma_stats(conn)
            got = conn.execute(f"SELECT id, body FROM docs WHERE id IN ({','.join(map(str, ids))}) ORDER BY id").fetchall()
            self.assertEqual(got, want, label)
            if label == "unlimited":
                self.assertGreater(s["requests"], 6 * s["rounds"] - 6)   # at least one big round
                self.assertEqual(s["overfetch"], 0)
            else:
                self.assertLessEqual(s["requests"], 6 * s["rounds"], label)
                self.assertGreater(s["planned_rounds"], 0, label)
                if label == "coalesce":
                    self.assertGreater(s["overfetch"], 0)
                    # the over-fetched blocks were cached: reading them again is free
                    self.assertEqual(self.pragma_stats(conn)["requests"], s["requests"])
                else:
                    self.assertEqual(s["overfetch"], 0)
                    self.assertGreater(s["multipart_requests"], 0)


class PlannerTest(unittest.TestCase):
    """httpvfs_plan(), the per-round request planner, called through ctypes
    and checked against brute force over every way to cut the ranges."""

    @classmethod
    def setUpClass(cls):
        import ctypes as C
        cls.C = C
        lib = C.CDLL(os.path.join(NATIVE, "httpvfs.so"))
        f = lib.httpvfs_plan
        f.restype = C.c_int
        f.argtypes = [C.c_int, C.POINTER(C.c_int64), C.POINTER(C.c_int64), C.c_int, C.c_int,
                      C.c_double, C.c_double, C.c_int64, C.c_int, C.c_int, C.POINTER(C.c_int)]
        cls.fn = f

    def plan(self, ranges, bs=4096, max_req=6, rtt=100.0, bw=1250.0, max_blocks=0, mp=0, max_parts=100):
        C = self.C
        m = len(ranges)
        rs = (C.c_int64 * m)(*[a for a, _ in ranges])
        re = (C.c_int64 * m)(*[b for _, b in ranges])
        grp = (C.c_int * m)()
        g = self.fn(m, rs, re, bs, max_req, rtt, bw, max_blocks, mp, max_parts, grp)
        return g, list(grp)

    @staticmethod
    def cost(ranges, grp, bs, max_req, rtt, bw):
        spans = {}
        for (a, b), g in zip(ranges, grp):
            lo, hi = spans.get(g, (a, b))
            spans[g] = (min(lo, a), max(hi, b))
        nblk = sum(hi - lo for lo, hi in spans.values())
        return math.ceil(len(spans) / max_req) * rtt + nblk * bs / bw, nblk

    def random_ranges(self, rnd, m):
        out, pos = [], rnd.randrange(0, 50)
        for _ in range(m):
            pos += rnd.choice([1, 1, 2, 3, 5, 10, 40, 200])
            n = rnd.choice([1, 1, 1, 2, 4])
            out.append((pos, pos + n))
            pos += n
        return out

    def test_trivial_cases(self):
        self.assertEqual(self.plan([]), (0, []))
        r = [(i * 10, i * 10 + 1) for i in range(6)]
        self.assertEqual(self.plan(r), (6, list(range(6))))        # m <= max_req
        r = [(i * 10, i * 10 + 1) for i in range(20)]
        self.assertEqual(self.plan(r, max_req=0), (20, list(range(20))))   # no budget
        # zero latency: never worth merging; huge latency: one wave of six
        self.assertEqual(self.plan(r, rtt=0.0)[0], 20)
        g, grp = self.plan(r, rtt=1e6)
        self.assertEqual(g, 6)
        self.assertEqual(grp, sorted(grp))

    def test_matches_brute_force(self):
        import itertools
        import random
        rnd = random.Random(7)
        for trial in range(300):
            m = rnd.randrange(2, 11)
            ranges = self.random_ranges(rnd, m)
            max_req = rnd.choice([1, 2, 3, 4, 6])
            rtt = rnd.choice([5.0, 20.0, 70.0, 165.0, 562.5])
            bw = rnd.choice([180.0, 1012.5, 1500.0, 6250.0])     # bytes per ms
            max_blocks = rnd.choice([0, 0, 8, 30])
            bs = 4096
            g, grp = self.plan(ranges, bs, max_req, rtt, bw, max_blocks)
            self.assertEqual(len(set(grp)), g)
            self.assertEqual(grp, sorted(grp))
            self.assertEqual(grp[0], 0)
            t, nblk = self.cost(ranges, grp, bs, max_req, rtt, bw)
            best = math.inf
            for cuts in itertools.product([0, 1], repeat=m - 1):
                gg, k = [0], 0
                for c in cuts:
                    k += c
                    gg.append(k)
                tt, nb = self.cost(ranges, gg, bs, max_req, rtt, bw)
                if max_blocks and k + 1 < m and nb > max_blocks:
                    continue
                best = min(best, tt)
            self.assertLessEqual(t, best + 1e-6, (trial, ranges, max_req, rtt, bw, grp))
            if max_blocks and g < m:
                self.assertLessEqual(nblk, max_blocks)

    def test_multipart_groups(self):
        import random
        rnd = random.Random(3)
        for _ in range(200):
            m = rnd.randrange(1, 120)
            ranges = self.random_ranges(rnd, m)
            max_req = rnd.choice([2, 6, 8])
            max_parts = rnd.choice([3, 16, 100])
            g, grp = self.plan(ranges, max_req=max_req, mp=1, max_parts=max_parts)
            self.assertEqual(grp, sorted(grp))
            self.assertEqual(sorted(set(grp)), list(range(g)))
            counts = [grp.count(k) for k in range(g)]
            self.assertLessEqual(max(counts), max_parts)
            if m <= max_req:
                self.assertEqual(g, m)
            else:
                self.assertEqual(g, max(max_req, math.ceil(m / max_parts)))
        # byte balance: 60 one-block ranges into 6 requests of 10
        g, grp = self.plan([(i * 3, i * 3 + 1) for i in range(60)], mp=1)
        self.assertEqual([grp.count(k) for k in range(g)], [10] * 6)



if __name__ == "__main__":
    unittest.main(verbosity=2)
