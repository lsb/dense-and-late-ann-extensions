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

    def test_pinned_batch_larger_than_cache(self):
        """A prefetch batch 30x the cache budget arrives in one round and
        stays cached (pinned) until it is read, so a full scan costs no
        further round; PRAGMA httpvfs_release shrinks the cache back to its
        budget. The batch is cut at the hard limit (cache_max_kb)."""
        npages = os.path.getsize(DB) // 4096
        want = sqlite3.connect(f"file:{DB}?mode=ro", uri=True).execute(
            "SELECT count(*), sum(length(body)) FROM docs").fetchone()
        conn = self.open(cache_kb=64, readahead_kb=0, cache_max_kb=1 << 20)   # budget 16 blocks
        conn.execute("SELECT id FROM docs LIMIT 1").fetchall()                # schema
        conn.execute("PRAGMA httpvfs_reset=cache")
        conn.execute("PRAGMA cache_size=0")
        conn.execute("SELECT httpvfs_prefetch_pages(1, 1)").fetchone()        # header block
        conn.execute("PRAGMA httpvfs_reset")
        self.assertEqual(conn.execute("SELECT httpvfs_prefetch_pages(2, ?)", (npages - 1,)).fetchone()[0], 0)
        s = self.stats(conn)
        self.assertGreater(npages, 30 * s["cache_blocks"])
        self.assertEqual(s["rounds"], 1)
        self.assertEqual(s["prefetch_blocks"], npages - 1)
        self.assertEqual(s["pinned_blocks"], npages - 1)
        self.assertEqual(s["pin_evictions"], 0)
        self.assertEqual(conn.execute("SELECT count(*), sum(length(body)) FROM docs").fetchone(), want)
        s = self.stats(conn)
        self.assertEqual(s["rounds"], 1)                 # every page read came from the batch
        self.assertEqual(s["cache_misses"], 0)
        self.assertLess(s["pinned_blocks"], npages - 1)  # read blocks are unpinned
        conn.execute("PRAGMA httpvfs_release").fetchall()
        s = self.stats(conn)
        self.assertEqual(s["pinned_blocks"], 0)
        self.assertLessEqual(s["cached_blocks"], s["cache_blocks"])
        self.assertEqual(conn.execute("SELECT count(*), sum(length(body)) FROM docs").fetchone(), want)

        # hard limit: 64 blocks; the rest of the batch is left to on-demand reads
        conn = self.open(cache_kb=64, readahead_kb=0, cache_max_kb=256)
        conn.execute("SELECT id FROM docs LIMIT 1").fetchall()
        conn.execute("PRAGMA httpvfs_reset")
        conn.execute("SELECT httpvfs_prefetch_pages(2, ?)", (npages - 1,)).fetchone()
        s = self.stats(conn)
        self.assertEqual(s["cache_max_blocks"], 64)
        self.assertLessEqual(s["prefetch_blocks"], 64)
        self.assertLessEqual(s["peak_blocks"], 64)
        self.assertEqual(s["pin_evictions"], 0)
        self.assertEqual(conn.execute("SELECT count(*), sum(length(body)) FROM docs").fetchone(), want)

    def test_pinning_random_workload(self):
        """Random prefetch batches, lookups, scans and releases on a small
        cache (growth past the budget, hash resizing, eviction of pinned
        blocks at the hard limit, shrinking): results always match."""
        import random
        rnd = random.Random(5)
        npages = os.path.getsize(DB) // 4096
        plain = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
        for cache_kb, max_kb in [(64, 4096), (64, 96), (256, 3000)]:
            conn = self.open(cache_kb=cache_kb, cache_max_kb=max_kb)
            conn.execute("PRAGMA cache_size=0")
            for it in range(150):
                op = rnd.random()
                if op < 0.3:
                    first = rnd.randint(1, npages)
                    conn.execute("SELECT httpvfs_prefetch_pages(?, ?)",
                                 (first, rnd.randint(1, npages - first + 1))).fetchone()
                elif op < 0.4:
                    conn.execute("PRAGMA httpvfs_release").fetchall()
                elif op < 0.45:
                    q = "SELECT count(*), sum(length(body)) FROM docs"
                    self.assertEqual(conn.execute(q).fetchone(), plain.execute(q).fetchone())
                else:
                    i = rnd.randint(1, 4000)
                    q = "SELECT body FROM docs WHERE id=?"
                    self.assertEqual(conn.execute(q, (i,)).fetchone(), plain.execute(q, (i,)).fetchone())
                s = self.stats(conn)
                self.assertLessEqual(s["cached_blocks"], s["cache_max_blocks"])
                self.assertLessEqual(s["pinned_blocks"], s["cached_blocks"])
            conn.execute("PRAGMA httpvfs_release").fetchall()
            s = self.stats(conn)
            self.assertLessEqual(s["cached_blocks"], s["cache_blocks"])
            q = "SELECT count(*), sum(length(body)) FROM docs"
            self.assertEqual(conn.execute(q).fetchone(), plain.execute(q).fetchone())

    def test_auto_cache_budget(self):
        """cache_kb=auto (the default): 1/64 of the file, at least 4 MiB."""
        s = self.stats(self.open())
        self.assertEqual(s["cache_blocks"], 1024)            # the 2 MB test file: the 4 MiB floor
        self.assertEqual(s["cache_max_blocks"], 4096)
        s = self.stats(self.open(cache_min_kb=64))
        self.assertEqual(s["cache_blocks"], max(16, os.path.getsize(DB) // 64 // 4096))
        s = self.stats(self.open(cache_kb=8192))
        self.assertEqual(s["cache_blocks"], 2048)

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
