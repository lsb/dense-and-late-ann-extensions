"""Native tests: Python's sqlite3 on the project's SQLite build, with httpvfs.so
and demo_ext.so loaded as extensions.

    LD_LIBRARY_PATH=build/native python3 wasm/test/test_native.py
(make test-native sets LD_LIBRARY_PATH itself; without it the system SQLite is
used, which also works for these tests.)
"""
import json
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
