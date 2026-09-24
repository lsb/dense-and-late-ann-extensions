#!/usr/bin/env python3
"""Cross-check the extension's own page trace against every read SQLite and
the extension actually issue, recorded by an APSW VFS shim (bench/countvfs.py).

For each configuration, opens a fresh connection through the counting VFS,
runs one warm-up query (session start: schema, static data), then for N
queries records the distinct pages read by the whole statement and compares
them with the pages the extension reports in its trace.

  python3 ext/late/countcheck.py --db build/late/w10k-n2-k16384.db --configs "layout=warp nprobe=4"
"""
import argparse
import json
import sys
from pathlib import Path

import apsw
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "bench"))
sys.path.insert(0, str(HERE))
from countvfs import CountingVFS  # noqa: E402
from bench import load_queries  # noqa: E402

EXT = str(HERE / "build" / "late.so")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--queries", default="words-10k")
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("-n", type=int, default=20)
    a = ap.parse_args()
    qs = load_queries(a.queries)[:: max(1, 2000 // a.n)][: a.n]
    vfs = CountingVFS("latecount")
    for opts in a.configs:
        con = apsw.Connection(a.db, vfs="latecount", flags=apsw.SQLITE_OPEN_READONLY)
        con.enable_load_extension(True)
        con.load_extension(EXT, "sqlite3_late_init")
        ps = con.execute("PRAGMA page_size").fetchall()[0][0]
        vfs.reset()
        st0 = json.loads(con.execute("SELECT stats FROM t WHERE t MATCH ? AND k=10 AND opts=?",
                                     (qs[0]["vec"].tobytes(), opts + " trace=1")).fetchall()[0][0])
        first = {o // ps + 1 for o, _ in vfs.reads}
        rows = []
        for q in qs:
            vfs.reset()
            st = json.loads(con.execute("SELECT stats FROM t WHERE t MATCH ? AND k=10 AND opts=?",
                                        (q["vec"].tobytes(), opts + " trace=1")).fetchall()[0][0])
            vfs_pages = {o // ps + 1 for o, _ in vfs.reads}
            ext_pages = {p for r in st["trace"] for p in r}
            rows.append((len(vfs_pages), len(ext_pages), len(vfs_pages - ext_pages), len(ext_pages - vfs_pages)))
        r = np.array(rows)
        print(f"{opts:45s} session start: {len(first)} pages ({st0['static_rounds']} static rounds); per query: "
              f"VFS pages {r[:, 0].mean():.1f}, trace pages {r[:, 1].mean():.1f}, "
              f"VFS-only {r[:, 2].mean():.2f}, trace-only {r[:, 3].mean():.2f}")
        con.close()


if __name__ == "__main__":
    main()
