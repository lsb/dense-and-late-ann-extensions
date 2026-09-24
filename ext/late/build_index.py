#!/usr/bin/env python3
"""Build a late_plaid index database from precomputed token embeddings.

The heavy lifting is done by the extension's C builder ('build_npy'), which
streams the float16 vectors from the .npy file; this script only creates the
database, runs 'build' and 'finalize', and records sizes and timings.

  python3 ext/late/build_index.py --data words-10k --nbits 2 --centroids 16384 \
      --out build/late/w10k-n2-k16384.db [--layout both] [--ppc 256] [--iters 4]

Writes <out>.json with the parameters, build time and per-table sizes.
"""
import argparse
import json
import os
import time
from pathlib import Path

import apsw

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
EXT = str(HERE / "build" / "late.so")


def connect(path, vfs=None):
    con = apsw.Connection(str(path), vfs=vfs) if vfs else apsw.Connection(str(path))
    con.enable_load_extension(True)
    con.load_extension(EXT, "sqlite3_late_init")
    return con


def table_sizes(con, name):
    out = {}
    for tb, n, pages, payload in con.execute(
            "SELECT name, count(*), sum(pgsize), sum(payload) FROM dbstat "
            "WHERE name LIKE ? || '%' GROUP BY name", (name,)):
        out[tb] = {"pages": n, "bytes": pages, "payload": payload}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="words-10k")
    ap.add_argument("--vectors")
    ap.add_argument("--offsets")
    ap.add_argument("--out", required=True)
    ap.add_argument("--nbits", type=int, default=2)
    ap.add_argument("--centroids", type=int, default=0, help="0 = fast-plaid formula")
    ap.add_argument("--layout", default="both")
    ap.add_argument("--ppc", type=int, default=256, help="k-means points per centroid")
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--sample-docs", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--mem-mb", type=int, default=1024)
    ap.add_argument("--page-size", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    vec = a.vectors or str(REPO / "data" / "emb" / f"{a.data}.lateon.vectors.npy")
    off = a.offsets or str(REPO / "data" / "emb" / f"{a.data}.lateon.offsets.npy")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("", "-journal"):
        if os.path.exists(str(out) + suffix):
            os.remove(str(out) + suffix)
    con = connect(out)
    con.execute(f"PRAGMA page_size={a.page_size}")
    con.execute("PRAGMA journal_mode=delete")
    con.execute(
        f"CREATE VIRTUAL TABLE t USING late_plaid(dim=48, nbits={a.nbits}, centroids={a.centroids}, "
        f"layout={a.layout}, kmeans_iters={a.iters}, kmeans_ppc={a.ppc}, sample_docs={a.sample_docs}, "
        f"threads={a.threads}, mem_mb={a.mem_mb}, seed={a.seed}, verbose=1)")
    t0 = time.time()
    con.execute("INSERT INTO t(t) VALUES (?)", (f"build_npy {vec} {off}",))
    t_build = time.time() - t0
    t0 = time.time()
    con.execute("INSERT INTO t(t) VALUES ('finalize')")
    t_fin = time.time() - t0
    sizes = table_sizes(con, "t_")
    con.close()
    meta = {
        "data": a.data, "vectors": vec, "offsets": off, "nbits": a.nbits, "centroids": a.centroids,
        "layout": a.layout, "ppc": a.ppc, "iters": a.iters, "sample_docs": a.sample_docs,
        "threads": a.threads, "page_size": a.page_size, "seed": a.seed,
        "build_s": round(t_build, 2), "finalize_s": round(t_fin, 2),
        "db_bytes": os.path.getsize(out), "tables": sizes,
    }
    Path(str(out) + ".json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
