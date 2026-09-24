#!/usr/bin/env python3
"""Benchmark for the dense_ann SQLite extension.

Builds an index over a dataset, then sweeps search parameters and reports
recall@k (against exact float32 search and against exact PQ search), native
QPS, and the httpvfs-relevant costs of each query: dependent fetch rounds,
pages and bytes read, nodes expanded, distance computations.

Datasets:
  synth        clustered Gaussian vectors with low-rank structure (--n)
  words-10k    MiniLM embeddings of data/corpora/words-10k.txt; queries are
               200 held-out documents (the next documents of the same stream)
               and 200 single words, encoded with enc/minilm.py

Examples:
  python3 ext/dense/bench.py --data words-10k
  python3 ext/dense/bench.py --data synth --n 100000 --page-size 65536
  python3 ext/dense/bench.py --data synth --n 1000000 --sweep small

Run `make -C ext/dense` first. Results go to ext/dense/results/<tag>.json.
"""
import argparse
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
EXT = str(HERE / "build" / "dense_ann")
COUNTVFS = str(HERE / "build" / "countvfs")
CACHE = HERE / "build"


# ----------------------------------------------------------------- data

def normalize(x):
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


def synth(n, nq, dim=384, seed=0):
    """Clustered data: unit cluster centres + a shared rank-48 component +
    small isotropic noise, normalised. Queries come from the same model."""
    rng = np.random.default_rng(seed)
    ncl = max(16, int(np.sqrt(n)))
    centers = normalize(rng.standard_normal((ncl, dim)).astype(np.float32))
    A = rng.standard_normal((48, dim)).astype(np.float32)
    A /= np.linalg.norm(A, axis=1, keepdims=True)

    def gen(m, r):
        out = np.empty((m, dim), np.float32)
        for s in range(0, m, 100_000):
            e = min(m, s + 100_000)
            c = r.integers(0, ncl, e - s)
            z = r.standard_normal((e - s, 48)).astype(np.float32) * (0.8 / np.sqrt(48))
            iso = r.standard_normal((e - s, dim)).astype(np.float32) * (0.3 / np.sqrt(dim))
            out[s:e] = normalize(centers[c] + z @ A + iso)
        return out

    return gen(n, rng), gen(nq, np.random.default_rng(seed + 1))


def words_queries(nq_docs=200, nq_words=200):
    """Held-out documents 10000.. of the corpus stream, and single words."""
    path = CACHE / f"queries-words-{nq_docs}-{nq_words}.npy"
    if path.exists():
        return np.load(path)
    docs = []
    with open(REPO / "data" / "corpora" / "words-1m.txt") as f:
        for i, line in enumerate(f):
            if i >= 10000:
                docs.append(line.strip())
            if len(docs) == nq_docs:
                break
    with open(REPO / "data" / "words" / "shuffled-seed0.txt") as f:
        words = [w.strip() for _, w in zip(range(nq_words), f)]
    sys.path.insert(0, str(REPO))
    from enc.minilm import MiniLM
    q = MiniLM().encode(docs + words, batch_size=1).astype(np.float32)
    CACHE.mkdir(exist_ok=True)
    np.save(path, q)
    return q


def load_data(args):
    if args.data == "synth":
        return synth(args.n, args.queries, seed=args.seed)
    if args.data.startswith("words-") or args.data == "llm-paragraphs":
        X = np.load(REPO / "data" / "emb" / f"{args.data}.minilm.npy").astype(np.float32)
        return normalize(X), normalize(words_queries())
    raise SystemExit(f"unknown dataset {args.data}")


def exact_topk(X, Q, k):
    out = np.empty((len(Q), k), np.int64)
    for s in range(0, len(Q), 64):
        sims = Q[s:s + 64] @ X.T
        idx = np.argpartition(-sims, k, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(sims, idx, axis=1), axis=1)
        out[s:s + 64] = np.take_along_axis(idx, order, axis=1)
    return out + 1   # rowids are 1-based


# ---------------------------------------------------------------- build

def connect(path, vfs=None):
    if vfs:
        db = sqlite3.connect(f"file:{path}?vfs={vfs}", uri=True, isolation_level=None)
    else:
        db = sqlite3.connect(path, isolation_level=None)
    db.enable_load_extension(True)
    db.load_extension(EXT)
    return db


def build_db(path, X, params, page_size):
    if os.path.exists(path):
        os.remove(path)
    db = connect(path)
    db.execute(f"PRAGMA page_size={page_size}")
    db.execute(f"CREATE VIRTUAL TABLE v USING dense_ann(dim={X.shape[1]}{', ' + params if params else ''})")
    t = time.time()
    db.execute("BEGIN")
    db.executemany("INSERT INTO v(rowid, embedding) VALUES (?, ?)",
                   ((i + 1, X[i].tobytes()) for i in range(len(X))))
    db.execute("COMMIT")
    t_insert = time.time() - t
    t = time.time()
    db.execute("INSERT INTO v(v) VALUES ('build')")
    t_build = time.time() - t
    t = time.time()
    db.execute("INSERT INTO v(v) VALUES ('finalize')")
    t_final = time.time() - t
    cfg = {k: v for k, v in db.execute("SELECT key, value FROM v_config") if not isinstance(v, bytes)}
    db.close()
    size = os.path.getsize(path)
    return {
        "insert_s": t_insert, "build_s": t_build, "finalize_s": t_final,
        "db_bytes": size, "bytes_per_doc": size / len(X), "config": cfg,
    }


def table_bytes(path):
    """Bytes per shadow table (dbstat)."""
    db = sqlite3.connect(path)
    try:
        rows = db.execute("SELECT name, sum(pgsize) FROM dbstat GROUP BY name ORDER BY 2 DESC").fetchall()
    except sqlite3.OperationalError:
        rows = []
    db.close()
    return dict(rows)


# -------------------------------------------------------------- queries

SQL = ("SELECT rowid, distance, stats FROM v WHERE embedding MATCH ?1 AND k = ?2 AND ef = ?3 "
       "AND beam = ?4 AND rerank = ?5 AND trace = ?6")


def run_queries(db, Q, k, ef, beam, rerank, trace=False):
    ids, stats = [], []
    t = time.time()
    for q in Q:
        rows = db.execute(SQL, (q.tobytes(), k, ef, beam, rerank, 1 if trace else 0)).fetchall()
        ids.append([r[0] for r in rows])
        stats.append(json.loads(rows[0][2]) if rows else {})
    return ids, stats, time.time() - t


def recall(found, truth, k):
    return float(np.mean([len(set(f[:k]) & set(t[:k])) / k for f, t in zip(found, truth)]))


def exact_pq(db, Q, k):
    ids = []
    for q in Q:
        ids.append([r[0] for r in db.execute(
            "SELECT rowid FROM v WHERE embedding MATCH ?1 AND k = ?2 AND exact = 2", (q.tobytes(), k))])
    return ids


def warm_cache_sim(stats):
    """Replay page traces through an unbounded cache shared across queries
    (a browser session). Returns mean missed pages and rounds with a miss
    over the second half of the queries."""
    cache, miss_pages, miss_rounds = set(), [], []
    for st in stats:
        mp = mr = 0
        for rnd in st.get("trace", []):
            new = [p for p in rnd if p not in cache]
            if new:
                mr += 1
                mp += len(new)
                cache.update(new)
        miss_pages.append(mp)
        miss_rounds.append(mr)
    h = len(stats) // 2
    return float(np.mean(miss_pages[h:])), float(np.mean(miss_rounds[h:]))


def cold_query_pages(path, Q, k, ef, beam, rerank, nq=10):
    """Distinct file pages read by a brand-new connection answering one
    query (schema, config, codebook, entry set, search), via the counting VFS."""
    boot = sqlite3.connect(":memory:")
    boot.enable_load_extension(True)
    boot.load_extension(COUNTVFS)
    out = []
    for q in Q[:nq]:
        db = connect(path, vfs="countvfs")
        db.load_extension(COUNTVFS)          # registers the SQL functions here
        db.execute("SELECT countvfs_reset()")
        db.execute(SQL, (q.tobytes(), k, ef, beam, rerank, 0)).fetchall()
        out.append(json.loads(db.execute("SELECT countvfs_stats()").fetchone()[0]))
        db.close()
    boot.close()
    return {key: float(np.mean([o[key] for o in out])) for key in out[0]}


SWEEPS = {
    "full": [(ef, beam, rr) for ef in (16, 32, 64, 128) for beam in (1, 4, 8) for rr in (0, 1, 2)],
    "small": [(ef, beam, rr) for ef in (32, 64, 128) for beam in (4,) for rr in (0, 1, 2)] + [(64, 1, 2), (64, 8, 2)],
    "main": [(ef, beam, rr) for ef in (32, 64, 128) for beam in (4, 8, 16) for rr in (0, 2)] + [(256, 16, 2)],
    "wide": [(ef, beam, 2) for ef in (32, 64, 128) for beam in (8, 16, 32, 64)],
    "rr": [(ef, 4, rr) for ef in (16, 32, 64, 128, 256) for rr in (0, 2, 3)],
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="synth")
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--queries", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--params", default="", help="extra dense_ann parameters, e.g. 'M=16, store_vectors=int8'")
    ap.add_argument("--page-size", type=int, default=4096)
    ap.add_argument("--sweep", default="full", choices=sorted(SWEEPS))
    ap.add_argument("--db", default=None, help="database path (default: ext/dense/build/<tag>.db)")
    ap.add_argument("--reuse", action="store_true", help="reuse an existing database")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    tag = args.tag or f"{args.data}{'-' + str(args.n) if args.data == 'synth' else ''}-p{args.page_size}"
    path = args.db or str(CACHE / f"{tag}.db")
    X, Q = load_data(args)
    print(f"# {tag}: {len(X)} x {X.shape[1]}, {len(Q)} queries, params '{args.params}'", flush=True)
    t = time.time()
    gt = exact_topk(X, Q, args.k)
    print(f"ground truth in {time.time() - t:.1f}s", flush=True)

    if args.reuse and os.path.exists(path):
        info = {"reused": True, "db_bytes": os.path.getsize(path), "bytes_per_doc": os.path.getsize(path) / len(X)}
    else:
        info = build_db(path, X, args.params, args.page_size)
        c = info["config"]
        print(f"insert {info['insert_s']:.1f}s, build {info['build_s']:.1f}s "
              f"(pq {c.get('build_ms_pq', 0) / 1e3:.1f}s, graph {c.get('build_ms_graph', 0) / 1e3:.1f}s, "
              f"write {c.get('build_ms_write', 0) / 1e3:.1f}s), finalize {info['finalize_s']:.1f}s, "
              f"row {c.get('row_size')} B, avg degree {c.get('avg_degree', 0):.1f}", flush=True)
    del X
    info["tables"] = table_bytes(path)
    print(f"db {info['db_bytes'] / 2**20:.1f} MiB = {info['bytes_per_doc']:.0f} B/doc; "
          + ", ".join(f"{k} {v / 2**20:.1f} MiB" for k, v in info["tables"].items()), flush=True)

    db = connect(path)
    pq_ids = exact_pq(db, Q, args.k)
    pq_recall = recall(pq_ids, gt, args.k)
    print(f"exact PQ search recall@{args.k} vs float: {pq_recall:.3f}", flush=True)
    if args.data.startswith("words-"):
        print(f"  held-out docs {recall(pq_ids[:200], gt[:200], args.k):.3f}, "
              f"single words {recall(pq_ids[200:], gt[200:], args.k):.3f}", flush=True)

    rows = []
    groups = {"docs": slice(0, 200), "words": slice(200, 400)} if args.data.startswith("words-") else {}
    hdr = ("| ef | W | rerank | recall@10 | vs PQ-exact | rounds | pages | KiB | expanded | dists "
           "| warm miss pages | warm miss rounds | QPS |" + "".join(f" R@10 {g} |" for g in groups))
    print(hdr)
    print("|---" * (hdr.count("|") - 1) + "|")
    run_queries(db, Q[:5], args.k, 64, 4, 0)   # load codebook / entry set, warm caches
    for ef, beam, rr in SWEEPS[args.sweep]:
        ids, stats, secs = run_queries(db, Q, args.k, ef, beam, rr, trace=True)
        wp, wr = warm_cache_sim(stats)
        r = {
            "ef": ef, "beam": beam, "rerank": rr,
            "recall": recall(ids, gt, args.k), "recall_vs_pq": recall(ids, pq_ids, args.k),
            "rounds": float(np.mean([s["rounds"] for s in stats])),
            "pages": float(np.mean([s["pages"] for s in stats])),
            "kib": float(np.mean([s["bytes"] for s in stats])) / 1024,
            "expanded": float(np.mean([s["expanded"] for s in stats])),
            "dist": float(np.mean([s["dist"] for s in stats])),
            "fallback": float(np.mean([s["fallback"] for s in stats])),
            "warm_miss_pages": wp, "warm_miss_rounds": wr,
            "qps": len(Q) / secs, "ms_internal": float(np.mean([s["ms"] for s in stats])),
        }
        for g, sl in groups.items():
            r["recall_" + g] = recall(ids[sl], gt[sl], args.k)
        rows.append(r)
        print(f"| {ef} | {beam} | {rr} | {r['recall']:.3f} | {r['recall_vs_pq']:.3f} | {r['rounds']:.1f} "
              f"| {r['pages']:.1f} | {r['kib']:.0f} | {r['expanded']:.1f} | {r['dist']:.0f} "
              f"| {wp:.1f} | {wr:.1f} | {r['qps']:.0f} |"
              + "".join(f" {r['recall_' + g]:.3f} |" for g in groups), flush=True)
        if r["fallback"]:
            print(f"  warning: {r['fallback']:.1f} SQL fallbacks per query (stale page hints?)")
    db.close()

    cold = cold_query_pages(path, Q, args.k, 64, 16, 2)
    print(f"cold connection, one query (ef=64, W=16, rerank=2): {cold['distinct']:.0f} distinct pages, "
          f"{cold['reads']:.0f} reads, {cold['bytes'] / 1024:.0f} KiB", flush=True)

    out = HERE / "results"
    out.mkdir(exist_ok=True)
    with open(out / f"{tag}.json", "w") as f:
        json.dump({"tag": tag, "args": vars(args), "info": info, "pq_exact_recall": pq_recall,
                   "cold_first_query": cold, "sweep": rows}, f, indent=1)


if __name__ == "__main__":
    main()
