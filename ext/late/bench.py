#!/usr/bin/env python3
"""Evaluate late_plaid indexes: quality, native latency, and httpvfs cost.

For every query of a query set (encoded by encode_queries.py) and every
search configuration, runs

  SELECT rowid, score, stats FROM t WHERE t MATCH ?1 AND k = ?2 AND opts = ?3

and records:
  * recall@10 against exact MaxSim over the original float16 vectors
    (ground truth computed here with NumPy), and against the relevance
    labels: recall@10, MRR@10, nDCG@10, success@1, AUC (bench/metrics.py);
  * native wall time per query (noisy: the machine is shared);
  * the extension's own instrumentation: centroids probed, candidates,
    list bytes, documents fetched, dependent rounds, pages per round;
  * simulated network time (netsim/simulate.py) of the recorded page
    trace under 4g, lte and slow-4g, with HTTP/2-like concurrency (h2) and
    the HTTP/1.1 six-connection limit (h1). Adjacent pages of a round are
    coalesced into one range request.

The static data (centroids etc.) is loaded once per connection; its cost is
reported separately ("session start"). Before each query the extension's
interior-page cache is kept (steady state of a browser session), or dropped
with --cold.

  python3 ext/late/bench.py --db build/late/w10k-n2-k16384.db --queries words-10k \
      --configs "layout=warp nprobe=4" "layout=plaid approx=codes nprobe=4 ndocs=256" --tag x
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "bench"))
sys.path.insert(0, str(REPO / "netsim"))

import metrics  # noqa: E402
from netsim import netmodel as nm  # noqa: E402
from netsim import simulate as sim  # noqa: E402
from build_index import connect  # noqa: E402

PRESETS = ["4g,h2", "lte,h2", "slow-4g,h2", "4g,h1"]


def load_queries(name):
    z = np.load(REPO / "build" / "late" / f"queries-{name}.npz", allow_pickle=True)
    V, O, K, I, R = z["vectors"], z["offsets"], z["kinds"], z["qids"], z["relevant"]
    qs = []
    for i in range(len(O) - 1):
        qs.append({
            "vec": np.ascontiguousarray(V[O[i]:O[i + 1]], np.float32),
            "kind": str(K[i]), "qid": str(I[i]),
            "relevant": set(int(x) for x in R[i]),
        })
    return qs


def ground_truth(data, qname, qs, depth=100):
    path = REPO / "build" / "late" / f"gt-{data}-{qname}.npz"
    if path.exists():
        z = np.load(path)
        return z["ids"], z["scores"]
    V = np.load(REPO / "data" / "emb" / f"{data}.lateon.vectors.npy").astype(np.float32)
    O = np.load(REPO / "data" / "emb" / f"{data}.lateon.offsets.npy")
    ids = np.zeros((len(qs), depth), np.int64)
    scores = np.zeros((len(qs), depth), np.float32)
    t = time.time()
    for i, q in enumerate(qs):
        s = V @ q["vec"].T                                  # [T, nq]
        per_doc = np.maximum.reduceat(s, O[:-1], axis=0).sum(1)
        top = np.argsort(-per_doc, kind="stable")[:depth]
        ids[i, :len(top)] = top
        scores[i, :len(top)] = per_doc[top]
        if i % 500 == 0:
            print(f"  ground truth {i}/{len(qs)} ({time.time() - t:.0f} s)", file=sys.stderr)
    np.savez(path, ids=ids, scores=scores)
    return ids, scores


def trace_of(stats, skip_rounds=0):
    """netsim trace (list of rounds of (offset, length)) from the page trace."""
    ps = stats["page_size"]
    rounds = []
    for rnd in stats.get("trace", [])[skip_rounds:]:
        rounds.append([((p - 1) * ps, ps) for p in sorted(rnd)])
    return rounds


def sim_ms(rounds, preset, seed=0, merge=True):
    if not rounds:
        return 0.0, 0
    tr = sim.make_trace(rounds)
    if merge:
        tr = tr.merged(0)
    return sim.simulate(tr, nm.preset(preset), seed=seed).total_ms, len(tr.reads)


def run_config(con, qs, gt_ids, opts, k, n_docs, warm=False, sim_every=1):
    rows = []
    for qi, q in enumerate(qs):
        if not warm:
            con.execute("INSERT INTO t(t) VALUES ('drop_cache')")
        t = time.perf_counter()
        res = list(con.execute("SELECT rowid, score, stats FROM t WHERE t MATCH ? AND k = ? AND opts = ?",
                               (q["vec"].tobytes(), k, opts + " trace=1")))
        wall = (time.perf_counter() - t) * 1000
        st = json.loads(res[0][2]) if res else {}
        ranked = [r[0] for r in res]
        exact10 = set(int(x) for x in gt_ids[qi, :10])
        row = {
            "kind": q["kind"],
            "recall10_exact": len(set(ranked[:10]) & exact10) / 10,
            "recall10": metrics.recall_at(ranked, q["relevant"], 10),
            "mrr10": metrics.mrr_at(ranked, q["relevant"], 10),
            "ndcg10": metrics.ndcg_at(ranked, q["relevant"], 10),
            "success1": metrics.success_at(ranked, q["relevant"], 1),
            "auc": metrics.auc(ranked, q["relevant"], n_docs),
            "wall_ms": wall, "ext_ms": st.get("ms", 0), "cpu_ms": st.get("cpu_ms", 0),
            "probed": st.get("probed", 0), "candidates": st.get("candidates", 0),
            "list_bytes": st.get("list_bytes", 0), "docs_fetched": st.get("docs_fetched", 0),
            "rounds": st.get("rounds", 0) - st.get("static_rounds", 0),
            "pages": sum(len(r) for r in st.get("trace", [])[st.get("static_rounds", 0):]),
            "tokens_decoded": st.get("tokens_decoded", 0),
        }
        row["kb"] = row["pages"] * st.get("page_size", 4096) / 1024
        if qi % sim_every == 0:
            rounds = trace_of(st, st.get("static_rounds", 0))
            for p in PRESETS:
                row["net_" + p], row["req_" + p] = sim_ms(rounds, p, seed=qi)
        rows.append(row)
    return rows


def summarize(rows):
    out = {}
    keys = [k for k in rows[0] if isinstance(rows[0][k], (int, float))]
    for kk in keys:
        vals = [r[kk] for r in rows if kk in r and not (isinstance(r[kk], float) and np.isnan(r[kk]))]
        if not vals:
            continue
        out[kk] = float(np.mean(vals))
        if kk in ("wall_ms", "rounds", "kb") or kk.startswith("net_"):
            out[kk + "_p50"] = float(np.percentile(vals, 50))
            out[kk + "_p90"] = float(np.percentile(vals, 90))
    return out


def session_start(db):
    """Cost of loading the static data on a fresh connection."""
    con = connect(db)
    q = np.random.default_rng(0).standard_normal((4, 48)).astype(np.float32)
    res = list(con.execute("SELECT stats FROM t WHERE t MATCH ? AND k=1 AND opts='trace=1'", (q.tobytes(),)))
    st = json.loads(res[0][0])
    rounds = trace_of(st)[:st["static_rounds"]]
    out = {"static_bytes": st["static_bytes"], "static_rounds": st["static_rounds"],
           "static_pages": sum(len(r) for r in rounds)}
    for p in PRESETS:
        out["net_" + p] = sim_ms(rounds, p)[0]
    con.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--data", default="words-10k")
    ap.add_argument("--queries", default=None)
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--limit", type=int, default=0, help="use only the first N queries of each kind")
    ap.add_argument("--cold", action="store_true", help="drop the interior-page cache before every query")
    ap.add_argument("--sim-every", type=int, default=1)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    qname = a.queries or a.data
    qs = load_queries(qname)
    if a.limit:
        byk = {}
        qs = [q for q in qs if byk.setdefault(q["kind"], []).append(q) or len(byk[q["kind"]]) <= a.limit]
    n_docs = len(np.load(REPO / "data" / "emb" / f"{a.data}.lateon.offsets.npy")) - 1
    gt_ids, _ = ground_truth(a.data, qname, load_queries(qname))
    if a.limit:
        allq = load_queries(qname)
        keep = {q["qid"] for q in qs}
        gt_ids = gt_ids[[i for i, q in enumerate(allq) if q["qid"] in keep]]
    meta_path = Path(a.db + ".json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    out = {"db": a.db, "build": meta, "queries": qname, "n_queries": len(qs), "warm": not a.cold,
           "session": session_start(a.db), "results": []}
    # reference: exact MaxSim over the original float16 vectors
    ref = []
    for qi, q in enumerate(qs):
        ranked = [int(x) for x in gt_ids[qi]]
        ref.append({"kind": q["kind"], "recall10_exact": 1.0,
                    "recall10": metrics.recall_at(ranked, q["relevant"], 10),
                    "mrr10": metrics.mrr_at(ranked, q["relevant"], 10),
                    "ndcg10": metrics.ndcg_at(ranked, q["relevant"], 10),
                    "success1": metrics.success_at(ranked, q["relevant"], 1),
                    "auc": metrics.auc(ranked, q["relevant"], n_docs)})
    res = {"opts": "float-exact", "all": summarize(ref)}
    for kind in sorted({r["kind"] for r in ref}):
        res[kind] = summarize([r for r in ref if r["kind"] == kind])
    out["results"].append(res)
    s = res["all"]
    print(f"{'float-exact (reference)':50s} nDCG={s['ndcg10']:.3f} MRR={s['mrr10']:.3f}", flush=True)
    con = connect(a.db)
    for opts in a.configs:
        t = time.time()
        rows = run_config(con, qs, gt_ids, opts, a.k, n_docs, not a.cold, a.sim_every)
        res = {"opts": opts, "all": summarize(rows)}
        for kind in sorted({r["kind"] for r in rows}):
            res[kind] = summarize([r for r in rows if r["kind"] == kind])
        out["results"].append(res)
        s = res["all"]
        print(f"{opts:50s} R@10ex={s['recall10_exact']:.3f} nDCG={s['ndcg10']:.3f} MRR={s['mrr10']:.3f} "
              f"rounds={s['rounds']:.2f} KB={s['kb']:.0f} cand={s['candidates']:.0f} "
              f"ms={s['wall_ms']:.1f} 4g={s.get('net_4g,h2', 0):.0f}ms ({time.time() - t:.0f}s)", flush=True)
    res_dir = HERE / "results"
    res_dir.mkdir(exist_ok=True)
    (res_dir / f"{a.tag}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
