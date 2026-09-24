#!/usr/bin/env python3
"""End-to-end check of the per-round request budget: the WebAssembly build
(Node, Asyncify, bench/matrix.mjs) against netsim/rangeserver.py shaped with
<profile>,h1 (six requests in service, as a browser over HTTP/1.1), for a few
configurations of the benchmark matrix, in three modes:

  baseline    open() defaults in Node: no request budget
  coalesce    {maxRequests: 6}: merging under the cost model, RTT and
              bandwidth estimated online from the defaults (100 ms, 10 Mbit/s)
  multipart   {maxRequests: 6, multipart: true}: multi-range requests

Every mode runs the same queries (warm: one connection, after one warm-up
query; cold: a new connection per query). The baseline run's own request
logs are also re-simulated with bench/coalesce_eval.py's transforms, so the
table shows the simulator's prediction for coalesce and multipart next to the
measured times.

  python3 bench/coalesce_real.py words-10k --profiles 4g,lte
"""
import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "bench"))

import coalesce_eval as ce  # noqa: E402
import matrix as mx  # noqa: E402
from netsim import netmodel as nm  # noqa: E402
from netsim import simulate as sim  # noqa: E402

MODES = {
    "baseline": {},
    "coalesce": {"maxRequests": 6},
    "multipart": {"maxRequests": 6, "multipart": True},
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("corpus")
    ap.add_argument("--matrix-dir", default=str(REPO / "build" / "matrix"))
    ap.add_argument("--queries-dir", default=str(REPO / "build" / "queries"))
    ap.add_argument("--profiles", default="4g,lte")
    ap.add_argument("--warm", type=int, default=20)
    ap.add_argument("--cold", type=int, default=8)
    ap.add_argument("--configs", default=None, help="default: the systems of coalesce_eval.py")
    ap.add_argument("--out-dir", default=str(REPO / "build" / "coalesce"))
    ap.add_argument("--parallel", type=int, default=2)
    args = ap.parse_args()
    mx.BUILD = Path(args.matrix_dir)
    mx.QDIR = Path(args.queries_dir)
    cfg = mx.load_config()
    man, meta = mx.manifest(args.corpus), mx.qmeta(args.corpus)
    fam = "llm" if args.corpus.startswith("llm") else "words"
    names = args.configs.split(",") if args.configs else list(ce.SYSTEMS[fam].values())
    confs = [c for c in mx.configs_for(cfg, man) if c["id"] in names]
    qs = mx.all_queries(cfg, meta)
    warm_q = qs[::max(1, len(qs) // args.warm)][:args.warm]
    cold_q = qs[1::max(1, len(qs) // args.cold)][:args.cold]
    runs = []
    for c in confs:
        runs.append(mx.node_run(c, cfg["run"]["k"], "warm", warm_q, [qs[-1]]))
        runs.append(mx.node_run(c, cfg["run"]["k"], "cold", cold_q, []))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def one(profile):
        spec = f"{profile},h1"
        srv = mx.Server(spec, seed=1)
        mx.BUILD = Path(args.matrix_dir)
        res = {}
        try:
            for mode, opts in MODES.items():
                out = out_dir / f"{args.corpus}.{profile}_h1.{mode}.jsonl"
                t = time.time()
                job = {"url": f"{srv.base}/{man['file']}", "queries": str(mx.QDIR / f"{args.corpus}.json"),
                       "open": opts, "phases": [{"runs": runs}]}
                job_path = Path(str(out) + ".job.json")
                job_path.write_text(json.dumps(job))
                out.unlink(missing_ok=True)
                r = subprocess.run(["node", str(REPO / "bench" / "matrix.mjs"), str(job_path), str(out)],
                                   capture_output=True, text=True)
                job_path.unlink()
                if r.returncode:
                    raise RuntimeError(r.stderr[-3000:])
                res[mode] = [json.loads(line) for line in open(out)]
                print(f"[{spec}] {mode}: {len(res[mode])} queries in {time.time() - t:.0f} s", flush=True)
        finally:
            srv.close()
        return profile, res

    with ThreadPoolExecutor(args.parallel) as ex:
        results = dict(ex.map(one, args.profiles.split(",")))

    rows = []
    for profile, res in results.items():
        prof = nm.preset(f"{profile},h1")
        base = nm.preset(profile)
        for c in confs:
            for regime in ("warm", "cold"):
                row = {"profile": profile, "config": c["id"], "regime": regime}
                recs = {m: [r for r in res[m] if r["config"] == c["id"] and r["regime"] == regime] for m in MODES}
                ids_ok = all(r["ids"] == b["ids"] for m in MODES for r, b in zip(recs[m], recs["baseline"]))
                row["same_results"] = ids_ok
                for m in MODES:
                    w = [r["wall_ms"] for r in recs[m]]
                    row[m] = {"p50": ce.pct(w, 0.5), "p95": ce.pct(w, 0.95), "n": len(w),
                              "requests": sum(r["stats"]["requests"] for r in recs[m]) / max(1, len(w)),
                              "kb": sum(r["stats"]["bytes"] for r in recs[m]) / max(1, len(w)) / 1024,
                              "rounds": sum(r["stats"]["rounds"] for r in recs[m]) / max(1, len(w))}
                # Prediction from the baseline run's own logs.
                for m, mode, cc in [("sim_baseline", "baseline", 0), ("sim_coalesce", "coalesce", 6),
                                    ("sim_multipart", "multipart", 6)]:
                    t = []
                    for r in recs["baseline"]:
                        tr, _ = ce.transform(ce.record_trace(r), mode, cc, base.latency_ms, base.link_kbps)
                        t.append(sim.simulate(tr, prof, seed=r["qi"]).total_ms)
                    row[m] = {"p50": ce.pct(t, 0.5), "p95": ce.pct(t, 0.95)}
                rows.append(row)
    path = REPO / "results" / "coalesce" / f"{args.corpus}.real.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"corpus": args.corpus, "profiles": args.profiles.split(","), "warm": len(warm_q),
                                "cold": len(cold_q), "rows": rows}, indent=1))
    print(f"-> {path}")
    for r in rows:
        print(f"{r['profile']:>7} {r['regime']:>4} {r['config']:<15} same={r['same_results']!s:<5} " + "  ".join(
            f"{m} {r[m]['p50']:.0f}/{r[m]['p95']:.0f}" for m in
            ["baseline", "sim_baseline", "coalesce", "sim_coalesce", "multipart", "sim_multipart"]))


if __name__ == "__main__":
    main()
