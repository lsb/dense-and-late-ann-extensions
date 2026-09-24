#!/usr/bin/env python3
"""Graph vs IVF at matched recall, from bench.py result files.

For each (dataset, index) pair and target recall, picks the swept query
configuration with the lowest simulated 4g time whose recall@10 reaches the
target, and prints a markdown table with rounds, bytes, simulated 4g /
slow-4g times (netsim presets; one query on a warm connection), the one-time
setup cost, index bytes per document and build time.

  python3 ext/dense/compare.py [--targets=0.8,0.9,0.95] synth1m-graph:synth1m-ivf-cpq w10k-graph:w10k-ivf ...
"""
import json
import sys
from pathlib import Path

RES = Path(__file__).resolve().parent / "results"


def load(tag):
    return json.loads((RES / f"{tag}.json").read_text())


def best(res, target):
    ok = [r for r in res["sweep"] if r["recall"] >= target and "net_ms_4g" in r]
    return min(ok, key=lambda r: r["net_ms_4g"]) if ok else None


def build_s(res):
    info = res["info"]
    return info.get("build_s")


def main(pairs, targets=(0.9, 0.95)):
    print("| data | target | index | config | recall@10 | rounds | KiB/query | 4g ms | slow-4g ms "
          "| setup KiB | setup 4g ms | B/doc | build s |")
    print("|---" * 13 + "|")
    for pair in pairs:
        for target in targets:
            for tag in pair.split(":"):
                res = load(tag)
                r = best(res, target)
                setup = res.get("setup", {})
                name = "IVF-PQ" if "nprobe" in (res["sweep"][0] if res["sweep"] else {}) else "graph"
                bs = build_s(res)
                cells = [res["args"]["data"] + (f"-{res['args']['n']}" if res["args"]["data"] == "synth" else ""),
                         f"{target:.2f}", f"{name} ({tag})"]
                if r is None:
                    cells += ["not reached"] + [""] * 9
                else:
                    cells += [r["config"], f"{r['recall']:.3f}", f"{r['rounds']:.1f}", f"{r['kib']:.0f}",
                              f"{r['net_ms_4g']:.0f}", f"{r['net_ms_slow-4g']:.0f}",
                              f"{setup.get('setup_bytes', 0) / 1024:.0f}", f"{setup.get('setup_ms_4g', 0):.0f}",
                              f"{res['info'].get('bytes_per_doc', 0):.0f}", f"{bs:.0f}" if bs else "reused"]
                print("| " + " | ".join(cells) + " |")


if __name__ == "__main__":
    args = sys.argv[1:]
    targets = (0.9, 0.95)
    if args and args[0].startswith("--targets="):
        targets = tuple(float(t) for t in args.pop(0).split("=", 1)[1].split(","))
    main(args, targets)
