#!/usr/bin/env python3
"""Markdown tables from ext/late/results/*.json (used to write NOTES.md).

  python3 ext/late/report.py w10k-n2-k16384 [more tags...] [--kind all|word|known] [--net]
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def fmt(x, nd=3):
    return "–" if x is None else f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--kind", default="all")
    ap.add_argument("--net", action="store_true", help="network columns instead of quality")
    ap.add_argument("--label", action="store_true", help="prefix rows with the tag")
    a = ap.parse_args()
    if a.net:
        print("| Configuration | Rounds | KB read (p50 / p90) | Candidates | Native ms | 4g h2 (p50 / p90) | lte h2 | slow-4g h2 | 4g h1 |")
        print("|---|---|---|---|---|---|---|---|---|")
    else:
        print("| Configuration | R@10 vs exact | nDCG@10 | MRR@10 | Recall@10 | Success@1 | AUC | Rounds | KB read | 4g h2 ms |")
        print("|---|---|---|---|---|---|---|---|---|---|")
    for tag in a.tags:
        d = json.loads((HERE / "results" / f"{tag}.json").read_text())
        for r in d["results"]:
            s = r.get(a.kind) or r["all"]
            name = (tag + ": " if a.label else "") + r["opts"]
            if a.net:
                if "rounds" not in s:
                    continue
                print(f"| {name} | {s['rounds']:.2f} | {s['kb_p50']:.0f} / {s['kb_p90']:.0f} | {s['candidates']:.0f} | "
                      f"{s['wall_ms']:.1f} | {s['net_4g,h2_p50']:.0f} / {s['net_4g,h2_p90']:.0f} | "
                      f"{s['net_lte,h2_p50']:.0f} | {s['net_slow-4g,h2_p50']:.0f} | {s['net_4g,h1_p50']:.0f} |")
            else:
                print(f"| {name} | {fmt(s.get('recall10_exact'))} | {fmt(s['ndcg10'])} | {fmt(s['mrr10'])} | "
                      f"{fmt(s['recall10'])} | {fmt(s['success1'])} | {fmt(s['auc'])} | "
                      f"{fmt(s.get('rounds'), 2)} | {fmt(s.get('kb'), 0)} | {fmt(s.get('net_4g,h2_p50'), 0)} |")


if __name__ == "__main__":
    main()
