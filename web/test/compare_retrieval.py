#!/usr/bin/env python3
"""Does the browser/Python encoding difference change retrieval results?

Exact search over words-10k (MiniLM cosine, LateOn MaxSim over the float16
corpus vectors) with the browser encodings (build/web/browser-embeddings.json,
written by web/test/web.test.mjs) and with the Python reference encodings
(build/web/reference-embeddings.json); prints the mean top-10 overlap. For
scale it also compares the Python encoder with and without ONNX Runtime graph
optimisations, two encodings that are equally "correct".

  python3 web/test/compare_retrieval.py [--data words-10k] [--k 10]
"""
import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="words-10k")
    ap.add_argument("--k", type=int, default=10)
    a = ap.parse_args()
    ref = json.loads((REPO / "build/web/reference-embeddings.json").read_text())["queries"]
    br = json.loads((REPO / "build/web/browser-embeddings.json").read_text())
    emb = REPO / "data" / "emb"
    D = np.load(emb / f"{a.data}.minilm.npy").astype(np.float32)
    V = np.load(emb / f"{a.data}.lateon.vectors.npy").astype(np.float32)
    off = np.load(emb / f"{a.data}.lateon.offsets.npy")
    k = a.k

    def top_dense(q):
        return set(np.argsort(-(D @ np.asarray(q, np.float32)))[:k])

    def top_late(q):
        q = np.asarray(q, np.float32).reshape(-1, 48)
        s = q @ V.T                                        # [nq, T]
        per_doc = np.maximum.reduceat(s, off[:-1], axis=1)  # [nq, N]
        return set(np.argsort(-per_doc.sum(0))[:k])

    out = {}
    for which, top in (("minilm", top_dense), ("lateon", top_late)):
        ov_b, ov_o = [], []
        for i, q in enumerate(ref):
            r = top(q[which])
            ov_b.append(len(r & top(br[which][i])) / k)
            ov_o.append(len(r & top(q[which + "_noopt"])) / k)
        out[which] = {"browser_vs_python": float(np.mean(ov_b)), "python_opt_vs_noopt": float(np.mean(ov_o)),
                      "browser_identical_top10": float(np.mean(np.array(ov_b) == 1.0)), "n": len(ref)}
        print(f"{which}: top-{k} overlap browser vs Python {np.mean(ov_b):.3f} "
              f"(identical for {100 * np.mean(np.array(ov_b) == 1.0):.0f}% of queries); "
              f"Python optimised vs unoptimised {np.mean(ov_o):.3f}")
    (REPO / "build/web/retrieval-agreement.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
