"""Compare the qint8 MiniLM ONNX model against the fp32 reference and time encoding.

    python3 -m enc.validate_minilm [--fp32 /path/to/model.onnx] [--threads 2]

Reports per-text cosine similarity between qint8 and fp32 embeddings, retrieval
agreement (queries = the LLM corpus seed words, documents = LLM paragraphs plus
random-word documents), and encode throughput for 50-word documents.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from enc.minilm import MiniLM, REPO


def first_words(text, n=50):
    return " ".join(text.split()[:n])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp32", default="/tmp/claude-0/npmx/package/onnx/model.onnx")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--n-words-docs", type=int, default=1000)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(REPO / "data/llm/paragraphs-10k.jsonl")]
    paras = [first_words(r["text"]) for r in rows]
    words = [r["word"] for r in rows]
    wdocs = open(REPO / "data/corpora/words-10k.txt").read().split("\n")[: args.n_words_docs]
    docs = paras + wdocs
    queries = [f"{w}" for w in words] + [f"a paragraph about {w}" for w in words]

    q8 = MiniLM(threads=args.threads)
    f32 = MiniLM(model_path=args.fp32, threads=args.threads)

    Dq, Df = q8.encode(docs), f32.encode(docs)
    Qq, Qf = q8.encode(queries), f32.encode(queries)
    cos_d = (Dq * Df).sum(1)
    cos_q = (Qq * Qf).sum(1)
    allcos = np.concatenate([cos_d, cos_q])
    print(f"texts compared: {len(allcos)}")
    print(f"cosine(qint8, fp32): mean {allcos.mean():.4f}  min {allcos.min():.4f}  "
          f"p1 {np.percentile(allcos, 1):.4f}  median {np.median(allcos):.4f}")
    print(f"  LLM paragraphs: mean {cos_d[:len(paras)].mean():.4f}; random-word docs: mean "
          f"{cos_d[len(paras):].mean():.4f}; queries: mean {cos_q.mean():.4f}")

    Sq, Sf = Qq @ Dq.T, Qf @ Df.T
    k = 10
    tq = np.argsort(-Sq, 1)[:, :k]
    tf = np.argsort(-Sf, 1)[:, :k]
    top1 = (tq[:, 0] == tf[:, 0]).mean()
    overlap = np.mean([len(set(a) & set(b)) / k for a, b in zip(tq, tf)])
    gold = np.concatenate([np.arange(len(paras))] * 2)
    r1q = (tq[:, 0] == gold).mean()
    r1f = (tf[:, 0] == gold).mean()
    r10q = np.mean([g in t for g, t in zip(gold, tq)])
    r10f = np.mean([g in t for g, t in zip(gold, tf)])
    # Spearman correlation of full score vectors, averaged over queries
    def ranks(x):
        return np.argsort(np.argsort(x, 1), 1).astype(np.float64)
    rq, rf = ranks(Sq), ranks(Sf)
    rq -= rq.mean(1, keepdims=True); rf -= rf.mean(1, keepdims=True)
    spear = ((rq * rf).sum(1) / np.sqrt((rq ** 2).sum(1) * (rf ** 2).sum(1))).mean()
    print(f"retrieval over {len(docs)} docs, {len(queries)} queries:")
    print(f"  top-1 agreement {top1:.3f}, top-{k} overlap {overlap:.3f}, mean Spearman {spear:.4f}")
    print(f"  recall@1 of the seed-word paragraph: qint8 {r1q:.3f}, fp32 {r1f:.3f}; "
          f"recall@10: qint8 {r10q:.3f}, fp32 {r10f:.3f}")

    # throughput on 50-word random-word documents
    bench = open(REPO / "data/corpora/words-10k.txt").read().split("\n")[:1000]
    for name, m in (("qint8", q8), ("fp32", f32)):
        m.encode(bench[:64])
        t = time.perf_counter(); m.encode(bench); dt = time.perf_counter() - t
        print(f"throughput {name}: {len(bench) / dt:.1f} docs/s (50-word random-word docs, "
              f"{args.threads} threads)")
    ntok = np.mean([len(e.ids) for e in q8.tokenize(bench)])
    ntokp = np.mean([len(e.ids) for e in q8.tokenize(paras)])
    print(f"mean word pieces incl. [CLS]/[SEP]: random-word {ntok:.1f}, LLM paragraphs {ntokp:.1f}")


if __name__ == "__main__":
    main()
