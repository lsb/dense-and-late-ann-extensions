"""Sanity checks, token statistics and throughput for the LateOn-Code-edge encoder.

    python3 -m enc.validate_lateon [--threads 2]
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from enc.lateon import LateOn, maxsim, REPO

EXAMPLES = [
    ("how do I open a database connection in python",
     ["import sqlite3\nconn = sqlite3.connect('app.db')\ncur = conn.cursor()",
      "def fib(n):\n    return n if n < 2 else fib(n-1) + fib(n-2)",
      "The recipe calls for two cups of flour and a pinch of salt.",
      "body { margin: 0; font-family: sans-serif; }"]),
    ("reverse a string",
     ["def reverse(s):\n    return s[::-1]",
      "SELECT name FROM users WHERE age > 30;",
      "The Eiffel Tower is located in Paris, France.",
      "for i in range(10): print(i)"]),
    ("what is the capital of France",
     ["Paris is the capital and largest city of France.",
      "Photosynthesis converts light energy into chemical energy in plants.",
      "The stock market fell sharply on Monday amid inflation fears.",
      "def add(a, b): return a + b"]),
    ("effects of caffeine on sleep",
     ["Drinking coffee late in the day can delay sleep onset and reduce deep sleep.",
      "The Pacific Ocean is the largest and deepest of Earth's oceans.",
      "Use git rebase to rewrite commit history on a feature branch.",
      "Tigers are the largest living cat species."]),
    ("sort a list of dictionaries by a key",
     ["people.sort(key=lambda p: p['age'])",
      "The violin has four strings tuned in perfect fifths.",
      "Mount Everest is 8,849 metres tall.",
      "x = requests.get(url).json()"]),
]


def first_words(text, n=50):
    return " ".join(text.split()[:n])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    m = LateOn(threads=args.threads)

    # 1. handmade ranking examples (relevant document is listed first)
    ok = 0
    for q, docs in EXAMPLES:
        qe = m.encode_queries([q])[0]
        de = m.encode_documents(docs)
        s = [maxsim(qe, d) for d in de]
        ok += int(np.argmax(s) == 0)
        print(f"{q!r}: " + ", ".join(f"{x:.2f}" for x in s) + ("  OK" if np.argmax(s) == 0 else "  WRONG"))
    print(f"handmade examples ranked correctly: {ok}/{len(EXAMPLES)}")

    # 2. padding / batching invariance and unit norms
    docs = [d for _, ds in EXAMPLES for d in ds]
    alone = [m.encode_documents([d])[0] for d in docs]
    batched = m.encode_documents(docs, batch_size=len(docs))
    diff = max(float(np.abs(a - b).max()) for a, b in zip(alone, batched))
    norms = np.concatenate([np.linalg.norm(b, axis=1) for b in batched])
    print(f"max |alone - padded batch| = {diff:.2e}; token norms in [{norms.min():.4f}, {norms.max():.4f}]")
    q = m.encode_queries(["Reverse a string"])[0]
    print(f"query 'Reverse a string' -> {q.shape[0]} vectors, ids {m.token_ids(['Reverse a string'], True)[0]}")

    # 3. token statistics for 50-word documents
    words = open(REPO / "data/corpora/words-10k.txt").read().split("\n")[:10000]
    words = [w for w in words if w]
    rows = [json.loads(l) for l in open(REPO / "data/llm/paragraphs-10k.jsonl")]
    paras = [first_words(r["text"]) for r in rows]
    for name, corpus in (("random-word (words-10k)", words), (f"LLM paragraphs ({len(paras)} rows)", paras)):
        seqs = m.token_ids(corpus, False)
        n_in = np.array([len(s) for s in seqs])
        n_out = np.array([int((~np.isin(np.asarray(s), m.skiplist_ids)).sum()) for s in seqs])
        print(f"{name}: model input tokens mean {n_in.mean():.1f} (min {n_in.min()}, max {n_in.max()}); "
              f"stored vectors after skiplist mean {n_out.mean():.1f}")

    # 4. throughput
    bench = words[:600]
    m.encode_documents(bench[:32])
    for bs in (16, 64):
        t = time.perf_counter(); m.encode_documents(bench, batch_size=bs); dt = time.perf_counter() - t
        print(f"throughput (random-word 50-word docs, batch {bs}, {args.threads} threads): {len(bench) / dt:.1f} docs/s")
    bench = paras[:300]
    t = time.perf_counter(); m.encode_documents(bench); dt = time.perf_counter() - t
    print(f"throughput (LLM 50-word docs, batch 32, {args.threads} threads): {len(bench) / dt:.1f} docs/s")
    qs = [r["word"] for r in rows[:200]]
    t = time.perf_counter(); m.encode_queries(qs); dt = time.perf_counter() - t
    print(f"query encoding (single words): {len(qs) / dt:.1f} queries/s")


if __name__ == "__main__":
    main()
