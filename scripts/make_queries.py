#!/usr/bin/env python3
"""Build query sets with relevance judgements for the random-word corpora.

Two query kinds, 1,000 of each per corpus (fewer if the corpus is smaller):

* ``word``  — a single word drawn uniformly from the 74,744-word vocabulary,
  restricted to words that occur in the corpus. Relevant documents are all
  documents containing that exact word. This is the natural keyword query.
* ``known`` — known-item search: pick a target document uniformly, then three
  distinct words from it; the query is those words in random order. The
  target is the relevant document (plus any other document that happens to
  contain all three words, which is rare).

Randomness comes from SplitMix64 (scripts/detshuffle.py), seeded per corpus
size, so the query sets are reproducible.

Usage: make_queries.py [label ...]   (default: 100 10k 1m)
Writes data/queries/words-<label>.jsonl, one JSON object per line:
  {"qid": ..., "kind": "word"|"known", "text": ..., "relevant": [doc ids]}
Document ids are 0-based line numbers in data/corpora/words-<label>.txt.
"""
import json, pathlib, sys
from collections import defaultdict
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from detshuffle import splitmix64, shuffled

ROOT = pathlib.Path(__file__).resolve().parent.parent
N_QUERIES = 1000
SEEDS = {"100": 100, "10k": 10_000, "1m": 1_000_000}


def main():
    labels = sys.argv[1:] or ["100", "10k", "1m"]
    out = ROOT / "data/queries"; out.mkdir(parents=True, exist_ok=True)
    for label in labels:
        docs = (ROOT / f"data/corpora/words-{label}.txt").read_text(encoding="utf-8").split("\n")[:-1]
        postings = defaultdict(list)
        for i, d in enumerate(docs):
            for w in set(d.split(" ")):
                postings[w].append(i)
        rng = splitmix64(SEEDS[label])
        rows = []
        vocab = sorted(postings)
        for k, w in enumerate(shuffled(vocab, SEEDS[label])[:N_QUERIES]):
            rows.append({"qid": f"word-{k}", "kind": "word", "text": w, "relevant": postings[w]})
        n_known = min(N_QUERIES, len(docs))
        targets = shuffled(range(len(docs)), SEEDS[label] + 1)[:n_known]
        for k, t in enumerate(targets):
            ws = shuffled(sorted(set(docs[t].split(" "))), next(rng))[:3]
            rel = sorted(set.intersection(*(set(postings[w]) for w in ws)))
            rows.append({"qid": f"known-{k}", "kind": "known", "text": " ".join(ws), "target": t, "relevant": rel})
        with (out / f"words-{label}.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        nrel = [len(r["relevant"]) for r in rows if r["kind"] == "word"]
        print(label, len(rows), "queries; word-query mean relevant docs", sum(nrel) / len(nrel))


if __name__ == "__main__":
    main()
