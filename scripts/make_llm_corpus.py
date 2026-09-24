#!/usr/bin/env python3
"""Build the LLM corpora and their query sets from the generated jsonl files.

Corpus: document i is the first 50 whitespace-separated words of the paragraph
the model wrote about word i (data/llm/paragraphs-10k.jsonl), joined by single
spaces. The 100-document corpus is the first 100 documents of the 10k corpus.

Queries (all known-item: the relevant document is the one written about the
query's word):
  word  — the word itself, e.g. "gemstone";
  llmq  — the model's own search query for the word
          (data/llm/queries-10k.jsonl), cleaned of boilerplate such as
          'Search query:' and surrounding quotes. The prompt asked the model
          not to use the word; "contains_word" records whether it did anyway.

Writes data/corpora/llm-{100,10k}.txt and data/queries/llm-{100,10k}.jsonl.
"""
import json, pathlib, re

ROOT = pathlib.Path(__file__).resolve().parent.parent
WORDS_PER_DOC = 50
PREFIX = re.compile(r'^\s*(search(\s+engine)?(\s+query)?|query)\s*[:\-]?\s*', re.I)


def clean_query(text):
    q = text.strip().split("\n")[0].strip()
    for _ in range(2):
        q = PREFIX.sub("", q).strip()
        q = q.strip('"“”\'` ').strip()
    return q


def load(path):
    rows = {}
    if path.exists():
        for line in path.open(encoding="utf-8"):
            r = json.loads(line)
            rows[r["i"]] = r
    return rows


def main():
    paras = load(ROOT / "data/llm/paragraphs-10k.jsonl")
    queries = load(ROOT / "data/llm/queries-10k.jsonl")
    n = 0
    while n in paras:
        n += 1
    print(f"{len(paras)} paragraphs, contiguous prefix {n}; {len(queries)} generated queries")
    for label, size in (("100", 100), ("10k", 10_000)):
        if n < size:
            print(f"llm-{label}: not enough paragraphs yet"); continue
        docs = [" ".join(paras[i]["text"].split()[:WORDS_PER_DOC]) for i in range(size)]
        (ROOT / f"data/corpora/llm-{label}.txt").write_text("\n".join(docs) + "\n", encoding="utf-8")
        out = []
        for i in range(size):
            w = paras[i]["word"]
            out.append({"qid": f"word-{i}", "kind": "word", "text": w, "relevant": [i]})
        for i in range(size):
            if i in queries:
                q = clean_query(queries[i]["text"])
                if q:
                    out.append({"qid": f"llmq-{i}", "kind": "llmq", "text": q, "relevant": [i],
                                "contains_word": paras[i]["word"].lower() in q.lower()})
        with (ROOT / f"data/queries/llm-{label}.jsonl").open("w", encoding="utf-8") as f:
            for r in out:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        nq = sum(r["kind"] == "llmq" for r in out)
        print(f"llm-{label}: {size} docs, {size} word queries, {nq} llm queries")


if __name__ == "__main__":
    main()
