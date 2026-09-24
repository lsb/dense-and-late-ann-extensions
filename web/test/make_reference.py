#!/usr/bin/env python3
"""Reference tokenizations and encodings from the Python encoders (enc/).

Writes build/web/reference-tokens.json (token ids of many texts, for the
tokenizer parity check in Node) and build/web/reference-embeddings.json (MiniLM
and LateOn query encodings of a few queries, for the browser check).

  python3 web/test/make_reference.py [--out-dir build/web] [--n-tokens 3000]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from enc.minilm import MiniLM, MAX_SEQ_LENGTH  # noqa: E402
from enc.lateon import LateOn  # noqa: E402

# Queries whose encodings the browser must reproduce.
EMBED_QUERIES = [
    "pinwheel",
    "gossiping player",
    "Nantucket",
    "a paragraph about moonlight",
    "what is the capital of France",
    "reverse a string in python",
    "def fib(n): return n if n < 2 else fib(n-1) + fib(n-2)",
    "Crème brûlée à la française, naïve café",
    "HTTP range requests over 4G!",
    "wineries Android formatted yeshivah commuters gemstone interface moonlight",
    "  leading and trailing spaces  ",
    "UPPER lower MiXeD 12345 3.14159",
]

# Awkward inputs for the tokenizer comparison.
EDGE_TEXTS = [
    "", " ", "a", "A", "hello world", "Hello, World!", "don't stop", "e-mail", "U.S.A.",
    "naïve café résumé", "Ångström Øresund straße", "İstanbul ıi", "ﬁnance ﬂow", "Ⅻ ①",
    "日本語のテキスト", "中文分词测试", "한국어", "emoji 😀👍🏽 family 👨‍👩‍👧", "tab\there\nnewline\r\n",
    "zero​width", "nbsp space", "soft­hyphen", "́combining", "x" * 300,
    "supercalifragilisticexpialidocious " * 40, "[CLS] [SEP] [MASK] [Q] [D] literal", "<s></s>",
    "for (int i = 0; i < n; ++i) { sum += a[i]; }", "SELECT * FROM t WHERE x = 'y';",
    "path/to/file.py:42", "0x1F 1e-9 3,141.59 $100 50%", "ÉCOLE", "ǅ ǈ", "Σίσυφος ΣΊΣΥΦΟΣ",
    "Привет мир", "مرحبا بالعالم", "שלום", "हिन्दी", "ไทย", "   multiple   spaces   ",
]


def load_texts(n):
    texts = list(EDGE_TEXTS)
    qs = [json.loads(l)["text"] for l in open(REPO / "data/queries/words-10k.jsonl")]
    texts += qs[:n]
    docs = (REPO / "data/corpora/words-10k.txt").read_text().split("\n")[:n]
    texts += docs
    llm = [json.loads(l)["text"] for l in open(REPO / "data/llm/paragraphs-10k.jsonl")][: n // 2]
    texts += llm
    return texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(REPO / "build" / "web"))
    ap.add_argument("--n-tokens", type=int, default=3000)
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ml = MiniLM(threads=1)
    lo = LateOn(threads=1)

    texts = load_texts(a.n_tokens)
    ml_ids = [e.ids for e in ml.tokenize(texts)]
    lo_ids = lo.token_ids(texts, is_query=True)
    (out / "reference-tokens.json").write_text(json.dumps(
        {"texts": texts, "minilm": ml_ids, "lateon_query": lo_ids,
         "minilm_max_len": MAX_SEQ_LENGTH, "lateon_query_length": lo.cfg["query_length"]}))

    emb = []
    for q in EMBED_QUERIES:  # one query per batch, as in the browser
        m = ml.encode([q], batch_size=1)[0]
        l = lo.encode_queries([q], batch_size=1)[0]
        emb.append({"text": q, "minilm": np.round(m, 7).tolist(),
                    "minilm_ids": ml.tokenize([q])[0].ids,
                    "lateon": np.round(l, 7).tolist(),
                    "lateon_ids": lo.token_ids([q], True)[0]})
    (out / "reference-embeddings.json").write_text(json.dumps({"queries": emb}))
    print(f"wrote {len(texts)} tokenizations and {len(emb)} encodings to {out}")


if __name__ == "__main__":
    main()
