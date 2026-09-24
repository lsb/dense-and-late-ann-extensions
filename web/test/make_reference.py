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
import onnxruntime as ort

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
    ap.add_argument("--n-embed", type=int, default=100, help="extra queries from data/queries/words-10k.jsonl")
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

    # The same encoders with ONNX Runtime graph optimisations disabled. enc/ uses
    # ORT_ENABLE_ALL, whose fused x86 kernels round floats slightly differently;
    # with dynamic int8 quantisation such differences can flip a quantisation
    # step and move a vector by a cosine of up to ~0.01 (see web/NOTES.md).
    ml0 = MiniLM(threads=1)
    lo0 = LateOn(threads=1)
    for enc in (ml0, lo0):
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        enc.session = ort.InferenceSession(enc.session._model_path, so, providers=["CPUExecutionProvider"])

    qs = [json.loads(l) for l in open(REPO / "data/queries/words-10k.jsonl")]
    kinds = {}
    for q in qs:
        kinds.setdefault(q["kind"], []).append(q["text"])
    queries = list(EMBED_QUERIES)
    for k in sorted(kinds):
        queries += kinds[k][: a.n_embed // len(kinds)]
    emb = []
    for q in queries:  # one query per batch, as in the browser
        emb.append({"text": q,
                    "minilm": np.round(ml.encode([q], batch_size=1)[0], 7).tolist(),
                    "minilm_noopt": np.round(ml0.encode([q], batch_size=1)[0], 7).tolist(),
                    "minilm_ids": ml.tokenize([q])[0].ids,
                    "lateon": np.round(lo.encode_queries([q], batch_size=1)[0], 7).tolist(),
                    "lateon_noopt": np.round(lo0.encode_queries([q], batch_size=1)[0], 7).tolist(),
                    "lateon_ids": lo.token_ids([q], True)[0]})
    # Noise floor: the same Python encoder with and without graph optimisations.
    def cos_rows(x, y):
        x, y = np.atleast_2d(np.asarray(x, np.float64)), np.atleast_2d(np.asarray(y, np.float64))
        return (x * y).sum(1) / np.linalg.norm(x, axis=1) / np.linalg.norm(y, axis=1)
    floor = {}
    for key in ("minilm", "lateon"):
        c = np.concatenate([cos_rows(e[key], e[key + "_noopt"]) for e in emb])
        floor[key] = {"min": float(c.min()), "mean": float(c.mean()), "frac_ge_0999": float((c >= 0.999).mean()), "n": int(len(c))}
    print("python ENABLE_ALL vs DISABLE_ALL:", json.dumps(floor))
    (out / "reference-embeddings.json").write_text(json.dumps({"queries": emb, "python_opt_vs_noopt": floor}))
    print(f"wrote {len(texts)} tokenizations and {len(emb)} encodings to {out}")


if __name__ == "__main__":
    main()
