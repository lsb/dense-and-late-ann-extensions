#!/usr/bin/env python3
"""Precompute query embeddings for the benchmark matrix, one query at a time.

Both ONNX models use dynamic int8 quantisation, so batching would change the
vectors (enc/NOTES.md); every query is encoded alone, as a browser would. The
per-query encode time of each model is recorded as well (native ONNX Runtime,
--threads intra-op threads, after a warm-up).

Outputs, for query set NAME (data/queries/NAME.jsonl), under build/queries/:
  NAME.minilm.f32   float32 [n, 384], raw little-endian
  NAME.lateon.f32   float32 [T, 48], queries concatenated
  NAME.json         qids, kinds, texts, relevant, lateon offsets, encode times

A cache entry is reused when the query file's SHA-256 is unchanged.

  python3 tools/encode_queries.py words-100 words-10k [--threads 1]
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "build" / "queries"
sys.path.insert(0, str(REPO))


def encode(name, threads, force=False):
    src = REPO / "data" / "queries" / f"{name}.jsonl"
    digest = hashlib.sha256(src.read_bytes()).hexdigest()
    meta_path = OUT / f"{name}.json"
    if not force and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("source_sha256") == digest:
            print(f"{name}: cached")
            return meta
    from enc.minilm import MiniLM
    from enc.lateon import LateOn
    rows = [json.loads(line) for line in open(src, encoding="utf-8") if line.strip()]
    texts = [r["text"] for r in rows]
    OUT.mkdir(parents=True, exist_ok=True)

    dense = MiniLM(threads=threads)
    dense.encode(["warm up"], batch_size=1)
    D = np.empty((len(rows), 384), np.float32)
    t_dense = []
    for i, t in enumerate(texts):
        s = time.perf_counter()
        D[i] = dense.encode([t], batch_size=1)[0]
        t_dense.append((time.perf_counter() - s) * 1e3)
    del dense

    late = LateOn(threads=threads)
    late.encode_queries(["warm up"], batch_size=1)
    vecs, offs, t_late = [], [0], []
    for t in texts:
        s = time.perf_counter()
        v = late.encode_queries([t], batch_size=1)[0]
        t_late.append((time.perf_counter() - s) * 1e3)
        vecs.append(np.asarray(v, np.float32))
        offs.append(offs[-1] + len(v))
    L = np.concatenate(vecs).astype(np.float32)

    D.astype("<f4").tofile(OUT / f"{name}.minilm.f32")
    L.astype("<f4").tofile(OUT / f"{name}.lateon.f32")
    meta = {
        "name": name, "source": str(src.relative_to(REPO)), "source_sha256": digest, "n": len(rows),
        "qids": [r["qid"] for r in rows], "kinds": [r["kind"] for r in rows], "texts": texts,
        "relevant": [r["relevant"] for r in rows],
        "minilm": {"file": f"{name}.minilm.f32", "dim": 384, "encode_ms": [round(x, 3) for x in t_dense]},
        "lateon": {"file": f"{name}.lateon.f32", "dim": 48, "offsets": offs,
                   "encode_ms": [round(x, 3) for x in t_late]},
        "contains_word": [r.get("contains_word") for r in rows],
        "encoders": {"minilm": str(Path(MiniLM.__init__.__defaults__[0]).relative_to(REPO)),
                     "lateon": str(Path(LateOn.__init__.__defaults__[0]).relative_to(REPO))},
        "threads": threads, "batch_size": 1,
        "encoded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    meta_path.write_text(json.dumps(meta))
    p = lambda xs, q: float(np.percentile(xs, q))  # noqa: E731
    print(f"{name}: {len(rows)} queries; MiniLM p50 {p(t_dense, 50):.2f} ms, LateOn p50 {p(t_late, 50):.2f} ms, "
          f"{offs[-1] / len(rows):.1f} LateOn vectors/query")
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="+")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    for n in args.names:
        encode(n, args.threads, args.force)


if __name__ == "__main__":
    main()
