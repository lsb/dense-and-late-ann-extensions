#!/usr/bin/env python3
"""Encode a query set with LateOn-Code-edge, one query at a time.

Writes build/late/queries-<name>.npz with `vectors` (float32 [sum nq, 48]),
`offsets` (int64 [n+1]), `qids`, `kinds`, and `relevant` (object array).
Usage: python3 ext/late/encode_queries.py words-10k [words-100 ...]
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from enc.lateon import LateOn  # noqa: E402


def main():
    enc = LateOn()
    out_dir = REPO / "build" / "late"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in sys.argv[1:]:
        rows = [json.loads(l) for l in open(REPO / "data" / "queries" / f"{name}.jsonl")]
        vecs, offs = [], [0]
        for r in rows:
            v = enc.encode_queries([r["text"]], batch_size=1)[0]
            vecs.append(v)
            offs.append(offs[-1] + len(v))
        np.savez(out_dir / f"queries-{name}.npz",
                 vectors=np.concatenate(vecs).astype(np.float32),
                 offsets=np.array(offs, np.int64),
                 qids=np.array([r["qid"] for r in rows]),
                 kinds=np.array([r["kind"] for r in rows]),
                 texts=np.array([r["text"] for r in rows]),
                 relevant=np.array([np.array(r["relevant"], np.int64) for r in rows], dtype=object))
        print(name, len(rows), "queries,", offs[-1], "vectors")


if __name__ == "__main__":
    main()
