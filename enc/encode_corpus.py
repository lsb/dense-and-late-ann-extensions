"""Encode a corpus with MiniLM (dense) and/or LateOn-Code-edge (late interaction).

Input formats
  *.txt    one document per line (the random-word corpora in data/corpora/)
  *.jsonl  LLM corpus rows; the document is the first 50 whitespace-separated words of
           the "text" field (``--words`` changes 50)

Outputs (under --out-dir, default data/emb/, which is git-ignored), for corpus NAME
(default: input file stem):
  NAME.minilm.npy          float16 [n, 384]  unit-norm dense vectors
  NAME.lateon.vectors.npy  float16 [T, 48]   all stored token vectors, documents concatenated
  NAME.lateon.offsets.npy  int64   [n + 1]   document i owns vectors[offsets[i]:offsets[i+1]]
  NAME.lateon.doclens.npy  int32   [n]       np.diff(offsets)
  NAME.<model>.json        metadata (counts, settings, timings)

Work is split into chunks of --chunk-size documents written to NAME.<model>.chunks/;
a finished chunk is never recomputed, so an interrupted run resumes where it stopped.
A chunk shorter than --chunk-size (the tail of a file that is still growing, such as
the LLM corpus) is recomputed on the next run.  The final arrays are assembled from the
chunks with memory-mapped writes, so the 1M corpus does not need to fit in RAM.

Documents are encoded one at a time by default (--batch-size 1).  Both ONNX models use
dynamic int8 quantisation, whose scales are computed over the whole batch tensor, so a
document's vectors otherwise depend on which documents share its batch (measured per-
token cosine between batch 1 and batch 32 as low as 0.48 for LateOn).  Batch size 1
makes document vectors reproducible and identical to what a browser computes for a
single text, at roughly 25-30% lower throughput.

Examples
  python3 -m enc.encode_corpus data/corpora/words-10k.txt --model both
  python3 -m enc.encode_corpus data/llm/paragraphs-10k.jsonl --model late --name llm-10k
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent


def iter_docs(path: Path, words: int):
    if path.suffix == ".jsonl":
        with open(path) as f:
            for line in f:
                if not line.endswith("\n"):
                    break  # partially written last line of a growing file
                line = line.strip()
                if line:
                    yield " ".join(json.loads(line)["text"].split()[:words])
    else:
        with open(path) as f:
            for line in f:
                line = line.rstrip("\n")
                if line:
                    yield line


def atomic_save(path: Path, arr: np.ndarray):
    tmp = path.with_name(path.name + ".tmp.npy")
    np.save(tmp, arr)
    os.replace(tmp, path)


class DenseJob:
    kind = "minilm"

    def __init__(self, threads):
        from enc.minilm import MiniLM
        self.m = MiniLM(threads=threads)

    def chunk_done(self, d: Path, k: int, size: int) -> bool:
        p = d / f"{k:06d}.npy"
        return p.exists() and np.load(p, mmap_mode="r").shape[0] == size

    def run_chunk(self, d: Path, k: int, docs, bs):
        atomic_save(d / f"{k:06d}.npy", self.m.encode(docs, batch_size=bs).astype(np.float16))

    def assemble(self, d: Path, nchunks: int, out: Path, name: str, delete: bool):
        n = sum(np.load(d / f"{k:06d}.npy", mmap_mode="r").shape[0] for k in range(nchunks))
        dst = out / f"{name}.minilm.npy"
        tmp = dst.with_name(dst.name + ".tmp.npy")
        mm = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16, shape=(n, 384))
        o = 0
        for k in range(nchunks):
            p = np.load(d / f"{k:06d}.npy")
            mm[o:o + len(p)] = p
            o += len(p)
            if delete:
                (d / f"{k:06d}.npy").unlink()
        mm.flush(); del mm
        os.replace(tmp, dst)
        return {"n_docs": n, "dim": 384, "dtype": "float16", "files": [dst.name]}


class LateJob:
    kind = "lateon"

    def __init__(self, threads):
        from enc.lateon import LateOn
        self.m = LateOn(threads=threads)

    def chunk_done(self, d: Path, k: int, size: int) -> bool:
        p = d / f"{k:06d}.doclens.npy"
        return (p.exists() and (d / f"{k:06d}.vectors.npy").exists()
                and np.load(p).shape[0] == size)

    def run_chunk(self, d: Path, k: int, docs, bs):
        flat, lens = self.m.encode_documents_flat(docs, batch_size=bs)
        atomic_save(d / f"{k:06d}.vectors.npy", flat.astype(np.float16))
        atomic_save(d / f"{k:06d}.doclens.npy", lens.astype(np.int32))  # written last = commit

    def assemble(self, d: Path, nchunks: int, out: Path, name: str, delete: bool):
        lens = np.concatenate([np.load(d / f"{k:06d}.doclens.npy") for k in range(nchunks)])
        offsets = np.zeros(len(lens) + 1, np.int64)
        np.cumsum(lens, out=offsets[1:])
        T = int(offsets[-1])
        dst = out / f"{name}.lateon.vectors.npy"
        tmp = dst.with_name(dst.name + ".tmp.npy")
        mm = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16, shape=(T, 48))
        o = 0
        for k in range(nchunks):
            v = np.load(d / f"{k:06d}.vectors.npy")
            mm[o:o + len(v)] = v
            o += len(v)
            if delete:
                mm.flush()
                (d / f"{k:06d}.vectors.npy").unlink()
        assert o == T
        mm.flush(); del mm
        os.replace(tmp, dst)
        atomic_save(out / f"{name}.lateon.offsets.npy", offsets)
        atomic_save(out / f"{name}.lateon.doclens.npy", lens.astype(np.int32))
        if delete:
            for k in range(nchunks):
                (d / f"{k:06d}.doclens.npy").unlink()
        return {"n_docs": int(len(lens)), "n_vectors": T, "dim": 48, "dtype": "float16",
                "mean_doclen": float(lens.mean()) if len(lens) else 0.0,
                "files": [dst.name, f"{name}.lateon.offsets.npy", f"{name}.lateon.doclens.npy"]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path)
    ap.add_argument("--model", choices=["dense", "late", "both"], default="both")
    ap.add_argument("--out-dir", type=Path, default=REPO / "data" / "emb")
    ap.add_argument("--name", help="output name (default: input file stem)")
    ap.add_argument("--chunk-size", type=int, default=10000)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("ENC_THREADS", "2")))
    ap.add_argument("--words", type=int, default=50, help="jsonl: words of 'text' per document")
    ap.add_argument("--limit", type=int, help="encode only the first N documents")
    ap.add_argument("--delete-chunks", action="store_true",
                    help="delete each chunk once copied into the final array (halves peak disk "
                         "use; an interrupted assembly then re-encodes the deleted chunks)")
    args = ap.parse_args(argv)

    name = args.name or args.input.stem
    args.out_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    if args.model in ("dense", "both"):
        jobs.append(DenseJob)
    if args.model in ("late", "both"):
        jobs.append(LateJob)

    for Job in jobs:
        job = Job(args.threads)
        cdir = args.out_dir / f"{name}.{job.kind}.chunks"
        cdir.mkdir(exist_ok=True)
        meta_path = args.out_dir / f"{name}.{job.kind}.json"
        if meta_path.exists() and not any(cdir.iterdir()):
            print(f"[{job.kind}] {name}: already assembled and chunks deleted; remove "
                  f"{meta_path.name} to re-encode", file=sys.stderr)
            continue
        docs_iter = iter_docs(args.input, args.words)
        if args.limit:
            docs_iter = itertools.islice(docs_iter, args.limit)
        t0 = time.perf_counter()
        n_done = n_new = 0
        k = 0
        encode_time = 0.0
        while True:
            chunk = list(itertools.islice(docs_iter, args.chunk_size))
            if not chunk:
                break
            if job.chunk_done(cdir, k, len(chunk)):
                n_done += len(chunk)
            else:
                t = time.perf_counter()
                job.run_chunk(cdir, k, chunk, args.batch_size)
                dt = time.perf_counter() - t
                encode_time += dt
                n_new += len(chunk)
                n_done += len(chunk)
                print(f"[{job.kind}] {name} chunk {k}: {len(chunk)} docs in {dt:.1f}s "
                      f"({len(chunk) / dt:.1f} docs/s), {n_done} total", file=sys.stderr, flush=True)
            k += 1
        meta = job.assemble(cdir, k, args.out_dir, name, args.delete_chunks)
        meta.update({
            "input": str(args.input.resolve().relative_to(REPO) if args.input.resolve().is_relative_to(REPO) else args.input),
            "model": job.kind, "chunk_size": args.chunk_size, "n_chunks": k,
            "batch_size": args.batch_size, "threads": args.threads,
            "encoded_this_run": n_new, "encode_seconds_this_run": round(encode_time, 2),
            "docs_per_second_this_run": round(n_new / encode_time, 1) if encode_time else None,
            "words_per_doc_jsonl": args.words if args.input.suffix == ".jsonl" else None,
        })
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")
        print(f"[{job.kind}] {name}: {meta['n_docs']} docs, wall {time.perf_counter() - t0:.1f}s -> "
              f"{', '.join(meta['files'])}", file=sys.stderr)


if __name__ == "__main__":
    main()
