#!/usr/bin/env python3
"""Reassemble the LFM2.5-350M q4f16 ONNX external-data files.

The weights are checked in as four equal chunks (weights.part0..3) of the
concatenation of the six original external-data files. The ONNX graph refers
to the original six file names, so this script splits the concatenation back
out using external_data_manifest.txt and verifies each SHA-256.
Output goes next to model_q4f16.onnx (the files are git-ignored).
"""
import hashlib, pathlib, sys

D = pathlib.Path(__file__).resolve().parent.parent / "models/lfm2.5-350m/onnx"

def main():
    lines = (D / "external_data_manifest.txt").read_text().split("\n")
    sizes = [(l.split()[0], int(l.split()[1])) for l in lines if l and len(l.split()) == 2 and l.split()[1].isdigit()]
    sums = {l.split()[1]: l.split()[0] for l in lines if l and len(l.split()) == 2 and not l.split()[1].isdigit()}
    parts = [D / f"weights.part{i}" for i in range(4)]
    done = all((D / n).exists() and (D / n).stat().st_size == s for n, s in sizes)
    if done and "--force" not in sys.argv:
        print("already assembled"); return
    stream = (p.open("rb") for p in parts)
    cur = next(stream)
    for name, size in sizes:
        h = hashlib.sha256(); left = size
        with (D / name).open("wb") as out:
            while left:
                buf = cur.read(min(left, 1 << 22))
                if not buf:
                    cur = next(stream); continue
                out.write(buf); h.update(buf); left -= len(buf)
        if h.hexdigest() != sums[name]:
            sys.exit(f"checksum mismatch for {name}")
        print(f"{name}: ok")

if __name__ == "__main__":
    main()
