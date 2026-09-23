#!/usr/bin/env python3
"""Build the random-word corpora and the word selections for the LLM corpus.

Word list: data/words/american-english (Debian wamerican 2020.12.07-2), with
entries containing an apostrophe (possessives such as "Aaron's") removed.

Random-word corpus: pass p (p = 0, 1, 2, ...) is the word list shuffled with
seed p; the passes are concatenated into one infinite word stream and document
i is words [50*i, 50*i + 50) of that stream. The 100 and 10k corpora are thus
prefixes of the 1M corpus.

LLM word selection: the first N words of pass 0.

Usage: make_corpora.py [N ...]   (default: 100 10000 1000000)
Writes data/corpora/words-<label>.txt (one document per line).
"""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from detshuffle import shuffled

ROOT = pathlib.Path(__file__).resolve().parent.parent
WORDS_PER_DOC = 50


def load_words():
    words = (ROOT / "data/words/american-english").read_text(encoding="utf-8").split("\n")
    return [w for w in words if w and "'" not in w]


def label(n):
    return {100: "100", 10_000: "10k", 1_000_000: "1m"}.get(n, str(n))


def word_stream(words):
    p = 0
    while True:
        yield from shuffled(words, p)
        p += 1


def main():
    ns = [int(a) for a in sys.argv[1:]] or [100, 10_000, 1_000_000]
    words = load_words()
    out = ROOT / "data/corpora"; out.mkdir(parents=True, exist_ok=True)
    pass0 = shuffled(words, 0)
    (ROOT / "data/words/shuffled-seed0.txt").write_text("\n".join(pass0) + "\n", encoding="utf-8")
    n_max = max(ns)
    files = {n: (out / f"words-{label(n)}.txt").open("w", encoding="utf-8") for n in ns}
    stream = word_stream(words)
    for i in range(n_max):
        doc = " ".join(next(stream) for _ in range(WORDS_PER_DOC))
        for n, f in files.items():
            if i < n:
                f.write(doc + "\n")
    for f in files.values():
        f.close()
    print(f"{len(words)} words; wrote corpora {ns}")


if __name__ == "__main__":
    main()
