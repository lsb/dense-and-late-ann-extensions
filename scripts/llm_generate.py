#!/usr/bin/env python3
"""Generate text with LiquidAI LFM2.5-350M (q4f16 ONNX) on CPU.

For each word w in the first N words of the seed-0 shuffled word list, the
user turn is a template such as "please write a paragraph about {w}".
Prompts are batched by token length so no padding is needed (the exported
graph has no position_ids input and its short-convolution layers would see
pad tokens). Sampling follows Liquid AI's recommended settings (temperature
0.3, min_p 0.15, repetition penalty 1.05) with a per-word seeded RNG, so a
row's output does not depend on which batch it was in (up to floating point).

Usage:
  llm_generate.py paragraphs N out.jsonl
  llm_generate.py queries N out.jsonl
Output is appended; rows already present are skipped (resumable).
"""
import json, pathlib, sys, time
from collections import defaultdict
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

ROOT = pathlib.Path(__file__).resolve().parent.parent
MD = ROOT / "models/lfm2.5-350m"
IM_END = 7
TEMPLATES = {
    "paragraphs": ("please write a paragraph about {w}", 256),
    "queries": ("Write a short search engine query that someone looking for a paragraph about \"{w}\" might type, without using the word \"{w}\" itself. Reply with only the query.", 32),
}
TEMPERATURE, MIN_P, REP_PENALTY = 0.3, 0.15, 1.05


def chat(user):
    return f"<|startoftext|><|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"


def sample(logits, prev, rng):
    logits = logits.astype(np.float32).copy()
    if prev:
        idx = np.fromiter(set(prev), dtype=np.int64)
        v = logits[idx]
        logits[idx] = np.where(v > 0, v / REP_PENALTY, v * REP_PENALTY)
    logits /= TEMPERATURE
    p = np.exp(logits - logits.max())
    p /= p.sum()
    p[p < MIN_P * p.max()] = 0
    p /= p.sum()
    return int(rng.choice(len(p), p=p))


class LFM:
    def __init__(self, threads=4):
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        self.s = ort.InferenceSession(str(MD / "onnx/model_q4f16.onnx"), so, providers=["CPUExecutionProvider"])
        self.inputs = [i.name for i in self.s.get_inputs()]
        self.outputs = [o.name for o in self.s.get_outputs()]
        self.tok = Tokenizer.from_file(str(MD / "tokenizer.json"))

    def generate(self, prompt_ids, seeds, max_new):
        B, L = len(prompt_ids), len(prompt_ids[0])
        feed = {}
        for n in self.inputs:
            if n.startswith("past_conv"):
                feed[n] = np.zeros((B, 1024, 3), np.float16)
            elif n.startswith("past_key_values"):
                feed[n] = np.zeros((B, 8, 0, 64), np.float16)
        cur = np.array(prompt_ids, dtype=np.int64)
        total = L
        rngs = [np.random.default_rng(s) for s in seeds]
        gen = [[] for _ in range(B)]
        done = [False] * B
        for _ in range(max_new):
            feed["input_ids"] = cur
            feed["attention_mask"] = np.ones((B, total), np.int64)
            feed["num_logits_to_keep"] = np.array(1, np.int64)
            r = dict(zip(self.outputs, self.s.run(None, feed)))
            nxt = np.zeros(B, np.int64)
            for b in range(B):
                if done[b]:
                    nxt[b] = IM_END
                    continue
                t = sample(r["logits"][b, -1], gen[b], rngs[b])
                nxt[b] = t
                if t == IM_END:
                    done[b] = True
                else:
                    gen[b].append(t)
            if all(done):
                break
            for n in self.outputs:
                if n.startswith("present_conv"):
                    feed["past_conv" + n[len("present_conv"):]] = r[n]
                elif n.startswith("present."):
                    feed["past_key_values." + n[len("present."):]] = r[n]
            cur = nxt[:, None]
            total += 1
        return [(self.tok.decode(g), len(g), d) for g, d in zip(gen, done)]


def main():
    kind, n, out = sys.argv[1], int(sys.argv[2]), pathlib.Path(sys.argv[3])
    batch = int(sys.argv[4]) if len(sys.argv) > 4 else 16
    template, max_new = TEMPLATES[kind]
    words = (ROOT / "data/words/shuffled-seed0.txt").read_text(encoding="utf-8").split("\n")[:n]
    have = set()
    if out.exists():
        for line in out.open(encoding="utf-8"):
            have.add(json.loads(line)["i"])
    m = LFM()
    groups = defaultdict(list)
    for i, w in enumerate(words):
        if i not in have:
            ids = m.tok.encode(chat(template.format(w=w)), add_special_tokens=False).ids
            groups[len(ids)].append((i, w, ids))
    todo = sorted((i for g in groups.values() for i in g), key=lambda x: (len(x[2]), x[0]))
    t0, ntok, ndone = time.time(), 0, 0
    with out.open("a", encoding="utf-8") as f:
        for L, g in sorted(groups.items()):
            for k in range(0, len(g), batch):
                chunk = g[k:k + batch]
                res = m.generate([c[2] for c in chunk], [[1234, 0 if kind == "paragraphs" else 1, c[0]] for c in chunk], max_new)
                for (i, w, _), (text, nt, fin) in zip(chunk, res):
                    f.write(json.dumps({"i": i, "word": w, "prompt": template.format(w=w), "text": text.strip(), "n_tokens": nt, "finished": fin}, ensure_ascii=False) + "\n")
                    ntok += nt
                f.flush()
                ndone += len(chunk)
                el = time.time() - t0
                print(f"{ndone}/{len(todo)} rows, {ntok/el:.0f} tok/s, {el:.0f}s", flush=True)


if __name__ == "__main__":
    main()
