# Research log

A dated, append-only record of what was done, what was measured, and why decisions were made.

## 2026-09-23 — Environment and inputs

**Machine.** 4 vCPUs, 15 GB RAM, about 30 GB free disk. Outbound network goes through a proxy: GitHub, PyPI, npm, crates.io and the Emscripten download host are reachable; huggingface.co, sqlite.org, cdn.jsdelivr.net and download.pytorch.org are not.

**Models.**
- *all-MiniLM-L6-v2*: `model_qint8_arm64.onnx` (inputs `input_ids`, `attention_mask`, `token_type_ids`; output `last_hidden_state`, 384 dimensions; embeddings require mean pooling and L2 normalisation). The matching `tokenizer.json` was not in the repository and Hugging Face is unreachable, so it was taken from the npm package `@xcidos/genesis-memory-model` 0.1.0-alpha.1, which pins sentence-transformers/all-MiniLM-L6-v2 revision `c9745ed1`. SHA-256 `be50c362…2037`.
- *LateOn-Code-edge*: `model_int8.onnx`, a ModernBERT-style encoder (vocabulary 50,370, hidden size 256) that outputs one 48-dimensional vector per token. The tokenizer contains the PyLate prefix tokens `[Q] ` (id 50368) and `[D] ` (id 50369); `[MASK]` (50284) is the padding token, as used for ColBERT query expansion.
- *LFM2.5-350M*: copied from `lsb/sidechat` (trunk, commit `b0287661`), `model_q4f16.onnx` plus six external-data files (207,419,392 bytes in total). These were concatenated and split into four equal 51,854,848-byte chunks to stay well under GitHub's 100 MB file limit. `scripts/assemble_lfm.py` restores the six original files and checks each SHA-256.

**Word list.** `/usr/share/dict` is absent in the container; the Debian package `wamerican` 2020.12.07-2 (104,334 lines, SHA-256 `9f513f1c…6a32`) was extracted into `data/words/american-english`. The 29,590 entries containing an apostrophe (mostly possessives) are dropped, leaving 74,744 words.

## 2026-09-23 — Deterministic shuffling and random-word corpora

GNU `shuf` is deterministic only for a fixed random source *and* coreutils version, and Python's `random` is tied to its implementation. The project therefore uses its own shuffle: Fisher–Yates from the last index down, with index `j = r mod (i+1)`, where `r` is drawn from SplitMix64 seeded with the pass number (`scripts/detshuffle.py`). This is a few lines in any language, which matters because the browser may need to regenerate data.

The random-word corpus is an endless stream formed by concatenating shuffle passes 0, 1, 2, …; document *i* is words 50*i* to 50*i*+49. The 100, 10k and 1M corpora are prefixes of one another. Generating all three takes 28 s; the 1M corpus is 452 MB and is not checked in.

## 2026-09-23 — LLM corpus ("LLM slop")

The LLM corpus uses the first 10,000 words of shuffle pass 0 (the first 100 give the 100-document corpus). For each word *w* the model is prompted with `please write a paragraph about w` in the LFM chat template.

- *Batching.* The exported graph has no `position_ids` input and includes short causal convolutions, so padded batches would be wrong. Prompts are grouped by token length and batched without padding.
- *Sampling.* Liquid AI's recommended settings (temperature 0.3, min-p 0.15, repetition penalty 1.05), with one NumPy generator per word, seeded `[1234, kind, i]`, so a word's text does not depend on its batch.
- *Throughput.* 123 tokens/s at batch 1 and about 160–235 tokens/s batched on 4 threads. Paragraphs average about 120 tokens, so 10,000 paragraphs take roughly 2–3 hours.
- *Queries.* A second pass asks the model for a short search query about *w* that avoids the word itself, giving paraphrase-style queries whose relevant document is known.
