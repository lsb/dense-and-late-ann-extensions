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

## 2026-09-23 — Query sets

`scripts/make_queries.py` writes `data/queries/words-{100,10k,1m}.jsonl`. Each corpus has 1,000 **word** queries (a single vocabulary word occurring in the corpus; relevant = every document containing it exactly; on average 1.0, 6.7 and 669 relevant documents for the 100, 10k and 1M corpora) and 1,000 **known-item** queries (three distinct words drawn from one target document, in random order; relevant = documents containing all three, almost always just the target). The 100-document corpus has only 100 known-item queries.

## 2026-09-23 — FTS5 baseline (native, APSW / SQLite 3.53.4)

Method (`bench/fts5_bench.py`): an external-content FTS5 table over `docs(id, body)`, built with `'rebuild'`, then `'optimize'` and `VACUUM`, page size 4 KiB. Queries are `SELECT rowid FROM fts WHERE fts MATCH ? ORDER BY rank LIMIT 10` (bm25), with every term quoted. "Cold pages" are distinct database pages read by a fresh connection (schema already loaded), counted with an APSW VFS shim (`bench/countvfs.py`); this approximates what an httpvfs client must fetch with an empty cache. An LLM generation job was using all 4 cores during these runs, so timings are pessimistic.

| Corpus | Bulk build | Row-by-row insert | DB size | FTS index | Known-item (AND) p50 | Word p50 | Cold pages, known-item | Cold pages, word |
|---|---|---|---|---|---|---|---|---|
| 10k | 30,100 docs/s | 26,900 docs/s | 7.6 MB | 2.4 MB | 0.031 ms | 0.025 ms | 14.0 | 16.3 |
| 1M | 22,800 docs/s (44 s) | 30,200 docs/s | 723 MB | 204 MB | 0.13 ms | 1.17 ms | 21.5 | 688 |

Quality: known-item queries are solved perfectly (recall@10 = MRR = 1). Word queries have success@1 of 0.98–0.99, not 1, because the `unicode61` tokenizer folds case and diacritics (so "Polish" matches "polish") while the relevance labels are exact string matches. Word-query recall@10 on the 1M corpus is only 0.015 simply because there are ~669 relevant documents and 10 results; for the same reason, AUC computed from a top-10 list (0.507) is not informative there. AUC needs deeper result lists and will be reported at a larger k in later matrices.

**Finding: bm25 ranking is the httpvfs bottleneck for FTS5.** On the 1M corpus a ranked single-word query touches about 688 cold pages (2.8 MB), but the same query without `ORDER BY rank` touches 10 pages, and `count(*)` 10.7. bm25 needs each matching document's length, so FTS5 reads one `fts_docsize` row per matching document, scattered over the 2,450-page `fts_docsize` table. The OR form of a known-item query (≈2,000 matching documents) touches 1,539 pages. These reads do not depend on each other and could be fetched in one parallel round, but SQLite's synchronous VFS asks for them one by one. Possible remedies to evaluate later: prefetching all of `fts_docsize` (10 MB for 1M documents), parallel prefetch driven by the doclist, or a ranking that needs no per-document lengths.

## 2026-09-23 — Network simulator (`netsim/`)

`netsim/` contains a byte-range HTTP/1.1 server with network shaping (`rangeserver.py`), a discrete-event simulator that uses the same model (`simulate.py`), a trace replayer (`replay.py`) and tests (28 passing, about 7 s). The model charges each request a queue wait for a connection slot (FIFO, optional per-connection setup cost), a latency drawn from a fixed or random distribution (normal, lognormal, exponential, Pareto, plus rare "tail" delays), and a transfer time on a link whose capacity is shared fairly among concurrent responses, with an optional per-request cap, TCP slow start and capacity jitter. A read trace is a list of dependent *rounds*: the requests in a round are issued in parallel, and the next round waits for all of them. The whole model is seeded.

Presets are taken from Chrome DevTools (`4g` = "Fast 4G": 165 ms, 8.1 Mbps; `slow-4g`: 562.5 ms, 1.44 Mbps; `slow-3g`: 2,000 ms, 400 kbps; latency and throughput already include DevTools' calibration multipliers) and from WebPageTest (`lte`: 70 ms, 12 Mbps; `3g`: 300 ms, 1.6 Mbps; and others). Both sets were checked against their source files. The `wifi`, `5g`, `starlink` and `lte-poor` presets are illustrative, and modifiers such as `h1` (6 connections) and `h2` (100 streams) set concurrency.

Validation: for seven traces, from latency-bound to bandwidth-bound to concurrency-limited, the real server is 0.2–1 % slower than the simulator and never faster (for example 1,675 ms simulated against 1,682–1,686 ms measured for a mixed trace on `4g,h1,cold`). Without shaping the server handles about 3,650 requests/s for 4 KB ranges on one connection, far above the fastest preset, so it is not the bottleneck. Known simplifications: only HTTP/1.1 is spoken (HTTP/2 is represented only by a higher concurrency limit), slow start restarts for every request, packet loss is not modelled, and CORS preflights are not simulated.
