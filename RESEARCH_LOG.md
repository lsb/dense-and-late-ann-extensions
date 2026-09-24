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

## 2026-09-23 — Encoders (`enc/`) and the LateOn-Code-edge configuration

**LateOn-Code-edge settings, pinned exactly.** The model's `onnx_config.json` could not be downloaded, so the candidate configurations the next-plaid exporter would write were generated and hashed. The GitHub project oimiragieo/tensor-grep (`src/tensor_grep/core/retrieval_late.py`) lists SHA-256 digests for Hugging Face revision `07ef20f4` of LateOn-Code-edge. Our `model_int8.onnx` matches its digest (`eac35bda…`), and exactly one candidate configuration matches its `onnx_config.json` digest (`fa4fef89…`). That file is now `models/lateon-code-edge/onnx_config.json`. It says:
- queries are prefixed with `[Q] ` and documents with `[D] `, inserted right after `[CLS]`;
- `query_length` is 256 and `document_length` 2048;
- **no** `[MASK]` query expansion;
- punctuation tokens are skipped on the document side;
- text is lower-cased.

The ONNX graph already includes both projection layers (256 → 512 → 48) and per-token L2 normalisation.

**Dynamic quantisation makes batching change the output.** Both ONNX models use dynamic int8 quantisation: activation scales are computed over the whole batch tensor, so a document's vectors depend on which other documents share its batch (per-token cosine similarity as low as 0.48 for LateOn). All corpora are therefore encoded one document at a time, which is deterministic and matches what a browser computes for a single query.

**MiniLM qint8 compared with fp32** (fp32 model from the same npm package as the tokenizer), over 2,152 texts:
- mean cosine similarity 0.988 (minimum 0.960);
- the top-1 neighbour agrees 95.2 % of the time;
- on the LLM paragraphs, recall@1 of the paragraph written about the query word is 0.855 for qint8 and 0.862 for fp32.

**Throughput and sizes** (2 threads, shared CPU):

| Model | Speed, one document at a time | Size |
|---|---|---|
| MiniLM | ≈130 docs/s | 768 bytes per document (fp16) |
| LateOn | 150–190 docs/s on random-word docs; ≈490 docs/s on LLM docs | 96 bytes per token vector (fp16) |

A 50-word random-word document yields 111.4 LateOn token vectors; an LLM paragraph truncated to 50 words yields 65.3 tokens, 58.7 after the punctuation skiplist. Encoding the 1M random-word corpus is estimated at about 2 h per model and about 10.7 GB of fp16 token vectors for LateOn.

## 2026-09-23 — PLAID study (`docs/plaid.md`)

`docs/plaid.md` specifies fast-plaid's algorithm with source references: centroid count K = 2^⌊log₂(16√T)⌋ for T token vectors, k-means, the residual codec (bucket cutoffs at quantiles i/2ⁿ, bucket weights at quantiles (i+0.5)/2ⁿ), the IVF, and the search stages with their defaults (`n_ivf_probe` 8 per query token, `n_full_scores` 4096). At dimension 48 a residual costs 6, 12 or 24 bytes at 1, 2 or 4 bits. With bit-packed centroid ids and IVF entries, the total is about 10.6, 16.6 or 28.6 bytes per token, against 96 bytes for fp16. That projects to about 1.2, 1.9 or 3.2 GB for the 1M random-word corpus.

**Finding: default PLAID probing touches too much of the corpus for httpvfs.** A NumPy simulation on words-10k with `n_ivf_probe` = 8 probes about 69 centroid cells per query and gathers about 2,800 candidate documents (28 % of the corpus), whose codes all have to be fetched for approximate scoring. PLAID's centroid-score pruning threshold (0.4 in next-plaid) removes almost nothing for this model. Omar Khattab's measurements in the Hugging Face blog (1-bit residuals, bit-packed ids, document-side pruning) point towards small indexes. A centroid-major layout in the style of WARP, where each centroid's posting list stores document ids together with residuals, lets a query finish in one parallel round after choosing centroids, and is the main candidate for the SQLite design.

## 2026-09-24 — Build toolchain and the WASM HTTP VFS (`wasm/`, `Makefile`)

**SQLite source.** SQLite 3.53.4 comes from the GitHub mirror at tag `version-3.53.4`, because sqlite.org is blocked. The amalgamation is generated with the JimTcl bundled in the source tree. `wasm/scripts/fetch-sqlite.sh` pins SHA-256 digests of the generated files, and a fresh clone reproduced them. Emscripten 6.0.10.

**Builds.** `make native` produces an `sqlite3` shell and `libsqlite3.so` with FTS5, plus loadable extensions. `make wasm` builds three WebAssembly variants. `make test` runs 4 native, 7 Node and 5 Chromium tests, all passing. Any directory `ext/*/` whose C files define `sqlite3_<name>_init` is compiled in. In static builds a generated registry calls `sqlite3_auto_extension` for each such extension.

**The HTTP VFS** (`wasm/src/httpvfs.c`) is one C file shared by the native and WASM builds. Its features:
- an LRU block cache, sequential readahead in the style of sql.js-httpvfs, and merging of adjacent ranges;
- a log of every request (`offset`, `length`, `round`, start and end time), in the trace format of `netsim/simulate.py`;
- counters for requests, bytes, rounds and cache hits.

All network access goes through one backend call, "fetch these N ranges and return when all have arrived". Each call is one **round**. Extensions reach the VFS through `sqlite3_file_control`, so there is no link-time dependency; the calls do nothing on an ordinary file. The C API is:
- `httpvfs_prefetch(db, offsets, lengths, n)` for batches of byte ranges;
- *speculative batching*: during a pass, uncached reads return zero-filled blocks and are only recorded; the recorded blocks are then fetched in one round and the pass is repeated. This batches B-tree lookups whose byte offsets an extension cannot know in advance.

**Parallel fetching in the browser.** Synchronous XMLHttpRequest, as used by sql.js-httpvfs, can only fetch one range at a time, so three variants were built from the same objects:

| Variant | Mechanism | Size of `.wasm` with both extensions (raw / gzip) | Requirements |
|---|---|---|---|
| Asyncify | Emscripten stack unwinding around `Promise.all` | 2.37 MB / 846 KB | none; works in every browser and Node |
| JSPI | WebAssembly JavaScript Promise Integration | 1.40 MB / 574 KB | Chromium 137+ (not Node 22) |
| sync | worker performs fetches while the SQLite thread waits in `Atomics.wait` | 1.40 MB / 574 KB | SharedArrayBuffer, so COOP/COEP headers in browsers |

The JS API (`wasm/pkg/index.mjs`) picks JSPI when available and otherwise Asyncify.

**Measurements.** Timings are noisy because the machine load average was 15–18 on 4 CPUs; round counts are exact.
- *Cold lookups by rowid.* With a 20 ms server delay, 16 cold rowid lookups take 18 rounds (about 420 ms) one at a time and 3 rounds (85–165 ms) with speculative batching.
- *Readahead.* It reduces a full table scan from 336 requests to 12.
- *CPU overhead.* With a warm cache, a small FTS5 query costs a fixed 0.1–0.25 ms extra in WASM. A heavier FTS5 query costs 1.4–1.8 times native with Asyncify and 0.8–1.3 times with JSPI or sync. Both are negligible next to one mobile round trip.

**Browser behaviour discovered.**
- *Cache lock.* Chromium serialises concurrent range requests for the same URL through its HTTP cache unless `fetch` uses `cache: 'no-store'`, which is now the default.
- *Connection limit.* Over HTTP/1.1 Chromium sends at most 6 requests per host at once.
- *Worker messaging.* A thread blocked in `Atomics.wait` does not get messages delivered to its nested worker; a `MessageChannel` works.

**Open issues.**
- The Asyncify build is larger than the others.
- The sync variant is untested in Firefox and Safari.
- The native build has no real HTTP backend; it only simulates a delay per round.
- Speculative batching is tested only for rowid lookups.
- Index construction must stay native, because the WASM build is single-threaded.

## 2026-09-24 — CPU contention

With the LLM job, the 1M MiniLM encoding and the agents' builds and benchmarks all running at once (load average 15–18 on 4 cores), LLM generation fell to 46 tokens/s and encoding to 12 documents/s. ONNX Runtime's intra-op threads spin-wait by default, which wastes CPU when cores are oversubscribed. All ONNX sessions now set `session.intra_op.allow_spinning = 0`. The 1M encoding is paused, at a chunk boundary (120,000 documents done), until the LLM corpus is finished. The LLM job restarted at 79 tokens/s. Timings measured during this period are marked as noisy.

## 2026-09-24 — Dense ANN extension, first version (`ext/dense/`)

**Design.** `dense_ann` is a virtual table.
- *Build.* Vectors inserted before the first `'build'` command are buffered. `'build'` trains PQ-64 (64 sub-quantizers of 6 dimensions × 256 centroids, 64 bytes per vector) on them and builds an HNSW graph in memory from full-precision vectors, with 4 threads natively.
- *Storage.* Only layer 0 of HNSW is stored. The upper layers are replaced by an *entry set* (the 1,024 highest-level nodes, kept with the PQ codebook in `_blobs` and fetched once per connection), so the descent through the upper layers costs no network rounds.
- *Node rows.* Each node is one fixed 3,152-byte row that fits in one 4 KiB page: its own PQ code and fp16 vector, 32 neighbour ids, *each neighbour's PQ code* (co-location, as in DiskANN), and a *page hint* per neighbour.
- *Search.* A beam search expands W nodes per step, the step's pages are prefetched in one round, and final candidates are reranked with the stored fp16 vectors.
- *Page hints.* `'finalize'` records the b-tree leaf page of every row. Queries then read those pages directly and parse the cells, so they never walk b-tree interior pages.

API and all options are in `ext/dense/NOTES.md`; 12 tests pass. The round counter agrees exactly with the WASM VFS's own counter.

**Layout comparison** (100k synthetic vectors, ef = 64, W = 4, all variants at recall@10 = 0.991):

| Variant | Rounds | Pages | KiB/query |
|---|---|---|---|
| co-located codes, inline vectors (default) | 17.3 | 68 | 272 |
| neighbour codes in a separate table (naive) | 34.3 | 622 | 2,487 |
| vectors in a separate table | 18.3 | 136 | 544 |
| 64 KiB pages | 16.1 | 38 | 2,405 |

Co-location halves the rounds and cuts bytes about 9×. 64 KiB pages cut rounds only slightly while reading about 9× more bytes, so 4 KiB pages are preferred.

**1M synthetic vectors** (4 KiB pages, W = 16): recall@10 = 0.966 at ef = 64 costs 7.5 rounds and 427 KiB per query; 0.993 at ef = 128 costs 7.0 rounds (W = 32) and 826 KiB. Setup is about 3 rounds and 300 KiB once per connection. Rounds ≈ expanded/W + 2 and pages ≈ ef + W. On the `4g` profile that is roughly 1.1 s per query. Building took 25 minutes (HNSW 22 minutes on 4 contended threads). The database is 4,119 bytes per document (3.9 GiB).

**Real MiniLM embeddings (words-10k).** Exhaustive search over the 64-byte PQ codes reaches only recall@10 = 0.57, and faiss `IndexPQ(384, 64, 8)` gives the same 0.567, so the codec is behaving as expected. Random-word documents are nearly tied (cosine similarity of the 1st, 10th and 100th neighbours: 0.686, 0.655, 0.620). The true top 10 is almost always inside the PQ top 100 (0.98), so reranking with stored vectors recovers quality: recall@10 = 0.81 / 0.89 / 0.95 at ef = 64 / 128 / 256, costing 6.7 / 9.9 / 17.3 rounds. In faiss, an OPQ rotation raises exhaustive PQ-64 recall to 0.650 (top 10 within the top 100: 0.998); the extension's own OPQ option reaches 0.614.

**Observation on size.** The PQ code is 64 bytes, but co-location and inline vectors make the stored row about 4 KB per document, 64 times more. This buys fewer rounds, since one page read gives the distances to all 32 neighbours. For a strict size budget, the alternative is an IVF-PQ layout (≈70 bytes per document plus centroids, with posting lists clustered on contiguous pages), which also needs only one or two dependent rounds after the centroids are cached. It is scheduled as a comparison.

## 2026-09-24 — Late-interaction extension, first version (`ext/late/`)

**Design.** `late_plaid` is a virtual table implementing PLAID-style retrieval with LateOn-Code-edge's 48-dimensional token vectors. The residual codec follows fast-plaid exactly (quantile cutoffs and weights, same bit order); ids and codes are bit-packed to ⌈log₂N⌉ and ⌈log₂K⌉ bits, following Omar Khattab's note. Two storage layouts can be built side by side:
- *plaid*: an IVF from centroid to bit-packed document ids, plus one row per document holding its codes and residuals;
- *warp* (centroid-major, after WARP): each centroid's posting list holds document ids followed by their packed residuals, so after choosing centroids one parallel round yields every token score needed. Documents missing from a query token's probed lists get an imputed score.

Lists are stored as *paged streams*: rows of exactly `page_size − 39` bytes, one per page, so nothing spills into overflow chains. Page hints make any set of lists one round with no b-tree walk. Six tests pass, including a pure-Python decoder that must reproduce the C exact scores. The page trace matches an APSW counting VFS page for page.

**Results on words-10k** (nbits = 2, K = 16,384 centroids, 2,000 queries; simulated p50 network time with HTTP/2-like concurrency; fp16 exhaustive MaxSim scores nDCG@10 = 0.359):

| Configuration | Rounds | nDCG@10 | KB read | `4g` ms | `slow-4g` ms |
|---|---|---|---|---|---|
| exhaustive over decompressed vectors | 1 | 0.357 | 20,000 | 20,392 | 114,340 |
| faithful fast-plaid (nprobe 8) | 2 | 0.357 | 16,803 | 17,436 | 97,347 |
| plaid, approximate scores from IVF only, nprobe 8, 64 docs | 2 | 0.327 | 351 | 682 | 3,105 |
| warp, nprobe 8 | 1 | 0.259 | 158 | 315 | 1,404 |
| warp, nprobe 32 | 1 | 0.329 | 510 | 673 | 3,418 |
| warp, nprobe 8, rerank 64 | 2 | 0.350 | 413 | 739 | 3,423 |

The warp rows use the stoplist described below. Faithful PLAID needs only two rounds but reads every candidate document, 4–17 MB per query, which is far too much for a phone. The warp layout plus a rerank round gets within 0.01 nDCG of exhaustive search for 0.4 MB. At nbits = 2 the codec itself costs only 0.002 nDCG; nbits 1 / 2 / 4 give 0.328 / 0.350 / 0.353 after reranking, at 7.9 / 13.9 / 26.1 bytes per token in warp postings.

**Findings.**
- *Stop vectors.* The query's `[SEP]` vector probes lists containing nearly every document, and it accounted for about 75 % of fetched posting entries. Skipping any query vector whose probed lists average more than 2 % of the corpus (`stoplist=0.02`) and giving it a constant score leaves rankings unchanged and cuts bytes by a third. At 1M documents this is required.
- *Session setup.* Loading the static data costs 1.7 MB in 2 rounds with K = 16,384 (2.0 s on `4g`); with K = 4,096 it is 0.43 MB. A two-level centroid scheme shrinks this to 15–60 KB but currently loses 0.05–0.09 nDCG.
- *Pages dominate at small scale.* Warp at nprobe 8 needs 29 KB of posting data but reads 152 KB of 4 KiB pages. Sub-page range reads would help.
- *HTTP/1.1.* With 6 connections one warp round becomes several latency steps (923 ms against 315 ms on `4g`).
- *The model is weak on random words.* LateOn-Code-edge reaches exhaustive nDCG@10 of only 0.12 on single-word queries over random-word documents, which compresses the differences between methods. The LLM-paragraph corpus is the more meaningful test.

**Projected 1M index.**
- *Centroids.* K = 65,536, rather than fast-plaid's formula value of 131,072, to keep the build tractable and the static data at 6.8 MB.
- *Build.* About 3 hours and 2.5 GB RAM.
- *Size.* The warp layout would take 0.96 / 1.63 / 2.98 GB at nbits 1 / 2 / 4.
- *Query cost.* Roughly 0.65 MB in one round per query. Recall at that scale is still to be measured.

## 2026-09-24 — Browser demo and client library (`web/`)

**What was built.**
- `web/lib/index.mjs` provides `openIndex(url)` and `search(text, {system, k, ...})`. SQLite runs in one Web Worker, and each query encoder runs in its own worker using onnxruntime-web 1.30.0 and `@huggingface/tokenizers` (vendored in `web/vendor/`; no CDN).
- The library discovers the FTS5, `dense_ann` and `late_plaid` tables from the schema. Every result carries its stats: encode, SQL and network time, rounds, requests and bytes.
- `web/index.html` shows the three systems side by side, with a network-profile selector that drives `netsim/rangeserver.py`.
- `make web` builds the demo database and runs the Playwright test (3 tests pass); `make serve PRESET=4g` serves the repository at `http://localhost:8000/web/`.

**Browser and Python agree.** Tokenization is identical to Python for 6,540 texts. Two edge cases had to be fixed: a word-final Σ is lower-cased differently in Rust and JS, and Python's `strip()` removes a different set of whitespace from JS `trim()`.

The encodings are close but not bit-identical:

| Model | Mean cosine | Minimum cosine | Exact top-10 overlap with Python |
|---|---|---|---|
| MiniLM | 0.9989 | 0.981 | 0.975 |
| LateOn | 0.9997 | 0.987 | 0.995 |

The integer matrix products in onnxruntime-web are bit-exact. Float operators such as LayerNorm differ by about 10⁻⁶, which occasionally flips a step of the dynamic int8 quantisation, and the difference then grows through the layers; the same effect appears inside Python's onnxruntime between optimised and unoptimised graphs. This is smaller than the batch-composition effect. For the same query vector, the browser returns exactly the rows the native build returns.

**Sizes and first-visit cost** (raw / gzip):

| Component | Raw | gzip |
|---|---|---|
| Library JS | 40 KB | 14 KB |
| SQLite WASM (JSPI) | 1.55 MB | 0.62 MB |
| ONNX Runtime WASM | 14.2 MB | 3.7 MB |
| MiniLM model | 23.0 MB | 17.4 MB |
| LateOn model and tokenizer | 20.8 MB | 14.0 MB |

A first visit using both encoders transfers about 60 MB: 41 s on `lte` and 60 s on `4g`. Later visits load the encoders from Cache Storage in about 2.5 s, mostly ONNX session creation. Warm query encoding takes 7–11 ms (MiniLM) and 3.5–4 ms (LateOn) for a short query.

**End-to-end query times** in headless Chromium, words-10k, HTTP/1.1 (6 connections), after the first query:

| System | Unshaped | `lte` | `4g` |
|---|---|---|---|
| FTS5 | 0.08 s | 0.58 s | 1.3 s |
| Dense graph | 0.22 s | 1.5 s | 3.1 s |
| Late (warp) | 0.30 s | 1.5 s | 3.3 s |

The first query of a session takes 4.7–6.5 s on `4g` because it also loads each index's static data. On `4g` the browser's six-connections-per-host limit costs more than the round trips themselves: a round of 16–100 requests becomes several waits in sequence. HTTP/2 hosting, or fewer and larger range requests per round, is therefore a priority.

## 2026-09-24 — Dense ANN: IVF-PQ layout compared with the graph

`dense_ann` gained a `layout=ivf` option (15 tests pass).
- *Coarse quantiser.* k-means with nlist lists; the centroids are stored as fp16, int8 or PQ codes in a head fetched once per connection.
- *Posting lists.* PQ-64 codes of residuals in rows of at most 4,000 bytes, 69 bytes per entry (rowid, page delta, code), written list by list so each list is one contiguous page range.
- *Queries.* Round 1 fetches all `nprobe` lists in parallel; round 2 fetches the stored vectors of the top `rerank_k` candidates for reranking. A query is therefore two rounds regardless of corpus size.

**Matched-recall comparison.** Recall@10 is measured against exact search; times are netsim estimates on `4g` / `slow-4g` for a warm connection.

| Data | Target recall | Graph (co-located PQ, HNSW-built) | IVF-PQ |
|---|---|---|---|
| synthetic 1M | 0.95 | 0.972 · 5.9 rounds · 606 KiB · 1,526 / 6,519 ms | 0.995 · 2 rounds · 404 KiB · 738 / 3,421 ms |
| synthetic 1M | 0.90 | 1,336 ms | 635 ms |
| words-10k MiniLM | 0.95 | 0.954 · 9.8 rounds · 1,058 KiB · 2,671 / 11,471 ms | 0.957 · 2 rounds · 1,049 KiB · 1,394 / 7,111 ms |
| words-120k MiniLM | 0.80 | 0.815 · 12.5 rounds · 2.4 MB · 4,453 ms | 0.834 · 3.2 MB · 3,609 ms |
| words-120k MiniLM | 0.90 | not reached | 0.924 · 5.8 MB · 6,242 / 34,379 ms (nprobe 512 of 1,386) |

Storage at 1M:
- IVF: 917 bytes per document (80 bytes of list entry and 821 bytes of fp16 vector). int8 rerank vectors cut this to about 500 bytes at the same recall; with no stored vectors it is about 95 bytes, but recall stops near 0.6.
- Graph: 4,119 bytes per document.

Setup per connection:
- graph: 288 KiB;
- IVF: 540 KiB at 10k and 736 KiB at 1M, the latter with PQ-compressed centroids (fp16 centroids would be 3.3 MB).

**Interpretation.**
- *Synthetic data flatters IVF*, because its clusters line up with the lists.
- *On real MiniLM data the bottleneck is coarse probe coverage.* To reach 0.92 recall a query must probe about 37 % of the lists at both 10k and 120k, so bytes per query grow linearly with corpus size.
- *The graph needs more rounds but fewer bytes*, and it levels off near recall 0.8 for single-word queries, which are out of distribution for 50-word documents. Queries that are themselves documents reach 0.93.
- *Recall@10 against exact search is partly noise here.* Random-word documents are near-ties, so exact search's own top 10 depends on small encoder differences. A comparison against relevance labels on the LLM corpus should decide between the layouts.

`ext/dense/run_words1m.sh` runs both layouts on the real 1M embeddings once they exist.

## 2026-09-24 — LLM corpus finished

All 10,000 paragraphs were generated (9,995 ended with the end-of-turn token within 256 new tokens; mean 127 tokens). 39 paragraphs are shorter than 50 words and are used whole. `scripts/make_llm_corpus.py` wrote `data/corpora/llm-{100,10k}.txt` (document *i* = the first 50 words of the paragraph about word *i*) and `data/queries/llm-{100,10k}.jsonl`, with two known-item query kinds per document:
- **word**: the word itself;
- **llmq**: the model's own search query.

The generated queries are short (3.1 words on average) and often vague. Common ones are "creative ideas" (70 times), "historical significance" (68) and "negative impact" (68). 21.8 % contain the target word despite the instruction. Because each has exactly one relevant document, llmq is a hard, noisy known-item task; it resembles what a small model produces rather than what a careful user types. The LLM corpora were encoded with both models, one document at a time, into `data/emb/llm-{100,10k}.*`.

## 2026-09-24 — First full benchmark matrix (`results/matrix/`)

**Method.** `tools/build_db.py` builds one deployable database per corpus: documents, FTS5, the dense graph and dense IVF-PQ indexes, and the late index in both the warp and plaid layouts, VACUUMed and finalized, with 4 KiB pages. `bench/matrix.py` then does four things:
- measures quality natively against the relevance labels;
- runs every configuration through the Asyncify WASM build in Node against the shaped range server, recording each query's request trace;
- replays the traces through `netsim/simulate.py` under 8 profiles × {HTTP/1.1 with 6 connections, HTTP/2-like with 100};
- validates the simulation against real shaped-server runs.

WASM returned exactly the native top-10 list for every query and configuration. Real runs were 0.3–3 % slower than simulated on cellular profiles (localhost overhead of 2–6 ms per round on a loaded machine). "Warm" means a query within a session with a 4 MiB block cache; "cold" means a fresh connection. AUC uses a separate k = 100 run.

**Headline** (simulated p50 on `4g`, HTTP/2-like; the full tables are in `results/matrix/README.md` and the charts in `results/matrix/index.html`):

| Corpus | Method | Index size | nDCG@10 | AUC@100 | Warm p50 | Cold p50 |
|---|---|---|---|---|---|---|
| words-10k | FTS5 bm25 | 2.5 MB | 0.994 | 1.000 | ≈0 ms (cached) | 3.0 s |
| words-10k | dense graph, ef 64 | 41.5 MB | 0.075 | 0.588 | 1.4 s | 2.4 s |
| words-10k | dense IVF, nprobe 64 | 9.9 MB | 0.075 | 0.583 | 0.48 s | 2.1 s |
| words-10k | late warp 8 + rerank 64 | 39.7 MB | 0.350 | 0.754 | 0.69 s | 3.3 s |
| llm-10k | FTS5 bm25 (OR) | 1.8 MB | 0.456 | 0.775 | ≈0 ms (cached) | 2.7 s |
| llm-10k | dense graph, ef 64 | 41.5 MB | 0.585 | 0.866 | 1.2 s | 2.3 s |
| llm-10k | dense IVF, nprobe 128 | 9.9 MB | 0.567 | — | 0.40 s | 2.0 s |
| llm-10k | late warp 8 | 19.5 MB | 0.431 | — | ≈0 ms (cached) | 1.6 s |

References for exhaustive search on llm-10k: MiniLM exact nDCG@10 = 0.586, LateOn exact MaxSim = 0.519.

**Findings.**
- *Random-word corpora are a lexical task.* FTS5 is essentially perfect, MiniLM is near chance (nDCG 0.075), and LateOn is in between (0.35), because its token vectors can match exact words.
- *On the LLM corpus the dense model is best.* MiniLM beats LateOn-Code-edge for both query kinds, and both beat FTS5 on paraphrase (llmq) queries: nDCG@10 of 0.31 (dense), 0.16–0.18 (late) and 0.15 (FTS5 OR). LateOn-Code-edge is a code-retrieval model, so this is not a verdict on late interaction in general.
- *ANN costs almost nothing in quality.* The dense graph at ef 64 is within 0.001 nDCG of exhaustive MiniLM search on llm-10k, and IVF nprobe 128 within 0.02.
- *Session start dominates the first query.* A cold query costs 1.6–3.3 s on `4g` for every method, mostly per-connection static data (late centroids ≈1.7 MB, IVF centroids, PQ codebook and entry set). Within a session, IVF answers in about 0.4 s and the graph in about 1.2 s.
- *Session caching can make the "warm" regime look free.* At 10k documents the 4 MiB block cache ends up holding much of the hot data, so warm-session numbers depend on query order and cache size; the "Ext. KB" column shows the extension's reads before the cache.
- *Exhaustive dense search in the extension (`exact=1`) reads one page per round* (about 2,000 sequential rounds at 10k), so it is useful only as a quality baseline.

## 2026-09-24 — Packaging (`docs/INSTALL.md`)

The project is distributed through two packages that share one database format, both provisionally named `dense-late-ann`. Nothing has been published.
- **npm package** (`packages/npm/`), 2.2 MB packed. It contains the three SQLite WASM builds, the JS API and the search client, with hand-written TypeScript declarations. `onnxruntime-web` and `@huggingface/tokenizers` are optional peer dependencies, needed only for in-browser query encoding.
- **Python package** (`pyproject.toml`, `python/dense_late_ann/`). It compiles both extensions as loadable modules and provides `load(conn)` for `sqlite3` or APSW connections, plus a `dense-late-ann` command (`check`, `build-db`, `search`, `encode-query`) that wraps `tools/build_db.py` and `enc/`. Models are not bundled.

**Tested from scratch.**
- *Python.* `pip install .` in a fresh virtualenv, then `dense-late-ann build-db` on llm-100, gave top-10 lists identical to the benchmark database.
- *npm.* The packed tarball was installed into a fresh project; its results were identical to native in Node and Chromium.
- *sqlite3 shell.* The extensions load in the shell with `.load`.

**Workflows.**
- `.github/workflows/ci.yml` builds native and WASM and runs all tests.
- `.github/workflows/release.yml` builds wheels (cibuildwheel for Linux and macOS), the sdist and the npm tarball on version tags. Publishing stays off until credentials are configured and `PUBLISH_PACKAGES` is set. Neither workflow has been run on GitHub yet.

Windows is unsupported because index building uses pthreads.

## 2026-09-24 — The int8 encoders are wrong on CPUs without VNNI; switch to weight-only int8

**Symptom.** The first GitHub Actions run failed a packaging test: for a query copied from document 3, the late index did not return document 3 at all. Locally it ranked first. A build with AddressSanitizer and UndefinedBehaviorSanitizer found nothing wrong in the C code.

**Cause.** Both published int8 ONNX exports use onnxruntime's *dynamic* quantization: each matrix product becomes `DynamicQuantizeLinear(x) → MatMulInteger(x_q, W_q) → Cast → Mul(scale)`, with uint8 activations and int8 weights. On x86 CPUs without AVX-512 VNNI or AVX-VNNI (GitHub's runners are AMD EPYC with AVX2 only), onnxruntime's u8s8 kernel sums pairs of products in saturating 16-bit integers. Running the encoder under `qemu-x86_64 -cpu Haswell`, which emulates an AVX2-only CPU, reproduced this. LateOn-Code-edge's token vectors on such a CPU have a mean cosine similarity of only 0.24–0.40 with those computed on a VNNI CPU; MiniLM drifts less (0.985–0.994 against fp32). The same mechanism also explains the batch dependence recorded earlier, because the activation scale is computed over the whole batch.

**Fix.** `scripts/dequantize_activations.py` rewrites each chain as `MatMul(x, DequantizeLinear(W_q, W_scale, W_zero_point))`, or as a plain float `MatMul` where both operands are activations (MiniLM's attention products). The weights stay int8 in the file, so the sizes are unchanged (17.2 MB and 23.0 MB); onnxruntime constant-folds the dequantization when the session is created and computes in float32. The results are `models/*/model_w8.onnx` ("w8"), now the default in `enc/`, `web/lib/` and the Python package.

| Check | Original int8 | w8 |
|---|---|---|
| LateOn, VNNI CPU vs emulated AVX2 CPU (per-token cosine) | 0.24–0.40 | 0.9999 |
| MiniLM, VNNI vs AVX2 | 0.985–0.995 (vs fp32) | 0.9999 |
| Batch of 4 vs one at a time | 0.99 | 1.0 (identical) |
| MiniLM vs fp32 model | 0.991–0.995 | 0.998–0.999 |
| Browser (onnxruntime-web) vs Python, share of vectors with cosine ≥ 0.999 | 84 % / 91 % | 100 % / 99.9 % |

**Quality**, exhaustive search, nDCG@10 on 500 word queries and 500 llmq queries of llm-10k:

| Model | Variant | Word queries | Paraphrase queries |
|---|---|---|---|
| MiniLM | int8 | 0.853 | 0.288 |
| MiniLM | w8 | 0.854 | 0.288 |
| MiniLM | fp32 | 0.854 | 0.287 |
| LateOn | int8 | 0.803 | 0.199 |
| LateOn | w8 | 0.816 | 0.211 |

Dynamic activation quantization cost LateOn about 0.012 nDCG even on VNNI hardware. The price of w8 is speed: native encoding one document at a time falls from 177 to 93 documents/s (MiniLM) and from 224 to 150 (LateOn) on the shared CPU. In the browser a short query now takes about 30 ms (MiniLM) and 13 ms (LateOn) to encode, instead of 7–11 ms and 3.5–4 ms.

**Consequence.** All corpora are re-encoded with the w8 models, and the benchmark matrix is rebuilt. The dense words-1m ANN comparison that was already running uses int8 MiniLM embeddings. Its conclusions about recall against exact search, rounds and bytes do not depend on the encoder variant, so it is left to finish.

## 2026-09-24 — Dense ANN on the real words-1m embeddings

Both `dense_ann` layouts were built over the 1M random-word corpus's MiniLM embeddings (the int8 encoder, computed before the w8 switch) and queried with 1,000 held-out random-word documents and single words. Recall@10 is measured against exact float search. All times are warm-connection `4g` estimates.

**Graph layout** (HNSW-built, M = 16, 4 KiB pages):

| ef | Best W | Recall@10 | Rounds | KiB | `4g` |
|---|---|---|---|---|---|
| 64 | 32 | 0.434 | 8.2 | 649 | 2.0 s |
| 128 | 64 | 0.515 | 8.5 | 1,257 | 2.6 s |
| 256 | 64 | 0.574 | 10.7 | 1,710 | 3.5 s |

**IVF-PQ layout** (1,000 lists):

| nprobe | Rerank R | Recall@10 | KiB | `4g` |
|---|---|---|---|---|
| 128 | 256 | 0.796 | 12,945 | 13.6 s |
| 512 | 128 | 0.901 | 47,278 | 48 s |
| 1,024 (all lists) | 256 | 0.962 | 72,957 | 74 s |

With the default 4√N ≈ 4,000 lists, 0.922 needs 29 MB per query (30 s on `4g`). Setup is 0.5–0.7 MB per connection. Build times are 166 s with 1,000 lists and 681 s with the default number.

**Interpretation.** Random-word documents make a pathological dense corpus. Each document is a bag of 50 unrelated words, so MiniLM places all million documents in a narrow, nearly structureless region: neighbours are near-ties, and the top 10 is fragile.
- *PQ-64 cannot separate near-ties.* The graph's recall against exhaustive PQ search is only about 0.3.
- *Coarse clustering does not follow the data.* IVF's clusters do not line up with query neighbourhoods, so recall grows only slowly as more lists are probed.

The same code on synthetic clustered 1M data reaches recall 0.97–0.995 in 2–8 rounds and 0.4–0.6 MB. And on llm-10k the graph matches exhaustive MiniLM search (nDCG 0.585 against 0.586). Recall against exact search at 1M random words is therefore a stress test rather than a prediction for real text. A realistic 1M-scale dense evaluation would need a realistic 1M-document corpus: generating one with the local LLM would take about 10 days on this machine, so it is out of scope.

## 2026-09-24 — Ranked FTS5 over httpvfs (`docs/fts5-httpvfs.md`, `ext/fts5rank/`)

**Problem.** FTS5's `bm25()` reads one `fts_docsize` row per matching document. Over httpvfs each read is a separate round trip, because SQLite's synchronous VFS asks for them one at a time.

**Fix.** The new extension `ext/fts5rank` adds `bm25c()`: bm25 with every document treated as having the average length, so ranking reads nothing per document. It is now the client's default (`rank: 'bm25c'` in `web/lib/search-core.mjs`); `rank: 'bm25'` restores the old behaviour. Two exact alternatives are also available:
- `bm25-rerank` re-scores bm25c's top 50 with their stored lengths (3 more rounds, about 200 KB at 1M; its top 10 matches bm25's 97.5–100 % of the time).
- `bm25-prefetch` fetches every matching `fts_docsize` row in one `httpvfs_warm` batch (13 rounds, but still 2.7 MB at 1M).

**Results through the WASM client.** Cold queries on words-1m, simulated on `4g`:

| Query | bm25 | bm25c |
|---|---|---|
| Single word | 685 rounds, 2.9 MB, 115 s | 10 rounds, 39 KB, 1.7 s |
| Three-word OR | 1,092 rounds, 191 s | 17 rounds, 3.0 s |

Real shaped runs agree with the simulation: bm25 took 51 s on `lte` and bm25c 1.71 s on `4g`.

**Quality.** Unchanged on words-10k, words-1m and llm-10k, whose documents have (nearly) equal lengths. On untruncated LLM paragraphs (11–205 words), bm25c loses 0.008 nDCG@10 on word queries, and rerank recovers it.

**Alternatives that did not help.**
- Reading the whole `fts_docsize` table costs 9.6 MB per session at 1M.
- The FTS5 options `columnsize=0` and `detail=column`/`none` make ranking re-tokenise the matching documents (about 1,000 pages), so none of them helps.

After this fix, a cold FTS5 query still costs 6–17 rounds. Those come from FTS5's own chain of dependent lookups (structure record, `%_idx`, leaves), so prefetching the structure pages when the database opens is the next lever.

**Caveat for the matrix.** Ties are broken by rowid, and in the LLM query sets query *i* is about document *i*. A subset of the first 1,000 queries therefore favours methods with many ties; over all 10,000 queries this washes out.

## 2026-09-24 — Request budget per round for HTTP/1.1 (`wasm/src/httpvfs.c`)

**Problem.** Browsers run at most six requests per host over HTTP/1.1, so a round of 16–100 range requests costs several latencies in sequence.

**Design.** The VFS now plans each round against a request budget C (`maxRequests: 'auto'`: 6 over HTTP/1.x, 100 over HTTP/2, off in Node). It has two strategies:
- *Coalescing.* It merges nearby ranges into at most C requests, choosing merges exactly under the cost model ⌈g/C⌉·RTT + bytes/bandwidth. RTT and bandwidth are estimated online from recent rounds (in Chromium on `4g,h1` the estimate converged to 167–169 ms, against a true 165 ms). Over-fetched pages go into the block cache.
- *Multi-range requests* (opt-in). It sends at most C requests of the form `Range: bytes=a-b,c-d,…`, which over-fetch nothing. A server that ignores or refuses multi-range requests is detected once per URL (a 200 reply is aborted after its headers), and the VFS falls back to coalescing, at a cost of one extra round trip.

**Results.** Warm-session p50 on `4g`, re-simulating the words-10k matrix traces with the C planner:

| System | HTTP/1.1 before | Coalescing, C = 6 | Multi-range | HTTP/2 |
|---|---|---|---|---|
| Dense graph (ef 64) | 2.60 s | 2.58 s | 1.43 s | 1.42 s |
| Dense IVF (nprobe 64) | 1.13 s | 0.95 s | 0.47 s | 0.47 s |
| Late warp (nprobe 8, rerank 64) | 2.09 s | 1.92 s | 0.63 s | 0.62 s |

- Multi-range requests make HTTP/1.1 match HTTP/2 within 1 % for every system and profile.
- Coalescing alone gains 15–23 % for IVF, 3–8 % for warp, and 1 % or less for the graph, whose 16 nodes per round are scattered over 40 MB.
- Node + WASM runs against the h1-shaped server agree with the simulator within 2–4 % and return identical results.
- In Chromium on `4g,h1` the demo's later queries go from 3.5 / 3.4 s (graph / late) to 3.3 / 3.0 s with coalescing and to 1.8 / 1.0 s with multi-range requests.

**Recommendation.** Host on HTTP/2, or enable multi-range requests on HTTP/1.1 servers that support them (nginx, Apache and Caddy do; S3 and R2 do not). One caveat: a cross-origin page sends one CORS preflight per URL for multi-range requests.

## 2026-09-24 — Cold start: warming and shrinking per-connection static data

**Warming.** Both extensions accept `SELECT … WHERE t MATCH 'warm'`, which loads their per-connection static data without running a query:
- dense: the codebook and entry set, or the IVF centroids;
- late: the centroids, the list lengths and the document table's interior pages.

The browser client (`openIndex({warm: 'auto'})`) runs it in the SQLite worker at low priority as soon as an encoder starts loading, so the fetch overlaps with model loading. A search that arrives meanwhile goes first.

**Shrinking.** The following changes cost nothing measurable on llm-10k and words-10k (within ±0.002 of float16 in nDCG and recall against exact search):
- int8 late centroids (late format 3, which also stores list lengths as varints);
- int8 PQ codebooks (dense format 2);
- int8 or PQ IVF centroids (`ivf_centroids=auto`).

They halve the static data and are now the defaults:

| Index | Before | After |
|---|---|---|
| late, llm-10k | 832 KB | 418 KB |
| late, words-10k | 1,664 KB | 836 KB |
| dense graph head | 264 KB | 168 KB |
| dense IVF head | 497 KB | 252 KB |

A cold query becomes 0.1–0.9 s faster on `4g`. Two further reductions were rejected because they lose quality:
- int4 centroids: +0.015 nDCG on llm-10k but −0.007 to −0.014 on words-10k;
- halving K: −0.008 to −0.015 nDCG.

A *lazy cells* mode stores the flat centroids in cells fetched on demand, and it matches flat quality at cprobe 16. That makes it better than the trained two-level mode (−0.05 to −0.09 nDCG, a loss that came from per-cell k-means, not from routing), but it saves little at 10k and is kept for 1M-scale use.

**Time to first result in Chromium** (llm-10k; a real HTTP/2 proxy, `web/test/h2proxy.mjs`, allows true h2 measurements):
- *Repeat visit, query typed 3 s after the page opens.* The first query is 30–55 % faster, for example 2.9 s → 1.4 s for late on `4g`/h2 and 3.1 s → 2.1 s for IVF on `4g`/h1. It now costs what a query within a session costs.
- *Repeat visit, query issued the moment the index opens.* The gain is 0.1–1.4 s. Opening the page, the SQLite build and the database (1.9–3.4 s) is then the long pole, not the cached encoder (1.1–1.7 s).
- *First visit.* The roughly 37 MB download of ONNX Runtime and the models dominates (28–44 s), and the gain is 0.5–1.2 s.

**Parser bug.** `late_plaid` ignored every option after the first when options were space-separated, which is how `tools/build_db.py` writes them. Earlier matrix late indexes were therefore built with default parameters. At 10k these equal the intended ones; at 1M they would have meant 131,072 centroids. The bug is fixed (with a test), and all matrix databases are being rebuilt.

## 2026-09-24 — CI fixes after the round-2 merges

CI on GitHub had been red since the FTS5 merge, for two reasons.

1. **npm package test.** It compared the WASM client's FTS5 results, now ranked with `bm25c` by default, against a native reference ranked with plain `bm25`. The test now asks for `rank: 'bm25'` explicitly.
2. **Stoplist on tiny corpora.** Once the late option parser was fixed, the stoplist (`stoplist=0.02`) actually took effect. At N = 100, 2 % of N is 2 entries, so nearly every query token counted as a stop token and a top-10 query returned only 6 documents. A stop token must now also exceed a floor of 64 list entries (`LATE_STOPLIST_MIN_ENTRIES`). This changes nothing at 10k (where 2 % is 200) or at 1M, and only affects corpora below 3,200 documents.

## 2026-09-24 — Disk housekeeping

The w8 1M encodings are complete: LateOn in 3.0 h, MiniLM in 3.0 h, one document at a time on 2 threads with a shared CPU. The words-1m late index (warp layout, K = 65,536, int8 centroids, nbits = 2) was built from the LateOn vectors into `build/matrix/words-1m--late.db` (1.6 GB). The 10.7 GB file `data/emb/words-1m.lateon.vectors.npy` was then deleted to make room for the 1M dense indexes, because the 30 GB disk allowance is nearly full. The offsets and document lengths are kept. Exact MaxSim ground truth is skipped at this scale anyway (111M token vectors), and the vectors can be regenerated with `enc/encode_corpus.py --model late data/corpora/words-1m.txt` in about 3 hours.
