# Benchmark matrix: notes

The benchmark matrix measures every retrieval configuration of the project end to end: the size it adds to the deployed database, its retrieval quality against the relevance labels, and its fetch costs and latency when the WebAssembly SQLite build reads the database over HTTP range requests. Results are in `results/matrix/` (`README.md` with tables, `index.html` with charts, `<corpus>.json` with every number).

**Encoders.** All results in `results/matrix/` use the weight-only int8 encoders (`models/*/model_w8.onnx`, the `enc/` defaults since commit 54a9095) for both the corpus embeddings and the queries. The earlier matrix, made with the dynamically quantised int8 files, was discarded; see RESEARCH_LOG.md, "The int8 encoders are wrong on CPUs without VNNI". `build/queries/<set>.json` records the model files used, and each database manifest records the embedding metadata.

**Index formats.** The databases are built with the extensions as of commit 7c4270d: late_plaid format 3 (int8 centroids, varint list lengths) and dense_ann format 2 (int8 PQ codebook; IVF centroids `auto`, which means int8 below 1,024 lists and PQ above). Earlier late indexes were affected by a parser bug that ignored every space-separated option after the first. At 10k the defaults happened to equal the intended parameters, but at 1M they would not have. `build_db.py` now decodes each late index's header after building, records the stored parameters in the manifest (`ext_config.late`: format, K, nbits, layout, centroid type) and stops if they differ from the requested ones. The words-1m FTS5 database contains no vector index and was not rebuilt.

## Files

| File | Purpose |
|---|---|
| `tools/build_db.py` | Builds one deployable database per corpus (`build/matrix/<corpus>.db`) and its manifest (`.db.json`); `--split` also builds one database per index |
| `tools/encode_queries.py` | Encodes a query set with MiniLM and LateOn, one query at a time, and times each encode (`build/queries/`) |
| `bench/matrix_config.json` | Corpora, indexes (with per-corpus parameter overrides), query configurations (SQL), profiles, run sizes |
| `bench/matrix.py` | Driver: `quality`, `trace`, `sim`, `real`, `report`, `all` |
| `bench/matrix.mjs` | Node 22 runner: executes a job of queries through `wasm/pkg` (Asyncify) and writes one JSON line per query |
| `bench/matrix_report.py` | Generates `results/matrix/README.md` and `index.html` |

## Design decisions

**One database per corpus.** A deployment ships one file that holds the documents, FTS5, and every vector index, so that is what is measured. Attribution is still clean: `dbstat` gives each index's pages, and a query reads only its own index's pages plus page 1 and the schema. This was checked against per-index databases (`build_db.py --split`) on words-100, 40 cold queries per configuration:

| Configuration | Combined DB: rounds / requests / KB | Own DB: rounds / requests / KB |
|---|---|---|
| FTS5 bm25 | 7.8 / 7.8 / 31.3 | 7.8 / 7.8 / 31.3 |
| graph ef 64 | 8.0 / 58.5 / 488 | 7.0 / 57.5 / 484 |
| IVF nprobe 64 | 6.0 / 6.3 / 347 | 5.0 / 5.3 / 343 |
| warp nprobe 8 + rerank 64 | 6.0 / 16.5 / 352 | 6.0 / 16.5 / 352 |

The only difference is that the combined schema occupies a second page, which the dense extension reads in its own round when a connection starts (one round, 4 KB). Warm-session costs are identical.

**Build and query binaries.** Indexes are built with threaded copies of the extension sources, compiled by `build_db.py` into `build/matrix/bin/<source digest>/` (the `make native` libraries are single-threaded), with `threads=2`. Queries run on `make native` (quality) and `make wasm` (costs and latency) builds of the same sources. The manifest records the SHA-256 of the extension sources and the git HEAD. The order is: documents, FTS5 (`rebuild`, `optimize`), vector indexes (`build`), `VACUUM`, then each extension's `finalize` (page hints must follow the last VACUUM).

**Data-driven configurations.** Each index in `matrix_config.json` has a kind, a table name and parameters; each query configuration names its index and gives its SQL with `:q`, `:k` and `{table}`. A configuration runs only when its index is in the database, so a new layout (for example a further dense layout) is added by one index entry and its configurations, with no code change, provided it is a `dense_ann`, `late_plaid` or `fts5` table.

**Quality natively, costs in WASM.** Quality needs every query at two depths (k = 10, and k = 100 for AUC), which is fastest natively. The WASM runs return the same ids (the report checks agreement per configuration). AUC uses the k = 100 list, treating unreturned documents as tied below the returned ones; for the graph index k = 100 raises the candidate list to at least 100, so its AUC describes a slightly more expensive search than its other metrics. Exhaustive configurations are evaluated on all queries when N × queries ≤ 3·10⁷ and their costs on 200 queries.

**Warm and cold.** *Warm* is one connection answering the whole query sequence (kinds interleaved) after one warm-up query, with the default 4 MiB block cache: per-connection static data is loaded and pages shared between queries are cache hits. *Cold* opens a new connection per query and includes the open, the schema and the static data. HTTP connection setup (TCP, TLS) is not included; `netsim`'s `cold` modifier could add it.

**Simulated latency for all queries, real latency for a subset.** Every query's request log is recorded against the unshaped server with timestamps, so the CPU time between rounds (WASM, Asyncify) is known; the log is turned into a netsim trace (one round per backend call) and simulated under every profile with `h1` and `h2`. The real-network runs use the same runner against a server shaped with each profile (one server per profile, four at a time) on 50 queries per kind warm and 10 per kind cold. Exhaustive configurations run for real only on `none` and `lan`. The current results have real runs on `4g` and `lte`, each with `h1` and `h2`; the previous (int8) matrix also ran all eight profiles, with the same agreement. The report compares the real wall times with the pipeline estimate for the same queries.

## Corpora

| Corpus | Database | Measured |
|---|---|---|
| words-100, llm-100, words-10k, llm-10k | one combined database each | quality and traces for every query (up to 1,000 per kind), simulation for all profiles × h1/h2, real runs on `4g` and `lte` × h1/h2 |
| words-1m | one database per index (`--split`), no VACUUM | as above, on 250 queries per kind; late index warp-only; no exhaustive rows |

**Query subsets.** Where a kind has more queries than the cap (llm-10k: 10,000 per kind, cap 1,000; words-1m: 1,000 per kind, cap 250), the evaluated queries are a seeded random sample of that kind (`interleave_kinds`), not the first N. In the LLM query sets query *i* is about document *i*, and FTS5 breaks ties by rowid, so a first-N subset favours methods that produce many ties: on the first 1,000 per kind bm25c scores 0.787 against 0.757 for bm25, a gap that disappears over all 10,000. A random sample was chosen over evaluating all queries so that quality, traces and real runs use the same queries; the real-run subset (50 per kind) is a prefix of the same sample. Sets within the cap (words-100, words-10k, llm-100) are evaluated completely.

**FTS5 ranking.** `fts-bm25c-or` ranks with `bm25c()` from `ext/fts5rank` (bm25 with every document at the average length, docs/fts5-httpvfs.md), which needs no per-document `fts_docsize` reads; it is the FTS5 row of the headline table. `fts-bm25-or` (standard bm25) is kept next to it. Its R@10-vs-exact is measured against standard bm25.

The LLM query kinds are `word` (the word the paragraph was written about) and `llmq` (the model's own search query for it). `llmq` results are also reported split by `contains_word`, that is, by whether the model used the word despite being asked not to.

## words-1m

Disk is the constraint: about 20 GB are free and the LateOn input alone is 10.7 GB. A combined 1M database would be about 8 GB, and VACUUM needs a second copy. The run (`build/matrix/chain1m.sh`) therefore handles one index at a time:

- It builds `words-1m--<index>.db` with `--split-only --no-vacuum`. Without VACUUM, the float32 build buffer of a dense index (1.5 GB) stays on the free list. It occupies disk but is never fetched, and `dbstat` sizes exclude it.
- It runs quality, trace, simulation and the real runs, then deletes the database, keeping its manifest (`.db.json`, sizes and parameters), traces and results. Only about 9 GB of disk were free, so no words-1m database is kept. Rebuild one with the recipe below to query it.
- `bench/matrix.py report words-1m` merges the per-index results into one table (`merge_splits`).

Index parameters at 1M:

- **Late:** `layout=warp nbits=2 centroids=65536 fast_assign=1 sample_tokens=2000000`. The rerank and plaid configurations need the plaid document rows, so they are skipped automatically (`needs_layout`).
- **Dense:** same parameters as at 10k (nlist = 4√N = 4,000).
- **Quality:** measured on 250 queries per kind (`max_per_kind`). Exhaustive rows are skipped (`skip_exhaustive`). Float-exact MaxSim is not computed above 60 M token vectors, so late R@10-vs-exact is empty at 1M.

Recipe, by hand:

```sh
python3 tools/encode_queries.py words-1m
python3 tools/build_db.py words-1m --split-only --indexes fts            # then late, dense_ivf, dense_graph with --no-vacuum
python3 bench/matrix.py quality words-1m--fts   # likewise trace, sim, real --profiles 4g,lte
python3 bench/matrix.py report
```

## Findings worth knowing when reading the tables

- **Warm sessions cache a lot.** At 10k the VFS block cache (4 MiB) holds a large share of the smaller indexes after a few hundred queries: FTS5 needs 0.2–0.3 rounds per warm query, and at llm-10k the warp posting stream (about 8 MB) is served almost entirely from cache (3 KB transferred against 125 KB read by the extension). The *Ext. KB* column shows the extension's own reads before the cache.
- **The dense exhaustive baseline (`exact=1`) reads its vector table one page per round** (2,006 rounds per query at 10k): the scan is neither prefetched nor detected as sequential by readahead. It is a baseline for quality only; over a network it would need batching.
- **HTTP/1.1 costs a lot, and multi-range requests win it back.** The per-corpus section *HTTP/1.1 with the request budget* re-simulates the traces with `bench/coalesce_eval.py` (the VFS's own planner via ctypes). At 10k on 4g, warm, plain h1 is 1.5–3.4× slower than h2 for the multi-request systems (graph 2,409 vs 1,240 ms at llm-10k; warp + rerank 2,086 vs 622 ms at words-10k). With at most six multi-range requests per round, h1 comes within 1 % of h2. Coalescing with over-fetch recovers only part of the gap.
- **Cold starts are dominated by static data.** Late interaction's centroid table (1.7 MB at K = 16,384) makes its first query 2.5–3.5 s on 4g even though a warm warp query is 0.2 s.

## FTS5 ranking

FTS5's `bm25()` reads one `fts_docsize` row per matching document, one round trip each: 685 rounds for a cold single-word query at 1M. The client therefore ranks with `bm25c()` (`ext/fts5rank`, bm25 at the average document length) by default. The configuration `fts-bm25c-or` measures it next to `fts-bm25-or` in future matrix runs. The current `results/matrix/` tables predate it and show `bm25()`.

The comparison of ranking methods is not part of the matrix:

- `bench/fts5_rank.py quality|cost|report` compares bm25, bm25c, bm25-rerank, bm25-prefetch and rowid order on quality, cold and warm rounds and bytes, and simulated and real 4G/LTE latency, through `web/lib/search-core.mjs` in the WASM build;
- `bench/fts5_options.py` compares the FTS5 table options.

Results are in `results/fts5-rank/*.json`, write-up in `docs/fts5-httpvfs.md`.

## Caveats

- The machine was shared with an LLM generation job and other agents' builds (load average 8–11 during these runs), so CPU gaps in the traces, and hence the `none` and `lan` latencies, are pessimistic. Round, request and byte counts are exact.
- The Asyncify build has more CPU overhead than JSPI (wasm/NOTES.md); CPU is a few ms per query and matters only on the fastest profiles.
- Node's `fetch` has no per-host connection limit; `h1` is enforced by the server (six requests in service at once, FIFO queue), matching the simulator.
- The HTTP/2 profile is modelled as 100 concurrent HTTP/1.1 requests (see netsim/NOTES.md).
- `lte-poor` is random; real and simulated runs agree in distribution, not per query.
