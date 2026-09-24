# Benchmark matrix: notes

The benchmark matrix measures every retrieval configuration of the project end to end: the size it adds to the deployed database, its retrieval quality against the relevance labels, and its fetch costs and latency when the WebAssembly SQLite build reads the database over HTTP range requests. Results are in `results/matrix/` (`README.md` with tables, `index.html` with charts, `<corpus>.json` with every number).

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

**Simulated latency for all queries, real latency for a subset.** Every query's request log is recorded against the unshaped server with timestamps, so the CPU time between rounds (WASM, Asyncify) is known; the log is turned into a netsim trace (one round per backend call) and simulated under every profile with `h1` and `h2`. The real-network runs use the same runner against a server shaped with each profile (one server per profile, four profiles at a time) on 50 queries per kind (10 on `slow-4g`, `3g`, `lte-poor`) warm and a fifth of that cold; exhaustive configurations run for real only on `none`, `lan` and `wifi`. The report compares the real wall times with the pipeline estimate for the same queries.

## Recipes for the remaining corpora

The inputs for `words-1m` (MiniLM and LateOn embeddings) and for the LLM corpora (`data/corpora/llm-{100,10k}.txt`, `data/queries/llm-{100,10k}.jsonl`, embeddings `data/emb/llm-10k.*`) are still being produced. Once they exist:

```sh
# LLM corpora (after scripts/make_llm_corpus.py and
#   python3 -m enc.encode_corpus data/corpora/llm-10k.txt --model both --name llm-10k)
python3 tools/build_db.py llm-10k && python3 tools/build_db.py llm-100   # llm-100 uses the first 100 llm-10k vectors
python3 tools/encode_queries.py llm-100 llm-10k
python3 bench/matrix.py all llm-100 llm-10k

# words-1m (after enc.encode_corpus data/corpora/words-1m.txt --model both)
python3 tools/build_db.py words-1m          # hours: late K = 65,536 with fast_assign, graph build
python3 tools/encode_queries.py words-1m
python3 bench/matrix.py quality words-1m    # float-exact MaxSim is skipped above 60 M token vectors
python3 bench/matrix.py trace words-1m && python3 bench/matrix.py sim words-1m
python3 bench/matrix.py real words-1m --parallel 4
python3 bench/matrix.py report
```

Expected cost at 1M, from the extension notes: database of roughly 4 GB (graph) + 0.3 GB (IVF, with f16 vectors 0.8 GB) + 3.9 GB (late, both layouts) + 0.7 GB (docs and FTS5), about 9–10 GB in one file, so disk must be freed first (the words-1m LateOn input alone is 10.7 GB). If that is too large, build the late index with `layout=warp` or split per index (`--split`), which the attribution check above shows gives the same costs. The exhaustive configurations should be dropped at 1M (their quality run is capped automatically; their cost runs read the whole index per query).

## Caveats

- The machine was shared with an LLM generation job and other agents' builds (load average 8–11 during these runs), so CPU gaps in the traces, and hence the `none` and `lan` latencies, are pessimistic. Round, request and byte counts are exact.
- The Asyncify build has more CPU overhead than JSPI (wasm/NOTES.md); CPU is a few ms per query and matters only on the fastest profiles.
- Node's `fetch` has no per-host connection limit; `h1` is enforced by the server (six requests in service at once, FIFO queue), matching the simulator.
- The HTTP/2 profile is modelled as 100 concurrent HTTP/1.1 requests (see netsim/NOTES.md).
- `lte-poor` is random; real and simulated runs agree in distribution, not per query.
