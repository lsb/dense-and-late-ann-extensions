# Browser demo and client library

`web/` makes the project usable in a browser. A page opens one SQLite database over HTTP range requests, encodes the query in the browser, and searches with FTS5, dense ANN (MiniLM + `dense_ann`) or late interaction (LateOn-Code-edge + `late_plaid`), alone or side by side. Nothing is loaded from a CDN; every file comes from this repository.

## Running it

```sh
make native wasm                          # build/native and wasm/pkg/dist (see wasm/NOTES.md)
python3 web/make_demo_db.py               # build/web/words-10k.db (94 MB), see "Demo database"
python3 netsim/rangeserver.py --dir . --port 8000 --preset 4g
# open http://localhost:8000/web/
```

With `--preset none` the page is fast; switching the "Network profile" selector posts to `/__netsim/profile`, so the next requests are shaped. The profile applies to every file the server sends, including the 14 MB ONNX Runtime binary and the 40 MB of models, so on `4g` the encoders take about a minute to download the first time. After that they come from Cache Storage (see below). The page picks the first database it finds among `build/web/words-10k.db`, `build/matrix/words-10k.db` and `build/matrix/words-100.db`; `?db=…`, `?variant=asyncify|jspi`, `?q=…&system=all|fts|dense|late` override or auto-run.

Tests and measurements:

```sh
python3 web/test/make_reference.py        # Python reference tokenizations/encodings -> build/web/
node web/test/tokenizer_parity.mjs        # tokenizer check in Node, 6,540 texts
node --test web/test/web.test.mjs         # Chromium: encoder parity, one query per system, browser == native
python3 web/test/compare_retrieval.py     # does the encoding difference change exact top-10? (after web.test)
node web/test/profile_run.mjs --profiles 'none;lte,h1;4g,h1'   # page/encoder/query timings per profile
node web/test/ttfr.mjs --dbs 'a=build/x/a.db;b=build/x/b.db' --variants 'a:off,a:auto,b:auto' \
     --profiles '4g,h1;4g,h2' --visits first,repeat --delays 0,3000   # time to first result (HTTP/2 via h2proxy.mjs)
./web/vendor.sh                           # refresh web/vendor/ from npm (versions pinned in package.json)
```

`make web` runs these steps (`make_demo_db.py`, then `node --test web/test/web.test.mjs`), and `make serve PRESET=4g` starts the shaped server.

## Architecture

```
page (index.html, demo.mjs)
  └─ lib/index.mjs            openIndex(url, opts) -> SearchIndex
       ├─ Worker lib/sqlite-worker.mjs       wasm/pkg (SQLite + httpvfs + dense_ann + late_plaid)
       ├─ Worker lib/encoder-worker.mjs      MiniLM   (onnxruntime-web + tokenizers.js)
       └─ Worker lib/encoder-worker.mjs      LateOn-Code-edge
```

| File | Role |
|---|---|
| `lib/index.mjs` | Public API. Starts the workers, discovers the indexes, routes a search to its encoder and to SQLite, collects timings. |
| `lib/sqlite-worker.mjs` | Opens the database with `wasm/pkg/index.mjs` and runs `search-core.mjs` on it. Requests are handled one at a time so that the counters of a search belong to it alone. |
| `lib/search-core.mjs` | `SearchDb`: reads `sqlite_schema` to find FTS5, `dense_ann` and `late_plaid` tables (and the layout of each); builds the query SQL; fetches document text; returns VFS counter deltas per phase. No workers or encoders, so it also runs in Node with precomputed query vectors. |
| `lib/paths.mjs` | Default URLs of the SQLite module, ONNX Runtime, the tokenizers and the models (the npm package ships its own version; see `docs/INSTALL.md`). |
| `lib/encoder-worker.mjs` | Loads one model and encodes queries. |
| `lib/encoder-core.mjs` | Encoding without environment assumptions (also runs in Node): tokenize, run ONNX, mean-pool and normalise (MiniLM) or return the per-token vectors (LateOn). |
| `lib/tokenize.mjs` | Reproduces the Python preprocessing on top of `@huggingface/tokenizers`: Python `str.strip()` whitespace set, truncation that keeps `[SEP]`, MiniLM's 256-token limit, LateOn's lower-casing, 256-token query length and `[Q]` prefix after `[CLS]`. |
| `lib/fetch-cache.mjs` | `fetch` with progress, through Cache Storage. |
| `lib/rpc.mjs` | Small postMessage request/response helper with progress notifications. |

### API

```js
import { openIndex } from './web/lib/index.mjs';
const ix = await openIndex('/build/web/words-10k.db', { preload: ['minilm', 'lateon'] });
ix.systems;   // [{id:'fts', kind:'fts'}, {id:'dense_graph', kind:'dense', layout:'graph'}, {id:'dense_ivf', …}, {id:'late', kind:'late'}]
const r = await ix.search('pinwheel gossiping', { system: 'late', k: 10, nprobe: 8, rerank: 64 });
r.rows;       // [{id, score, text}]
r.stats;      // {wallMs, encodeMs, tokenizeMs, inferMs, encoderLoadWaitMs, searchMs, docsMs,
              //  rounds, requests, bytes, netMs, phases: {search: {...}, docs: {...}}}
r.ext;        // the extension's own per-query stats JSON (rounds, pages, candidates, …)
r.embedding;  // the query vector(s), Float32Array
await ix.encode('text', 'minilm');      // {vectors, ids, n, dim, ms, …}
await ix.loadEncoder('lateon', onProgress);   // load stats: bytes, download ms, session ms, …
```

`system` is `'fts'`, `'dense'` (the graph index if present, else IVF), `'late'`, or a table name. Other options are passed to the extension as query constraints, with defaults from `DEFAULT_PARAMS`: dense graph `ef=64 beam=16 rerank=2`; dense IVF `nprobe=16 rerank_k=64 rerank=2`; late `nprobe=8 layout=warp stoplist=0.02 rerank=64` (the recommendation in `ext/late/NOTES.md`); FTS5 `mode=or` (each word quoted and OR-ed) and `rank=bm25c` (bm25 with every document at the average length, from `ext/fts5rank`: FTS5's own `bm25()` reads one `fts_docsize` row, that is one round trip, per matching document; `rank=bm25` restores it, `bm25-prefetch` gives the exact bm25 with those rows fetched in one batch, `bm25-rerank` re-scores bm25c's top 50 exactly, `none` is rowid order; measurements in `docs/fts5-httpvfs.md`). A build without `bm25c` falls back to `bm25`. Numeric parameters are checked to be integers before they go into SQL; the late `opts` string is bound as a parameter. `cold: true` empties the VFS block cache and SQLite's page cache first; the extensions' own per-connection caches (dense entry set and codebook, late static data and interior pages) stay. `openIndex` options: `variant` (`auto` picks JSPI where available, else Asyncify), `pageCacheBytes` (default `'auto'`: 1/64 of the database, clamped to `pageCacheMinBytes`, 16 MiB, and 64 MiB), `blockSize`, `readaheadBytes`, `maxParallel`, `modelBase`, `models`, `modelCache`, `encoderThreads`, and the file locations `sqliteModule`, `ortBase`, `tokenizersModule` (defaults in `lib/paths.mjs`).

**Warm-up.** Each index has per-connection static data (dense: PQ codebook and entry set or IVF centroids; late: centroids and list lengths; FTS5: structure record) that the first search would otherwise fetch before doing any work. `ix.warm(systems?)` asks the SQLite worker to load it now, by running each extension's `MATCH 'warm'` statement (`warmSql` in `search-core.mjs`; for FTS5 a lookup of a term that does not occur); it resolves to `[{table, ms, rounds, requests, bytes}]` and warms each index once per connection. The worker runs warm-ups at low priority: a search that arrives while warm-ups are queued goes first (one already running is not interrupted). `openIndex(url, {warm})` chooses when: `'auto'` (default) warms, as soon as an encoder starts loading, the index that `search({system: 'dense' | 'late'})` uses (the graph if the database has one) (through `preload` or a first search), so the static data arrives while the model loads and the query is encoded; `true` warms every index at once, FTS5 included; `false` or a list of systems limits it. FTS5 is not warmed automatically: its "static data" is a few b-tree pages reached in about five dependent rounds (1.4 s on `4g` with HTTP/1.1), which would only delay a vector search that arrives meanwhile. See *Time to first result* below.

Document text comes from the FTS5 table's content table (`docs(id, body)` in `tools/build_db.py` databases). The lookups are batched with `httpvfs_warm` (from `wasm/demo_ext.c`), so fetching ten documents costs one or two rounds instead of ten.

### Model loading and caching

The netsim server sends `Cache-Control: no-store`, so the browser's HTTP cache never keeps the models. The encoder workers therefore keep the ONNX files, the tokenizers and the ONNX Runtime binary in Cache Storage (`dense-late-demo-models-v1`; "keep models in Cache Storage" on the page, `modelCache: false` in the API). The 14 MB ONNX Runtime binary is fetched once by the page and handed to both workers as `env.wasm.wasmBinary`; before that change both workers downloaded it. Cache Storage needs a secure context (`localhost` qualifies); elsewhere the cache is skipped.

ONNX Runtime runs single-threaded unless the page is cross-origin isolated (`rangeserver.py --isolate`) and `encoderThreads` > 1; that path was not measured. The two models run in separate workers, so they encode in parallel on a multi-core device.

## Sizes

Raw bytes and `gzip -9` (the netsim server does not compress, so the raw size is what it transfers).

| What | Raw | gzip |
|---|---:|---:|
| `web/lib/*.mjs` + `demo.mjs` (our code) | 40 KB | 14 KB |
| `index.html` + `demo.css` | 8.8 KB | 3.1 KB |
| `vendor/tokenizers.min.mjs` (@huggingface/tokenizers 0.2.0) | 37 KB | 11 KB |
| `vendor/ort/ort.wasm.min.mjs` + `ort-wasm-simd-threaded.mjs` (onnxruntime-web 1.30.0) | 74 KB | 25 KB |
| `vendor/ort/ort-wasm-simd-threaded.wasm` | 14.2 MB | 3.66 MB |
| `wasm/pkg` JSPI build (`.wasm` + `.mjs`) + `index.mjs` | 1.55 MB | 621 KB |
| `wasm/pkg` Asyncify build (`.wasm` + `.mjs`), used where JSPI is missing | 2.52 MB | 895 KB |
| MiniLM `model_qint8_arm64.onnx` + `tokenizer.json` | 23.0 + 0.47 MB | 17.4 + 0.19 MB |
| LateOn `model_int8.onnx` + `tokenizer.json` | 17.2 + 3.58 MB | 13.4 + 0.64 MB |
| Demo database `words-10k.db` (read by range, never downloaded whole) | 94.1 MB | |

A first visit that uses both encoders transfers about 60 MB uncompressed (about 36 MB with gzip or brotli from a CDN), almost all of it the two models and ONNX Runtime. FTS5 alone needs only the SQLite build (1.6 MB). `web/vendor/` is 14 MB, of which 14.2 MB is the ONNX Runtime binary; it is committed so that the demo works offline, but it could instead be fetched by `vendor.sh` and ignored by git. Licenses are in `web/vendor/LICENSES.md` (tokenizers: Apache-2.0; onnxruntime-web: MIT).

The database has FTS5 2.4 MB, docs 4.8 MB, `dense_graph` 39.6 MB, `dense_ivf` 9.4 MB and `late` 37.8 MB (both layouts).

## Encoder correctness

**Tokenization is exact.** `tokenizer_parity.mjs` compares `lib/tokenize.mjs` with the Python `tokenizers` output that `enc/` uses on 6,540 texts (40 edge cases with accents, CJK, emoji, combining marks, zero-width and non-breaking spaces, literal special tokens and 300-character words; 3,000 queries; 3,000 random-word documents; 500 LLM paragraphs, some past the 256-token limit): all identical for both models. Two fixes were needed. First, the Rust BERT normalizer lower-cases code point by code point, while JavaScript's `toLowerCase()` on a whole string turns a word-final `Σ` into `ς`; `Σ` is mapped to `σ` before tokenizing (lower-casing the whole text first would break the matching of literal `[CLS]`-style special tokens). Second, Python's `strip()` and JavaScript's `trim()` remove different whitespace sets, so `pyStrip` copies Python's.

**Vectors agree closely but not to 0.999 everywhere.** On 112 queries (12 handmade, 50 single words and 50 three-word queries from `data/queries/words-10k.jsonl`), each encoded alone as in the browser, against `enc/minilm.py` and `enc/lateon.py`:

| | MiniLM (112 vectors) | LateOn (928 token vectors) |
|---|---|---|
| Browser vs Python (`enc/`, ORT graph optimisations on): mean cosine | 0.99888 | 0.99974 |
| min / 1st percentile | 0.9811 / 0.987 | 0.9869 / 0.996 |
| share of vectors with cosine ≥ 0.999 | 84 % | 91 % |
| For scale: Python optimised vs Python unoptimised, mean / min | 0.99979 / 0.9962 | 0.99996 / 0.9962 |
| Exact top-10 overlap on words-10k, browser vs Python encodings | 0.975 (identical for 85 % of queries) | 0.995 (identical for 96 %) |
| Same, Python optimised vs unoptimised | 0.993 | 0.998 |

The target of cosine ≥ 0.999 per vector is met by 84–91 % of vectors, not all. The cause was traced and is not a bug in the browser path:

- Both models use dynamic int8 quantisation: before every `MatMulInteger`, `DynamicQuantizeLinear` picks a scale from the tensor's range and rounds every activation to 8 bits.
- Feeding identical inputs, the first `MatMulInteger` gives bit-identical int32 results in onnxruntime-web and in Python's onnxruntime, and both equal an exact NumPy computation. The integer kernels are correct.
- The float operators differ in the last bits (LayerNorm, Softmax, GELU: differences of 1–5·10⁻⁶). When a value lies near a rounding boundary, one quantised activation flips by one step; for the query in the worst case above, one activation of 5,760 flipped after the embedding LayerNorm, 21 flipped by the next quantisation, and the differences grow through six layers.
- Python's onnxruntime shows the same effect against itself: with graph optimisations on (fused kernels, what `enc/` uses) or off, its outputs differ by up to 0.004 in cosine. onnxruntime-web gives the same output at every optimisation level, and matches neither Python variant exactly.

The effect on retrieval is small (top-10 overlap 0.975 and 0.995 on exact search) and smaller than the batch-size effect recorded in `enc/NOTES.md` (mean cosine 0.988 between a text encoded alone and in a batch of 64). A phone's ARM kernels will differ again, in the same way. Getting 0.999 everywhere would need float32 models, 4× larger (90 MB for MiniLM). The Playwright test therefore requires identical token ids, a mean cosine ≥ 0.995 and a minimum ≥ 0.95, and reports the rest.

**The WASM extensions return exactly what the native build returns.** For the same query vector (taken from the browser) and the same parameters, the dense graph, dense IVF and late (warp + rerank) results in the browser are identical, row for row, to Python + `build/native` with extensions compiled from the same sources; FTS5 likewise for the same MATCH expression.

## Timings

All in headless Chromium 141 on the 4-core container while an LLM generation job, a 1M-document encoding job and other agents' benchmarks kept the load average at 8–11. CPU timings are noisy and probably pessimistic; network timings come from netsim and are close to its model.

**Encoding, warm** (after loading; single thread; `web.test.mjs`):

| | MiniLM | LateOn |
|---|---:|---:|
| short query (median 7–8 tokens), median | 7–11 ms | 3.5–4 ms |
| 50-word query (114–117 tokens), median | 64 ms | 29 ms |
| tokenization | 0.1 ms | 0.1 ms |
| worker round trip added | 1–2 ms | 0.5–1 ms |

**Loading an encoder** (from localhost, no shaping): download of the model 0.2–0.4 s, ONNX session creation 0.6–1.9 s, first inference 20–250 ms; 0.9–2.2 s in total, and about the same from Cache Storage, since session creation dominates. The LateOn tokenizer (3.6 MB of JSON) takes about 60 ms to parse.

**The whole page under network profiles** (`profile_run.mjs`: a fresh browser profile, open the page, load both encoders, then run five three-word queries through each system in turn; "first" is the first query of the session, "later" the median of the other four; all numbers are wall times seen by the caller, including encoding and fetching the text of the ten hits):

| | none | lte,h1 (70 ms, 12 Mbit/s) | 4g,h1 (165 ms, 8.1 Mbit/s) |
|---|---:|---:|---:|
| page + SQLite + open database | 0.2 s | 1.9 s | 3.4 s |
| both encoders, first visit (ORT 14 MB + models 40 MB) | 2.2 s | 41 s | 60 s |
| both encoders, next visit (Cache Storage) | 2.4 s | 2.3 s | 2.9 s |
| FTS5 first / later query | 0.25 / 0.08 s | 2.1 / 0.58 s | 4.7 / 1.3 s |
| dense graph (ef 64, W 16) first / later | 0.31 / 0.22 s | 2.3 / 1.5 s | 4.8 / 3.1 s |
| late (warp nprobe 8 + rerank 64) first / later | 0.64 / 0.30 s | 3.6 / 1.5 s | 6.5 / 3.3 s |

Later queries fetch: FTS5 6 rounds, 14 requests, 56 KB; dense 7 rounds, 82 requests, 328 KB; late 3 rounds, 97 requests, 420 KB (all including the document text: 1–2 rounds, about 11 requests). The first query of a session reads more because it also loads static data; its totals are FTS5 25 rounds and 168 KB (structure pages), dense 14 rounds and 650 KB (codebook and entry set), late 7 rounds and 2.3 MB (centroid table).

What this says:

- **Encoding is not the bottleneck** once the models are loaded: 4–20 ms per short query, against 1–3 s of network time per query on 4G.
- **The first visit is dominated by downloading the models**: a minute on 4G. Cache Storage fixes repeat visits. Smaller models (int4 or distilled) or loading only the encoder a query needs would help the first one; FTS5 works immediately.
- **HTTP/1.1 costs more than round trips here.** A dense round fetches 16 pages and a late round up to 100 lists and documents; with six connections per host these become several latency steps each. On 4g the later dense query takes 3.1 s for 7 rounds (1.2 s of latency if every round were one parallel step), and the late query 3.3 s for 3 rounds. HTTP/2 from a real CDN, or multi-range requests, would bring both close to rounds × RTT; netsim's simulator (`h2`) predicts that, but a browser talking to this HTTP/1.1 server cannot show it. *Update:* the VFS now has a per-round request budget (6 over HTTP/1.1, detected automatically). With multi-range requests (`?multipart=1`), the same `4g,h1` measurement gives 1.8 s for the dense graph and 1.0 s for late; with coalescing only, 3.3 s and 3.0 s (wasm/NOTES.md, "Request budget per round").
- Fetching the text of the hits costs 1–2 more rounds (0.2 s on lte, 0.4 s on 4g). Storing a short snippet next to each vector, or starting that fetch while re-ranking, would hide it.

## Time to first result

`web/test/ttfr.mjs` measures how long a page takes to show its first result, in Chromium (headless shell 141) against `netsim/rangeserver.py`: `web/test/ttfr.html` opens the database with `preload` of the encoder the system needs and searches as soon as the index is open (*delay 0*), or 3 s later, as if the user were typing (*delay 3,000*; then the time is counted from submitting the query). HTTP/1.1 profiles (`h1`) talk to the range server directly, so Chromium opens at most six connections; the `h2` profiles go through `web/test/h2proxy.mjs`, a real HTTP/2 (TLS) proxy whose upstream range server runs the `,h2` preset. *First visit*: an empty browser profile, so ONNX Runtime and the model are downloaded through the shaped link. *Repeat visit*: the same profile after one unshaped visit, so the encoder comes from Cache Storage (the page, the SQLite build and the database pages are fetched again, because the range server sends `no-store`). Database: llm-10k with every index. *before* is the matrix database built by the previous code (float16 codebooks and centroids, late format 2), *after* is rebuilt with the new defaults (int8 codebooks, IVF centroids and late centroids); *off* / *auto* is `openIndex(…, {warm})` (the harness names the queried index, as `'auto'` does for `dense` and `late`). Query "volcanic eruption", k = 10 with document text, default parameters (graph ef 64, IVF nprobe 16, late warp nprobe 8 + rerank 64). Medians of 2 runs (repeat) or 1 run (first visit), in seconds; all runs are in `results/static/ttfr-llm-10k.json`.

Repeat visit, query typed 3 s after the index opened (time from submitting the query):

| System | Profile | before, no warm-up | before, warm-up | after, no warm-up | **after, warm-up** |
|---|---|---|---|---|---|
| dense graph | 4g h1 | 4.70 | 3.81 | 4.38 | **3.66** |
| dense graph | 4g h2 | 2.88 | 2.06 | 2.78 | **2.06** |
| dense graph | lte h1 | 2.22 | 1.87 | 2.08 | **1.76** |
| dense graph | lte h2 | 1.56 | 1.07 | 1.47 | **1.09** |
| dense IVF | 4g h1 | 3.14 | 2.29 | 2.93 | **2.05** |
| dense IVF | 4g h2 | 2.29 | 1.21 | 2.02 | **1.23** |
| dense IVF | lte h1 | 1.67 | 1.00 | 1.43 | **1.01** |
| dense IVF | lte h2 | 1.27 | 0.67 | 1.08 | **0.66** |
| late | 4g h1 | 4.81 | 3.17 | 4.24 | **3.08** |
| late | 4g h2 | 2.92 | 1.36 | 2.50 | **1.38** |
| late | lte h1 | 2.33 | 1.42 | 2.07 | **1.42** |
| late | lte h2 | 1.64 | 0.76 | 1.37 | **0.75** |

Repeat visit, query submitted as soon as the index is open (time from navigation start):

| System | Profile | before, no warm-up | before, warm-up | after, no warm-up | **after, warm-up** |
|---|---|---|---|---|---|
| dense graph | 4g h1 | 8.09 | 7.98 | 7.79 | **7.62** |
| dense graph | 4g h2 | 6.33 | 6.27 | 6.20 | **6.09** |
| dense graph | lte h2 | 3.48 | 3.43 | 3.45 | **3.35** |
| dense IVF | 4g h1 | 6.68 | 6.92 | 6.73 | **6.34** |
| dense IVF | 4g h2 | 5.73 | 5.57 | 5.35 | **5.33** |
| dense IVF | lte h2 | 3.23 | 3.14 | 3.01 | **2.96** |
| late | 4g h1 | 9.20 | 8.35 | 8.16 | **7.77** |
| late | 4g h2 | 6.28 | 6.31 | 5.85 | **5.90** |
| late | lte h2 | 3.59 | 3.54 | 3.33 | **3.32** |

First visit, query submitted as soon as the index is open (time from navigation start; the encoder download dominates):

| System | Profile | before, no warm-up | before, warm-up | **after, warm-up** |
|---|---|---|---|---|
| dense graph | 4g h1 / h2 | 46.0 / 44.2 | 45.5 / 43.7 | **45.2 / 43.8** |
| dense graph | lte h1 / h2 | 30.1 / 29.5 | 30.0 / 29.2 | **29.7 / 29.2** |
| dense IVF | 4g h1 / h2 | 44.4 / 43.7 | 43.8 / 43.1 | **43.6 / 42.9** |
| dense IVF | lte h1 / h2 | 29.5 / 29.2 | 29.2 / 28.9 | **29.1 / 28.9** |
| late | 4g h1 / h2 | 43.6 / 41.8 | 42.9 / 41.2 | **42.4 / 40.6** |
| late | lte h1 / h2 | 28.6 / 28.0 | 28.4 / 27.7 | **27.9 / 27.3** |

What this shows:

- **When there is anything to hide behind, the warm-up removes the static data from the first query entirely**: with 3 s of typing, the first query costs what a query within a session costs. It is 30–55 % faster (late on `4g` h2: 2.92 → 1.38 s; IVF 2.29 → 1.23 s; graph 2.88 → 2.06 s). Smaller static data then no longer matters for this query, only for the bandwidth it takes from other downloads.
- **On a repeat visit the encoder is not the long pole here.** It loads from Cache Storage in 1.1–1.7 s (session creation), while opening the page, the SQLite build (1.5 MB, fetched again because of `no-store`) and the database takes 1.9 s on `lte` and 3.4 s on `4g` h1. A query submitted the moment the index opens therefore finds nothing loaded yet, and the warm-up can only overlap with the encoding (tens of ms). The gains there (0.1–1.4 s) come mostly from the smaller static data. On a phone, where creating the ONNX session takes several times longer, the warm-up would hide more. HTTP caching of the page and the SQLite build would shorten the open.
- **On a first visit the warm-up saves only 0.3–0.7 s of 28–46 s**: the static data competes for the same link as the 14 MB ONNX Runtime and the 17–23 MB model, so moving it earlier saves its round trips, not its transfer time; shrinking it (after) saves a further 0.2–0.5 s.
- Warming an index the page does not query costs: in a first version `'auto'` also warmed FTS5 (five dependent rounds, 1.4 s on `4g` h1) and every dense index, and a search that arrived while one of those was running waited for it (IVF first query 1.6 → 1.9 s on `lte` h1). `'auto'` now warms only the index the default search uses.

## Demo database

`web/make_demo_db.py` produces `build/web/words-10k.db` with the benchmark matrix's builder, `tools/build_db.py` (documents, FTS5 `rebuild` + `optimize`, `dense_ann` graph and IVF layouts, `late_plaid` with both layouts and K = 16,384, then VACUUM and every extension's `finalize`; parameters in `bench/matrix_config.json`). If `build/matrix/words-10k.db` exists, is complete and was built from the current extension sources (SHA-256 digests in its manifest), the script copies it; otherwise it runs the builder with `--out-dir build/web` (about 8 minutes here, mostly the late index's k-means). The copy used for the measurements above was built by the matrix agent from the tree of 2026-09-24 01:3x.

## Limitations and open issues

- Only Chromium was tested (the only browser here). The JSPI build is used there; other browsers get the Asyncify build. Cache Storage, module workers and WebAssembly SIMD are needed: Chrome 80+, Firefox 114+, Safari 16.4+ (untested).
- Encoder vectors match the Python reference to a mean cosine of 0.999–0.9997, not 0.999 per vector (see above).
- The page shows the extension's parameters but not every option (`impute`, `cross`, `tcs`, `exact`); the API accepts them.
- `wallMs` in "all side by side" mode includes waiting for the other systems, because one SQLite worker serves all three; `searchMs` and `docsMs` are each system's own time.
- `cold` clears the VFS and SQLite caches but not the extensions' per-connection caches; reopen the database for a truly cold start.
- No mobile device was measured; the network profiles simulate the link, not a phone's CPU. ONNX inference on a mid-range phone is probably 3–5× slower than here, which still leaves encoding well under the network time.
- The netsim profile applies to the page's own assets and models too, which is realistic for a first visit but makes switching profiles mid-download confusing.
- `wasm/pkg/dist` must be rebuilt (`make wasm`) whenever the extension sources change; a stale JSPI build (it lacked the dense IVF layout when this work started) makes queries on new layouts fail.
