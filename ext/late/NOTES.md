# late_plaid: PLAID-style late-interaction search for httpvfs

`late_plaid` is a SQLite virtual table for late-interaction (ColBERT-style, MaxSim) retrieval. It follows the PLAID index of [fast-plaid](https://github.com/lightonai/fast-plaid) (centroids, a residual codec with 1, 2 or 4 bits per dimension, an inverted file) and adds a centroid-major layout in the spirit of WARP. It is written for read-only databases that a browser reads lazily over HTTP range requests ("httpvfs"), where the number of *dependent* round trips and the bytes read matter far more than CPU time. The specification it follows is `docs/plaid.md`.

The extension is plain C11 with SQLite and libm only. Index building may use POSIX threads (`-DLATE_THREADS`); the query path is single-threaded. It compiles unchanged with Emscripten (`emcc -r`, checked) and as a loadable extension.

## Summary

- Two storage layouts share one set of centroids and one residual codec: **plaid** (fast-plaid's document-major rows plus a bit-packed IVF) and **warp** (centroid-major posting lists of document id + residual). Lists are stored as page-sized rows whose leaf pages are known in advance, so a batch of lists is one parallel round with no b-tree walk.
- On words-10k (nbits = 2, K = 16,384), exact MaxSim over the compressed vectors scores nDCG@10 0.357 against 0.359 for float16 vectors: the codec costs almost nothing. The faithful fast-plaid pipeline also reaches 0.357, but reads every candidate's row: 17 MB per query with its default parameters (17 s on a simulated 4G phone).
- The warp layout answers in **one round**: nDCG@10 0.26 / 0.30 / 0.33 at nprobe 8 / 16 / 32, reading 0.15–0.5 MB (0.3–0.7 s on 4g). Adding a second round that re-ranks the best 64 documents exactly from their plaid rows gives **0.350** for 0.4 MB (0.74 s on 4g, 0.42 s on lte). Ranking candidates from the IVF alone and decompressing only the best 64–128 (plaid layout, `approx=ivf`) gives 0.33–0.34 in two rounds for 0.35–0.6 MB.
- **Stop tokens.** The query's `[SEP]` vector probes lists that contain almost every document and accounted for 75 % of the posting entries fetched. Skipping query vectors whose probed lists are longer than 2 % of N (`stoplist=0.02`) and scoring them with a constant leaves every ranking unchanged and cuts candidates by 70 % and bytes by a third. At 1M documents this is essential.
- Residual bits hardly affect the first stage; nbits = 2 is the sensible default (nbits = 1 costs 0.02 nDCG after re-ranking and saves 43 % of the postings). Four to sixteen thousand centroids behave alike at equal bytes; halving the k-means training set costs about 0.01 nDCG.
- Recommended starting point: `layout=both nbits=2 order=1`, query `layout=warp nprobe=8 stoplist=0.02 rerank=64` (or `nprobe=16 stoplist=0.02` when one round matters more than quality). For 1M documents, K = 65,536 with `fast_assign=1` (about 3 hours to build here, 1.6 GB for the warp layout at nbits = 2, 6.8 MB of static data).

## Files

| File | Contents |
|---|---|
| `late_plaid.c` | Virtual table: configuration, build, finalize, query algorithms, instrumentation |
| `late_codec.c`, `late_codec.h` | k-means, two-level assignment, residual codec (fast-plaid bit order) |
| `late_page.c`, `late_page.h` | Direct b-tree page reader: batched lookups one level per round, interior-page cache, page trace, VFS prefetch hook |
| `late_common.h` | float16, RNG, dot product, bit packing, parallel-for |
| `Makefile` | `make` builds `build/late.so` (entry point `sqlite3_late_init`); `make test` |
| `build_index.py` | Builds an index database from `.npy` embeddings through the C builder |
| `bench.py` | Quality, latency, page traces and simulated network time |
| `countcheck.py` | Cross-checks the extension's page trace against an APSW counting VFS |
| `encode_queries.py` | Encodes the query sets with LateOn-Code-edge, one query at a time |
| `pyref.py` | Pure-Python reader of the storage format (used by the tests) |
| `report.py` | Markdown tables from `results/*.json` |
| `test/test_late.py` | Unit tests (format checked against `pyref.py`, layouts, rounds, VACUUM, two-level) |
| `results/*.json` | Raw benchmark results |

The top-level `wasm/build.mk` finds this directory automatically (`ext/late` → `sqlite3_late_init`); all non-entry symbols are `static` or prefixed `lc_`, `lpg_`, `lt_`, so the extension links statically next to `ext/dense`.

## SQL interface

```sql
CREATE VIRTUAL TABLE t USING late_plaid(
    dim=48, nbits=2,          -- residual bits per dimension: 1, 2 or 4
    centroids=16384,          -- K; 0 = fast-plaid's 2^floor(log2(16*sqrt(T)))
    coarse=0,                 -- G > 0: two-level centroids (G coarse cells)
    layout=both,              -- plaid | warp | both
    kmeans_iters=4, kmeans_ppc=256, sample_docs=0, sample_tokens=4000000,
    assign_probe=8, refine=2, order=1, fast_assign=0,
    centroid_type=int8,       -- int8 | f16 | int4: storage of the centroids (format 3)
    lazy_cells=0,             -- G > 0: flat centroids stored in G cells fetched on demand
    threads=4, mem_mb=1024, seed=42, input=f32);

-- documents: one blob of n_tokens x dim float32 (or float16 with input=f16)
INSERT INTO t(rowid, vectors) VALUES (?1, ?2);
INSERT INTO t(t) VALUES ('build');                         -- from the inserted rows
INSERT INTO t(t) VALUES ('build_npy VECTORS.npy OFFSETS.npy'); -- or stream from files
INSERT INTO t(t) VALUES ('finalize');                      -- store page hints; rerun after VACUUM
SELECT rowid FROM t WHERE t MATCH 'warm';                  -- load the static data (and the document
                                                           -- table's interior pages) now; no rows

SELECT rowid, score, stats FROM t
 WHERE t MATCH ?1                 -- query: float32 blob [n_query_tokens x dim]
   AND k = 10 [AND nprobe = 4]
   [AND opts = 'layout=warp cprobe=4 impute=1 cross=1 rerank=0 approx=codes ndocs=256 tcs=0 exact=0 trace=0'];
```

`build_npy` reads a C-order float16 or float32 `[T, dim]` array and an int64 `[N+1]` offsets array with `fopen`/`fread`, so a 10 GB input is streamed, never loaded whole; documents get rowids 0 … N−1. The `'build'` path reads the rows inserted beforehand (stored as float16 in `t_buffer`, deleted after the build) and keeps arbitrary rowids through a rowid map.

Query options:

| Option | Default | Meaning |
|---|---|---|
| `k` | 10 | results (a `LIMIT` is also honoured) |
| `nprobe` | 4 | centroids probed per query token |
| `layout` | `warp` if built, else `plaid` | which layout answers the query |
| `approx` | `codes` | plaid layout: `codes` = fast-plaid (read every candidate's row and rank by centroid interaction); `ivf` = rank candidates from the IVF lists alone, read only the `ndocs` best rows |
| `ndocs` | 256 | plaid layout: documents decompressed and scored exactly (fast-plaid's `n_full_scores/4`) |
| `tcs` | 0 | next-plaid's centroid score threshold for probed cells |
| `stoplist` | 0 | query tokens whose probed lists hold on average more than `stoplist`·N entries are "stop tokens": their lists are not fetched and every document gets the token's best centroid score (0.02 recommended, see below) |
| `impute` | 1 | warp: missing (document, query token) pairs get the score of the token's last probed centroid (1) or 0 (0) |
| `cross` | 1 | warp: score every fetched token against every query token (1), or only against the tokens that probed its centroid (0) |
| `rerank` | 0 | warp: re-score the best `rerank` documents exactly from their plaid rows (needs `layout=both`, one more round) |
| `cprobe` | 4 | two-level: coarse cells examined per query token |
| `exact` | 0 | 1 = brute-force MaxSim over every decompressed document (ground truth for the codec) |
| `trace` | 0 | 1 = add the page numbers read in each round to `stats` |

`stats` (hidden column, same value on every row) is a JSON object with the per-query instrumentation: centroids probed, candidates, list entries and bytes, documents fetched, tokens decompressed, dependent rounds (and how many were for loading static data or cells), pages and bytes read, interior-page cache hits, hint misses, SQL fallbacks, time, and optionally the page trace. Other commands: `'drop_cache'` forgets cached interior pages and cells, `'reload'` also forgets the static data.

## Storage

All shadow tables are `(id INTEGER PRIMARY KEY, data BLOB)`.

| Table | Rows | Contents |
|---|---|---|
| `t_meta` | paged stream | static data: parameters, codec (cutoffs, weights), centroids (float16) or, in two-level mode, only the coarse centroids and cell boundaries; list lengths; page hints |
| `t_cells` | paged stream | two-level only: per cell, its fine centroids (float16) and their list lengths |
| `t_ivf` | paged stream | plaid: per centroid, the sorted distinct ids of documents with a token assigned to it, bit-packed at ⌈log₂ N⌉ bits |
| `t_post` | paged stream | warp: per centroid, the ids (⌈log₂ N⌉ bits each) then the packed residuals of every token assigned to it |
| `t_docs` | one per document | plaid: `u16 n_tokens`, centroid ids at ⌈log₂ K⌉ bits, packed residuals (`dim·nbits/8` bytes per token) |
| `t_rowids` | one per document | document number → user rowid, only when rowids are not 0 … N−1 |
| `t_buffer` | one per document | float16 input before `'build'` |

**Paged streams.** Lists are concatenated in centroid order into one byte stream, which is cut into rows of exactly `page_size − 39` bytes (4,057 bytes for 4 KiB pages). With a 4-byte record header the payload is then `usable − 35` bytes, the largest payload SQLite keeps entirely on the leaf page, so each row fills exactly one page and no list ever touches an overflow chain. A list of *n* bytes costs ⌈*n*/4057⌉ or one more pages, whatever its length, and its location follows from the list lengths, which are part of the static data. Fetching several lists is a batch of rowid lookups.

**Page hints.** A rowid lookup normally walks root → interior → leaf, one dependent read per level. `'finalize'` walks each stream's b-tree once and stores the leaf page of every row, run-length encoded as (first row, first page) pairs; because rows are appended in order, runs are long (a new run starts only where SQLite placed an interior page), so the hints for a 1.6 GB stream take a few kilobytes. At query time the leaves are read directly (`xRead` on the database file) and each is validated (table-leaf page, rowid present, payload local); a stale hint (for example after `VACUUM`) silently falls back to the level-by-level walk. `'finalize'` must run after anything that moves pages.

**Batched b-tree walks.** For rows without hints (document rows, the rowid map), `lpg_multiget` walks the tree breadth-first: all pages needed at one level are announced to the VFS together with `httpvfs_prefetch_pages()` (the file-control interface of `wasm/src/httpvfs.h`, a no-op on ordinary files) and then read. *n* lookups cost one round per level rather than *n* per level. Interior pages are cached in the extension (at most 16,384 pages), so in a warm session a batch of document rows costs one round (their leaves). The static data is loaded the same way by a full breadth-first scan: one round per level of `t_meta` (two rounds for a 1.7 MB table).

**Document rows** (plaid layout) are ordinary rows, 1.5 KB for a 111-token document at nbits = 2, so two fit on a 4 KiB page and the page fill is about 75 %.

## Build

1. **Sample.** fast-plaid's document sample: min(1 + ⌊16 √(120 N)⌋, N) documents in seeded random order, cut at `sample_tokens` (default 4 M tokens, 768 MB as float32) to bound memory. For N = 10,000 this is the whole corpus; for N = 1 M it would be 175,272 documents (19.5 M tokens; `docs/plaid.md` says 17,528, which is the N = 10,000 value), of which the first 4 M tokens are kept.
2. **Training points.** At most K · `kmeans_ppc` of the sample tokens, chosen at random (fast-plaid: 256 per centroid).
3. **k-means** (fast-plaid `FastKMeans`): K distinct random points as initial centroids, `kmeans_iters` Lloyd iterations with squared Euclidean distance (argmax of x·c − ‖c‖²/2), empty clusters re-seeded, then L2-normalised and rounded to float16. The assignment kernel stores centroids transposed in blocks of 64 so that the inner loop vectorises, and runs on `threads` threads.
4. **Two-level centroids** (`coarse=G`): k-means with G centroids on G · 256 training points; every training point is assigned to its coarse cell; cell *g* receives K_g ∝ (its share of the points) fine centroids (largest remainders, at least one, at most its points), trained by k-means on the cell's points only, cells in parallel. Fine ids are cell-major. Every vector is then assigned approximately: the `assign_probe` (8) best cells by coarse score, then the best fine centroid inside them.
   `refine` (2) global Lloyd iterations with this assignment follow, so that points near cell boundaries can move to a neighbouring cell's centroid.
5. **Ordering** (`order=1`). Centroids are renumbered so that similar centroids have adjacent ids: in two-level mode the cells follow a greedy nearest-neighbour chain over the coarse centroids and the fine centroids a chain inside each cell; in flat mode the same with √K groups from a k-means over the centroids. Lists probed by one query token then tend to lie next to each other in the streams, which saves pages and range requests. With `fast_assign=1` (flat mode) these √K groups are also used to assign vectors approximately (the `assign_probe` best groups, then the best centroid inside them), which makes encoding about five times faster.
6. **Codec** (fast-plaid `create.rs`): the held-out set is the last min(5 % of the sample tokens, 50,000) sample tokens; residuals against their nearest centroid; `bucket_cutoffs` = quantiles i/2ⁿ and `bucket_weights` = quantiles (i + ½)/2ⁿ of all residual components (NumPy "linear" interpolation); `cluster_threshold` = 0.75 quantile of residual norms (stored, unused).
7. **Encoding.** Code = argmax c·e (exact, or two-level approximate); bucket = number of cutoffs strictly below the residual component (`torch.bucketize`, right = False); bits of each bucket least-significant first, stream packed most-significant bit first (fast-plaid/NumPy `packbits` order). Decoding uses a 256-entry table from byte value to the weights of its 8/nbits dimensions and normalises centroid + weights, as fast-plaid does. `test/test_late.py` decodes the database in Python (`pyref.py`) with these rules and checks that brute-force MaxSim agrees with the C code.
8. **Writing.** Document rows are written while encoding. The IVF is built from the codes array (4 bytes per token in memory) with a "last document seen" array per centroid. Posting lists are written in centroid ranges that fit in `mem_mb`; the residuals are kept in memory if they fit, otherwise every pass re-reads the input and recomputes the residuals of its range (the 100 k test below forced three passes).

## Query algorithms

**Stage 1 (all layouts).** Scores S = C · Qᵀ for the candidate centroids (all K, or in two-level mode the fine centroids of the `cprobe` best cells of each query token, whose blocks are fetched in one round if not cached); for each query token its `nprobe` best centroids; their union P, optionally pruned by `tcs`.

**Layout plaid, `approx=codes` (fast-plaid).** Round 1 reads the IVF lists of P; the candidates are the union of their documents. Round 2 reads the row of every candidate (this is the expensive read), scores each by centroid interaction Σᵢ maxⱼ S[code_j, i], keeps the best `ndocs`, and scores those exactly by decompressing their tokens (the residuals arrived with the codes, so there is no third round).

**Layout plaid, `approx=ivf`.** Round 1 as above, but each candidate is ranked from the IVF lists alone: for query token *i*, the best S[c, i] over the probed centroids c whose list contains the document, with missing pairs imputed as in the warp layout. Round 2 reads only the rows of the best `ndocs` candidates and scores them exactly. This is the "IVF-only approximate scoring" of `docs/plaid.md`, recommendation 5.

**Layout warp.** One round reads the posting lists of P. Every entry is decompressed (centroid + bucket weights, normalised) and scored against the query tokens (all of them with `cross=1`, only those that probed its centroid with `cross=0`); for each (document, query token) the maximum is kept, together with a flag saying whether one of the token's own probed centroids contributed. The document score is Σᵢ sᵢ where sᵢ is the maximum found if the flag is set, and otherwise max(maximum found, mᵢ), with mᵢ the score of token *i*'s last probed centroid (`impute=1`, WARP's missing-similarity estimate) or 0 (`impute=0`). With `rerank=R` the R best documents are re-scored exactly from their plaid rows in a second round.

**Stop tokens** (`stoplist=f`, all layouts). A query token whose probed lists hold on average more than f·N entries is treated like a stop word: its lists are not fetched, and every document receives its best centroid score for that token (a constant, so the ranking is decided by the other tokens). This catches the `[SEP]`-like vectors that every document shares; see the results.

**Exact** (`exact=1`) scans all document rows (or all posting lists if only the warp layout exists) and returns exact MaxSim over the decompressed vectors.

## Results

### Method

- **Data.** words-10k: 10,000 random-word documents, 1,113,741 LateOn-Code-edge token vectors (111.4 per document), 2,000 queries (1,000 single-word, 1,000 three-word known-item; 7.7 query vectors on average). words-100: 100 documents, 11,086 vectors, 1,100 queries. Queries were encoded one at a time with `enc/lateon.py` (`encode_queries.py`).
- **Ground truth.** Exact MaxSim over the original float16 vectors, computed with NumPy ("float-exact" rows). "R@10 vs exact" is the overlap of the top 10 with float-exact's top 10. The label metrics (nDCG@10, MRR@10, recall@10, success@1, AUC over the collection) come from `bench/metrics.py`; results are requested at k = 100 so that AUC sees a deeper list.
- **Costs.** Every query records its page trace (pages per dependent round). "KB read" counts whole 4 KiB pages. The trace was checked against an APSW counting VFS (`countcheck.py`): per query the VFS sees exactly the traced pages plus page 1 (SQLite's schema-cookie check), and nothing else.
- **Cache state.** Unless marked *cold*, the numbers are for a warm session: the static data and the b-tree interior pages cached by the extension persist between queries (the posting lists and documents themselves are read afresh every query). The session start (loading the static data) is reported separately.
- **Network.** Each query's trace was replayed in `netsim/simulate.py` (adjacent pages of a round coalesced into one range request) under `4g` (165 ms, 8.1 Mbit/s), `lte` (70 ms, 12 Mbit/s) and `slow-4g` (562.5 ms, 1.44 Mbit/s), with HTTP/2-like concurrency (`h2`, 100 streams) and the HTTP/1.1 limit of six connections (`h1`); every 10th query (20th for secondary runs), p50 unless stated. CPU time is not added.
- **Timing caveat.** The machine (4 vCPUs) was shared with an LLM generation job, an encoding job and another agent's benchmarks (load average 12–19). Build times and native latencies are therefore pessimistic and noisy; relative comparisons within one run are more reliable than absolute numbers. Native latency is the wall time of one `SELECT` from Python (APSW), including result parsing.

### Index size (words-10k)

| Index | Build (s) | k-means (s) | DB | warp postings, B/token | plaid document rows, B/token | IVF, B/token | Static data | warp-only, B/doc | plaid-only, B/doc |
|---|---|---|---|---|---|---|---|---|---|
| nbits 1, K 16384 | 480 | 373 | 22.7 MB | 7.86 | 9.22 | 1.71 | 1.73 MB | 1048 | 1390 |
| nbits 2, K 16384 | 545 | 457 | 39.7 MB | 13.92 | 18.43 | 1.71 | 1.73 MB | 1724 | 2416 |
| nbits 4, K 16384 | 357 | 290 | 73.7 MB | 26.07 | 36.86 | 1.71 | 1.73 MB | 3077 | 4469 |
| nbits 2, K 4096 | 111 | 82 | 38.3 MB | 13.92 | 18.43 | 1.66 | 0.44 MB | 1595 | 2282 |
| nbits 2, K 8192 | 176 | 139 | 38.8 MB | 13.92 | 18.43 | 1.69 | 0.87 MB | 1638 | 2329 |
| nbits 2, K 16384, 32 points/centroid | 195 | 123 | 39.7 MB | 13.92 | 18.43 | 1.70 | 1.73 MB | 1724 | 2416 |
| nbits 2, K 16384, two-level G 128 | 24 | 6 | 39.7 MB | 13.92 | 18.43 | 1.71 | 20 KB + 1.72 MB cells | 1725 | 2418 |
| nbits 2, K 16384, two-level G 512 | 16 | 6 | 39.8 MB | 13.92 | 18.43 | 1.72 | 66 KB + 1.72 MB cells | 1730 | 2423 |

A posting entry costs ⌈log₂ N⌉ = 14 bits of document id plus the residual (6, 12 or 24 bytes), and the page overhead of the stream is 1 %: 7.9, 13.9 and 26.1 bytes per token, against 96 for float16 vectors. Document rows cost more than their payload (13.8 bytes per token at nbits = 2) because two 1.5 KB rows fill only 75 % of a 4 KiB page. The IVF is small (0.96 distinct (centroid, document) pairs per token at 14 bits each). A database with both layouts is the sum; a deployment would normally ship one layout, or the warp layout plus the document rows for re-ranking (3.2 KB per document at nbits = 2).

### Main results (words-10k, nbits = 2, K = 16,384, ordered centroids)

Quality:

| Configuration | R@10 vs exact | nDCG@10 | MRR@10 | Recall@10 | Success@1 | AUC | Rounds | KB read | 4g h2 (ms) |
|---|---|---|---|---|---|---|---|---|---|
| float-exact (reference) | 1.000 | 0.359 | 0.404 | 0.430 | 0.318 | 0.794 | – | – | – |
| exact decompressed (`exact=1`) | 0.553 | 0.357 | 0.395 | 0.424 | 0.318 | 0.804 | 1 | 20,000 | 20,392 |
| warp nprobe 2, stoplist 0.02 | 0.299 | 0.159 | 0.178 | 0.224 | 0.119 | 0.716 | 1 | 48 | 214 |
| warp nprobe 4, stoplist 0.02 | 0.346 | 0.209 | 0.235 | 0.277 | 0.172 | 0.751 | 1 | 87 | 250 |
| warp nprobe 8, stoplist 0.02 | 0.382 | 0.259 | 0.288 | 0.328 | 0.221 | 0.774 | 1 | 158 | 315 |
| warp nprobe 8, no stoplist | 0.382 | 0.259 | 0.288 | 0.329 | 0.219 | 0.775 | 1 | 236 | 392 |
| warp nprobe 16, stoplist 0.02 | 0.413 | 0.297 | 0.332 | 0.364 | 0.262 | 0.788 | 1 | 284 | 444 |
| warp nprobe 32, stoplist 0.02 | 0.459 | 0.329 | 0.365 | 0.397 | 0.289 | 0.797 | 1 | 510 | 673 |
| warp nprobe 16, `cross=0` | 0.392 | 0.290 | 0.321 | 0.359 | 0.253 | 0.787 | 1 | 284 | 444 |
| warp nprobe 16, `impute=0` | 0.274 | 0.290 | 0.317 | 0.355 | 0.250 | 0.767 | 1 | 284 | 444 |
| warp nprobe 8 + rerank 32 | 0.488 | 0.339 | 0.380 | 0.387 | 0.311 | 0.719 | 2 | 286 | 611 |
| warp nprobe 8 + rerank 64 | 0.512 | 0.350 | 0.388 | 0.407 | 0.316 | 0.754 | 2 | 413 | 739 |
| warp nprobe 16 + rerank 64 | 0.531 | 0.356 | 0.394 | 0.421 | 0.318 | 0.770 | 2 | 540 | 866 |
| plaid-ivf nprobe 4, ndocs 64 | 0.343 | 0.302 | 0.332 | 0.347 | 0.278 | 0.701 | 2 | 314 | 646 |
| plaid-ivf nprobe 8, ndocs 64 | 0.372 | 0.327 | 0.359 | 0.381 | 0.297 | 0.726 | 2 | 351 | 682 |
| plaid-ivf nprobe 8, ndocs 128 | 0.438 | 0.343 | 0.377 | 0.402 | 0.311 | 0.766 | 2 | 604 | 1094 |
| plaid-ivf nprobe 16, ndocs 128 | 0.458 | 0.348 | 0.383 | 0.407 | 0.315 | 0.773 | 2 | 664 | 1154 |
| plaid-codes nprobe 1, ndocs 256 | 0.427 | 0.325 | 0.354 | 0.379 | 0.293 | 0.748 | 2 | 4,103 | 4,773 |
| plaid-codes nprobe 2, ndocs 256 | 0.491 | 0.352 | 0.384 | 0.416 | 0.312 | 0.787 | 2 | 7,018 | 7,489 |
| plaid-codes nprobe 8, ndocs 1024 (fast-plaid defaults) | 0.547 | 0.357 | 0.394 | 0.424 | 0.318 | 0.803 | 2 | 16,803 | 17,436 |

("plaid-ivf" is `layout=plaid approx=ivf stoplist=0.02`; "plaid-codes" is the faithful `approx=codes`.)

By query kind (warp, stoplist 0.02): known-item queries reach nDCG@10 0.331 / 0.423 / 0.488 / 0.545 at nprobe 4 / 8 / 16 / 32 against 0.599 for float-exact; single-word queries 0.088 / 0.096 / 0.105 / 0.113 against 0.119. The single-word queries are nearly ties for this code-search model (7 relevant documents that differ from the others by one token), which is also why R@10 against exact stays low even for exact decompressed search.

Network and cost:

| Configuration | Rounds | KB read (p50 / p90) | List payload KB | Requests | Candidates | Native ms | 4g h2 p50 / p90 (ms) | lte h2 | slow-4g h2 | 4g h1 |
|---|---|---|---|---|---|---|---|---|---|---|
| warp nprobe 4 | 1 | 84 / 132 | 14.6 | 16 | 988 | 3.4 | 250 / 303 | 127 | 1,040 | 563 |
| warp nprobe 8 | 1 | 152 / 240 | 28.7 | 27 | 1,844 | 3.9 | 315 / 408 | 171 | 1,404 | 923 |
| warp nprobe 16 | 1 | 268 / 436 | 56.9 | 46 | 3,312 | 4.7 | 444 / 606 | 258 | 2,133 | 1,455 |
| warp nprobe 32 | 1 | 492 / 776 | 114.3 | 74 | 5,530 | 6.7 | 673 / 942 | 413 | 3,418 | 2,267 |
| warp nprobe 8 + rerank 64 | 2 | 404 / 496 | 28.7 + 64 docs | 90 | 1,844 | 6.9 | 739 / 828 | 416 | 3,423 | 2,997 |
| plaid-ivf nprobe 8, ndocs 64 | 2 | 346 / 404 | 3.6 + 64 docs | 79 | 1,844 | 5.9 | 682 / 739 | 378 | 3,105 | 2,645 |
| plaid-ivf nprobe 8, ndocs 128 | 2 | 600 / 656 | 3.6 + 128 docs | 140 | 1,844 | 7.3 | 1,094 / 1,154 | 614 | 5,055 | 4,540 |
| plaid-codes nprobe 2, ndocs 256 | 2 | 7,190 / 9,730 | 1,951 docs | 1,096 | 1,951 | 17.3 | 7,499 / 10,258 | 4,979 | 41,448 | 33,286 |
| plaid-codes nprobe 8, ndocs 1024 (fast-plaid) | 2 | 18,216 / 18,888 | 6,135 docs | 674 | 6,135 | 50.4 | 17,436 / 19,441 | 11,687 | 97,347 | 25,534 |
| exact decompressed | 1 | 20,000 | all rows | 11 | 10,000 | 251 | 20,392 | 13,723 | 114,340 | 20,392 |

Session start (once per connection): the flat K = 16,384 static data is 1.70 MB in 422 contiguous pages, read in 2 rounds (b-tree root, then all leaves as one range): 2.0 s on 4g, 1.3 s on lte, 10.7 s on slow-4g. With K = 4,096 it is 0.43 MB (0.76 s on 4g); in two-level mode 15–60 KB (0.35–0.40 s on 4g), after which each query pays for the cells it touches.

### Stop tokens

About 75 % of the posting entries a query fetched at 10k came from the query's `[SEP]` vector: document `[SEP]` vectors are all alike, k-means gives them centroids with lists of thousands of entries, and `[SEP]`'s probed lists held 3,377 entries on average, against 176 for a word-piece token (`[CLS]` 326, `[Q]` 104). Such a token contributes an almost constant amount to every document's score. `stoplist=f` detects these tokens from the list lengths alone (mean probed-list length above f·N; the extension never sees token ids) and scores them with their best centroid score for every document instead of fetching their lists. With f = 0.02 only `[SEP]` is caught at 10k; the ranking is unchanged in every configuration tested (identical nDCG and R@10 to three decimals), while candidates fall by 70 % and bytes by 33 % (warp nprobe 8: 236 → 158 KB, 403 → 315 ms on 4g). The threshold must be per list, not per token: a first version that summed a token's lists wrongly stopped word-piece tokens at nprobe 32 and lost half the quality. At 1M documents the `[SEP]` lists grow in proportion to N, so this is not an optimisation but a requirement.

### Centroid ordering and page size

Renumbering centroids by similarity (`order=1`) changes no result but puts the lists a token probes next to each other: warp nprobe 8 (no stoplist) read 301 KB before and 236 KB after (−22 %), mean 4g time 469 → 403 ms. Even so, pages dominate at this scale: warp nprobe 8 needs 28.7 KB of list payload but reads 152 KB of pages, because a probed list averages 650 bytes and costs one or two 4 KiB pages. With 1 KiB database pages (same index otherwise) warp nprobe 8 reads 59 KB instead of 148 KB and takes 226 instead of 315 ms on 4g h2, but it needs more separate requests, so under HTTP/1.1's six connections it is slower (1,018 vs 927 ms), and the static data then needs 3 rounds. The plaid document rows (1.5 KB) do not fit a 1 KiB page and spill to overflow pages, which the direct reader hands to SQL (65 extra dependent page reads per query in the counting VFS), so small pages suit the warp layout only.

### Residual bits and number of centroids

nDCG@10 (R@10 vs exact in parentheses), all 2,000 queries, no stoplist, unordered builds:

| Index | warp 4 | warp 8 | warp 16 | warp 32 | warp 8 + rerank 64 | plaid-ivf 4/64 | plaid-ivf 8/128 | plaid-codes 2/256 |
|---|---|---|---|---|---|---|---|---|
| nbits 1, K 16384 | 0.211 (0.30) | 0.257 (0.32) | 0.291 (0.35) | 0.318 (0.37) | 0.328 (0.39) | 0.293 (0.30) | 0.324 (0.36) | 0.332 (0.39) |
| nbits 2, K 16384 | 0.209 (0.35) | 0.259 (0.38) | 0.297 (0.41) | 0.329 (0.46) | 0.350 (0.51) | 0.302 (0.34) | 0.343 (0.44) | 0.352 (0.49) |
| nbits 4, K 16384 | 0.205 (0.37) | 0.256 (0.43) | 0.299 (0.47) | 0.331 (0.54) | 0.353 (0.62) | 0.303 (0.36) | 0.349 (0.49) | 0.358 (0.59) |
| nbits 2, K 4096 | 0.227 (0.35) | 0.266 (0.39) | 0.293 (0.43) | 0.307 (0.46) | 0.327 (0.47) | 0.230 (0.22) | 0.276 (0.31) | 0.310 (0.41) |
| nbits 2, K 8192 | 0.224 (0.35) | 0.266 (0.39) | 0.306 (0.42) | 0.323 (0.46) | 0.334 (0.49) | 0.268 (0.28) | 0.310 (0.38) | 0.333 (0.46) |
| nbits 2, K 16384, 32 pts/centroid | 0.198 (0.34) | 0.244 (0.37) | 0.281 (0.41) | 0.316 (0.45) | 0.347 (0.50) | 0.283 (0.33) | 0.333 (0.43) | 0.343 (0.48) |
| same, `fast_assign=1` | – | 0.236 (0.37) | 0.272 (0.40) | – | 0.336 (0.49) | – | 0.320 (0.41) | – |

KB read / candidates for the same cells:

| Index | warp 4 | warp 8 | warp 16 | warp 32 | plaid-codes 2/256 |
|---|---|---|---|---|---|
| nbits 1, K 16384 | 134 / 3,974 | 252 / 6,142 | 457 / 7,988 | 823 / 9,727 | 5,741 / 1,954 |
| nbits 2, K 16384 | 163 / 3,974 | 301 / 6,142 | 533 / 7,988 | 950 / 9,727 | 7,044 / 1,954 |
| nbits 4, K 16384 | 210 / 3,974 | 384 / 6,142 | 664 / 7,988 | 1,168 / 9,727 | 7,874 / 1,954 |
| nbits 2, K 4096 | 273 / 8,955 | 426 / 9,483 | 724 / 9,905 | 1,237 / 10,000 | 19,540 / 8,379 |
| nbits 2, K 8192 | 163 / 3,792 | 320 / 6,583 | 604 / 9,318 | 1,054 / 9,962 | 8,417 / 2,421 |
| nbits 2, K 16384, 32 pts/centroid | 138 / 2,334 | 268 / 3,955 | 515 / 6,642 | 946 / 8,660 | 5,541 / 1,502 |

- **Residual bits.** The first stage (warp alone, plaid-ivf ranking) is insensitive to nbits: its errors come from which lists are probed, not from residual precision. Exact scoring of a short list gains a little from more bits (warp 8 + rerank 64: 0.328 / 0.350 / 0.353 for 1 / 2 / 4 bits), and R@10 against float-exact rises steadily (0.39 / 0.51 / 0.62). nbits = 2 is the balance point: nbits = 1 saves 43 % of the postings (7.9 vs 13.9 B/token) for −0.02 nDCG after re-ranking; nbits = 4 doubles the size for +0.003.
- **Number of centroids.** Fewer centroids give longer lists, so at equal nprobe the warp layout reads more and scores slightly better (K 4096 vs 16384 at nprobe 4: 0.227 vs 0.209 for 1.7× the bytes); at equal bytes the difference disappears. The IVF-based modes prefer more centroids (plaid-ivf 8/128: 0.276 / 0.310 / 0.343 for K 4096 / 8192 / 16384), because the candidate ranking is by centroid scores. The static data grows with K (104 bytes per centroid).
- **k-means budget.** Training on 32 instead of 68 points per centroid (524 k instead of 1.1 M) cut k-means time 3.7× and cost 0.003–0.019 nDCG; it also produced shorter probed lists (fewer candidates for the same nprobe). `fast_assign` (approximate assignment through √K centroid groups, `assign_probe=8`) made encoding 5× faster (14 s instead of 72 s) for a further 0.008–0.013 nDCG.
- **Two-level centroids** (coarse = 128 or 512 cells, K = 16,384) build 20–30× faster and shrink the static data to 15–60 KB, but cost 0.05–0.09 nDCG in warp mode at equal nprobe (nprobe 8 / cprobe 8: 0.210 at G = 128 and 0.175 at G = 512, cold, against 0.259 flat), and each cold query spends a second round and 100–400 KB fetching cell blocks. The per-cell centroids fit query vectors worse (mean best centroid score 0.635–0.644 against 0.654 flat) and attract more of the `[SEP]` mass (probed lists 1.9× longer). Global refinement iterations (`refine`) improve the fit but lengthen the probed lists further. This mode needs more work before it is useful; it is kept because at 1M documents the flat centroid table becomes the largest single read (below).

Two-level results (cold: cells dropped before every query):

| Configuration | nDCG@10 | R@10 vs exact | Rounds | KB read | 4g h2 (ms) |
|---|---|---|---|---|---|
| G 128, warp nprobe 8, cprobe 4 | 0.171 | 0.305 | 2 | 622 | 941 |
| G 128, warp nprobe 8, cprobe 8 | 0.210 | 0.338 | 2 | 911 | 1,224 |
| G 128, warp nprobe 16, cprobe 8 | 0.238 | 0.367 | 2 | 1,031 | 1,343 |
| G 512, warp nprobe 8, cprobe 8 | 0.175 | 0.322 | 2 | 426 | 747 |
| G 512, warp nprobe 16, cprobe 8 | 0.198 | 0.348 | 2 | 538 | 842 |

### Cold interior pages

Without the extension's interior-page cache (dropped before each query), the plaid layout pays two more rounds for the document table's b-tree (root and interior level) before its leaves: plaid-ivf nprobe 4/64 (no stoplist) takes 4 rounds and 1,075 ms on 4g instead of 2 rounds and 698 ms warm. The warp layout is unaffected because its lists are read through page hints (1 round, 444 ms at nprobe 8 without stoplist either way). The interior pages of the document table are 0.25 % of it (about 50 KB here), so a session fetches them once.

### words-100

| Configuration | nDCG@10 | MRR@10 | R@10 vs exact | Rounds | KB read | 4g h2 (ms) | slow-4g h2 (ms) |
|---|---|---|---|---|---|---|---|
| float-exact | 0.454 | 0.400 | 1 | – | – | – | – |
| exact decompressed | 0.430 | 0.375 | 0.697 | 1 | 200 | 367 | 1,700 |
| warp nprobe 4 | 0.350 | 0.308 | 0.504 | 1 | 61 | 226 | 904 |
| warp nprobe 8 | 0.372 | 0.324 | 0.582 | 1 | 91 | 254 | 1,063 |
| warp nprobe 4 + rerank 16 | 0.372 | 0.335 | 0.584 | 2 | 119 | 447 | 1,785 |
| plaid-codes nprobe 4, ndocs 32 | 0.385 | 0.342 | 0.634 | 2 | 208 | 540 | 2,308 |

With 100 documents the whole index (200 KB of document rows, 107 KB of static data at K = 1,024) is smaller than one slow round trip's worth of transfer, and brute force over all documents in one round is the best choice.

### Rounds and bytes

| Layout / mode | Dependent rounds per query (warm) | What is read |
|---|---|---|
| warp | 1 | posting lists of the probed centroids (page hints, no b-tree walk) |
| warp + rerank R | 2 | + R document rows |
| plaid, approx=ivf | 2 | IVF lists; `ndocs` document rows |
| plaid, approx=codes (fast-plaid) | 2 | IVF lists; every candidate's document row (codes and residuals together, so fast-plaid's third stage needs no extra round) |
| two-level, cold cells | +1 | cell blocks (fine centroids) of the probed cells |
| plaid document table, cold interior pages | +2 | root and interior pages of `t_docs` |
| non-identity rowids | +1 | rowid map leaves for the top k |
| session start | 2 (+ schema) | static data: one b-tree level per round |

The faithful PLAID pipeline is round-efficient (2 rounds) but reads every candidate's row: 4–17 MB per query at 10k, 5–17 s on a 4G phone. Ranking candidates from the IVF alone (`approx=ivf`) keeps 2 rounds and reads 0.35–0.66 MB for nDCG 0.33–0.35, close to float-exact's 0.359. The centroid-major warp layout needs only 1 round, reads 0.08–0.5 MB and answers in 0.25–0.67 s on 4g, but ranks worse by itself (0.21–0.33); one more round to re-rank its best 32–64 documents exactly brings it to 0.34–0.36 in 0.6–0.9 s. On a 4G link (165 ms per round trip, 8.1 Mbit/s) one round costs about as much as 170 KB of transfer, so the second round is worth taking whenever it buys quality.

### Static data: precision, number of centroids, lazy cells (2026-09-24)

The static data (centroids, list lengths, page hints) is what a new connection reads before its first query; with float16 centroids it was 104 bytes per centroid, 1.7 MB at K = 16,384 (words-10k) and 0.83 MB at K = 8,192 (llm-10k, where the fast-plaid formula gives 8,192). Three ways to shrink it were measured with `bench/static_eval.py` (one late index per database, `layout=both nbits=2 order=1`, all 2,000 queries of each corpus for quality; 40 cold queries through the WASM build, each on a new connection, replayed through netsim, p50):

- **Centroid precision** (`centroid_type=f16|int8|int4`, format 3). Each centroid is stored with a float16 scale and 48 int8 (50 B) or int4 (26 B) values. The build rounds the centroids to their stored values *before* the codec is trained and the residuals are computed, so decompression is exact with respect to what the query sees. Format 3 also stores the list lengths as varints (about 2.6 instead of 8 bytes per centroid).
- **Fewer centroids** (K/2, K/4), all int8.
- **Lazy cells** (`lazy_cells=G`, new): the same flat k-means centroids, but stored in G cells of similar centroids (the ordering groups); the static data keeps only the G cell centroids (14 KB), and a query fetches the cells its tokens probe (`cprobe` best cells per query token) in one extra round. Unlike `coarse=G`, the index is identical to the flat one, so with enough cells probed the results are identical too. It needs every centroid for re-ranking or the plaid layout, so it suits `layout=warp` without `rerank`.

| Corpus | Variant | Static KB | warp 8: nDCG@10 | R@10 vs exact | warp 8 + rerank 64: nDCG@10 | R@10 vs exact | Cold KB (warp 8 + rr 64) | Cold 4g h1 / h2 ms | Cold lte h1 / h2 ms |
|---|---|---|---|---|---|---|---|---|---|
| llm-10k | f16, format 2 (before) | 832 | 0.444 | 0.389 | 0.443 | 0.462 | 1,226 | 4,332 / 2,400 | 2,140 / 1,332 |
| llm-10k | f16, format 3 | 786 | 0.444 | 0.389 | 0.443 | 0.462 | 1,178 | 4,284 / 2,353 | 2,108 / 1,301 |
| llm-10k | **int8 (new default)** | **418** | 0.444 | 0.389 | 0.442 | 0.462 | 808 | 3,909 / 1,977 | 1,854 / 1,046 |
| llm-10k | int4 | 226 | 0.453 | 0.406 | 0.458 | 0.490 | 610 | 3,697 / 1,776 | 1,711 / 910 |
| llm-10k | int8, K = 4,096 | 212 | 0.429 | 0.393 | 0.427 | 0.444 | 614 | 3,707 / 1,781 | 1,718 / 914 |
| llm-10k | int8, K = 2,048 | 107 | 0.421 | 0.331 | 0.418 | 0.405 | 474 | 3,410 / 1,641 | 1,558 / 820 |
| words-10k | f16, format 2 (before) | 1,664 | 0.313 | 0.413 | 0.409 | 0.539 | 2,154 | 5,811 / 3,342 | 2,994 / 1,969 |
| words-10k | f16, format 3 | 1,572 | 0.313 | 0.413 | 0.409 | 0.539 | 2,058 | 5,715 / 3,245 | 2,929 / 1,904 |
| words-10k | **int8 (new default)** | **836** | 0.313 | 0.412 | 0.408 | 0.537 | 1,304 | 4,921 / 2,481 | 2,414 / 1,388 |
| words-10k | int4 | 451 | 0.299 | 0.413 | 0.401 | 0.543 | 914 | 4,513 / 2,088 | 2,139 / 1,122 |
| words-10k | int8, K = 8,192 | 422 | 0.334 | 0.418 | 0.400 | 0.516 | 914 | 4,494 / 2,086 | 2,126 / 1,121 |
| words-10k | int8, K = 4,096 | 215 | 0.268 | 0.361 | 0.355 | 0.455 | 672 | 4,051 / 1,841 | 1,867 / 955 |

Lazy cells (128 cells, int8, `layout=warp`, warp nprobe 8, no rerank; cold = static directory + the probed cells + postings):

| Corpus | cprobe | nDCG@10 | R@10 vs exact | Cold rounds | Cold KB | Cold 4g h1 / h2 ms | Flat int8, same query: cold KB, 4g h1 / h2 ms |
|---|---|---|---|---|---|---|---|
| llm-10k | 4 | 0.376 | 0.327 | 5 | 192 | 1,241 / 1,030 | 532, 1,509 / 1,204 |
| llm-10k | 8 | 0.411 | 0.363 | 5 | 268 | 1,513 / 1,101 | |
| llm-10k | 16 | 0.444 | 0.389 | 5 | 354 | 1,547 / 1,186 | (flat: 0.444, 0.389) |
| words-10k | 4 | 0.291 | 0.392 | 5 | 344 | 1,750 / 1,192 | 1,000, 2,314 / 1,694 |
| words-10k | 8 | 0.310 | 0.410 | 5 | 506 | 2,073 / 1,370 | |
| words-10k | 16 | 0.313 | 0.412 | 5 | 702 | 2,232 / 1,552 | (flat: 0.313, 0.412) |

What this shows:

- **int8 centroids are free**: every configuration is within ±0.002 nDCG and ±0.002 recall of float16 on both corpora (plaid-ivf and warp nprobe 32 too; `results/static/`), for half the static data. Measured against the float16 format-2 index, a cold first query is about 420 ms (llm-10k) and 860 ms (words-10k) faster on `4g` with HTTP/2-like concurrency, and 290 / 580 ms faster on `lte`. This is the new default (`centroid_type=int8`, also in `bench/matrix_config.json`).
- **int4 is not at matched quality**: it *gains* 0.01–0.015 nDCG on llm-10k but loses 0.007–0.014 on words-10k. It is kept as an option.
- **Fewer centroids cost quality** with the default re-ranked configuration (−0.015 nDCG at K/2 on llm-10k, −0.008 on words-10k; the IVF-ranked plaid mode loses 0.04), although on words-10k K = 8,192 improves the warp first stage alone (0.334 against 0.313), as earlier: fewer centroids mean longer lists, i.e. more candidates per probe. At equal static bytes int8 at full K is better than float16 at K/2.
- **Lazy cells recover flat quality at cprobe 16** (identical nDCG and recall), whereas the trained two-level mode lost 0.05–0.09: the loss came from the per-cell k-means, not from routing through cells. Cold, they save 0.2–0.3 MB at K = 16,384 (140 ms on `4g` h2 at cprobe 16, 320 ms at cprobe 8 for −0.003 nDCG) and nothing measurable at K = 8,192, because the extra round costs about as much as the bytes saved. Every later query also pays that round until its cells are cached, and re-ranking needs all cells. With the static data loaded in the background (`MATCH 'warm'`, below), loading everything is better at 10k; lazy cells are for indexes whose flat table is too large to load up front (K = 65,536 at 1M: 3.4 MB int8).
- **Format 3's varint list lengths** save 6% (46 KB at llm-10k, 92 KB at words-10k).

**`MATCH 'warm'`.** `SELECT rowid FROM t WHERE t MATCH 'warm'` loads the static data now and also the interior pages of the document table (read level by level, stopping above the leaf level, whose size is estimated from N and the row size), which the first re-ranking query would otherwise walk in two extra rounds. It returns no rows. The browser client runs it while the query encoder loads (`web/NOTES.md`).

**Format and compatibility.** Format 3 adds the centroid type after the version-2 header and stores the flat table's list lengths as varints; in two-level and lazy modes the centroid type also applies to the cell blocks. This build reads formats 1–3; a newer format is refused with "index format version N is not supported by this build". Older builds cannot read format 3 (they report "cannot load index"); `finalize` rewrites any index it is run on in format 3.

**Option parsing (fixed).** `tools/build_db.py` passes the late index parameters as one space-separated argument (`dim=48 nbits=2 …`), which the parser used to read as `dim` with a long value: every other option, `threads` included, was silently ignored and the defaults were used. The earlier matrix databases therefore had the default parameters (for words-1m: K from the formula, both layouts, no `fast_assign`); at 10k the defaults happened to equal the intended parameters. Options may now be separated by commas or white space.

## Scaling to 1M documents

The 1M LateOn embeddings do not exist yet, so this section is a projection from the 10k measurements and a 100k test.

**Streaming build test (100k).** words-10k tiled ten times (100,000 documents, 11.1 M vectors, 1.07 GB float16 `.npy`), `centroids=32768 coarse=256 nbits=2 layout=both mem_mb=64`: 309 s in total (k-means on 6.2 M sample tokens 43 s, encoding 250 s, IVF 1 s, postings in 3 passes that each re-read the input 15 s), peak RSS 2.4 GB (before the `sample_tokens` cap and the per-cell gathering that now bound the k-means memory), database 391 MB (postings 159 MB = 14.3 B/token, document rows 205 MB, IVF 23 MB). Duplicated documents received bit-identical scores in the warp layout, and the plaid and exact modes agreed, which checks the multi-pass posting writer.

**Choice of K.** fast-plaid's formula gives K = 2¹⁷ = 131,072 for 111.4 M vectors. That is not affordable here for three reasons. (1) *Building*: exact assignment costs T·K·48 multiply-adds, 7·10¹⁴ at K = 2¹⁷; this machine sustained about 10¹⁰ per second on 4 loaded cores, i.e. about 20 hours for encoding alone (k-means on K·256 = 33 M points would add days). (2) *Session start*: the flat centroid table (with list lengths) is 104 bytes per centroid, 13.6 MB at 2¹⁷, 14 s on 4g before the first query. (3) *Quality at 10k did not depend strongly on K*: 4,096–16,384 centroids gave the same warp quality at equal bytes, and the IVF modes preferred more centroids only mildly. The warp layout's per-query bytes scale as T/K while the static data scales as K, so the total over a session of Q queries, K·104 + Q·(lists probed)·(T/K)·14.5 bytes, is minimised near K = √(Q · 1.2·10⁹): 35 k for one query per session, 110 k for ten. The recommendation for 1M is therefore **K = 65,536** (6.8 MB static data, 1,700 vectors per list), trained on the `sample_tokens` = 4 M sample tokens (61 per centroid, which the ppc = 32 experiment shows costs about 0.01 nDCG) with 4 iterations, and encoded with `fast_assign=1` (256 groups, `assign_probe=8`, about 2,300 dot products per vector instead of 65,536; −0.01 nDCG at 10k). If the static data must be small, the two-level mode (`coarse=1024`, 110 KB static) is the fallback, at the quality cost measured above.

**Projected build time** (this loaded machine): k-means with exact assignment on 4 M points, 4 iterations, 5·10¹³ multiply-adds, about 1.5 h; encoding 111 M vectors with `fast_assign`, extrapolated from 12.6 µs per vector at 10k (3 threads, 1,152 dot products per vector with 128 groups) to about 25 µs (256 groups of 256, 2,300 dot products), about 45–60 min (the 100k two-level test ran at 22 µs per vector); postings in two passes at `mem_mb=1024`, each re-reading the 10.7 GB input, about 10 min. Roughly 3 hours in total, dominated by k-means; `sample_tokens=2000000` halves the k-means part. Peak memory: 445 MB of centroid codes (4 bytes per vector), 768 MB of sample, 267 MB of IVF, `mem_mb` of posting buffer, about 2.5 GB. Disk: the input (10.7 GB) plus the database.

**Projected sizes** (N = 10⁶, T = 111.4 M, 20-bit document ids, K = 65,536):

| nbits | warp postings | plaid document rows | IVF | Static | warp only | both layouts |
|---|---|---|---|---|---|---|
| 1 | 0.96 GB (8.6 B/token) | 1.02 GB (4 rows/page) | 0.27 GB | 6.8 MB | 0.97 GB | 2.25 GB |
| 2 | 1.63 GB (14.6 B/token) | 2.05 GB (2 rows/page) | 0.27 GB | 6.8 MB | 1.64 GB | 3.95 GB |
| 4 | 2.98 GB (26.8 B/token) | 4.10 GB (1 row/page) | 0.27 GB | 6.8 MB | 2.99 GB | 7.35 GB |

**Projected query cost** (warp, K = 65,536, stoplist 0.02): at 10k a word-piece token's probed list held about 44 vectors against a mean of 68; scaling the mean to 1,700 gives about 1,100 vectors (16 KB) per probed list, so nprobe 8 over about five non-stop query vectors reads roughly 40 lists, 0.65 MB, in one round: about 0.8 s on 4g, 0.6 s on lte and 4 s on slow-4g. Re-ranking 64 documents adds one round and about 100 KB. How recall behaves with 100 times more documents per list cannot be extrapolated from 10k and has to be measured once the embeddings exist.

## Open questions

1. **Quality ceiling of the first stage.** Warp alone reaches 0.26–0.33 nDCG@10 against 0.359 for exact search; plaid-ivf and warp + re-ranking come within 0.01–0.02. The missing-score imputation is the weak point: `impute=0` scores better at small nprobe (0.242 vs 0.209 at nprobe 4, 0.277 vs 0.259 at nprobe 8) and worse at large (0.290 vs 0.297 at nprobe 16), and always lowers R@10 against exact. A better estimate (per-token score distributions, or WARP's cumulative-size threshold t′) should be tried.
2. **Two-level centroids** lose 0.05–0.09 nDCG at 10k. Candidates: balancing the special-token mass across cells, more coarse probes during assignment, k-means++ seeding inside cells, or using the cells only to *fetch* centroid blocks while keeping flat k-means centroids.
3. **Sub-page reads.** At 10k the posting payload is 5× smaller than the pages read. Exact byte ranges within pages (the offset of a row inside its page follows from the cell layout, and `HTTPVFS_FCNTL_PREFETCH` takes byte ranges) would help if the VFS block size allows it; at 1M lists span several pages and this matters less.
4. **HTTP/1.1.** One warp round is 16–74 range requests; under six connections that is 3–12 latency steps (0.9 s instead of 0.3 s at nprobe 8 on 4g). Multi-range requests (`Range: bytes=a-b,c-d`) or HTTP/2 remove this; netsim does not model multipart responses.
5. **Document rows** fill 75 % of a page and need the b-tree interior pages (no hints: a per-document directory would cost 4 bytes per document of static data). Grouping documents into page-sized rows, or ordering documents by content so that one query's candidates share pages, are untested.
6. **The model.** LateOn-Code-edge is a code-search model; on single random words it is barely better than chance even with exact search (nDCG 0.12), which compresses all differences between methods. The LLM-paragraph corpus should be evaluated when it is complete.
7. **Browser CPU.** Query-side work is small (decompressing 1–9 k vectors, 3–8 ms natively including Python overhead), but query encoding with the ONNX model is not included and will dominate on a phone.
8. **next-plaid's lookup-table scoring** (no decompression, per-token inverse norms) is not implemented; decompression is cheap at these candidate counts.

## Reproducing

```sh
make -C ext/late && make -C ext/late test
python3 ext/late/encode_queries.py words-100 words-10k            # build/late/queries-*.npz
python3 ext/late/build_index.py --data words-10k --nbits 2 --centroids 16384 \
    --out build/late/w10k-n2-k16384-ord.db                         # ~8 min here
python3 ext/late/bench.py --db build/late/w10k-n2-k16384-ord.db --data words-10k \
    --sim-every 10 --tag mytag --configs "layout=warp nprobe=8 stoplist=0.02 rerank=64"
python3 ext/late/report.py mytag [--net]
python3 ext/late/countcheck.py --db build/late/w10k-n2-k16384-ord.db --configs "layout=warp nprobe=8"
```

`bench.py` caches the float ground truth in `build/late/gt-*.npz`; result files are in `results/` (the tags used above: `w10k-n2-k16384-ord`, `w10k-n2-k16384-ord-stop`, `w10k-n{1,2,4}-k16384`, `w10k-n2-k{4096,8192}`, `w10k-n2-k16384-ppc32`, `w10k-n2-k16384-ppc32-fa`, `*-pg`, `*-g128-ord-cold`, `*-g512-ord-cold`, `w10k-n2-k16384-cold`, `w100-n2-k1024`).
