# late_plaid: PLAID-style late-interaction search for httpvfs

`late_plaid` is a SQLite virtual table for late-interaction (ColBERT-style, MaxSim) retrieval. It follows the PLAID index of [fast-plaid](https://github.com/lightonai/fast-plaid) (centroids, a residual codec with 1, 2 or 4 bits per dimension, an inverted file) and adds a centroid-major layout in the spirit of WARP. It is written for read-only databases that a browser reads lazily over HTTP range requests ("httpvfs"), where the number of *dependent* round trips and the bytes read matter far more than CPU time. The specification it follows is `docs/plaid.md`.

The extension is plain C11 with SQLite and libm only. Index building may use POSIX threads (`-DLATE_THREADS`); the query path is single-threaded. It compiles unchanged with Emscripten (`emcc -r`, checked) and as a loadable extension.

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
    assign_probe=8, order=1, threads=4, mem_mb=1024, seed=42, input=f32);

-- documents: one blob of n_tokens x dim float32 (or float16 with input=f16)
INSERT INTO t(rowid, vectors) VALUES (?1, ?2);
INSERT INTO t(t) VALUES ('build');                         -- from the inserted rows
INSERT INTO t(t) VALUES ('build_npy VECTORS.npy OFFSETS.npy'); -- or stream from files
INSERT INTO t(t) VALUES ('finalize');                      -- store page hints; rerun after VACUUM

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
5. **Ordering** (`order=1`). Centroids are renumbered so that similar centroids have adjacent ids: in two-level mode the cells follow a greedy nearest-neighbour chain over the coarse centroids and the fine centroids a chain inside each cell; in flat mode the same with √K groups from a k-means over the centroids. Lists probed by one query token then tend to lie next to each other in the streams, which saves pages and range requests.
6. **Codec** (fast-plaid `create.rs`): the held-out set is the last min(5 % of the sample tokens, 50,000) sample tokens; residuals against their nearest centroid; `bucket_cutoffs` = quantiles i/2ⁿ and `bucket_weights` = quantiles (i + ½)/2ⁿ of all residual components (NumPy "linear" interpolation); `cluster_threshold` = 0.75 quantile of residual norms (stored, unused).
7. **Encoding.** Code = argmax c·e (exact, or two-level approximate); bucket = number of cutoffs strictly below the residual component (`torch.bucketize`, right = False); bits of each bucket least-significant first, stream packed most-significant bit first (fast-plaid/NumPy `packbits` order). Decoding uses a 256-entry table from byte value to the weights of its 8/nbits dimensions and normalises centroid + weights, as fast-plaid does. `test/test_late.py` decodes the database in Python (`pyref.py`) with these rules and checks that brute-force MaxSim agrees with the C code.
8. **Writing.** Document rows are written while encoding. The IVF is built from the codes array (4 bytes per token in memory) with a "last document seen" array per centroid. Posting lists are written in centroid ranges that fit in `mem_mb`; the residuals are kept in memory if they fit, otherwise every pass re-reads the input and recomputes the residuals of its range (the 100 k test below forced three passes).

## Query algorithms

**Stage 1 (all layouts).** Scores S = C · Qᵀ for the candidate centroids (all K, or in two-level mode the fine centroids of the `cprobe` best cells of each query token, whose blocks are fetched in one round if not cached); for each query token its `nprobe` best centroids; their union P, optionally pruned by `tcs`.

**Layout plaid, `approx=codes` (fast-plaid).** Round 1 reads the IVF lists of P; the candidates are the union of their documents. Round 2 reads the row of every candidate (this is the expensive read), scores each by centroid interaction Σᵢ maxⱼ S[code_j, i], keeps the best `ndocs`, and scores those exactly by decompressing their tokens (the residuals arrived with the codes, so there is no third round).

**Layout plaid, `approx=ivf`.** Round 1 as above, but each candidate is ranked from the IVF lists alone: for query token *i*, the best S[c, i] over the probed centroids c whose list contains the document, with missing pairs imputed as in the warp layout. Round 2 reads only the rows of the best `ndocs` candidates and scores them exactly. This is the "IVF-only approximate scoring" of `docs/plaid.md`, recommendation 5.

**Layout warp.** One round reads the posting lists of P. Every entry is decompressed (centroid + bucket weights, normalised) and scored against the query tokens (all of them with `cross=1`, only those that probed its centroid with `cross=0`); for each (document, query token) the maximum is kept, together with a flag saying whether one of the token's own probed centroids contributed. The document score is Σᵢ sᵢ where sᵢ is the maximum found if the flag is set, and otherwise max(maximum found, mᵢ), with mᵢ the score of token *i*'s last probed centroid (`impute=1`, WARP's missing-similarity estimate) or 0 (`impute=0`). With `rerank=R` the R best documents are re-scored exactly from their plaid rows in a second round.

**Exact** (`exact=1`) scans all document rows (or all posting lists if only the warp layout exists) and returns exact MaxSim over the decompressed vectors.

## Results

RESULTS_PLACEHOLDER
