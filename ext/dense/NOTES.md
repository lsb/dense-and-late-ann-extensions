# dense_ann: notes

`dense_ann` is a SQLite virtual table for approximate nearest-neighbour (ANN) search over dense embeddings: all-MiniLM-L6-v2, 384 dimensions, cosine distance. It is designed for a read-only database that a browser fetches lazily with HTTP range requests, where each dependent round trip costs 50–150 ms but many requests can be in flight at once.

It offers two index layouts behind one API:
- **Graph (default).** An HNSW-built graph with the neighbours' PQ codes stored in each node row.
- **IVF-PQ (`layout=ivf`).** Coarse lists over about 70-byte PQ entries. It costs 2 rounds per query.

At matched recall, IVF is 2–2.5× faster on synthetic data and on 10k real MiniLM documents. On 120k real MiniLM documents both layouts become expensive; IVF is the only one that reaches recall 0.9, at about 6 MB per query. See [IVF-PQ layout](#ivf-pq-layout-layoutivf).

- **Language and dependencies.** C11, depending only on SQLite and libm. The query path is single-threaded. Index building can optionally use pthreads (`-DDENSE_ANN_THREADS`, on in the native Makefile).
- **Files.**
  - `dense_ann.c`: the virtual table.
  - `pq.c`: product quantisation (PQ), with optional OPQ.
  - `hnsw.c`: in-memory HNSW construction.
  - `ivf.c`: coarse k-means and list probing for `layout=ivf`.
  - `rawpage.c`: direct page reads and the prefetch hook.
  - `test/countvfs.c`: a VFS that counts reads, used by the tests.
  - `bench.py`: the benchmark (recall, rounds, bytes and netsim `4g`/`slow-4g` times per configuration).
  - `compare.py`: graph vs IVF at matched recall, from the result files.
  - `run_words1m.sh`: one-command graph + IVF run on the real 1M MiniLM corpus.
  - `test/test_dense_ann.py`: the test suite.

## API

```sql
CREATE VIRTUAL TABLE v USING dense_ann(
    dim=384,              -- required
    pq_m=64,              -- PQ sub-quantisers x 8 bits = bytes per code (default 64 when dim % 64 == 0)
    M=16,                 -- HNSW degree; layer-0 degree M0 = 2M (override with M0=)
    ef_construction=200,
    store_vectors=f16,    -- none | int8 | f16 | f32: vectors kept for reranking and exact search
    vectors=inline,       -- inline (inside the node row) | table (separate shadow table)
    layout=colocated,     -- colocated (neighbour PQ codes in the node row) | separate | ivf
    nlist=0,              -- ivf: number of lists (0 = 4 sqrt(n))
    ivf_centroids=auto,   -- ivf: auto | f16 | int8 | pq storage of the cached centroids
                          --   (auto: int8 below 1,024 lists, pq from 1,024)
    codebook=int8,        -- f16 | int8: storage of the cached PQ codebook(s) (format 2)
    ivf_residual=1,       -- ivf: PQ on residuals x - centroid (IVFADC); 0 = on raw vectors
    nprobe=32, rerank_k=64,  -- ivf: query defaults
    metric=cosine,        -- cosine (normalises on insert/query) | ip | l2
    entry_points=1024,    -- size of the cached entry set
    opq=0,                -- OPQ iterations (0 = plain PQ)
    pq_train=65536, kmeans_iters=25, seed=42, alpha=1.0,
    reorder=auto,         -- auto | none | pack (pack graph neighbourhoods into pages)
    page_size_hint=0,     -- page size assumed by reorder (0 = current page size)
    threads=4, ef_search=64, beam=16, verbose=0);

INSERT INTO v(rowid, embedding) VALUES (?, ?);  -- float32 blob (dim*4 bytes) or JSON text '[...]'
INSERT INTO v(v) VALUES ('build');              -- train PQ on everything inserted so far, build the graph
INSERT INTO v(v) VALUES ('finalize');           -- store page hints; run last (after any VACUUM)

SELECT rowid, distance FROM v
 WHERE embedding MATCH ?1 AND k = 10            -- or LIMIT 10
   [AND ef = 64]        -- candidate list size
   [AND beam = 16]      -- W: nodes expanded (fetched in parallel) per round
   [AND rerank = 0|1|2|3]  -- default 2 when vectors are stored, else 0
   [AND exact = 1|2]    -- brute force: 1 = stored vectors, 2 = PQ codes
   [AND trace = 1]      -- stats JSON includes the page numbers fetched per round
   [AND nprobe = 32]    -- ivf: lists scanned
   [AND rerank_k = 64]; -- ivf: candidates reranked with stored vectors (R)
-- the hidden column `stats` returns per-query JSON:
--   rounds, pages, bytes, expanded, dist, entry_dist, rerank, fallback, setup_rounds, setup_pages, ms, ...

SELECT rowid FROM v WHERE embedding MATCH 'warm'; -- load the cached head now; returns no rows

SELECT embedding FROM v WHERE rowid = ?;        -- stored vector as float32 (NULL with store_vectors=none)
DELETE FROM v WHERE rowid = ?;  UPDATE ...;     -- supported (see "Updates")
```

The `rerank` modes:
- 0: order by PQ distance.
- 1: rerank the final `ef` candidates with the stored vectors.
- 2: rerank every expanded node with its stored vector, as DiskANN does. With `vectors=inline` this reads no extra pages.
- 3: like 2, but an expanded node's exact distance also replaces its PQ estimate during navigation (inline vectors only).

Other behaviour:
- **Before the first `build`**, inserts are buffered as float32 rows in `v_buffer`.
- **The first `MATCH` query builds the index automatically** if nobody has run `build` yet. On a read-only database, where that build fails, it answers by brute force over the buffer instead.
- **Inserts after the build** are encoded with the existing codebook and linked into the graph incrementally.
- **Loading.** Natively: `sqlite3_denseann_init`, or `.load build/dense_ann`. For a static or WASM build, call `sqlite3_denseann_init(db, 0, 0)` (compile with `-DSQLITE_CORE`).

## Design

### Why HNSW-built, single-layer-persisted, with a cached entry set

The graph is built with HNSW, as the owner suggested. It is built in memory from the full-precision vectors, and HNSW supports incremental inserts. Only layer 0 is persisted, one row per node. The upper layers are replaced at query time by an **entry set**: the `entry_points` nodes with the highest HNSW levels, which is effectively a random sample. The entry set is stored with its PQ codes and loaded once per connection.

A query first computes the PQ distance to every entry node in memory. This costs about 1,024 lookup-table sums, or 0.05 ms, and no network. The best entries then seed a DiskANN-style beam search on layer 0.

Why not keep the upper layers?
- **They are sequential.** Descending HNSW's upper layers is greedy, so each hop is a dependent round trip, which is exactly the resource we lack.
- **They are too big to cache.** Layer 1 alone holds n/M nodes, 62,500 at 1M. That is about 8 MB with co-located codes, too large to fetch once.
- **What remains.** Level ≥ 2 at 1M is about 3,900 nodes. Only the top of the hierarchy is small enough to cache, and a flat sample scanned in memory serves the same purpose (the Vamana/DiskANN "medoid plus cache" idea) with zero dependent fetches.

Vamana (DiskANN) was the alternative builder. The layer-0 graphs are very similar: the HNSW heuristic is Vamana's robust prune with α = 1, and α is exposed as `alpha`. HNSW needs no second pass and its incremental insert is standard, so it was the simpler choice.

### Storage schema (shadow tables)

Every per-node table is `(id INTEGER PRIMARY KEY, data BLOB)`. Node ids are dense internal ids assigned in storage order.

| table | row | notes |
|---|---|---|
| `v_config(key, value)` | parameters, state, chunk lists | read once per connection |
| `v_blobs` | 3,800-byte chunks | float16 PQ codebook (196,608 B = 52 chunks), entry set (1,024 × 72 B = 20 chunks), optional OPQ rotation (294,912 B) |
| `v_nodes` | one node | see below |
| `v_codes` | 64-byte PQ code | `layout=separate` only |
| `v_vectors` | stored vector | `vectors=table` only |
| `v_rowids(rowid, node)` | user rowid → node id | not on the query path |
| `v_buffer(rowid, vec)` | float32 vectors before `build` | emptied by `build` |

The node row is a fixed-size little-endian blob:

```
u8 flags (bit 0 = deleted) | u8 reserved | u16 degree | u32 vector-row page hint | i64 user rowid   (16 B)
own PQ code                                                                          (m = 64 B)
own vector, if vectors=inline                                                        (f16: 768 B, int8: 388 B)
M0 neighbour entries: u32 node id, u32 page hint [, u32 code-row page hint]          (32 x 8 B)
M0 neighbour PQ codes, if layout=colocated                                           (32 x 64 B)
```

With the defaults (M = 16, co-located), a row is 2,384 B, or 3,152 B with inline float16 vectors.
- **4 KiB pages.** Both sizes fit one row per page with no overflow; the local payload limit is 4,061 B, and even inline float32 fits at 3,920 B.
- **64 KiB pages.** About 20 rows per page.
- **Fixed size is load-bearing.** Rows have a fixed size so that in-place rewrites (adding a back-link, setting a page hint, setting the deleted flag) never move a row. SQLite overwrites a same-size cell in place.

### Page hints: reading node rows without walking the b-tree

This is the most important httpvfs-specific decision.

**The problem.** A normal `SELECT data FROM v_nodes WHERE id = ?` walks the table b-tree from the root through the interior pages to the leaf. At 1M rows with 4 KiB pages, that is about 2,000 second-level interior pages, roughly 8 MB, so they cannot all be cached. Each lookup therefore costs two or more dependent fetches. Worse, the pages for the W nodes of a search step cannot be requested together, because SQLite does the lookups one at a time.

**Page hints.** `finalize` walks the committed b-trees directly and records, next to every reference to a node, the page number of the leaf holding that node's row:
- in each neighbour entry;
- in each entry-set entry;
- in the config chunk lists;
- for vector rows, in the node header.

**Query-time reads.** For each search step, the extension does four things:
1. It collects the hinted pages of the W nodes to expand.
2. It announces them to the VFS in one prefetch call.
3. It reads each page with the main database file's `xRead`, obtained through `SQLITE_FCNTL_FILE_POINTER`.
4. It parses the table-leaf cell itself: a binary search on rowid, then the record header (`rawpage.c`, about 200 lines).

**Validation and fallback.** Every hint is checked: the page must be a table leaf, the rowid must be present, and the payload must be local (no overflow). A stale hint falls back to the SQL lookup, counted as `fallback` in the stats.

**When direct reads are allowed.** Only when all of these hold:
- the query is in autocommit mode, so the file on disk equals the committed state;
- the database is not in WAL mode (checked in the file header);
- the index has been finalized.

A database destined for httpvfs meets these conditions.

**Workflow.** The order is build → optional `PRAGMA page_size=…; VACUUM` → `finalize`. Anything that moves pages afterwards (another VACUUM, a page-size change) needs a new `finalize`. Inserts after `finalize` keep existing hints valid, because rows are appended at the end of the b-tree or updated in place. New nodes have no hint until the next `finalize`.

**Result.** Expanding W nodes costs exactly one round of W page fetches (fewer if pages repeat), with no interior pages. The one-time setup has three parts:
- page 1 and the schema;
- the config row;
- the codebook and entry-set chunks, all in one round.

### Co-located neighbour codes (DiskANN-style) versus the separate layout

In the co-located layout, a node row carries its neighbours' 64-byte codes. Expanding a node is therefore one page read, after which the distances to all its neighbours can be computed with no further reads.

The price is storage:
- every code is stored about 23 times, once per in-neighbour (the average degree is 23);
- the node table is about 2.4 KB per document instead of about 0.2 KB.

At 4 KiB pages the row costs a whole page anyway, so the slack holds the inline vector for free (see below).

In the separate layout, a node row holds ids and hints only, and the codes are 64-byte rows in `v_codes`, about 60 per 4 KiB page. Each step then needs:
- one round for the node rows;
- a second, dependent round for the neighbours' codes.

With random ids, a step touches about W × deg distinct code pages. The separate layout therefore doubles the rounds and multiplies the pages by about 1 + deg. Measurements are below.

### Reranking and where to keep full vectors

With PQ at 64 bytes, the ranking of near-ties is poor on real MiniLM data (see results), so reranking with stored vectors is essential.
- **`vectors=table`** costs one extra round plus about one page per reranked candidate: vectors are 768 B, five per page, in random order.
- **`vectors=inline`** puts the node's own float16 vector in its node row. At 4 KiB pages this costs **no extra bytes transferred and no extra rounds**: the 3,152-byte row still occupies one page. The database is even smaller, because the separate vector table disappears. Every expanded node's exact distance becomes available for free, which enables `rerank=2` (DiskANN's "rerank everything visited").

Recommendation: at 4 KiB pages, use `vectors=inline` with `store_vectors=f16`, and query with `rerank=2`.

### Search algorithm

DiskANN beam search. The candidate list L (size `ef`, sorted by PQ distance) is seeded from the entry set. Each round:
1. expand the W closest unexpanded candidates in L, fetching their rows in parallel;
2. score all unvisited neighbours by ADC from the co-located codes;
3. insert the good ones into L.

The search stops when every candidate in L has been expanded. So `expanded ≥ ef`, `rounds ≈ expanded / W` plus a few, and `pages ≈ expanded`.

`rerank=2` then rescores every expanded node with its stored vector and returns the top k.

ADC uses a per-query table of m × 256 floats (64 KiB). The OPQ rotation, if any, is applied to the query first.

### Updates

Inserts after `build` do the following:
1. PQ-encode the new vector with the existing codebook.
2. Run the same beam search with `ef = ef_construction` to find candidates.
3. Pick neighbours with the HNSW heuristic, using PQ-decoded vectors.
4. Write the new node row.
5. Add back-links. A full neighbour list is re-pruned with the heuristic, using the co-located codes, so no extra reads are needed.

Deletes set the deleted flag on the node row, which stays as a routing node, and remove the row from `v_rowids`. Search results skip deleted nodes. An UPDATE is a delete followed by an insert.

Neither operation changes the codebook or the entry set. Re-training requires a rebuild; there is no `rebuild` command yet, so recreate the table.

### Prefetch interface wanted from the HTTP VFS

Before each dependent step, the extension calls

```c
sqlite3_file_control(db, "main", DENSE_ANN_FCNTL_PREFETCH /* 0x44414e01 */, &req);
typedef struct { int n; int len; const sqlite3_int64 *offsets; } dense_ann_prefetch_req;
```

`offsets` are sorted, distinct file offsets of whole pages, and `len` is the page size. The VFS should:
1. start fetching all ranges in parallel, coalescing adjacent ones;
2. return when all of them are cached;
3. serve the `xRead` calls that immediately follow from that cache.

An unknown opcode returns `SQLITE_NOTFOUND`, so natively the call is a no-op.

A static build can instead register a function pointer with `dense_ann_set_prefetch_hook(fn, ctx)`. A browser-side implementation needs the fetch to block from the C side's point of view: sync XHR cannot run in parallel, so it would need Atomics.wait with a fetch worker, or JSPI/Asyncify. The `countvfs` test shim implements the opcode and counts the calls; a test asserts that every search round (plus the setup round) is announced exactly once.

## Results

**Setup.** Measured 2026-09-23/24 on 4 vCPUs shared with a running LLM generation job and a corpus encoding job; the load average was 10–16 throughout. Build times and QPS are therefore noisy and pessimistic, roughly 2–3× slower than on an idle machine. Round, page and recall numbers are deterministic up to thread scheduling in the graph build.

**Common configuration.** Unless stated otherwise:
- M = 16 (M0 = 32), `ef_construction` = 200, PQ 64 × 8 bits trained on a 65,536-vector sample;
- float16 vectors inline, co-located codes, 4 KiB pages, entry set of 1,024;
- k = 10; 200 queries for the synthetic data, 400 for words-10k.

**Measurement methods.**
- *QPS* is native, single-threaded, from Python including SQLite overhead, with a warm OS cache.
- *Recall@10* is against exact float32 search in numpy.
- *Rounds, pages and KiB* come from the extension's stats: dependent fetch steps, and distinct pages read per query (per-query cache only, no cross-query cache).
- *Warm miss* replays the page traces through an unbounded cache shared by all queries of the run (a long browser session) and averages over the second half of the queries.
- The `rounds` stat was checked against the project's native httpvfs (`build/native/httpvfs.so`, `wasm/`): its VFS-side round counter equals the extension's count exactly, plus one schema round on a cold connection.

**Datasets.**
- *synth-N*: unit cluster centres (√N clusters), plus a shared rank-48 component, plus small isotropic noise, normalised. Queries come from the same generator.
- *words-10k*: the enc agent's MiniLM embeddings (`data/emb/words-10k.minilm.npy`). Queries are 200 held-out documents of the same random-word stream (documents 10000–10199) and 200 single words, encoded with `enc/minilm.py`.

### Headline: 1M synthetic, 4 KiB pages (`results/synth1m-*.json`)

| ef | W | recall@10 (rerank=2) | recall@10 (PQ only) | rounds | pages | KiB/query | QPS |
|---|---|---|---|---|---|---|---|
| 32 | 16 | 0.876 | 0.525 | 6.2 | 79 | 315 | 1350 |
| 32 | 32 | 0.904 | – | 5.5 | 121 | 484 | 1256 |
| 64 | 8 | 0.958 | 0.530 | 11.4 | 86 | 345 | 2000 |
| **64** | **16** | **0.966** | 0.532 | **7.5** | **107** | **427** | 1858 |
| 64 | 32 | 0.972 | – | 5.9 | 152 | 607 | 923 |
| 128 | 16 | 0.993 | 0.531 | 10.9 | 167 | 669 | 1330 |
| 128 | 32 | 0.993 | – | 7.0 | 207 | 826 | 786 |
| 128 | 64 | 0.993 | – | 5.5 | 294 | 1176 | 608 |

Other measurements at 1M:
- **Exact PQ search** (`exact=2`) has recall@10 of 0.531, so reranking is what gives high recall.
- **Build.** `build` took 1,492 s:
  - PQ training and encoding: 127 s;
  - HNSW with 4 threads: 1,309 s;
  - writing: 31 s.

  `finalize` took 39 s. Inserting 1M float32 rows through Python's `executemany` took 564 s, mostly Python and the buffer table.
- **Memory.** Peak RSS was 3.3 GB, and that includes the benchmark's own 1.5 GB copy of the vectors. The extension holds the float32 vectors, the graph (132 MB) and the codes (64 MB).
- **Database size.** 3.93 GiB, which is 4,119 B/doc: 4,096 B for the node page plus the rowid map.
- **Cold first query** on a new connection (countvfs; ef = 64, W = 16) reads 176 distinct pages, 701 KiB:
  - schema and config: about 3 pages;
  - codebook plus entry set: 72 pages, 288 KiB, in one round;
  - the search itself.

  Through httpvfs this is 9 VFS rounds including setup.

### 100k synthetic: layout, page size, vector placement, entry set (`results/synth100k-*.json`)

All rows are ef = 64, W = 4, rerank = 2 unless noted. Recall@10 is 0.991 in every row: the graph and the rerank are identical, and only the storage changes.

| variant | DB B/doc | rounds | distinct pages | KiB/query | notes |
|---|---|---|---|---|---|
| co-located, inline f16, 4 KiB (default) | 4,121 | 17.3 | 68.0 | 272 | W=16: 6.0 rounds, 87 pages, 349 KiB |
| **separate** codes table (naive), 4 KiB | 2,053 | **34.3** | **622** | **2,487** | two dependent rounds per step; about 9 code pages per expanded node |
| vectors in a table (`vectors=table`), rerank=2 | 4,942 | 18.3 | 136 | 544 | +1 round, +1 page per reranked node; rerank=0: 17.3 / 68 |
| 64 KiB pages, `reorder=pack` | 3,298 | 16.1 | 37.6 | **2,405** | packing gives 1.8 expanded nodes per page, but each page is 16× larger |
| 64 KiB pages, no reorder | 3,298 | 17.3 | 67.6 | 4,324 | |
| entry set 256 | 4,120 | 18.4 | 72.0 | 288 | setup 18 KiB smaller |
| entry set 8,192 | 4,127 | 16.3 | 64.4 | 258 | setup +504 KiB (cold query 1,145 KiB) |

### words-10k, real MiniLM embeddings (`results/w10k-*.json`)

| ef | W | recall@10 | held-out docs | single words | rounds | pages | KiB |
|---|---|---|---|---|---|---|---|
| 32 | 16 | 0.718 | 0.798 | 0.639 | 5.2 | 51 | 204 |
| 64 | 16 | 0.805 | 0.885 | 0.726 | 6.7 | 77 | 307 |
| 64 | 32 | 0.839 | 0.912 | 0.766 | 5.2 | 98 | 393 |
| 64 | 64 | 0.870 | 0.934 | 0.807 | 4.4 | 138 | 550 |
| 128 | 16 | 0.889 | 0.956 | 0.822 | 9.9 | 135 | 539 |
| 128 | 64 | 0.917 | 0.970 | 0.864 | 5.1 | 189 | 758 |
| 256 | 16 | 0.952 | 0.988 | 0.915 | 17.3 | 259 | 1,037 |

All rows use rerank = 2.

**PQ is the bottleneck on real data.**
- Exact PQ search reaches recall@10 of only **0.572** (held-out documents 0.547, single words 0.597).
- Faiss `IndexPQ(384, 64, 8)` gives 0.567 on the same data, which validates the implementation.
- The reason is near-ties. Averaged over held-out-document queries, the 1st, 10th and 100th neighbours have cosine 0.686, 0.655 and 0.620; for single words, 0.412, 0.356 and 0.312. PQ-64 error is larger than those gaps.
- PQ is still a good candidate generator: the true top-10 is 98% inside the PQ top-100 (faiss). So recall is bought by expanding more nodes (larger ef, hence more pages), and W converts pages into fewer rounds.
- For reference (faiss, exact search over codes): PQ96 reaches 0.706, PQ128 0.797 and SQ4 (192 B) 0.836.
- The enc notes report that the qint8 and fp32 MiniLM models themselves agree on only 77% of top-10 for random-word documents. Recall@10 above about 0.8 is therefore partly fitting encoder noise on this corpus.

**OPQ** (`opq=10`: 10 alternations on a 20k sample, Jacobi-SVD Procrustes) does help:
- exact-PQ recall rises 0.572 → 0.614 (faiss OPQ: 0.592);
- end-to-end recall at ef = 128, W = 4 rises 0.885 → 0.900;
- ef = 64 rises 0.782 → 0.795.

The float16 rotation, however, adds 288 KiB to the one-time setup (cold first query 563 → 878 KiB), so it is opt-in.

**Exact navigation** (`rerank=3`, which uses an expanded node's inline vector instead of its PQ estimate) adds 1–2 points of recall for about 10% more expansions, which is about the same as raising ef. It is kept as an option.

**Latency check with the native httpvfs at 100 ms latency per round** (bandwidth not simulated), words-10k, ef = 64, W = 16:
- first query: 810 ms (setup + 6 rounds);
- later queries: 500–900 ms (5–9 rounds);
- that is, about rounds × 100 ms.

## Fetch-pattern analysis: what a 1M-document query costs

**Row geometry.** Default row 3,152 B: 16 B header + 64 B code + 768 B float16 vector + 32 × 8 B neighbour ids and hints + 32 × 64 B neighbour codes.

**4 KiB pages.** One node per page; `v_nodes` is 1M leaf pages (3.9 GiB).

*Setup, once per connection:*
- page 1;
- `sqlite_schema` and `v_config` (two to three small dependent reads);
- one parallel round of 52 codebook and 20 entry-set chunk pages (288 KiB).

That is about **3 rounds and 300 KiB**, plus 288 KiB more with OPQ.

*Per query:* `rounds ≈ ⌈expanded / W⌉ + ~2` and `pages ≈ expanded ≈ ef + W`. Measured:
- ef = 64, W = 16: 7.5 rounds and 107 pages (427 KiB), recall 0.966 (synthetic);
- ef = 128, W = 32: 7 rounds and 826 KiB for 0.993.

On MiniLM data, the same recall needs roughly twice the ef.

*A 4G phone estimate* (100 ms RTT, about 10 Mbit/s = 1.25 MB/s): 7.5 × 100 ms + 427 KiB / 1.25 MB/s ≈ 0.75 s + 0.35 s ≈ **1.1 s**, plus about 0.55 s the first time. One round costs about as much as 30 pages. That puts the optimum near W = 16–32 for ef = 64 and W = 32–64 for ef = 128, which is why the default beam is 16.

Cross-query caching barely helps at 1M. After 100 queries, a query still misses 86 of its 107 pages and 7.1 of its 7.5 rounds. Only the pages near the entry set get reused.

**64 KiB pages.** About 20 nodes per page. Packing graph neighbourhoods into pages (`reorder=pack`) roughly halves the distinct pages (100k: 68 → 38), but every page is 16× larger. The result is **about 2.4 MB per query versus 272 KiB**, with the same number of rounds. The setup is about 5 pages of 64 KiB instead of 72 of 4 KiB, which is about the same bytes. Large pages only pay off if the VFS fetches sub-page ranges (possible, because we parse cells ourselves: only the cell's byte range is needed) or if nodes per page approach nodes needed per query. Neither holds here, so **use 4 KiB pages** (or 8 KiB with 2 nodes per page).

**Naive layout** (neighbour ids only; codes in their own table). Each step is 2 dependent rounds, and the code fetch touches about deg × W random code pages. At 100k the measured cost was 2× the rounds and 9× the pages. At 1M the codes span 16k pages, so almost every code is a distinct page: expect about 2 × 17 rounds and about 1,500–2,500 pages (6–10 MB) per query. The naive layout is not viable over HTTP. Co-location costs about 2 KB/doc of duplicated codes, which is disk and CDN storage, not transfer.

**Vectors in a separate table.** Costs +1 round and +1 page per reranked node (100k: 136 pages instead of 68). Inline is free at 4 KiB pages because the row fills a page anyway.

**Without page hints** (plain SQL lookups through the b-tree). Each node read costs 2–3 dependent page reads (root and level-1 interior pages get cached, level-2 interior pages do not at 1M), and SQLite issues the W lookups of a step one after another. That multiplies rounds by about 2 × W. `finalize` removes this entirely. The wasm agent's `httpvfs_speculate_*` could batch the interior-page reads, at about one round per uncached level.

## IVF-PQ layout (`layout=ivf`)

The owner asked for about 64 bytes per document. The graph's node rows cost about 4 KB per document, because neighbour codes are duplicated about 23 times and a row fills a 4 KiB page. IVF-PQ stores each PQ code once.

### Design
- **Coarse quantiser.** k-means (10 Lloyd iterations on a sample of 40 points per list, at least 65,536) with `nlist` centroids. The default is 4√N: 400 at 10k, 1,265 at 100k, 4,000 at 1M. The assignment kernel dots four points against each centroid at once (`ivf.c`).
- **Codes.** PQ-64 × 8 bits on the residual x − c(x) (IVFADC, the default), or on the raw vector (`ivf_residual=0`). The residual is taken against the *stored* (decoded) centroid, so the stored centroid and the codes are consistent. OPQ (`opq=N`) applies to the codes. For cosine/IP, d(q, c + r) = ADC(q, r) − ⟨q, c⟩, so one ADC table serves every list. For L2, a table per probed list is built from q − c.
- **Cached head.** Loaded once per connection, in one round together with the PQ codebook (float16, 192 KiB):
  - the centroids: `f16` (768 B each), `int8` (388 B) or `pq` (64-byte codes plus their own float16 codebook, 192 KiB);
  - a directory of 12 B per list: number of chunks, first page, last page.
- **Posting lists** go in `v_lists(id = list << 20 | j, data)`. Each chunk is at most 4,000 bytes, so it fits a 4 KiB page without overflow:
  - header: `u16 count | u16 reserved | u32 first slot | u32 first vector page`;
  - 57 entries of `u32 rowid | u8 vector-page delta | 64 B code`, i.e. **69 B per document**.
- **Contiguous lists.** Lists are written list by list, so each one is a contiguous page range. `finalize` records the range in the directory, and a query fetches all `nprobe` ranges in one round; adjacent pages coalesce into one range request.
  - To keep the ranges contiguous, `build` frees the float32 buffer only *after* writing. Reusing scattered freelist pages broke 52 of 400 lists on the first attempt. Run VACUUM (before `finalize`) to reclaim the buffer.
  - A list whose chunks turn out non-contiguous or out of order loses its range and falls back to SQL lookups (counted in `fallback`).
- **Rerank vectors** go in `v_vectors(id = slot, data)`, where slots are positions in (list, rowid) order. A list's vectors are therefore on consecutive pages, and candidates from the same list often share a page. Each chunk stores the page of its first slot plus a one-byte delta per entry, so the vector rows of the top R candidates are fetched **in one round with no b-tree lookups**.
- **The rowid map** `v_rowids(rowid, node = list << 32 | slot)` serves deletes and exact search only. Rowids must fit in 32 bits.

### Query cost
- **Round 1:** the nprobe list ranges, about nprobe × (N/nlist × 69 B / 4 KiB + a partial page) pages. At 1M with 4,000 lists this is 5.7 pages per list.
- **Round 2:** about R pages of vectors, fewer when candidates share pages. Without rerank there is only round 1.

### Updates
- **Insert after build** assigns the vector to its nearest list and appends a one-entry chunk. Such chunks are found through SQL until the next `finalize`, and many of them fragment the lists; rebuild after large changes.
- **Delete** flags the entry in its chunk (delta `0xFE`).
- The coarse centroids and the PQ codebook are never retrained.

### Results

**Setup.** Same machine and methods as above: 4 vCPUs shared with the LLM and late-interaction jobs, threads = 2 for the IVF builds.
- **Time estimates.** `4g` (165 ms, 8.1 Mbit/s) and `slow-4g` (562.5 ms, 1.44 Mbit/s) are simulated with `netsim/simulate.py` from each query's recorded per-round page lists (adjacent pages merged into one range), averaged over 50 queries on a warm connection. The one-time setup is listed separately; it is modelled as page 1, the schema page, then the head in one range.
- **Selection.** `compare.py` picks the configuration with the lowest `4g` time that reaches each target recall.
- **Graph runs:** M = 16, inline float16 vectors, `rerank=2`.
- **IVF runs:** float16 rerank vectors, residual PQ-64.

| data | target | index | best config | recall@10 | rounds | KiB/query | 4g ms | slow-4g ms | setup KiB | setup 4g ms | index B/doc | build s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| synth 1M | 0.90 | graph | ef=32 W=32 | 0.905 | 5.5 | 483 | 1,336 | 5,616 | 288 | 794 | 4,119 | 891 (4 thr) |
| synth 1M | 0.90 | IVF, pq centroids | nprobe=8 R=32 | 0.940 | 2 | 302 | **635** | **2,839** | 736 | 1,247 | 917 | 944 (2 thr) |
| synth 1M | 0.95 | graph | ef=64 W=32 | 0.972 | 5.9 | 606 | 1,526 | 6,519 | 288 | 794 | 4,119 | 891 |
| synth 1M | 0.95 | IVF, pq centroids | nprobe=8 R=64 | 0.995 | 2 | 404 | **738** | **3,421** | 736 | 1,247 | 917 | 944 |
| synth 100k | 0.90 | graph | ef=32 W=16 | 0.939 | 4.8 | 245 | 1,040 | 4,093 | 288 | 794 | 4,121 | 78 |
| synth 100k | 0.95 | graph | ef=32 W=32 | 0.953 | 4.3 | 378 | 1,098 | 4,587 | 288 | 794 | 4,121 | 78 |
| synth 100k | 0.95 | IVF, pq centroids | nprobe=8 R=32 | 0.986 | 2 | 172 | **505** | **2,107** | 520 | 1,029 | 933 | 136 |
| words-10k | 0.90 | graph | ef=128 W=32 | 0.902 | 6.7 | 597 | 1,719 | 7,199 | 288 | 794 | 4,149 | 8 |
| words-10k | 0.90 | IVF (nlist 400) | nprobe=128 R=64 | 0.934 | 2 | 826 | **1,167** | **5,834** | 540 | 1,049 | 989 | 7 |
| words-10k | 0.90 | IVF (nlist 100) | nprobe=64 R=64 | 0.945 | 2 | 859 | 1,204 | 6,043 | 292 | 798 | 956 | 5 |
| words-10k | 0.95 | graph | ef=256 W=32 | 0.954 | 9.8 | 1,058 | 2,671 | 11,471 | 288 | 794 | 4,149 | 8 |
| words-10k | 0.95 | IVF (nlist 400) | nprobe=128 R=128 | 0.957 | 2 | 1,049 | **1,394** | **7,111** | 540 | 1,049 | 989 | 7 |
| words-10k | 0.95 | IVF, int8 vectors | nprobe=256 R=64 | 0.960 | 2 | 1,118 | 1,464 | 7,505 | 540 | 1,049 | **578** | 8 |

Notes on the table:
- Index B/doc excludes free pages (as after VACUUM).
- The 1M graph build used 4 threads, the IVF builds 2.

**Headline (synthetic data and words-10k).** At matched recall, IVF-PQ is **2–2.5× faster** than the graph on `4g` and **1.6–2.2× faster** on `slow-4g`. It needs 2 rounds instead of 4–10, similar or fewer bytes, and a 4.5× smaller database. The price is a larger one-time setup (520–740 KiB versus 288 KiB). Even including setup, a cold first query is still faster on IVF at synthetic 1M: 1,247 + 738 ms versus 794 + 1,526 ms.

At 120k real MiniLM documents both layouts become expensive; see the next table.

**Caveat: synthetic data flatters IVF.** The generator draws points around √N cluster centres, so with 4√N lists `nprobe=8` already covers 100% of the true top-10.

**On real MiniLM data (words-10k), coarse probing is the bottleneck.** PQ plus rerank recovers essentially everything inside the probed lists. Coverage of the true top-10 by the probed lists (nlist = 400):

| nprobe | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|
| coverage | 0.36 | 0.52 | 0.68 | 0.84 | 0.96 |

- 0.95 recall needs about 30% of the corpus scanned at 10k. This is the random-word near-tie problem again: neighbours are spread over many lists.
- Whether the scanned fraction falls at 1M is **the** open question for the real corpus (`run_words1m.sh`). If it does not, IVF bytes grow linearly with N while the graph's grow roughly logarithmically, and the graph would win at 1M on MiniLM.
- The table suggests the break-even point: in the `4g` model, about 165 KiB (165 ms at 8.1 Mbit/s) costs as much as one round, so IVF stays ahead while its extra bytes are under about (graph rounds − 2) × 165 KiB.

**Real MiniLM at 120k documents** (`results/w120k-*.json`). The enc agent's 1M encoding stopped after 12 of 100 chunks, which is the first 120,000 documents of words-1m. Queries are held-out documents 120,000–120,199 plus the 200 single words. Exact search over PQ codes reaches recall 0.51–0.53.

| target | index | best config | recall@10 | rounds | KiB/query | 4g ms | slow-4g ms | setup KiB | B/doc | build s |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.80 | graph | ef=512 W=64 | 0.815 | 12.5 | 2,405 | 4,453 | 20,551 | 288 | 4,121 | 107 |
| 0.80 | IVF, 1,386 lists | nprobe=256 R=128 | 0.834 | 2 | 3,207 | **3,609** | **19,571** | 528 | 927 | 144 |
| 0.90 | graph | not reached (best: 0.815) | | | | | | | | |
| 0.90 | IVF, 1,386 lists | nprobe=512 R=128 | 0.924 | 2 | 5,786 | 6,242 | 34,379 | 528 | 927 | 144 |
| 0.95 | IVF, 1,386 lists | nprobe=512 R=256 | 0.951 | 2 | 6,266 | 6,727 | 37,109 | 528 | 927 | 144 |
| 0.95 | IVF, 346 lists | nprobe=256 R=128 | 0.951 | 2 | 7,677 | 8,140 | 45,057 | 492 | 918 | 68 |

What the 120k results show:
- **Both layouts get harder with N on this corpus.** IVF at 0.92 scans 37% of the lists (62k documents, 5.8 MB), the same fraction as at 10k. So bytes per query grow linearly with N, and at 1M this would be tens of MB, which is unusable.
- **The graph's recall collapses on single-word queries.** It gets 0.70 at ef=512, versus 0.93 for held-out documents. Short queries lie off the manifold of 50-word documents, a known weakness of graph indexes with out-of-distribution queries (RoarGraph-style query-aware graphs address it). IVF handles them better (0.91–0.97).
- **The corpus itself is part of the problem.** Random-word documents are near-ties: the 10th and 100th neighbours differ by about 0.03 in cosine. Recall@10 against exact search therefore rewards reproducing encoder noise; recall of *relevant* documents is a better target. The next evaluation should use `data/llm` (the LLM paragraphs with known relevant paragraphs per query) at scale.

**Variants (words-10k).** Unless noted, nprobe = 128, R = 128. The results files are `results/w10k-ivf*.json`.

| variant | recall@10 | KiB/query | setup KiB | B/doc | note |
|---|---|---|---|---|---|
| default: residual PQ, f16 centroids, f16 vectors | 0.957 | 1,049 | 540 | 989 | |
| `ivf_residual=0` (PQ on raw vectors) | 0.951 | 1,059 | 540 | 989 | residuals help a little: exact-PQ 0.589 vs 0.572 |
| `ivf_centroids=int8` | 0.956 | 1,048 | 380 | 973 | no loss |
| `ivf_centroids=pq` | 0.951 | 1,038 | 448 | 980 | pays off from about 1,000 lists (1M: 736 KiB instead of about 3.3 MB with f16) |
| `store_vectors=int8` | 0.948 | 1,004 | 540 | **578** | halves the DB; transfer about the same (random vector pages either way) |
| `opq=10` | 0.957 (R=64: 0.941 vs 0.934) | 1,050 | 852 | 1,021 | small gain, +300 KiB setup |
| `nlist=100` (√N) | 0.945 at nprobe=64 R=64 | 859 | 292 | 956 | fewer, larger lists: better here |
| `nlist=1600` (16√N) | 0.935 at nprobe=256 | 1,151 | 1,524 | 1,083 | worse, and a large head |
| no rerank (`rerank=0`) | 0.58 at nprobe=128 | 586 (1 round) | 540 | ~140 without vectors | capped by PQ-64 (exact PQ: 0.589) |

**Storage at 1M** (`ivf_centroids=pq`), 917 B/doc in total:
- posting lists: 79.5 B/doc (69 B entries plus chunk headers and partial pages);
- float16 vectors: 821 B/doc (five per 4 KiB page);
- rowid map: 15 B/doc.

The minimal read-only variants are:
- without stored vectors: about **95 B/doc**, but recall is capped near 0.6 on MiniLM;
- with int8 vectors: about 500 B/doc.

**Build at 1M:** 944 s with 2 threads:
- coarse k-means (4,000 lists, 160k sample, 10 iterations) plus the assignment of 1M vectors: 497 s;
- PQ training and encoding of residuals: 427 s;
- writing: 5 s.

Peak RSS was 4.7 GB, including the benchmark's own copy of the vectors and a float32 residual buffer (1.5 GB) that could be streamed.

**Recommendation.**
- **Data with cluster structure** (the synthetic sets, and most topical text corpora): use `layout=ivf` with `ivf_centroids=pq` (or `int8` below about 1,000 lists) and `store_vectors=int8` or `f16`. Start at `nprobe` ≈ 1–4% of `nlist` with `rerank_k=64`. It is 2 rounds, about 0.4–1 MB per query and about 0.7–1.2 s on `4g`, plus a one-time setup of about 0.5–0.75 MB.
- **Near-tie data such as random-word MiniLM:** neither layout is cheap beyond about 100k documents. IVF is the better of the two (2 rounds; the only layout that reaches 0.9), but its bytes grow linearly with N.
- **What would change that:**
  - spilling each vector into two lists (SOAR);
  - larger codes;
  - query-aware graphs;
  - accepting a lower recall@10 against exact search when it tracks relevance.

  Evaluate on relevance-labelled data before choosing.

### Real 1M MiniLM corpus: not run yet
`data/emb/words-1m.minilm.npy` does not exist yet. The encoder stopped after 12 of 100 chunks, i.e. 120k documents, which were used above via `--emb`. Once the file exists, run:

```sh
ext/dense/run_words1m.sh
```

The script:
- builds and sweeps the graph (`graphw` sweep), IVF with 4,000 lists and IVF with 1,000 lists (`ivfw` sweeps up to nprobe = 1,024), with 2 threads;
- deletes each multi-GB database after use;
- prints the matched-recall table.

Queries are 200 held-out documents 1,000,000–1,000,199 regenerated from the deterministic word stream (`bench.heldout_docs`), plus 200 single words. It takes about 1–1.5 h on this machine and peaks around 6 GB of disk.

## Per-connection static data (2026-09-24)

Every new connection first loads the *index head*: the PQ codebook plus the entry set (graph) or the coarse centroids and list directory (IVF). On a phone this is most of the first query's cost, so it was shrunk and made loadable ahead of time.

- **`MATCH 'warm'`.** `SELECT rowid FROM v WHERE embedding MATCH 'warm'` loads the head and returns no rows. The browser client runs it in the background while the query encoder loads (`web/NOTES.md`), so the first search no longer waits for it.
- **`codebook=int8` (new default, format 2).** Each of the 64 sub-quantisers stores a float32 scale and its 256 × 6 values as int8, 98.6 KB instead of 196.6 KB as float16. The trained codebook is rounded to the stored values *before* encoding, so codes and query-time tables agree. The same applies to the centroid-PQ codebook of `ivf_centroids=pq`.
- **`ivf_centroids=auto` (new default).** `int8` below 1,024 lists, `pq` from 1,024 lists (where 64-byte codes plus a 98.6 KB codebook become smaller than 388 B per int8 centroid).
- **Format.** The head format is the config key `format` (2 when written by this build) with `codebook` giving the codebook type. An index without these keys (format 1) is read as before: float16 codebooks. A build that finds a newer format refuses with `dense_ann: index format N is not supported by this build`. Older builds cannot read an int8 codebook (they report that the index head cannot be loaded); build with `codebook=f16` for them.

Measured with `bench/static_eval.py` (one index per database, built with the matrix parameters; all 2,000 queries of each corpus for quality; 40 cold queries through the WASM build, each on a new connection, replayed through netsim; p50):

| Corpus | Index | Head | Head KB | Config | nDCG@10 | R@10 vs exact | Cold rounds | Cold KB | 4g h1 ms | 4g h2 ms | lte h1 ms | lte h2 ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| llm-10k | graph | f16 codebook (before) | 264 | ef 64 | 0.587 | 0.984 | 9 | 596 | 3,575 | 2,086 | 1,669 | 1,037 |
| llm-10k | graph | **int8 codebook** | **168** | ef 64 | 0.587 | 0.984 | 9 | 492 | 3,468 | 1,978 | 1,597 | 964 |
| llm-10k | graph | int8, 256 entries | 114 | ef 64 | 0.586 | 0.985 | 9 | 460 | 3,607 | 1,957 | 1,649 | 949 |
| llm-10k | graph | int8, 4,096 entries | 384 | ef 64 | 0.586 | 0.987 | 8 | 690 | 3,342 | 2,022 | 1,595 | 1,035 |
| llm-10k | IVF | f16 centroids + codebook (before) | 497 | nprobe 16 | 0.488 | 0.739 | 5 | 774 | 2,352 | 1,612 | 1,175 | 882 |
| llm-10k | IVF | **int8 centroids + codebook** | **252** | nprobe 16 | 0.487 | 0.739 | 5 | 508 | 2,079 | 1,345 | 990 | 702 |
| llm-10k | IVF | pq centroids, int8 codebooks | 222 | nprobe 16 | 0.479 | 0.734 | 5 | 472 | 2,033 | 1,306 | 959 | 675 |
| llm-10k | IVF | f16 (before) | 497 | nprobe 64 | 0.545 | 0.909 | 5 | 976 | 3,432 | 1,814 | 1,655 | 1,018 |
| llm-10k | IVF | **int8** | 252 | nprobe 64 | 0.544 | 0.909 | 5 | 714 | 3,159 | 1,549 | 1,471 | 839 |
| words-10k | graph | f16 codebook (before) | 264 | ef 64 | 0.073 | 0.730 | 10 | 608 | 3,752 | 2,265 | 1,747 | 1,117 |
| words-10k | graph | **int8 codebook** | 168 | ef 64 | 0.072 | 0.730 | 10 | 508 | 3,651 | 2,166 | 1,680 | 1,050 |
| words-10k | graph | int8, 256 entries | 114 | ef 64 | 0.071 | 0.723 | 10 | 492 | 3,967 | 2,168 | 1,809 | 1,051 |
| words-10k | graph | int8, 4,096 entries | 384 | ef 64 | 0.078 | 0.806 | 9 | 704 | 3,523 | 2,202 | 1,675 | 1,115 |
| words-10k | IVF | f16 (before) | 497 | nprobe 64 | 0.071 | 0.802 | 5 | 1,088 | 3,978 | 1,927 | 1,900 | 1,095 |
| words-10k | IVF | **int8** | 252 | nprobe 64 | 0.071 | 0.801 | 5 | 826 | 3,708 | 1,662 | 1,718 | 916 |
| words-10k | IVF | pq | 222 | nprobe 64 | 0.070 | 0.794 | 5 | 788 | 3,670 | 1,624 | 1,692 | 890 |

(graph ef 128 and IVF nprobe 16 on words-10k behave the same way; every row is in `results/static/`.)

- **int8 codebooks cost nothing measurable** (nDCG within 0.001, recall against exact search within 0.001) and remove 96 KB from every head: the graph head shrinks 264 → 168 KB, 100 ms less on a cold `4g` query.
- **int8 IVF centroids are free too** (as on words-10k earlier): with the int8 codebook the IVF head halves (497 → 252 KB) and a cold query is 250–270 ms faster on `4g`. PQ centroids save another 30 KB at 400 lists but lose 0.008 nDCG at nprobe 16, so `auto` keeps int8 at this size.
- **The entry set is worth its bytes.** 256 entries save 54 KB but cost recall and, under HTTP/1.1, a little time; 4,096 entries (+216 KB) raise recall against exact search on words-10k from 0.730 to 0.806 at ef 64 and save a round. The default stays at 1,024; a deployment that warms the head in the background can afford a larger one (`entry_points=4096`).

## Comparison with libSQL and sqlite-vec

**libSQL `libsql_vector_idx` (DiskANN).** Verified by reading `libsql-sqlite3/src/vectordiskann.c` and `vectorIndexInt.h` (main branch, cloned 2026-09-24).
- **Graph and storage.** It is a Vamana/DiskANN graph (FreshDiskANN and LM-DiskANN are cited) stored in a shadow table `<idx>_shadow(index key, data BLOB)`. One row per node, one "block" per row.
- **Block layout.** `[u64 rowid][u16 nEdges][pad][node vector][edge vector × maxEdges][(u32 unused, f32 distance, u64 edge rowid) × nEdges]`. That is the same co-location idea as here: neighbours' vectors live in the node's block. They are "compressed" by `compress_neighbors` (float8, float1bit, …) rather than PQ, with the node's own vector at full type for reranking.
- **Defaults.**
  - max_neighbors = min(3(√D + 1), …), which is 60 at D = 384;
  - alpha = 1.2, insert L = 70, search L = 200;
  - the block size is derived from these.
- **Reads.** Each block is read with `sqlite3_blob_open` by rowid, i.e. a b-tree walk per node.
- **Search order.** Candidates are expanded **one at a time** (beam width 1, `diskAnnSearchCtxFindClosestCandidateIdx`), starting from a **random row**.
- **Consequences over HTTP.**
  - With 60 float8 neighbour vectors of 384 dimensions, a block is about 25 KB. It overflows 4 KiB pages into an overflow chain, which means sequential reads.
  - Search L = 200 means about 200+ dependent node reads, each needing several page reads.
  - It is designed for local disk and has no parallel-prefetch path.
- **Our differences:** PQ-64 edge codes (64 B/edge rather than 392+), rows that fit one page, page hints (no b-tree walk), beam width W with parallel prefetch, and an in-memory entry set instead of a random start.

**sqlite-vec (`vec0`)**, v0.1.10-alpha on main, checked in its repository:
- **Brute force only.** Vectors are stored in chunked blobs (`<t>_vector_chunksNN`) and scanned exhaustively. Float32, int8 and bit (binary quantisation) are supported, with metadata and partition columns.
- **ANN on branches.** The repository's `TODO.md` describes an `ann` branch with DiskANN (e.g. R = 48, L = 128, binary quantiser), IVF and Annoy implementations. None of it is in a release, as far as the main branch shows.
- **Over HTTP**, brute force means downloading every vector (1M × 384 float32 = 1.5 GB; binary 48 MB), so it only suits small collections.

## Open questions and next steps

- **Real MiniLM beyond 10k.** At 120k the scanned fraction IVF needs for 0.9 did not shrink (about 37% of the lists), and the graph plateaus near 0.8, mostly on single-word queries. Questions for the full 1M corpus and for relevance-labelled data (`data/llm`):
  - Does spilling each vector into its two nearest lists (SOAR/ScaNN, doubling the 69 B entries) buy coverage cheaply?
  - Would a query-aware graph fix single-word queries?
  - The IVF rerank round costs about 1 page per candidate; clustering co-candidates' vector rows, or int4 vectors, would cut it.

- **Real 1M MiniLM corpus.** The embeddings are coming from the enc agent. The synthetic data is easier than MiniLM: at 10k, words need about 2× the ef of synthetic data for the same recall. Rerun `bench.py --data words-1m` once `data/emb/words-1m.minilm.npy` exists (the loader already handles the `words-*` names).
- **PQ quality versus bytes.** At fixed transfer, is PQ-96 or PQ-128 with a smaller M0 (the same row size) better than PQ-64 with M0 = 32?
  - Row size per neighbour is 8 + m bytes, so M0 = 24 with m = 96 is about the same row.
  - Alternatively, 4-bit fast-scan PQ with more subspaces.
  - OPQ helps a little; its 288 KiB setup could shrink to 144 KiB with int8 storage, or be replaced by a cheap permutation plus PCA if that captures most of the gain.
- **Fewer rounds.**
  - The measured floor is 4–5 rounds (hop depth from the entry set) even with W = 64.
  - Options:
    - a larger entry set (8,192 entries saves 1 round for about 500 KiB of setup);
    - a two-level entry structure: a cached top layer that picks one of ~64 cached "region entry lists";
    - speculative prefetch of the second-best candidates' neighbours;
    - early termination once the top-k is stable.
- **The page-hint workflow.** `finalize` must run after any VACUUM or page-size change, in rollback-journal mode. Should `build` finalize automatically when it can (autocommit, not WAL)? It cannot run inside the same statement's transaction, because it reads the committed file.
- **Incremental inserts** do not update the entry set or re-train PQ, and new rows get no page hints until the next `finalize`. Deletes leave tombstoned routing nodes. There is no compaction or rebuild command yet.
- **Build speed.** HNSW with `ef_construction` = 200 is the dominant cost: 22 min for 1M on a loaded 4-core machine. `ef_construction` = 100 or a Vamana two-pass build are options. The build keeps all float32 vectors in memory (1.5 GB at 1M).
- **`insert` pruning** uses PQ-decoded vectors, not the stored float16 ones, so graph quality after many incremental inserts is untested at scale.
- **Browser runtime.** Rows are parsed by our own cell parser. It assumes little-endian hosts for float32 blobs from SQL (true for x86, ARM and wasm) and no reserved-byte extensions beyond the header's value. The query path is single-threaded and WASM SIMD-friendly: the ADC loop is m lookups per neighbour (about 23 × 64 per expansion).
