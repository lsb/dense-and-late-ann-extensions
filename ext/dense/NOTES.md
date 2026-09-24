# dense_ann: notes

`dense_ann` is a SQLite virtual table for approximate nearest-neighbour (ANN) search over dense embeddings: all-MiniLM-L6-v2, 384 dimensions, cosine distance. It is designed for a read-only database that a browser fetches lazily with HTTP range requests, where each dependent round trip costs 50–150 ms but many requests can be in flight at once.

- **Language and dependencies.** C11, depending only on SQLite and libm. The query path is single-threaded. Index building can optionally use pthreads (`-DDENSE_ANN_THREADS`, on in the native Makefile).
- **Files.**
  - `dense_ann.c`: the virtual table.
  - `pq.c`: product quantisation (PQ), with optional OPQ.
  - `hnsw.c`: in-memory HNSW construction.
  - `rawpage.c`: direct page reads and the prefetch hook.
  - `test/countvfs.c`: a VFS that counts reads, used by the tests.
  - `bench.py`: the benchmark.
  - `test/test_dense_ann.py`: the test suite.

## API

```sql
CREATE VIRTUAL TABLE v USING dense_ann(
    dim=384,              -- required
    pq_m=64,              -- PQ sub-quantisers x 8 bits = bytes per code (default 64 when dim % 64 == 0)
    M=16,                 -- HNSW degree; layer-0 degree M0 = 2M (override with M0=)
    ef_construction=200,
    store_vectors=f16,    -- none | int8 | f16 | f32: vectors kept for reranking and exact search
    vectors=table,        -- inline (inside the node row) | table (separate shadow table)
    layout=colocated,     -- colocated (neighbour PQ codes in the node row) | separate
    metric=cosine,        -- cosine (normalises on insert/query) | ip | l2
    entry_points=1024,    -- size of the cached entry set
    opq=0,                -- OPQ iterations (0 = plain PQ)
    pq_train=65536, kmeans_iters=25, seed=42, alpha=1.0,
    reorder=auto,         -- auto | none | pack (pack graph neighbourhoods into pages)
    page_size_hint=0,     -- page size assumed by reorder (0 = current page size)
    threads=4, ef_search=64, beam=4, verbose=0);

INSERT INTO v(rowid, embedding) VALUES (?, ?);  -- float32 blob (dim*4 bytes) or JSON text '[...]'
INSERT INTO v(v) VALUES ('build');              -- train PQ on everything inserted so far, build the graph
INSERT INTO v(v) VALUES ('finalize');           -- store page hints; run last (after any VACUUM)

SELECT rowid, distance FROM v
 WHERE embedding MATCH ?1 AND k = 10            -- or LIMIT 10
   [AND ef = 64]        -- candidate list size
   [AND beam = 4]       -- W: nodes expanded (fetched in parallel) per round
   [AND rerank = 0|1|2|3]
   [AND exact = 1|2]    -- brute force: 1 = stored vectors, 2 = PQ codes
   [AND trace = 1];     -- stats JSON includes the page numbers fetched per round
-- the hidden column `stats` returns per-query JSON:
--   rounds, pages, bytes, expanded, dist, entry_dist, rerank, fallback, setup_rounds, setup_pages, ms, ...

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

(RESULTS_PLACEHOLDER)
