# WebAssembly runtime and build notes

This directory holds the build toolchain and the browser/Node runtime: SQLite compiled to WebAssembly with FTS5 and the project's extensions linked in, reading a read-only database over HTTP range requests through a VFS called **httpvfs**. The same VFS also builds natively, so extensions can be developed and their network round trips counted without a browser.

## Layout

| Path | Contents |
|---|---|
| `Makefile` (repo root) | Includes `wasm/build.mk`. Default target: `native`. |
| `wasm/build.mk` | All build rules: SQLite, native CLI and libraries, extensions, WASM variants, tests. |
| `wasm/scripts/fetch-sqlite.sh` | Fetches the pinned SQLite source and generates the amalgamation, checking SHA-256. |
| `wasm/src/httpvfs.c` | The VFS: block cache, readahead, batching, speculation, request log, counters. |
| `wasm/src/httpvfs.h` | Header for extensions: `httpvfs_prefetch()`, `httpvfs_prefetch_pages()`, speculation, stats. |
| `wasm/src/httpvfs-pre.js` | JavaScript backend of the VFS (`--pre-js`): parallel `fetch()` with `Range`. |
| `wasm/src/hv_init.c` | `SQLITE_EXTRA_INIT` hook: registers the VFS and every static extension. |
| `wasm/demo_ext.c` | Placeholder extension: `hello()`, `httpvfs_warm()`, `httpvfs_stats()`. |
| `wasm/pkg/` | The npm-shaped package: `index.mjs` (API), `sync-fetch*.mjs`, `dist/` (build output, git-ignored). |
| `wasm/test/` | Native (Python), Node and Chromium tests, a minimal range server, shared helpers. |
| `wasm/bench/overhead.mjs` | CPU overhead of the WASM builds against native SQLite. |

Build output goes to `build/sqlite`, `build/native`, `build/wasm` (git-ignored) and `wasm/pkg/dist`.

## Building

```sh
# once: Emscripten, outside the repository
git clone https://github.com/emscripten-core/emsdk.git /home/user/emsdk
cd /home/user/emsdk && ./emsdk install 6.0.10 && ./emsdk activate 6.0.10

make sqlite        # build/sqlite: amalgamation + shell.c, SHA-256 checked
make native        # build/native: sqlite3 CLI, libsqlite3.so.0, httpvfs.so, demo_ext.so, ext/<name>.so
make wasm          # wasm/pkg/dist: asyncify, jspi and sync variants
make test          # test-native, test-node, test-browser
make print-exts    # which ext/* directories are being linked
```

The Makefile finds `emcc` on `PATH` or in `$(EMSDK)/upstream/emscripten` (default `EMSDK=/home/user/emsdk`), so sourcing `emsdk_env.sh` is not required. If you do source it, note that it puts `/home/user/emsdk` on `PATH`, whose `node/` directory shadows `node` for `make`; `build.mk` resolves `NODE` through the shell to avoid this.

A clean `make native` takes about 75 s and `make wasm` about 2 minutes on the 4-core container, most of it compiling `sqlite3.c` (twice natively: once for the CLI, once as a shared library).

### Pinned versions

| Component | Version | Source |
|---|---|---|
| SQLite | 3.53.4 (source id `2026-07-24 19:02:57 bf7c7f30…bcc`) | `github.com/sqlite/sqlite` tag `version-3.53.4`, commit `b09c88c14082339b66c7b7158d609a771e64ca69`; amalgamation generated with SQLite's bundled JimTcl (no system Tcl) |
| `sqlite3.c` | SHA-256 `b1dd5d74ec7f29055a6684fa06fb3c2f6821c87dd38f9a458dfd2e8a1db28189` | generated |
| `sqlite3.h` | SHA-256 `919e7f2e8ed1d8f56ac17b412b8971c76aa5d1a879752cc6058f75e7d5910e1d` | generated |
| `sqlite3ext.h` | SHA-256 `ac9645e5c9ff0cf176efdd6e75cb5e98f46295d38e02db5c4d208826a39ab4be` | generated |
| `shell.c` | SHA-256 `8011ed018aa12969f93573b7bb1eae2d939d64d0f451b297ff847a0211c85179` | generated |
| Emscripten | 6.0.10 (clang 24.0.0git, releases hash `666337b5…ead64`) | emsdk commit `e566f7bd…3c93145` |
| Node | 22.22.2 | preinstalled |
| Playwright / Chromium | 1.56.1 / HeadlessChrome 141.0.7390.37 | preinstalled (`/opt/pw-browsers`) |

sqlite.org is blocked from the container, so the source comes from the official GitHub mirror. The amalgamation is deterministic: a fresh clone reproduces all four hashes. (The `better-sqlite3` npm package also bundles 3.53.4, but its amalgamation is generated with its own options and carries a patch, so it was not used.)

### SQLite options

All builds: `SQLITE_ENABLE_FTS5`, `ENABLE_DBSTAT_VTAB`, `ENABLE_DBPAGE_VTAB` (page-level inspection), `ENABLE_MATH_FUNCTIONS`, `DQS=0`, `DEFAULT_MEMSTATUS=0`, `LIKE_DOESNT_MATCH_BLOBS`. The WASM build adds `THREADSAFE=0`, `OMIT_LOAD_EXTENSION`, `OMIT_DEPRECATED`, `OMIT_SHARED_CACHE`, `TEMP_STORE=3` and is compiled with `-O2 -msimd128`, so extensions can use WebAssembly SIMD (supported by Chrome, Firefox and Safari 16.4+). The native shared library keeps the deprecated and shared-cache APIs because Python's `_sqlite3` module links against them.

## Extensions: how they are linked

Each directory `ext/<dir>/` is one extension once one of its `.c` files defines an entry point, found by searching the sources for `int sqlite3_<x>_init(`. Directories without one are skipped, so a half-written extension does not break the build. `EXT_DIRS="…"` overrides the list, and `make print-exts` shows what was found. All `.c` files of a directory are compiled together. Currently: `ext/dense` (`sqlite3_denseann_init`, virtual table `dense_ann`), `ext/late` (`sqlite3_late_init`, virtual table `late_plaid`) and `ext/fts5rank` (`sqlite3_fts5rank_init`, FTS5 ranking functions `bm25c()` and `bm25dl()` that read no per-document lengths; see `docs/fts5-httpvfs.md`).

- **Native loadable:** `build/native/ext/<x>.so`, named after the entry point so that SQLite's default entry-point rule finds it (`denseann.so`, `late.so`).
- **Static (WASM and the native CLI):** compiled with `-DSQLITE_CORE`, which turns `SQLITE_EXTENSION_INIT1/2` into no-ops. The Makefile generates `ext_registry.c`, which calls `sqlite3_auto_extension()` for `demo_ext` and every `ext/*` entry point; `hv_init.c` runs it from `SQLITE_EXTRA_INIT`, at the end of `sqlite3_initialize()`. Every connection therefore gets every extension without any JavaScript involvement.
- Because the extensions are linked into one module, non-entry symbols should be `static` or prefixed.

This was checked with a two-file test extension (native CLI, Python `load_extension`, and the WASM module). With the current tree, the WASM module's `pragma_module_list` includes `dense_ann`, `late_plaid`, `fts5`, `dbstat` and `sqlite_dbpage`, and all tests pass with both extensions linked in.

### Python

This Python (3.11, system SQLite 3.45.1) has `enable_load_extension`, and its `_sqlite3` module links `libsqlite3.so.0` dynamically, so

```sh
LD_LIBRARY_PATH=build/native python3 …   # sqlite3.sqlite_version == '3.53.4', FTS5 on
```

runs Python on the project's SQLite build. Extensions built against 3.53 headers that call only APIs present in 3.45 also load into the system SQLite. `pysqlite3` or `apsw` would be the fallback on a Python without `enable_load_extension`.

To use the VFS natively, load `build/native/httpvfs.so` once (it registers the VFS process-wide) and open databases with a URI:

```python
boot = sqlite3.connect(":memory:"); boot.enable_load_extension(True)
boot.load_extension("build/native/httpvfs")
db = sqlite3.connect("file:/path/db.sqlite?vfs=httpvfs&latency_ms=80&cache_kb=4096", uri=True)
```

The native backend reads the local file with `pread()` and sleeps `latency_ms` once per round, so round counts, request counts and bytes are the same as in the browser. The CLI has everything built in: `build/native/sqlite3 'file:db?vfs=httpvfs'`, then `PRAGMA httpvfs_stats;` or `PRAGMA httpvfs_reset=cache;`.

## The parallel-fetch mechanism

The research question for the ANN extensions is how many *sequential* round trips a query needs; they need to fetch, say, 8 graph nodes or 16 IVF lists at once. SQLite's VFS interface is synchronous, and a synchronous XHR in a worker fetches one range at a time. Options considered:

| Option | Parallel fetches | Portability | Cost |
|---|---|---|---|
| Sync XHR in a worker (sql.js-httpvfs) | No: strictly one request per blocking read | Everywhere | None, but rules out the research goal |
| Sync XHR with multi-range `Range: bytes=a-b,c-d` | One request, many ranges | Needs server `multipart/byteranges`; S3, R2 and most CDNs do not support it | Parsing multipart |
| **Asyncify** (wa-sqlite style), `EM_ASYNC_JS` + `Promise.all(fetch…)` | Yes | Every browser, Node; main thread or worker; no special headers | +70 % code size, CPU overhead on instrumented code |
| **JSPI** (same C and JS, `-sJSPI`) | Yes | Chromium 137+ only today; Firefox and Safari not yet; Node 22 lacks the current API | None measurable in size |
| **Sync build + worker pool with SharedArrayBuffer/Atomics.wait** | Yes | Node; browsers only in a Worker of a cross-origin-isolated page (COOP/COEP) | Headers break some embeddings; GitHub Pages cannot set them |
| Sync VFS + "prefetch hook" without stack switching | Only if the hook can block on parallel fetches, which is the previous row | | |

**Choice.** The C VFS does not care: it calls one backend function, "fetch these N ranges, return when all are in", and one such call is one *round*. That function is implemented three ways, from the same objects, as three builds:

- `sqlite-httpvfs` (**Asyncify**, the portable default): works in every browser and Node, on the main thread or in a worker, with no COOP/COEP.
- `sqlite-httpvfs-jspi` (**JSPI**): same behaviour, no Asyncify size or speed cost; selected automatically where `WebAssembly.Suspending` exists (`variant: 'auto'`, the default).
- `sqlite-httpvfs-sync` (**SharedArrayBuffer + Atomics.wait**): no stack switching at all; a fetch worker does the parallel `fetch()`es and wakes the SQLite thread. Needs a Worker and cross-origin isolation in browsers; works anywhere in Node (the server must be in another thread or process).

All three pass the same tests in Chromium, and asyncify and sync pass them in Node. The extension-facing API is identical in all of them and natively.

### What extensions call (`wasm/src/httpvfs.h`)

Everything goes through `sqlite3_file_control()` with private opcodes, so there is no link-time dependency: the calls work in a loadable `.so`, in the static WASM build, and return `SQLITE_NOTFOUND` harmlessly on an ordinary database.

```c
#include "httpvfs.h"
int httpvfs_prefetch(sqlite3 *db, const sqlite3_int64 *offsets, const int *lengths, int n);
int httpvfs_prefetch_pages(sqlite3 *db, const unsigned int *pgnos, int n);
int httpvfs_speculate_begin(sqlite3 *db);
int httpvfs_speculate_end(sqlite3 *db);     /* > 0: blocks fetched in one round */
int httpvfs_stats(sqlite3 *db, HttpvfsStats *out);
int httpvfs_reset_stats(sqlite3 *db);
int httpvfs_release(sqlite3 *db);           /* end of statement: unpin, shrink to budget */
```

`httpvfs_prefetch` fetches every uncached block of the given ranges in one parallel round; later `xRead`s hit the cache. The VFS also accepts `DENSE_ANN_FCNTL_PREFETCH` (`0x44414e01`) from `ext/dense/rawpage.h` with the same meaning, so that extension works unchanged.

**Speculative batching** is for reads whose byte offsets the extension cannot know, such as rows of an ordinary table found through a B-tree:

```c
do {
  httpvfs_speculate_begin(db);
  for (i = 0; i < n; i++) { bind(ids[i]); while (sqlite3_step(st) == SQLITE_ROW) {} sqlite3_reset(st); }
} while (httpvfs_speculate_end(db) > 0);
/* now run the lookups for real: all cache hits */
```

While speculating, uncached blocks read as zeros and are only recorded; SQLite may return garbage or `SQLITE_CORRUPT`, which the caller ignores. `httpvfs_speculate_end()` fetches all recorded blocks in one round and calls `sqlite3_db_release_memory()` so the zero-filled pages leave SQLite's own page cache. Each pass reveals one more B-tree level for all lookups at once, so N lookups cost about one round per uncached level instead of N per level. Block 0 (header and page 1) is never speculated. `demo_ext.c`'s `httpvfs_warm(sql, 'id,id,…')` is a working example and doubles as a SQL-level helper. Caveat: statements stepped during a pass must be reset before `httpvfs_speculate_end()`, and the schema should already be loaded (any earlier prepare does it).

## The VFS

- **Block cache.** LRU over fixed-size blocks (default 4096 bytes; set it to the database page size or a multiple), with the blocks of the current prefetch batch pinned; see [Block cache: pinned batches and the default budget](#block-cache-pinned-batches-and-the-default-budget). The budget comes from `pageCacheBytes` (JS) or `cache_kb` (URI); by default (`'auto'`) it is 1/64 of the database file, clamped to 4–64 MiB. A read larger than the cache still works (copied straight from the fetched buffers). SQLite's own page cache sits on top (the JS API sets `PRAGMA cache_size` to 2 MiB by default).
- **Readahead** (as in sql.js-httpvfs): when a read starts where the previous one ended, the miss is extended by 1, 2, 4, … blocks up to `readaheadBytes` (default 1 MiB, 0 disables). A full scan of the `docs` table in the 2.2 MB test database takes 12 requests (2.1 MB) instead of 336 (1.4 MB) without readahead; the extra bytes are readahead running past the end of the table.
- **Coalescing.** Adjacent missing blocks in a batch become one request; `coalesceGapBytes` also merges ranges separated by small gaps. A per-round request budget (`maxRequests`, 6 over HTTP/1.1 in browsers) merges or groups the rest; see [Request budget per round](#request-budget-per-round-http11).
- **Size discovery.** The first request (block 0, which SQLite reads first anyway) learns the file size from `Content-Range`, so opening costs one round.
- **Read-only.** `SQLITE_IOCAP_IMMUTABLE`, writes return `SQLITE_READONLY`; a WAL-mode header is presented as rollback mode. Non-main files (temp, journals) go to the default VFS.
- **Instrumentation.** Every request is logged as `{offset, length, round, tStart, tEnd}` (ms, `performance.now()` clock of the calling thread). `round` is the number of the backend call; the requests of one batch share a round. Counters: `requests`, `bytes`, `rounds`, `reads`, `cacheHits`, `cacheMisses`, `prefetchCalls`, `prefetchBlocks`, `specMisses`, `netMs` (wall time blocked on the network), plus `fileSize`, `blockSize`, `cacheBlocks` (the budget), `cachedBlocks`, `cacheMaxBlocks` (the hard limit), `pinnedBlocks`, `peakBlocks` (most blocks held at once since the last reset) and `pinEvictions`.
- **URI parameters** (native and WASM): `cache_kb` (a number or `auto`), `cache_min_kb`, `cache_max_kb`, `block`, `readahead_kb`, `gap_kb`, `log_max`, `max_req`, `rtt_ms`, `bw_kbps`, `multipart`, `max_parts`, `net_auto`, and natively `latency_ms`.

Chunked databases (sql.js-httpvfs "chunked" mode) are not implemented; the backend interface would take them without changes to the C side.

### Block cache: pinned batches and the default budget

**Problem.** At words-1m one IVF query at nprobe 128 prefetches about 870 pages (3.5 MB) in one batch. The VFS used to cap a batch at half the cache and insert it into a plain LRU. With the old default of 4 MiB the batch was cut, and part of it was evicted by the rest of the batch before SQLite read it. Those pages were then fetched again one at a time: 104 rounds (median, warm) for the 2 rounds the extension issued, and 1,148 for nprobe 512. A 64 MiB cache fixed nprobe 128 but not nprobe 512 (13 MB per query), which evicted its own batch once the cache was full (median 11–16 rounds).

**Pinning.** The cache now has two lists: the LRU list and a list of *pinned* blocks.
- Every block that an extension asks for in `httpvfs_prefetch`, `httpvfs_prefetch_pages`, `DENSE_ANN_FCNTL_PREFETCH` or a speculation pass (`httpvfs_speculate_end`) is pinned, including blocks of the batch that were already cached. Blocks over-fetched by coalescing, readahead blocks and on-demand misses are not pinned.
- A pinned block is unpinned when it is first read (it becomes the most recently used LRU block), when the next batch starts (the extensions read a batch before announcing the next one), or at the end of the statement.
- Eviction takes the LRU tail. While blocks are pinned, the cache may exceed its budget to hold them, up to a hard limit (`cache_max_kb` / `pageCacheMaxBytes`, default 4 × the budget). The unpinned part then keeps a floor of a quarter of the budget, so the pages read most recently (B-tree interior pages, the previous batch) survive a large batch. Only when the cache would exceed the hard limit with nothing unpinned left is the oldest pinned block evicted (`pinEvictions`). A batch larger than the hard limit is cut there, and the rest is read on demand as before.
- As pinned blocks are read the cache trims back towards the budget. At the end of a statement (`httpvfs_release()`, `PRAGMA httpvfs_release`, file control `HTTPVFS_FCNTL_RELEASE`) the remaining pins are dropped, the cache is trimmed to its budget, and the memory above the budget is returned: blocks are stored in chunks of 256, and the blocks in chunks past the budget are moved down before those chunks are freed. The JavaScript API calls it after every statement. Native users who never call it get the same round counts; only unread pinned blocks linger until the next batch.
- The per-round cap of half the cache is gone. The coalescing planner may still over-fetch at most half the budget beyond the requested blocks, because those blocks are not pinned.

With an immutable database SQLite takes no locks, so the VFS sees no transaction boundaries: the end of a statement has to be signalled, and first read is the pin's natural end.

**Default budget.** `pageCacheBytes: 'auto'` (the default; `cache_kb=auto` or no `cache_kb` in a URI) sets the budget to clamp(file size / 64, 4 MiB, 64 MiB) once the size is known (after the first request). The reasoning:
- at 10k documents (75–100 MB) this is the old 4 MiB, which already holds a query's working set there;
- at 1M it gives 11 MiB (FTS5, 0.7 GB), 24 MiB (late, 1.6 GB), 44 MiB (IVF, 3.0 GB, of which 1 GB is live data) and 61 MiB (graph, 4.1 GB). That is enough for one query's batch and for the pages that successive queries share. With pinning, the batch no longer has to fit anyway;
- 64 MiB (plus up to 4 × that, briefly, for a pinned batch) is a reasonable ceiling for a browser tab.

`pageCacheMinBytes` (`cache_min_kb`) raises the floor. The web client (`web/lib/index.mjs`) uses 'auto' with a 16 MiB floor, its previous fixed size, because its single combined database serves four indexes. An explicit `pageCacheBytes` still sets a fixed budget.

**Results.** words-1m, WASM (Asyncify) against the unshaped range server, a warm session of 40 queries per configuration after one warm-up query; rounds per query (median / mean), requests and KB per query (mean). *Before* is the previous VFS with its 4 MiB default; *after* is this VFS with the automatic budget.

| Configuration | Extension rounds | Before (4 MiB) | After (auto) | Before, 64 MiB | After, 64 MiB | After, 4 MiB |
|---|---|---|---|---|---|---|
| IVF nprobe 128 | 2 | 104 / 118 · 311 req · 3,771 KB | 2 / 2.0 · 167 req · 1,892 KB | 2 / 2.35 · 158 req · 1,697 KB | 2 / 2.0 · 158 req · 1,693 KB | 2 / 2.0 · 226 req · 3,363 KB |
| IVF nprobe 512 | 2 | 1,148 / 1,158 · 1,338 req · 16,473 KB | 2 / 2.0 · 281 req · 5,046 KB | 11.5 / 19.8 · 220 req · 3,051 KB | 2 / 2.0 · 202 req · 2,881 KB | 2 / 2.0 · 515 req · 12,349 KB |
| graph ef 64 | 9.9 | 10 / 9.9 · 114 req · 455 KB | 10 / 9.85 · 109 req · 437 KB | 10 / 9.85 · 109 req · 437 KB | 10 / 9.85 · 109 req · 437 KB | 10 / 9.88 · 114 req · 455 KB |
| warp nprobe 8 | 1 | 2 / 2.9 · 39 req · 872 KB | 1 / 1.0 · 31 req · 723 KB | 1 / 1.0 · 31 req · 723 KB | 1 / 1.0 · 31 req · 723 KB | 1 / 1.0 · 37 req · 855 KB |
| warp nprobe 32 | 1 | 174 / 190 · 265 req · 4,120 KB | 1 / 1.0 · 107 req · 2,735 KB | 1 / 2.9 · 90 req · 2,302 KB | 1 / 1.0 · 88 req · 2,278 KB | 1 / 1.48 · 133 req · 3,408 KB |

- Every configuration now takes exactly the rounds its extension issues, whatever the budget. Over 200 queries at 64 MiB, IVF nprobe 512 goes from a median of 11 rounds (mean 18.8) to 2.
- The larger automatic budget also saves bytes across queries (IVF nprobe 128: 1.9 MB against 3.4 MB per query with a 4 MiB budget).
- Cold queries (a new connection per query, 10 queries): IVF nprobe 128 goes from 140 to 5 rounds, nprobe 512 from 1,152 to 5, warp nprobe 8 from 14 to 5, warp nprobe 32 from 287 to 5; graph stays at 13.
- The top 10 are identical in every run. words-10k and llm-10k (graph, IVF nprobe 64, warp nprobe 8, FTS5 bm25c) keep their 4 MiB budget and give the same or slightly fewer rounds, requests and bytes (for example words-10k IVF 2.15 → 1.70 rounds, llm-10k IVF 1.90 → 1.63), because blocks of a batch that were already cached can no longer be evicted before they are read. FTS5 at 1M is unchanged (7.55 rounds).
- Tests: `test_pinned_batch_larger_than_cache` (native: a 538-block batch against a 16-block budget arrives in one round, a full scan then costs no further round, and `PRAGMA httpvfs_release` shrinks the cache back), `test_pinning_random_workload` (random batches, lookups, scans and releases against small budgets and hard limits; the results always match), `test_auto_cache_budget`, and the Node tests for the same batch inside one statement and for the default budget.

## JavaScript API (`wasm/pkg/index.mjs`)

```js
import { open } from './wasm/pkg/index.mjs';
const db = await open('https://host/db.sqlite', {
  pageCacheBytes: 'auto', blockSize: 4096, readaheadBytes: 1 << 20,
  coalesceGapBytes: 0, maxParallel: undefined, variant: 'auto',
  maxRequests: 'auto', multipart: false,   // request budget per round, see below
});
await db.query('SELECT rowid FROM t_fts WHERE t_fts MATCH ? LIMIT 10', ['word']);  // [{rowid: …}]
await db.queryRaw(sql, params);   // {columns, rows: [[…]]}
await db.exec(sql);               // several statements, no results
db.stats(); db.log(); db.netState();   await db.setNetOptions({ multipart: true });
await db.resetStats({ clearCache: true });   // cold start for the next query
await db.close();
```

Parameters can be an array or an object of named parameters; integers come back as `Number` when safe, else `BigInt`; blobs as `Uint8Array`. Calls into the module are serialised (Asyncify cannot run two suspended calls at once), so concurrent `query()` calls are safe and run in order. Fetch errors surface in the thrown `SqliteError` message (for example `unable to open database file (httpvfs: HTTP 404 …)`).

## Measurements

All measured on the shared 4-core container while a long LLM job kept the load average around 17–18, so timings are noisy; the round counts are exact.

### Binary size (FTS5, httpvfs, demo_ext; `-O2 -msimd128`)

| Build | `.wasm` raw | gzip -9 | JS glue |
|---|---|---|---|
| Asyncify | 2,164,554 | 763,788 | 83 KB |
| JSPI | 1,258,349 | 511,105 | 78 KB |
| Sync | 1,258,338 | 511,123 | 76 KB |

With `ext/dense` and `ext/late` linked in as well (current tree): Asyncify 2,372,896 / 846,460; JSPI and sync 1,404,490 / 574,237. Asyncify costs about +72 % raw and +50 % gzipped, because SQLite reaches `xRead` through function pointers, so almost every function that makes an indirect call is instrumented. Linking with `-Os` instead of `-O2` saves only 2 %. `ASYNCIFY_IGNORE_INDIRECT` with an explicit list would shrink it but is fragile with extensions calling back into SQLite (virtual tables and SQL functions are indirect calls), so it was not used.

### Round trips (16 cold lookups by rowid, 20 ms server latency)

| Where | Sequential | With `httpvfs_warm` |
|---|---|---|
| Native (10 ms simulated latency) | 18 rounds, 189 ms | 3 rounds, 31 ms |
| Node, asyncify | 18 rounds, 411 ms | 3 rounds, 83 ms; the 16-request round took 33 ms |
| Node, sync | 18 rounds, 393 ms | 3 rounds, 92 ms |
| Chromium worker, asyncify / jspi / sync | 18 rounds, 430–560 ms | 3 rounds, 155–210 ms |

In Chromium the 16 parallel requests go out in waves of six: the netsim server speaks HTTP/1.1, and browsers open at most six connections per host. Over HTTP/2 they would all be in flight at once. Node's `fetch` (undici) does not have that limit.

### CPU overhead against native (warm cache, median of 200 runs, ms)

Native is Python's `sqlite3` on the same SQLite 3.53.4 build (default VFS). WASM numbers go through the JS API, so they include `ccall`, one promise per `sqlite3_step` and row conversion. Two runs; each cell gives the range.

| Query | Native | Node asyncify | Node sync | Chromium asyncify | Chromium JSPI | Chromium sync |
|---|---|---|---|---|---|---|
| FTS5 term, top 10 by rank | 0.045 | 0.26 (5.6×) | 0.14–0.15 (3.1–3.2×) | 0.26–0.29 (5.6–6.3×) | 0.15–0.16 (3.2–3.5×) | 0.16 (3.4×) |
| FTS5 prefix `a*`, count | 7.6–7.7 | 10.4–14.0 | 9.2–9.8 | 10.0–13.1 | 6.1–10.2 | 9.1–9.4 |
| FTS5 OR + join, top 5 | 0.06 | 0.21 | 0.10 | 0.20–0.23 | 0.12 | 0.11–0.21 |
| rowid lookup | 0.004 | 0.010–0.013 | 0.006–0.007 | 0.010–0.015 | 0.010 | 0.005–0.010 |

For short queries the cost is a fixed 0.1–0.25 ms per query in the JS wrapper and stack switching. For compute-bound work (the prefix query) Asyncify costs roughly 1.3–1.8× native, and JSPI and the sync build roughly 1.0–1.3×. Both are small next to a single mobile round trip (70–300 ms), which is why round counts, not CPU, are the thing to optimise. Raw results: `results/wasm-overhead.json` (rerun with `node wasm/bench/overhead.mjs --out …` on an idle machine for cleaner numbers).

## Request budget per round (HTTP/1.1)

Over HTTP/1.1 a browser runs at most six requests per host at once. A round of the dense graph (16 nodes, 7–8 rounds), of IVF (16–128 lists) or of warp (up to about 100 posting and document ranges) therefore becomes several waves of latency, and on `4g` this cost more than the dependent rounds themselves (web/NOTES.md). The VFS now plans every round against a **request budget** *C*. Extension code is unchanged: the planning happens in `hvFetchBlocks`, below every prefetch, speculation and `xRead` miss.

### Design

- **Input.** A round's uncached blocks, sorted, after adjacent blocks and `gap_kb` gaps have been merged into *m* ranges. If *m* ≤ *C* nothing changes.
- **Coalescing** (the default when *m* > *C*). `httpvfs_plan()` chooses *g* contiguous requests. It minimises the estimated round time
  *T*(*g*) = ⌈*g*/*C*⌉ · RTT + bytes(*g*) / bandwidth,
  where bytes(*g*) is the smallest total span of *g* requests, obtained by cutting at the *g*−1 largest gaps. Between multiples of *C* the number of waves is constant and more requests fetch fewer bytes, so only *g* = *m* and *g* = *C*, 2*C*, … can be optimal. All candidates are evaluated exactly, in O(*m* log *m*); a test checks the result against brute force over every set of cuts. The total span may exceed the requested blocks by at most half the cache budget. Blocks fetched in the gaps are real data and go into the cache, unpinned. They are inserted *before* the requested blocks, and the requested blocks of a prefetch batch are pinned, so a round never evicts its own blocks.
- **Only RTT × bandwidth matters.** The decision depends only on the bandwidth-delay product, the number of bytes worth one extra wave. That product is similar across the profiles: 167 KB on `4g`, 105 KB on `lte`, 101 KB on `slow-4g` and 125 KB on `wifi`. The defaults (100 ms, 10 Mbit/s: 125 KB) are therefore close everywhere. Scaling the RTT given to the planner by 0.5–2× changed the simulated warm `4g` p50 by less than 1 % for graph and IVF and by at most 7 % for warp (2× was 2 % *better*: the model slightly underestimates a wave). Too-large values cost a lot, because the planner then over-fetches: 8× costs 26 % (graph) to 79 % (IVF).
- **Online estimate** (`netAutoEstimate`, on by default). The VFS keeps the last 64 rounds that ran without client-side queueing (at most *C* requests), as pairs of bytes and wall time. The bandwidth estimate is the slope between the medians of the smaller and larger halves, used only when their sizes differ by 2× or more. The RTT estimate is a low quartile of *T* − *B*/bandwidth over the smaller half. In Chromium on `4g,h1` it converged to RTT 167–169 ms (true value 165) and 4.8–7.4 Mbit/s (true 8.1); against a localhost server it converged to 1.3 ms.
- **Multi-range requests** (`multipart: true`, opt-in). The ranges are split in offset order into min(*C*, *m*) requests of similar byte size, each with at most `maxRangesPerRequest` (100) ranges, sent as `Range: bytes=a-b,c-d,…`. Nothing is over-fetched. The `multipart/byteranges` parser (`wasm/pkg/multipart.mjs`) takes each part's length from its `Content-Range`, so bodies that contain the boundary string are safe. It accepts parts in any order, and parts the server merged. The same module is compiled into the glue as a `--pre-js`, with its `export`s stripped, and is imported by the sync fetch worker.
- **Fallback.** Until a URL is known to support multi-range requests, the first multi-range request of a round is sent alone:
  - a `200` reply is aborted after its headers, so the whole file is never downloaded;
  - on a `416`/`4xx` reply, a single range that does not cover everything, or a multipart reply that lacks some ranges, the missing ranges are fetched singly.

  The URL is then marked as unsupported, shared by every connection in the module. The C side is told through a flag and switches to coalescing. The detection costs one extra round trip, once per URL.
- **Default *C*** (`maxRequests: 'auto'`). After the open request, the Resource Timing entry's `nextHopProtocol` gives the protocol. The default is 6 for `http/1.x` or when the protocol is not visible (for example, sync-variant fetches run in a nested worker), 100 for `h2`/`h3`, and 0 (off) in Node, whose `fetch` has no per-host connection limit. The benchmark matrix, which runs in Node, is therefore unchanged. In the web demo Chromium detected `http/1.1` and used 6.
- **API.** Options of `open()` and `openIndex()`: `maxRequests`, `rttMs`, `bandwidthKbps`, `netAutoEstimate`, `multipart`, `maxRangesPerRequest`. Runtime calls: `db.setNetOptions({...})` and `db.netState()`, which returns the budget, the estimates, over-fetched bytes, planned rounds, multi-range requests and the fallback flag. URI parameters: `max_req`, `rtt_ms`, `bw_kbps`, `multipart`, `max_parts`, `net_auto`. `PRAGMA httpvfs_stats` shows the counters. The demo page accepts `?maxRequests=0|6|auto&multipart=1`. The request log has one entry per range; `req` numbers the request within its round, so the ranges of one multi-range request share it. `bench/matrix.mjs` logs record it, and `bench/matrix.py` merges those ranges back into one read.
- **Cross-origin caveat.** A multi-range `Range` header is not CORS-safelisted (only a single `bytes=a-b` is), so a cross-origin page sends one preflight per URL (cached for `Access-Control-Max-Age`). Same-origin hosting avoids it.

### Evaluation: re-simulated matrix traces

`bench/coalesce_eval.py words-10k` re-simulates the recorded queries of `build/matrix/words-10k.trace.jsonl` round by round. It uses `httpvfs_plan` from `build/native/httpvfs.so` through ctypes, so the planner is the C code itself. The simulator is netsim's, with the recorded CPU gaps kept. It covers 500 warm queries and 200 cold queries per configuration, on four configurations: FTS5 `fts-bm25`, dense graph `graph-ef64`, dense IVF `ivf-np64`, and late `warp-np8-rr64`. Each multipart part is charged 100 bytes of framing.

Two effects are left out, so coalescing is evaluated slightly pessimistically:
- the over-fetched blocks are not fed back into the cache;
- the fixed-default and oracle RTT and bandwidth are used in place of the online estimate.

Columns:
- *h1* is `<profile>,h1` (six requests in service, FIFO); *h2* is `<profile>,h2` (100 streams).
- *coalesce* uses the profile's RTT and bandwidth (with the defaults of 100 ms and 10 Mbit/s the results are within ±5 %; see `results/coalesce/words-10k.json`).
- *h2 coalesce C=6* is HTTP/2 mistaken for HTTP/1.1.
- *h2 coalesce C=100*, the HTTP/2 default, equals *h2 baseline* everywhere, because no round has more than 100 requests. It is omitted.


**Warm** (within a session; p50 / p95 ms; requests per query; over-fetch per query with coalescing):

| Profile | System | h1 baseline | h1 coalesce | h1 multipart | h2 baseline | h2, C = 6 | Requests h1: base / coal. / multi | Over-fetch |
|---|---|---|---|---|---|---|---|---|
| 4g | FTS5 | 0 / 169 | 0 / 169 | 0 / 169 | 0 / 169 | 0 / 169 | 0 / 0 / 0 | 0 KB |
| 4g | dense graph | 2600 / 3251 | 2582 / 3119 | 1429 / 1809 | 1423 / 1802 | 1444 / 1935 | 65 / 64 / 35 | 40 KB |
| 4g | dense IVF | 1128 / 1771 | 946 / 1467 | 471 / 974 | 468 / 971 | 550 / 1086 | 31 / 23 / 7 | 90 KB |
| 4g | late warp | 2086 / 2602 | 1915 / 2325 | 628 / 954 | 622 / 948 | 735 / 1093 | 61 / 57 / 12 | 79 KB |
| lte | dense graph | 1176 / 1450 | 1168 / 1412 | 676 / 847 | 671 / 843 | 679 / 877 | 65 / 65 / 35 | 17 KB |
| lte | dense IVF | 513 / 782 | 436 / 663 | 236 / 451 | 234 / 449 | 267 / 479 | 31 / 25 / 7 | 50 KB |
| lte | late warp | 954 / 1179 | 923 / 1092 | 342 / 480 | 338 / 476 | 375 / 510 | 61 / 58 / 12 | 39 KB |
| slow-4g | dense graph | 9503 / 11698 | 9426 / 11407 | 5477 / 6861 | 5440 / 6823 | 5486 / 7106 | 65 / 65 / 35 | 16 KB |
| slow-4g | dense IVF | 4145 / 6301 | 3531 / 5411 | 1917 / 3649 | 1899 / 3632 | 2172 / 3889 | 31 / 25 / 7 | 48 KB |
| slow-4g | late warp | 7691 / 9492 | 7465 / 8796 | 2781 / 3881 | 2745 / 3847 | 3026 / 4142 | 61 / 58 / 12 | 37 KB |
| wifi | dense graph | 328 / 406 | 326 / 394 | 185 / 233 | 184 / 232 | 186 / 242 | 65 / 64 / 35 | 22 KB |
| wifi | dense IVF | 144 / 220 | 121 / 185 | 64 / 125 | 63 / 124 | 73 / 134 | 31 / 25 / 7 | 64 KB |
| wifi | late warp | 268 / 332 | 251 / 307 | 92 / 131 | 91 / 130 | 103 / 140 | 61 / 58 / 12 | 51 KB |

**Cold** (new connection per query; p50 / p95 ms; requests per query; over-fetch per query with coalescing):

| Profile | System | h1 baseline | h1 coalesce | h1 multipart | h2 baseline | h2, C = 6 | Requests h1: base / coal. / multi | Over-fetch |
|---|---|---|---|---|---|---|---|---|
| 4g | FTS5 | 2538 / 2880 | 2538 / 2880 | 2538 / 2880 | 2538 / 2880 | 2538 / 2880 | 15 / 15 / 15 | 0 KB |
| 4g | dense graph | 3930 / 4477 | 3929 / 4461 | 2453 / 2963 | 2446 / 2956 | 2448 / 2956 | 83 / 83 / 40 | 7 KB |
| 4g | dense IVF | 4152 / 4409 | 3181 / 3338 | 2103 / 2147 | 2093 / 2138 | 2683 / 2825 | 94 / 42 / 16 | 580 KB |
| 4g | late warp | 5760 / 6286 | 5328 / 5735 | 3347 / 3454 | 3337 / 3443 | 3518 / 3705 | 105 / 93 / 22 | 166 KB |
| lte | FTS5 | 1093 / 1241 | 1093 / 1241 | 1093 / 1241 | 1093 / 1241 | 1093 / 1241 | 15 / 15 / 15 | 0 KB |
| lte | dense graph | 1826 / 2071 | 1825 / 2065 | 1202 / 1421 | 1197 / 1417 | 1198 / 1417 | 83 / 83 / 40 | 2 KB |
| lte | dense IVF | 1976 / 2105 | 1641 / 1747 | 1172 / 1202 | 1166 / 1196 | 1409 / 1479 | 94 / 53 / 16 | 347 KB |
| lte | late warp | 2981 / 3213 | 2844 / 3043 | 1973 / 2046 | 1966 / 2037 | 2032 / 2136 | 105 / 96 / 22 | 94 KB |
| slow-4g | FTS5 | 8781 / 9974 | 8781 / 9974 | 8781 / 9974 | 8781 / 9974 | 8781 / 9974 | 15 / 15 / 15 | 0 KB |
| slow-4g | dense graph | 14780 / 16762 | 14779 / 16719 | 9762 / 11538 | 9718 / 11498 | 9739 / 11498 | 83 / 83 / 40 | 2 KB |
| slow-4g | dense IVF | 16033 / 17083 | 13414 / 14292 | 9622 / 9875 | 9568 / 9819 | 11433 / 12094 | 94 / 54 / 16 | 325 KB |
| slow-4g | late warp | 24310 / 26165 | 23234 / 24841 | 16226 / 16811 | 16167 / 16746 | 16702 / 17489 | 105 / 96 / 22 | 93 KB |
| wifi | FTS5 | 312 / 354 | 312 / 354 | 312 / 354 | 312 / 354 | 312 / 354 | 15 / 15 / 15 | 0 KB |
| wifi | dense graph | 504 / 572 | 504 / 570 | 325 / 387 | 324 / 386 | 324 / 386 | 83 / 83 / 40 | 3 KB |
| wifi | dense IVF | 540 / 575 | 431 / 463 | 303 / 310 | 301 / 308 | 368 / 395 | 94 / 49 / 16 | 412 KB |
| wifi | late warp | 792 / 861 | 751 / 809 | 502 / 523 | 501 / 521 | 520 / 551 | 105 / 95 / 22 | 113 KB |

(Warm FTS5 is almost always answered from the cache, the same on every profile; shown for `4g` only.)

- **Multi-range requests make HTTP/1.1 as fast as HTTP/2** (within 1 %) for every system and profile. The warm `4g` p50 is 2.6 → 1.4 s (graph), 1.1 → 0.47 s (IVF) and 2.1 → 0.63 s (warp). Cold p50s drop by 38–49 %.
- **Coalescing helps where the ranges are clustered, and only there.** With merging alone, IVF is 15–16 % faster warm and 16–23 % faster cold, for 50–90 KB (warm) or 330–580 KB (cold) of over-fetch per query; warp is 3–8 % faster. The graph gains ≤ 1 %, because its 16 nodes per round lie anywhere in a 40 MB index, and merging even two of them costs more bytes than a wave is worth. The planner correctly declines. Coalescing closes 10–50 % of the gap between h1 and h2 for IVF and warp, not all of it.
- **Guessing wrong is cheap.** Applying C = 6 on HTTP/2 costs 0–28 % (IVF most, cold). Applying no budget on HTTP/1.1 is the status quo.
- **FTS5 is unaffected:** its rounds are single pages.

### Evaluation: end to end

**Node + WASM against `netsim/rangeserver.py --preset <profile>,h1.** `bench/coalesce_real.py words-10k --profiles 4g,lte` ran 20 warm and 8 cold queries per configuration and mode. Coalescing used the online estimate, starting from the defaults. Every mode returned identical top-10 lists. The table gives the p50 in ms, measured, with the prediction obtained by re-simulating the baseline run's own logs in brackets:

| Profile | Regime | Config | Baseline | Coalesce | Multipart | Requests per query (base / coal. / multi) |
|---|---|---|---|---|---|---|
| 4g,h1 | warm | fts-bm25 | 177 (173) | 174 (173) | 176 (173) | 2 / 2 / 2 |
| 4g,h1 | cold | fts-bm25 | 2423 (2378) | 2406 (2378) | 2417 (2378) | 14 / 14 / 14 |
| 4g,h1 | warm | graph-ef64 | 2780 (2752) | 2745 (2712) | 1319 (1293) | 68 / 68 / 35 |
| 4g,h1 | cold | graph-ef64 | 3974 (3932) | 3981 (3932) | 2578 (2537) | 82 / 82 / 39 |
| 4g,h1 | warm | ivf-np64 | 1326 (1304) | 980 (1012) | 514 (503) | 41 / 25 / 8 |
| 4g,h1 | cold | ivf-np64 | 4113 (4138) | 3175 (3142) | 2140 (2108) | 93 / 39 / 16 |
| 4g,h1 | warm | warp-np8-rr64 | 2220 (2236) | 1967 (1917) | 635 (619) | 65 / 58 / 12 |
| 4g,h1 | cold | warp-np8-rr64 | 6082 (6046) | 5544 (5548) | 3457 (3411) | 116 / 100 / 22 |
| lte,h1 | warm | fts-bm25 | 77 (73) | 78 (73) | 77 (73) | 2 / 2 / 2 |
| lte,h1 | cold | fts-bm25 | 1068 (1030) | 1051 (1030) | 1059 (1030) | 14 / 14 / 14 |
| lte,h1 | warm | graph-ef64 | 1268 (1238) | 1242 (1235) | 647 (626) | 68 / 68 / 35 |
| lte,h1 | cold | graph-ef64 | 1865 (1829) | 1863 (1829) | 1272 (1239) | 82 / 82 / 39 |
| lte,h1 | warm | ivf-np64 | 599 (591) | 461 (500) | 263 (257) | 41 / 28 / 8 |
| lte,h1 | cold | ivf-np64 | 1990 (1968) | 1648 (1618) | 1198 (1178) | 93 / 51 / 16 |
| lte,h1 | warm | warp-np8-rr64 | 1030 (1014) | 902 (893) | 347 (336) | 65 / 61 / 12 |
| lte,h1 | cold | warp-np8-rr64 | 3149 (3115) | 2964 (2942) | 2053 (2020) | 116 / 100 / 22 |

The simulator predicts real multipart and coalesce times within 2–4 %. The exception is warm coalescing on IVF, where the real run is 5–8 % *faster* than predicted: the over-fetched blocks it caches are hit by later queries, an effect the re-simulation leaves out.

**Chromium (Playwright, JSPI worker) with the web demo** (`web/test/profile_run.mjs --profiles '4g,h1' --queries 11 --setup-unshaped`). The page and the encoders loaded unshaped; then the profile was switched and the database reopened. Medians are over 10 queries after the first:

| `4g,h1`, Chromium | FTS5 (+ docs) | Dense graph | Late warp | Requests per query (graph / warp) | Estimate at the end |
|---|---|---|---|---|---|
| no budget (before) | 0.92 s | 3.48 s | 3.42 s | 81 / 95 | — |
| auto = 6, coalescing | 0.92 s | 3.33 s | 2.98 s | 80 / 86 | 167 ms, 4.8 Mbit/s |
| 6, multi-range | 0.74 s | 1.82 s | 1.02 s | 40 / 18 | 169 ms, 7.4 Mbit/s |

`node --test web/test/web.test.mjs` passes with the default (`auto`: `http/1.1`, 6). Its results equal the native build's row for row. On the unshaped server the planner never merged anything: the estimated RTT was about 1 ms, so the bandwidth-delay product was tiny.

### Recommendation

- Keep `maxRequests: 'auto'` (on by default in browsers).
- Serve over HTTP/2 where possible.
- On an HTTP/1.1 host that supports multi-range requests (nginx, Apache, Caddy and netsim do), turn on `multipart: true`. It is the only way to reach HTTP/2 latency over six connections. When the server lacks support, the fallback costs one round trip once.

## Things learned on the way

- **Chromium serialises parallel range requests for the same URL** unless `fetch()` uses `cache: 'no-store'`: its HTTP cache takes a per-URL lock while an entry is being written. With the default cache mode, 16 "parallel" requests completed one after another (430 ms instead of about 60 ms). Both backends now default to `no-store` (option `httpCache`).
- **Nested workers and Atomics.wait:** in Chromium, messages posted to a nested Worker are not delivered while its parent is blocked in `Atomics.wait`, but messages on a `MessageChannel` port are. `sync-fetch.mjs` therefore sends batches over a transferred port.
- `sqlite3_file_control(SQLITE_FCNTL_PRAGMA)` must return a non-NULL string on success; a NULL result becomes a NULL column name, which Python reports as `MemoryError`.
- The sync build in Node needs the HTTP server in another thread or process, since the calling thread blocks.

## Tests

- `make test-native`: Python against `build/native` (FTS5 results equal to the default VFS, read-only, speculative batching with simulated latency, `PRAGMA httpvfs_stats`).
- `make test-node`: `node --test wasm/test/node.test.mjs`. Builds `build/test/fts-test.db` natively (4,000 40-word documents from the project word list, FTS5 index, a blob table), serves it with `netsim/rangeserver.py` in a child process (or `wasm/test/rangeserver.mjs` if netsim is absent), and for the asyncify and sync builds checks FTS5 results against the native CLI, the log/counter invariants, speculative batching (at most 4 rounds, at least 8 requests in one round, overlapping in time), readahead on a full scan, parameters and blobs, errors, and serialisation of concurrent queries.
- `make test-node` also runs `wasm/test/coalesce.test.mjs`. It unit-tests `multipart.mjs`: `Content-Range` and boundary parsing, bodies with CRLF and LF line ends, and bodies containing the boundary string. It tests `hvFetchMultiRange` against a fake server that answers multipart (reordered, merged parts), one covering range, only the first range, 200 (the body must be cancelled) or 416. Against netsim limited to six concurrent requests, for asyncify and sync, it checks that baseline, coalescing and multi-range runs return identical rows with at most 6 requests per round. It checks the fallback against `wasm/test/rangeserver.mjs --multi refuse|ignore|first`: exactly one multi-range request is ever sent, and the results are correct. It also checks the online RTT and bandwidth estimate.
- `make test-native` also checks `httpvfs_plan` (ctypes on `httpvfs.so`) against brute force over every set of cuts for 300 random rounds, the multi-range grouping invariants, and `max_req` natively: at most 6 requests per round, over-fetch counted and cached, and results unchanged.
- `make test-browser`: the same checks in a Chromium Worker via Playwright for asyncify (with and without cross-origin isolation), jspi, auto (must pick jspi) and sync.

## Open issues

- Asyncify's code size; JSPI will make it moot once Firefox and Safari ship it. The `auto` variant already uses JSPI where available.
- The sync build in browsers requires COOP/COEP; on hosts that cannot set headers a service-worker shim (coi-serviceworker) would be needed. Not tested in Firefox or Safari (only Chromium is available here).
- No native HTTP backend: native runs read a local file and simulate latency per round. A libcurl-multi backend would let native benchmarks run through `netsim`.
- Speculation relies on SQLite tolerating zero-filled pages (it reports `SQLITE_CORRUPT` and moves on). It is tested for rowid lookups; extensions that speculate through virtual tables should test their own paths, and should cap passes (the demo caps at 16).
- The WASM build is single-threaded (`THREADSAFE=0`, no `-pthread`): pthread calls in extension code (for example `ext/dense/hnsw.c`) link against Emscripten's stubs, so index construction belongs in the native build.
- Chunked databases are not done. (The HTTP/1.1 connection limit is handled by the request budget; batches larger than the cache by pinning.)
