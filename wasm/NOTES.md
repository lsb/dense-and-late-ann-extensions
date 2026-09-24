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

- **Block cache.** LRU over fixed-size blocks (default 4096 bytes; set it to the database page size or a multiple). Size from `pageCacheBytes` (JS) or `cache_kb` (URI), default 4 MiB. A read larger than the cache still works (copied straight from the fetched buffers). SQLite's own page cache sits on top (the JS API sets `PRAGMA cache_size` to 2 MiB by default).
- **Readahead** (as in sql.js-httpvfs): when a read starts where the previous one ended, the miss is extended by 1, 2, 4, … blocks up to `readaheadBytes` (default 1 MiB, 0 disables). A full scan of the `docs` table in the 2.2 MB test database takes 12 requests (2.1 MB) instead of 336 (1.4 MB) without readahead; the extra bytes are readahead running past the end of the table.
- **Coalescing.** Adjacent missing blocks in a batch become one request; `coalesceGapBytes` also merges ranges separated by small gaps.
- **Size discovery.** The first request (block 0, which SQLite reads first anyway) learns the file size from `Content-Range`, so opening costs one round.
- **Read-only.** `SQLITE_IOCAP_IMMUTABLE`, writes return `SQLITE_READONLY`; a WAL-mode header is presented as rollback mode. Non-main files (temp, journals) go to the default VFS.
- **Instrumentation.** Every request is logged as `{offset, length, round, tStart, tEnd}` (ms, `performance.now()` clock of the calling thread). `round` is the number of the backend call; the requests of one batch share a round. Counters: `requests`, `bytes`, `rounds`, `reads`, `cacheHits`, `cacheMisses`, `prefetchCalls`, `prefetchBlocks`, `specMisses`, `netMs` (wall time blocked on the network), plus `fileSize`, `blockSize`, `cacheBlocks`, `cachedBlocks`.
- **URI parameters** (native and WASM): `cache_kb`, `block`, `readahead_kb`, `gap_kb`, `log_max`, and natively `latency_ms`.

Chunked databases (sql.js-httpvfs "chunked" mode) are not implemented; the backend interface would take them without changes to the C side.

## JavaScript API (`wasm/pkg/index.mjs`)

```js
import { open } from './wasm/pkg/index.mjs';
const db = await open('https://host/db.sqlite', {
  pageCacheBytes: 8 << 20, blockSize: 4096, readaheadBytes: 1 << 20,
  coalesceGapBytes: 0, maxParallel: undefined, variant: 'auto',
});
await db.query('SELECT rowid FROM t_fts WHERE t_fts MATCH ? LIMIT 10', ['word']);  // [{rowid: …}]
await db.queryRaw(sql, params);   // {columns, rows: [[…]]}
await db.exec(sql);               // several statements, no results
db.stats(); db.log();
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

## Things learned on the way

- **Chromium serialises parallel range requests for the same URL** unless `fetch()` uses `cache: 'no-store'`: its HTTP cache takes a per-URL lock while an entry is being written. With the default cache mode, 16 "parallel" requests completed one after another (430 ms instead of about 60 ms). Both backends now default to `no-store` (option `httpCache`).
- **Nested workers and Atomics.wait:** in Chromium, messages posted to a nested Worker are not delivered while its parent is blocked in `Atomics.wait`, but messages on a `MessageChannel` port are. `sync-fetch.mjs` therefore sends batches over a transferred port.
- `sqlite3_file_control(SQLITE_FCNTL_PRAGMA)` must return a non-NULL string on success; a NULL result becomes a NULL column name, which Python reports as `MemoryError`.
- The sync build in Node needs the HTTP server in another thread or process, since the calling thread blocks.

## Tests

- `make test-native`: Python against `build/native` (FTS5 results equal to the default VFS, read-only, speculative batching with simulated latency, `PRAGMA httpvfs_stats`).
- `make test-node`: `node --test wasm/test/node.test.mjs`. Builds `build/test/fts-test.db` natively (4,000 40-word documents from the project word list, FTS5 index, a blob table), serves it with `netsim/rangeserver.py` in a child process (or `wasm/test/rangeserver.mjs` if netsim is absent), and for the asyncify and sync builds checks FTS5 results against the native CLI, the log/counter invariants, speculative batching (at most 4 rounds, at least 8 requests in one round, overlapping in time), readahead on a full scan, parameters and blobs, errors, and serialisation of concurrent queries.
- `make test-browser`: the same checks in a Chromium Worker via Playwright for asyncify (with and without cross-origin isolation), jspi, auto (must pick jspi) and sync.

## Open issues

- Asyncify's code size; JSPI will make it moot once Firefox and Safari ship it. The `auto` variant already uses JSPI where available.
- The sync build in browsers requires COOP/COEP; on hosts that cannot set headers a service-worker shim (coi-serviceworker) would be needed. Not tested in Firefox or Safari (only Chromium is available here).
- No native HTTP backend: native runs read a local file and simulate latency per round. A libcurl-multi backend would let native benchmarks run through `netsim`.
- Speculation relies on SQLite tolerating zero-filled pages (it reports `SQLITE_CORRUPT` and moves on). It is tested for rowid lookups; extensions that speculate through virtual tables should test their own paths, and should cap passes (the demo caps at 16).
- The WASM build is single-threaded (`THREADSAFE=0`, no `-pthread`): pthread calls in extension code (for example `ext/dense/hnsw.c`) link against Emscripten's stubs, so index construction belongs in the native build.
- Chunked databases, a `maxParallel` default tuned for HTTP/1.1 (six connections), and an eviction policy that pins blocks of the current batch are not done.
