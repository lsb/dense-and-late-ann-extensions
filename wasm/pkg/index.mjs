// sqlite-httpvfs: read-only SQLite over HTTP range requests, in Node and in
// browsers, with the project's extensions statically linked.
//
//   import { open } from './index.mjs';
//   const db = await open('https://example.org/db.sqlite', { pageCacheBytes: 8 << 20 });
//   const rows = await db.query('SELECT * FROM t WHERE id = ?', [42]);
//   console.log(db.stats());
//
// All calls into the WebAssembly module are serialised (Asyncify cannot run
// two suspended calls at once), so it is safe to issue queries concurrently;
// they simply run one after the other.

const SQLITE_OK = 0, SQLITE_ROW = 100, SQLITE_DONE = 101;
const SQLITE_OPEN_READONLY = 0x1, SQLITE_OPEN_URI = 0x40;
const SQLITE_TRANSIENT = -1;
const SQLITE_INTEGER = 1, SQLITE_FLOAT = 2, SQLITE_TEXT = 3, SQLITE_BLOB = 4;

const STAT_NAMES = [
  'requests', 'bytes', 'rounds', 'reads', 'cacheHits', 'cacheMisses',
  'prefetchCalls', 'prefetchBlocks', 'specMisses', 'fileSize', 'blockSize',
  'cacheBlocks', 'cachedBlocks', 'netMs',
];

const modules = new Map();   // variant -> Promise<Module>

/**
 * Load (once) the WebAssembly module. variant: 'auto' (default), 'asyncify'
 * (every browser and Node), 'jspi' (Chromium 137+, not Node 22) or 'sync' (blocking
 * fetches through a worker and Atomics.wait: Node, or a browser Worker in a
 * cross-origin-isolated page).
 */
const FILES = {
  asyncify: './dist/sqlite-httpvfs.mjs',
  jspi: './dist/sqlite-httpvfs-jspi.mjs',
  sync: './dist/sqlite-httpvfs-sync.mjs',
};
let defaultSyncFetcher = null;

/** 'jspi' where JSPI is available (WebAssembly.Suspending), else 'asyncify'. */
export function autoVariant() {
  return typeof WebAssembly !== 'undefined' && typeof WebAssembly.Suspending === 'function'
    ? 'jspi' : 'asyncify';
}

export function loadModule(variant = 'auto', moduleArgs = {}) {
  if (variant === 'auto') variant = autoVariant();
  if (!modules.has(variant)) {
    const file = FILES[variant];
    if (!file) throw new Error(`unknown variant ${variant}`);
    modules.set(variant, import(file).then(async (m) => {
      const M = await m.default(moduleArgs);
      M.queue = Promise.resolve();
      M.nextFile = 1;
      return M;
    }));
  }
  return modules.get(variant);
}

/** Run fn after every earlier call on this module has finished. */
function serial(M, fn) {
  const p = M.queue.then(fn);
  M.queue = p.catch(() => {});
  return p;
}

function allocString(M, s) {
  const n = M.lengthBytesUTF8(s) + 1;
  const p = M._malloc(n);
  M.stringToUTF8(s, p, n);
  return p;
}

async function callAsync(M, name, ret, types, args) {
  return M.ccall(name, ret, types, args, { async: true });
}

export class SqliteError extends Error {
  constructor(message, code) {
    super(message);
    this.code = code;
  }
}

/**
 * Open a read-only database served at url (must support HTTP Range).
 * Options:
 *   pageCacheBytes  VFS block cache size (default 4 MiB)
 *   blockSize       cache block and minimum request size (default 4096;
 *                   use the database page size or a multiple of it)
 *   readaheadBytes  maximum sequential readahead (default 1 MiB, 0 = off)
 *   coalesceGapBytes  merge ranges of one batch separated by at most this
 *   maxParallel     concurrent requests per round (default unlimited)
 *   maxRequests     request budget per round: a round that needs more range
 *                   requests is merged into at most this many (nearby ranges
 *                   joined, over-fetching the pages between them, when the
 *                   cost model says it is faster), or sent as multi-range
 *                   requests with `multipart`. 'auto' (default): 6 in
 *                   browsers over HTTP/1.x (or when the protocol cannot be
 *                   seen), 100 over HTTP/2 and HTTP/3, 0 (off) in Node,
 *                   whose fetch has no per-host connection limit. 0 = off.
 *   rttMs, bandwidthKbps  initial cost-model latency (default 100 ms) and
 *                   bandwidth (default 10000 kbit/s); only their product
 *                   (bytes worth one extra round trip) matters
 *   netAutoEstimate refit rttMs and bandwidthKbps from the observed rounds
 *                   (default true)
 *   multipart       send one multi-range request (multipart/byteranges) per
 *                   connection instead of merging ranges (default false; many
 *                   CDNs and S3 do not support it; a server that answers 200
 *                   or a single range is detected and single ranges are used)
 *   maxRangesPerRequest  ranges per multi-range request (default 100)
 *   headers         extra request headers
 *   fetch           fetch implementation (default globalThis.fetch)
 *   httpCache       fetch() cache mode (default 'no-store'; other modes let
 *                   Chromium serialise parallel requests on its cache lock)
 *   sqliteCacheKiB  SQLite's own page cache (PRAGMA cache_size), default 2048
 *   variant         'auto' (default: jspi if supported, else asyncify) |
 *                   'asyncify' | 'jspi' | 'sync'
 *   fetchSync       sync variant: (url, offsets, lengths) => [{buf, total}]
 *                   (default: a worker from sync-fetch.mjs)
 */
export async function open(url, opts = {}) {
  const variant = !opts.variant || opts.variant === 'auto' ? autoVariant() : opts.variant;
  const M = await loadModule(variant, opts.moduleArgs || {});
  let fetchSync = opts.fetchSync;
  if (variant === 'sync' && !fetchSync) {
    if (!defaultSyncFetcher) {
      defaultSyncFetcher = import('./sync-fetch.mjs').then((m) => m.createSyncFetcher({ fetchHeaders: opts.headers }));
    }
    fetchSync = await defaultSyncFetcher;
  }
  return serial(M, async () => {
    const name = `/httpvfs/${M.nextFile++}`;
    M.httpvfsRegister(name, {
      url: String(url),
      fetch: opts.fetch,
      headers: opts.headers,
      maxParallel: opts.maxParallel,
      cache: opts.httpCache,
      fetchSync,
    });
    const params = new URLSearchParams({
      cache_kb: String(Math.max(1, Math.round((opts.pageCacheBytes ?? 4 << 20) / 1024))),
      block: String(opts.blockSize ?? 4096),
      readahead_kb: String(Math.round((opts.readaheadBytes ?? 1 << 20) / 1024)),
      gap_kb: String(Math.round((opts.coalesceGapBytes ?? 0) / 1024)),
      immutable: '1',
    });
    const uri = `file:${name}?${params}`;
    const pDb = M._malloc(4);
    const pUri = allocString(M, uri);
    const pVfs = allocString(M, 'httpvfs');
    let rc, db;
    try {
      rc = await callAsync(M, 'sqlite3_open_v2', 'number',
        ['number', 'number', 'number', 'number'],
        [pUri, pDb, SQLITE_OPEN_READONLY | SQLITE_OPEN_URI, pVfs]);
      db = M.HEAPU32[pDb >> 2];
    } finally {
      M._free(pDb); M._free(pUri); M._free(pVfs);
    }
    if (rc !== SQLITE_OK) {
      const cfg = M.httpvfsFiles.get(name);
      let msg = db ? M.UTF8ToString(M._sqlite3_errmsg(db)) : `open failed (${rc})`;
      if (cfg?.lastError) msg += ` (${cfg.lastError.message || cfg.lastError})`;
      if (db) M._sqlite3_close_v2(db);
      M.httpvfsUnregister(name);
      throw new SqliteError(`${msg}: ${url}`, rc);
    }
    const d = new Database(M, db, name, url);
    d.variant = variant;
    d.protocol = detectProtocol(String(url));
    let maxRequests = opts.maxRequests ?? 'auto';
    if (maxRequests === 'auto') maxRequests = autoMaxRequests(d.protocol);
    d._netConfig({
      maxRequests, rttMs: opts.rttMs, bandwidthKbps: opts.bandwidthKbps,
      multipart: opts.multipart, netAutoEstimate: opts.netAutoEstimate,
      maxRangesPerRequest: opts.maxRangesPerRequest,
    });
    const kib = opts.sqliteCacheKiB ?? 2048;
    await d._exec(`PRAGMA cache_size=-${kib}`);
    return d;
  });
}

const isNode = typeof process !== 'undefined' && !!process.versions?.node;

/** The protocol the first request of url used ('http/1.1', 'h2', ...), if the
 * Resource Timing API shows it (cross-origin: needs Timing-Allow-Origin). */
function detectProtocol(url) {
  try {
    const abs = new URL(url, globalThis.location?.href).href;
    const es = performance.getEntriesByName(abs);
    for (let i = es.length - 1; i >= 0; i--) if (es[i].nextHopProtocol) return es[i].nextHopProtocol;
  } catch {}
  return '';
}

/** Default request budget: browsers allow 6 connections per host over
 * HTTP/1.x; HTTP/2 and HTTP/3 multiplex (100 streams is a common limit);
 * Node's fetch opens as many connections as needed. */
export function autoMaxRequests(protocol) {
  if (isNode) return 0;
  return /^(h2|h3|http\/2|http\/3)/i.test(protocol || '') ? 100 : 6;
}

const NET_NAMES = ['maxRequests', 'rttMs', 'bandwidthKbps', 'multipart', 'netAutoEstimate',
  'overfetchBytes', 'plannedRounds', 'multipartRequests', 'multipartFailed', 'observedRounds'];

export class Database {
  constructor(M, db, name, url) {
    this.M = M;
    this.db = db;
    this.name = name;
    this.url = url;
  }

  _check(rc) {
    if (rc !== SQLITE_OK && rc !== SQLITE_ROW && rc !== SQLITE_DONE) {
      let msg = this.M.UTF8ToString(this.M._sqlite3_errmsg(this.db));
      const cfg = this.M.httpvfsFiles.get(this.name);
      if ((rc & 0xff) === 10 && cfg?.lastError) {   // SQLITE_IOERR: say why
        msg += ` (${cfg.lastError.message || cfg.lastError})`;
        cfg.lastError = null;
      }
      throw new SqliteError(msg, rc);
    }
    return rc;
  }

  async _exec(sql) {
    const M = this.M;
    const p = allocString(M, sql);
    try {
      const rc = await callAsync(M, 'sqlite3_exec', 'number',
        ['number', 'number', 'number', 'number', 'number'], [this.db, p, 0, 0, 0]);
      this._check(rc);
    } finally {
      M._free(p);
    }
  }

  /** Run one or more statements, discarding results. */
  exec(sql) {
    return serial(this.M, () => this._exec(sql));
  }

  _bind(stmt, params) {
    const M = this.M;
    const bindOne = (i, v) => {
      let rc;
      if (v === null || v === undefined) rc = M._sqlite3_bind_null(stmt, i);
      else if (typeof v === 'bigint') rc = M._sqlite3_bind_int64(stmt, i, v);
      else if (typeof v === 'number') {
        rc = Number.isSafeInteger(v) ? M._sqlite3_bind_int64(stmt, i, BigInt(v))
                                     : M._sqlite3_bind_double(stmt, i, v);
      } else if (typeof v === 'boolean') rc = M._sqlite3_bind_int64(stmt, i, v ? 1n : 0n);
      else if (typeof v === 'string') {
        const p = allocString(M, v);
        rc = M._sqlite3_bind_text(stmt, i, p, -1, SQLITE_TRANSIENT);
        M._free(p);
      } else if (v instanceof Uint8Array || v instanceof ArrayBuffer || ArrayBuffer.isView(v)) {
        const u8 = v instanceof Uint8Array ? v
          : v instanceof ArrayBuffer ? new Uint8Array(v)
          : new Uint8Array(v.buffer, v.byteOffset, v.byteLength);
        const p = M._malloc(Math.max(1, u8.length));
        M.HEAPU8.set(u8, p);
        rc = M._sqlite3_bind_blob(stmt, i, p, u8.length, SQLITE_TRANSIENT);
        M._free(p);
      } else throw new TypeError(`cannot bind ${typeof v}`);
      this._check(rc);
    };
    if (Array.isArray(params)) {
      params.forEach((v, k) => bindOne(k + 1, v));
    } else if (params && typeof params === 'object') {
      for (const [k, v] of Object.entries(params)) {
        const key = /^[:@$?]/.test(k) ? k : ':' + k;
        const pk = allocString(M, key);
        const i = M._sqlite3_bind_parameter_index(stmt, pk);
        M._free(pk);
        if (i === 0) throw new SqliteError(`no parameter ${key}`, 25);
        bindOne(i, v);
      }
    }
  }

  _row(stmt, ncol) {
    const M = this.M;
    const row = new Array(ncol);
    for (let c = 0; c < ncol; c++) {
      switch (M._sqlite3_column_type(stmt, c)) {
        case SQLITE_INTEGER: {
          const v = M._sqlite3_column_int64(stmt, c);
          row[c] = (v >= -9007199254740991n && v <= 9007199254740991n) ? Number(v) : v;
          break;
        }
        case SQLITE_FLOAT: row[c] = M._sqlite3_column_double(stmt, c); break;
        case SQLITE_TEXT: row[c] = M.UTF8ToString(M._sqlite3_column_text(stmt, c)); break;
        case SQLITE_BLOB: {
          const p = M._sqlite3_column_blob(stmt, c);
          const n = M._sqlite3_column_bytes(stmt, c);
          row[c] = M.HEAPU8.slice(p, p + n);
          break;
        }
        default: row[c] = null;
      }
    }
    return row;
  }

  /**
   * Run one statement with optional parameters (array, or object of named
   * parameters). Returns { columns, rows } with rows as arrays.
   */
  queryRaw(sql, params) {
    const M = this.M;
    return serial(M, async () => {
      const pSql = allocString(M, sql);
      const pStmt = M._malloc(4);
      let stmt = 0;
      try {
        this._check(await callAsync(M, 'sqlite3_prepare_v2', 'number',
          ['number', 'number', 'number', 'number', 'number'], [this.db, pSql, -1, pStmt, 0]));
        stmt = M.HEAPU32[pStmt >> 2];
        if (!stmt) return { columns: [], rows: [] };
        if (params !== undefined) this._bind(stmt, params);
        const ncol = M._sqlite3_column_count(stmt);
        const columns = [];
        for (let c = 0; c < ncol; c++) columns.push(M.UTF8ToString(M._sqlite3_column_name(stmt, c)));
        const rows = [];
        for (;;) {
          const rc = await callAsync(M, 'sqlite3_step', 'number', ['number'], [stmt]);
          if (rc === SQLITE_ROW) rows.push(this._row(stmt, ncol));
          else { this._check(rc); break; }
        }
        return { columns, rows };
      } finally {
        if (stmt) M._sqlite3_finalize(stmt);
        M._free(pStmt); M._free(pSql);
      }
    });
  }

  /** Run one statement; returns an array of row objects keyed by column name. */
  async query(sql, params) {
    const { columns, rows } = await this.queryRaw(sql, params);
    return rows.map((r) => Object.fromEntries(columns.map((c, i) => [c, r[i]])));
  }

  /** VFS counters for this database (cumulative since open or resetStats). */
  stats() {
    const M = this.M;
    const p = M._malloc(8 * STAT_NAMES.length);
    try {
      if (M._httpvfs_stats_array(this.db, p) !== SQLITE_OK) return null;
      const out = {};
      STAT_NAMES.forEach((k, i) => { out[k] = M.HEAPF64[(p >> 3) + i]; });
      return out;
    } finally {
      M._free(p);
    }
  }

  _netConfig(o) {
    const num = (v, d = -1) => (v === undefined || v === null ? d : Number(v));
    const flag = (v) => (v === undefined || v === null ? -1 : v ? 1 : 0);
    this.M._httpvfs_net_config(this.db, num(o.maxRequests), num(o.rttMs), num(o.bandwidthKbps),
      flag(o.multipart), flag(o.netAutoEstimate), num(o.maxRangesPerRequest));
  }

  /** Change the request planning options of open() on an open database
   * (maxRequests, rttMs, bandwidthKbps, multipart, netAutoEstimate,
   * maxRangesPerRequest); omitted ones keep their value. */
  setNetOptions(o) {
    return serial(this.M, async () => {
      const x = { ...o };
      if (x.maxRequests === 'auto') x.maxRequests = autoMaxRequests(this.protocol);
      this._netConfig(x);
    });
  }

  /** Request planner state: the budget, the current cost-model estimates and
   * counters (over-fetched bytes, rounds the planner changed, multi-range
   * requests; since open or resetStats). */
  netState() {
    const M = this.M;
    const p = M._malloc(8 * NET_NAMES.length);
    try {
      if (M._httpvfs_net_state(this.db, p) !== SQLITE_OK) return null;
      const out = { protocol: this.protocol };
      NET_NAMES.forEach((k, i) => { out[k] = M.HEAPF64[(p >> 3) + i]; });
      out.multipart = !!out.multipart; out.netAutoEstimate = !!out.netAutoEstimate;
      out.multipartFailed = !!out.multipartFailed;
      return out;
    } finally {
      M._free(p);
    }
  }

  /**
   * Request log: [{offset, length, round, tStart, tEnd, req}], times in ms
   * from performance.now(). One entry per range; the ranges of one
   * multi-range request share `req` (the request's number in its round).
   */
  log() {
    const M = this.M;
    const n = M._httpvfs_log_count(this.db);
    if (n === 0) return [];
    const p = M._malloc(48 * n);
    try {
      const got = M._httpvfs_log_copy(this.db, 0, n, p);
      const out = new Array(got);
      const b = p >> 3;
      for (let i = 0; i < got; i++) {
        const F = M.HEAPF64;
        out[i] = { offset: F[b + 6 * i], length: F[b + 6 * i + 1], round: F[b + 6 * i + 2],
                   tStart: F[b + 6 * i + 3], tEnd: F[b + 6 * i + 4], req: F[b + 6 * i + 5] };
      }
      return out;
    } finally {
      M._free(p);
    }
  }

  /**
   * Zero the counters and the log. With {clearCache: true} also empty the
   * VFS cache and SQLite's page cache, so the next query starts cold.
   */
  resetStats({ clearCache = false } = {}) {
    const M = this.M;
    return serial(M, async () => {
      if (clearCache) M._sqlite3_db_release_memory(this.db);
      M._httpvfs_reset(this.db, clearCache ? 1 : 0);
    });
  }

  close() {
    const M = this.M;
    return serial(M, async () => {
      if (this.db) {
        M._sqlite3_close_v2(this.db);
        M.httpvfsUnregister(this.name);
        this.db = 0;
      }
    });
  }
}
