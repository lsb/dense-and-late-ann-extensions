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
 * Load (once) the WebAssembly module. variant: 'asyncify' (default, every
 * browser and Node) or 'jspi' (Chromium 137+, not Node 22).
 */
export function loadModule(variant = 'asyncify', moduleArgs = {}) {
  if (!modules.has(variant)) {
    const file = variant === 'jspi' ? './dist/sqlite-httpvfs-jspi.mjs' : './dist/sqlite-httpvfs.mjs';
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
 *   headers         extra request headers
 *   fetch           fetch implementation (default globalThis.fetch)
 *   sqliteCacheKiB  SQLite's own page cache (PRAGMA cache_size), default 2048
 *   variant         'asyncify' | 'jspi'
 */
export async function open(url, opts = {}) {
  const M = await loadModule(opts.variant || 'asyncify', opts.moduleArgs || {});
  return serial(M, async () => {
    const name = `/httpvfs/${M.nextFile++}`;
    M.httpvfsRegister(name, {
      url: String(url),
      fetch: opts.fetch,
      headers: opts.headers,
      maxParallel: opts.maxParallel,
      cache: opts.httpCache,
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
      const msg = db ? M.UTF8ToString(M._sqlite3_errmsg(db)) : `open failed (${rc})`;
      if (db) M._sqlite3_close_v2(db);
      M.httpvfsUnregister(name);
      throw new SqliteError(`${msg}: ${url}`, rc);
    }
    const d = new Database(M, db, name, url);
    const kib = opts.sqliteCacheKiB ?? 2048;
    await d._exec(`PRAGMA cache_size=-${kib}`);
    return d;
  });
}

export class Database {
  constructor(M, db, name, url) {
    this.M = M;
    this.db = db;
    this.name = name;
    this.url = url;
  }

  _check(rc) {
    if (rc !== SQLITE_OK && rc !== SQLITE_ROW && rc !== SQLITE_DONE) {
      const msg = this.M.UTF8ToString(this.M._sqlite3_errmsg(this.db));
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

  /**
   * Request log: [{offset, length, round, tStart, tEnd}], times in ms from
   * performance.now().
   */
  log() {
    const M = this.M;
    const n = M._httpvfs_log_count(this.db);
    if (n === 0) return [];
    const p = M._malloc(40 * n);
    try {
      const got = M._httpvfs_log_copy(this.db, 0, n, p);
      const out = new Array(got);
      const b = p >> 3;
      for (let i = 0; i < got; i++) {
        const F = M.HEAPF64;
        out[i] = { offset: F[b + 5 * i], length: F[b + 5 * i + 1], round: F[b + 5 * i + 2],
                   tStart: F[b + 5 * i + 3], tEnd: F[b + 5 * i + 4] };
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
