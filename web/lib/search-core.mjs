// Search over a database built by tools/build_db.py (or the Python package's
// `dense-late-ann build-db`), without workers or encoders: discovers the FTS5,
// dense_ann and late_plaid tables, builds the query SQL and fetches document
// text. The SQLite worker (sqlite-worker.mjs) runs this in the browser; in
// Node it can be used directly with precomputed query vectors:
//
//   import { open } from '../../wasm/pkg/index.mjs';     // npm: 'dense-late-ann'
//   import { SearchDb } from './search-core.mjs';       // npm: 'dense-late-ann/search-core'
//   const s = await SearchDb.open(await open('http://host/db.sqlite'));
//   await s.search({ system: 'fts', text: 'pinwheel', k: 10 });
//   await s.search({ system: 'dense', vector: float32Array384, k: 10 });
//   await s.search({ system: 'late', vector: float32ArrayNx48, k: 10 });

const VFS_KEYS = ['rounds', 'requests', 'bytes', 'netMs', 'reads', 'cacheHits', 'cacheMisses'];

function delta(a, b) {
  const d = {};
  for (const k of VFS_KEYS) d[k] = (b?.[k] ?? 0) - (a?.[k] ?? 0);
  return d;
}

/** Default query parameters per index kind (see the extensions' NOTES.md). */
export const DEFAULT_PARAMS = {
  // rank: 'bm25c' (default) ranks with ext/fts5rank's bm25c(): bm25 with
  // every document at the average length, which reads no per-document
  // lengths; 'bm25' is FTS5's built-in bm25() (exact, but one %_docsize row,
  // i.e. one sequential round trip, per matching document); 'bm25-prefetch'
  // is the exact bm25() after fetching those rows in one batch;
  // 'bm25-rerank' re-scores bm25c's top 50 with their stored lengths (exact
  // bm25 formula on the candidates); 'none' is rowid order. See
  // docs/fts5-httpvfs.md.
  fts: { mode: 'or', rank: 'bm25c' },
  'dense:graph': { ef: 64, beam: 16, rerank: 2 },
  'dense:ivf': { nprobe: 16, rerank_k: 64, rerank: 2 },
  late: { nprobe: 8, layout: 'warp', stoplist: 0.02, rerank: 64 },
};

export function paramsKey(ix) { return ix.kind === 'dense' ? `dense:${ix.layout}` : ix.kind; }

/** Parse the module arguments of CREATE VIRTUAL TABLE … USING mod(args). */
function parseVtab(sql) {
  const m = /USING\s+(\w+)\s*\(([\s\S]*)\)\s*$/i.exec(sql || '');
  if (!m) return null;
  const params = {};
  for (const part of m[2].split(/[,\s]+/)) {
    const kv = /^([\w]+)\s*=\s*'?([^']*)'?$/.exec(part.trim());
    if (kv) params[kv[1].toLowerCase()] = kv[2];
  }
  return { module: m[1].toLowerCase(), params };
}

/** Find the search indexes and the document table of an open database. */
export async function discover(db) {
  const rows = await db.query("SELECT name, sql FROM sqlite_schema WHERE type = 'table' AND sql LIKE 'CREATE VIRTUAL TABLE%'");
  const indexes = [];
  for (const { name, sql } of rows) {
    const v = parseVtab(sql);
    if (!v) continue;
    if (v.module === 'fts5') {
      indexes.push({ table: name, kind: 'fts', module: 'fts5', content: v.params.content || null,
                     contentRowid: v.params.content_rowid || 'rowid' });
    } else if (v.module === 'dense_ann') {
      indexes.push({ table: name, kind: 'dense', module: 'dense_ann',
                     layout: v.params.layout === 'ivf' ? 'ivf' : 'graph', params: v.params });
    } else if (v.module === 'late_plaid') {
      indexes.push({ table: name, kind: 'late', module: 'late_plaid',
                     layout: v.params.layout || 'both', params: v.params });
    }
  }
  const tables = new Set((await db.query("SELECT name FROM sqlite_schema WHERE type = 'table'")).map((r) => r.name));
  const fts = indexes.find((i) => i.kind === 'fts' && i.content);
  const docs = fts ? { table: fts.content, key: fts.contentRowid === 'rowid' ? 'rowid' : fts.contentRowid }
    : tables.has('docs') ? { table: 'docs', key: 'id' } : null;
  if (docs) {
    const cols = (await db.query(`SELECT name FROM pragma_table_info('${docs.table.replace(/'/g, "''")}')`)).map((r) => r.name);
    docs.column = cols.includes('body') ? 'body' : cols.find((c) => c !== docs.key) || null;
  }
  return { indexes, docs };
}

const q = (s) => `"${String(s).replace(/"/g, '""')}"`;

/** Integers only: these values are interpolated into SQL as constraints. */
function intParam(v, name) {
  const n = Number(v);
  if (!Number.isInteger(n)) throw new Error(`parameter ${name} must be an integer, got ${v}`);
  return n;
}

/** FTS5 query from free text: each word quoted, joined with AND or OR. */
export function ftsQuery(text, mode = 'or') {
  const words = String(text).toLowerCase().match(/[\p{L}\p{N}]+/gu) || [];
  const uniq = [...new Set(words)];
  return uniq.map((w) => `"${w}"`).join(mode === 'and' ? ' AND ' : ' OR ');
}

/** FTS5 ranking methods (params.rank); see DEFAULT_PARAMS. */
export const FTS_RANKS = ['bm25c', 'bm25', 'bm25-prefetch', 'bm25-rerank', 'none'];

/** The SQL of one search: {sql, extra} where extra are parameters after ?1. */
export function buildSql(ix, k, params) {
  const t = q(ix.table);
  if (ix.kind === 'fts') {
    const rank = params.rank || 'bm25';
    if (!FTS_RANKS.includes(rank)) throw new Error(`unknown FTS rank ${rank} (use ${FTS_RANKS.join(', ')})`);
    const lim = `LIMIT ${intParam(k, 'k')}`;
    if (rank === 'none') return { sql: `SELECT rowid AS id, NULL AS score FROM ${t} WHERE ${t} MATCH ?1 ${lim}` };
    if (rank === 'bm25c') return { sql: `SELECT rowid AS id, rank AS score FROM ${t} WHERE ${t} MATCH ?1 AND rank MATCH 'bm25c()' ORDER BY rank ${lim}` };
    return { sql: `SELECT rowid AS id, rank AS score FROM ${t} WHERE ${t} MATCH ?1 ORDER BY rank ${lim}` };
  }
  if (ix.kind === 'dense') {
    const allowed = ix.layout === 'ivf' ? ['nprobe', 'rerank_k', 'rerank', 'exact'] : ['ef', 'beam', 'rerank', 'exact'];
    let where = `embedding MATCH ?1 AND k = ${intParam(k, 'k')}`;
    for (const name of allowed) if (params[name] !== undefined && params[name] !== '') where += ` AND ${name} = ${intParam(params[name], name)}`;
    return { sql: `SELECT rowid AS id, distance AS score, stats FROM ${t} WHERE ${where}` };
  }
  if (ix.kind === 'late') {
    let where = `${t} MATCH ?1 AND k = ${intParam(k, 'k')}`;
    if (params.nprobe !== undefined && params.nprobe !== '') where += ` AND nprobe = ${intParam(params.nprobe, 'nprobe')}`;
    const opts = [];
    for (const name of ['layout', 'approx', 'ndocs', 'stoplist', 'rerank', 'impute', 'cross', 'tcs', 'cprobe', 'exact']) {
      const v = params[name];
      if (v === undefined || v === '') continue;
      if (!/^[\w.]+$/.test(String(v))) throw new Error(`bad value for ${name}: ${v}`);
      opts.push(`${name}=${v}`);
    }
    return { sql: `SELECT rowid AS id, score, stats FROM ${t} WHERE ${where} AND opts = ?2`, extra: [opts.join(' ')] };
  }
  throw new Error(`unknown index kind ${ix.kind}`);
}

/**
 * The statement that makes an index load its per-connection static data
 * (dense: PQ codebook and entry set or IVF centroids; late: centroids, list
 * lengths and the document table's interior pages; FTS5: the structure
 * record and the pages a term lookup walks). It returns no rows.
 */
export function warmSql(ix) {
  const t = q(ix.table);
  if (ix.kind === 'dense') return `SELECT rowid FROM ${t} WHERE embedding MATCH 'warm'`;
  if (ix.kind === 'late') return `SELECT rowid FROM ${t} WHERE ${t} MATCH 'warm'`;
  if (ix.kind === 'fts') return `SELECT rowid FROM ${t} WHERE ${t} MATCH '"zzwarmzz"' LIMIT 1`;
  throw new Error(`unknown index kind ${ix.kind}`);
}

/** A database opened with wasm/pkg's open(), plus its discovered indexes. */
export class SearchDb {
  static async open(db) {
    const s = new SearchDb();
    s.db = db;
    Object.assign(s, await discover(db));
    return s;
  }

  /** Pick the index for a system name: 'fts' | 'dense' | 'late' | a table name. */
  resolve(system) {
    const ix = this.indexes.find((i) => i.table === system)
      || (system === 'dense' && (this.indexes.find((i) => i.kind === 'dense' && i.layout === 'graph')
                                 || this.indexes.find((i) => i.kind === 'dense')))
      || this.indexes.find((i) => i.kind === system);
    if (!ix) throw new Error(`this database has no ${system} index (have: ${this.indexes.map((i) => i.table).join(', ')})`);
    return ix;
  }

  /** Text of documents by id: Map(id -> text). */
  async fetchDocs(ids) {
    const { db } = this;
    const d = this.docs;
    if (!d || !d.column || !ids.length) return new Map();
    const list = ids.map((x) => intParam(x, 'id')).join(',');
    const key = d.key === 'rowid' ? 'rowid' : q(d.key);
    const lookup = `SELECT ${q(d.column)} FROM ${q(d.table)} WHERE ${key} = ?`;
    // httpvfs_warm (wasm/demo_ext.c) runs the lookups speculatively so that all
    // of them fetch their b-tree pages together, one round per tree level.
    try { await db.query('SELECT httpvfs_warm(?, ?)', [lookup, list]); } catch { /* optional */ }
    const rows = await db.queryRaw(`SELECT ${key}, ${q(d.column)} FROM ${q(d.table)} WHERE ${key} IN (${list})`);
    return new Map(rows.rows.map((r) => [r[0], r[1]]));
  }

  /**
   * Fetch the %_docsize rows of documents `ids` in one batch with
   * httpvfs_warm (speculative batching: about one round per uncached b-tree
   * level), so that later per-row length lookups hit the cache.
   */
  async warmDocsizes(ix, ids) {
    if (!ids.length) return false;
    const lookup = `SELECT sz FROM ${q(ix.table + '_docsize')} WHERE id = ?`;
    try { await this.db.query('SELECT httpvfs_warm(?, ?)', [lookup, ids.map((x) => intParam(x, 'id')).join(',')]); } catch { return false; }
    return true;
  }

  /**
   * Run one FTS5 query with the ranking method params.rank (FTS_RANKS):
   *  bm25-prefetch: list the matching rowids (reads only the doclists), fetch
   *    their %_docsize rows in one batch, then run bm25() on a warm cache; the
   *    batch is skipped when more than params.prefetchCap (4000) documents
   *    match, since the rows would not fit the block cache.
   *  bm25-rerank: the top params.rerankDepth (50) documents by bm25c(), their
   *    %_docsize rows in one batch, then those candidates re-scored with the
   *    exact bm25 formula (bm25dl() with the stored length).
   * Returns {res, sql, ext}.
   */
  async ftsRun(ix, arg, k, params) {
    const { db } = this;
    const t = q(ix.table);
    const rank = params.rank;
    if (rank === 'bm25-prefetch') {
      const cap = intParam(params.prefetchCap ?? 4000, 'prefetchCap');
      const ids = (await db.queryRaw(`SELECT rowid FROM ${t} WHERE ${t} MATCH ?1 LIMIT ${cap + 1}`, [arg])).rows.map((r) => r[0]);
      const warmed = ids.length <= cap && await this.warmDocsizes(ix, ids);
      const { sql } = buildSql(ix, k, params);
      return { res: await db.queryRaw(sql, [arg]), sql, ext: { matches: ids.length, warmed } };
    }
    if (rank === 'bm25-rerank') {
      const depth = Math.max(intParam(params.rerankDepth ?? 50, 'rerankDepth'), intParam(k, 'k'));
      const cand = (await db.queryRaw(buildSql(ix, depth, { ...params, rank: 'bm25c' }).sql, [arg])).rows.map((r) => r[0]);
      if (!cand.length) return { res: { columns: ['id', 'score'], rows: [] }, sql: null, ext: { candidates: 0 } };
      const warmed = await this.warmDocsizes(ix, cand);
      const list = cand.map((x) => intParam(x, 'id')).join(',');
      // The candidate test sits in the select list, not in WHERE, so that it is
      // not handed to FTS5 as a rowid constraint (one xFilter per value).
      const sql = `SELECT id, score FROM (SELECT rowid AS id, CASE WHEN rowid IN (${list}) THEN `
        + `bm25dl(${t}, (SELECT sz FROM ${q(ix.table + '_docsize')} WHERE id = ${t}.rowid)) END AS score `
        + `FROM ${t} WHERE ${t} MATCH ?1) WHERE score IS NOT NULL ORDER BY score LIMIT ${intParam(k, 'k')}`;
      return { res: await db.queryRaw(sql, [arg]), sql, ext: { candidates: cand.length, warmed } };
    }
    const { sql } = buildSql(ix, k, params);
    return { res: await db.queryRaw(sql, [arg]), sql, ext: null };
  }

  /**
   * One search. `system` or `table` names the index; FTS takes `text`, dense
   * and late take `vector` (Float32Array: [384] for MiniLM, [n × 48] for
   * LateOn). `params` are the extension's query parameters; missing ones take
   * DEFAULT_PARAMS. Returns {rows: [{id, score, text}], ext, sql, stats}.
   */
  async search({ system, table, text, match, vector, k = 10, params = {}, cold = false, fetchDocs = true, ftsMode = 'or' }) {
    const { db } = this;
    const ix = table ? this.indexes.find((i) => i.table === table) : this.resolve(system || 'fts');
    if (!ix) throw new Error(`no index ${table}`);
    params = { ...DEFAULT_PARAMS[paramsKey(ix)], ...params };
    if (ix.kind === 'fts' && this.noBm25c && /^bm25-?(c|rerank)$/.test(params.rank)) params.rank = 'bm25';
    if (cold) await db.resetStats({ clearCache: true });
    let { sql, extra = [] } = buildSql(ix, k, params);
    let arg;
    if (ix.kind === 'fts') {
      // `match`: a ready FTS5 expression (benchmarks); else built from `text`.
      arg = match ?? ftsQuery(text, params.mode || ftsMode);
      if (!arg) return { table: ix.table, kind: ix.kind, rows: [], sql, match: arg, stats: {} };
    } else {
      if (!(vector instanceof Float32Array)) throw new Error(`${ix.table}: vector must be a Float32Array`);
      arg = new Uint8Array(vector.buffer, vector.byteOffset, vector.byteLength);
    }
    const s0 = db.stats();
    const t0 = performance.now();
    let res, ext = null;
    if (ix.kind === 'fts') {
      try {
        ({ res, sql, ext } = await this.ftsRun(ix, arg, k, params));
      } catch (e) {
        // A SQLite build without ext/fts5rank: fall back to the built-in bm25().
        if (!/no such function: bm25(c|dl)|bm25c|bm25dl/.test(String(e.message || e)) || params.rank === 'bm25') throw e;
        this.noBm25c = true;
        params.rank = 'bm25';
        ({ res, sql, ext } = await this.ftsRun(ix, arg, k, params));
      }
    } else {
      res = await db.queryRaw(sql, [arg, ...extra]);
    }
    const t1 = performance.now();
    const s1 = db.stats();
    const ci = Object.fromEntries(res.columns.map((c, i) => [c, i]));
    const rows = res.rows.map((r) => ({ id: r[ci.id], score: r[ci.score] }));
    if (ci.stats !== undefined && res.rows.length) {
      try { ext = JSON.parse(res.rows[0][ci.stats]); } catch { ext = res.rows[0][ci.stats]; }
    }
    let docsMs = 0, s2 = s1;
    if (fetchDocs && rows.length) {
      const t2 = performance.now();
      const texts = await this.fetchDocs(rows.map((r) => r.id));
      docsMs = performance.now() - t2;
      s2 = db.stats();
      for (const r of rows) r.text = texts.get(r.id) ?? null;
    }
    const search = delta(s0, s1), docs = delta(s1, s2), total = delta(s0, s2);
    return {
      table: ix.table, kind: ix.kind, layout: ix.layout, sql, sqlArgs: extra,
      match: ix.kind === 'fts' ? arg : undefined, params,
      rows, ext,
      stats: { searchMs: t1 - t0, docsMs, ...total, phases: { search, docs } },
    };
  }

  /**
   * Load an index's static data now (see warmSql), so that its first search
   * does not wait for it. Returns {table, kind, ms, rounds, requests, bytes};
   * with an extension build that has no 'warm' command, {skipped: reason}.
   */
  async warm(table) {
    const ix = this.indexes.find((i) => i.table === table) || this.resolve(table);
    const s0 = this.db.stats();
    const t0 = performance.now();
    let skipped;
    try { await this.db.queryRaw(warmSql(ix)); } catch (e) { skipped = String(e.message || e); }
    const d = delta(s0, this.db.stats());
    return { table: ix.table, kind: ix.kind, ms: performance.now() - t0, rounds: d.rounds, requests: d.requests,
             bytes: d.bytes, ...(skipped ? { skipped } : {}) };
  }

  close() { return this.db.close(); }
}
