// SQLite worker: opens the database over HTTP range requests with the WASM
// build in wasm/pkg, discovers the search indexes it holds, and runs searches.
//
// Methods:
//   open({url, sqliteModule, ...openOptions}) -> {indexes, docs, stats, openMs, variant}
//   search({table, kind, layout, text | vector, k, params, cold, fetchDocs}) -> result
//   stats() / log() / resetStats({clearCache}) / close()

import { serve } from './rpc.mjs';

let db = null;
let schema = null;

const VFS_KEYS = ['rounds', 'requests', 'bytes', 'netMs', 'reads', 'cacheHits', 'cacheMisses'];

function delta(a, b) {
  const d = {};
  for (const k of VFS_KEYS) d[k] = (b?.[k] ?? 0) - (a?.[k] ?? 0);
  return d;
}

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

async function discover() {
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

function buildSql(ix, k, params) {
  const t = q(ix.table);
  if (ix.kind === 'fts') {
    return { sql: `SELECT rowid AS id, rank AS score FROM ${t} WHERE ${t} MATCH ?1 ORDER BY rank LIMIT ${intParam(k, 'k')}` };
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

async function fetchDocs(ids) {
  const d = schema.docs;
  if (!d || !d.column || !ids.length) return new Map();
  const list = ids.map((x) => intParam(x, 'id')).join(',');
  const lookup = `SELECT ${q(d.column)} FROM ${q(d.table)} WHERE ${d.key === 'rowid' ? 'rowid' : q(d.key)} = ?`;
  // httpvfs_warm (wasm/demo_ext.c) runs the lookups speculatively so that all
  // of them fetch their b-tree pages together, one round per tree level.
  try { await db.query('SELECT httpvfs_warm(?, ?)', [lookup, list]); } catch { /* optional */ }
  const rows = await db.queryRaw(
    `SELECT ${d.key === 'rowid' ? 'rowid' : q(d.key)}, ${q(d.column)} FROM ${q(d.table)} WHERE ${d.key === 'rowid' ? 'rowid' : q(d.key)} IN (${list})`);
  return new Map(rows.rows.map((r) => [r[0], r[1]]));
}

const handlers = {
  async open({ url, sqliteModule, ...opts }) {
    const t0 = performance.now();
    const mod = await import(sqliteModule);
    db = await mod.open(url, opts);
    schema = await discover();
    return { ...schema, variant: db.variant, stats: db.stats(), openMs: performance.now() - t0 };
  },

  async search({ table, text, vector, k = 10, params = {}, cold = false, fetchDocs: wantDocs = true, ftsMode = 'or' }) {
    const ix = schema.indexes.find((i) => i.table === table);
    if (!ix) throw new Error(`no index ${table}`);
    if (cold) await db.resetStats({ clearCache: true });
    const { sql, extra = [] } = buildSql(ix, k, params);
    let arg;
    if (ix.kind === 'fts') {
      arg = ftsQuery(text, params.mode || ftsMode);
      if (!arg) return { rows: [], sql, match: arg, stats: {} };
    } else {
      arg = new Uint8Array(vector.buffer, vector.byteOffset, vector.byteLength);
    }
    const s0 = db.stats();
    const t0 = performance.now();
    const res = await db.queryRaw(sql, [arg, ...extra]);
    const t1 = performance.now();
    const s1 = db.stats();
    const ci = Object.fromEntries(res.columns.map((c, i) => [c, i]));
    const rows = res.rows.map((r) => ({ id: r[ci.id], score: r[ci.score] }));
    let ext = null;
    if (ci.stats !== undefined && res.rows.length) {
      try { ext = JSON.parse(res.rows[0][ci.stats]); } catch { ext = res.rows[0][ci.stats]; }
    }
    let docsMs = 0, s2 = s1;
    if (wantDocs && rows.length) {
      const t2 = performance.now();
      const texts = await fetchDocs(rows.map((r) => r.id));
      docsMs = performance.now() - t2;
      s2 = db.stats();
      for (const r of rows) r.text = texts.get(r.id) ?? null;
    }
    const search = delta(s0, s1), docs = delta(s1, s2), total = delta(s0, s2);
    return {
      table, kind: ix.kind, layout: ix.layout, sql, sqlArgs: extra, match: ix.kind === 'fts' ? arg : undefined,
      rows, ext,
      stats: { searchMs: t1 - t0, docsMs, ...total, phases: { search, docs } },
    };
  },

  async query({ sql, params }) { return db.queryRaw(sql, params); },
  async stats() { return db.stats(); },
  async log() { return db.log(); },
  async resetStats({ clearCache = false } = {}) { await db.resetStats({ clearCache }); return db.stats(); },
  async close() { if (db) await db.close(); db = null; return true; },
};
// Serial: the VFS counter deltas of a search must belong to that search alone.
serve(self, handlers, { serial: true });
