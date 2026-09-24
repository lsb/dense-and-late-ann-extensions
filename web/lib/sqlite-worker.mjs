// SQLite worker: opens the database over HTTP range requests with the WASM
// build in wasm/pkg, discovers the search indexes it holds, and runs searches
// (the search logic itself is in search-core.mjs, which also runs in Node).
//
// Methods:
//   open({url, sqliteModule, ...openOptions}) -> {indexes, docs, stats, openMs, variant}
//   search({table, text | vector, k, params, cold, fetchDocs}) -> result
//   warm({table}) -> {table, ms, rounds, requests, bytes}: load the index's
//     per-connection static data now (queued behind any search)
//   query({sql, params}) / stats() / log() / resetStats({clearCache}) / close()

import { serve } from './rpc.mjs';
import { SearchDb } from './search-core.mjs';

let db = null;
let sdb = null;

const handlers = {
  async open({ url, sqliteModule, ...opts }) {
    const t0 = performance.now();
    const mod = await import(sqliteModule);
    db = await mod.open(url, opts);
    sdb = await SearchDb.open(db);
    return { indexes: sdb.indexes, docs: sdb.docs, variant: db.variant, stats: db.stats(), openMs: performance.now() - t0 };
  },
  async search(args) { return sdb.search(args); },
  async warm({ table }) { return sdb.warm(table); },
  async query({ sql, params }) { return db.queryRaw(sql, params); },
  async stats() { return db.stats(); },
  async log() { return db.log(); },
  async resetStats({ clearCache = false } = {}) { await db.resetStats({ clearCache }); return db.stats(); },
  async close() { if (db) await db.close(); db = null; sdb = null; return true; },
};
// Serial: the VFS counter deltas of a search must belong to that search alone.
// Warm-ups are background work: a search that arrives while they are queued
// runs first (one that is already running is not interrupted).
serve(self, handlers, { serial: true, priority: (method) => (method === 'warm' ? 1 : 0) });
