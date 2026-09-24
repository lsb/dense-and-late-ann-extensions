// SQLite worker: opens the database over HTTP range requests with the WASM
// build in wasm/pkg, discovers the search indexes it holds, and runs searches
// (the search logic itself is in search-core.mjs, which also runs in Node).
//
// Methods:
//   open({url, sqliteModule, ...openOptions}) -> {indexes, docs, stats, net, openMs, variant}
//   search({table, text | vector, k, params, cold, fetchDocs}) -> result
//   query({sql, params}) / stats() / log() / netState() / setNetOptions(o) /
//   resetStats({clearCache}) / close()

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
    return { indexes: sdb.indexes, docs: sdb.docs, variant: db.variant, stats: db.stats(), net: db.netState?.() ?? null,
             openMs: performance.now() - t0 };
  },
  async search(args) { return sdb.search(args); },
  async query({ sql, params }) { return db.queryRaw(sql, params); },
  async stats() { return db.stats(); },
  async log() { return db.log(); },
  async netState() { return db.netState?.() ?? null; },
  async setNetOptions(o) { await db.setNetOptions(o); return db.netState(); },
  async resetStats({ clearCache = false } = {}) { await db.resetStats({ clearCache }); return db.stats(); },
  async close() { if (db) await db.close(); db = null; sdb = null; return true; },
};
// Serial: the VFS counter deltas of a search must belong to that search alone.
serve(self, handlers, { serial: true });
