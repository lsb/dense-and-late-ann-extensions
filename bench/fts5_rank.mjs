#!/usr/bin/env node
// FTS5 ranking-method runner for bench/fts5_rank.py: runs FTS queries through
// the WASM build (Asyncify) and web/lib/search-core.mjs's SearchDb.search
// against a byte-range server, one JSON line per query with the VFS counters
// and the request log.
//
//   node bench/fts5_rank.mjs JOB.json OUT.jsonl
//
// JOB: {url, queries: [{qid, kind, match}], methods: ['bm25', ...],
//       cold: N, warm: N, pageCacheBytes}
// Regimes: "cold" clears the VFS block cache and SQLite's page cache before
// each query (the connection and its parsed schema stay, as for the first
// query after opening); "warm" is one connection per method and query kind
// answering the kind's queries in order after one warm-up query. "session" records one-off costs:
// opening the database, and reading the whole fts_docsize table (method C).

import { readFileSync, createWriteStream } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { performance } from 'node:perf_hooks';

const REPO = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const { open } = await import(resolve(REPO, 'wasm/pkg/index.mjs'));
const { SearchDb } = await import(resolve(REPO, 'web/lib/search-core.mjs'));

const [jobPath, outPath] = process.argv.slice(2);
const job = JSON.parse(readFileSync(jobPath, 'utf8'));
const out = createWriteStream(outPath);
const emit = (o) => new Promise((res) => (out.write(JSON.stringify(o) + '\n') ? res() : out.once('drain', res)));
const openOpts = { variant: 'asyncify', pageCacheBytes: job.pageCacheBytes || 16 << 20 };

function record(db, extra, t0) {
  const tEnd = performance.now();
  const s = db.stats();
  return {
    ...extra, wall_ms: tEnd - t0,
    stats: { rounds: s.rounds, requests: s.requests, bytes: s.bytes, cacheHits: s.cacheHits, cacheMisses: s.cacheMisses },
    log: db.log().map((r) => [r.offset, r.length, r.round, +(r.tStart - t0).toFixed(3), +(r.tEnd - t0).toFixed(3)]),
  };
}

// Group queries by kind, keeping order; take the first n of each kind.
function firstPerKind(n) {
  const seen = new Map();
  return job.queries.filter((q) => { const c = seen.get(q.kind) || 0; seen.set(q.kind, c + 1); return c < n; });
}

async function runQuery(s, q, method, regime) {
  const db = s.db;
  await db.resetStats({ clearCache: regime === 'cold' });
  const t0 = performance.now();
  const r = await s.search({ table: 'fts', match: q.match, k: 10, params: { rank: method }, fetchDocs: false });
  return record(db, { regime, method, kind: q.kind, qid: q.qid, ids: r.rows.map((x) => x.id), ext: r.ext }, t0);
}

// Session costs: open + discover, then a full read of fts_docsize.
if (job.session !== false) {
  let t0 = performance.now();
  const db = await open(job.url, openOpts);
  const s = await SearchDb.open(db);
  await emit(record(db, { regime: 'session', method: 'open', kind: '-' }, t0));
  await db.resetStats({ clearCache: true });
  t0 = performance.now();
  const [row] = await db.query('SELECT count(*) AS n, sum(length(sz)) AS b FROM fts_docsize');
  await emit(record(db, { regime: 'session', method: 'docsize-scan', kind: '-', rows: row.n }, t0));
  // After the scan the rows are cached (if they fit): a bm25 query is then warm.
  if (job.queries.length) {
    await db.resetStats();
    t0 = performance.now();
    const r = await s.search({ table: 'fts', match: job.queries[0].match, k: 10, params: { rank: 'bm25' }, fetchDocs: false });
    await emit(record(db, { regime: 'session', method: 'bm25-after-scan', kind: job.queries[0].kind, ids: r.rows.map((x) => x.id) }, t0));
  }
  await db.close();
}

for (const method of job.methods) {
  const s = await SearchDb.open(await open(job.url, openOpts));
  for (const q of firstPerKind(job.cold)) await emit(await runQuery(s, q, method, 'cold'));
  await s.close();
  // Warm: one connection per kind (so that the AND and OR forms of the same
  // query do not share a cache), warmed up with the kind's last query.
  const qs = firstPerKind(job.warm);
  for (const kind of [...new Set(qs.map((q) => q.kind))]) {
    const kq = qs.filter((q) => q.kind === kind);
    const w = await SearchDb.open(await open(job.url, openOpts));
    const last = job.queries.filter((q) => q.kind === kind).at(-1);
    await w.search({ table: 'fts', match: last.match, k: 10, params: { rank: method }, fetchDocs: false });
    for (const q of kq) await emit(await runQuery(w, q, method, 'warm'));
    await w.close();
  }
  process.stderr.write(`${method} done\n`);
}
await new Promise((res) => out.end(res));
