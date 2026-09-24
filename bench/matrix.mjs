#!/usr/bin/env node
// Benchmark-matrix runner: executes queries through the WebAssembly SQLite
// build (wasm/pkg, Asyncify variant: JSPI is not available in Node 22) against
// a byte-range server, normally netsim/rangeserver.py, and writes one JSON line
// per query with its results, wall time, VFS counters and request log.
//
//   node bench/matrix.mjs JOB.json OUT.jsonl
//
// JOB.json (written by bench/matrix.py):
// {
//   "url": "http://127.0.0.1:8000/words-10k.db",
//   "control": "http://127.0.0.1:8000/__netsim/profile",   // optional
//   "queries": "build/queries/words-10k.json",             // from tools/encode_queries.py
//   "open": { ...options of open() },                      // optional
//   "phases": [
//     { "profile": "4g,h2",                                // posted to control first (optional)
//       "runs": [ { "config": "graph-ef64", "sql": "...", "input": "minilm",
//                   "k": 10, "regime": "warm" | "cold", "qidx": [0, 5, ...],
//                   "warmup": [17], "log": true } ] } ]
// }
//
// Regimes. "cold": every query opens a new database connection (empty VFS
// and SQLite caches, extension static data not loaded), so the measurement
// includes opening the file, reading the schema and any per-connection setup;
// the HTTP connections themselves may be reused (Node's fetch keeps them
// alive). "warm": one connection answers the queries in order after the
// "warmup" queries (not recorded); caches persist across queries, as in a
// browser session.

import { readFileSync, createWriteStream } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { performance } from 'node:perf_hooks';

const REPO = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const { open } = await import(resolve(REPO, 'wasm/pkg/index.mjs'));

const [jobPath, outPath] = process.argv.slice(2);
if (!jobPath || !outPath) {
  console.error('usage: node bench/matrix.mjs JOB.json OUT.jsonl');
  process.exit(2);
}
const job = JSON.parse(readFileSync(jobPath, 'utf8'));
const qmeta = JSON.parse(readFileSync(resolve(REPO, job.queries), 'utf8'));
const qdir = dirname(resolve(REPO, job.queries));

function readF32(file) {
  const b = readFileSync(resolve(qdir, file));
  return new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4);
}
const minilm = readF32(qmeta.minilm.file);
const lateon = readF32(qmeta.lateon.file);
const lateOff = qmeta.lateon.offsets;

function ftsQuery(text, op) {
  const terms = text.split(/\s+/).filter(Boolean).map((t) => `"${t.replace(/"/g, '""')}"`);
  return terms.join(op === 'or' ? ' OR ' : ' ');
}

function bytesOf(f32) {
  return new Uint8Array(f32.buffer, f32.byteOffset, f32.byteLength);
}

function queryParam(input, qi) {
  switch (input) {
    case 'fts_and': return ftsQuery(qmeta.texts[qi], 'and');
    case 'fts_or': return ftsQuery(qmeta.texts[qi], 'or');
    case 'minilm': return bytesOf(minilm.subarray(qi * 384, (qi + 1) * 384));
    case 'lateon': return bytesOf(lateon.subarray(lateOff[qi] * 48, lateOff[qi + 1] * 48));
    default: throw new Error(`unknown input ${input}`);
  }
}

const out = createWriteStream(outPath, { flags: 'a' });
function emit(obj) {
  return new Promise((res) => { if (!out.write(JSON.stringify(obj) + '\n')) out.once('drain', res); else res(); });
}

const STAT_KEYS = ['requests', 'bytes', 'rounds', 'reads', 'cacheHits', 'cacheMisses', 'prefetchCalls',
  'prefetchBlocks', 'specMisses', 'netMs'];

async function runOne(db, run, qi, t0, regime) {
  const param = queryParam(run.input, qi);
  let ids = [], ext = null, err = null;
  const tq = performance.now();
  try {
    const { columns, rows } = await db.queryRaw(run.sql, { q: param, k: run.k });
    const ci = columns.indexOf('id'), cs = columns.indexOf('stats');
    ids = rows.map((r) => r[ci]);
    if (cs >= 0 && rows.length && typeof rows[0][cs] === 'string') {
      try { ext = JSON.parse(rows[0][cs]); delete ext.trace; } catch { ext = null; }
    }
  } catch (e) {
    err = String(e.message || e);
  }
  const tEnd = performance.now();
  const s = db.stats();
  const st = {};
  for (const k of STAT_KEYS) st[k] = s[k];
  const rec = {
    config: run.config, regime, qi, k: run.k, ids, wall_ms: tEnd - t0, query_ms: tEnd - tq, stats: st, ext,
  };
  if (err) rec.error = err;
  if (run.log) {
    rec.t0 = t0; rec.t_end = tEnd;
    rec.log = db.log().map((r) => [r.offset, r.length, r.round, +(r.tStart - t0).toFixed(3), +(r.tEnd - t0).toFixed(3)]);
  }
  return rec;
}

async function setProfile(profile) {
  if (!job.control || !profile) return;
  const body = typeof profile === 'string' ? { preset: profile, seed: job.seed ?? 1 } : profile;
  const r = await fetch(job.control, { method: 'POST', body: JSON.stringify(body) });
  if (!r.ok) throw new Error(`profile ${JSON.stringify(profile)}: HTTP ${r.status}`);
  await r.arrayBuffer();
}

const openOpts = { variant: 'asyncify', ...(job.open || {}) };
for (const phase of job.phases) {
  await setProfile(phase.profile);
  for (const run of phase.runs) {
    const tag = { profile: phase.profile ?? null };
    if (run.regime === 'cold') {
      for (const qi of run.qidx) {
        const t0 = performance.now();
        const db = await open(job.url, openOpts);
        // no resetStats: a new connection starts with zero counters, so the
        // record includes the open (block 0) and the schema reads
        const rec = await runOne(db, run, qi, t0, 'cold');
        await db.close();
        await emit({ ...tag, ...rec });
      }
    } else {
      const db = await open(job.url, openOpts);
      for (const qi of run.warmup || []) {
        await db.queryRaw(run.sql, { q: queryParam(run.input, qi), k: run.k });
      }
      for (const qi of run.qidx) {
        await db.resetStats();
        const t0 = performance.now();
        const rec = await runOne(db, run, qi, t0, 'warm');
        await emit({ ...tag, ...rec });
      }
      await db.close();
    }
  }
}
await new Promise((res) => out.end(res));
