// What the demo feels like on a given network: page open, encoder load (cold
// and from Cache Storage after a reload) and query latency per system, in
// Chromium against netsim/rangeserver.py.
//   node web/test/profile_run.mjs [--profiles none,lte,h1;4g,h1] [--db build/web/words-10k.db] [--queries 6]
//        [--params maxRequests=0] [--setup-unshaped]
// Profiles are separated by ';'. Writes build/web/profile-run.json.
// --params: extra page URL parameters (e.g. maxRequests=0 turns the request
// budget off, multipart=1 uses multi-range requests). --setup-unshaped: load
// the page and the encoders without shaping, then set the profile and reopen
// the database (queries only; much faster on slow profiles).
import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';
import { execSync } from 'node:child_process';
import { ROOT, startServerProcess } from '../../wasm/test/helpers.mjs';

const arg = (name, def) => { const i = process.argv.indexOf(`--${name}`); return i > 0 ? process.argv[i + 1] : def; };
const PROFILES = arg('profiles', 'none;lte,h1;4g,h1').split(';');
const DB = arg('db', 'build/web/words-10k.db');
const NQ = Number(arg('queries', '6'));
const OUT = arg('out', path.join(ROOT, 'build/web/profile-run.json'));
const PARAMS = arg('params', '');
const SETUP_UNSHAPED = process.argv.includes('--setup-unshaped');
const QUERIES = fs.readFileSync(path.join(ROOT, 'data/queries/words-10k.jsonl'), 'utf8').trim().split('\n')
  .map((l) => JSON.parse(l)).filter((q) => q.kind !== 'word').map((q) => q.text);

const require = createRequire(import.meta.url);
let pw;
try { pw = require('playwright'); } catch { pw = require(path.join(execSync('npm root -g').toString().trim(), 'playwright')); }
process.env.PLAYWRIGHT_BROWSERS_PATH ||= '/opt/pw-browsers';

const median = (xs) => { const s = [...xs].sort((a, b) => a - b); return s[s.length >> 1]; };
const setProfile = (srv, preset) => fetch(`${srv.url}/__netsim/profile`, { method: 'POST', body: JSON.stringify({ preset }) }).then((r) => r.json());

async function openPage(ctx, srv) {
  const page = await ctx.newPage();
  const t0 = Date.now();
  await page.goto(`${srv.url}/web/?db=../${DB}${PARAMS ? '&' + PARAMS : ''}`);
  await page.waitForFunction(() => window.demo && window.demo.ready, null, { timeout: 600000 });
  await page.evaluate(() => window.demo.ready);
  const openMs = Date.now() - t0;
  const status = await page.evaluate(() => document.getElementById('dbStatus').textContent);
  return { page, openMs, status };
}

async function loadEncoders(page) {
  return page.evaluate(async () => {
    const t0 = performance.now();
    const [m, l] = await Promise.all([window.demo.ix.loadEncoder('minilm'), window.demo.ix.loadEncoder('lateon')]);
    return { bothMs: performance.now() - t0, minilm: m, lateon: l };
  });
}

async function queries(page, qs) {
  return page.evaluate(async (qs) => {
    const ix = window.demo.ix;
    const out = { fts: [], dense: [], late: [] };
    for (const q of qs) {
      for (const system of Object.keys(out)) {
        const r = await ix.search(q, { system });
        const s = r.stats;
        out[system].push({ wall: s.wallMs, enc: s.encodeMs, sql: s.searchMs, docs: s.docsMs,
                           rounds: s.phases.search.rounds + s.phases.docs.rounds, requests: s.requests, bytes: s.bytes });
      }
    }
    return out;
  }, qs);
}

const srv = await startServerProcess(ROOT, { latencyMs: 0 });
const browser = await pw.chromium.launch();
const results = { db: DB, when: new Date().toISOString(), params: PARAMS, setupUnshaped: SETUP_UNSHAPED, profiles: {} };
try {
  for (const prof of PROFILES) {
    await setProfile(srv, 'none');
    const ctx = await browser.newContext();          // empty Cache Storage
    if (!SETUP_UNSHAPED) await setProfile(srv, prof);
    const res = {};
    const a = await openPage(ctx, srv);
    res.pageOpenMs = a.openMs;
    res.dbStatus = a.status;
    res.encodersCold = await loadEncoders(a.page);
    if (SETUP_UNSHAPED) {
      await setProfile(srv, prof);
      res.dbReopenMs = await a.page.evaluate(async () => { const t0 = performance.now(); await window.demo.openDb(); return performance.now() - t0; });
      res.dbStatus = await a.page.evaluate(() => document.getElementById('dbStatus').textContent);
    }
    const q = await queries(a.page, QUERIES.slice(0, NQ));
    res.net = await a.page.evaluate(() => window.demo.ix.sql('netState'));
    res.firstQuery = Object.fromEntries(Object.entries(q).map(([k, v]) => [k, v[0]]));
    res.laterQueries = Object.fromEntries(Object.entries(q).map(([k, v]) => {
      const rest = v.slice(1);
      return [k, Object.fromEntries(Object.keys(rest[0]).map((f) => [f, median(rest.map((x) => x[f]))]))];
    }));
    await a.page.close();
    if (!SETUP_UNSHAPED) {
      const b = await openPage(ctx, srv);             // reload: models come from Cache Storage
      res.pageReopenMs = b.openMs;
      res.encodersCached = await loadEncoders(b.page);
      await b.page.close();
    }
    await ctx.close();
    results.profiles[prof] = res;
    const f = (x) => Math.round(x);
    console.log(`${prof}: page+db open ${f(res.pageOpenMs)} ms; encoders cold ${f(res.encodersCold.bothMs)} ms` +
      (res.encodersCached ? `, cached ${f(res.encodersCached.bothMs)} ms; ` : '; ') +
      Object.entries(res.laterQueries).map(([k, v]) => `${k} first ${f(res.firstQuery[k].wall)} / later ${f(v.wall)} ms (${v.rounds} rounds, ${f(v.bytes / 1024)} KB)`).join('; '));
  }
} finally {
  await setProfile(srv, 'none').catch(() => {});
  await browser.close();
  srv.close();
  fs.mkdirSync(path.dirname(OUT), { recursive: true });
  fs.writeFileSync(OUT, JSON.stringify(results, null, 1));
}
