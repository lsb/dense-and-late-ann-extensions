// Time to first result in Chromium: open a page (web/test/ttfr.html) that
// opens a database, starts loading the query encoder and searches as soon as
// the index is open; the result time is measured from navigation start.
// Compares databases and warm-up modes under network profiles, on a first
// visit (empty Cache Storage: the encoder is downloaded) and a repeat visit
// (encoder files from Cache Storage; page, SQLite and ONNX Runtime code are
// fetched again, as the range server sends no-store).
//
//   node web/test/ttfr.mjs --dbs 'before=build/x/old.db;after=build/x/new.db' \
//        --variants 'before:off,before:auto,after:auto' --systems dense,dense_ivf,late \
//        --profiles '4g,h1;lte,h1;4g,h2;lte,h2' --visits first,repeat --reps 3 --out build/web/ttfr.json
//
// h1 profiles talk HTTP/1.1 to netsim/rangeserver.py (Chromium opens at most
// six connections per host); h2 profiles go through web/test/h2proxy.mjs, a
// real HTTP/2 (TLS) front whose upstream range server runs the ',h2' preset.
import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';
import { execSync } from 'node:child_process';
import { ROOT, startServerProcess } from '../../wasm/test/helpers.mjs';
import { startH2Proxy } from './h2proxy.mjs';

const arg = (name, def) => { const i = process.argv.indexOf(`--${name}`); return i > 0 ? process.argv[i + 1] : def; };
const DBS = Object.fromEntries(arg('dbs', 'db=build/web/words-10k.db').split(';').map((kv) => kv.split('=')));
const VARIANTS = arg('variants', Object.keys(DBS).flatMap((d) => [`${d}:off`, `${d}:auto`]).join(',')).split(',')
  .map((v) => { const [db, warm] = v.split(':'); return { db, warm, id: v }; });
const SYSTEMS = arg('systems', 'dense,dense_ivf,late').split(',');
const PROFILES = arg('profiles', '4g,h1;lte,h1;4g,h2;lte,h2').split(';');
const VISITS = arg('visits', 'first,repeat').split(',');
const REPS = Number(arg('reps', '3'));
const QUERY = arg('q', 'volcanic eruption');
const DELAYS = arg('delays', '0').split(',').map(Number);   // ms between index open and query submission
const OUT = arg('out', path.join(ROOT, 'build/web/ttfr.json'));

const require = createRequire(import.meta.url);
let pw;
try { pw = require('playwright'); } catch { pw = require(path.join(execSync('npm root -g').toString().trim(), 'playwright')); }
process.env.PLAYWRIGHT_BROWSERS_PATH ||= '/opt/pw-browsers';

const median = (xs) => { const s = [...xs].sort((a, b) => a - b); return s.length ? s[s.length >> 1] : null; };

const srv = await startServerProcess(ROOT, { latencyMs: 0 });
const proxy = await startH2Proxy(srv.url);
const setProfile = (preset) => fetch(`${srv.url}/__netsim/profile`, { method: 'POST', body: JSON.stringify({ preset, seed: 1 }) })
  .then((r) => r.json());
const browser = await pw.chromium.launch();

async function run(ctx, base, v, system) {
  const page = await ctx.newPage();
  page.on('pageerror', (e) => console.log('# pageerror', e.message));
  const q = new URLSearchParams({ db: `/${DBS[v.db]}`, system, q: QUERY, warm: v.warm, delay: String(v.delay || 0) });
  await page.goto(`${base}/web/test/ttfr.html?${q}`);
  const r = await page.waitForFunction(() => window.ttfr, null, { timeout: 900000 })
    .then(() => page.evaluate(() => window.ttfr));
  await page.close();
  if (r.error) throw new Error(`${v.id} ${system}: ${r.error}`);
  return r;
}

const results = { query: QUERY, dbs: DBS, when: new Date().toISOString(), runs: [] };
try {
  for (const prof of PROFILES) {
    const base = prof.includes('h2') ? proxy.url : srv.url;
    for (const visit of VISITS) {
      for (const system of SYSTEMS) {
        for (const delay of DELAYS) for (const v of VARIANTS) {
          const reps = visit === 'first' ? 1 : REPS;
          for (let rep = 0; rep < reps; rep++) {
            const ctx = await browser.newContext({ ignoreHTTPSErrors: true });
            if (visit === 'repeat') {                 // prime Cache Storage at full speed
              await setProfile('none');
              await run(ctx, base, v, system);
            }
            await setProfile(prof);
            const r = await run(ctx, base, { ...v, delay }, system);
            await ctx.close();
            const rec = {
              profile: prof, visit, system, variant: v.id, delay, rep, ttfr: r.marks.result, openAt: r.marks.open,
              submitToResult: r.marks.result - r.marks.submit,
              encoderMs: r.encoder?.totalMs, encoderFromCache: r.encoder?.modelFromCache,
              encoderLoadWaitMs: r.stats.encoderLoadWaitMs, searchMs: r.stats.searchMs, docsMs: r.stats.docsMs,
              rounds: r.stats.rounds, bytes: r.stats.bytes, staticRounds: r.ext?.static_rounds ?? r.ext?.setup_rounds,
              warm: r.warm, rows: r.rows,
            };
            results.runs.push(rec);
            console.log(`${prof} ${visit} ${system} ${v.id} delay ${delay}#${rep}: first result at ${rec.ttfr.toFixed(0)} ms, ${rec.submitToResult.toFixed(0)} ms after submit ` +
              `(open ${rec.openAt.toFixed(0)}, encoder ${rec.encoderMs?.toFixed(0)} ms, waited ${rec.encoderLoadWaitMs?.toFixed(0)}, ` +
              `search ${rec.searchMs.toFixed(0)} ms / ${rec.rounds} rounds / ${(rec.bytes / 1024).toFixed(0)} KB, ` +
              `warm ${r.warm.map((w) => `${w.table} ${(w.ms || 0).toFixed(0)} ms ${((w.bytes || 0) / 1024).toFixed(0)} KB${w.skipped ? ' skipped' : ''}`).join(', ')})`);
            fs.mkdirSync(path.dirname(OUT), { recursive: true });
            fs.writeFileSync(OUT, JSON.stringify(results, null, 1));
          }
        }
      }
    }
  }
} finally {
  await setProfile('none').catch(() => {});
  await browser.close();
  proxy.close();
  srv.close();
}

// Summary: median time to first result per profile / visit / system / variant.
// Delay 0: from navigation start; delay > 0: from query submission.
const groups = {};
for (const r of results.runs) {
  const g = (groups[`${r.profile}|${r.visit}|${r.system}|${r.delay}`] ||= {});
  (g[r.variant] ||= []).push(r.delay ? r.submitToResult : r.ttfr);
}
console.log(`\n| Profile | Visit | System | Delay | ${VARIANTS.map((v) => v.id).join(' | ')} |`);
console.log(`|---|---|---|---|${VARIANTS.map(() => '---').join('|')}|`);
for (const [k, g] of Object.entries(groups)) {
  const [p, vis, s, d] = k.split('|');
  console.log(`| ${p} | ${vis} | ${s} | ${d} | ${VARIANTS.map((v) => (g[v.id] ? (median(g[v.id]) / 1000).toFixed(2) + ' s' : '–')).join(' | ')} |`);
}
