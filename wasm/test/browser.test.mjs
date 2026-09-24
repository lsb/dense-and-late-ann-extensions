// Chromium (Playwright) tests: the same database and checks as the Node test,
// run inside a browser Worker for each build variant.
//   node --test wasm/test/browser.test.mjs    (after make native wasm)
// Uses the globally installed playwright package and the preinstalled browsers
// (PLAYWRIGHT_BROWSERS_PATH); it does not download anything.
import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import { createRequire } from 'node:module';
import { execSync } from 'node:child_process';
import { ROOT, makeTestDb, nativeQuery, startServerProcess } from './helpers.mjs';

function loadPlaywright() {
  const require = createRequire(import.meta.url);
  try { return require('playwright'); } catch {}
  const globalRoot = execSync('npm root -g').toString().trim();
  return require(path.join(globalRoot, 'playwright'));
}

const LATENCY = 20;
let dbPath, server, plainServer, browser;

before(async () => {
  process.env.PLAYWRIGHT_BROWSERS_PATH ||= '/opt/pw-browsers';
  dbPath = makeTestDb();
  // Serve the repository root with COOP/COEP so that the "sync" variant can
  // use SharedArrayBuffer.
  server = await startServerProcess(ROOT, { latencyMs: LATENCY, isolate: true });
  plainServer = await startServerProcess(ROOT, { latencyMs: LATENCY });
  const { chromium } = loadPlaywright();
  browser = await chromium.launch();
});
after(async () => { await browser?.close(); server?.close(); plainServer?.close(); });

async function runVariant(variant, srv = server) {
  const page = await browser.newPage();
  const errors = [];
  page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
  const db = '/' + path.relative(ROOT, dbPath);
  await page.goto(`${srv.url}/wasm/test/browser/index.html?variant=${variant}&db=${db}`);
  const r = await page.evaluate(() => window.testResult);
  await page.close();
  if (r.error) assert.fail(`${variant}: ${r.error}\n${errors.join('\n')}`);
  return r;
}

function check(r) {
  for (const { q, rows } of r.fts) assert.deepEqual(rows, nativeQuery(dbPath, q), q);
  assert.equal(r.hello, 'hello, browser');
  assert.equal(r.stats.requests, r.log.length);
  assert.equal(r.stats.bytes, r.log.reduce((a, e) => a + e.length, 0));
  assert.equal(new Set(r.log.map((e) => e.round)).size, r.stats.rounds);
  const ids = r.sequential.rows.map((row) => row[0]);
  const want = nativeQuery(dbPath, `SELECT id, body FROM docs WHERE id IN (${ids}) ORDER BY id`);
  assert.deepEqual(r.sequential.rows, want);
  assert.deepEqual(r.batched.rows, want);
  assert.ok(r.sequential.rounds >= ids.length);
  assert.ok(r.batched.rounds <= 4);
  const counts = new Map();
  for (const e of r.batched.log) counts.set(e.round, (counts.get(e.round) || 0) + 1);
  const biggest = Math.max(...counts.values());
  assert.ok(biggest >= 8, `largest round ${biggest}`);
  console.log(`# ${r.variant} (isolated=${r.isolated}): sequential ${r.sequential.rounds} rounds ` +
    `${r.sequential.ms.toFixed(0)} ms, batched ${r.batched.rounds} rounds ${r.batched.ms.toFixed(0)} ms, ` +
    `largest round ${biggest} requests`);
}

test('chromium worker: asyncify build', async () => check(await runVariant('asyncify')));
test('chromium worker: asyncify build without cross-origin isolation', async () => {
  const r = await runVariant('asyncify', plainServer);
  assert.equal(r.isolated, false);
  check(r);
});
test('chromium worker: jspi build', async () => check(await runVariant('jspi')));
test('chromium worker: auto picks jspi', async () => {
  const r = await runVariant('auto');
  assert.equal(r.resolvedVariant, 'jspi');
  check(r);
});
test('chromium worker: sync build (SharedArrayBuffer + Atomics.wait)', async () => check(await runVariant('sync')));
