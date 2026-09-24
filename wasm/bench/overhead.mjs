// CPU overhead of the WebAssembly builds against native SQLite, for small FTS5
// queries on build/test/fts-test.db with a warm cache (no network in the
// timed loop). Node: asyncify and sync builds. Chromium (Playwright, in a
// Worker): asyncify, jspi and sync builds. Native: Python sqlite3 on the same
// SQLite version. Prints a JSON report.
//   node wasm/bench/overhead.mjs [--no-browser] [--out results/wasm-overhead.json]
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { createRequire } from 'node:module';
import { ROOT, makeTestDb, startServerProcess } from '../test/helpers.mjs';
import { BENCH_QUERIES, timeQueries } from './queries.mjs';
import { open } from '../pkg/index.mjs';

const dbPath = makeTestDb();
const server = await startServerProcess(ROOT, { isolate: true });
const dbUrl = `${server.url}/${path.relative(ROOT, dbPath)}`;
const report = { db: path.relative(ROOT, dbPath), queries: BENCH_QUERIES };

const native = JSON.parse(execFileSync('python3', [
  path.join(ROOT, 'wasm/bench/native_overhead.py'), dbPath, JSON.stringify(BENCH_QUERIES),
  path.join(ROOT, 'build/native/httpvfs')],
  { env: { ...process.env, LD_LIBRARY_PATH: path.join(ROOT, 'build/native') } }).toString());
Object.assign(report, native);

for (const variant of ['asyncify', 'sync']) {
  const db = await open(dbUrl, { variant, pageCacheBytes: 16 << 20 });
  report[`node_${variant}`] = await timeQueries((sql) => db.queryRaw(sql));
  await db.close();
}

if (!process.argv.includes('--no-browser')) {
  process.env.PLAYWRIGHT_BROWSERS_PATH ||= '/opt/pw-browsers';
  const require = createRequire(import.meta.url);
  let pw;
  try { pw = require('playwright'); } catch {
    pw = require(path.join(execFileSync('npm', ['root', '-g']).toString().trim(), 'playwright'));
  }
  const browser = await pw.chromium.launch();
  for (const variant of ['asyncify', 'jspi', 'sync']) {
    const page = await browser.newPage();
    await page.goto(`${server.url}/wasm/bench/bench.html?variant=${variant}&db=/${report.db}`);
    const r = await page.evaluate(() => window.benchResult);
    if (r.error) throw new Error(r.error);
    report[`chromium_${variant}`] = r.timings;
    await page.close();
  }
  await browser.close();
}
server.close();

// Summary: median ms per query and ratio to native (default VFS).
const rows = [];
for (const [k, v] of Object.entries(report)) {
  if (!v?.fts_term_top10?.median_ms) continue;
  const row = { config: k };
  for (const q of Object.keys(BENCH_QUERIES)) {
    row[q] = `${v[q].median_ms.toFixed(3)} (${(v[q].median_ms / report.native_default_vfs[q].median_ms).toFixed(2)}x)`;
  }
  rows.push(row);
}
console.table(rows);
report.date = new Date().toISOString();
report.loadavg = (await import('node:os')).loadavg();
const outIdx = process.argv.indexOf('--out');
if (outIdx > 0) (await import('node:fs')).writeFileSync(process.argv[outIdx + 1], JSON.stringify(report, null, 1) + '\n');
else console.log(JSON.stringify(report, null, 1));
