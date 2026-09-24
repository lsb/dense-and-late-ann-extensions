// Runs inside a fresh project that has the packed dense-late-ann installed
// (see run.sh). Serves this directory with a range server and checks:
//  Node:     'dense-late-ann' + 'dense-late-ann/search-core' with the query
//            vectors of reference.json return exactly the expected ids
//            (FTS5, dense graph, dense IVF, late) with the asyncify and sync
//            builds.
//  Chromium: page.html does the same in the page, then openIndex() from
//            'dense-late-ann/search' encodes the queries in the browser
//            (onnxruntime-web and @huggingface/tokenizers from node_modules)
//            and searches through its workers.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { spawn, execSync } from 'node:child_process';
import { createRequire } from 'node:module';
import { open } from 'dense-late-ann';
import { SearchDb } from 'dense-late-ann/search-core';

const rangeserver = process.argv[2];
const ref = JSON.parse(fs.readFileSync('reference.json', 'utf8'));
const SYSTEMS = ['fts', 'dense_graph', 'dense_ivf', 'late'];

function startServer(dir) {
  const args = rangeserver
    ? ['python3', [rangeserver, '--dir', dir, '--port', '0', '--preset', 'none']]
    : null;
  if (!args) throw new Error('usage: node smoke.mjs path/to/netsim/rangeserver.py');
  const child = spawn(args[0], args[1], { stdio: ['ignore', 'pipe', 'inherit'] });
  return new Promise((resolve, reject) => {
    let buf = '';
    child.stdout.on('data', (d) => {
      buf += d;
      const m = /(http:\/\/[\d.]+:\d+)/.exec(buf);
      if (m) resolve({ url: m[1], close: () => child.kill() });
    });
    child.on('exit', (c) => reject(new Error(`server exited ${c}`)));
  });
}

const server = await startServer(process.cwd());
let failed = false;
try {
  // ---------------------------------------------------------------- Node
  for (const variant of ['asyncify', 'sync']) {
    const db = await open(`${server.url}/search.db`, { variant });
    const s = await SearchDb.open(db);
    assert.deepEqual(s.indexes.map((i) => i.table).sort(), [...SYSTEMS].sort());
    for (const q of ref.queries) {
      for (const table of SYSTEMS) {
        const vector = table === 'late' ? Float32Array.from(q.lateon) : table === 'fts' ? undefined : Float32Array.from(q.minilm);
        const r = await s.search({ table, text: q.text, vector, k: 10 });
        assert.deepEqual(r.rows.map((x) => x.id), q.expected[table], `${variant} ${table} "${q.text}"`);
        assert.ok(r.rows.every((x) => typeof x.text === 'string' && x.text.length > 0));
      }
    }
    const st = db.stats();
    console.log(`node ${variant}: ${ref.queries.length} queries x ${SYSTEMS.length} systems identical to native; ${st.requests} requests, ${st.bytes} bytes`);
    await db.close();
  }

  // ---------------------------------------------------------------- Chromium
  const require = createRequire(import.meta.url);
  let pw;
  try { pw = require('playwright'); } catch { pw = require(path.join(execSync('npm root -g').toString().trim(), 'playwright')); }
  const browser = await pw.chromium.launch();
  try {
    const page = await browser.newPage();
    page.on('console', (m) => console.log(`  [page] ${m.text()}`));
    page.on('pageerror', (e) => console.log(`  [pageerror] ${e.message}`));
    await page.goto(`${server.url}/page.html`);
    const out = await page.evaluate(() => window.run(), null);
    for (const [variant, got] of Object.entries(out.exact)) {
      ref.queries.forEach((q, i) => {
        for (const t of SYSTEMS) assert.deepEqual(got[i][t], q.expected[t], `chromium ${variant} ${t} "${q.text}"`);
      });
      console.log(`chromium ${variant}: identical to native with the reference vectors`);
    }
    // In-browser encoding differs slightly from Python's (web/NOTES.md), so
    // require the same FTS5 results and a large top-10 overlap elsewhere.
    let overlap = 0, n = 0;
    ref.queries.forEach((q, i) => {
      assert.deepEqual(out.encoded[i].fts, q.expected.fts);
      for (const t of ['dense_graph', 'dense_ivf', 'late']) {
        overlap += out.encoded[i][t].filter((id) => q.expected[t].includes(id)).length / 10; n++;
        assert.equal(out.encoded[i][t][0], q.expected[t][0], `top hit, browser-encoded ${t} "${q.text}"`);
      }
    });
    console.log(`chromium openIndex (variant ${out.variant}, in-browser encoders): FTS5 identical, top hit identical, mean top-10 overlap ${(overlap / n).toFixed(3)}`);
    assert.ok(overlap / n >= 0.8);
  } finally {
    await browser.close();
  }
  console.log('npm package smoke test: ok');
} catch (e) {
  failed = true;
  console.error(e);
} finally {
  server.close();
}
process.exit(failed ? 1 : 0);
