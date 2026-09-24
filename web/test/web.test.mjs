// End-to-end test of the browser demo (Chromium via Playwright):
//   node --test web/test/web.test.mjs
// Serves the repository with netsim/rangeserver.py, opens web/index.html,
//  1. checks that the browser query encoders (onnxruntime-web + tokenizers.js)
//     reproduce enc/minilm.py and enc/lateon.py (token ids exactly; vectors by
//     cosine), and times model loading and encoding;
//  2. runs one query per system through the page and checks the results:
//     FTS5 hits contain the query word, and the dense and late hits equal a
//     native (Python + build/native) search with the same query vectors.
// Environment: WEB_TEST_DB (database path relative to the repo root; default
// build/web/words-10k.db, else build/matrix/words-100.db), WEB_TEST_REPORT
// (JSON report path; default build/web/test-report.json).
import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';
import { execSync, execFileSync } from 'node:child_process';
import { ROOT, startServerProcess } from '../../wasm/test/helpers.mjs';

function loadPlaywright() {
  const require = createRequire(import.meta.url);
  try { return require('playwright'); } catch {}
  return require(path.join(execSync('npm root -g').toString().trim(), 'playwright'));
}

const REF_DIR = path.join(ROOT, 'build/web');
const DB = process.env.WEB_TEST_DB
  || ['build/web/words-10k.db', 'build/matrix/words-10k.db', 'build/matrix/words-100.db']
    .find((p) => fs.existsSync(path.join(ROOT, p)) && !fs.existsSync(path.join(ROOT, p + '-journal')));
const REPORT = process.env.WEB_TEST_REPORT || path.join(REF_DIR, 'test-report.json');
const report = { db: DB, when: new Date().toISOString() };
let server, browser, page;
const browserVectors = {};   // saved for web/test/compare_retrieval.py

before(async () => {
  assert.ok(DB, 'no database: run python3 web/make_demo_db.py');
  if (!fs.existsSync(path.join(REF_DIR, 'reference-embeddings.json'))) {
    execFileSync('python3', [path.join(ROOT, 'web/test/make_reference.py')], { stdio: 'inherit' });
  }
  process.env.PLAYWRIGHT_BROWSERS_PATH ||= '/opt/pw-browsers';
  server = await startServerProcess(ROOT, { latencyMs: 0 });
  const { chromium } = loadPlaywright();
  browser = await chromium.launch();
  page = await browser.newPage();
  page.on('pageerror', (e) => console.log('# pageerror', e.message));
  // A fresh browser profile: models are downloaded, not taken from Cache Storage.
  await page.goto(`${server.url}/web/?db=../${DB}`);
  await page.waitForFunction(() => window.demo && window.demo.ready, null, { timeout: 60000 });
  await page.evaluate(() => window.demo.ready);
  report.userAgent = await page.evaluate(() => navigator.userAgent);
});

after(async () => {
  fs.mkdirSync(path.dirname(REPORT), { recursive: true });
  fs.writeFileSync(REPORT, JSON.stringify(report, null, 1));
  if (browserVectors.minilm) {
    fs.writeFileSync(path.join(REF_DIR, 'browser-embeddings.json'), JSON.stringify(browserVectors));
  }
  await browser?.close();
  server?.close();
});

function cosines(a, b, dim) {
  const out = [];
  for (let i = 0; i < a.length / dim; i++) {
    let d = 0, x = 0, y = 0;
    for (let j = 0; j < dim; j++) { const u = a[i * dim + j], v = b[i * dim + j]; d += u * v; x += u * u; y += v * v; }
    out.push(d / Math.sqrt(x * y));
  }
  return out;
}

function summary(c) {
  const s = [...c].sort((p, q) => p - q);
  return { n: s.length, min: s[0], p1: s[Math.floor(0.01 * (s.length - 1))], median: s[s.length >> 1],
           mean: s.reduce((p, q) => p + q, 0) / s.length, fracGe0999: s.filter((x) => x >= 0.999).length / s.length };
}

const median = (xs) => { const s = [...xs].sort((a, b) => a - b); return s[s.length >> 1]; };

test('browser encoders match the Python reference', async () => {
  const ref = JSON.parse(fs.readFileSync(path.join(REF_DIR, 'reference-embeddings.json')));
  report.encoders = {};
  for (const [which, dim] of [['minilm', 384], ['lateon', 48]]) {
    const texts = ref.queries.map((q) => q.text);
    const r = await page.evaluate(async ({ which, texts }) => {
      const ix = window.demo.ix;
      const load = await ix.loadEncoder(which);
      const out = [];
      for (const t of texts) {
        const e = await ix.encode(t, which);
        out.push({ ids: e.ids, v: Array.from(e.vectors), ms: e.ms, tokenizeMs: e.tokenizeMs, inferMs: e.inferMs });
      }
      return { load, out };
    }, { which, texts });
    const cOpt = [], cNoopt = [];
    ref.queries.forEach((q, i) => {
      const got = r.out[i];
      assert.deepEqual(got.ids, q[`${which}_ids`], `${which} token ids for ${JSON.stringify(q.text)}`);
      const want = q[which].flat(), want0 = q[`${which}_noopt`].flat();
      assert.equal(got.v.length, want.length, `${which} vector count for ${JSON.stringify(q.text)}`);
      cOpt.push(...cosines(got.v, want, dim));
      cNoopt.push(...cosines(got.v, want0, dim));
    });
    browserVectors[which] = r.out.map((o) => o.v);
    const sOpt = summary(cOpt), sNoopt = summary(cNoopt);
    const lens = r.out.map((o) => o.ids.length);
    report.encoders[which] = {
      load: r.load,
      vsPythonDefault: sOpt, vsPythonNoOpt: sNoopt, pythonOptVsNoOpt: ref.python_opt_vs_noopt?.[which],
      encodeMs: { median: median(r.out.map((o) => o.ms)), max: Math.max(...r.out.map((o) => o.ms)),
                  medianInfer: median(r.out.map((o) => o.inferMs)), medianTokenize: median(r.out.map((o) => o.tokenizeMs)) },
      tokens: { median: median(lens), max: Math.max(...lens) },
    };
    console.log(`# ${which}: load ${r.load.totalMs.toFixed(0)} ms (model ${(r.load.modelBytes / 1e6).toFixed(1)} MB in ` +
      `${r.load.modelMs.toFixed(0)} ms, session ${r.load.sessionMs.toFixed(0)} ms); encode median ` +
      `${report.encoders[which].encodeMs.median.toFixed(1)} ms; cosine vs Python: min ${sOpt.min.toFixed(4)}, ` +
      `mean ${sOpt.mean.toFixed(5)}, ${(100 * sOpt.fracGe0999).toFixed(1)}% >= 0.999 (n=${sOpt.n})`);
    // Dynamic int8 quantisation makes exact agreement impossible across
    // kernels (see web/NOTES.md): Python's own fused and unfused graphs differ
    // by up to 0.004. Require close agreement on average and no outliers.
    assert.ok(sOpt.mean >= 0.995, `${which} mean cosine ${sOpt.mean}`);
    assert.ok(sOpt.min >= 0.95, `${which} min cosine ${sOpt.min}`);
  }
});

function native(queries) {
  const out = execFileSync('python3', [path.join(ROOT, 'web/test/native_search.py')],
    { input: JSON.stringify({ db: path.join(ROOT, DB), queries }), maxBuffer: 64 << 20 });
  return JSON.parse(out.toString());
}

test('one query per system through the page', async () => {
  // Two words that each occur in 7 documents of words-10k (in 1 of words-100).
  const [QUERY, WORDS] = DB.includes('words-100') ? ['book great', /\bbook\b|\bgreat\b/] : ['pinwheel gossiping', /\bpinwheel\b|\bgossiping\b/];
  const res = await page.evaluate(async (q) => {
    document.getElementById('q').value = q;
    const r = await window.demo.runSearch(q, 'all');
    // The IVF layout too, if the database has it (the page shows one dense index at a time).
    if (window.demo.ix.indexes.some((i) => i.table === 'dense_ivf')) r.dense_ivf = await window.demo.ix.search(q, { system: 'dense_ivf' });
    for (const v of Object.values(r)) if (v.embedding) v.embedding = Array.from(v.embedding);
    return r;
  }, QUERY);
  report.search = {};
  const nat = [];
  for (const sys of ['fts', 'dense', 'dense_ivf', 'late']) {
    const r = res[sys];
    if (!r) { console.log(`# ${sys}: not in this database`); continue; }
    assert.ok(!r.error, `${sys}: ${r.error}`);
    if (sys === 'fts') assert.ok(r.rows.length >= 1 && r.rows.length <= 10, 'FTS returns 1..k rows');
    else assert.equal(r.rows.length, 10, `${sys} returns k rows`);
    for (const row of r.rows) assert.equal(typeof row.text, 'string', `${sys} row ${row.id} has text`);
    const scores = r.rows.map((x) => x.score);
    const sorted = [...scores].sort((a, b) => (sys === 'late' ? b - a : a - b));
    assert.deepEqual(scores, sorted, `${sys} rows are ranked`);
    if (sys === 'fts') {
      for (const row of r.rows) assert.match(row.text.toLowerCase(), WORDS, 'FTS hit contains a query word');
    }
    nat.push({ sys, sql: r.sql, args: r.kind === 'fts' ? [r.match] : [null, ...r.sqlArgs], vector: r.embedding ?? null, ids: r.rows.map((x) => x.id) });
    const s = r.stats;
    report.search[sys] = { table: r.table, params: r.params, rows: r.rows.map((x) => x.id), stats: s, ext: r.ext };
    console.log(`# ${sys} (${r.table}): wall ${s.wallMs.toFixed(0)} ms, encode ${s.encodeMs.toFixed(1)} ms, sql ${s.searchMs.toFixed(0)} ms, ` +
      `docs ${s.docsMs.toFixed(0)} ms, rounds ${s.phases.search.rounds}+${s.phases.docs.rounds}, ${s.requests} requests, ${(s.bytes / 1024).toFixed(0)} KB`);
  }
  // The WASM extensions must return exactly what the native build returns
  // for the same query vector (and FTS5 for the same MATCH expression).
  const got = native(nat.map(({ sql, args, vector }) => ({ sql, args, vector })));
  nat.forEach((q, i) => assert.deepEqual(q.ids, got[i], `${q.sys}: browser and native results differ`));
});

test('encode latency, warm (20 queries per model)', async () => {
  const texts = JSON.parse(fs.readFileSync(path.join(REF_DIR, 'reference-embeddings.json'))).queries.map((q) => q.text).slice(12, 32);
  // A long query too: a 50-word document (about 110 word pieces).
  const longText = fs.readFileSync(path.join(ROOT, 'data/corpora/words-10k.txt'), 'utf8').split('\n')[0];
  const r = await page.evaluate(async ({ texts, longText }) => {
    const out = {};
    for (const which of ['minilm', 'lateon']) {
      const ms = [];
      for (const t of texts) { const t0 = performance.now(); await window.demo.ix.encode(t, which); ms.push(performance.now() - t0); }
      out[which] = ms;
      const long = [];
      let tokens = 0;
      for (let i = 0; i < 5; i++) { const t0 = performance.now(); tokens = (await window.demo.ix.encode(longText, which)).ids.length; long.push(performance.now() - t0); }
      out[which + '_long'] = { ms: long, tokens };
    }
    return out;
  }, { texts, longText });
  for (const w of ['minilm', 'lateon']) {
    const l = r[w + '_long'];
    delete r[w + '_long'];
    report[`encodeLong_${w}`] = { tokens: l.tokens, median: median(l.ms) };
    console.log(`# ${w}: ${l.tokens}-token query median ${median(l.ms).toFixed(1)} ms`);
  }
  report.encodeRoundTrip = {};
  for (const [w, ms] of Object.entries(r)) {
    report.encodeRoundTrip[w] = { median: median(ms), max: Math.max(...ms), min: Math.min(...ms) };
    console.log(`# ${w}: encode incl. worker round trip median ${median(ms).toFixed(1)} ms (min ${Math.min(...ms).toFixed(1)}, max ${Math.max(...ms).toFixed(1)})`);
  }
});
