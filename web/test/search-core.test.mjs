// Node tests for web/lib/search-core.mjs's FTS5 ranking methods over httpvfs
// (docs/fts5-httpvfs.md): the WASM build (Asyncify) against a range server,
// compared with the native CLI (which has ext/fts5rank built in).
//   node --test web/test/search-core.test.mjs      (after make native wasm)
import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { ROOT, CLI, nativeQuery, startServerProcess } from '../../wasm/test/helpers.mjs';
import { open } from '../../wasm/pkg/index.mjs';
import { SearchDb, DEFAULT_PARAMS } from '../lib/search-core.mjs';

// 30,000 documents of 5 to 60 words (lengths vary, so bm25 and bm25c differ);
// 20 % of the words come from 50 common ones (each in about 12 % of the
// documents), the rest from 20,000 (a typical one in about 40 documents,
// scattered over the 75 pages of fts_docsize, as in a large corpus).
function makeDb() {
  const dir = path.join(ROOT, 'build/test');
  const db = path.join(dir, 'fts-rank-test.db');
  if (fs.existsSync(db)) return db;
  fs.mkdirSync(dir, { recursive: true });
  const words = fs.readFileSync(path.join(ROOT, 'data/words/american-english'), 'utf8')
    .split('\n').filter((w) => w && !w.includes("'")).slice(0, 20000).map((w) => w.toLowerCase());
  let x = 987654321;
  const rnd = () => { x ^= x << 13; x >>>= 0; x ^= x >>> 17; x ^= x << 5; x >>>= 0; return x; };
  const lines = ['PRAGMA page_size=4096;', 'BEGIN;', 'CREATE TABLE docs(id INTEGER PRIMARY KEY, body TEXT);'];
  for (let i = 0; i < 30000; i++) {
    const n = 5 + (rnd() % 56), w = [];
    for (let j = 0; j < n; j++) w.push(rnd() % 5 === 0 ? words[rnd() % 50] : words[rnd() % 20000]);
    lines.push(`INSERT INTO docs VALUES(${i}, '${w.join(' ')}');`);
  }
  lines.push("CREATE VIRTUAL TABLE fts USING fts5(body, content='docs', content_rowid='id');",
    "INSERT INTO fts(fts) VALUES('rebuild');", "INSERT INTO fts(fts) VALUES('optimize');", 'COMMIT;', 'VACUUM;');
  execFileSync(CLI, [db + '.tmp'], { input: lines.join('\n') });
  fs.renameSync(db + '.tmp', db);
  return db;
}

let dbPath, server, url, common, sparse;
before(async () => {
  dbPath = makeDb();
  server = await startServerProcess(path.dirname(dbPath));
  url = `${server.url}/${path.basename(dbPath)}`;
  // A common word (hundreds of matches) and a rare one.
  common = nativeQuery(dbPath, "CREATE VIRTUAL TABLE temp.v USING fts5vocab(main, 'fts', 'row'); "
    + 'SELECT term FROM temp.v ORDER BY doc DESC LIMIT 1')[0][0];
  sparse = nativeQuery(dbPath, "CREATE VIRTUAL TABLE temp.v USING fts5vocab(main, 'fts', 'row'); "
    + "SELECT term FROM temp.v WHERE term GLOB '[a-z]*' ORDER BY abs(doc - 40), term LIMIT 1")[0][0];
});
after(() => server?.close());

function native(sql) { return nativeQuery(dbPath, sql); }

test('native: bm25dl with the stored length is exactly bm25; bm25c is bm25 at the average length', () => {
  const m = `'"${common}" OR "zebra"'`;
  const a = native(`SELECT rowid, bm25(fts) FROM fts WHERE fts MATCH ${m} ORDER BY rowid`);
  const b = native(`SELECT rowid, bm25dl(fts, (SELECT sz FROM fts_docsize WHERE id = fts.rowid)) FROM fts WHERE fts MATCH ${m} ORDER BY rowid`);
  const c = native(`SELECT rowid, bm25dl(fts, 40) - bm25c(fts), bm25dl(fts, NULL) - bm25c(fts) FROM fts WHERE fts MATCH ${m} ORDER BY rowid`);
  assert.ok(a.length > 100);
  assert.deepEqual(b, a);
  assert.ok(c.some((r) => Math.abs(r[1]) > 1e-9), 'a fixed length differs from the average');
  assert.ok(c.every((r) => r[2] === 0), 'NULL length means the average');
});

test('SearchDb FTS ranking methods: results and fetch costs', async () => {
  assert.equal(DEFAULT_PARAMS.fts.rank, 'bm25c');
  const s = await SearchDb.open(await open(url, { variant: 'asyncify' }));
  try {
    const text = `${common} zebra`;
    const run = async (rank, cold = true) => s.search({ system: 'fts', text, k: 10, params: rank ? { rank } : {}, cold, fetchDocs: false });
    const def = await run(undefined);
    const r = {};
    for (const rank of ['bm25', 'bm25c', 'bm25-prefetch', 'bm25-rerank', 'none']) r[rank] = await run(rank);
    assert.equal(def.params.rank, 'bm25c');
    assert.deepEqual(def.rows, r.bm25c.rows);
    const ids = (x) => x.rows.map((y) => y.id);
    const m = def.match.replace(/'/g, "''");
    // Each method equals its native SQL.
    assert.deepEqual(ids(r.bm25), native(`SELECT rowid FROM fts WHERE fts MATCH '${m}' ORDER BY rank LIMIT 10`).map((x) => x[0]));
    assert.deepEqual(ids(r.bm25c), native(`SELECT rowid FROM fts WHERE fts MATCH '${m}' AND rank MATCH 'bm25c()' ORDER BY rank LIMIT 10`).map((x) => x[0]));
    assert.deepEqual(ids(r['bm25-rerank']), native(r['bm25-rerank'].sql.replace('?1', `'${m}'`)).map((x) => x[0]));
    // bm25-prefetch is bm25, scores included; bm25-rerank is bm25 restricted to bm25c's top 50.
    assert.deepEqual(r['bm25-prefetch'].rows, r.bm25.rows);
    assert.equal(r['bm25-prefetch'].ext.warmed, true);
    const cand = native(`SELECT rowid FROM fts WHERE fts MATCH '${m}' AND rank MATCH 'bm25c()' ORDER BY rank LIMIT 50`).map((x) => x[0]);
    const exact = native(`SELECT rowid FROM fts WHERE fts MATCH '${m}' AND rowid IN (${cand.join(',')}) ORDER BY rank LIMIT 10`).map((x) => x[0]);
    assert.deepEqual(ids(r['bm25-rerank']), exact);
    for (const [k, v] of Object.entries(r)) if (k !== 'none') {
      const sc = v.rows.map((y) => y.score);
      assert.deepEqual(sc, [...sc].sort((a, b) => a - b), `${k} is ranked`);
    }
    assert.notDeepEqual(ids(r.bm25c), ids(r.bm25), 'lengths vary, so bm25c differs from bm25 here');
    assert.ok(r['bm25-rerank'].ext.candidates === 50);
    // Cold fetch costs for a word in ~40 documents: bm25 reads their
    // fts_docsize rows one round each; the others need at most a few rounds.
    const c = {};
    for (const rank of ['bm25', 'bm25c', 'bm25-prefetch', 'bm25-rerank']) {
      c[rank] = (await s.search({ system: 'fts', text: sparse, params: { rank }, cold: true, fetchDocs: false })).stats.rounds;
    }
    assert.ok(c.bm25 > c.bm25c + 15, JSON.stringify(c));
    assert.ok(c['bm25-prefetch'] <= c.bm25c + 4, JSON.stringify(c));
    assert.ok(c['bm25-rerank'] <= c.bm25c + 4, JSON.stringify(c));
  } finally {
    await s.close();
  }
});

test('SearchDb falls back to bm25 on a SQLite build without bm25c', async () => {
  const db = await open(url, { variant: 'asyncify' });
  const s = await SearchDb.open(db);
  try {
    const orig = db.queryRaw.bind(db);
    db.queryRaw = (sql, p) => (/bm25(c|dl)\(/.test(sql) ? Promise.reject(new Error('no such function: bm25c')) : orig(sql, p));
    const a = await s.search({ system: 'fts', text: common, fetchDocs: false });
    assert.equal(a.params.rank, 'bm25');
    assert.equal(s.noBm25c, true);
    const b = await s.search({ system: 'fts', text: common, params: { rank: 'bm25' }, fetchDocs: false });
    assert.deepEqual(a.rows, b.rows);
  } finally {
    await s.close();
  }
});
