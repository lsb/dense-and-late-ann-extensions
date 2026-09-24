// Node tests for the WebAssembly build: a database made natively, served by a
// range server in another process, queried through the httpvfs VFS.
//   node --test wasm/test/node.test.mjs      (after make native wasm)
import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import { makeTestDb, nativeQuery, startServerProcess } from './helpers.mjs';
import { open } from '../pkg/index.mjs';

const LATENCY = 20;   // ms per response, so that parallelism is visible in timings
let dbPath, server, dbUrl;

before(async () => {
  dbPath = makeTestDb();
  server = await startServerProcess(path.dirname(dbPath), { latencyMs: LATENCY });
  dbUrl = `${server.url}/${path.basename(dbPath)}`;
});
after(() => server?.close());

const FTS_QUERIES = [
  "SELECT rowid, rank FROM docs_fts WHERE docs_fts MATCH 'baathist' ORDER BY rank LIMIT 10",
  "SELECT rowid FROM docs_fts WHERE docs_fts MATCH 'bab* OR bac*' ORDER BY rowid LIMIT 20",
  "SELECT count(*) FROM docs_fts WHERE docs_fts MATCH 'a*'",
  "SELECT d.id, d.title FROM docs_fts JOIN docs d ON d.id = docs_fts.rowid WHERE docs_fts MATCH 'baal OR baath' ORDER BY docs_fts.rank, d.id LIMIT 5",
];

function checkStatsAndLog(db) {
  const s = db.stats();
  const log = db.log();
  assert.equal(s.requests, log.length, 'one log entry per request');
  assert.equal(s.bytes, log.reduce((a, e) => a + e.length, 0), 'bytes match the log');
  const rounds = new Set(log.map((e) => e.round));
  assert.equal(rounds.size, s.rounds, 'rounds match the log');
  for (let i = 1; i < log.length; i++) assert.ok(log[i].round >= log[i - 1].round);
  for (const e of log) assert.ok(e.tEnd >= e.tStart && e.length > 0 && e.offset % 4096 === 0);
  return { s, log };
}

for (const variant of ['asyncify', 'sync']) {
  test(`${variant}: FTS5 results match native SQLite`, async () => {
    const db = await open(dbUrl, { variant });
    try {
      for (const q of FTS_QUERIES) {
        const got = (await db.queryRaw(q)).rows;
        const want = nativeQuery(dbPath, q);
        assert.deepEqual(got, want, q);
      }
      assert.deepEqual(await db.query("SELECT hello('wasm') AS h, sqlite_version() AS v"),
        [{ h: 'hello, wasm', v: '3.53.4' }]);
      const { s } = checkStatsAndLog(db);
      assert.ok(s.requests > 0 && s.rounds > 0 && s.bytes >= s.requests * 4096);
      assert.equal(s.fileSize, (await import('node:fs')).statSync(dbPath).size);
    } finally {
      await db.close();
    }
  });

  test(`${variant}: parallel prefetch collapses rounds`, async () => {
    const db = await open(dbUrl, { variant, readaheadBytes: 0 });
    try {
      await db.query('SELECT id FROM docs LIMIT 1');  // load the schema
      const ids = [17, 311, 555, 901, 1234, 1500, 1777, 2020, 2345, 2600, 2900, 3100, 3333, 3600, 3800, 3999];
      const want = nativeQuery(dbPath, `SELECT id, body FROM docs WHERE id IN (${ids}) ORDER BY id`);

      // Cold, one lookup at a time: each lookup waits for its own leaf page.
      await db.resetStats({ clearCache: true });
      let t0 = performance.now();
      const seq = [];
      for (const id of ids) seq.push((await db.queryRaw('SELECT id, body FROM docs WHERE id = ?', [id])).rows[0]);
      const tSeq = performance.now() - t0;
      const seqRounds = db.stats().rounds;
      assert.deepEqual(seq, want);
      assert.ok(seqRounds >= ids.length, `sequential rounds ${seqRounds}`);

      // Cold, speculative batching first: one round per uncached B-tree level.
      await db.resetStats({ clearCache: true });
      t0 = performance.now();
      const [{ passes }] = await db.query(
        'SELECT httpvfs_warm(?, ?) AS passes', ['SELECT body FROM docs WHERE id = ?', ids.join(',')]);
      const warmRounds = db.stats().rounds;
      const batched = [];
      for (const id of ids) batched.push((await db.queryRaw('SELECT id, body FROM docs WHERE id = ?', [id])).rows[0]);
      const tBatch = performance.now() - t0;
      assert.deepEqual(batched, want);
      const { s, log } = checkStatsAndLog(db);
      assert.equal(s.rounds, warmRounds, 'lookups after warming hit the cache');
      assert.ok(s.rounds <= 4, `batched rounds ${s.rounds} (passes ${passes})`);
      // The biggest round fetched many ranges concurrently.
      const byRound = new Map();
      for (const e of log) byRound.set(e.round, [...(byRound.get(e.round) || []), e]);
      const big = [...byRound.values()].sort((a, b) => b.length - a.length)[0];
      assert.ok(big.length >= 8, `largest round has ${big.length} requests`);
      const span = Math.max(...big.map((e) => e.tEnd)) - Math.min(...big.map((e) => e.tStart));
      assert.ok(span < big.length * LATENCY / 2, `round of ${big.length} took ${span.toFixed(1)} ms`);
      console.log(`# ${variant}: ${ids.length} cold lookups: sequential ${seqRounds} rounds ` +
        `${tSeq.toFixed(0)} ms; batched ${s.rounds} rounds ${tBatch.toFixed(0)} ms ` +
        `(largest round: ${big.length} requests in ${span.toFixed(1)} ms)`);
    } finally {
      await db.close();
    }
  });
}

test('httpvfs_prefetch by byte range and sequential readahead', async () => {
  const db = await open(dbUrl, { readaheadBytes: 1 << 20 });
  try {
    await db.resetStats({ clearCache: true });
    const [{ n, total }] = await db.query('SELECT count(*) AS n, sum(length(body)) AS total FROM docs');
    const want = nativeQuery(dbPath, 'SELECT count(*), sum(length(body)) FROM docs')[0];
    assert.deepEqual([n, total], want);
    const s = db.stats();
    // A full scan reads hundreds of pages; readahead should need far fewer requests.
    assert.ok(s.cacheMisses + s.cacheHits > 100);
    assert.ok(s.requests < (s.bytes / 4096) / 4, `requests ${s.requests} for ${s.bytes} bytes`);
  } finally {
    await db.close();
  }
});

test('parameters, blobs and errors', async () => {
  const db = await open(dbUrl);
  try {
    const rows = await db.query('SELECT id, data FROM blobs WHERE id BETWEEN :lo AND :hi ORDER BY id', { lo: 10, hi: 12 });
    assert.equal(rows.length, 3);
    assert.ok(rows[0].data instanceof Uint8Array);
    assert.equal(rows[0].data.length, 100 + 10 * 7);
    const hex = nativeQuery(dbPath, 'SELECT hex(data) FROM blobs WHERE id = 10')[0][0];
    assert.equal(Buffer.from(rows[0].data).toString('hex').toUpperCase(), hex);
    const [{ x }] = await db.query('SELECT ? AS x', [new Uint8Array([1, 2, 3])]);
    assert.deepEqual([...x], [1, 2, 3]);
    await assert.rejects(db.query('SELECT * FROM no_such_table'), /no such table/);
    await assert.rejects(db.exec('CREATE TABLE t(x)'), /readonly|read-only/i);
  } finally {
    await db.close();
  }
  await assert.rejects(open(`${server.url}/does-not-exist.db`), /unable to open.*HTTP 404/);
});

test('concurrent queries are serialised', async () => {
  const db = await open(dbUrl);
  try {
    const qs = [1, 2, 3, 4, 5].map((i) => db.query('SELECT title FROM docs WHERE id = ?', [i * 100]));
    const res = await Promise.all(qs);
    assert.deepEqual(res.map((r) => r[0].title), ['doc 100', 'doc 200', 'doc 300', 'doc 400', 'doc 500']);
  } finally {
    await db.close();
  }
});
