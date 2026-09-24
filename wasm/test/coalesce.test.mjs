// Tests of the per-round request budget: multi-range parsing and fallback
// (wasm/pkg/multipart.mjs, unit level with a fake fetch), and the WebAssembly
// build under an HTTP/1.1-like server limit of six concurrent requests, with
// range coalescing and with multi-range requests, including servers that do
// not support multi-range requests.
//   node --test wasm/test/coalesce.test.mjs      (after make native wasm)
import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import fs from 'node:fs';
import { makeTestDb, nativeQuery, startServerProcess } from './helpers.mjs';
import { startServer } from './rangeserver.mjs';
import {
  hvParseContentRange, hvMultipartBoundary, hvParseMultipart, hvFetchMultiRange, hvFetchRange, hvGroups,
} from '../pkg/multipart.mjs';
import { open } from '../pkg/index.mjs';

// ------------------------------------------------------------------ unit ---

const enc = (s) => new TextEncoder().encode(s);
function concat(...parts) {
  const bufs = parts.map((p) => (typeof p === 'string' ? enc(p) : p));
  const out = new Uint8Array(bufs.reduce((a, b) => a + b.length, 0));
  let o = 0;
  for (const b of bufs) { out.set(b, o); o += b.length; }
  return out;
}
// A 64 KiB "file" whose bytes contain the boundary string here and there.
const FILE = (() => {
  const f = new Uint8Array(65536);
  for (let i = 0; i < f.length; i++) f[i] = (i * 7 + (i >> 8)) & 0xff;
  f.set(enc('\r\n--BOUND\r\n'), 5000);
  return f;
})();
function multipartBody(boundary, ranges, eol = '\r\n') {
  const parts = [];
  for (const [a, b] of ranges) {
    parts.push(`${eol}--${boundary}${eol}Content-Type: application/octet-stream${eol}` +
      `Content-Range: bytes ${a}-${b}/${FILE.length}${eol}${eol}`, FILE.subarray(a, b + 1));
  }
  parts.push(`${eol}--${boundary}--${eol}`);
  return concat(...parts);
}

test('Content-Range and boundary parsing', () => {
  assert.deepEqual(hvParseContentRange('bytes 0-99/1000'), { start: 0, end: 99, total: 1000 });
  assert.deepEqual(hvParseContentRange('Bytes 5-9/*'), { start: 5, end: 9, total: -1 });
  assert.equal(hvParseContentRange('bytes */1000'), null);
  assert.equal(hvParseContentRange(null), null);
  assert.equal(hvMultipartBoundary('multipart/byteranges; boundary=abc123'), 'abc123');
  assert.equal(hvMultipartBoundary('multipart/byteranges;boundary="a b;c"'), 'a b;c');
  assert.equal(hvMultipartBoundary('application/octet-stream'), null);
  assert.deepEqual(hvGroups(5, [0, 0, 1, 2, 2]), [[0, 1], [2], [3, 4]]);
  assert.deepEqual(hvGroups(3, null), [[0], [1], [2]]);
});

test('multipart/byteranges bodies', () => {
  const ranges = [[4990, 5100], [0, 9], [65000, 65535]];
  for (const eol of ['\r\n', '\n']) {
    const parts = hvParseMultipart(multipartBody('BOUND', ranges, eol), 'BOUND');
    assert.equal(parts.length, 3);
    parts.forEach((p, i) => {
      assert.deepEqual([p.start, p.end, p.total], [...ranges[i], FILE.length]);
      assert.deepEqual(p.data, FILE.subarray(ranges[i][0], ranges[i][1] + 1));
    });
  }
  const body = multipartBody('BOUND', ranges);
  assert.throws(() => hvParseMultipart(body.subarray(0, 200), 'BOUND'), /truncated/);
  assert.throws(() => hvParseMultipart(body, 'OTHER'), /boundary not found/);
  assert.throws(() => hvParseMultipart(enc('--B\r\nContent-Type: x\r\n\r\nabc\r\n--B--'), 'B'), /Content-Range/);
});

// A fake fetch: mode says how the "server" answers multi-range requests.
function fakeFetch(mode, calls) {
  return async (url, init) => {
    const spec = init.headers.Range.replace(/^bytes=/, '');
    calls.push(spec);
    const ranges = spec.split(',').map((r) => r.split('-').map(Number))
      .map(([a, b]) => [a, Math.min(b, FILE.length - 1)]);
    const single = ([a, b]) => new Response(FILE.slice(a, b + 1), { status: 206,
      headers: { 'Content-Range': `bytes ${a}-${b}/${FILE.length}` } });
    if (ranges.length === 1) return single(ranges[0]);
    switch (mode) {
      case 'multipart': {   // reordered, and the first two merged into one part
        const [r0, r1, ...rest] = ranges;
        const parts = [...rest.reverse(), [r0[0], r1[1]]];
        return new Response(multipartBody('xyz', parts), { status: 206,
          headers: { 'Content-Type': 'multipart/byteranges; boundary="xyz"' } });
      }
      case 'span': return single([ranges[0][0], ranges[ranges.length - 1][1]]);
      case 'first': return single(ranges[0]);
      case 'ignore': {
        let cancelled = false;
        const body = new ReadableStream({ pull(c) { c.enqueue(FILE.slice(0, 1024)); }, cancel() { cancelled = true; } });
        const r = new Response(body, { status: 200 });
        r.wasCancelled = () => cancelled;
        calls.last = r;
        return r;
      }
      case 'refuse': return new Response(null, { status: 416, headers: { 'Content-Range': `bytes */${FILE.length}` } });
      default: throw new Error(mode);
    }
  };
}

test('multi-range requests and their fallbacks (fake server)', async () => {
  const offs = [0, 8192, 20480, 40960, 65536 - 4096 + 1000];
  const lens = [4096, 4096, 8192, 4096, 4096];   // the last one runs past the end
  for (const [mode, fallback, requests] of [['multipart', false, 1], ['span', false, 1], ['first', true, 5],
    ['ignore', true, 6], ['refuse', true, 6]]) {
    const calls = [];
    const { results, fallback: fb } = await hvFetchMultiRange(fakeFetch(mode, calls), 'u', {}, offs, lens);
    assert.equal(fb, fallback, mode);
    assert.equal(calls.length, requests, `${mode}: ${calls.join(' | ')}`);
    results.forEach((r, i) => {
      const want = FILE.subarray(offs[i], Math.min(offs[i] + lens[i], FILE.length));
      assert.deepEqual(r.buf, want, `${mode} range ${i}`);
      assert.equal(r.total, FILE.length);
    });
    if (mode === 'ignore') assert.ok(calls.last.wasCancelled(), 'a 200 response is not downloaded');
  }
  // A single range is a plain request; a server ignoring Range still works.
  const calls = [];
  const r = await hvFetchRange(async (u, init) => { calls.push(init.headers.Range); return new Response(FILE.slice()); },
    'u', {}, 100, 50);
  assert.deepEqual(r.buf, FILE.subarray(100, 150));
  assert.deepEqual(calls, ['bytes=100-149']);
});

// ----------------------------------------------------------- end to end ---

const LATENCY = 20;
const IDS = [17, 311, 555, 901, 1234, 1500, 1777, 2020, 2345, 2600, 2900, 3100, 3333, 3600, 3800, 3999];
let dbPath, h1, plain = {};

before(async () => {
  dbPath = makeTestDb();
  // netsim with an HTTP/1.1-like limit: six requests in service, FIFO queue.
  h1 = await startServerProcess(path.dirname(dbPath), { latencyMs: LATENCY, extraArgs: ['--max-concurrent', '6'] });
  for (const multi of ['refuse', 'ignore', 'first']) {
    plain[multi] = await startServer({ dir: path.dirname(dbPath), latencyMs: 5, multi });
  }
});
after(async () => {
  h1?.close();
  for (const s of Object.values(plain)) await s.close();
});

function requestsPerRound(log) {
  const m = new Map();
  for (const e of log) {
    if (!m.has(e.round)) m.set(e.round, new Set());
    m.get(e.round).add(e.req);
  }
  return [...m.values()].map((s) => s.size);
}

// Cold lookups of 16 scattered rows, batched with httpvfs_warm: the leaf
// round has 16 ranges.
async function warmLookups(db) {
  await db.query('SELECT id FROM docs LIMIT 1');
  await db.resetStats({ clearCache: true });
  const t0 = performance.now();
  await db.query('SELECT httpvfs_warm(?, ?) AS passes', ['SELECT body FROM docs WHERE id = ?', IDS.join(',')]);
  const ms = performance.now() - t0;
  const got = (await db.queryRaw(`SELECT id, body FROM docs WHERE id IN (${IDS}) ORDER BY id`)).rows;
  return { ms, got, stats: db.stats(), log: db.log(), net: db.netState() };
}

for (const variant of ['asyncify', 'sync']) {
  test(`${variant}: request budget of 6 under an HTTP/1.1 limit`, async () => {
    const url = `${h1.url}/${path.basename(dbPath)}`;
    const want = nativeQuery(dbPath, `SELECT id, body FROM docs WHERE id IN (${IDS}) ORDER BY id`);
    const common = { variant, readaheadBytes: 0 };
    const cases = {
      baseline: { ...common },   // Node: 'auto' means no budget
      coalesce: { ...common, maxRequests: 6, rttMs: 1000, bandwidthKbps: 100000, netAutoEstimate: false },
      multipart: { ...common, maxRequests: 6, multipart: true },
    };
    const res = {};
    for (const [name, opts] of Object.entries(cases)) {
      const db = await open(url, opts);
      try {
        res[name] = await warmLookups(db);
      } finally {
        await db.close();
      }
      const r = res[name];
      assert.deepEqual(r.got, want, name);
      const per = requestsPerRound(r.log);
      assert.equal(r.stats.requests, per.reduce((a, b) => a + b, 0), `${name}: requests counted per request`);
      assert.equal(r.stats.bytes, r.log.reduce((a, e) => a + e.length, 0));
      if (name === 'baseline') {
        assert.ok(Math.max(...per) >= 12, `baseline has a big round: ${per}`);
        assert.equal(r.net.maxRequests, 0);
      } else {
        assert.ok(Math.max(...per) <= 6, `${name}: requests per round ${per}`);
        assert.ok(r.net.plannedRounds >= 1);
      }
      if (name === 'coalesce') assert.ok(r.net.overfetchBytes > 0);
      if (name === 'multipart') {
        assert.ok(r.net.multipartRequests >= 1 && !r.net.multipartFailed, JSON.stringify(r.net));
        assert.equal(r.net.overfetchBytes, 0);
      }
    }
    console.log(`# ${variant}, ${LATENCY} ms, 6 concurrent: ` + Object.entries(res).map(([k, r]) =>
      `${k} ${r.ms.toFixed(0)} ms (${r.stats.rounds} rounds, ${r.stats.requests} requests, ` +
      `${(r.stats.bytes / 1024).toFixed(0)} KB)`).join('; '));
  });
}

test('multi-range fallback on servers without multipart support', async () => {
  const want = nativeQuery(dbPath, `SELECT id, body FROM docs WHERE id IN (${IDS}) ORDER BY id`);
  for (const [multi, srv] of Object.entries(plain)) {
    const db = await open(`${srv.url}/${path.basename(dbPath)}`, { readaheadBytes: 0, maxRequests: 6, multipart: true });
    try {
      srv.log.length = 0;
      const r = await warmLookups(db);
      assert.deepEqual(r.got, want, multi);
      assert.ok(r.net.multipartFailed, `${multi}: fallback detected`);
      // Only one multi-range request was ever sent; no full-file download
      // was completed.
      assert.equal(srv.log.filter((e) => e.range?.includes(',')).length, 1, multi);
      // Afterwards, single ranges only (coalesced when the cost model says so).
      await db.resetStats({ clearCache: true });
      const r2 = await warmLookups(db);
      assert.deepEqual(r2.got, want);
      assert.equal(r2.stats.requests, r2.log.length);
      assert.equal(srv.log.filter((e) => e.range?.includes(',')).length, 1, multi);
    } finally {
      await db.close();
    }
  }
});

test('sync variant: fallback when the server refuses multi-range', async () => {
  const srv = await startServerProcess(path.dirname(dbPath), { latencyMs: 5, preferNode: true, extraArgs: ['--multi', 'ignore'] });
  try {
    const db = await open(`${srv.url}/${path.basename(dbPath)}`, { variant: 'sync', readaheadBytes: 0, maxRequests: 6, multipart: true });
    try {
      const r = await warmLookups(db);
      assert.deepEqual(r.got, nativeQuery(dbPath, `SELECT id, body FROM docs WHERE id IN (${IDS}) ORDER BY id`));
      assert.ok(r.net.multipartFailed);
    } finally {
      await db.close();
    }
  } finally {
    srv.close();
  }
});

test('setNetOptions and the estimator', async () => {
  const db = await open(`${h1.url}/${path.basename(dbPath)}`, { readaheadBytes: 256 << 10 });
  try {
    await db.setNetOptions({ maxRequests: 6, rttMs: 500, bandwidthKbps: 1000 });
    let n = db.netState();
    assert.equal(n.maxRequests, 6);
    assert.equal(n.rttMs, 500);
    // A scan with readahead makes rounds of 1, 2, 4, ... 64 blocks: enough
    // spread in size to fit both. The server adds 20 ms per response and
    // does not limit bandwidth (localhost).
    await db.query('SELECT sum(length(body)) FROM docs');
    n = db.netState();
    assert.ok(n.observedRounds >= 4);
    assert.ok(n.rttMs > LATENCY * 0.8 && n.rttMs < LATENCY * 3, `rtt estimate ${n.rttMs}`);
    assert.ok(n.bandwidthKbps > 50000, `bandwidth estimate ${n.bandwidthKbps}`);
    console.log(`# estimator: rtt ${n.rttMs.toFixed(1)} ms, ${(n.bandwidthKbps / 1000).toFixed(0)} Mbit/s from ${n.observedRounds} rounds`);
  } finally {
    await db.close();
  }
});
