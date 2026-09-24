// Fetch worker for sync-fetch.mjs: fetches a batch of byte ranges in
// parallel (one request per group of ranges; a group of several ranges is a
// multi-range request), writes them into the shared buffer and wakes the
// waiting thread.
import { hvFetchRange, hvFetchMultiRange } from './multipart.mjs';

const isNode = typeof process !== 'undefined' && !!process.versions?.node;
// Per file: does the server answer multi-range requests? (true/false/unknown)
const multipartOk = new Map();

async function handle({ url, offs, lens, groups, sab, metaOff, dataOff, headers }) {
  const ctrl = new Int32Array(sab, 0, 4);
  const meta = new Float64Array(sab, metaOff, 4 * offs.length);
  const data = new Uint8Array(sab);
  // cache: 'no-store' avoids Chromium's per-URL HTTP cache lock, which
  // would serialise these requests.
  const init = { cache: 'no-store', headers: headers || {} };
  try {
    let p = dataOff;
    const starts = lens.map((l) => { const s = p; p += l; return s; });
    groups = (groups || offs.map((_, i) => [i])).slice();
    const one = async (idx) => {
      const t0 = performance.timeOrigin + performance.now();
      const o = idx.map((i) => offs[i]), l = idx.map((i) => lens[i]);
      let got;
      if (idx.length > 1 && multipartOk.get(url) === false) {
        got = await Promise.all(o.map((off, k) => hvFetchRange(fetch, url, init, off, l[k])));
      } else {
        const r = await hvFetchMultiRange(fetch, url, init, o, l);
        if (idx.length > 1) {
          if (r.fallback) multipartOk.set(url, false);
          else if (!multipartOk.has(url)) multipartOk.set(url, true);
        }
        got = r.results;
      }
      const t1 = performance.timeOrigin + performance.now();
      idx.forEach((i, k) => {
        const buf = got[k].buf.subarray(0, lens[i]);
        data.set(buf, starts[i]);
        meta[4 * i] = got[k].total;
        meta[4 * i + 1] = buf.length;
        meta[4 * i + 2] = t0;
        meta[4 * i + 3] = t1;
      });
    };
    // Until the server is known to answer multi-range requests, send one of
    // them alone first (a server that ignores them sends the whole file).
    if (!multipartOk.has(url)) {
      const k = groups.findIndex((g) => g.length > 1);
      if (k >= 0) await one(groups.splice(k, 1)[0]);
    }
    await Promise.all(groups.map(one));
    ctrl[2] = multipartOk.get(url) === false ? 1 : 0;
    Atomics.store(ctrl, 0, 1);
  } catch (e) {
    const msg = new TextEncoder().encode(String(e && e.message || e));
    const room = data.length - dataOff;
    data.set(msg.subarray(0, room), dataOff);
    ctrl[1] = Math.min(msg.length, room);
    Atomics.store(ctrl, 0, 2);
  }
  Atomics.notify(ctrl, 0);
}

if (isNode) {
  const { parentPort } = await import('node:worker_threads');
  parentPort.on('message', handle);
  parentPort.postMessage('ready');
} else {
  self.onmessage = (e) => {
    const port = e.data.port;
    port.onmessage = (m) => handle(m.data);
    port.postMessage('ready');
  };
}
