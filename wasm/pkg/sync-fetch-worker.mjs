// Fetch worker for sync-fetch.mjs: fetches a batch of byte ranges in
// parallel, writes them into the shared buffer and wakes the waiting thread.
const isNode = typeof process !== 'undefined' && !!process.versions?.node;

async function handle({ url, offs, lens, sab, metaOff, dataOff, headers }) {
  const ctrl = new Int32Array(sab, 0, 4);
  const meta = new Float64Array(sab, metaOff, 4 * offs.length);
  const data = new Uint8Array(sab);
  try {
    let p = dataOff;
    const starts = lens.map((l) => { const s = p; p += l; return s; });
    await Promise.all(offs.map(async (off, i) => {
      const t0 = performance.timeOrigin + performance.now();
      const resp = await fetch(url, { headers: Object.assign({}, headers || {},
        { Range: `bytes=${off}-${off + lens[i] - 1}` }) });
      let buf = new Uint8Array(await resp.arrayBuffer());
      let total = -1;
      if (resp.status === 206) {
        const m = /\/(\d+)\s*$/.exec(resp.headers.get('content-range') || '');
        if (m) total = Number(m[1]);
      } else if (resp.status === 200) {
        total = buf.length;
        buf = buf.subarray(off, off + lens[i]);
      } else {
        throw new Error(`HTTP ${resp.status} for ${url}`);
      }
      buf = buf.subarray(0, lens[i]);
      data.set(buf, starts[i]);
      meta[4 * i] = total;
      meta[4 * i + 1] = buf.length;
      meta[4 * i + 2] = t0;
      meta[4 * i + 3] = performance.timeOrigin + performance.now();
    }));
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
  self.onmessage = (e) => handle(e.data);
  self.postMessage('ready');
}
