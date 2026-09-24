// Blocking, parallel range fetches for the "sync" build (no Asyncify/JSPI).
//
// The calling thread posts a batch of ranges to a fetch worker and waits with
// Atomics.wait on a SharedArrayBuffer; the worker fetches all ranges
// concurrently with fetch() and wakes it. Works in Node (any thread) and in
// browsers inside a Worker of a cross-origin-isolated page (COOP/COEP), since
// SharedArrayBuffer needs isolation and the main thread may not block.

const isNode = typeof process !== 'undefined' && !!process.versions?.node;

export async function createSyncFetcher({ fetchHeaders } = {}) {
  const url = new URL('./sync-fetch-worker.mjs', import.meta.url);
  let worker, post;
  const ready = new Promise((resolve, reject) => {
    if (isNode) {
      import('node:worker_threads').then(({ Worker }) => {
        worker = new Worker(url);
        worker.once('message', resolve);
        worker.once('error', reject);
        post = (m) => worker.postMessage(m);
      }, reject);
    } else {
      // Batches go over a MessageChannel: in Chromium, messages posted to a
      // nested worker itself are not delivered while this thread is blocked
      // in Atomics.wait, but MessagePort messages are.
      worker = new Worker(url, { type: 'module' });
      const ch = new MessageChannel();
      ch.port1.onmessage = () => resolve();
      worker.onerror = reject;
      worker.postMessage({ port: ch.port2 }, [ch.port2]);
      post = (m) => ch.port1.postMessage(m);
    }
  });
  await ready;   // the worker must be running before we ever block
  if (isNode) worker.unref();

  function fetchSync(fileUrl, offs, lens) {
    const n = offs.length;
    const total = lens.reduce((a, b) => a + b, 0);
    const metaOff = 16, dataOff = 16 + 32 * n;
    const sab = new SharedArrayBuffer(dataOff + total);
    const ctrl = new Int32Array(sab, 0, 4);
    post({ url: String(fileUrl), offs, lens, sab, metaOff, dataOff, headers: fetchHeaders });
    Atomics.wait(ctrl, 0, 0);
    if (Atomics.load(ctrl, 0) !== 1) {
      throw new Error('httpvfs sync fetch failed: ' + new TextDecoder().decode(
        new Uint8Array(sab, dataOff, Math.min(total, ctrl[1])).slice()));
    }
    const meta = new Float64Array(sab, metaOff, 4 * n);
    const out = new Array(n);
    let p = dataOff;
    for (let i = 0; i < n; i++) {
      const got = meta[4 * i + 1];
      out[i] = { buf: new Uint8Array(sab, p, got).slice(), total: meta[4 * i],
                 t0: meta[4 * i + 2] - performance.timeOrigin,
                 t1: meta[4 * i + 3] - performance.timeOrigin };
      p += lens[i];
    }
    return out;
  }
  fetchSync.terminate = () => worker.terminate();
  return fetchSync;
}
