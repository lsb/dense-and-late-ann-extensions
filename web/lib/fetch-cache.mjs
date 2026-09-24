// fetch() into a Uint8Array with progress, optionally through Cache Storage so
// that a reload does not download large files (models, the ONNX Runtime
// binary) again. The netsim server sends Cache-Control: no-store, so the HTTP
// cache never keeps them. Works on the main thread and in workers.

export const CACHE_NAME = 'dense-late-demo-models-v1';
/** Returns {bytes, fromCache, ms, transferBytes}. */
export async function fetchBytes(url, { cache = true, onProgress } = {}) {
  const t0 = performance.now();
  let store = null;
  if (cache && typeof caches !== 'undefined') {
    try { store = await caches.open(CACHE_NAME); } catch { store = null; }
  }
  if (store) {
    const hit = await store.match(url);
    if (hit) {
      const buf = new Uint8Array(await hit.arrayBuffer());
      return { bytes: buf, fromCache: true, ms: performance.now() - t0, transferBytes: 0 };
    }
  }
  const resp = await fetch(url, { cache: 'no-store' });
  if (!resp.ok) throw new Error(`HTTP ${resp.status} for ${url}`);
  const total = Number(resp.headers.get('content-length')) || 0;
  const reader = resp.body.getReader();
  const chunks = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    onProgress?.({ url, loaded: got, total });
  }
  const buf = new Uint8Array(got);
  let off = 0;
  for (const c of chunks) { buf.set(c, off); off += c.length; }
  if (store) {
    try {
      await store.put(url, new Response(buf, { headers: { 'content-type': 'application/octet-stream' } }));
    } catch { /* quota or opaque: ignore */ }
  }
  return { bytes: buf, fromCache: false, ms: performance.now() - t0, transferBytes: got };
}
