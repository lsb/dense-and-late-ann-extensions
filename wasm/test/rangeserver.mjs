// Minimal static HTTP server with single-range support, CORS and optional
// COOP/COEP headers and fixed per-response latency. Used by the tests when
// netsim/rangeserver.py is not wanted; the project's full network simulator is
// netsim/rangeserver.py.
//
//   node wasm/test/rangeserver.mjs --dir build --port 0 [--latency-ms 50] [--isolate]
//   import { startServer } from './rangeserver.mjs'
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';

const TYPES = { '.html': 'text/html', '.mjs': 'text/javascript', '.js': 'text/javascript',
  '.wasm': 'application/wasm', '.json': 'application/json' };

export function startServer({ dir, port = 0, latencyMs = 0, isolate = false, host = '127.0.0.1' }) {
  const root = path.resolve(dir);
  const log = [];
  const server = http.createServer((req, res) => {
    const url = new URL(req.url, 'http://x');
    const file = path.join(root, decodeURIComponent(url.pathname));
    const headers = {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Headers': 'Range',
      'Access-Control-Expose-Headers': 'Content-Range, Content-Length, Accept-Ranges',
      'Accept-Ranges': 'bytes',
      'Cache-Control': 'no-store',
    };
    if (isolate) {
      headers['Cross-Origin-Opener-Policy'] = 'same-origin';
      headers['Cross-Origin-Embedder-Policy'] = 'require-corp';
      headers['Cross-Origin-Resource-Policy'] = 'cross-origin';
    }
    if (req.method === 'OPTIONS') { res.writeHead(204, headers); res.end(); return; }
    if (!file.startsWith(root) || !fs.existsSync(file) || !fs.statSync(file).isFile()) {
      res.writeHead(404, headers); res.end(); return;
    }
    const size = fs.statSync(file).size;
    headers['Content-Type'] = TYPES[path.extname(file)] || 'application/octet-stream';
    let status = 200, start = 0, end = size - 1;
    const range = req.headers.range;
    if (range) {
      const m = /^bytes=(\d*)-(\d*)$/.exec(range);
      if (!m) { res.writeHead(416, headers); res.end(); return; }
      if (m[1] === '') { start = Math.max(0, size - Number(m[2])); }
      else { start = Number(m[1]); if (m[2] !== '') end = Math.min(Number(m[2]), size - 1); }
      if (start >= size || start > end) {
        headers['Content-Range'] = `bytes */${size}`;
        res.writeHead(416, headers); res.end(); return;
      }
      status = 206;
      headers['Content-Range'] = `bytes ${start}-${end}/${size}`;
    }
    headers['Content-Length'] = String(end - start + 1);
    log.push({ path: url.pathname, start, end, t: performance.now() });
    const send = () => {
      res.writeHead(status, headers);
      if (req.method === 'HEAD') { res.end(); return; }
      fs.createReadStream(file, { start, end }).pipe(res);
    };
    if (latencyMs > 0) setTimeout(send, latencyMs); else send();
  });
  server.keepAliveTimeout = 60000;
  return new Promise((resolve) => {
    server.listen(port, host, () => {
      const { port: p } = server.address();
      resolve({ url: `http://${host}:${p}`, port: p, log, server,
                setLatency(ms) { latencyMs = ms; },
                close: () => new Promise((r) => { server.closeAllConnections?.(); server.close(r); }) });
    });
  });
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const args = process.argv.slice(2);
  const get = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : d; };
  const s = await startServer({ dir: get('--dir', '.'), port: Number(get('--port', 8000)),
    latencyMs: Number(get('--latency-ms', 0)), isolate: args.includes('--isolate') });
  console.log(`serving at ${s.url}`);
}
