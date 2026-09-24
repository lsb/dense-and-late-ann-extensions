// HTTP/2 (TLS) front for netsim/rangeserver.py, so that a browser can be
// measured with HTTP/2 multiplexing instead of HTTP/1.1's six connections per
// host (rangeserver.py speaks HTTP/1.1 only). Every stream is forwarded as its
// own HTTP/1.1 request to the range server, which applies the network profile
// (use an ',h2' preset there so that its concurrency limit is 100); the proxy
// itself adds only loopback costs. The certificate is self-signed: launch the
// browser with ignoreHTTPSErrors.
//
//   import { startH2Proxy } from './h2proxy.mjs';
//   const p = await startH2Proxy('http://127.0.0.1:8000');   // -> {url, close}
import http from 'node:http';
import http2 from 'node:http2';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { execFileSync } from 'node:child_process';

function selfSigned() {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'h2proxy-'));
  const key = path.join(dir, 'key.pem'), cert = path.join(dir, 'cert.pem');
  execFileSync('openssl', ['req', '-x509', '-newkey', 'rsa:2048', '-nodes', '-keyout', key, '-out', cert,
    '-days', '2', '-subj', '/CN=localhost'], { stdio: 'ignore' });
  const out = { key: fs.readFileSync(key), cert: fs.readFileSync(cert) };
  fs.rmSync(dir, { recursive: true, force: true });
  return out;
}

export async function startH2Proxy(upstream, { port = 0 } = {}) {
  const up = new URL(upstream);
  const agent = new http.Agent({ keepAlive: true, maxSockets: Infinity });
  const server = http2.createSecureServer({ ...selfSigned(), allowHTTP1: false });
  server.on('stream', (stream, headers) => {
    const h = {};
    for (const [k, v] of Object.entries(headers)) if (!k.startsWith(':')) h[k] = v;
    const req = http.request({
      host: up.hostname, port: up.port, method: headers[':method'], path: headers[':path'], headers: h, agent,
    }, (res) => {
      const rh = { ':status': res.statusCode };
      for (const [k, v] of Object.entries(res.headers)) {
        if (!['connection', 'keep-alive', 'transfer-encoding'].includes(k)) rh[k] = v;
      }
      if (stream.destroyed) { res.resume(); return; }
      stream.respond(rh);
      res.pipe(stream);
    });
    req.on('error', () => { if (!stream.destroyed) { try { stream.respond({ ':status': 502 }); } catch {} stream.end(); } });
    stream.on('close', () => req.destroy());
    stream.pipe(req);
  });
  await new Promise((res) => server.listen(port, '127.0.0.1', res));
  return {
    url: `https://localhost:${server.address().port}`,
    close: () => { server.close(); agent.destroy(); },
  };
}
