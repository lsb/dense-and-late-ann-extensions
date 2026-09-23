// Load generator for measuring rangeserver overhead (Node.js, no dependencies).
// usage: node netsim/loadgen.js PORT CONNS SECONDS RANGE_BYTES PATH [FILE_SIZE]
const http = require('http');
const [,, port, conns, secs, len, path] = process.argv;
const agent = new http.Agent({keepAlive: true, maxSockets: +conns});
let done = 0, bytes = 0, lat = []; const end = Date.now() + secs*1000; const size = +(process.argv[7] || 5000000);
function one() {
  if (Date.now() > end) return Promise.resolve();
  const off = Math.floor(Math.random() * (size - len));
  const t = process.hrtime.bigint();
  return new Promise((res, rej) => {
    http.get({port: +port, path, agent, headers: {Range: `bytes=${off}-${off + +len - 1}`}}, r => {
      r.on('data', d => bytes += d.length); r.on('end', () => { done++; lat.push(Number(process.hrtime.bigint() - t)/1e6); res(); });
    }).on('error', rej);
  }).then(one);
}
const t0 = Date.now();
Promise.all(Array.from({length: +conns}, one)).then(() => {
  const s = (Date.now() - t0)/1000; lat.sort((a,b)=>a-b);
  console.log(JSON.stringify({conns:+conns, len:+len, rps: Math.round(done/s), MBps: +(bytes/s/1e6).toFixed(1), p50_ms: +lat[lat.length>>1].toFixed(3), p99_ms: +lat[Math.floor(lat.length*0.99)].toFixed(3)}));
  agent.destroy();
});
