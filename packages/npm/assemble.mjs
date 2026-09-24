// Copy the runtime files into this directory so that `npm pack` can package
// them: the SQLite WebAssembly builds and JS API from wasm/pkg, the search
// client from web/lib (all but paths.mjs, which this package replaces), and
// the license. Runs automatically before `npm pack` / `npm publish`.
//
//   node assemble.mjs           copy
//   node assemble.mjs --check   copy, then fail if a WebAssembly build is
//                               missing or older than the C sources it is
//                               built from (run `make wasm` first)
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, '../..');
const WASM_PKG = path.join(ROOT, 'wasm/pkg');
const WEB_LIB = path.join(ROOT, 'web/lib');
const BUILDS = ['sqlite-httpvfs', 'sqlite-httpvfs-jspi', 'sqlite-httpvfs-sync'];

function copy(from, to) {
  fs.mkdirSync(path.dirname(to), { recursive: true });
  fs.copyFileSync(from, to);
}

const missing = BUILDS.flatMap((b) => [`${b}.mjs`, `${b}.wasm`])
  .filter((f) => !fs.existsSync(path.join(WASM_PKG, 'dist', f)));
if (missing.length) {
  console.error(`assemble: missing wasm/pkg/dist/${missing.join(', ')}; run \`make wasm\` in ${ROOT}`);
  process.exit(1);
}

for (const f of ['index.mjs', 'multipart.mjs', 'sync-fetch.mjs', 'sync-fetch-worker.mjs']) copy(path.join(WASM_PKG, f), path.join(HERE, f));
fs.rmSync(path.join(HERE, 'dist'), { recursive: true, force: true });
for (const b of BUILDS) for (const ext of ['.mjs', '.wasm']) {
  copy(path.join(WASM_PKG, 'dist', b + ext), path.join(HERE, 'dist', b + ext));
}
for (const f of fs.readdirSync(WEB_LIB)) {
  if (f.endsWith('.mjs') && f !== 'paths.mjs') copy(path.join(WEB_LIB, f), path.join(HERE, 'search', f));
}
copy(path.join(ROOT, 'LICENSE'), path.join(HERE, 'LICENSE'));

if (process.argv.includes('--check')) {
  // A stale build silently lacks newer index layouts (web/NOTES.md), so
  // compare its time with the newest C source that goes into it.
  const srcs = ['ext/dense', 'ext/late', 'ext/fts5rank', 'wasm/src'].flatMap((d) =>
    fs.readdirSync(path.join(ROOT, d)).filter((f) => /\.[ch]$|\.js$/.test(f)).map((f) => path.join(ROOT, d, f)));
  const newest = Math.max(...srcs.map((f) => fs.statSync(f).mtimeMs));
  const stale = BUILDS.filter((b) => fs.statSync(path.join(WASM_PKG, 'dist', b + '.wasm')).mtimeMs < newest);
  if (stale.length) {
    console.error(`assemble: ${stale.join(', ')} older than the C sources; run \`make wasm\` (or set DLA_ALLOW_STALE=1)`);
    if (!process.env.DLA_ALLOW_STALE) process.exit(1);
  }
}
console.log(`assemble: copied wasm/pkg and web/lib into ${path.relative(ROOT, HERE)}`);
