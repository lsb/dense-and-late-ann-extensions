// Shared test helpers: build a test database with the native CLI and serve it
// over HTTP from a separate process (netsim/rangeserver.py when present,
// otherwise wasm/test/rangeserver.mjs). A separate process is required for the
// "sync" variant, which blocks the calling thread while it waits.
import { spawn, execFileSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

export const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
export const CLI = path.join(ROOT, 'build/native/sqlite3');

function xorshift(seed) {
  let x = seed >>> 0 || 1;
  return () => { x ^= x << 13; x >>>= 0; x ^= x >>> 17; x ^= x << 5; x >>>= 0; return x; };
}

/**
 * Create (once) build/test/fts-test.db: docs(id, title, body) with 4000
 * 40-word documents drawn from the project word list, an FTS5 index over it,
 * and a blob table. Page size 4096, rollback journal.
 */
export function makeTestDb(dir = path.join(ROOT, 'build/test')) {
  const db = path.join(dir, 'fts-test.db');
  if (fs.existsSync(db)) return db;
  fs.mkdirSync(dir, { recursive: true });
  const words = fs.readFileSync(path.join(ROOT, 'data/words/american-english'), 'utf8')
    .split('\n').filter((w) => w && !w.includes("'"));
  const rnd = xorshift(12345);
  const lines = ['PRAGMA page_size=4096;', 'PRAGMA journal_mode=delete;', 'BEGIN;',
    'CREATE TABLE docs(id INTEGER PRIMARY KEY, title TEXT, body TEXT);',
    'CREATE TABLE blobs(id INTEGER PRIMARY KEY, data BLOB);'];
  for (let i = 1; i <= 4000; i++) {
    const w = [];
    for (let k = 0; k < 40; k++) w.push(words[rnd() % 5000]);  // 5000-word vocabulary
    lines.push(`INSERT INTO docs VALUES(${i}, 'doc ${i}', '${w.join(' ')}');`);
  }
  lines.push("INSERT INTO blobs SELECT value, randomblob(100 + value * 7) FROM generate_series(1, 200);");
  lines.push('CREATE VIRTUAL TABLE docs_fts USING fts5(body, content=docs, content_rowid=id);');
  lines.push("INSERT INTO docs_fts(docs_fts) VALUES('rebuild');");
  lines.push('COMMIT;', 'VACUUM;');
  const tmp = db + '.tmp';
  fs.rmSync(tmp, { force: true });
  execFileSync(CLI, [tmp], { input: lines.join('\n') });
  fs.renameSync(tmp, db);
  return db;
}

/** Run a query with the native CLI; returns rows as arrays of strings/numbers. */
export function nativeQuery(db, sql) {
  const out = execFileSync(CLI, ['-json', db, sql]).toString().trim();
  if (!out) return [];
  return JSON.parse(out).map((o) => Object.values(o));
}

/**
 * Start a range server for dir in a child process. Returns {url, close}.
 * opts.latencyMs adds a fixed delay per response; opts.isolate sends COOP/COEP.
 */
export async function startServerProcess(dir, { latencyMs = 0, isolate = false, preferNode = false } = {}) {
  const netsim = path.join(ROOT, 'netsim/rangeserver.py');
  let cmd, args;
  if (!preferNode && fs.existsSync(netsim)) {
    cmd = 'python3';
    args = [netsim, '--dir', dir, '--port', '0', '--preset', 'none', '--latency-ms', String(latencyMs)];
    if (isolate) args.push('--isolate');
  } else {
    cmd = process.execPath;
    args = [path.join(ROOT, 'wasm/test/rangeserver.mjs'), '--dir', dir, '--port', '0',
      '--latency-ms', String(latencyMs)];
    if (isolate) args.push('--isolate');
  }
  const child = spawn(cmd, args, { stdio: ['ignore', 'pipe', 'inherit'] });
  const url = await new Promise((resolve, reject) => {
    let buf = '';
    child.stdout.on('data', (d) => {
      buf += d;
      const m = /(http:\/\/[\d.]+:\d+)/.exec(buf);
      if (m) resolve(m[1]);
    });
    child.on('exit', (code) => reject(new Error(`server exited (${code})`)));
    setTimeout(() => reject(new Error('server did not start')), 10000);
  });
  return { url, close: () => { child.kill(); } };
}
