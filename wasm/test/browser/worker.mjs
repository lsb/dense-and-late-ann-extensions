// Browser-side test body, run in a dedicated Worker: opens the test database
// over HTTP with the requested build variant, runs the FTS5 queries and the
// batched-lookup scenario, and reports rows, stats and logs to the page.
import { open } from '../../pkg/index.mjs';

export const FTS_QUERIES = [
  "SELECT rowid, rank FROM docs_fts WHERE docs_fts MATCH 'baathist' ORDER BY rank LIMIT 10",
  "SELECT rowid FROM docs_fts WHERE docs_fts MATCH 'bab* OR bac*' ORDER BY rowid LIMIT 20",
  "SELECT count(*) FROM docs_fts WHERE docs_fts MATCH 'a*'",
];
const IDS = [17, 311, 555, 901, 1234, 1500, 1777, 2020, 2345, 2600, 2900, 3100, 3333, 3600, 3800, 3999];

self.onmessage = async (e) => {
  const { variant, dbUrl, isolated } = e.data;
  const report = { variant, isolated, userAgent: navigator.userAgent };
  try {
    const db = await open(dbUrl, { variant });
    report.resolvedVariant = db.variant;
    report.fts = [];
    for (const q of FTS_QUERIES) report.fts.push({ q, rows: (await db.queryRaw(q)).rows });
    report.hello = (await db.query("SELECT hello('browser') AS h"))[0].h;
    report.stats = db.stats();
    report.log = db.log();

    await db.resetStats({ clearCache: true });
    let t0 = performance.now();
    const seq = [];
    for (const id of IDS) seq.push((await db.queryRaw('SELECT id, body FROM docs WHERE id = ?', [id])).rows[0]);
    report.sequential = { ms: performance.now() - t0, rounds: db.stats().rounds, rows: seq };

    await db.resetStats({ clearCache: true });
    t0 = performance.now();
    await db.query('SELECT httpvfs_warm(?, ?)', ['SELECT body FROM docs WHERE id = ?', IDS.join(',')]);
    const bat = [];
    for (const id of IDS) bat.push((await db.queryRaw('SELECT id, body FROM docs WHERE id = ?', [id])).rows[0]);
    report.batched = { ms: performance.now() - t0, rounds: db.stats().rounds, rows: bat, log: db.log() };
    await db.close();
  } catch (err) {
    report.error = String(err && err.stack || err);
  }
  self.postMessage(report);
};
