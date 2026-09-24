// Queries timed by the overhead benchmark (on build/test/fts-test.db).
export const BENCH_QUERIES = {
  fts_term_top10: "SELECT rowid, rank FROM docs_fts WHERE docs_fts MATCH 'baathist' ORDER BY rank LIMIT 10",
  fts_prefix_count: "SELECT count(*) FROM docs_fts WHERE docs_fts MATCH 'a*'",
  fts_or_join: "SELECT d.id, d.title FROM docs_fts JOIN docs d ON d.id = docs_fts.rowid WHERE docs_fts MATCH 'baal OR baath' ORDER BY docs_fts.rank, d.id LIMIT 5",
  rowid_lookup: "SELECT body FROM docs WHERE id = 1234",
};

// Time each query: warm-up runs, then `iters` timed runs; median and mean in ms.
export async function timeQueries(runQuery, { iters = 200, warmup = 20 } = {}) {
  const out = {};
  for (const [name, sql] of Object.entries(BENCH_QUERIES)) {
    for (let i = 0; i < warmup; i++) await runQuery(sql);
    const t = [];
    for (let i = 0; i < iters; i++) {
      const t0 = performance.now();
      await runQuery(sql);
      t.push(performance.now() - t0);
    }
    t.sort((a, b) => a - b);
    out[name] = { median_ms: t[t.length >> 1], mean_ms: t.reduce((a, b) => a + b, 0) / t.length };
  }
  return out;
}
