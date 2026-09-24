import { open } from '../pkg/index.mjs';
import { timeQueries } from './queries.mjs';
self.onmessage = async (e) => {
  try {
    const db = await open(e.data.dbUrl, { variant: e.data.variant, pageCacheBytes: 16 << 20 });
    const timings = await timeQueries((sql) => db.queryRaw(sql));
    await db.close();
    self.postMessage({ timings });
  } catch (err) {
    self.postMessage({ error: String(err && err.stack || err) });
  }
};
