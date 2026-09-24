// Browser client for a search database built by tools/build_db.py: FTS5,
// dense (MiniLM + dense_ann) and late-interaction (LateOn-Code-edge +
// late_plaid) search over one SQLite file fetched with HTTP range requests.
//
//   import { openIndex } from './lib/index.mjs';
//   const ix = await openIndex('/build/matrix/words-10k.db');
//   ix.systems                     // [{id:'fts', …}, {id:'dense', …}, {id:'late', …}, …]
//   const r = await ix.search('pinwheel gossiping', { system: 'late', k: 10, nprobe: 8 });
//   r.rows                         // [{id, score, text}]
//   r.stats                        // {encodeMs, searchMs, docsMs, wallMs, rounds, requests, bytes, …}
//
// SQLite runs in one module Worker (web/lib/sqlite-worker.mjs); each encoder
// runs in its own module Worker (web/lib/encoder-worker.mjs), loaded the
// first time a query needs it (or up front with opts.preload).

import { client } from './rpc.mjs';
import { fetchBytes } from './fetch-cache.mjs';
import { DEFAULT_PARAMS, paramsKey } from './search-core.mjs';
import * as PATHS from './paths.mjs';

export { DEFAULT_PARAMS };

const HERE = new URL('.', import.meta.url);

/** Absolute URL of an option (relative ones resolve against the page). */
function absUrl(v, name) {
  if (!v) throw new Error(`openIndex: no default for the ${name} option here; pass it explicitly`);
  return new URL(v, globalThis.location?.href).href;
}

export const MODEL_FILES = {
  minilm: { model: 'models/minilm-l6-v2/model_w8.onnx', tokenizer: 'models/minilm-l6-v2/tokenizer.json' },
  lateon: { model: 'models/lateon-code-edge/model_w8.onnx', tokenizer: 'models/lateon-code-edge/tokenizer.json' },
};

function label(ix) {
  if (ix.kind === 'fts') return 'FTS5 (bm25)';
  if (ix.kind === 'dense') return ix.layout === 'ivf' ? 'Dense IVF-PQ (MiniLM)' : 'Dense graph (MiniLM)';
  return 'Late interaction (LateOn-Code-edge)';
}

export class SearchIndex {
  constructor(opts) {
    this.opts = opts;
    this.encoders = {};   // which -> {call, worker, loaded: Promise<loadStats>}
    this.warming = {};    // table -> Promise<warm stats>
  }

  /**
   * Load the per-connection static data of indexes now (PQ codebooks, entry
   * sets, IVF and late-interaction centroids, FTS5 structure), in the
   * background: a search issued meanwhile is served first. systems: a list
   * of 'fts' | 'dense' | 'late' | table names (default: every index); each
   * index is warmed once. Resolves to [{table, ms, rounds, requests, bytes}].
   */
  warm(systems) {
    const tables = new Set();
    // system names resolve as in search(): 'dense' is the graph index if there is one
    for (const s of systems || this.indexes.map((i) => i.table)) tables.add(this.resolve(s).table);
    return Promise.all([...tables].map((t) => {
      this.warming[t] ||= this.sql('warm', { table: t }).catch((e) => ({ table: t, error: String(e.message || e) }));
      return this.warming[t];
    }));
  }

  /**
   * opts.warm = 'auto': when an encoder starts loading, warm the index that
   * search({system: 'dense' | 'late'}) would use. Only that one: a warm-up
   * that is running cannot be interrupted, so warming an index the page never
   * queries (say the graph when it uses IVF) would delay its first search.
   */
  _autoWarm(which) {
    if ((this.opts.warm ?? 'auto') !== 'auto' || !this.indexes) return;
    const kind = which === 'minilm' ? 'dense' : 'late';
    if (this.indexes.some((i) => i.kind === kind)) this.warm([this.resolve(kind).table]);
  }

  /**
   * Start loading an encoder ('minilm' | 'lateon'); resolves to load stats
   * {modelBytes, modelMs, modelFromCache, sessionMs, warmupMs, totalMs, …}.
   */
  loadEncoder(which, onProgress) {
    let e = this.encoders[which];
    if (!e) {
      const worker = new Worker(new URL('encoder-worker.mjs', HERE), { type: 'module', name: `encoder-${which}` });
      const call = client(worker);
      const base = this.opts.modelBase;
      e = this.encoders[which] = { worker, call, listeners: new Set() };
      // Both encoder workers use the same 14 MB ONNX Runtime binary: fetch it once.
      const ortBase = absUrl(this.opts.ortBase || PATHS.ortBase, 'ortBase');
      this.ortWasm ||= fetchBytes(new URL('ort-wasm-simd-threaded.wasm', ortBase).href,
        { cache: this.opts.modelCache !== false });
      e.loaded = this.ortWasm.then(async (wasm) => ({ ...await call('load', {
        ortWasm: wasm.bytes,
        which,
        modelUrl: new URL(this.opts.models?.[which]?.model || MODEL_FILES[which].model, base).href,
        tokenizerUrl: new URL(this.opts.models?.[which]?.tokenizer || MODEL_FILES[which].tokenizer, base).href,
        ortBase,
        tokenizersModule: absUrl(this.opts.tokenizersModule || PATHS.tokenizersModule, 'tokenizersModule'),
        numThreads: this.opts.encoderThreads || 1,
        cache: this.opts.modelCache !== false,
      }, (p) => { for (const l of e.listeners) l(p); }),
      ortWasmBytes: wasm.bytes.length, ortWasmMs: wasm.ms, ortWasmFromCache: wasm.fromCache }));
      e.loaded.catch(() => { delete this.encoders[which]; worker.terminate(); });
    }
    if (onProgress) e.listeners.add(onProgress);
    this._autoWarm(which);
    return e.loaded;
  }

  /** Encode a query: {vectors: Float32Array, n, dim, ids, tokenizeMs, inferMs, ms}. */
  async encode(text, which) {
    await this.loadEncoder(which);
    return this.encoders[which].call('encode', { text });
  }

  /** Pick the index for a system name: 'fts' | 'dense' | 'late' | a table name. */
  resolve(system) {
    const ix = this.indexes.find((i) => i.table === system)
      || (system === 'dense' && (this.indexes.find((i) => i.kind === 'dense' && i.layout === 'graph')
                                 || this.indexes.find((i) => i.kind === 'dense')))
      || this.indexes.find((i) => i.kind === system);
    if (!ix) throw new Error(`this database has no ${system} index (have: ${this.indexes.map((i) => i.table).join(', ')})`);
    return ix;
  }

  /**
   * Search. options: {system, k, cold, fetchDocs, ...params}; params are the
   * extension's query parameters (dense graph: ef, beam, rerank; dense IVF:
   * nprobe, rerank_k, rerank; late: nprobe, layout, stoplist, rerank, approx,
   * ndocs, …; fts: mode 'or'|'and'). Missing params take DEFAULT_PARAMS.
   */
  async search(text, { system = 'fts', k = 10, cold = false, fetchDocs = true, ...params } = {}) {
    const t0 = performance.now();
    const ix = this.resolve(system);
    const p = { ...DEFAULT_PARAMS[paramsKey(ix)], ...params };
    let enc = null;
    if (ix.kind !== 'fts') {
      const which = ix.kind === 'dense' ? 'minilm' : 'lateon';
      const tl = performance.now();
      await this.loadEncoder(which);
      const loadWaitMs = performance.now() - tl;
      enc = await this.encode(text, which);
      enc.loadWaitMs = loadWaitMs;
    }
    const r = await this.sql('search', {
      table: ix.table, text, vector: enc?.vectors, k, params: p, cold, fetchDocs,
    });
    const wallMs = performance.now() - t0;
    r.system = system;
    r.label = label(ix);
    r.params = p;
    r.query = { text, tokens: enc?.ids?.length, vectors: enc?.n, ids: enc?.ids };
    r.embedding = enc?.vectors;   // Float32Array: [dim] (dense) or [n × dim] (late)
    r.stats = {
      ...r.stats,
      encodeMs: enc ? enc.ms : 0,
      tokenizeMs: enc ? enc.tokenizeMs : 0,
      inferMs: enc ? enc.inferMs : 0,
      encoderLoadWaitMs: enc ? enc.loadWaitMs : 0,
      wallMs,
    };
    return r;
  }

  /** VFS counters of the SQLite connection (cumulative). */
  vfsStats() { return this.sql('stats'); }
  /** Per-request log [{offset, length, round, tStart, tEnd}] since the last reset. */
  vfsLog() { return this.sql('log'); }
  resetStats(opts) { return this.sql('resetStats', opts); }
  /** Raw SQL, for experiments: {columns, rows}. */
  query(sql, params) { return this.sql('query', { sql, params }); }

  async close() {
    try { await this.sql('close'); } catch { /* ignore */ }
    this.sqliteWorker.terminate();
    for (const e of Object.values(this.encoders)) e.worker.terminate();
    this.encoders = {};
  }
}

/**
 * Open a search database.
 * opts:
 *   variant          SQLite WASM build: 'auto' (JSPI where supported, else Asyncify) | 'asyncify' | 'jspi'
 *   pageCacheBytes, blockSize, readaheadBytes, maxParallel   passed to wasm/pkg open();
 *                    pageCacheBytes defaults to 'auto' (1/64 of the database, 16..64 MiB)
 *   maxRequests, multipart, rttMs, bandwidthKbps, netAutoEstimate   request planning under a
 *                    per-round request budget (wasm/pkg open(); maxRequests 'auto' by default:
 *                    6 over HTTP/1.x, 100 over HTTP/2)
 *   sqliteModule     URL of the SQLite module (default: see paths.mjs)
 *   ortBase          directory URL of onnxruntime-web's dist files (default: see paths.mjs)
 *   tokenizersModule URL of @huggingface/tokenizers' ES module (default: see paths.mjs)
 *   modelBase        base URL for the MODEL_FILES paths (default: see paths.mjs)
 *   models           {minilm: {model, tokenizer}, lateon: {…}} relative to modelBase
 *   preload          ['minilm', 'lateon']: start loading encoders now
 *   modelCache       keep model files in Cache Storage (default true)
 *   encoderThreads   ONNX Runtime threads (needs cross-origin isolation; default 1)
 *   warm             load indexes' static data in the background, while the
 *                    encoders load: 'auto' (default: the indexes of each
 *                    encoder as soon as it starts loading, e.g. through
 *                    preload), true (every index at once, FTS5 included),
 *                    false, or a list of systems / tables
 */
export async function openIndex(url, opts = {}) {
  const ix = new SearchIndex({ ...opts, modelBase: absUrl(opts.modelBase || PATHS.modelBase || './', 'modelBase') });
  for (const w of opts.preload || []) ix.loadEncoder(w).catch(() => {});
  ix.sqliteWorker = new Worker(new URL('sqlite-worker.mjs', HERE), { type: 'module', name: 'sqlite' });
  ix.sql = client(ix.sqliteWorker);
  const { variant = 'auto', pageCacheBytes = 'auto', pageCacheMinBytes = 16 << 20, blockSize, readaheadBytes, maxParallel,
    maxRequests, multipart, rttMs, bandwidthKbps, netAutoEstimate } = opts;
  const info = await ix.sql('open', {
    url: new URL(url, globalThis.location?.href).href,
    sqliteModule: absUrl(opts.sqliteModule || PATHS.sqliteModule, 'sqliteModule'),
    variant, pageCacheBytes, pageCacheMinBytes, blockSize, readaheadBytes, maxParallel,
    maxRequests, multipart, rttMs, bandwidthKbps, netAutoEstimate,
  });
  ix.url = url;
  ix.indexes = info.indexes;
  ix.docs = info.docs;
  ix.variant = info.variant;
  ix.openStats = { ...info.stats, openMs: info.openMs };
  ix.net = info.net;
  ix.systems = info.indexes.map((i) => ({
    id: i.table, kind: i.kind, layout: i.layout, table: i.table, label: label(i),
    encoder: i.kind === 'dense' ? 'minilm' : i.kind === 'late' ? 'lateon' : null,
    defaults: DEFAULT_PARAMS[paramsKey(i)],
  }));
  const w = opts.warm ?? 'auto';
  if (w === true) ix.warm();
  else if (Array.isArray(w)) ix.warm(w);
  else if (w === 'auto') for (const which of Object.keys(ix.encoders)) ix._autoWarm(which);
  return ix;
}
