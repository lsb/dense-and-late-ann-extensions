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

const HERE = new URL('.', import.meta.url);

/** Default query parameters per index kind (see the extensions' NOTES.md). */
export const DEFAULT_PARAMS = {
  fts: { mode: 'or' },
  'dense:graph': { ef: 64, beam: 16, rerank: 2 },
  'dense:ivf': { nprobe: 16, rerank_k: 64, rerank: 2 },
  late: { nprobe: 8, layout: 'warp', stoplist: 0.02, rerank: 64 },
};

export const MODEL_FILES = {
  minilm: { model: 'models/minilm-l6-v2/model_qint8_arm64.onnx', tokenizer: 'models/minilm-l6-v2/tokenizer.json' },
  lateon: { model: 'models/lateon-code-edge/model_int8.onnx', tokenizer: 'models/lateon-code-edge/tokenizer.json' },
};

function paramsKey(ix) { return ix.kind === 'dense' ? `dense:${ix.layout}` : ix.kind; }

function label(ix) {
  if (ix.kind === 'fts') return 'FTS5 (bm25)';
  if (ix.kind === 'dense') return ix.layout === 'ivf' ? 'Dense IVF-PQ (MiniLM)' : 'Dense graph (MiniLM)';
  return 'Late interaction (LateOn-Code-edge)';
}

export class SearchIndex {
  constructor(opts) {
    this.opts = opts;
    this.encoders = {};   // which -> {call, worker, loaded: Promise<loadStats>}
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
      this.ortWasm ||= fetchBytes(new URL('../vendor/ort/ort-wasm-simd-threaded.wasm', HERE).href,
        { cache: this.opts.modelCache !== false });
      e.loaded = this.ortWasm.then(async (wasm) => ({ ...await call('load', {
        ortWasm: wasm.bytes,
        which,
        modelUrl: new URL(this.opts.models?.[which]?.model || MODEL_FILES[which].model, base).href,
        tokenizerUrl: new URL(this.opts.models?.[which]?.tokenizer || MODEL_FILES[which].tokenizer, base).href,
        ortBase: new URL('../vendor/ort/', HERE).href,
        numThreads: this.opts.encoderThreads || 1,
        cache: this.opts.modelCache !== false,
      }, (p) => { for (const l of e.listeners) l(p); }),
      ortWasmBytes: wasm.bytes.length, ortWasmMs: wasm.ms, ortWasmFromCache: wasm.fromCache }));
      e.loaded.catch(() => { delete this.encoders[which]; worker.terminate(); });
    }
    if (onProgress) e.listeners.add(onProgress);
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
 *   pageCacheBytes, blockSize, readaheadBytes, maxParallel   passed to wasm/pkg open()
 *   sqliteModule     URL of wasm/pkg/index.mjs (default: ../../wasm/pkg/index.mjs)
 *   modelBase        base URL for models/ (default: the repository root, ../../)
 *   models           {minilm: {model, tokenizer}, lateon: {…}} relative to modelBase
 *   preload          ['minilm', 'lateon']: start loading encoders now
 *   modelCache       keep model files in Cache Storage (default true)
 *   encoderThreads   ONNX Runtime threads (needs cross-origin isolation; default 1)
 */
export async function openIndex(url, opts = {}) {
  const ix = new SearchIndex({ modelBase: new URL('../../', HERE).href, ...opts });
  for (const w of opts.preload || []) ix.loadEncoder(w).catch(() => {});
  ix.sqliteWorker = new Worker(new URL('sqlite-worker.mjs', HERE), { type: 'module', name: 'sqlite' });
  ix.sql = client(ix.sqliteWorker);
  const { variant = 'auto', pageCacheBytes = 16 << 20, blockSize, readaheadBytes, maxParallel } = opts;
  const info = await ix.sql('open', {
    url: new URL(url, globalThis.location?.href).href,
    sqliteModule: opts.sqliteModule || new URL('../../wasm/pkg/index.mjs', HERE).href,
    variant, pageCacheBytes, blockSize, readaheadBytes, maxParallel,
  });
  ix.url = url;
  ix.indexes = info.indexes;
  ix.docs = info.docs;
  ix.variant = info.variant;
  ix.openStats = { ...info.stats, openMs: info.openMs };
  ix.systems = info.indexes.map((i) => ({
    id: i.table, kind: i.kind, layout: i.layout, table: i.table, label: label(i),
    encoder: i.kind === 'dense' ? 'minilm' : i.kind === 'late' ? 'lateon' : null,
    defaults: DEFAULT_PARAMS[paramsKey(i)],
  }));
  return ix;
}
