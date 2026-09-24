// Encoder worker: loads one query encoder (MiniLM or LateOn-Code-edge) with
// onnxruntime-web and encodes queries. One worker per model, so the two
// models load and run in parallel.
//
// Methods:
//   load({which, modelUrl, tokenizerUrl, ortBase, numThreads, cache}) -> load stats
//   encode({text}) -> {ids, vectors (Float32Array, transferred), n, dim, tokenizeMs, inferMs, ms}

import { serve } from './rpc.mjs';
import { Encoder } from './encoder-core.mjs';
import { fetchBytes } from './fetch-cache.mjs';

let encoder = null;
let loading = null;

async function load({ which, modelUrl, tokenizerUrl, ortBase, ortWasm, numThreads = 1, cache = true }, notify) {
  const t0 = performance.now();
  const ort = await import(new URL('ort.wasm.min.mjs', ortBase).href);
  ort.env.wasm.wasmPaths = ortBase;
  ort.env.wasm.numThreads = self.crossOriginIsolated ? numThreads : 1;
  ort.env.wasm.proxy = false;
  const { Tokenizer } = await import('../vendor/tokenizers.min.mjs');
  const tLib = performance.now() - t0;
  // The ONNX Runtime WebAssembly binary (14 MB) is fetched once by the page
  // (through the same cache) and handed to both encoder workers.
  const [model, tok] = await Promise.all([
    fetchBytes(modelUrl, { cache, onProgress: (p) => notify({ kind: 'progress', which, ...p }) }),
    fetchBytes(tokenizerUrl, { cache }),
  ]);
  if (ortWasm) ort.env.wasm.wasmBinary = ortWasm;
  const tokenizerJson = JSON.parse(new TextDecoder().decode(tok.bytes));
  encoder = await Encoder.create(ort, Tokenizer, which, model.bytes, tokenizerJson);
  // One warm-up run: the first inference allocates buffers and is slower.
  const tw = performance.now();
  await encoder.encode('warm up');
  const warmupMs = performance.now() - tw;
  return {
    which,
    modelBytes: model.bytes.length, modelFromCache: model.fromCache, modelMs: model.ms,
    tokenizerBytes: tok.bytes.length, tokenizerFromCache: tok.fromCache, tokenizerMs: tok.ms,
    libMs: tLib, tokenizerInitMs: encoder.tokenizerMs, sessionMs: encoder.sessionMs, warmupMs,
    totalMs: performance.now() - t0,
    numThreads: ort.env.wasm.numThreads, crossOriginIsolated: self.crossOriginIsolated,
  };
}

serve(self, {
  async load(args, notify) {
    if (!loading) loading = load(args, notify).catch((e) => { loading = null; throw e; });
    return loading;
  },
  async encode({ text }) {
    if (!encoder) {
      if (!loading) throw new Error('encoder not loaded');
      await loading;
    }
    const t0 = performance.now();
    const r = await encoder.encode(text);
    r.ms = performance.now() - t0;
    return { __transfer: [r.vectors.buffer], value: r };
  },
}, { serial: true });
