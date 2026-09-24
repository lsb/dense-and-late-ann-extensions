// Query encoders on onnxruntime-web: all-MiniLM-L6-v2 (dense, 384-d) and
// LateOn-Code-edge (late interaction, 48-d per token). Environment-neutral:
// the caller passes the `ort` namespace and the Tokenizer class, so the same
// code runs in the encoder worker and in Node tests.
//
// Semantics follow enc/minilm.py and enc/lateon.py for a batch of one query:
//   MiniLM: tokens -> last_hidden_state -> mean over the (all-ones) mask -> L2 normalise
//   LateOn: tokens ([CLS] [Q] … [SEP]) -> output [n, 48], already unit-norm per token

import { makeTokenizers } from './tokenize.mjs';

export const MODELS = {
  minilm: { onnx: 'models/minilm-l6-v2/model_w8.onnx', tokenizer: 'models/minilm-l6-v2/tokenizer.json', dim: 384 },
  lateon: { onnx: 'models/lateon-code-edge/model_w8.onnx', tokenizer: 'models/lateon-code-edge/tokenizer.json', dim: 48 },
};

function int64Tensor(ort, ids) {
  return new ort.Tensor('int64', BigInt64Array.from(ids, (x) => BigInt(x)), [1, ids.length]);
}

export class Encoder {
  /**
   * @param ort       onnxruntime-web namespace
   * @param Tokenizer @huggingface/tokenizers Tokenizer class
   * @param which     'minilm' | 'lateon'
   * @param modelBytes Uint8Array of the .onnx file
   * @param tokenizerJson parsed tokenizer.json
   */
  static async create(ort, Tokenizer, which, modelBytes, tokenizerJson, sessionOptions = {}) {
    const e = new Encoder();
    e.which = which;
    e.ort = ort;
    const t0 = performance.now();
    e.tok = makeTokenizers(Tokenizer, which === 'minilm' ? { minilmJson: tokenizerJson } : { lateonJson: tokenizerJson });
    e.tokenizerMs = performance.now() - t0;
    const t1 = performance.now();
    e.session = await ort.InferenceSession.create(modelBytes, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
      ...sessionOptions,
    });
    e.sessionMs = performance.now() - t1;
    return e;
  }

  /** Encode one query. Returns {ids, vectors: Float32Array, n, dim, tokenizeMs, inferMs}. */
  async encode(text) {
    const { ort } = this;
    const t0 = performance.now();
    const ids = this.which === 'minilm' ? this.tok.minilm(text) : this.tok.lateonQuery(text);
    const t1 = performance.now();
    const feeds = {
      input_ids: int64Tensor(ort, ids),
      attention_mask: int64Tensor(ort, ids.map(() => 1)),
    };
    if (this.session.inputNames.includes('token_type_ids')) feeds.token_type_ids = int64Tensor(ort, ids.map(() => 0));
    const out = await this.session.run(feeds);
    const y = out[this.session.outputNames[0]];
    const data = y.data; // Float32Array [1, n, dim]
    const n = ids.length, dim = y.dims[2];
    let vectors;
    if (this.which === 'minilm') {
      vectors = new Float32Array(dim);
      for (let i = 0; i < n; i++) for (let d = 0; d < dim; d++) vectors[d] += data[i * dim + d];
      let norm = 0;
      for (let d = 0; d < dim; d++) { vectors[d] /= n; norm += vectors[d] * vectors[d]; }
      norm = Math.max(Math.sqrt(norm), 1e-12);
      for (let d = 0; d < dim; d++) vectors[d] /= norm;
    } else {
      vectors = Float32Array.from(data.subarray(0, n * dim));
    }
    for (const t of Object.values(out)) t.dispose?.();
    const t2 = performance.now();
    return { ids, vectors, n: this.which === 'minilm' ? 1 : n, dim, tokenizeMs: t1 - t0, inferMs: t2 - t1 };
  }
}
