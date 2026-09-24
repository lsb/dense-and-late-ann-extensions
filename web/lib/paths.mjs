// Default locations of the files the client loads at run time. In this
// repository they are the WASM build in wasm/pkg, the vendored ONNX Runtime
// and tokenizers in web/vendor, and the models in models/. The npm package
// ships its own version of this file (packages/npm/search/paths.mjs), which
// points at the package's own SQLite build and at the onnxruntime-web and
// @huggingface/tokenizers packages installed next to it. Every entry can be
// overridden with an openIndex() option of the same name.

/** URL of the SQLite module (wasm/pkg/index.mjs). */
export const sqliteModule = new URL('../../wasm/pkg/index.mjs', import.meta.url).href;
/** Directory URL holding ort.wasm.min.mjs and ort-wasm-simd-threaded.{mjs,wasm}. */
export const ortBase = new URL('../vendor/ort/', import.meta.url).href;
/** URL of the ES module build of @huggingface/tokenizers. */
export const tokenizersModule = new URL('../vendor/tokenizers.min.mjs', import.meta.url).href;
/** Base URL that MODEL_FILES paths are resolved against. */
export const modelBase = new URL('../../', import.meta.url).href;
