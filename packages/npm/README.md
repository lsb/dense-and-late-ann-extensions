# dense-late-ann

**dense-late-ann** is a read-only SQLite for browsers and Node that reads a database lazily over HTTP range requests, with full-text search (FTS5), dense approximate nearest-neighbour search (`dense_ann`) and late-interaction search (`late_plaid`) compiled in. A search client that encodes queries in the browser with all-MiniLM-L6-v2 and LateOn-Code-edge is included. It is the JavaScript distribution of the [dense-and-late-ann-extensions](https://github.com/lsb/dense-and-late-ann-extensions) research project.

Databases are built natively, for example with the Python package of the same name (`pip install 'dense-late-ann[encoders]'`, then `dense-late-ann build-db corpus.txt --out search.db`), and served as static files by any server that supports `Range` requests.

## Installation

```sh
npm install dense-late-ann                                           # SQLite + extensions (FTS5 and precomputed vectors)
npm install dense-late-ann onnxruntime-web @huggingface/tokenizers   # plus in-browser query encoding
```

`onnxruntime-web` (tested: 1.30.0) and `@huggingface/tokenizers` (0.2.0) are optional peer dependencies. They are needed only by the search client's encoders.

## Contents

| Import | Environment | Purpose |
|---|---|---|
| `dense-late-ann` | browser, worker, Node ≥ 22 | `open(url, options)`: a read-only `Database` with `query`, `queryRaw`, `exec`, `stats`, `log`, `resetStats`, `close` |
| `dense-late-ann/search-core` | browser, worker, Node | `SearchDb`: finds the indexes of a database and runs FTS5, dense and late searches with precomputed query vectors |
| `dense-late-ann/search` | browser | `openIndex(url, options)`: SQLite and each encoder in their own module Worker; `search(text, {system})` encodes and searches |
| `dense-late-ann/encoder` | browser, Node | `Encoder`: tokenization and ONNX inference for one query |

Three WebAssembly builds are included (`dist/`): Asyncify (every browser and Node), JSPI (Chromium 137 and later; chosen automatically where available) and a synchronous build for workers of cross-origin-isolated pages and Node. Each is about 0.6–0.9 MB compressed; a page loads only one.

## Node

```js
import { open } from 'dense-late-ann';
import { SearchDb } from 'dense-late-ann/search-core';

const db = await open('https://example.org/search.db');
const s = await SearchDb.open(db);
await s.search({ system: 'fts', text: 'volcanic eruption', k: 10 });
await s.search({ system: 'dense', vector: minilmVector, k: 10 });      // Float32Array(384)
await s.search({ system: 'late', vector: lateonVectors, k: 10 });      // Float32Array(n × 48)
console.log(db.stats());                                                // requests, bytes, rounds, …
```

## Browser

```js
import { openIndex } from 'dense-late-ann/search';
const ix = await openIndex('/search.db', { preload: ['minilm', 'lateon'], modelBase: '/' });
const r = await ix.search('volcanic eruption', { system: 'dense', k: 10 });
r.rows;   // [{id, score, text}]
```

The client loads its workers, the WebAssembly files and ONNX Runtime by URL at run time, so the package directory must be reachable by the browser. The tested set-up serves the project directory, including `node_modules/`, as static files and imports `/node_modules/dense-late-ann/search/index.mjs`. The defaults then find `onnxruntime-web` and `@huggingface/tokenizers` next to the package in `node_modules/`. Elsewhere, pass `sqliteModule`, `ortBase` (the directory holding `ort.wasm.min.mjs` and `ort-wasm-simd-threaded.wasm`) and `tokenizersModule` (the URL of `tokenizers.min.mjs`). Bundlers that rewrite `import.meta.url` need the package excluded from dependency pre-bundling (in Vite, `optimizeDeps.exclude: ['dense-late-ann']`); this has not been tested.

The models are not in the package. By default they are fetched from `models/minilm-l6-v2/model_qint8_arm64.onnx`, `models/minilm-l6-v2/tokenizer.json`, `models/lateon-code-edge/model_int8.onnx` and `models/lateon-code-edge/tokenizer.json` relative to `modelBase` (default: the page's directory); `models: {minilm: {model, tokenizer}, lateon: {…}}` overrides single files. The two models are about 44 MB and are kept in Cache Storage after the first visit.

## Server requirements

The server must answer `Range` requests with `206 Partial Content` and a `Content-Range` header, and allow them cross-origin if the database is on another origin (CORS: expose `Content-Range`). GitHub Pages, S3, R2 and most CDNs do. Python's `http.server` does not support ranges.

## License

AGPL-3.0-only. The models keep their upstream licenses.
