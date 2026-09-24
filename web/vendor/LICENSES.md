# Vendored third-party files

Copied from the npm registry by `web/vendor.sh` (versions pinned in `web/package.json`). Nothing is loaded from a CDN.

| File | Package | Version | License | SHA-256 |
|---|---|---|---|---|
| `tokenizers.min.mjs` | `@huggingface/tokenizers` (`dist/tokenizers.min.mjs`) | 0.2.0 | Apache-2.0 (`tokenizers.LICENSE`) | `258e4633…b1d3a` |
| `ort/ort.wasm.min.mjs` | `onnxruntime-web` (`dist/`) | 1.30.0 | MIT | `219e6a1f…6cee3b` |
| `ort/ort-wasm-simd-threaded.mjs` | `onnxruntime-web` (`dist/`) | 1.30.0 | MIT | `e13f7f94…299b` |
| `ort/ort-wasm-simd-threaded.wasm` | `onnxruntime-web` (`dist/`) | 1.30.0 | MIT | `3398c10d…dee2` |

The `onnxruntime-web` npm package does not ship a license file; its `package.json` declares MIT. The license text (from github.com/microsoft/onnxruntime) is:

```
MIT License

Copyright (c) Microsoft Corporation

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

The model files the demo loads (`models/`) keep their upstream licenses (all-MiniLM-L6-v2: Apache-2.0; LateOn-Code-edge: see its model card).
