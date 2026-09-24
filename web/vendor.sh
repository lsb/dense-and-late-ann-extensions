#!/bin/sh
# Refresh web/vendor/ from the npm registry (versions pinned in package.json).
# Only the files the demo loads are copied; see web/vendor/LICENSES.md.
set -eu
cd "$(dirname "$0")"
npm install --no-audit --no-fund --silent
mkdir -p vendor/ort
cp node_modules/@huggingface/tokenizers/dist/tokenizers.min.mjs vendor/
cp node_modules/@huggingface/tokenizers/LICENSE vendor/tokenizers.LICENSE
for f in ort.wasm.min.mjs ort-wasm-simd-threaded.mjs ort-wasm-simd-threaded.wasm; do
  cp node_modules/onnxruntime-web/dist/$f vendor/ort/
done
ls -l vendor vendor/ort
