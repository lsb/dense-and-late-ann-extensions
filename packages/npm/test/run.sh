#!/bin/sh
# Pack the npm package, install the tarball into a fresh project, and test it
# in Node and in Chromium (Playwright) against a database and the reference
# results written by python/tests/test_package.py.
#
#   packages/npm/test/run.sh OUT_DIR MODELS_DIR
#
# OUT_DIR must hold search.db and reference.json. The fresh project is created
# in OUT_DIR/project. Needs network access to the npm registry (for the peer
# dependencies onnxruntime-web and @huggingface/tokenizers) and Playwright with
# Chromium (set PLAYWRIGHT_BROWSERS_PATH if the browsers are not in the default
# place; `npx playwright install chromium` installs them).
set -eu
OUT=$(cd "$1" && pwd)
MODELS=$(cd "$2" && pwd)
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
PROJECT="$OUT/project"

rm -rf "$PROJECT" && mkdir -p "$PROJECT"
(cd "$ROOT/packages/npm" && npm pack --silent --pack-destination "$PROJECT")
cd "$PROJECT"
TARBALL=$(ls dense-late-ann-*.tgz)
npm init -y >/dev/null
npm pkg set type=module
npm install --no-audit --no-fund --silent "./$TARBALL" onnxruntime-web@1.30.0 @huggingface/tokenizers@0.2.0
cp "$OUT/search.db" "$OUT/reference.json" "$HERE/smoke.mjs" "$HERE/page.html" .
mkdir -p models
cp -r "$MODELS/minilm-l6-v2" "$MODELS/lateon-code-edge" models/
node smoke.mjs "$ROOT/netsim/rangeserver.py"
