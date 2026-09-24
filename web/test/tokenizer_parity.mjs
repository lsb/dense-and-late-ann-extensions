// Node check: web/lib/tokenize.mjs (on @huggingface/tokenizers) against the
// Python tokenizers used by enc/ (reference from web/test/make_reference.py).
//   node web/test/tokenizer_parity.mjs [build/web/reference-tokens.json]
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Tokenizer } from '../vendor/tokenizers.min.mjs';
import { makeTokenizers } from '../lib/tokenize.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const refPath = process.argv[2] || path.join(ROOT, 'build/web/reference-tokens.json');
const ref = JSON.parse(fs.readFileSync(refPath, 'utf8'));
const tok = makeTokenizers(Tokenizer, {
  minilmJson: JSON.parse(fs.readFileSync(path.join(ROOT, 'models/minilm-l6-v2/tokenizer.json'))),
  lateonJson: JSON.parse(fs.readFileSync(path.join(ROOT, 'models/lateon-code-edge/tokenizer.json'))),
});
let bad = { minilm: 0, lateon: 0 };
const t0 = performance.now();
ref.texts.forEach((text, i) => {
  const a = tok.minilm(text), b = tok.lateonQuery(text);
  if (JSON.stringify(a) !== JSON.stringify(ref.minilm[i])) {
    if (bad.minilm++ < 10) console.log('minilm mismatch', JSON.stringify(text.slice(0, 80)), a.slice(0, 20), ref.minilm[i].slice(0, 20));
  }
  if (JSON.stringify(b) !== JSON.stringify(ref.lateon_query[i])) {
    if (bad.lateon++ < 10) console.log('lateon mismatch', JSON.stringify(text.slice(0, 80)), b.slice(0, 20), ref.lateon_query[i].slice(0, 20));
  }
});
const ms = performance.now() - t0;
console.log(`${ref.texts.length} texts: minilm mismatches ${bad.minilm}, lateon mismatches ${bad.lateon} (${(ms / ref.texts.length / 2).toFixed(3)} ms per tokenization)`);
process.exit(bad.minilm + bad.lateon ? 1 : 0);
