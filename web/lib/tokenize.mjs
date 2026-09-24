// Query tokenization for the two encoders, reproducing enc/minilm.py and
// enc/lateon.py on top of @huggingface/tokenizers (vendored in web/vendor/).
//
//   const tok = await loadTokenizers({ minilm: tokenizerJson, lateon: tokenizerJson2 });
//   tok.minilm('some query')        -> [101, ..., 102]      (<= 256 ids)
//   tok.lateonQuery('some query')   -> [50281, 50368, ..., 50282]  (<= 256 ids)
//
// The Tokenizer class is passed in so this module works with any import path
// (browser: ../vendor/tokenizers.min.mjs, Node: the npm package).

export const MINILM_MAX_LEN = 256;           // sentence-transformers max_seq_length
export const LATEON_QUERY_LENGTH = 256;      // onnx_config.json query_length
export const LATEON_QUERY_PREFIX_ID = 50368; // "[Q] "

// Python's str.strip() removes Unicode whitespace; JS trim() removes a
// slightly different set (it also strips U+FEFF, but not U+001C..U+001F).
// Match Python exactly.
const PY_WS = '\\t\\n\\x0b\\x0c\\r\\x1c-\\x1f \\x85\\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000';
const STRIP_RE = new RegExp(`^[${PY_WS}]+|[${PY_WS}]+$`, 'g');
export function pyStrip(s) { return String(s).replace(STRIP_RE, ''); }

/** HF "LongestFirst" truncation of a single [CLS] … [SEP] sequence to maxLen ids. */
function truncateKeepSep(ids, maxLen) {
  if (ids.length <= maxLen) return ids;
  return ids.slice(0, maxLen - 1).concat(ids[ids.length - 1]);
}

export function makeTokenizers(Tokenizer, { minilmJson, lateonJson }) {
  const out = {};
  if (minilmJson) {
    const t = new Tokenizer(minilmJson, {});
    out.minilm = (text) => {
      // The Rust BertNormalizer lower-cases code point by code point, with no
      // final-sigma rule; JS toLowerCase() on a whole string maps a word-final
      // capital sigma to ς instead of σ. U+03A3 is the only context-dependent
      // mapping in toLowerCase(), so pre-map it (special tokens such as [CLS]
      // are matched on the raw text and must not be lower-cased here).
      const lowered = pyStrip(text).replace(/\u03a3/g, '\u03c3');
      const ids = t.encode(lowered, { add_special_tokens: true }).ids;
      return truncateKeepSep(ids, MINILM_MAX_LEN);
    };
  }
  if (lateonJson) {
    const t = new Tokenizer(lateonJson, {});
    out.lateonQuery = (text) => {
      // enc/lateon.py: strip, lower(), tokenize with [CLS] … [SEP], truncate
      // to query_length - 1 keeping [SEP], insert the [Q] prefix after [CLS].
      const ids = t.encode(pyStrip(text).toLowerCase(), { add_special_tokens: true }).ids;
      const tr = truncateKeepSep(ids, LATEON_QUERY_LENGTH - 1);
      return [tr[0], LATEON_QUERY_PREFIX_ID, ...tr.slice(1)];
    };
  }
  return out;
}
