# Encoders: notes and measurements

This directory holds the Python reference encoders for the two embedding models and a corpus encoding tool.

| File | Purpose |
|---|---|
| `minilm.py` | all-MiniLM-L6-v2 dense encoder (qint8 ONNX), returns float32 `[n, 384]` unit vectors |
| `lateon.py` | LateOn-Code-edge late-interaction encoder (int8 ONNX), returns one `[n_tokens, 48]` float32 matrix per text |
| `lateon_onnx_config.json` | Byte-exact reconstruction of the model's `onnx_config.json` (see below) |
| `encode_corpus.py` | Resumable, chunked corpus encoder writing to `data/emb/` |
| `validate_minilm.py` | qint8 vs fp32 comparison and throughput |
| `validate_lateon.py` | MaxSim sanity examples, token statistics and throughput |

All scripts run from the repository root as modules, for example `python3 -m enc.validate_lateon`. The ONNX Runtime thread count defaults to 2 (environment variable `ENC_THREADS` or `--threads`).

## all-MiniLM-L6-v2

The encoder follows sentence-transformers: strip the text, WordPiece-tokenize with the BERT normalizer (lower-casing, accent handling as configured in `tokenizer.json`), add `[CLS]`/`[SEP]`, truncate to 256 word pieces including the special tokens, mean-pool `last_hidden_state` over the attention mask, then L2-normalize. The `tokenizer.json` taken from npm carries the Hugging Face tokenizer default of truncation at 128 with fixed padding to 128; the encoder overrides both, because sentence-transformers uses `max_seq_length = 256` and dynamic padding. At 50 words per document nothing reaches either limit (random-word documents average 111 word pieces, LLM paragraphs 64).

### qint8 vs fp32 (measured 2026-09-23)

Reference: `model.onnx` from the npm package `@xcidos/genesis-memory-model` 0.1.0-alpha.1, an fp32 export of the same checkpoint. Texts: 384 LLM paragraphs (first 50 words), 1,000 random-word documents and 768 queries (each seed word, and "a paragraph about *word*").

| Measure | Value |
|---|---|
| Cosine(qint8, fp32), mean over 2,152 texts | 0.9877 (min 0.9595, 1st percentile 0.9766) |
| LLM paragraphs / random-word documents / queries | 0.9925 / 0.9848 / 0.9891 |
| Top-1 agreement (768 queries over 1,384 documents) | 95.2 % |
| Top-10 overlap | 77.0 % (lower because random-word documents score in near ties) |
| Mean Spearman correlation of full score vectors | 0.980 |
| Recall@1 of the seed-word paragraph, qint8 / fp32 | 0.855 / 0.862 |
| Recall@10, qint8 / fp32 | 0.906 / 0.910 |

The quantized model loses under one point of recall on this task; rankings of clearly relevant documents agree, while near-ties among irrelevant documents reorder.

### Batch dependence of the quantized models

Both ONNX files use dynamic int8 quantization (`DynamicQuantizeLinear`), whose scale and zero point are computed over the whole activation tensor of the batch, padding included. A text's embedding therefore depends on the other texts in its batch. Measured cosine between encoding alone and in a batch: MiniLM mean 0.988 (min 0.980, batch 64); LateOn per-token mean 0.994, minimum 0.48 (batch 32). Because a browser encodes a single query alone, and because reproducibility matters, `encode_corpus.py` encodes one document at a time by default (`--batch-size 1`). Re-running it gives bit-identical output.

### Throughput (2 threads, CPU shared with an LLM generation job)

| Setting | docs/s |
|---|---|
| qint8, batch 64, 50-word random-word documents | 166 |
| qint8, batch 1 | 126–133 |
| fp32, batch 64 | 83 |

## LateOn-Code-edge

### Settings and evidence

The model's `config_sentence_transformers.json` and `onnx_config.json` are not downloadable here. The settings were recovered as follows.

1. **Our files are the published revision.** The project `oimiragieo/tensor-grep` (`src/tensor_grep/core/retrieval_late.py`) pins `lightonai/LateOn-Code-edge` revision `07ef20f4…` with SHA-256 hashes of `model_int8.onnx` (`eac35bda…11c7`, 17,228,399 bytes), `tokenizer.json` (`a388b949…05d0`, 3,583,847 bytes) and `onnx_config.json` (`fa4fef89…6651`, 792 bytes). Our two model files have exactly these hashes.
2. **The config was reconstructed byte-for-byte.** next-plaid-onnx's exporter writes `onnx_config.json` with `json.dump(config, indent=2)` and a fixed key order (`next-plaid/next-plaid-onnx/python/src/colbert_export/export.py:249-273`). Enumerating the plausible values of the unknown fields and hashing each candidate gives exactly one 792-byte file with the pinned SHA-256; it is saved as `enc/lateon_onnx_config.json`. This is conclusive evidence for every field:

| Field | Value |
|---|---|
| `query_prefix` / `document_prefix` | `"[Q] "` (id 50368) / `"[D] "` (id 50369) |
| `query_length` / `document_length` | 256 / 2048 |
| `do_query_expansion` | **false** |
| `attend_to_expansion_tokens` | false |
| `skiplist_words` | the 32 characters of Python's `string.punctuation` |
| `do_lower_case` | **true** |
| `uses_token_type_ids` | false (the graph has only `input_ids`, `attention_mask`) |
| `mask_token_id` / `pad_token_id` | 50284 / 50284 (PyLate sets the pad token to `[MASK]`) |
| `embedding_dim` | 48 |

Corroborating evidence, weaker on its own: the LateOn-Code training scripts in PyLate (`pylate/examples/train/lateon_code/pre-training.py:189-208`) use `document_length = 2048`, `query_length = 256`, starting from `mixedbread-ai/mxbai-edge-colbert-v0-17m`; the stale truncation of 2047 (= 2048 − 1) and `[MASK]` batch padding frozen into `tokenizer.json`; and two third-party projects (`mrsladoje/sweet-search` `docs/LATE_INTERACTION.md`, `mrsladoje/alice-ingest` `tools/embed/adapters.py`) that copied the Hugging Face model card: `Transformer({'max_seq_length': 2047, 'do_lower_case': True})`, then `Dense(256→512)`, `Dense(512→48)`, no bias, identity activation. The training scripts pass `skiplist_words=[]`, which disagrees with the published config; the published config is what inference uses.

### Preprocessing, exactly

Following PyLate `ColBERT.tokenize` (`pylate/models/colbert.py:1144-1202`) and next-plaid-onnx `prepare_batch_from_tokenizer_encodings` (`next-plaid/next-plaid-onnx/src/lib.rs:2140-2285`):

1. `text.strip()`, then `text.lower()` (sentence-transformers applies `do_lower_case` after stripping).
2. Tokenize with special tokens: `[CLS] t1 … tn [SEP]` (ids 50281 … 50282), with no truncation or padding from `tokenizer.json`.
3. Truncate to `L − 1` tokens (L = 256 for queries, 2048 for documents), keeping the final `[SEP]`: if the sequence is longer than `L − 1`, keep its first `L − 2` tokens and append `[SEP]`.
4. Insert the prefix id at position 1, right after `[CLS]`: `[CLS] [Q] t1 … [SEP]`. The prefix is attended.
5. Queries: no `[MASK]` expansion. A query yields one vector per real token, including `[CLS]`, `[Q] ` and `[SEP]` (3 + number of word pieces). Nothing is filtered.
6. Documents: all tokens are attended; afterwards the output vectors of tokens whose id is in the skiplist are dropped. The skiplist is the ids of the 32 punctuation characters as whole tokens (ids 2–16, 27–33, 60–65, 92–95). Byte-level BPE tokens with a leading space, such as `Ġ(`, are different ids and are kept. `[CLS]`, `[D] ` and `[SEP]` are kept.
7. The ONNX graph ends with the two projections and `ReduceL2 → Clip → Div`, so every output vector already has unit norm; PyLate's extra normalization is a no-op.

Score: MaxSim, `score(q, d) = Σ_i max_j q_i · d_j`.

When queries are short, `do_query_expansion = false` matters: an expanded query would carry 256 vectors, which would multiply the cost of every PLAID stage. Note that next-plaid-onnx, if expansion were on, would attend to the `[MASK]` tokens regardless of `attend_to_expansion_tokens` (`lib.rs:2202-2210`), whereas PyLate attends to them only when that flag is set (`colbert.py:1199-1201`); with this model the difference does not arise.

### Sanity checks

Five handmade query/document sets (code and English), relevant document first: all five rank the relevant document first, for example "what is the capital of France": 6.62 for the Paris sentence against 4.07–4.49 for the others; "reverse a string": 4.58 for `return s[::-1]` against 2.52–2.83.

### Tokens per 50-word document

| Corpus | Model input tokens | Stored vectors (after skiplist) |
|---|---|---|
| Random words (`words-10k`, 10,000 docs) | 111.4 (min 90, max 138) | 111.4 |
| LLM paragraphs (first 50 words, 480 rows available) | 65.3 (min 35, max 122) | 58.7 |

The random-word corpus is expensive for a byte-level BPE vocabulary: rare dictionary words split into about two tokens each, and there is no punctuation to drop. The LLM paragraphs lose about 10 % of vectors to the skiplist.

### Throughput (2 threads, shared CPU)

| Setting | docs/s |
|---|---|
| Random-word documents, batch 16 / 64 | 272 / 266 |
| Random-word documents, batch 1 (corpus default) | 150–187 |
| LLM 50-word documents, batch 32 | 489 |
| Single-word queries | about 3,600 queries/s |

## Corpus encodings (`data/emb/`, git-ignored)

Produced on 2026-09-23 with `python3 -m enc.encode_corpus <input> --model both` (batch size 1, 2 threads):

| Name | Docs | MiniLM | LateOn vectors | LateOn size |
|---|---|---|---|---|
| `words-100` | 100 | 77 KB | 11,086 | 1.1 MB |
| `words-10k` | 10,000 | 7.4 MB (78 s) | 1,114,000 (111.4/doc) | 102 MB (65 s) |
| `llm-paragraphs` | 640 rows present at the time | 0.5 MB | 58.7/doc | 3.5 MB |

File formats: `NAME.minilm.npy` float16 `[n, 384]`; `NAME.lateon.vectors.npy` float16 `[T, 48]`; `NAME.lateon.offsets.npy` int64 `[n + 1]`; `NAME.lateon.doclens.npy` int32 `[n]`; `NAME.<model>.json` metadata. Rerunning the LLM corpus later re-encodes only its incomplete tail chunk.

### Estimate for the 1M random-word corpus (not run)

| Model | Time at measured rate (2 threads) | Final output | Peak disk |
|---|---|---|---|
| MiniLM | 1,000,000 / ~130 docs/s ≈ 2.1 h | 768 MB | 1.5 GB (0.8 GB with `--delete-chunks`) |
| LateOn | 1,000,000 / ~155 docs/s ≈ 1.8 h | 111.4 M vectors × 96 B ≈ 10.7 GB, plus 8 MB offsets | 21.4 GB (≈ 11 GB with `--delete-chunks`) |

About 26 GB of disk is free, so the LateOn run should use `--delete-chunks`. Running the two models in parallel on 2 threads each would take about 2 hours if the CPU were otherwise idle. Throughput could rise about 30 % with batching, at the cost of batch-dependent (non-reproducible) vectors, as explained above.
