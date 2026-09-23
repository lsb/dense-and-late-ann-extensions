# dense-and-late-ann-extensions

**dense-and-late-ann-extensions** is a research project that builds SQLite extensions for approximate nearest-neighbour (ANN) search over text embeddings, designed to run in a web browser against a database fetched over HTTP range requests (the "httpvfs" technique). It covers two retrieval families:

- **Dense retrieval**: one vector per document, from [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (384 dimensions), indexed with a graph index (HNSW / DiskANN style) over product-quantized (PQ) codes of 64 bytes per document.
- **Late interaction**: one vector per token, from [LateOn-Code-edge](https://huggingface.co/lightonai/LateOn-Code-edge) (48 dimensions per token), indexed with a PLAID-style centroid-and-residual index following [fast-plaid](https://github.com/lightonai/fast-plaid).

SQLite's own full-text search (FTS5) is the lexical baseline.

## Status

Work in progress. See [RESEARCH_LOG.md](RESEARCH_LOG.md) for a dated record of decisions and measurements.

## Repository layout

| Path | Contents |
|---|---|
| `models/minilm-l6-v2/` | all-MiniLM-L6-v2 ONNX (qint8) and tokenizer |
| `models/lateon-code-edge/` | LateOn-Code-edge ONNX (int8) and tokenizer |
| `models/lfm2.5-350m/` | LiquidAI LFM2.5-350M q4f16 ONNX, weights split into four chunks; run `scripts/assemble_lfm.py` before use |
| `data/words/` | Word list (Debian `wamerican`) and its deterministic shuffle |
| `data/corpora/` | Random-word corpora of 50-word documents (the 1M corpus is regenerated, not checked in) |
| `data/llm/` | LLM-generated paragraphs and queries ("LLM slop") |
| `scripts/` | Data generation scripts |

## Reproducing the data

```sh
pip install onnxruntime tokenizers numpy
python3 scripts/make_corpora.py            # words-100, words-10k, words-1m
python3 scripts/assemble_lfm.py            # rebuild LFM external-data files
python3 scripts/llm_generate.py paragraphs 10000 data/llm/paragraphs-10k.jsonl 32
```

## License

AGPL-3.0; see [LICENSE](LICENSE). Model files keep their upstream licenses.
