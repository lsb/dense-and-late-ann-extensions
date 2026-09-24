# dense-late-ann (Python)

**dense-late-ann** packages the SQLite extensions of the [dense-and-late-ann-extensions](https://github.com/lsb/dense-and-late-ann-extensions) research project for native use, together with a command-line tool that builds databases for serving over HTTP range requests.

The wheel contains two loadable SQLite extensions, compiled from `ext/dense` and `ext/late` against the headers of SQLite 3.53.4:

| Extension | File | Virtual table |
|---|---|---|
| Dense ANN (PQ-64, HNSW-built graph or IVF-PQ) | `denseann.so` (`.dylib` on macOS) | `dense_ann` |
| Late interaction (PLAID and centroid-major layouts) | `late.so` | `late_plaid` |

## Installation

```sh
pip install dense-late-ann                 # extensions and CLI only
pip install 'dense-late-ann[encoders]'     # plus numpy, onnxruntime, tokenizers for encoding text
pip install 'dense-late-ann[apsw]'         # for Pythons whose sqlite3 cannot load extensions
```

## Use from Python

```python
import sqlite3, dense_late_ann
db = sqlite3.connect("search.db")
dense_late_ann.load(db)                    # dense_ann and late_plaid
print(dense_late_ann.extension_path("dense"))
db = dense_late_ann.connect("search.db")   # sqlite3, or apsw as a fallback, with both loaded
```

## Command-line tool

```sh
dense-late-ann check
dense-late-ann build-db corpus.txt --out search.db --fts --dense --late --models path/to/models
dense-late-ann search search.db "some query" --system dense --models path/to/models
dense-late-ann path dense                  # for the sqlite3 shell: .load /that/path
```

`build-db` reads one document per line (or JSON lines with a `text` field), encodes the documents with all-MiniLM-L6-v2 and LateOn-Code-edge unless precomputed vectors are given (`--minilm-npy`, `--lateon-npy`, `--lateon-offsets`), and writes the `docs` table, the FTS5 index and the chosen vector indexes. The models are not part of the package; `--models` (or `DENSE_LATE_ANN_MODELS`) names a directory with `minilm-l6-v2/{model_qint8_arm64.onnx,tokenizer.json}` and `lateon-code-edge/{model_int8.onnx,tokenizer.json}`.

The resulting file is read in browsers and Node by the npm package `dense-late-ann`. Full instructions: `docs/INSTALL.md` in the repository.

## License

AGPL-3.0-only.
