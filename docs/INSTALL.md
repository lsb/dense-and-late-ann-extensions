# Installation and distribution

This document describes how the project is packaged for use outside the repository: an npm package for browsers and Node, a Python package for building databases natively, and loadable extensions for the `sqlite3` shell. It also describes the continuous-integration and release workflows that build and test these packages.

## Proposed README section

The following text is intended for the "Install" section of the top-level README.

````markdown
## Install

The project is distributed as two packages that share one database format. Both are named `dense-late-ann`; neither is published yet, so for now they are built from this repository (see [docs/INSTALL.md](docs/INSTALL.md)).

| Package | For | Contents |
|---|---|---|
| npm `dense-late-ann` | browsers and Node ≥ 22 | SQLite 3.53.4 in WebAssembly with FTS5, `dense_ann`, `late_plaid` and the HTTP range-request VFS; a search client with in-browser query encoders; TypeScript declarations |
| PyPI `dense-late-ann` | building databases (Linux, macOS) | the two extensions as loadable modules, `load(conn)`, and the `dense-late-ann` command |

Build a database natively, serve it as a static file, and query it in the browser:

```sh
pip install 'dense-late-ann[encoders]'
dense-late-ann build-db corpus.txt --out search.db --models models   # docs, FTS5, dense graph + IVF, late
```

```js
import { openIndex } from 'dense-late-ann/search';        // npm install dense-late-ann onnxruntime-web @huggingface/tokenizers
const ix = await openIndex('/search.db', { preload: ['minilm'] });
const { rows } = await ix.search('volcanic eruption', { system: 'dense', k: 10 });
```

The `sqlite3` shell loads the extensions with `.load` (paths from `dense-late-ann path`). The models (all-MiniLM-L6-v2 and LateOn-Code-edge, about 44 MB) are not part of either package; they are in `models/`.
````

## Overview

The research code is left where it is; the packages wrap it.

| Channel | Source of the contents | Built by | Tested by |
|---|---|---|---|
| npm package `dense-late-ann` | `wasm/pkg/` (SQLite WebAssembly builds and JavaScript API), `web/lib/` (search client), `packages/npm/` (package metadata, `search/paths.mjs`, type declarations) | `make wasm`, then `npm pack` in `packages/npm/` | `packages/npm/test/run.sh` |
| Python package `dense-late-ann` | `ext/dense/`, `ext/late/` (C sources), `enc/` (encoders), `tools/build_db.py` (database recipe), `python/dense_late_ann/` (API and command-line tool) | `pip install .`, `python -m build` | `python/tests/test_package.py` |
| Loadable extensions | `ext/dense/`, `ext/late/` | `make native` or the Python wheel | `make test-native`, `dense-late-ann check` |

Both packages carry the version 0.1.0 and the license AGPL-3.0-only. The release workflow refuses to run when the versions in `packages/npm/package.json`, `pyproject.toml` and `python/dense_late_ann/__init__.py` differ from each other or from the tag.

## npm package

### Contents

The package contains the three WebAssembly builds of SQLite (Asyncify, JSPI and the synchronous build for workers), the JavaScript API of `wasm/pkg/index.mjs`, the search client of `web/lib/`, and hand-written TypeScript declarations. The tarball is about 2.2 MB, of which 5.3 MB unpacked are the three `.wasm` files; a page loads only one of them.

| Export | Environment | Contents |
|---|---|---|
| `dense-late-ann` | browser, worker, Node ≥ 22 | `open(url, options)` returning a read-only `Database` |
| `dense-late-ann/search-core` | browser, worker, Node | `SearchDb`: index discovery, query SQL, document text; takes precomputed query vectors |
| `dense-late-ann/search` | browser | `openIndex(url, options)`: SQLite and the encoders in module Workers |
| `dense-late-ann/encoder` | browser, Node | `Encoder`: tokenization and ONNX inference for one query |

`onnxruntime-web` (tested with 1.30.0) and `@huggingface/tokenizers` (0.2.0) are optional peer dependencies. A user who needs only FTS5, or who computes query vectors elsewhere, installs the package alone (5.5 MB in `node_modules`); a user of the in-browser encoders installs all three.

`SearchDb` was factored out of the SQLite worker (`web/lib/search-core.mjs`) so that the same search code runs in Node, in a page and in the worker. The file `web/lib/paths.mjs` holds the default locations of the files the client loads at run time; the package replaces it with `packages/npm/search/paths.mjs`, which points at the package's own SQLite build and at the peer dependencies next to it in `node_modules/`. Every default can be overridden with an `openIndex()` option (`sqliteModule`, `ortBase`, `tokenizersModule`, `modelBase`, `models`).

### Usage

In Node:

```js
import { open } from 'dense-late-ann';
import { SearchDb } from 'dense-late-ann/search-core';
const s = await SearchDb.open(await open('https://example.org/search.db'));
await s.search({ system: 'fts', text: 'volcanic eruption', k: 10 });
await s.search({ system: 'dense', vector: minilmVector, k: 10 });   // Float32Array(384)
await s.search({ system: 'late', vector: lateonVectors, k: 10 });   // Float32Array(n × 48)
```

In a browser, the client loads Workers, WebAssembly files and ONNX Runtime by URL, so the package must be served as files. The tested arrangement serves a project directory, including `node_modules/`, with a static server and imports `/node_modules/dense-late-ann/search/index.mjs`. Bundlers that rewrite `import.meta.url` need the package excluded from dependency pre-bundling; this has not been tested.

The database server must support `Range` requests (206 responses with `Content-Range`) and, across origins, CORS that exposes `Content-Range`. Static hosts such as GitHub Pages, S3 and R2 do; Python's `http.server` does not. For local testing, `python3 netsim/rangeserver.py --dir DIR --preset none` serves a directory.

### Building the tarball

```sh
make sqlite wasm                 # wasm/pkg/dist
cd packages/npm && npm pack      # dense-late-ann-0.1.0.tgz
```

`npm pack` runs `assemble.mjs`, which copies `wasm/pkg` and `web/lib` into `packages/npm/` (the copies are ignored by git) and fails if a WebAssembly build is missing or older than the C sources, since a stale build silently lacks newer index layouts.

## Python package

### Contents

The wheel contains the two extensions compiled as plain loadable modules (`denseann.so` and `late.so`, `.dylib` on macOS) with multi-threaded index construction, as `tools/build_db.py` compiles them. SQLite loads them; Python does not import them. They therefore do not depend on the Python version, and the wheel is tagged `py3-none-<platform>`: one wheel per platform serves every Python 3. The research modules are installed in place, without copies: `enc/` becomes `dense_late_ann.enc` and `tools/` becomes `dense_late_ann.tools`, so the database recipe and the encoders have one source.

The extensions are compiled against the headers of the pinned SQLite 3.53.4 amalgamation. `setup.py` looks for `sqlite3.h` and `sqlite3ext.h` in `$SQLITE_INCLUDE_DIR`, in the sdist's copy, and in `build/sqlite/`, and otherwise runs `wasm/scripts/fetch-sqlite.sh`, which clones SQLite from GitHub and checks the SHA-256 of the result. The sdist includes the two headers, so it builds without network access.

### Installation

```sh
pip install dense-late-ann                  # extensions, load(), CLI (FTS5-only databases)
pip install 'dense-late-ann[encoders]'      # + numpy, onnxruntime, tokenizers: encode text, build vector indexes
pip install 'dense-late-ann[numpy]'         # + numpy only: vector indexes from precomputed vectors
pip install 'dense-late-ann[apsw]'          # for Pythons whose sqlite3 module cannot load extensions
pip install .                               # from a checkout
```

A C compiler is needed when no wheel exists for the platform. Windows is not supported: the extensions use POSIX threads.

### Python interface

```python
import sqlite3, dense_late_ann
db = sqlite3.connect("search.db")
dense_late_ann.load(db)                      # dense_ann and late_plaid; also accepts an apsw.Connection
dense_late_ann.extension_path("late")        # absolute path, for other drivers
db = dense_late_ann.connect("search.db")     # sqlite3, falling back to apsw, with both extensions loaded
```

The standard `sqlite3` module needs to have been built with extension loading (`enable_load_extension`). Most Linux distributions' Pythons have it; some builds, notably on macOS, do not, and `connect()` then falls back to apsw. The extensions work with the system SQLite as long as it has FTS5 (tested with 3.45.1) and with apsw's bundled SQLite.

### Command-line tool

| Command | Purpose |
|---|---|
| `dense-late-ann check` | load both extensions into an in-memory database and create one table of each kind |
| `dense-late-ann path [dense\|late]` | print the extension paths |
| `dense-late-ann build-db CORPUS --out DB` | build a deployable database |
| `dense-late-ann search DB TEXT --system fts\|dense\|dense_ivf\|late` | query a database natively (`--json` for machine-readable output) |
| `dense-late-ann encode-query TEXT --model minilm\|lateon` | write a query's vector or vectors as raw float32 |

`build-db` reads one document per line (`.txt`) or JSON lines with a `text` field (`.jsonl`); document ids are 0, 1, 2, … in file order, skipping empty lines. It writes the same database as `tools/build_db.py`: a `docs(id, body)` table, the FTS5 index `fts`, the `dense_ann` graph index `dense_graph`, the IVF-PQ index `dense_ivf` and the `late_plaid` index `late`, followed by `VACUUM` and each extension's `finalize`. The flags `--fts`, `--dense`, `--dense-ivf` and `--late` select indexes (all four by default); `--dense-params`, `--dense-ivf-params` and `--late-params` add or override index parameters; `--threads` sets encoder and build threads. Documents are encoded one at a time by default (`--batch-size 1`), because the int8 models give batch-dependent vectors and single-document encoding equals what a browser computes (see `enc/NOTES.md`). For large corpora, vectors computed beforehand with `enc/encode_corpus.py` are passed with `--minilm-npy`, `--lateon-npy` and `--lateon-offsets`.

The index parameters are those of `bench/matrix_config.json`; `python/tests/test_package.py` checks that the two agree.

## Loadable extensions for the sqlite3 shell

`make native` builds `build/native/ext/denseann.so` and `build/native/ext/late.so` (single-threaded index construction) together with a `sqlite3` shell that has both extensions, FTS5 and the HTTP VFS compiled in. With another `sqlite3` shell, the extensions are loaded by path; the file name gives SQLite's default entry point (`sqlite3_denseann_init`, `sqlite3_late_init`), so none needs to be named:

```
sqlite> .load /path/to/denseann
sqlite> .load /path/to/late
```

The Python wheel's copies (`dense-late-ann path`) work the same way and are built with multi-threaded index construction. The extensions are also attached to each GitHub release inside the wheels, which are zip files.

## Models

Neither package includes the models, which are about 44 MB. The repository's `models/` directory holds the files that all measurements used:

| File | Size | SHA-256 | Upstream |
|---|---:|---|---|
| `minilm-l6-v2/model_qint8_arm64.onnx` | 23.0 MB | `4278337f…02474` | `sentence-transformers/all-MiniLM-L6-v2`, `onnx/model_qint8_arm64.onnx` |
| `minilm-l6-v2/tokenizer.json` | 0.47 MB | `be50c362…72037` | same repository, revision `c9745ed1` |
| `lateon-code-edge/model_int8.onnx` | 17.2 MB | `eac35bda…f11c7` | `lightonai/LateOn-Code-edge`, revision `07ef20f4` |
| `lateon-code-edge/tokenizer.json` | 3.58 MB | `a388b949…505d0` | same |

The upstream paths and revisions are those recorded in `RESEARCH_LOG.md` and `enc/NOTES.md`; Hugging Face was not reachable from the development container, so a download from there was not tested. Other exports of the same models (for example float32, or other quantizations) produce different vectors and must not be mixed with a database built from these files.

The Python tool looks for the models in `--models DIR`, then `$DENSE_LATE_ANN_MODELS`, then `./models`, and in a source checkout in the repository's `models/`. The browser client fetches them relative to `modelBase` (by default the page's directory, `models/minilm-l6-v2/…` and `models/lateon-code-edge/…`), and keeps them in Cache Storage after the first visit.

## Continuous integration and releases

`.github/workflows/ci.yml` runs on every push and pull request on Ubuntu 24.04. It sets up Node 22, Python 3.12, Emscripten 6.0.10 (`mymindstorm/setup-emsdk`) and Playwright 1.56.1 with Chromium; builds `make sqlite native wasm`; runs `make test` (native, Node and Chromium tests of the WebAssembly runtime); installs the Python package with its encoders into a fresh virtual environment and runs `python/tests/test_package.py`, which builds a database from `data/corpora/llm-100.txt` and queries it; and finally runs `packages/npm/test/run.sh`, which packs the npm package, installs the tarball into a fresh project and checks it in Node and Chromium against the Python results.

`.github/workflows/release.yml` runs on tags `v*`:

1. `versions` checks that the tag and the three version fields agree.
2. `npm` builds the WebAssembly targets and packs the npm tarball.
3. `sdist` builds the Python source distribution, which includes the SQLite headers.
4. `wheels` builds wheels from the sdist with cibuildwheel on Linux x86_64 and aarch64 (manylinux) and macOS arm64 and x86_64, with one interpreter per platform, and runs `dense-late-ann check` on each.
5. `github-release` attaches the tarball, the sdist and the wheels to a GitHub release.
6. `publish-pypi` and `publish-npm` publish the packages, but only when the repository variable `PUBLISH_PACKAGES` is `true`. PyPI publishing uses trusted publishing (no stored token); npm publishing uses the secret `NPM_TOKEN` and adds provenance.

## Verification

The package tests can be repeated from a checkout after `make sqlite native wasm`:

```sh
python3 -m venv /tmp/venv && /tmp/venv/bin/pip install '.[encoders]'
/tmp/venv/bin/python python/tests/test_package.py --out-dir /tmp/dla     # builds /tmp/dla/search.db
packages/npm/test/run.sh /tmp/dla models                                   # packs and tests the npm package
```

The following was run in the development container (Linux x86_64, Python 3.11, Node 22.22, Chromium 141 from `/opt/pw-browsers`).

- **Python package from a checkout.** `python3 -m venv venv && venv/bin/pip install .` produced `dense_late_ann-0.1.0-py3-none-linux_x86_64.whl`; `dense-late-ann check` loaded both extensions into the system SQLite 3.45.1. Without the `encoders` extra, an FTS5-only `build-db` worked and a vector index stopped with a message naming the extra. After `pip install '.[encoders]'`, `dense-late-ann build-db data/corpora/llm-100.txt --out llm-100.db --models models` built all four indexes in 1.8 s (1.38 MiB).
- **Same results as the benchmark database.** For the 200 queries of `data/queries/llm-100.jsonl`, the dense graph and late (warp) top-10 lists of that database were identical, query by query, to those of `build/matrix/llm-100.db` built by `tools/build_db.py`.
- **Python package from the sdist.** `python -m build --sdist` included the SQLite headers; `pip wheel` on the sdist built the wheel, which was installed into a second fresh environment, where `python/tests/test_package.py` passed. `load()` also worked with an `apsw.Connection` (SQLite 3.53.4).
- **sqlite3 shell.** `.load` of the wheel's `denseann` and `late` in `build/native/sqlite3` answered a `dense_ivf` query.
- **npm package.** `packages/npm/test/run.sh` packed the tarball, installed it with the two peer dependencies into a fresh project, and checked: in Node, the asyncify and sync builds returned the same ids as the native extensions for 4 queries × 4 indexes; in Chromium, the asyncify and JSPI builds did the same in a page importing the package from `node_modules/`; and `openIndex()` with in-browser encoding returned identical FTS5 results, the same top hit everywhere, and a mean top-10 overlap of 0.992 with the Python-encoded results. Installing the tarball alone pulled in no other package. The type declarations compiled with `tsc --strict`.
- **Repository tests after the client refactoring.** `node --test web/test/web.test.mjs` (demo page, encoder parity, browser results equal native) passed, as did `make test`.
- **Workflows.** Both workflow files passed `actionlint` 1.7.7. The version check, `make sqlite` with the revised `fetch-sqlite.sh` in a new directory, `python -m build --sdist` and `npm pack` were run locally; cibuildwheel and the GitHub-hosted runners themselves were not, and the macOS build is untested.

## Decisions for the maintainer

- **Package names.** `dense-late-ann` was free on both npm and PyPI on 2026-09-24 and is used as a placeholder for both, with the import name `dense_late_ann` and the command `dense-late-ann`. A scoped npm name (for example `@owner/dense-late-ann`) needs only the `name` field changed; `search/paths.mjs` finds the peer dependencies wherever `node_modules/` is.
- **Publishing credentials.** PyPI: register a trusted publisher for this repository, workflow `release.yml`, environment `pypi`. npm: create an automation token and store it as the secret `NPM_TOKEN`. Then set the repository variable `PUBLISH_PACKAGES` to `true`.
- **Models.** Whether to host the model files (for example as GitHub release assets, or by pointing `modelBase` at Hugging Face after checking the files match the digests above) or to keep asking users to copy `models/`.
- **License.** Both packages are AGPL-3.0-only, like the repository. A page that loads the npm package from a public site is distributing it, which the AGPL's terms then cover.
- **Platforms.** Windows wheels would need the extensions' threads ported to Windows threads; musllinux wheels could be added to `CIBW_BUILD` if wanted.
