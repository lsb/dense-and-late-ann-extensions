# Top-level build. Targets:
#   make sqlite    fetch and verify the pinned SQLite amalgamation (build/sqlite)
#   make native    sqlite3 CLI, libsqlite3.so.0, loadable extensions (build/native)
#   make wasm      WebAssembly builds into wasm/pkg/dist (needs emsdk)
#   make test      native, Node and Chromium tests
#   make web       demo database (build/web) and the browser demo test
#   make serve     serve the repo with the shaped range server; open /web/
# Details and options: wasm/NOTES.md. The rules live in wasm/build.mk; other
# parts of the project can add their own .mk files and include them here.

.DEFAULT_GOAL := native
include wasm/build.mk

.PHONY: web serve
web: native wasm
	python3 web/make_demo_db.py
	node --test web/test/web.test.mjs

PRESET ?= 4g
serve:
	python3 netsim/rangeserver.py --dir . --port 8000 --preset $(PRESET)
