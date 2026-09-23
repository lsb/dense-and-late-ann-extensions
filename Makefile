# Top-level build. Targets:
#   make sqlite    fetch and verify the pinned SQLite amalgamation (build/sqlite)
#   make native    sqlite3 CLI, libsqlite3.so.0, loadable extensions (build/native)
#   make wasm      WebAssembly builds into wasm/pkg/dist (needs emsdk)
#   make test      native, Node and Chromium tests
# Details and options: wasm/NOTES.md. The rules live in wasm/build.mk; other
# parts of the project can add their own .mk files and include them here.

.DEFAULT_GOAL := native
include wasm/build.mk
