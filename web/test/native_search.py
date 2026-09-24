#!/usr/bin/env python3
"""Run searches natively (Python sqlite3 on build/native, extensions built by
tools/build_db.py from the current sources) for the browser test.

stdin: JSON {"db": path, "queries": [{"sql": ..., "args": [...], "vector": [floats] | null}]}
       (a query's first argument is the FTS5 MATCH text, or the float32 vector)
stdout: JSON [[rowid, ...], ...]
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
NATIVE = REPO / "build" / "native"

if os.environ.get("_NATIVE_SEARCH_REEXEC") is None:
    env = dict(os.environ, LD_LIBRARY_PATH=f"{NATIVE}:{os.environ.get('LD_LIBRARY_PATH', '')}",
               _NATIVE_SEARCH_REEXEC="1")
    os.execve(sys.executable, [sys.executable] + sys.argv, env)

sys.path.insert(0, str(REPO / "tools"))
import build_db  # noqa: E402

req = json.load(sys.stdin)
db = build_db.connect(req["db"])  # loads dense_ann and late_plaid
db.load_extension(str(NATIVE / "ext" / "fts5rank"))  # bm25c() for the default FTS ranking
out = []
for q in req["queries"]:
    args = list(q["args"])
    if q.get("vector") is not None:
        args[0] = np.asarray(q["vector"], np.float32).tobytes()
    out.append([r[0] for r in db.execute(q["sql"], args)])
print(json.dumps(out))
