#!/usr/bin/env python3
"""Test an installed dense-late-ann Python package, end to end.

  python python/tests/test_package.py --corpus data/corpora/llm-100.txt \
      --models models --out-dir /tmp/dla

1. `dense-late-ann check` loads both extensions.
2. In a repository checkout, the CLI's index definitions equal those of
   bench/matrix_config.json.
3. `dense-late-ann build-db` builds a database with all four indexes.
4. Each system answers a few queries (`dense-late-ann search`); a query made
   from a document's own words finds that document first by FTS5.
5. OUT_DIR/reference.json records the query vectors and the expected result
   ids, which packages/npm/test/run.sh uses to check the npm package (the WASM
   build must return exactly the same ids for the same vectors).

Only the standard library is needed besides the package and its [encoders]
extra; run it with the Python the package is installed in.
"""
import argparse
import array
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
QUERIES = ["volcanic eruption", "wine regions and vineyards", "how do computers store numbers"]
SYSTEMS = ["fts", "dense_graph", "dense_ivf", "late"]


def cli(*args, capture=True):
    cmd = [sys.executable, "-m", "dense_late_ann", *map(str, args)]
    print("$", " ".join(cmd[2:]), flush=True)
    r = subprocess.run(cmd, check=True, text=True, capture_output=capture)
    return r.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(REPO / "data/corpora/llm-100.txt"))
    ap.add_argument("--models", default=str(REPO / "models"))
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    import dense_late_ann
    from dense_late_ann import cli as dcli
    print(cli("check"), end="")

    cfg_path = REPO / "bench" / "matrix_config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())["indexes"]
        for key, spec in dcli.INDEXES.items():
            for field in ("kind", "table", "params", "embedding"):
                assert spec.get(field) == cfg[key].get(field), (key, field, spec.get(field), cfg[key].get(field))
        assert set(cfg) == set(dcli.INDEXES), (set(cfg), set(dcli.INDEXES))
        print("index definitions match bench/matrix_config.json")

    db = out / "search.db"
    cli("build-db", args.corpus, "--out", db, "--force", "--models", args.models, capture=False)

    docs = dcli.read_corpus(args.corpus)
    own = " ".join(docs[3].split()[:6])   # words of document 3
    ref = {"db": db.name, "corpus": str(args.corpus), "queries": []}
    for text in QUERIES + [own]:
        q = {"text": text, "expected": {}}
        for model in ("minilm", "lateon"):
            raw = subprocess.run([sys.executable, "-m", "dense_late_ann", "encode-query", text, "--model", model,
                                  "--models", args.models], check=True, capture_output=True).stdout
            v = array.array("f")
            v.frombytes(raw)
            q[model] = list(v)
        for system in SYSTEMS:
            r = json.loads(cli("search", db, text, "--system", system, "--json", "--models", args.models))
            q["expected"][system] = [row["id"] for row in r["rows"]]
            assert system == "fts" or len(r["rows"]) == 10, (system, text, r)
        ref["queries"].append(q)
    assert ref["queries"][-1]["expected"]["fts"][0] == 3, ref["queries"][-1]["expected"]["fts"]
    for system in ("dense_graph", "dense_ivf", "late"):
        assert 3 in ref["queries"][-1]["expected"][system][:3], (system, ref["queries"][-1]["expected"][system])
    (out / "reference.json").write_text(json.dumps(ref))
    print(f"ok: {db} and {out / 'reference.json'} (package {dense_late_ann.__version__})")


if __name__ == "__main__":
    main()
