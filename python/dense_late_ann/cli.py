"""Command-line interface: `dense-late-ann <command>` (or `python -m dense_late_ann`).

  path [dense|late]           print the path of the loadable extensions
  check                       load both extensions into an in-memory database
  build-db CORPUS --out DB    build a deployable database (docs, FTS5, dense, late)
  encode-query TEXT --model   write a query's vector(s) as raw float32
  search DB TEXT              query a database natively

The database recipe is tools/build_db.py's (installed as
dense_late_ann.tools.build_db), and the encoders are enc/minilm.py and
enc/lateon.py (dense_late_ann.enc). See docs/INSTALL.md.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

from . import EXTENSIONS, connect, extension_path

# Index definitions; the same as the "indexes" of bench/matrix_config.json in
# the repository (python/tests/test_package.py checks that they agree).
INDEXES = {
    "fts": {"kind": "fts5", "table": "fts"},
    "dense_graph": {"kind": "dense_ann", "table": "dense_graph", "embedding": "minilm",
                    "params": "dim=384, layout=colocated, vectors=inline, store_vectors=f16, M=16, "
                              "ef_construction=200, threads=2"},
    "dense_ivf": {"kind": "dense_ann", "table": "dense_ivf", "embedding": "minilm",
                  "params": "dim=384, layout=ivf, store_vectors=f16, threads=2"},
    "late": {"kind": "late_plaid", "table": "late", "embedding": "lateon",
             "params": "dim=48 nbits=2 centroids=0 layout=both order=1 threads=2 input=f16"},
}
PAGE_SIZE = 4096

# Query defaults, as in web/lib/search-core.mjs (DEFAULT_PARAMS).
QUERY_DEFAULTS = {
    "dense:graph": {"ef": 64, "beam": 16, "rerank": 2},
    "dense:ivf": {"nprobe": 16, "rerank_k": 64, "rerank": 2},
    "late": {"nprobe": 8, "layout": "warp", "stoplist": 0.02, "rerank": 64},
}

MODEL_FILES = {
    "minilm": ("minilm-l6-v2", "model_w8.onnx", "tokenizer.json"),
    "lateon": ("lateon-code-edge", "model_w8.onnx", "tokenizer.json"),
}


# ------------------------------------------------------------------ helpers

def _build_db():
    from .tools import build_db
    return build_db


def models_dir(arg):
    """The directory holding minilm-l6-v2/ and lateon-code-edge/."""
    here = Path(__file__).resolve()
    candidates = [arg, os.environ.get("DENSE_LATE_ANN_MODELS"), Path.cwd() / "models",
                  here.parent.parent.parent / "models"]   # a source checkout (editable install)
    for c in candidates:
        if c and any((Path(c) / sub).is_dir() for sub, _, _ in MODEL_FILES.values()):
            return Path(c)
    sys.exit("models not found: pass --models DIR or set DENSE_LATE_ANN_MODELS to a directory holding "
             "minilm-l6-v2/{model_w8.onnx,tokenizer.json} and "
             "lateon-code-edge/{model_w8.onnx,tokenizer.json} (see docs/INSTALL.md)")


def encoder(which, mdir, threads):
    sub, model, tok = MODEL_FILES[which]
    m, t = Path(mdir) / sub / model, Path(mdir) / sub / tok
    for p in (m, t):
        if not p.is_file():
            sys.exit(f"missing model file {p}")
    try:
        if which == "minilm":
            from .enc.minilm import MiniLM
            return MiniLM(m, t, threads=threads)
        from .enc.lateon import LateOn
        return LateOn(m, t, threads=threads)
    except ImportError as e:
        sys.exit(f"{e}; the encoders need: pip install 'dense-late-ann[encoders]'")


def read_corpus(path):
    """One document per line (.txt), or the "text" field of each line (.jsonl).
    Empty lines are skipped; document ids are 0, 1, 2, ... in file order."""
    docs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            docs.append(json.loads(line)["text"] if str(path).endswith(".jsonl") else line.rstrip("\n"))
    return docs


def params_with(spec, extra, threads):
    """spec's params with `extra` (k=v list) and threads=N overriding."""
    over = " ".join(x for x in (extra or "", f"threads={threads}" if threads else "") if x)
    return _build_db().index_params({**spec, "per_corpus": {"_": over}} if over else spec, "_")


# ------------------------------------------------------------------ commands

def cmd_path(args):
    for n in ([args.name] if args.name else EXTENSIONS):
        print(extension_path(n))


def cmd_check(args):
    db = connect(":memory:")
    db.execute("CREATE VIRTUAL TABLE d USING dense_ann(dim=384)")
    db.execute("CREATE VIRTUAL TABLE l USING late_plaid(dim=48, nbits=2, centroids=256)")
    db.execute("CREATE VIRTUAL TABLE f USING fts5(body)")
    version = db.execute("SELECT sqlite_version()").fetchone()[0]
    print(f"ok: dense_ann, late_plaid and fts5 available (SQLite {version}, {type(db).__module__})")
    for n in EXTENSIONS:
        print(f"  {n}: {extension_path(n)}")


def cmd_build_db(args):
    bd = _build_db()
    keys = [k for k in args.indexes.split(",") if k] if args.indexes else []
    for flag, key in (("fts", "fts"), ("dense", "dense_graph"), ("dense_ivf", "dense_ivf"), ("late", "late")):
        if getattr(args, flag) and key not in keys:
            keys.append(key)
    keys = keys or list(INDEXES)
    unknown = [k for k in keys if k not in INDEXES]
    if unknown:
        sys.exit(f"unknown index {unknown}; known: {', '.join(INDEXES)}")
    docs = read_corpus(args.corpus)
    n = len(docs)
    print(f"{args.corpus}: {n} documents; indexes {', '.join(keys)}", flush=True)
    need = {INDEXES[k].get("embedding") for k in keys} - {None}
    out = Path(args.out)
    if out.exists():
        if not args.force:
            sys.exit(f"{out} exists (use --force)")
        out.unlink()
    out.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=args.tmp_dir) as tmp:
        X = vec = off = None
        mdir = None
        if need:
            try:
                import numpy as np
            except ImportError:
                sys.exit("vector indexes need numpy: pip install 'dense-late-ann[encoders]' (or [numpy] "
                         "with precomputed vectors)")
        if "minilm" in need:
            if args.minilm_npy:
                X = np.load(args.minilm_npy, mmap_mode="r")
            else:
                mdir = mdir or models_dir(args.models)
                t = time.time()
                enc = encoder("minilm", mdir, args.threads)
                X = np.stack([enc.encode([d], batch_size=1)[0] for d in docs]) if args.batch_size == 1 \
                    else enc.encode(docs, batch_size=args.batch_size)
                print(f"minilm: encoded {n} documents in {time.time() - t:.1f} s", flush=True)
            if len(X) < n:
                sys.exit(f"{len(X)} MiniLM vectors for {n} documents")
            X = X[:n]
        if "lateon" in need:
            if args.lateon_npy:
                if not args.lateon_offsets:
                    sys.exit("--lateon-npy needs --lateon-offsets")
                vec, off = args.lateon_npy, args.lateon_offsets
            else:
                mdir = mdir or models_dir(args.models)
                t = time.time()
                enc = encoder("lateon", mdir, args.threads)
                embs = enc.encode_documents(docs, batch_size=args.batch_size)
                offs = np.zeros(n + 1, np.int64)
                offs[1:] = np.cumsum([len(e) for e in embs])
                vec, off = os.path.join(tmp, "lateon.vectors.npy"), os.path.join(tmp, "lateon.offsets.npy")
                np.save(vec, np.concatenate(embs).astype(np.float16))
                np.save(off, offs)
                print(f"lateon: encoded {n} documents ({offs[-1]} token vectors) in {time.time() - t:.1f} s",
                      flush=True)

        db = connect(out, isolation_level=None)   # build_db issues BEGIN/COMMIT itself
        db.execute(f"PRAGMA page_size={PAGE_SIZE}")
        db.execute("PRAGMA journal_mode=DELETE")
        bd.build_docs(db, docs)
        extra = {"dense_graph": args.dense_params, "dense_ivf": args.dense_ivf_params, "late": args.late_params}
        for key in keys:
            spec = INDEXES[key]
            t = time.time()
            params = params_with(spec, extra.get(key), args.threads) if spec["kind"] != "fts5" else ""
            print(f"building {key} ({spec['kind']}{': ' + params if params else ''})", flush=True)
            if spec["kind"] == "fts5":
                bd.build_fts(db, spec)
            elif spec["kind"] == "dense_ann":
                bd.build_dense(db, spec, params, X)
            else:
                bd.build_late(db, spec, params, vec, off)
            print(f"  {key}: {time.time() - t:.1f} s", flush=True)
        db.execute("VACUUM")
        for key in keys:
            bd.finalize(db, INDEXES[key])
        try:
            _, per = bd.table_sizes(db, {k: INDEXES[k] for k in keys})
            sizes = ", ".join(f"{k} {v['bytes'] / 2**20:.2f} MiB" for k, v in sorted(per.items(), key=lambda kv: -kv[1]["bytes"]))
        except Exception:  # noqa: BLE001  (dbstat is not compiled into every SQLite)
            sizes = ""
        db.close()
    print(f"{out}: {out.stat().st_size / 2**20:.2f} MiB" + (f" ({sizes})" if sizes else ""))


def cmd_encode_query(args):
    enc = encoder(args.model, models_dir(args.models), args.threads)
    v = enc.encode([args.text], batch_size=1)[0] if args.model == "minilm" else enc.encode_queries([args.text])[0]
    data = v.astype("<f4").tobytes()
    if args.out == "-":
        sys.stdout.buffer.write(data)
    else:
        Path(args.out).write_bytes(data)
        print(f"{args.out}: {v.shape} float32", file=sys.stderr)


def discover(db):
    rows = db.execute("SELECT name, sql FROM sqlite_schema WHERE type = 'table' "
                      "AND sql LIKE 'CREATE VIRTUAL TABLE%'").fetchall()
    out = []
    for name, sql in rows:
        low = sql.lower()
        if "using fts5" in low:
            out.append((name, "fts", None))
        elif "using dense_ann" in low:
            out.append((name, "dense", "ivf" if "layout=ivf" in low.replace(" ", "") else "graph"))
        elif "using late_plaid" in low:
            out.append((name, "late", None))
    return out


def cmd_search(args):
    import re
    db = connect(args.db)
    indexes = discover(db)
    pick = [i for i in indexes if i[0] == args.system] or [i for i in indexes if i[1] == args.system]
    if args.system == "dense":
        pick.sort(key=lambda i: i[2] != "graph")
    if not pick:
        sys.exit(f"{args.db} has no {args.system} index (have: {', '.join(i[0] for i in indexes)})")
    table, kind, layout = pick[0]
    k = int(args.k)
    t = f'"{table}"'
    extra = dict(kv.split("=", 1) for kv in (args.param or []))
    if kind == "fts":
        words = list(dict.fromkeys(re.findall(r"\w+", args.text.lower())))
        sql = f"SELECT rowid, rank FROM {t} WHERE {t} MATCH ? ORDER BY rank LIMIT {k}"
        arg = [" OR ".join(f'"{w}"' for w in words)]
    else:
        which = "minilm" if kind == "dense" else "lateon"
        enc = encoder(which, models_dir(args.models), args.threads)
        v = enc.encode([args.text], batch_size=1)[0] if which == "minilm" else enc.encode_queries([args.text])[0]
        p = {**QUERY_DEFAULTS["dense:" + layout if kind == "dense" else "late"], **extra}
        if kind == "dense":
            allowed = ("ef", "beam", "rerank", "exact") if layout == "graph" else ("nprobe", "rerank_k", "rerank", "exact")
            where = "".join(f" AND {name} = {int(p[name])}" for name in allowed if name in p)
            sql = f"SELECT rowid, distance FROM {t} WHERE embedding MATCH ? AND k = {k}{where}"
            arg = [v.astype("<f4").tobytes()]
        else:
            opts = " ".join(f"{name}={p[name]}" for name in p if name != "nprobe")
            sql = f"SELECT rowid, score FROM {t} WHERE {t} MATCH ? AND k = {k} AND nprobe = {int(p['nprobe'])} AND opts = ?"
            arg = [v.astype("<f4").tobytes(), opts]
    t0 = time.time()
    rows = db.execute(sql, arg).fetchall()
    ms = (time.time() - t0) * 1000
    texts = {}
    if rows and any(r[0] == "docs" for r in db.execute("SELECT name FROM sqlite_schema WHERE type='table'")):
        ids = ",".join(str(int(r[0])) for r in rows)
        texts = dict(db.execute(f"SELECT id, body FROM docs WHERE id IN ({ids})").fetchall())
    if args.json:
        print(json.dumps({"table": table, "sql": sql, "ms": ms,
                          "rows": [{"id": r[0], "score": r[1], "text": texts.get(r[0])} for r in rows]}))
        return
    print(f"# {table} ({kind}{'/' + layout if layout else ''}), {len(rows)} rows in {ms:.1f} ms")
    for r in rows:
        print(f"{r[0]}\t{r[1]:.4f}\t{(texts.get(r[0]) or '')[:100]}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="dense-late-ann", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("path", help="print the path of the loadable extensions")
    p.add_argument("name", nargs="?", choices=sorted(EXTENSIONS))
    p.set_defaults(func=cmd_path)

    p = sub.add_parser("check", help="load both extensions into an in-memory database")
    p.set_defaults(func=cmd_check)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--models", help="directory with minilm-l6-v2/ and lateon-code-edge/ "
                                         "(default: $DENSE_LATE_ANN_MODELS, ./models)")
    common.add_argument("--threads", type=int, default=2, help="encoder and index-build threads (default 2)")

    p = sub.add_parser("build-db", parents=[common], help="build a deployable database from a corpus",
                       description="Build a database with a docs(id, body) table and the chosen indexes, "
                                   "as tools/build_db.py does. Without index flags, all four are built.")
    p.add_argument("corpus", help="one document per line (.txt) or JSON lines with a \"text\" field (.jsonl)")
    p.add_argument("--out", required=True, help="output database")
    p.add_argument("--force", action="store_true", help="overwrite --out")
    p.add_argument("--fts", action="store_true", help="FTS5 index 'fts'")
    p.add_argument("--dense", action="store_true", help="dense_ann graph index 'dense_graph' (MiniLM)")
    p.add_argument("--dense-ivf", action="store_true", help="dense_ann IVF-PQ index 'dense_ivf' (MiniLM)")
    p.add_argument("--late", action="store_true", help="late_plaid index 'late' (LateOn-Code-edge)")
    p.add_argument("--indexes", help="comma-separated: " + ",".join(INDEXES))
    p.add_argument("--dense-params", help="extra dense_graph parameters, e.g. 'M=32, ef_construction=400'")
    p.add_argument("--dense-ivf-params", help="extra dense_ivf parameters, e.g. 'nlist=1024'")
    p.add_argument("--late-params", help="extra late_plaid parameters, e.g. 'nbits=4 centroids=4096'")
    p.add_argument("--minilm-npy", help="precomputed MiniLM vectors [n, 384] (float16 or float32 .npy)")
    p.add_argument("--lateon-npy", help="precomputed LateOn token vectors [T, 48] float16 .npy")
    p.add_argument("--lateon-offsets", help="int64 .npy [n + 1]: document i owns vectors[off[i]:off[i+1]]")
    p.add_argument("--batch-size", type=int, default=1,
                   help="documents per encoder call (default 1: reproducible, equal to in-browser encoding)")
    p.add_argument("--tmp-dir", help="directory for temporary files")
    p.set_defaults(func=cmd_build_db)

    p = sub.add_parser("encode-query", parents=[common], help="encode a query to raw little-endian float32")
    p.add_argument("text")
    p.add_argument("--model", choices=sorted(MODEL_FILES), required=True)
    p.add_argument("--out", default="-", help="output file (default stdout)")
    p.set_defaults(func=cmd_encode_query)

    p = sub.add_parser("search", parents=[common], help="query a database natively")
    p.add_argument("db")
    p.add_argument("text")
    p.add_argument("--system", default="fts", help="fts | dense | late | a table name (default fts)")
    p.add_argument("-k", default=10, type=int)
    p.add_argument("--param", action="append", metavar="NAME=VALUE", help="query parameter (repeatable)")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_search)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
