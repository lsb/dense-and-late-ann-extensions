#!/usr/bin/env python3
"""Build the deployable SQLite database of a corpus for the benchmark matrix.

One database per corpus holds everything a browser client needs:

  docs(id INTEGER PRIMARY KEY, body TEXT)   the documents, id = line number (0-based)
  fts                                       FTS5 external-content index over docs.body
  dense_graph                               dense_ann, graph layout (MiniLM)
  dense_ivf                                 dense_ann, IVF-PQ layout (MiniLM)
  late                                      late_plaid, warp + plaid layouts (LateOn)

The set of indexes and their parameters come from bench/matrix_config.json
("indexes"), so a new layout is added there, not here. Build order: documents,
FTS5 ('rebuild', 'optimize'), each vector index ('build'), VACUUM, then every
extension's 'finalize' (page hints must be computed after the last VACUUM).
A manifest <db>.json records per-table and per-index sizes (dbstat), build
times, parameters, and SHA-256 digests of the extension sources.

With --split, each index is also built into its own database
(<corpus>--<index>.db; the FTS one also holds docs, which it needs), so its
size and fetch counts can be attributed without any shared pages.

Index construction uses threaded builds of the extension sources compiled into
build/matrix/bin (the `make native` .so files are single-threaded); the on-disk
format is the same, and queries use `make native` / `make wasm` builds.

  python3 tools/build_db.py words-10k                 # build/matrix/words-10k.db
  python3 tools/build_db.py words-100 --split
  python3 tools/build_db.py words-1m --split-only --indexes late --no-vacuum
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
NATIVE = REPO / "build" / "native"
OUT = REPO / "build" / "matrix"
BIN = OUT / "bin"
CONFIG = REPO / "bench" / "matrix_config.json"


def reexec_with_native_sqlite():
    """Python's sqlite3 links libsqlite3.so.0 dynamically; point it at 3.53.4."""
    import sqlite3
    if sqlite3.sqlite_version == "3.53.4":
        return
    if os.environ.get("_BUILD_DB_REEXEC"):
        sys.exit(f"sqlite3 is {sqlite3.sqlite_version} even with LD_LIBRARY_PATH={NATIVE}; run `make native`")
    env = dict(os.environ, LD_LIBRARY_PATH=f"{NATIVE}:{os.environ.get('LD_LIBRARY_PATH', '')}",
               _BUILD_DB_REEXEC="1")
    os.execve(sys.executable, [sys.executable] + sys.argv, env)


def load_config():
    return json.loads(CONFIG.read_text())


# ------------------------------------------------------------ builders

EXT_SOURCES = {
    "dense_ann": (["ext/dense/dense_ann.c", "ext/dense/hnsw.c", "ext/dense/ivf.c", "ext/dense/pq.c",
                   "ext/dense/rawpage.c"], "-DDENSE_ANN_THREADS", "denseann", "sqlite3_denseann_init"),
    "late_plaid": (["ext/late/late_codec.c", "ext/late/late_page.c", "ext/late/late_plaid.c"],
                   "-DLATE_THREADS", "late", "sqlite3_late_init"),
}


def ext_digest(kind):
    srcs, *_ = EXT_SOURCES[kind]
    d = Path(srcs[0]).parent
    h = hashlib.sha256()
    for p in sorted((REPO / d).glob("*.[ch]")):
        h.update(p.name.encode()); h.update(p.read_bytes())
    return h.hexdigest()


def builder_so(kind):
    """Compile a threaded loadable build of an extension (cached by source digest)."""
    srcs, flag, name, entry = EXT_SOURCES[kind]
    dig = ext_digest(kind)[:16]
    so = BIN / dig / f"{name}.so"   # file name gives SQLite's default entry point
    if not so.exists():
        so.parent.mkdir(parents=True, exist_ok=True)
        cmd = ["cc", "-O2", "-std=c11", "-D_GNU_SOURCE", "-fPIC", "-shared", "-pthread", flag,
               f"-I{REPO / 'build' / 'sqlite'}", f"-I{REPO / 'wasm' / 'src'}",
               *[str(REPO / s) for s in srcs], "-o", str(so) + ".tmp", "-lm"]
        subprocess.run(cmd, check=True)
        os.replace(str(so) + ".tmp", so)
    return str(so), entry


def connect(path):
    import sqlite3
    db = sqlite3.connect(str(path), isolation_level=None)
    db.enable_load_extension(True)
    for kind in EXT_SOURCES:
        so, entry = builder_so(kind)
        db.load_extension(so)
    return db


# ------------------------------------------------------------ inputs

def corpus_docs(cfg, corpus):
    path = REPO / cfg["corpora"][corpus]["text"]
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f if line.strip()]


def emb_paths(cfg, corpus, model):
    c = cfg["corpora"][corpus]
    base = REPO / "data" / "emb" / c["emb"]
    if model == "minilm":
        return {"vectors": Path(f"{base}.minilm.npy")}
    return {"vectors": Path(f"{base}.lateon.vectors.npy"), "offsets": Path(f"{base}.lateon.offsets.npy")}


def emb_meta(cfg, corpus):
    """The encoder metadata files (data/emb/NAME.<model>.json) the indexes were built from."""
    out = {}
    for m in ("minilm", "lateon"):
        p = REPO / "data" / "emb" / f"{cfg['corpora'][corpus]['emb']}.{m}.json"
        if p.exists():
            out[m] = json.loads(p.read_text())
    return out


def load_minilm(cfg, corpus, n):
    import numpy as np
    X = np.load(emb_paths(cfg, corpus, "minilm")["vectors"], mmap_mode="r")
    if len(X) < n:
        raise SystemExit(f"{corpus}: {len(X)} MiniLM vectors for {n} documents")
    return X[:n]


def lateon_npy(cfg, corpus, n, tmpdir):
    """Paths of the token-vector and offset .npy files for the first n documents
    (a prefix of a larger corpus's encoding is written to tmpdir)."""
    import numpy as np
    p = emb_paths(cfg, corpus, "lateon")
    off = np.load(p["offsets"])
    if len(off) - 1 < n:
        raise SystemExit(f"{corpus}: {len(off) - 1} LateOn documents for {n} documents")
    if len(off) - 1 == n:
        return str(p["vectors"]), str(p["offsets"])
    V = np.load(p["vectors"], mmap_mode="r")
    vp, op = Path(tmpdir) / "vectors.npy", Path(tmpdir) / "offsets.npy"
    np.save(vp, np.ascontiguousarray(V[:off[n]]))
    np.save(op, off[:n + 1])
    return str(vp), str(op)


# ------------------------------------------------------------ building

def index_params(spec, corpus):
    p = spec.get("params", "")
    extra = spec.get("per_corpus", {}).get(corpus)
    if extra:
        sep = ", " if spec["kind"] == "dense_ann" else " "
        keys = {kv.split("=")[0].strip() for kv in extra.replace(",", " ").split()}
        parts = [kv for kv in p.replace(",", " ").split() if kv.split("=")[0].strip() not in keys]
        p = sep.join(parts + extra.replace(",", " ").split())
    return p


def build_docs(db, docs):
    db.execute("CREATE TABLE docs(id INTEGER PRIMARY KEY, body TEXT)")
    db.execute("BEGIN")
    db.executemany("INSERT INTO docs(id, body) VALUES (?, ?)", enumerate(docs))
    db.execute("COMMIT")


def build_fts(db, spec):
    t = spec["table"]
    db.execute(f"CREATE VIRTUAL TABLE {t} USING fts5(body, content='docs', content_rowid='id')")
    db.execute(f"INSERT INTO {t}({t}) VALUES ('rebuild')")
    db.execute(f"INSERT INTO {t}({t}) VALUES ('optimize')")


def build_dense(db, spec, params, X):
    import numpy as np
    t = spec["table"]
    db.execute(f"CREATE VIRTUAL TABLE {t} USING dense_ann({params})")
    db.execute("BEGIN")
    step = 10000
    for s in range(0, len(X), step):
        blk = np.asarray(X[s:s + step], dtype=np.float32)
        db.executemany(f"INSERT INTO {t}(rowid, embedding) VALUES (?, ?)",
                       ((s + i, blk[i].tobytes()) for i in range(len(blk))))
    db.execute("COMMIT")
    db.execute(f"INSERT INTO {t}({t}) VALUES ('build')")


def build_late(db, spec, params, vec, off):
    t = spec["table"]
    db.execute(f"CREATE VIRTUAL TABLE {t} USING late_plaid({params})")
    db.execute(f"INSERT INTO {t}({t}) VALUES ('build_npy {vec} {off}')")


def finalize(db, spec):
    t = spec["table"]
    if spec["kind"] in ("dense_ann", "late_plaid"):
        db.execute(f"INSERT INTO {t}({t}) VALUES ('finalize')")


def late_header(db, table):
    """Parameters actually stored in a late_plaid index (header of <table>_meta;
    layout of ext/late/pyref.py)."""
    import struct
    blob = db.execute(f"SELECT data FROM {table}_meta ORDER BY id LIMIT 1").fetchone()[0]
    magic, ver = struct.unpack_from("<II", blob, 0)
    dim, nbits, K, layout, kbits, idbits, chunk, rid = struct.unpack_from("<8I", blob, 8)
    N, T, ivf = struct.unpack_from("<3Q", blob, 40)
    p = 64 + 132 + 8
    G = cq = 0
    if ver >= 2:
        G, _ = struct.unpack_from("<II", blob, p)
        p += 8
    if ver >= 3:
        (cq,) = struct.unpack_from("<I", blob, p)
    return {"format": ver, "dim": dim, "nbits": nbits, "centroids": K, "coarse": G,
            "layout": {1: "plaid", 2: "warp", 3: "both"}.get(layout, layout),
            "centroid_type": {0: "f16", 1: "int8", 2: "int4"}.get(cq, cq), "n_docs": N, "n_tokens": T}


def check_late(db, table, params):
    """Fail if the stored index does not have the requested parameters (a parser
    bug once silently ignored all but the first space-separated option)."""
    h = late_header(db, table)
    want = dict(kv.split("=", 1) for kv in params.replace(",", " ").split() if "=" in kv)
    bad = []
    for k in ("dim", "nbits", "layout", "centroid_type"):
        if k in want and str(h[k]) != want[k]:
            bad.append(f"{k}: asked {want[k]}, got {h[k]}")
    if want.get("centroids", "0") != "0" and str(h["centroids"]) != want["centroids"]:
        bad.append(f"centroids: asked {want['centroids']}, got {h['centroids']}")
    if bad:
        raise SystemExit(f"late index {table} was not built as requested: " + "; ".join(bad))
    return h


def owner(name, indexes):
    """Which index (or 'docs' / 'schema') a b-tree belongs to."""
    base = name
    if base.startswith("sqlite_autoindex_"):
        base = base[len("sqlite_autoindex_"):].rsplit("_", 1)[0]
    if base == "sqlite_schema":
        return "schema"
    if base == "docs":
        return "docs"
    for key, spec in indexes.items():
        t = spec["table"]
        if base == t or base.startswith(t + "_"):
            return key
    return "other"


def table_sizes(db, indexes):
    rows = db.execute("SELECT name, count(*), sum(pgsize), sum(payload) FROM dbstat GROUP BY name").fetchall()
    tables = {n: {"pages": c, "bytes": b, "payload": p, "owner": owner(n, indexes)} for n, c, b, p in rows}
    per = {}
    for n, v in tables.items():
        o = per.setdefault(v["owner"], {"bytes": 0, "pages": 0, "tables": []})
        o["bytes"] += v["bytes"]; o["pages"] += v["pages"]; o["tables"].append(n)
    return tables, per


def build(cfg, corpus, keys, path, with_docs=True, vacuum=True):
    indexes = {k: cfg["indexes"][k] for k in keys}
    docs = corpus_docs(cfg, corpus)
    n = len(docs)
    if path.exists():
        path.unlink()
    for suf in ("-journal", "-wal"):
        Path(str(path) + suf).unlink(missing_ok=True)
    db = connect(path)
    db.execute(f"PRAGMA page_size={cfg['page_size']}")
    db.execute("PRAGMA journal_mode=DELETE")
    times, params_used = {}, {}
    t0 = time.time()
    if with_docs:
        build_docs(db, docs)
        times["docs"] = time.time() - t0
    with tempfile.TemporaryDirectory(dir=OUT) as tmp:
        for key, spec in indexes.items():
            t = time.time()
            params = index_params(spec, corpus)
            params_used[key] = params
            print(f"[{corpus}] building {key} ({spec['kind']}: {params})", flush=True)
            if spec["kind"] == "fts5":
                if not with_docs:
                    build_docs(db, docs)
                build_fts(db, spec)
            elif spec["kind"] == "dense_ann":
                build_dense(db, spec, params, load_minilm(cfg, corpus, n))
            elif spec["kind"] == "late_plaid":
                build_late(db, spec, params, *lateon_npy(cfg, corpus, n, tmp))
            else:
                raise SystemExit(f"unknown index kind {spec['kind']}")
            times[key] = time.time() - t
            print(f"[{corpus}] {key}: {times[key]:.1f} s", flush=True)
    t = time.time()
    if vacuum:
        db.execute("VACUUM")
        times["vacuum"] = time.time() - t
    t = time.time()
    for spec in indexes.values():
        finalize(db, spec)
    times["finalize"] = time.time() - t
    free = db.execute("PRAGMA freelist_count").fetchone()[0]
    tables, per = table_sizes(db, indexes)
    ext_cfg = {}
    for key, spec in indexes.items():
        if spec["kind"] == "late_plaid":
            ext_cfg[key] = check_late(db, spec["table"], params_used[key])
    for key, spec in indexes.items():
        if spec["kind"] in ("dense_ann", "late_plaid"):
            try:
                cfgtab = f"{spec['table']}_config" if spec["kind"] == "dense_ann" else None
                if cfgtab:
                    ext_cfg[key] = {k: v for k, v in db.execute(f"SELECT key, value FROM {cfgtab}")
                                    if not isinstance(v, bytes)}
            except Exception as e:  # noqa: BLE001
                ext_cfg[key] = {"error": str(e)}
    page_size = db.execute("PRAGMA page_size").fetchone()[0]
    jm = db.execute("PRAGMA journal_mode").fetchone()[0]
    db.close()
    manifest = {
        "corpus": corpus, "file": path.name, "n_docs": n, "bytes": path.stat().st_size,
        "page_size": page_size, "journal_mode": jm, "freelist_pages": free,
        "indexes": {k: {**indexes[k], "params_used": params_used.get(k)} for k in indexes},
        "has_docs": with_docs or any(s["kind"] == "fts5" for s in indexes.values()),
        "sizes": per, "tables": tables, "build_seconds": times, "ext_config": ext_cfg,
        "ext_sources_sha256": {k: ext_digest(k) for k in EXT_SOURCES},
        "git_head": subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True,
                                   text=True).stdout.strip(),
        "vacuumed": vacuum,
        "sqlite_version": "3.53.4",
        "embeddings": emb_meta(cfg, corpus),
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    Path(str(path) + ".json").write_text(json.dumps(manifest, indent=1))
    print(f"[{corpus}] {path} {path.stat().st_size / 2**20:.2f} MiB: "
          + ", ".join(f"{k} {v['bytes'] / 2**20:.2f}" for k, v in sorted(per.items(), key=lambda kv: -kv[1]['bytes'])),
          flush=True)
    return manifest


def main():
    reexec_with_native_sqlite()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("corpus")
    ap.add_argument("--indexes", default="", help="comma-separated keys of matrix_config.json 'indexes' (default: all)")
    ap.add_argument("--split", action="store_true", help="also build one database per index")
    ap.add_argument("--split-only", action="store_true", help="only the per-index databases")
    ap.add_argument("--out-dir", default=str(OUT))
    ap.add_argument("--no-vacuum", action="store_true",
                    help="skip VACUUM (saves the temporary copy's disk space at 1M; the pages freed by the "
                         "dense build buffer stay on the free list: they take disk space but are never fetched)")
    args = ap.parse_args()
    cfg = load_config()
    if args.corpus not in cfg["corpora"]:
        raise SystemExit(f"unknown corpus {args.corpus}; known: {', '.join(cfg['corpora'])}")
    keys = [k for k in args.indexes.split(",") if k] or list(cfg["indexes"])
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not args.split_only:
        build(cfg, args.corpus, keys, out / f"{args.corpus}.db", vacuum=not args.no_vacuum)
    if args.split or args.split_only:
        for k in keys:
            build(cfg, args.corpus, [k], out / f"{args.corpus}--{k}.db", with_docs=False,
                  vacuum=not args.no_vacuum)


if __name__ == "__main__":
    main()
