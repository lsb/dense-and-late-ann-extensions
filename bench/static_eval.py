#!/usr/bin/env python3
"""Per-connection static data of the vector indexes: size, cold-query cost and
quality of storage variants (centroid / codebook precision, number of
centroids, entry-set size, IVF centroid storage).

Each variant is one index of bench/matrix_config.json built alone into
build/static/<corpus>--<name>.db with different parameters. For it this script
measures

  * static data: what a new connection loads before its first query, as the
    extension reports it after `MATCH 'warm'` (payload bytes, pages, rounds);
  * quality: the matrix configurations of that index (bench/matrix_config.json)
    run natively on every query of the corpus; nDCG@10 against the relevance
    labels and recall@10 against the model's exact search (as bench/matrix.py);
  * cold cost: `--cold N` queries through the WASM build (bench/matrix.mjs,
    Asyncify) against the unshaped range server, each on a new connection,
    replayed through netsim under 4g / lte with h1 and h2 (p50 ms), plus the
    rounds and KB each cold query transferred.

  python3 bench/static_eval.py build llm-10k late int8 "dim=48 nbits=2 centroids=0 layout=both order=1 threads=2 input=f16 centroid_type=int8"
  python3 bench/static_eval.py measure llm-10k int8 --cold 40
  python3 bench/static_eval.py table llm-10k words-10k        # markdown summary of results/static/*.json

A version-2 (pre-format-3) copy of a late index can be made with
`downgrade CORPUS NAME NEWNAME` (float16 centroids only), to measure the
static data of the previous format with the current code.
"""
import argparse
import json
import os
import re
import shutil
import struct
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "bench"))
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "ext" / "late"))

import numpy as np  # noqa: E402

import build_db  # noqa: E402
import matrix  # noqa: E402

OUT = REPO / "build" / "static"
RES = REPO / "results" / "static"
PROFILES = ["4g,h1", "4g,h2", "lte,h1", "lte,h2"]


def db_path(corpus, name):
    return OUT / f"{corpus}--{name}.db"


def manifest_of(corpus, name):
    return json.loads(Path(str(db_path(corpus, name)) + ".json").read_text())


# ------------------------------------------------------------------ build

def cmd_build(a):
    build_db.reexec_with_native_sqlite()
    cfg = build_db.load_config()
    spec = dict(cfg["indexes"][a.index])
    spec["params"] = a.params
    spec.pop("per_corpus", None)
    cfg["indexes"] = {a.index: spec}
    OUT.mkdir(parents=True, exist_ok=True)
    build_db.OUT = OUT
    man = build_db.build(cfg, a.corpus, [a.index], db_path(a.corpus, a.name), with_docs=False)
    man["variant"] = {"index": a.index, "name": a.name, "params": a.params}
    Path(str(db_path(a.corpus, a.name)) + ".json").write_text(json.dumps(man, indent=1))


def cmd_downgrade(a):
    """Rewrite a late index's static data in format 2 (u32 list lengths)."""
    import sqlite3
    import pyref
    src, dst = db_path(a.corpus, a.name), db_path(a.corpus, a.newname)
    shutil.copy(src, dst)
    man = manifest_of(a.corpus, a.name)
    man["file"] = dst.name
    man["variant"] = {**man["variant"], "name": a.newname, "note": f"{a.name} with static data rewritten in format 2"}
    t = man["variant"].get("table", "late")
    db = sqlite3.connect(dst)
    blob = pyref._stream(db, f"{t}_meta")
    ref = pyref.Index(db, t)
    assert ref.cq == 0 and ref.G == 0, "format 2 has float16 flat centroids only"
    head = bytearray(blob[:212])
    struct.pack_into("<I", head, 4, 2)
    p = 216
    K, D = ref.K, ref.dim
    body, p = blob[p:p + 2 * K * D], p + 2 * K * D
    parts = []
    for flag in (1, 2):
        if ref.layout & flag:
            v, p = pyref._varints(blob, p, K)
            parts.append(v.astype("<u4").tobytes())
    v2 = bytes(head) + body + b"".join(parts) + blob[p:]
    db.execute(f"DELETE FROM {t}_meta")
    for i in range(0, len(v2), ref.chunk):
        db.execute(f"INSERT INTO {t}_meta(id, data) VALUES (?, ?)", (i // ref.chunk + 1, v2[i:i + ref.chunk]))
    db.commit()
    db.execute("VACUUM")
    db.close()
    # page hints of the streams moved with VACUUM: finalize would rewrite format 3,
    # so re-run it on a scratch copy only to check nothing else is needed
    Path(str(dst) + ".json").write_text(json.dumps(man, indent=1))
    print(f"{dst}: static data {len(blob)} -> {len(v2)} bytes (format 2)")


# ------------------------------------------------------------------ eval

def exact_refs(cfg, corpus, meta, qs, inp, n_docs):
    cache = OUT / f"{corpus}.exact-{inp}.json"
    if cache.exists():
        return {int(k): v for k, v in json.loads(cache.read_text()).items()}
    matrix.manifest = lambda c: {"n_docs": n_docs}
    t = time.time()
    g = matrix.exact_dense(cfg, corpus, meta, qs, 10) if inp == "minilm" else matrix.exact_late(cfg, corpus, meta, qs, 10)
    cache.write_text(json.dumps(g))
    print(f"[{corpus}] exact {inp}: {time.time() - t:.0f} s", flush=True)
    return g


def static_info(path, spec):
    """Load the static data on a fresh native connection with MATCH 'warm'."""
    import sqlite3
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    db.enable_load_extension(True)
    for so in sorted(matrix.NATIVE_EXT.glob("*.so")):
        db.load_extension(str(so))
    t = spec["table"]
    col = "embedding" if spec["kind"] == "dense_ann" else t
    cur = db.execute(f"SELECT stats FROM {t} WHERE {col} MATCH 'warm'")
    cur.fetchall()
    out = {}
    if spec["kind"] == "late_plaid":
        # the warm statement returns no row: read the stats of a one-token query instead
        q = np.zeros((1, 48), np.float32); q[0, 0] = 1
        st = json.loads(db.execute(f"SELECT stats FROM {t} WHERE {t} MATCH ? AND k = 1", (q.tobytes(),)).fetchone()[0])
        out["static_payload"] = st["static_bytes"]
        n = db.execute(f"SELECT count(*), sum(length(data)) FROM {t}_meta").fetchone()
        out["static_pages"] = n[0]
    else:
        n = db.execute(f"SELECT count(*), sum(length(data)) FROM {t}_blobs").fetchone()
        out["static_payload"] = n[1]
        out["static_pages"] = n[0]
    db.close()
    return out


def cmd_eval(a):
    cfg = matrix.load_config()
    man = manifest_of(a.corpus, a.name)
    key = man["variant"]["index"]
    spec = man["indexes"][key]
    meta = matrix.qmeta(a.corpus)
    n_docs = man["n_docs"]
    path = db_path(a.corpus, a.name)
    man["corpus"] = a.corpus
    confs = [c for c in matrix.configs_for(cfg, man) if not c.get("exhaustive")]
    if a.configs:
        confs = [c for c in confs if c["id"] in a.configs.split(",")]
    if a.opts:   # extra late_plaid query options, e.g. cprobe=16
        confs = [{**c, "id": f"{c['id']}+{a.opts.replace(' ', '+')}",
                  "sql_resolved": c["sql_resolved"].replace("opts = '", f"opts = '{a.opts} ")} for c in confs]
    qs = matrix.all_queries(cfg, meta)
    inputs = matrix.QueryInputs(meta)
    db = matrix.native_db(path)
    out = {"corpus": a.corpus, "name": a.name, "index": key, "params": man["variant"]["params"],
           "index_bytes": man["sizes"].get(key, {}).get("bytes"), "n_queries": len(qs),
           "static": static_info(path, spec), "configs": {}}
    for c in confs:
        t = time.time()
        ref = exact_refs(cfg, a.corpus, meta, qs, c["input"], n_docs)
        rows = []
        for qi in qs:
            ids = [r[0] for r in db.execute(c["sql_resolved"], {"q": inputs.param(c["input"], qi), "k": 10})]
            m = matrix.per_query_metrics(ids, None, meta["relevant"][qi], n_docs, ref.get(qi))
            m["kind"] = meta["kinds"][qi]
            rows.append(m)
        summ = matrix.summarize_metrics(rows)
        out["configs"][c["id"]] = {"label": c["label"], "ndcg10": summ["all"]["ndcg@10"],
                                   "r10_exact": summ["all"].get("r10_vs_exact"),
                                   "by_kind": {k: v["ndcg@10"] for k, v in summ.items() if k not in ("all", "n")}}
        print(f"[{a.corpus}/{a.name}] {c['id']}: nDCG@10 {summ['all']['ndcg@10']:.4f} "
              f"R@10 vs exact {summ['all'].get('r10_vs_exact') or 0:.4f} ({time.time() - t:.0f} s)", flush=True)
    if a.cold:
        cold = cold_costs(cfg, man, meta, confs, qs[:a.cold], path)
        for cid, v in cold.items():
            out["configs"][cid]["cold"] = v
    RES.mkdir(parents=True, exist_ok=True)
    tag = f"--{a.opts.replace(' ', '_').replace('=', '')}" if a.opts else ""
    (RES / f"{a.corpus}--{a.name}{tag}.json").write_text(json.dumps(out, indent=1))


def cold_costs(cfg, man, meta, confs, qidx, path):
    from concurrent.futures import ProcessPoolExecutor
    matrix.BUILD = OUT
    srv = matrix.Server("none")
    tr = OUT / f"{path.stem}.cold.jsonl"
    try:
        runs = [matrix.node_run(c, 10, "cold", qidx, []) for c in confs]
        matrix.run_node({"url": f"{srv.base}/{path.name}", "queries": str(matrix.QDIR / f"{man['corpus']}.json"),
                         "phases": [{"runs": runs}]}, tr)
    finally:
        srv.close()
    recs = [json.loads(line) for line in open(tr)]
    tr.unlink()
    with ProcessPoolExecutor(2) as ex:
        sims = list(ex.map(matrix.sim_record, [(r, PROFILES, r["qi"]) for r in recs], chunksize=16))
    out = {}
    for c in confs:
        rs = [(r, s) for r, s in zip(recs, sims) if r["config"] == c["id"]]
        errs = [r.get("error") for r, _ in rs if r.get("error")]
        if errs:
            raise SystemExit(f"{c['id']}: {errs[0]}")
        out[c["id"]] = {
            "n": len(rs),
            "rounds": float(np.median([r["stats"]["rounds"] for r, _ in rs])),
            "kb": float(np.median([r["stats"]["bytes"] for r, _ in rs])) / 1024,
            "requests": float(np.median([r["stats"]["requests"] for r, _ in rs])),
            "ms": {p: float(np.median([s[i] for _, s in rs])) for i, p in enumerate(PROFILES)},
        }
        v = out[c["id"]]
        print(f"  cold {c['id']}: rounds {v['rounds']:.0f}, {v['kb']:.0f} KB, "
              + ", ".join(f"{p} {v['ms'][p]:.0f} ms" for p in PROFILES), flush=True)
    return out


# ------------------------------------------------------------------ table

def cmd_table(a):
    rows = []
    for corpus in a.corpora:
        for p in sorted(RES.glob(f"{corpus}--*.json")):
            rows.append(json.loads(p.read_text()))
    print("| Corpus | Index | Variant | Static KB | Config | nDCG@10 | R@10 vs exact | Cold rounds | Cold KB | "
          "4g h1 | 4g h2 | lte h1 | lte h2 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        for cid, c in r["configs"].items():
            cold = c.get("cold") or {}
            ms = cold.get("ms", {})
            print(f"| {r['corpus']} | {r['index']} | {r['name']} | {r['static']['static_payload'] / 1024:.0f} | {cid} | "
                  f"{c['ndcg10']:.3f} | {(c['r10_exact'] or 0):.3f} | {cold.get('rounds', float('nan')):.0f} | "
                  f"{cold.get('kb', float('nan')):.0f} | "
                  + " | ".join(f"{ms[p]:.0f}" if p in ms else "–" for p in PROFILES) + " |")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("corpus"); b.add_argument("index"); b.add_argument("name"); b.add_argument("params")
    d = sub.add_parser("downgrade")
    d.add_argument("corpus"); d.add_argument("name"); d.add_argument("newname")
    e = sub.add_parser("measure")
    e.add_argument("corpus"); e.add_argument("name")
    e.add_argument("--configs", default="")
    e.add_argument("--opts", default="", help="extra late_plaid query options appended to each config's opts")
    e.add_argument("--cold", type=int, default=0, help="cold queries through WASM (0 = skip)")
    t = sub.add_parser("table")
    t.add_argument("corpora", nargs="+")
    a = ap.parse_args()
    {"build": cmd_build, "downgrade": cmd_downgrade, "measure": cmd_eval, "table": cmd_table}[a.cmd](a)


if __name__ == "__main__":
    main()
