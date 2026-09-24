#!/usr/bin/env python3
"""End-to-end benchmark matrix: quality, sizes, fetch costs and latency of every
retrieval configuration in bench/matrix_config.json, measured through the
WebAssembly build in Node against netsim's shaped range server.

Steps (each writes build/matrix/<corpus>.<step>.*; `all` runs them in order):

  quality   native SQLite (build/native) runs every query of every config at
            k = 10 and at k = 100 (for AUC), and computes recall@10, success@1,
            MRR@10, nDCG@10, AUC and recall@10 against the system's exact search
  metrics   recompute the metrics from the stored result lists
  trace     Node + WASM (Asyncify) against the unshaped server: every query in
            the warm regime (one connection), a subset in the cold regime (new
            connection per query); records VFS counters and the request log
  sim       replays every recorded trace, with its measured CPU gaps, through
            netsim/simulate.py under every profile x {h1, h2}
  real      Node + WASM against the server shaped with each profile x {h1, h2}
            on a query subset (profiles run in parallel, one server each)
  report    results/matrix/<corpus>.json, results/matrix/README.md and
            results/matrix/index.html (all corpora with results)

  python3 bench/matrix.py all words-10k
  python3 bench/matrix.py real words-10k --parallel 4 --profiles 4g,lte
  python3 bench/matrix.py report
"""
import argparse
import json
import re
import math
import os
import socket
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "bench"))
sys.path.insert(0, str(REPO / "tools"))

import numpy as np  # noqa: E402

import metrics as M  # noqa: E402  (bench/metrics.py)

BUILD = REPO / "build" / "matrix"
QDIR = REPO / "build" / "queries"
RESULTS = REPO / "results" / "matrix"
CONFIG = REPO / "bench" / "matrix_config.json"
NATIVE_EXT = REPO / "build" / "native" / "ext"


def load_config():
    return json.loads(CONFIG.read_text())


def manifest(corpus):
    p = BUILD / f"{corpus}.db.json"
    if not p.exists():
        raise SystemExit(f"{p} missing: run tools/build_db.py {corpus}")
    return json.loads(p.read_text())


def base(corpus):
    """'words-1m--late' (a per-index database built with --split) -> 'words-1m'."""
    return corpus.split("--")[0]


def qmeta(corpus):
    p = QDIR / f"{base(corpus)}.json"
    if not p.exists():
        raise SystemExit(f"{p} missing: run tools/encode_queries.py {corpus}")
    return json.loads(p.read_text())


def configs_for(cfg, man):
    """Configs whose index exists in the database, with {table} substituted."""
    out = []
    ccfg = cfg["corpora"].get(man["corpus"], {})
    for c in cfg["configs"]:
        spec = man["indexes"].get(c["index"])
        if spec is None:
            continue
        if c.get("exhaustive") and ccfg.get("skip_exhaustive"):
            continue
        layout = re.search(r"layout=(\w+)", spec.get("params_used") or "")
        if c.get("needs_layout") and layout and layout.group(1) not in (c["needs_layout"], "both"):
            continue
        out.append({**c, "sql_resolved": c["sql"].replace("{table}", spec["table"]),
                    "system": spec.get("system", c["index"])})
    return out


def interleave_kinds(meta, per_kind=None):
    """Query indices of every kind, kinds interleaved round-robin (a session mixes
    query kinds). With `per_kind`, a kind with more queries than that is sampled
    at random (seeded, so every step and run sees the same queries, and a smaller
    cap gives a prefix of a larger one): in the LLM query sets query i is about
    document i, and ties broken by rowid would favour a first-N subset."""
    by = {}
    for i, k in enumerate(meta["kinds"]):
        by.setdefault(k, []).append(i)
    lists = []
    for kind, v in by.items():
        if per_kind and len(v) > per_kind:
            rng = np.random.default_rng([12345, len(v), sum(map(ord, kind))])
            v = [v[j] for j in rng.permutation(len(v))[:per_kind]]
        lists.append(v)
    out = []
    for j in range(max(len(v) for v in lists)):
        out.extend(v[j] for v in lists if j < len(v))
    return out


def kind_cap(cfg, meta):
    """Queries per kind evaluated for this corpus (matrix_config corpora.*.max_per_kind)."""
    return cfg["corpora"].get(meta.get("name", ""), {}).get("max_per_kind")


def all_queries(cfg, meta):
    return interleave_kinds(meta, kind_cap(cfg, meta))


def query_set(cfg, meta, c):
    if c.get("exhaustive"):
        return interleave_kinds(meta, cfg["run"]["exhaustive_max_queries"] // 2)
    return all_queries(cfg, meta)


# ================================================================ quality

def native_db(path):
    import sqlite3
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True, isolation_level=None)
    db.enable_load_extension(True)
    for so in sorted(NATIVE_EXT.glob("*.so")):
        db.load_extension(str(so))
    return db


class QueryInputs:
    def __init__(self, meta):
        self.meta = meta
        self.minilm = np.fromfile(QDIR / meta["minilm"]["file"], "<f4").reshape(-1, 384)
        self.lateon = np.fromfile(QDIR / meta["lateon"]["file"], "<f4").reshape(-1, 48)
        self.off = meta["lateon"]["offsets"]

    def fts(self, qi, op):
        terms = ['"' + t.replace('"', '""') + '"' for t in self.meta["texts"][qi].split()]
        return (" OR " if op == "or" else " ").join(terms)

    def param(self, inp, qi):
        if inp == "fts_and":
            return self.fts(qi, "and")
        if inp == "fts_or":
            return self.fts(qi, "or")
        if inp == "minilm":
            return self.minilm[qi].tobytes()
        if inp == "lateon":
            return self.lateon[self.off[qi]:self.off[qi + 1]].tobytes()
        raise ValueError(inp)


def exact_dense(cfg, corpus, meta, qi_all, k):
    import build_db
    X = build_db.load_minilm(cfg, base(corpus), manifest(corpus)["n_docs"])
    Q = np.fromfile(QDIR / meta["minilm"]["file"], "<f4").reshape(-1, 384)[qi_all]
    Q = Q / np.linalg.norm(Q, axis=1, keepdims=True)
    best_s = np.full((len(Q), k), -np.inf, np.float32)
    best_i = np.zeros((len(Q), k), np.int64)
    for s in range(0, len(X), 200_000):
        Xc = np.asarray(X[s:s + 200_000], np.float32)
        Xc /= np.linalg.norm(Xc, axis=1, keepdims=True)
        S = Q @ Xc.T
        best_s, best_i = _merge_topk(best_s, best_i, S, s, k)
    return {qi: best_i[j].tolist() for j, qi in enumerate(qi_all)}


def _merge_topk(best_s, best_i, S, base, k):
    kk = min(k, S.shape[1])
    idx = np.argpartition(-S, kk - 1, axis=1)[:, :kk]
    cs = np.take_along_axis(S, idx, axis=1)
    allS = np.concatenate([best_s, cs], axis=1)
    allI = np.concatenate([best_i, idx + base], axis=1)
    o = np.argsort(-allS, axis=1, kind="stable")[:, :k]
    return np.take_along_axis(allS, o, axis=1), np.take_along_axis(allI, o, axis=1)


def exact_late(cfg, corpus, meta, qi_all, k, max_tokens=60_000_000):
    """Float-exact MaxSim over the original float16 token vectors."""
    import build_db
    p = build_db.emb_paths(cfg, base(corpus), "lateon")
    n = manifest(corpus)["n_docs"]
    off = np.load(p["offsets"])[:n + 1]
    if off[-1] > max_tokens:
        return None
    V = np.load(p["vectors"], mmap_mode="r")
    L = np.fromfile(QDIR / meta["lateon"]["file"], "<f4").reshape(-1, 48)
    qoff = meta["lateon"]["offsets"]
    best_s = np.full((len(qi_all), k), -np.inf, np.float32)
    best_i = np.zeros((len(qi_all), k), np.int64)
    d0 = 0
    while d0 < n:
        d1 = int(np.searchsorted(off, off[d0] + 2_000_000, side="right")) - 1
        d1 = max(d1, d0 + 1)
        d1 = min(d1, n)
        Vc = np.asarray(V[off[d0]:off[d1]], np.float32)
        starts = (off[d0:d1] - off[d0]).astype(np.int64)
        for b in range(0, len(qi_all), 64):
            qis = qi_all[b:b + 64]
            Qb = np.concatenate([L[qoff[q]:qoff[q + 1]] for q in qis])
            S = Vc @ Qb.T                                 # tokens x query tokens
            mx = np.maximum.reduceat(S, starts, axis=0)   # docs x query tokens
            bounds = np.cumsum([0] + [qoff[q + 1] - qoff[q] for q in qis])
            doc = np.add.reduceat(mx, bounds[:-1], axis=1)  # docs x queries
            bs, bi = _merge_topk(best_s[b:b + 64], best_i[b:b + 64], doc.T.copy(), d0, k)
            best_s[b:b + 64], best_i[b:b + 64] = bs, bi
        d0 = d1
    return {qi: best_i[j].tolist() for j, qi in enumerate(qi_all)}


def per_query_metrics(ids10, ids100, rel, n_docs, ref10):
    rel = set(rel)
    m = {
        "recall@10": M.recall_at(ids10, rel, 10),
        "success@1": M.success_at(ids10, rel, 1),
        "mrr@10": M.mrr_at(ids10, rel, 10),
        "ndcg@10": M.ndcg_at(ids10, rel, 10),
        "auc@100": M.auc(ids100, rel, n_docs) if ids100 is not None else float("nan"),
    }
    if ref10:   # undefined when the exact search returns nothing (e.g. an AND query with no match)
        m["r10_vs_exact"] = len(set(ids10[:10]) & set(ref10[:10])) / max(1, min(10, len(ref10)))
    return m


def compute_metrics(out, all_confs, meta, n_docs):
    default_ref = {"minilm": "exact:minilm", "lateon": "exact:lateon"}
    for cid, r in out["configs"].items():
        c = next((x for x in all_confs if x["id"] == cid), None)
        ref_key = (c.get("ref") or default_ref.get(c["input"]) or cid) if c else cid
        ref = out["configs"].get(ref_key)
        refmap = dict(zip(ref["qidx"], ref["ids10"])) if ref else {}
        rows = []
        for qi, a, b in zip(r["qidx"], r["ids10"], r["ids100"]):
            m = per_query_metrics(a, b, meta["relevant"][qi], n_docs, refmap.get(qi))
            m["kind"] = meta["kinds"][qi]
            cw = (meta.get("contains_word") or [None] * len(meta["kinds"]))[qi]
            if cw is not None:   # LLM paraphrase queries: did the model use the word anyway?
                m["group"] = f"{m['kind']}, word {'used' if cw else 'absent'}"
            rows.append(m)
        r["ref"] = ref_key if ref else None
        r["metrics"] = summarize_metrics(rows)


def step_metrics(cfg, corpus, args):
    """Recompute metrics from the stored result lists (no queries are run)."""
    man, meta = manifest(corpus), qmeta(corpus)
    qpath = BUILD / f"{corpus}.quality.json"
    out = json.loads(qpath.read_text())
    compute_metrics(out, configs_for(cfg, man), meta, man["n_docs"])
    qpath.write_text(json.dumps(out))
    print(f"[{corpus}] metrics recomputed")


def step_quality(cfg, corpus, args):
    man, meta = manifest(corpus), qmeta(corpus)
    all_confs = configs_for(cfg, man)
    confs = [c for c in all_confs if not args.configs or c["id"] in args.configs.split(",")]
    inputs = QueryInputs(meta)
    db = native_db(BUILD / man["file"])
    n_docs = man["n_docs"]
    k, kd = cfg["run"]["k"], cfg["run"]["k_deep"]
    all_q = all_queries(cfg, meta)
    t = time.time()
    refs = {}
    if any(c["input"] == "minilm" for c in all_confs):
        refs["exact:minilm"] = exact_dense(cfg, corpus, meta, all_q, kd)
        print(f"[{corpus}] exact MiniLM ground truth {time.time() - t:.1f} s", flush=True)
    if any(c["input"] == "lateon" for c in all_confs):
        t = time.time()
        g = exact_late(cfg, corpus, meta, all_q, kd)
        if g is not None:
            refs["exact:lateon"] = g
            print(f"[{corpus}] exact MaxSim ground truth {time.time() - t:.1f} s", flush=True)
    out = {"corpus": corpus, "n_docs": n_docs, "k": k, "k_deep": kd, "configs": {}}
    qpath = BUILD / f"{corpus}.quality.json"
    if args.configs and qpath.exists():      # partial rerun: keep the other configs
        out["configs"] = json.loads(qpath.read_text())["configs"]
    for c in confs:
        t = time.time()
        # exhaustive configs: all queries when affordable natively, else the subset
        # the network runs use; their top 10 is the prefix of their top 100
        exh = bool(c.get("exhaustive"))
        qs = all_q if exh and n_docs * len(all_q) <= cfg["run"].get("exhaustive_quality_budget", 3e7) \
            else query_set(cfg, meta, c)
        ids10, ids100, ms = {}, {}, {}
        for qi in qs:
            p = inputs.param(c["input"], qi)
            s = time.perf_counter()
            if exh:
                ids100[qi] = [r[0] for r in db.execute(c["sql_resolved"], {"q": p, "k": kd})] if p else []
                ids10[qi] = ids100[qi][:k]
            else:
                ids10[qi] = [r[0] for r in db.execute(c["sql_resolved"], {"q": p, "k": k})] if p else []
            ms[qi] = (time.perf_counter() - s) * 1e3
            if not exh:
                ids100[qi] = [r[0] for r in db.execute(c["sql_resolved"], {"q": p, "k": kd})] if p else []
        out["configs"][c["id"]] = {"qidx": qs, "ids10": [ids10[q] for q in qs], "ids100": [ids100[q] for q in qs],
                                   "native_ms": [round(ms[q], 3) for q in qs]}
        print(f"[{corpus}] quality {c['id']}: {len(qs)} queries, {time.time() - t:.1f} s", flush=True)
    # exact references as pseudo-configs (no network cost)
    for key, g in refs.items():
        out["configs"][key] = {"qidx": all_q, "ids10": [g[q][:k] for q in all_q], "ids100": [g[q] for q in all_q]}
    compute_metrics(out, all_confs, meta, n_docs)
    qpath.write_text(json.dumps(out))
    print(f"[{corpus}] wrote {BUILD / f'{corpus}.quality.json'}")


def summarize_metrics(rows):
    keys = [k for k in rows[0] if k not in ("kind", "group")]
    def mean(rs, key):
        v = [r[key] for r in rs if key in r and not (isinstance(r[key], float) and math.isnan(r[key]))]
        return float(np.mean(v)) if v else None
    out = {"all": {k: mean(rows, k) for k in keys}, "n": len(rows)}
    for kind in sorted({r["kind"] for r in rows}):
        rs = [r for r in rows if r["kind"] == kind]
        out[kind] = {k: mean(rs, k) for k in keys}
        out[kind]["n"] = len(rs)
    for g in sorted({r["group"] for r in rows if "group" in r}):
        rs = [r for r in rows if r.get("group") == g]
        out[g] = {k: mean(rs, k) for k in keys}
        out[g]["n"] = len(rs)
    return out


# ================================================================ server / node

def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class Server:
    def __init__(self, preset="none", seed=1):
        self.port = free_port()
        self.proc = subprocess.Popen(
            [sys.executable, str(REPO / "netsim" / "rangeserver.py"), "--dir", str(BUILD), "--port", str(self.port),
             "--preset", preset, "--seed", str(seed)],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        for _ in range(100):
            try:
                socket.create_connection(("127.0.0.1", self.port), timeout=0.2).close()
                return
            except OSError:
                time.sleep(0.1)
        raise RuntimeError("range server did not start: " + self.proc.stderr.read().decode()[-2000:])

    @property
    def base(self):
        return f"http://127.0.0.1:{self.port}"

    def close(self):
        self.proc.terminate()
        try:
            self.proc.wait(5)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def run_node(job, out_path):
    job_path = Path(str(out_path) + ".job.json")
    job_path.write_text(json.dumps(job))
    if Path(out_path).exists():
        Path(out_path).unlink()
    r = subprocess.run(["node", str(REPO / "bench" / "matrix.mjs"), str(job_path), str(out_path)],
                       capture_output=True, text=True)
    job_path.unlink()
    if r.returncode != 0:
        raise RuntimeError(f"node failed ({r.returncode}): {r.stderr[-3000:]}")


def node_run(c, k, regime, qidx, warmup, log=True):
    return {"config": c["id"], "sql": c["sql_resolved"], "input": c["input"], "k": k, "regime": regime,
            "qidx": qidx, "warmup": warmup, "log": log}


def cold_subset(cfg, meta, c, n):
    return [q for q in query_set(cfg, meta, c)][:n]


def step_trace(cfg, corpus, args):
    man, meta = manifest(corpus), qmeta(corpus)
    confs = configs_for(cfg, man)
    if args.configs:
        confs = [c for c in confs if c["id"] in args.configs.split(",")]
    k = cfg["run"]["k"]
    runs = []
    for c in confs:
        qs = query_set(cfg, meta, c)
        runs.append(node_run(c, k, "warm", qs, [qs[-1]]))
        runs.append(node_run(c, k, "cold", cold_subset(cfg, meta, c, cfg["run"]["cold_queries"]), []))
    srv = Server("none")
    out = BUILD / f"{corpus}.trace.jsonl"
    try:
        # split runs over 2 Node processes (moderate CPU), then concatenate
        halves = [runs[0::2], runs[1::2]]
        parts = [BUILD / f"{corpus}.trace.part{i}.jsonl" for i in range(len(halves))]
        t = time.time()
        with ThreadPoolExecutor(len(halves)) as ex:
            futs = [ex.submit(run_node, {"url": f"{srv.base}/{man['file']}", "queries": str(QDIR / f"{base(corpus)}.json"),
                                         "phases": [{"runs": h}]}, p) for h, p in zip(halves, parts)]
            for f in futs:
                f.result()
        keep = []
        if args.configs and out.exists():   # partial rerun: keep the other configs' records
            ids = {c["id"] for c in confs}
            keep = [line for line in open(out) if json.loads(line)["config"] not in ids]
        with open(out, "w") as fo:
            fo.writelines(keep)
            for p in parts:
                fo.write(p.read_text())
                p.unlink()
        print(f"[{corpus}] traces: {sum(1 for _ in open(out))} records in {time.time() - t:.0f} s -> {out}")
    finally:
        srv.close()


# ================================================================ simulation

def record_trace(rec):
    """netsim Trace from a runner record: one round per VFS backend call,
    CPU gaps between rounds from the request timestamps. Log entries are
    [offset, length, round, t_start, t_end] or, from newer runners,
    [..., req]: ranges of one multi-range request (equal req within a round)
    become one read of their total length."""
    from netsim import simulate as sim
    log = rec.get("log") or []
    by = {}
    for i, e in enumerate(log):
        off, ln, rnd, ts, te = e[:5]
        req = e[5] if len(e) > 5 else -1 - i
        rs = by.setdefault(int(rnd), [])
        if rs and req >= 0 and rs[-1][4] == req:
            o, n, a, b, _ = rs[-1]
            rs[-1] = (o, n + int(ln), min(a, ts), max(b, te), req)
        else:
            rs.append((int(off), int(ln), ts, te, req))
    reads, cpu = [], {}
    prev_end = 0.0
    for j, rnd in enumerate(sorted(by)):
        rs = by[rnd]
        start = min(r[2] for r in rs)
        cpu[j] = max(0.0, start - prev_end)
        for off, ln, ts, te, _ in rs:
            reads.append(sim.Read(off, ln, j, 0.0, None, len(reads)))
        prev_end = max(r[3] for r in rs)
    tail = max(0.0, rec["wall_ms"] - prev_end)
    return sim.Trace(reads, cpu, tail)


_PROFILES = {}


def profile(spec):
    if spec not in _PROFILES:
        from netsim import netmodel as nm
        _PROFILES[spec] = nm.preset(spec)
    return _PROFILES[spec]


def sim_record(args):
    rec, specs, seed = args
    from netsim import simulate as sim
    tr = record_trace(rec)
    return [sim.simulate(tr, profile(s), seed=seed).total_ms for s in specs]


def profile_specs(cfg):
    return [f"{p},{c}" for p in cfg["profiles"] for c in cfg["concurrency"]]


def step_sim(cfg, corpus, args):
    specs = profile_specs(cfg)
    recs = [json.loads(line) for line in open(BUILD / f"{corpus}.trace.jsonl")]
    sim_path = BUILD / f"{corpus}.sim.json"
    old_recs = []
    if args.configs and sim_path.exists():   # partial rerun: simulate only these configs
        ids = set(args.configs.split(","))
        recs = [r for r in recs if r["config"] in ids]
        old = json.loads(sim_path.read_text())
        if old["specs"] == specs:
            old_recs = [r for r in old["records"] if r["config"] not in ids]
    t = time.time()
    jobs = [(r, specs, r["qi"]) for r in recs]
    with ProcessPoolExecutor(args.workers) as ex:
        res = list(ex.map(sim_record, jobs, chunksize=64))
    out = {"specs": specs, "records": old_recs}
    for r, ms in zip(recs, res):
        out["records"].append({"config": r["config"], "regime": r["regime"], "qi": r["qi"], "ms": [round(x, 2) for x in ms],
                               "rounds": r["stats"]["rounds"], "requests": r["stats"]["requests"],
                               "bytes": r["stats"]["bytes"], "cacheHits": r["stats"]["cacheHits"],
                               "wall_ms": r["wall_ms"], "query_ms": r["query_ms"], "ids": r["ids"],
                               "error": r.get("error"), "ext": r.get("ext")})
    sim_path.write_text(json.dumps(out))
    print(f"[{corpus}] simulated {len(recs)} traces x {len(specs)} profiles in {time.time() - t:.0f} s")


# ================================================================ real network

def step_real(cfg, corpus, args):
    man, meta = manifest(corpus), qmeta(corpus)
    confs = configs_for(cfg, man)
    if args.configs:
        confs = [c for c in confs if c["id"] in args.configs.split(",")]
    run = cfg["run"]
    k = run["k"]
    profiles = args.profiles.split(",") if args.profiles else cfg["profiles"]
    specs = [f"{p},{c}" for p in profiles for c in cfg["concurrency"]]
    fast = {"none", "lan"}   # exhaustive configs: thousands of rounds per query, only here
    jobs = []
    for spec in specs:
        p = spec.split(",")[0]
        slow = p in run["slow_profiles"]
        per_kind = run["real_per_kind_slow"] if slow else run["real_per_kind"]
        runs = []
        for c in confs:
            if c.get("exhaustive") and p not in fast:
                continue
            qs = interleave_kinds(meta, 5 if c.get("exhaustive") else per_kind)
            warm_up = [all_queries(cfg, meta)[-1]]
            runs.append(node_run(c, k, "warm", qs, warm_up))
            runs.append(node_run(c, k, "cold", interleave_kinds(meta, 2 if c.get("exhaustive") else max(2, per_kind // 5)), []))
        jobs.append((spec, runs))

    def one(job):
        spec, runs = job
        srv = Server(spec, seed=1)
        tag = ("." + args.configs.replace(",", "+")) if args.configs else ""
        out = BUILD / f"{corpus}.real.{spec.replace(',', '_')}{tag}.jsonl"
        t = time.time()
        try:
            run_node({"url": f"{srv.base}/{man['file']}", "queries": str(QDIR / f"{base(corpus)}.json"),
                      "phases": [{"profile": None, "runs": runs}]}, out)
        finally:
            srv.close()
        # tag records with the profile
        lines = [json.loads(line) for line in open(out)]
        with open(out, "w") as fo:
            for r in lines:
                r["profile"] = spec
                fo.write(json.dumps(r) + "\n")
        print(f"[{corpus}] real {spec}: {len(lines)} queries in {time.time() - t:.0f} s", flush=True)
        return out

    with ThreadPoolExecutor(args.parallel) as ex:
        list(ex.map(one, jobs))


def load_real(corpus):
    recs = []
    for p in sorted(BUILD.glob(f"{corpus}.real.*.jsonl")):
        recs.extend(json.loads(line) for line in open(p))
    return recs


# ================================================================ report

def pct(xs, q):
    xs = [x for x in xs if x is not None]
    return float(np.percentile(xs, q)) if xs else None


def aggregate(cfg, corpus):
    man, meta = manifest(corpus), qmeta(corpus)
    confs = configs_for(cfg, man)
    qual = json.loads((BUILD / f"{corpus}.quality.json").read_text())
    simp = BUILD / f"{corpus}.sim.json"
    simd = json.loads(simp.read_text()) if simp.exists() else {"specs": [], "records": []}
    specs = simd["specs"]
    res = {"corpus": corpus, "n_docs": man["n_docs"], "db_file": man["file"], "db_bytes": man["bytes"],
           "sizes": {k: v["bytes"] for k, v in man["sizes"].items()}, "tables": man["tables"],
           "index_params": {k: v.get("params_used") for k, v in man["indexes"].items()},
           "build_seconds": man["build_seconds"], "k": qual["k"], "k_deep": qual["k_deep"],
           "n_queries": len(all_queries(cfg, meta)), "n_queries_in_set": meta["n"],
           "query_kinds": {k: [meta["kinds"][i] for i in all_queries(cfg, meta)].count(k)
                           for k in sorted(set(meta["kinds"]))},
           "encode_ms": {m: {"p50": pct(meta[m]["encode_ms"], 50), "p95": pct(meta[m]["encode_ms"], 95),
                             "threads": meta["threads"]} for m in ("minilm", "lateon")},
           "profiles": specs, "configs": [], "references": [], "encoders": meta.get("encoders")}
    by = {}
    for r in simd["records"]:
        by.setdefault((r["config"], r["regime"]), []).append(r)
    qconf = qual["configs"]
    for c in confs:
        q = qconf.get(c["id"])
        row = {"id": c["id"], "label": c["label"], "system": c["system"], "index": c["index"],
               "index_bytes": man["sizes"].get(c["index"], {}).get("bytes"),
               "exhaustive": bool(c.get("exhaustive")), "sql": c["sql"],
               "quality": q["metrics"] if q else None, "quality_ref": q.get("ref") if q else None,
               "native_ms_p50": pct(q["native_ms"], 50) if q else None}
        # agreement of WASM results with native results
        if q:
            native = dict(zip(q["qidx"], q["ids10"]))
            recs = by.get((c["id"], "warm"), [])
            same = [r["ids"] == native.get(r["qi"]) for r in recs if r["qi"] in native]
            row["wasm_native_agreement"] = float(np.mean(same)) if same else None
        for regime in ("warm", "cold"):
            recs = by.get((c["id"], regime), [])
            if not recs:
                continue
            rr = {"n": len(recs), "errors": sum(1 for r in recs if r.get("error")),
                  "rounds": float(np.mean([r["rounds"] for r in recs])),
                  "requests": float(np.mean([r["requests"] for r in recs])),
                  "kb": float(np.mean([r["bytes"] for r in recs])) / 1024,
                  "kb_p95": pct([r["bytes"] / 1024 for r in recs], 95),
                  "cache_hits": float(np.mean([r["cacheHits"] for r in recs])),
                  "cpu_ms_p50": pct([r["query_ms"] if regime == "warm" else r["wall_ms"] for r in recs], 50),
                  "latency": {}}
            if regime == "warm":
                rr["rounds_p95"] = pct([r["rounds"] for r in recs], 95)
            eb = [r["ext"]["bytes"] for r in recs if isinstance(r.get("ext"), dict)
                  and isinstance(r["ext"].get("bytes"), (int, float))]
            rr["ext_kb"] = float(np.mean(eb)) / 1024 if eb else None   # read by the extension, before the VFS cache
            for j, s in enumerate(specs):
                xs = [r["ms"][j] for r in recs]
                rr["latency"][s] = {"p50": pct(xs, 50), "p95": pct(xs, 95), "mean": float(np.mean(xs))}
            row[regime] = rr
        res["configs"].append(row)
    for key in ("exact:minilm", "exact:lateon"):
        if key in qconf:
            res["references"].append({"id": key, "label": {"exact:minilm": "float exact MiniLM (numpy)",
                                                          "exact:lateon": "float exact MaxSim LateOn (numpy)"}[key],
                                      "quality": qconf[key]["metrics"]})
    res["sim_vs_real"] = sim_vs_real(cfg, corpus, simd)
    res["coalesce"] = load_coalesce(corpus)
    return res


COALESCE_STRATEGIES = ("h1 baseline", "h1 coalesce C=6", "h1 multipart C=6", "h2 baseline")


def load_coalesce(corpus):
    """Rows of results/coalesce/<corpus>.json (bench/coalesce_eval.py): recorded traces
    re-simulated with the VFS's HTTP/1.1 request budget."""
    p = REPO / "results" / "coalesce" / f"{corpus}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return {"profiles": d["profiles"],
            "rows": [r for r in d["rows"] if r["strategy"] in COALESCE_STRATEGIES]}


def sim_vs_real(cfg, corpus, simd):
    """Compare real shaped-server wall times with (a) the simulation of the same
    run's own trace and (b) the pipeline estimate: the unshaped trace of the same
    query and config, simulated under the profile."""
    real = load_real(corpus)
    if not real:
        return None
    from netsim import simulate as sim
    specs = simd["specs"]
    est = {(r["config"], r["regime"], r["qi"]): r["ms"] for r in simd["records"]}
    out = {}
    for spec in sorted({r["profile"] for r in real}, key=lambda s: specs.index(s) if s in specs else 99):
        rs = [r for r in real if r["profile"] == spec and not r.get("error")]
        j = specs.index(spec) if spec in specs else None
        rows = {"warm": [], "cold": []}
        for r in rs:
            own = sim.simulate(record_trace(r), profile(spec), seed=r["qi"]).total_ms
            pipe = est.get((r["config"], r["regime"], r["qi"]))
            rows[r["regime"]].append((r["wall_ms"], own, pipe[j] if (pipe and j is not None) else None,
                                      r["stats"]["rounds"]))
        o = {}
        for regime, xs in rows.items():
            if not xs:
                continue
            real_ms = [x[0] for x in xs]
            own = [x[1] for x in xs]
            pipe = [x[2] for x in xs if x[2] is not None]
            realp = [x[0] for x in xs if x[2] is not None]
            netx = [x for x in xs if x[2] is not None and x[3] > 0]
            o[regime] = {
                "n": len(xs),
                "real_p50": pct(real_ms, 50), "real_p95": pct(real_ms, 95),
                "own_sim_p50": pct(own, 50), "own_sim_p95": pct(own, 95),
                "pipeline_p50": pct(pipe, 50), "pipeline_p95": pct(pipe, 95),
                "real_over_own_median": pct([a / b for a, b, *_ in xs if b > 0], 50),
                "real_minus_pipeline_median_ms": pct([a - p for a, _, p, _ in netx], 50),
                "real_minus_pipeline_per_round_ms": pct([(a - p) / n for a, _, p, n in netx], 50),
                "real_over_pipeline_median": pct([a / b for a, b in zip(realp, pipe) if b > 0], 50),
                "real_minus_own_median_ms": pct([a - b for a, b, *_ in xs], 50),
                "abs_rel_err_pipeline_median": pct([abs(a - b) / a for a, b in zip(realp, pipe) if a > 0], 50),
                # the same, over queries that went to the network (real >= 20 ms)
                "n_net": sum(1 for a in realp if a >= 20),
                "abs_rel_err_pipeline_net_median": pct([abs(a - b) / a for a, b in zip(realp, pipe) if a >= 20], 50),
                "abs_rel_err_pipeline_net_p90": pct([abs(a - b) / a for a, b in zip(realp, pipe) if a >= 20], 90),
            }
        out[spec] = o
    return out


def merge_splits(corpus, parts):
    """One result for a corpus measured as per-index databases (build_db.py --split)."""
    r = dict(parts[0])
    r["corpus"] = corpus
    r["split"] = [p["db_file"] for p in parts]
    r["db_file"] = ", ".join(p["db_file"] for p in parts)
    r["db_bytes"] = sum(p["db_bytes"] for p in parts)
    sizes = {}
    for p in parts:
        for k, v in p["sizes"].items():
            sizes[k] = sizes.get(k, 0) + v
    r["sizes"] = sizes
    r["index_params"] = {k: v for p in parts for k, v in p["index_params"].items()}
    r["tables"] = {k: v for p in parts for k, v in p["tables"].items()}
    r["build_seconds"] = {p["db_file"]: p["build_seconds"] for p in parts}
    r["configs"] = [c for p in parts for c in p["configs"]]
    seen, refs = set(), []
    for p in parts:
        for ref in p["references"]:
            if ref["id"] not in seen:
                seen.add(ref["id"]); refs.append(ref)
    r["references"] = refs
    sv = {}
    for p in parts:
        for spec, o in (p.get("sim_vs_real") or {}).items():
            sv[f"{spec} ({p['db_file'].split('--')[-1].replace('.db', '')})"] = o
    r["sim_vs_real"] = sv or None
    co = [p["coalesce"] for p in parts if p.get("coalesce")]
    r["coalesce"] = {"profiles": co[0]["profiles"], "rows": [x for c in co for x in c["rows"]]} if co else None
    return r


def fmt(x, d=3):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    if isinstance(x, float) and d == 0:
        return f"{x:,.0f}"
    return f"{x:.{d}f}"


def pctfmt(x):
    return "–" if x is None else f"{100 * x:.1f} %"


def mb(b):
    return "–" if b is None else f"{b / 1e6:.2f}"


def ms(x):
    if x is None:
        return "–"
    return f"{x:,.0f}" if x >= 10 else f"{x:.1f}"


def md_corpus(cfg, r):
    L = []
    prof = cfg["profiles"]
    L.append(f"## {r['corpus']}\n")
    sub = (f" evaluated, the first of each kind out of {r['n_queries_in_set']:,}"
           if r.get("n_queries_in_set", r["n_queries"]) > r["n_queries"] else "")
    if r.get("split"):
        L.append("Measured as separate per-index databases (`build_db.py --split`, to fit the disk; the "
                 "attribution check in `bench/MATRIX.md` shows this gives the same fetch costs): "
                 + ", ".join(f"`{x}`" for x in r["split"]) + ". Sizes are summed over the files.\n")
    L.append(f"{r['n_docs']:,} documents; {r['n_queries']:,} queries{sub} ("
             + ", ".join(f"{v:,} {k}" for k, v in r["query_kinds"].items())
             + f"). Database `{r['db_file']}`: {r['db_bytes'] / 1e6:.2f} MB.\n")
    if r.get("encoders"):
        L.append("Encoders: " + ", ".join(f"`{v}`" for v in r["encoders"].values()) + ".\n")
    agr = [c.get("wasm_native_agreement") for c in r["configs"] if c.get("wasm_native_agreement") is not None]
    if agr:
        L.append(f"The top-10 lists returned through WASM are identical to the native ones for "
                 f"{100 * min(agr):.1f} % (worst configuration) of the queries.\n")
    L.append("**Database composition** (dbstat, bytes of pages):\n")
    L.append("| Part | MB | Share | Index parameters |\n|---|---|---|---|")
    for k, v in sorted(r["sizes"].items(), key=lambda kv: -kv[1]):
        L.append(f"| {k} | {mb(v)} | {100 * v / r['db_bytes']:.1f} % | {r['index_params'].get(k) or ''} |")
    L.append("")
    # quality table
    L.append(f"**Quality** (k = {r['k']}; AUC from k = {r['k_deep']} lists; "
             "R@10 vs exact = overlap of the top 10 with the system's own exact search):\n")
    L.append("| Config | Index MB | Recall@10 | Success@1 | MRR@10 | nDCG@10 | AUC@100 | R@10 vs exact | Queries |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for c in r["configs"]:
        q = (c["quality"] or {}).get("all", {})
        L.append(f"| {c['label']} | {mb(c['index_bytes'])} | {fmt(q.get('recall@10'))} | {fmt(q.get('success@1'))} | "
                 f"{fmt(q.get('mrr@10'))} | {fmt(q.get('ndcg@10'))} | {fmt(q.get('auc@100'))} | "
                 f"{fmt(q.get('r10_vs_exact'))} | {(c['quality'] or {}).get('n', '–')} |")
    for ref in r["references"]:
        q = ref["quality"]["all"]
        L.append(f"| *{ref['label']}* | – | {fmt(q.get('recall@10'))} | {fmt(q.get('success@1'))} | "
                 f"{fmt(q.get('mrr@10'))} | {fmt(q.get('ndcg@10'))} | {fmt(q.get('auc@100'))} | 1.000 | "
                 f"{ref['quality']['n']} |")
    L.append("")
    kinds = list(r["query_kinds"]) + [g for g in (r["configs"][0].get("quality") or {})
                                       if "word " in g and g not in r["query_kinds"]]
    L.append("nDCG@10 by query kind" + (" (LLM paraphrase queries also split by whether the model "
                                         "used the target word, `contains_word`)" if len(kinds) > len(r["query_kinds"]) else "")
             + ":\n")
    def n_of(k):
        q = (r["configs"][0].get("quality") or {}).get(k) or {}
        return f"{k} (n={q['n']})" if "n" in q else k
    L.append("| Config | " + " | ".join(n_of(k) for k in kinds) + " |\n|---|" + "---|" * len(kinds))
    for c in r["configs"] + r["references"]:
        q = c["quality"] or {}
        L.append(f"| {c['label']} | " + " | ".join(fmt((q.get(k) or {}).get("ndcg@10")) for k in kinds) + " |")
    L.append("")
    # cost + latency, warm h2
    for regime, title in (("warm", "Warm session"), ("cold", "Cold start (new connection per query)")):
        for conc in cfg["concurrency"]:
            specs = [f"{p},{conc}" for p in prof]
            L.append(f"**{title}, {conc}** ({'HTTP/2-like, 100 concurrent requests' if conc == 'h2' else 'HTTP/1.1, 6 connections'}); "
                     "p50 / p95 simulated latency in ms from the recorded traces:\n")
            L.append("| Config | Rounds | Requests | KB | Ext. KB | " + " | ".join(prof) + " |")
            L.append("|---|---|---|---|---|" + "---|" * len(prof))
            for c in r["configs"]:
                d = c.get(regime)
                if not d:
                    continue
                cells = []
                for s in specs:
                    lt = d["latency"].get(s)
                    cells.append(f"{ms(lt['p50'])} / {ms(lt['p95'])}" if lt else "–")
                ek = "–" if d.get("ext_kb") is None else f"{d['ext_kb']:,.0f}"
                L.append(f"| {c['label']} | {d['rounds']:.1f} | {d['requests']:.1f} | {d['kb']:,.0f} | {ek} | " + " | ".join(cells) + " |")
            L.append("")
    co = r.get("coalesce")
    if co:
        L.append("**HTTP/1.1 with the request budget** (`bench/coalesce_eval.py`: the recorded traces re-simulated "
                 "with the VFS's per-round request planner; *coalesce* merges nearby ranges into at most six requests, "
                 "over-fetching the gaps; *multi-range* sends at most six requests with several ranges each). "
                 "p50 simulated latency in ms, one configuration per system:\n")
        L.append("| System (config) | Regime | Profile | h1 | h1 + coalesce | h1 + multi-range | h2 |")
        L.append("|---|---|---|---|---|---|---|")
        cell = {}
        for x in co["rows"]:
            cell[(x["system"], x["config"], x["regime"], x["profile"], x["strategy"])] = x
        keys = sorted({(x["system"], x["config"], x["regime"], x["profile"]) for x in co["rows"]},
                      key=lambda k: (k[0], k[2] != "warm", co["profiles"].index(k[3]) if k[3] in co["profiles"] else 9))
        for sysname, conf, regime, prof in keys:
            vals = [cell.get((sysname, conf, regime, prof, st)) for st in COALESCE_STRATEGIES]
            L.append(f"| {sysname} ({conf}) | {regime} | {prof} | "
                     + " | ".join(ms(v["p50"]) if v else "–" for v in vals) + " |")
        L.append("")
    sv = r.get("sim_vs_real")
    if sv:
        L.append("**Simulated versus real** (real = wall time through the shaped server; own-trace = the same run's "
                 "request log simulated; pipeline = the unshaped trace of the same query simulated, which is what the "
                 "tables above use). Medians over the real-run subset:\n")
        L.append("| Profile | Regime | Queries | Real p50 | Pipeline p50 | Real p95 | Pipeline p95 | "
                 "Median real / own-trace | Median real / pipeline | Median real − pipeline, ms (per round) | "
                 "Queries ≥ 20 ms | Abs. rel. error there, median / p90 |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for spec, o in sv.items():
            for regime, d in o.items():
                L.append(f"| {spec} | {regime} | {d['n']} | {ms(d['real_p50'])} | {ms(d['pipeline_p50'])} | "
                         f"{ms(d['real_p95'])} | {ms(d['pipeline_p95'])} | {fmt(d['real_over_own_median'])} | "
                         f"{fmt(d['real_over_pipeline_median'])} | {fmt(d['real_minus_pipeline_median_ms'], 1)} "
                         f"({fmt(d['real_minus_pipeline_per_round_ms'], 1)}) | {d['n_net']} | "
                         f"{pctfmt(d['abs_rel_err_pipeline_net_median'])} / {pctfmt(d['abs_rel_err_pipeline_net_p90'])} |")
        L.append("")
    return "\n".join(L)


def step_report(cfg, corpora, args):
    RESULTS.mkdir(parents=True, exist_ok=True)
    allres = []
    for corpus in corpora:
        if (BUILD / f"{corpus}.quality.json").exists():
            r = aggregate(cfg, corpus)
        else:
            splits = [p.name[:-len(".db.json")] for p in sorted(BUILD.glob(f"{corpus}--*.db.json"))
                      if (BUILD / f"{p.name[:-len('.db.json')]}.quality.json").exists()]
            if not splits:
                continue
            r = merge_splits(corpus, [aggregate(cfg, sp) for sp in splits])
        (RESULTS / f"{corpus}.json").write_text(json.dumps(r, indent=1))
        allres.append(r)
        print(f"[{corpus}] wrote {RESULTS / f'{corpus}.json'}")
    if not allres:
        raise SystemExit("no corpus has results")
    import matrix_report
    (RESULTS / "README.md").write_text(matrix_report.readme(cfg, allres, md_corpus))
    (RESULTS / "index.html").write_text(matrix_report.html(cfg, allres))
    print(f"wrote {RESULTS / 'README.md'} and {RESULTS / 'index.html'}")


# ================================================================ main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["quality", "metrics", "trace", "sim", "real", "report", "all"])
    ap.add_argument("corpora", nargs="*")
    ap.add_argument("--configs", default="", help="comma-separated config ids (default: all applicable)")
    ap.add_argument("--profiles", default="", help="real: comma-separated profiles (default: all)")
    ap.add_argument("--parallel", type=int, default=4, help="real: profiles measured at once")
    ap.add_argument("--workers", type=int, default=2, help="sim: processes")
    args = ap.parse_args()
    import build_db
    build_db.reexec_with_native_sqlite()   # Python's sqlite3 on SQLite 3.53.4 (build/native)
    cfg = load_config()
    corpora = args.corpora or [c for c in cfg["corpora"] if (BUILD / f"{c}.db.json").exists()
                                or any(BUILD.glob(f"{c}--*.db.json"))]
    if args.step == "report":
        return step_report(cfg, corpora, args)
    for corpus in corpora:
        steps = ["quality", "trace", "sim", "real"] if args.step == "all" else [args.step]
        for s in steps:
            {"quality": step_quality, "metrics": step_metrics, "trace": step_trace, "sim": step_sim, "real": step_real}[s](cfg, corpus, args)
    if args.step == "all":
        step_report(cfg, corpora, args)


if __name__ == "__main__":
    main()
