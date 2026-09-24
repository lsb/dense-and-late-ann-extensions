#!/usr/bin/env python3
"""FTS5 ranking over httpvfs: quality and fetch costs of the ranking methods.

    python3 bench/fts5_rank.py quality DB QUERIES [--max-per-kind N] [--out J]
    python3 bench/fts5_rank.py cost DB QUERIES [--max-per-kind N] [--cold N] [--out J]
    python3 bench/fts5_rank.py report J [J ...]      (markdown tables)

DB is a database with an FTS5 table `fts` over docs(id, body) (tools/build_db.py);
QUERIES is data/queries/<corpus>.jsonl. Methods (docs/fts5-httpvfs.md):

  bm25           FTS5's bm25(): exact; reads one fts_docsize row per match
  bm25c          ext/fts5rank's bm25c(): bm25 with every document at the
                 average length; no per-document reads
  bm25-rerank    bm25c()'s top 50 re-scored with their stored lengths (exact
                 bm25 formula on the candidates; one batch of fts_docsize rows)
  bm25-prefetch  bm25() after fetching the fts_docsize rows of all matches in
                 one speculative batch (web/lib/search-core.mjs)
  none           rowid order (unranked)

`quality` runs natively (Python sqlite3 on build/native + fts5rank.so), k = 10
and k = 100 (AUC). `cost` serves DB with netsim/rangeserver.py (unshaped, or
--preset P for a real shaped run), runs the queries through the WASM build in
Node (bench/fts5_rank.mjs, which calls SearchDb.search of
web/lib/search-core.mjs), records each query's request log (build/fts5-rank/)
and simulates it on 4g and lte with h1 and h2 (netsim/simulate.py). It also
measures reading the whole fts_docsize table once (option b). Results merge
into the --out JSON (results/fts5-rank/<corpus>.json).
"""
import argparse, json, math, os, pathlib, subprocess, sys, time
from concurrent.futures import ProcessPoolExecutor

ROOT = pathlib.Path(__file__).resolve().parent.parent
NATIVE = ROOT / "build" / "native"
sys.path.insert(0, str(ROOT / "bench"))
sys.path.insert(0, str(ROOT))
from metrics import recall_at, success_at, mrr_at, ndcg_at, auc  # noqa: E402

if os.environ.get("_FTS5_RANK_REEXEC") is None:
    env = dict(os.environ, LD_LIBRARY_PATH=f"{NATIVE}:{os.environ.get('LD_LIBRARY_PATH', '')}",
               _FTS5_RANK_REEXEC="1")
    os.execve(sys.executable, [sys.executable] + sys.argv, env)

METHODS_Q = ["bm25", "bm25c", "bm25-rerank", "none"]
METHODS_C = ["bm25", "bm25-prefetch", "bm25c", "bm25-rerank", "none"]
PROFILES = ["4g,h1", "4g,h2", "lte,h1", "lte,h2"]


def fts_query(text, mode):
    terms = ['"' + t.replace('"', '""') + '"' for t in text.split()]
    return (" OR " if mode == "or" else " ").join(terms)


def load_queries(path, max_per_kind):
    qs = [json.loads(l) for l in open(path)]
    by = {}
    for q in qs:
        by.setdefault(q["kind"], []).append(q)
    return {k: v[:max_per_kind] if max_per_kind else v for k, v in by.items()}


SQL = {
    "bm25": "SELECT rowid FROM fts WHERE fts MATCH ? ORDER BY rank LIMIT ?",
    "bm25c": "SELECT rowid FROM fts WHERE fts MATCH ? AND rank MATCH 'bm25c()' ORDER BY rank LIMIT ?",
    "none": "SELECT rowid FROM fts WHERE fts MATCH ? LIMIT ?",
}


def rerank(db, match, k, depth=50):
    """The SQL of web/lib/search-core.mjs's 'bm25-rerank' (depth max(50, k))."""
    cand = [r[0] for r in db.execute(SQL["bm25c"], (match, max(depth, k)))]
    if not cand:
        return []
    lst = ",".join(str(int(c)) for c in cand)
    return [r[0] for r in db.execute(
        f"SELECT id FROM (SELECT rowid AS id, CASE WHEN rowid IN ({lst}) THEN bm25dl(fts, (SELECT sz FROM "
        f"fts_docsize WHERE id = fts.rowid)) END AS score FROM fts WHERE fts MATCH ?) WHERE score IS NOT NULL "
        f"ORDER BY score LIMIT ?", (match, k))]


def step_quality(a):
    import sqlite3
    db = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True)
    db.enable_load_extension(True)
    db.load_extension(str(NATIVE / "ext" / "fts5rank"))
    n_docs = db.execute("SELECT max(rowid) + 1 FROM docs").fetchone()[0]
    queries = load_queries(a.queries, a.max_per_kind)
    out = {"db": str(a.db), "n_docs": n_docs, "quality": {}}
    for kind, qs in queries.items():
        for mode in ("and", "or"):
            res = {}
            for m in METHODS_Q:
                rows, t = [], time.perf_counter()
                for q in qs:
                    match = fts_query(q["text"], mode)
                    r100 = rerank(db, match, 100) if m == "bm25-rerank" else \
                        [r[0] for r in db.execute(SQL[m], (match, 100))]
                    r10 = r100[:10]
                    rel = set(q["relevant"])
                    ref = res.get("bm25", {}).get("_ids", {}).get(q["qid"])
                    rows.append({"qid": q["qid"], "ids": r10, "ndcg@10": ndcg_at(r10, rel, 10),
                                 "mrr@10": mrr_at(r10, rel, 10), "recall@10": recall_at(r10, rel, 10),
                                 "success@1": success_at(r10, rel, 1), "auc@100": auc(r100, rel, n_docs),
                                 "vs_bm25@10": (len(set(r10) & set(ref)) / len(ref) if ref else float("nan"))
                                 if m != "bm25" else 1.0})
                s = {k: mean([r[k] for r in rows]) for k in
                     ("ndcg@10", "mrr@10", "recall@10", "success@1", "auc@100", "vs_bm25@10")}
                s["n"] = len(rows)
                s["ms_per_query"] = (time.perf_counter() - t) * 1000 / len(rows)
                s["_ids"] = {r["qid"]: r["ids"] for r in rows}
                res[m] = s
                print(f"{kind}/{mode} {m}: " + ", ".join(f"{k}={v:.4f}" for k, v in s.items() if k != "_ids"), flush=True)
            for m in res:
                res[m].pop("_ids")
            out["quality"][f"{kind}/{mode}"] = res
    write(a.out, out)


def mean(xs):
    xs = [x for x in xs if not (isinstance(x, float) and math.isnan(x))]
    return sum(xs) / len(xs) if xs else float("nan")


def write(path, obj):
    if path:
        p = pathlib.Path(path)
        old = json.loads(p.read_text()) if p.exists() else {}
        old.update(obj)
        p.write_text(json.dumps(old, indent=1))


# ------------------------------------------------------------------ costs

def record_trace(rec):
    """netsim Trace from a request log: one round per VFS backend call, CPU
    gaps between rounds from the timestamps (as bench/matrix.py)."""
    from netsim import simulate as sim
    by = {}
    for off, ln, rnd, ts, te in rec["log"]:
        by.setdefault(int(rnd), []).append((int(off), int(ln), ts, te))
    reads, cpu, prev_end = [], {}, 0.0
    for j, rnd in enumerate(sorted(by)):
        rs = by[rnd]
        cpu[j] = max(0.0, min(r[2] for r in rs) - prev_end)
        for off, ln, ts, te in rs:
            reads.append(sim.Read(off, ln, j, 0.0, None, len(reads)))
        prev_end = max(r[3] for r in rs)
    return sim.Trace(reads, cpu, max(0.0, rec["wall_ms"] - prev_end))


def sim_one(rec):
    from netsim import simulate as sim, netmodel as nm
    tr = record_trace(rec)
    return [sim.simulate(tr, nm.preset(p), seed=rec.get("qi", 0)).total_ms for p in PROFILES]


class Server:
    """netsim/rangeserver.py serving one directory, unshaped."""
    def __init__(self, directory, preset="none"):
        import socket
        s = socket.socket(); s.bind(("127.0.0.1", 0)); self.port = s.getsockname()[1]; s.close()
        self.proc = subprocess.Popen([sys.executable, str(ROOT / "netsim" / "rangeserver.py"), "--dir", str(directory),
                                      "--port", str(self.port), "--preset", preset],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(100):
            try:
                socket.create_connection(("127.0.0.1", self.port), timeout=0.2).close()
                return
            except OSError:
                time.sleep(0.1)
        raise RuntimeError("range server did not start")

    @property
    def base(self):
        return f"http://127.0.0.1:{self.port}"

    def close(self):
        self.proc.terminate()
        self.proc.wait(5)


def step_cost(a):
    db = pathlib.Path(a.db).resolve()
    queries = load_queries(a.queries, a.max_per_kind)
    jobq = []
    for kind, qs in queries.items():
        for mode in ("and", "or"):
            if a.kinds and f"{kind}/{mode}" not in a.kinds.split(","):
                continue
            for q in qs:
                jobq.append({"qid": q["qid"], "kind": f"{kind}/{mode}", "match": fts_query(q["text"], mode)})
    srv = Server(db.parent, a.preset)
    (ROOT / "build" / "fts5-rank").mkdir(parents=True, exist_ok=True)
    suffix = "" if a.preset == "none" else "." + a.preset.replace(",", "_")
    out_jsonl = ROOT / "build" / "fts5-rank" / (pathlib.Path(a.out or "fts5-rank").stem + suffix + ".cost.jsonl")
    methods = a.methods.split(",") if a.methods else METHODS_C
    job = {"url": f"{srv.base}/{db.name}", "queries": jobq, "methods": methods, "cold": a.cold,
           "warm": a.warm, "pageCacheBytes": a.cache_mb << 20,
           "session": a.preset == "none"}
    job_path = pathlib.Path(str(out_jsonl) + ".job.json")
    job_path.write_text(json.dumps(job))
    try:
        t = time.time()
        r = subprocess.run(["node", str(ROOT / "bench" / "fts5_rank.mjs"), str(job_path), str(out_jsonl)],
                           capture_output=True, text=True)
        if r.returncode:
            raise SystemExit(r.stderr[-3000:])
        print(r.stderr[-2000:], f"node: {time.time() - t:.0f} s", flush=True)
    finally:
        srv.close()
        job_path.unlink(missing_ok=True)
    recs = [json.loads(l) for l in open(out_jsonl)]
    with ProcessPoolExecutor(2) as ex:
        sims = list(ex.map(sim_one, recs, chunksize=16))
    agg = {}
    for rec, s in zip(recs, sims):
        key = (rec["regime"], rec.get("kind", "-"), rec["method"])
        a_ = agg.setdefault(key, {"n": 0, "rounds": [], "requests": [], "bytes": [], "wall_ms": [],
                                  **{p: [] for p in PROFILES}})
        a_["n"] += 1
        for k in ("rounds", "requests", "bytes"):
            a_[k].append(rec["stats"][k])
        a_["wall_ms"].append(rec["wall_ms"])
        for p, v in zip(PROFILES, s):
            a_[p].append(v)
    res = {}
    for (regime, kind, m), v in sorted(agg.items()):
        res.setdefault(regime, {}).setdefault(kind, {})[m] = {
            "n": v["n"], **{k: mean(v[k]) for k in ("rounds", "requests", "bytes", "wall_ms")},
            "wall_ms p50": pct(v["wall_ms"], 50),
            **{f"{p} p50": pct(v[p], 50) for p in PROFILES}, **{f"{p} mean": mean(v[p]) for p in PROFILES},
            "rounds_p95": pct(v["rounds"], 95)}
    for regime in res:
        for kind in res[regime]:
            for m, s in res[regime][kind].items():
                print(f"{regime} {kind} {m}: rounds {s['rounds']:.1f} req {s['requests']:.1f} "
                      f"KB {s['bytes'] / 1024:.0f} | 4g,h2 p50 {s['4g,h2 p50']:.0f} ms, lte,h2 p50 {s['lte,h2 p50']:.0f} ms",
                      flush=True)
    if a.preset != "none":   # real shaped run: keep wall times next to the simulation of the same logs
        write(a.out, {f"real_{a.preset}": res})
    else:
        write(a.out, {"cost": res})


def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    i = (len(xs) - 1) * q / 100
    lo, hi = math.floor(i), math.ceil(i)
    return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)


def step_report(paths):
    order = ["bm25", "bm25-prefetch", "bm25-rerank", "bm25c", "none"]
    for path in paths:
        d = json.loads(pathlib.Path(path).read_text())
        print(f"\n### {pathlib.Path(path).stem} ({d.get('n_docs', '?'):,} documents)\n")
        kinds = [k for k in d.get("quality", {}) if k.endswith("/or")] or list(d.get("cost", {}).get("cold", {}))
        head = ["Method"]
        for k in kinds:
            head += [f"{k} nDCG@10", f"{k} AUC@100", f"{k} top-10 = bm25"]
        print("| " + " | ".join(head) + " |")
        print("|" + "---|" * len(head))
        for m in order:
            qm = m if m != "bm25-prefetch" else "bm25"
            row = [m]
            for k in kinds:
                q = d.get("quality", {}).get(k, {}).get(qm)
                row += [f"{q['ndcg@10']:.4f}", f"{q['auc@100']:.4f}", f"{q['vs_bm25@10']:.3f}"] if q else ["–"] * 3
            print("| " + " | ".join(row) + " |")
        cost = d.get("cost")
        if not cost:
            continue
        ck = sorted(k for k in cost["cold"] if k.endswith("/or"))
        print()
        head = ["Method"] + [f"{k} {r}" for r in ("cold", "warm") for k in ck]
        print("| " + " | ".join(head) + " |")
        print("|" + "---|" * len(head))
        for m in order:
            row = [m]
            for r in ("cold", "warm"):
                for k in ck:
                    c = cost[r].get(k, {}).get(m)
                    row.append(f"{c['rounds']:.1f} r, {c['bytes'] / 1024:.0f} KB, {c['4g,h2 p50']:.0f} / {c['lte,h2 p50']:.0f} ms"
                               if c else "–")
            print("| " + " | ".join(row) + " |")
        ses = cost.get("session", {}).get("-", {})
        if "docsize-scan" in ses:
            c = ses["docsize-scan"]
            print(f"\nWhole fts_docsize read once: {c['rounds']:.0f} rounds, {c['bytes'] / 1024:.0f} KB, "
                  f"{c['4g,h2 p50']:.0f} ms on 4g,h2, {c['lte,h2 p50']:.0f} ms on lte,h2.")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "report":
        return step_report(sys.argv[2:])
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["quality", "cost"])
    ap.add_argument("db")
    ap.add_argument("queries")
    ap.add_argument("--max-per-kind", type=int, default=0)
    ap.add_argument("--cold", type=int, default=50, help="cold queries per kind/mode and method")
    ap.add_argument("--warm", type=int, default=200, help="warm queries per kind/mode and method")
    ap.add_argument("--cache-mb", type=int, default=16, help="VFS block cache (the web client's default)")
    ap.add_argument("--preset", default="none", help="shape the range server (a real-latency run), e.g. 4g,h1")
    ap.add_argument("--kinds", help="comma-separated subset of kind/mode, e.g. word/or")
    ap.add_argument("--methods", help="comma-separated subset of " + ",".join(METHODS_C))
    ap.add_argument("--out")
    a = ap.parse_args()
    {"quality": step_quality, "cost": step_cost}[a.step](a)


if __name__ == "__main__":
    main()
