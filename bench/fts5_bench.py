#!/usr/bin/env python3
"""FTS5 baseline: build time, size, query latency, quality and cold page reads.

Usage: fts5_bench.py LABEL [--page-size N] [--out results/fts5-LABEL.json]
Builds build/fts5-words-LABEL-pPAGESIZE.db from data/corpora/words-LABEL.txt,
then runs data/queries/words-LABEL.jsonl.

Each query kind is run two ways:
  and — the FTS5 default (all terms required), ranked by bm25;
  or  — any term, ranked by bm25.
"""
import argparse, json, os, pathlib, sys, time
import apsw
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from metrics import recall_at, success_at, mrr_at, ndcg_at, auc, summarize, percentile
from countvfs import CountingVFS, pages_touched

ROOT = pathlib.Path(__file__).resolve().parent.parent


def build(db_path, corpus, page_size):
    if db_path.exists():
        db_path.unlink()
    con = apsw.Connection(str(db_path))
    con.execute(f"PRAGMA page_size={page_size}; PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF;")
    con.execute("CREATE TABLE docs(id INTEGER PRIMARY KEY, body TEXT);"
                "CREATE VIRTUAL TABLE fts USING fts5(body, content='docs', content_rowid='id');")
    docs = corpus.read_text(encoding="utf-8").split("\n")[:-1]
    t = {}
    t0 = time.perf_counter()
    con.execute("BEGIN")
    con.executemany("INSERT INTO docs(id, body) VALUES (?, ?)", enumerate(docs))
    con.execute("COMMIT")
    t["insert_docs_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    con.execute("INSERT INTO fts(fts) VALUES ('rebuild')")
    t["fts_rebuild_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    con.execute("INSERT INTO fts(fts) VALUES ('optimize')")
    t["fts_optimize_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    con.execute("VACUUM")
    t["vacuum_s"] = time.perf_counter() - t0
    # Incremental insertion rate, measured separately on a copy-free path:
    # insert the last 1% of docs again through triggers-free direct fts insert.
    con.close()
    t["n_docs"] = len(docs)
    t["docs_per_s_bulk"] = len(docs) / (t["insert_docs_s"] + t["fts_rebuild_s"])
    return t


def incremental_insert_rate(corpus, page_size, n=20000):
    """Docs/s when inserting into an FTS5 table row by row in one transaction."""
    docs = corpus.read_text(encoding="utf-8").split("\n")[:n]
    con = apsw.Connection(":memory:")
    con.execute(f"PRAGMA page_size={page_size}")
    con.execute("CREATE VIRTUAL TABLE f USING fts5(body)")
    t0 = time.perf_counter()
    con.execute("BEGIN")
    for d in docs:
        con.execute("INSERT INTO f(body) VALUES (?)", (d,))
    con.execute("COMMIT")
    return len(docs) / (time.perf_counter() - t0)


def fts_query(text, mode):
    terms = ['"' + w.replace('"', '""') + '"' for w in text.split()]
    return (" OR " if mode == "or" else " ").join(terms)


def run_queries(db_path, queries, n_docs, page_size, k=10):
    vfs = CountingVFS()
    warm = apsw.Connection(str(db_path), flags=apsw.SQLITE_OPEN_READONLY)
    results = {}
    sql = "SELECT rowid FROM fts WHERE fts MATCH ? ORDER BY rank LIMIT ?"
    for kind in sorted({q["kind"] for q in queries}):
        for mode in ("and", "or"):
            rows, lat, cold_pages, cold_bytes = [], [], [], []
            for q in (q for q in queries if q["kind"] == kind):
                m = fts_query(q["text"], mode)
                t0 = time.perf_counter()
                ranked = [r[0] for r in warm.execute(sql, (m, k))]
                lat.append((time.perf_counter() - t0) * 1000)
                rel = set(q["relevant"])
                rows.append({"recall@10": recall_at(ranked, rel, k), "success@1": success_at(ranked, rel, 1),
                             "success@10": success_at(ranked, rel, k), "mrr@10": mrr_at(ranked, rel, k),
                             "ndcg@10": ndcg_at(ranked, rel, k), "auc": auc(ranked, rel, n_docs)})
                if len(cold_pages) < 200:  # cold-cache page reads on a subsample
                    cold = apsw.Connection(str(db_path), flags=apsw.SQLITE_OPEN_READONLY, vfs=vfs.name)
                    list(cold.execute("SELECT 1 FROM sqlite_schema LIMIT 1"))
                    vfs.reset()
                    list(cold.execute(sql, (m, k)))
                    cold_pages.append(pages_touched(vfs.reads, page_size))
                    cold_bytes.append(sum(n for _, n in vfs.reads))
                    cold.close()
            s = summarize(rows)
            s.update({"n": len(rows), "lat_ms_p50": percentile(lat, 50), "lat_ms_p95": percentile(lat, 95),
                      "lat_ms_mean": sum(lat) / len(lat), "cold_pages_mean": sum(cold_pages) / len(cold_pages),
                      "cold_pages_p95": percentile(cold_pages, 95), "cold_bytes_mean": sum(cold_bytes) / len(cold_bytes)})
            results[f"{kind}/{mode}"] = s
            print(f"  {kind}/{mode}: " + ", ".join(f"{a}={b:.4g}" for a, b in s.items()), flush=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label")
    ap.add_argument("--page-size", type=int, default=4096)
    ap.add_argument("--out")
    a = ap.parse_args()
    (ROOT / "build").mkdir(exist_ok=True)
    db = ROOT / f"build/fts5-words-{a.label}-p{a.page_size}.db"
    corpus = ROOT / f"data/corpora/words-{a.label}.txt"
    queries = [json.loads(l) for l in (ROOT / f"data/queries/words-{a.label}.jsonl").open()]
    print(f"building {db.name}", flush=True)
    res = {"label": a.label, "page_size": a.page_size, "sqlite": apsw.sqlite_lib_version(), "load": os.getloadavg()}
    res["build"] = build(db, corpus, a.page_size)
    res["build"]["incremental_docs_per_s"] = incremental_insert_rate(corpus, a.page_size)
    res["build"]["db_bytes"] = db.stat().st_size
    con = apsw.Connection(str(db))
    res["build"]["bytes_by_table"] = dict(con.execute(
        "SELECT name, SUM(pgsize) FROM dbstat GROUP BY name ORDER BY 2 DESC").fetchall()) if True else {}
    con.close()
    print(json.dumps(res["build"], indent=1), flush=True)
    res["queries"] = run_queries(db, queries, res["build"]["n_docs"], a.page_size)
    out = pathlib.Path(a.out) if a.out else ROOT / f"results/fts5-words-{a.label}-p{a.page_size}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
