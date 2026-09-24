#!/usr/bin/env python3
"""FTS5 table options and ranking reads (docs/fts5-httpvfs.md, option d).

    python3 bench/fts5_options.py CORPUS.txt|PARAGRAPHS.jsonl QUERIES.jsonl [--n 300] [--out J]

Builds one external-content FTS5 table per option set in a scratch database
(docs + fts, 'optimize', VACUUM, 4 KiB pages), then for the first N queries
of each kind (OR of the quoted words) counts the distinct pages a cold
connection reads (APSW VFS shim, bench/countvfs.py) for bm25() and bm25c(),
and nDCG@10 of both. The scratch databases are deleted afterwards.
"""
import argparse, json, os, pathlib, sys, tempfile, time
import apsw

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bench"))
from countvfs import CountingVFS, pages_touched  # noqa: E402
from metrics import ndcg_at  # noqa: E402

FTS5RANK = str(ROOT / "build" / "native" / "ext" / "fts5rank.so")
OPTIONS = ["", "columnsize=0", "detail=column", "detail=none"]
SQL = {"bm25": "SELECT rowid FROM fts WHERE fts MATCH ? ORDER BY rank LIMIT 10",
       "bm25c": "SELECT rowid FROM fts WHERE fts MATCH ? AND rank MATCH 'bm25c()' ORDER BY rank LIMIT 10"}


def load_docs(path):
    if path.endswith(".jsonl"):
        rows = {}
        for line in open(path):
            o = json.loads(line)
            rows[o["i"]] = o["text"]
        return [rows[i] for i in sorted(rows)]
    return open(path, encoding="utf-8").read().split("\n")[:-1]


def build(path, docs, opt):
    con = apsw.Connection(path)
    con.execute("PRAGMA page_size=4096; PRAGMA journal_mode=OFF;")
    con.execute("CREATE TABLE docs(id INTEGER PRIMARY KEY, body TEXT)")
    con.execute("BEGIN")
    con.executemany("INSERT INTO docs VALUES (?, ?)", enumerate(docs))
    con.execute("COMMIT")
    extra = (", " + opt) if opt else ""
    con.execute(f"CREATE VIRTUAL TABLE fts USING fts5(body, content='docs', content_rowid='id'{extra})")
    con.execute("INSERT INTO fts(fts) VALUES ('rebuild'); INSERT INTO fts(fts) VALUES ('optimize'); VACUUM;")
    size = dict(con.execute("SELECT name, SUM(pgsize) FROM dbstat WHERE name LIKE 'fts%' GROUP BY name"))
    con.close()
    return size


def connect(path, vfs=None):
    con = apsw.Connection(path, flags=apsw.SQLITE_OPEN_READONLY, **({"vfs": vfs} if vfs else {}))
    con.enable_load_extension(True)
    con.load_extension(FTS5RANK)
    return con


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("queries")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out")
    a = ap.parse_args()
    docs = load_docs(a.corpus)
    qs, per = [], {}
    for line in open(a.queries):
        q = json.loads(line)
        if per.get(q["kind"], 0) < a.n:
            per[q["kind"]] = per.get(q["kind"], 0) + 1
            qs.append(q)
    vfs = CountingVFS()
    res = {"corpus": a.corpus, "n_docs": len(docs), "options": {}}
    with tempfile.TemporaryDirectory(dir=ROOT / "build") as tmp:
        for opt in OPTIONS:
            path = os.path.join(tmp, "o.db")
            t = time.time()
            size = build(path, docs, opt)
            warm = connect(path)
            r = {"bytes": size, "build_s": time.time() - t}
            for m, sql in SQL.items():
                pages, nd, err = {}, {}, 0
                for q in qs:
                    match = " OR ".join('"' + w.replace('"', '""') + '"' for w in q["text"].split())
                    try:
                        ids = [x[0] for x in warm.execute(sql, (match,))]
                    except apsw.Error:
                        err += 1
                        continue
                    nd.setdefault(q["kind"], []).append(ndcg_at(ids, set(q["relevant"]), 10))
                    cold = connect(path, vfs.name)
                    list(cold.execute("SELECT 1 FROM sqlite_schema LIMIT 1"))
                    vfs.reset()
                    list(cold.execute(sql, (match,)))
                    pages.setdefault(q["kind"], []).append(pages_touched(vfs.reads, 4096))
                    cold.close()
                r[m] = {"errors": err, **{k: {"cold_pages": sum(v) / len(v), "ndcg@10": sum(nd[k]) / len(nd[k])}
                                         for k, v in pages.items()}}
            warm.close()
            os.unlink(path)
            res["options"][opt or "default"] = r
            print(opt or "default", json.dumps(r), flush=True)
    if a.out:
        p = pathlib.Path(a.out)
        old = json.loads(p.read_text()) if p.exists() else {}
        old["options"] = res
        p.write_text(json.dumps(old, indent=1))


if __name__ == "__main__":
    main()
