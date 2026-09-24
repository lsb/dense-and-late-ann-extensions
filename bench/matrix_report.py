"""README.md and index.html generation for the benchmark matrix (bench/matrix.py report)."""
import html as H
import math
import time

HEADLINE = (("fts-bm25c-or",), ("graph-ef64",), ("ivf-np64", "ivf-np128"), ("warp-np8",), ("warp-np8-rr64",),
            ("late-exact",))   # one slot per method; the first configuration a corpus has
FAMILY = {"FTS5": 0, "dense graph": 1, "dense IVF": 1, "late": 2}
SHAPE = {"FTS5": "circle", "dense graph": "circle", "dense IVF": "square", "late": "circle"}


def _family(system):
    return FAMILY.get(system, 2)


def readme(cfg, results, md_corpus):
    L = ["# Benchmark matrix", ""]
    L.append("This directory holds the project's end-to-end benchmark: every retrieval configuration in "
             "`bench/matrix_config.json`, run through the WebAssembly SQLite build (Asyncify variant, in Node 22) "
             "against `netsim/rangeserver.py`, with quality measured against the relevance labels. It is generated "
             f"by `bench/matrix.py report` ({time.strftime('%Y-%m-%d', time.gmtime())}); `index.html` shows the "
             "same data as charts, and `<corpus>.json` holds every number, including per-kind quality and all "
             "profiles.")
    L.append("")
    L.append("## Method")
    L.append("")
    L.append("- **One database per corpus** (`tools/build_db.py`), as it would be deployed: `docs`, an FTS5 index, "
             "two `dense_ann` indexes (graph and IVF-PQ layouts, MiniLM) and one `late_plaid` index (warp and "
             "plaid layouts, LateOn-Code-edge), built, `VACUUM`ed and then finalized, 4 KiB pages. The size of "
             "each index is its pages as counted by `dbstat`. A query touches only its own index's pages plus "
             "page 1 and the schema, so fetch counts are attributable per system without separate databases; "
             "`tools/build_db.py --split` builds per-index databases for checking this.")
    L.append("- **Encoders.** Documents and queries are embedded with the weight-only int8 models "
             "(`models/*/model_w8.onnx`), one text at a time (`tools/encode_queries.py`). FTS5 queries quote each term; AND is FTS5's implicit conjunction.")
    L.append("- **Quality** comes from native SQLite with the same extension code (the WASM results agree; see "
             "the agreement column in the JSON). Recall@10, success@1, MRR@10 and nDCG@10 use k = 10. "
             "**AUC is computed from a separate k = 100 run** (documents not returned count as tied below all "
             "returned ones), because a top-10 list says almost nothing about the rest of the collection; for "
             "the graph index k = 100 also raises the candidate list to at least 100. *R@10 vs exact* is the "
             "overlap of the top 10 with the system's own exact search: float32 cosine over the MiniLM vectors, "
             "float MaxSim over the float16 LateOn token vectors, and bm25 over all matches for FTS5.")
    L.append("- **Costs** are the WASM VFS's counters per query: *rounds* (dependent batches of parallel range "
             "requests), *requests* and *KB* transferred. *Ext. KB* is what the vector extension itself read per query, according to its own `stats` (before the VFS block cache), so the difference from *KB* is the effect of caching across queries in the session. *Warm*: one connection answers every query in turn "
             "after one warm-up query, with the default 4 MiB block cache, so per-connection static data "
             "(PQ codebooks, entry set, IVF centroids, late centroids) is already loaded and pages repeated "
             "across queries are cache hits. *Cold*: a new connection per query, so the numbers include "
             "opening the file, the schema and that static data (but not TCP/TLS handshakes).")
    L.append("- **Latency.** Every query's request log (offsets, lengths, round numbers, and the CPU gaps "
             "between rounds, measured in WASM) was recorded against the unshaped server and replayed through "
             "`netsim/simulate.py` under each profile, with `h1` (6 concurrent requests, HTTP/1.1) and `h2` "
             "(100). Tables give p50 / p95 over the queries. On a subset of queries the same runs were also "
             "made against the real shaped server, to validate the simulation (section *Simulated versus real*).")
    L.append("- Exhaustive rows read the whole index and are measured on at most "
             f"{cfg['run']['exhaustive_max_queries']} queries for costs; their quality uses all queries.")
    L.append("- Profiles: " + ", ".join(f"`{p}`" for p in cfg["profiles"]) + " (see `netsim/NOTES.md`).")
    L.append("")
    L.append("## Headline")
    L.append("")
    L.append("One configuration per method, all corpora. Latency is the simulated p50 on `4g` (165 ms, 8.1 Mbit/s) "
             "with HTTP/2-like concurrency; *warm* is a query within a session, *cold* a first query on a new "
             "connection.")
    L.append("")
    L.append("| Corpus | Config | Index MB | nDCG@10 | AUC@100 | Warm rounds | Warm KB | Warm p50 ms | Warm p95 ms | Cold KB | Cold p50 ms |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        have = {c["id"]: c for c in r["configs"]}
        for slot in HEADLINE:
            c = next((have[i] for i in slot if i in have), None)
            if c is None:
                continue
            q = (c.get("quality") or {}).get("all", {})
            w, co = c.get("warm") or {}, c.get("cold") or {}
            lw = (w.get("latency") or {}).get("4g,h2", {})
            lc = (co.get("latency") or {}).get("4g,h2", {})
            n = lambda v, d=0: "–" if v is None else f"{v:,.{d}f}"  # noqa: E731
            L.append(f"| {r['corpus']} | {c['label']} | {n((c.get('index_bytes') or 0) / 1e6, 1)} | "
                     f"{n(q.get('ndcg@10'), 3)} | {n(q.get('auc@100'), 3)} | {n(w.get('rounds'), 1)} | {n(w.get('kb'))} | "
                     f"{n(lw.get('p50'))} | {n(lw.get('p95'))} | {n(co.get('kb'))} | {n(lc.get('p50'))} |")
    L.append("")
    L.append("## Query encoding time")
    L.append("")
    L.append("Native ONNX Runtime, one query at a time, 1 intra-op thread, on the shared 4-core machine "
             "(browser encode times are measured separately, with the browser demo).")
    L.append("")
    L.append("| Query set | MiniLM p50 ms | MiniLM p95 ms | LateOn p50 ms | LateOn p95 ms |")
    L.append("|---|---|---|---|---|")
    for r in results:
        e = r["encode_ms"]
        L.append(f"| {r['corpus']} | {e['minilm']['p50']:.2f} | {e['minilm']['p95']:.2f} | "
                 f"{e['lateon']['p50']:.2f} | {e['lateon']['p95']:.2f} |")
    L.append("")
    for r in results:
        L.append(md_corpus(cfg, r))
    L.append("## How to reproduce")
    L.append("")
    L.append("```sh")
    L.append("make native wasm-asyncify                       # build/native, wasm/pkg/dist")
    L.append("python3 tools/build_db.py words-10k             # build/matrix/words-10k.db + .db.json")
    L.append("python3 tools/encode_queries.py words-10k       # build/queries/words-10k.*")
    L.append("python3 bench/matrix.py all words-10k           # quality, trace, sim, real, report")
    L.append("# or step by step: quality | trace | sim | real [--parallel 4 --profiles 4g,lte] | report")
    L.append("```")
    L.append("")
    L.append("The same commands work for `words-100`, `words-1m`, `llm-100` and `llm-10k` once their embeddings "
             "and query files exist (`bench/MATRIX.md`).")
    L.append("")
    return "\n".join(L)


# ------------------------------------------------------------------ HTML

CSS = """
:root{--surface:#fcfcfb;--text:#0b0b0b;--text2:#52514e;--grid:#e4e3df;--axis:#8a8983;
--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--border:#d9d8d3}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--surface:#1a1a19;--text:#fff;--text2:#c3c2b7;
--grid:#33322f;--axis:#77766f;--s1:#3987e5;--s2:#d95926;--s3:#199e70;--border:#3a3935}}
:root[data-theme="dark"]{--surface:#1a1a19;--text:#fff;--text2:#c3c2b7;--grid:#33322f;--axis:#77766f;
--s1:#3987e5;--s2:#d95926;--s3:#199e70;--border:#3a3935}
body{background:var(--surface);color:var(--text);font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif;
margin:0 auto;max-width:1100px;padding:16px}
h1{font-size:1.6em;margin:.3em 0}h2{font-size:1.25em;margin-top:2em;border-bottom:1px solid var(--border)}
p,li{color:var(--text2)}
.wrap{overflow-x:auto}
table{border-collapse:collapse;font-size:13px;margin:.5em 0 1.5em;font-variant-numeric:tabular-nums}
th,td{padding:4px 8px;border-bottom:1px solid var(--border);text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}
th{color:var(--text2);font-weight:600}
svg{max-width:100%;height:auto;display:block}
.pair{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:12px}
svg text{fill:var(--text2);font-size:11px}
svg .lbl{fill:var(--text);font-size:10.5px}
.legend{display:flex;gap:18px;flex-wrap:wrap;font-size:13px;color:var(--text2);margin:.3em 0}
.legend span{display:inline-flex;align-items:center;gap:6px}
.sw{width:10px;height:10px;border-radius:50%;display:inline-block}
.sq{border-radius:2px}
select{font:inherit;background:var(--surface);color:var(--text);border:1px solid var(--border);border-radius:4px;padding:2px 6px}
"""


def _fam_var(system):
    return f"var(--s{_family(system) + 1})"


def scatter(r, spec, metric="ndcg@10", regime="warm", W=720, Hh=380):
    pts = []
    for c in r["configs"]:
        d = c.get(regime)
        q = (c.get("quality") or {}).get("all", {})
        if not d or q.get(metric) is None or spec not in d["latency"]:
            continue
        pts.append((d["latency"][spec]["p50"], q[metric], c))
    if not pts:
        return "<p>No data.</p>"
    ml, mr, mt, mb = 50, 12, 22, 44
    xs = [max(1.0, p[0]) for p in pts]
    lo, hi = math.log10(min(xs)), math.log10(max(xs))
    lo, hi = math.floor(lo * 2) / 2, math.ceil(hi * 2) / 2
    if hi - lo < 1:
        hi = lo + 1
    ymax = min(1.0, math.ceil(max(p[1] for p in pts) * 10 + 0.5) / 10)
    X = lambda v: ml + (math.log10(max(1.0, v)) - lo) / (hi - lo) * (W - ml - mr)  # noqa: E731
    Y = lambda v: mt + (1 - v / ymax) * (Hh - mt - mb)  # noqa: E731
    s = [f'<svg viewBox="0 0 {W} {Hh}" role="img" aria-label="{H.escape(metric)} versus p50 latency, {spec}">']
    for k in range(0, 11):
        v = ymax * k / 10
        if k % 2 == 0:
            s.append(f'<line x1="{ml}" x2="{W - mr}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="var(--grid)"/>'
                     f'<text x="{ml - 6}" y="{Y(v) + 4:.1f}" text-anchor="end">{v:.1f}</text>')
    for d in range(math.floor(lo), math.ceil(hi) + 1):
        for m in ((1, 2, 5) if hi - lo <= 2.5 else (1,)):
            v = m * 10 ** d
            if not (lo - 1e-9 <= math.log10(v) <= hi + 1e-9):
                continue
            x = X(v)
            lab = f"{v / 1e6:,.0f}M" if v >= 1e6 else f"{v / 1000:,.0f}k" if v >= 10000 else f"{v:,.0f}"
            s.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{mt}" y2="{Hh - mb}" stroke="var(--grid)"/>'
                     f'<text x="{x:.1f}" y="{Hh - mb + 16}" text-anchor="middle">{lab}</text>')
    s.append(f'<line x1="{ml}" x2="{W - mr}" y1="{Hh - mb}" y2="{Hh - mb}" stroke="var(--axis)"/>')
    s.append(f'<text x="{(ml + W - mr) / 2}" y="{Hh - 8}" text-anchor="middle">p50 latency, ms (log scale), '
             f'{H.escape(spec)}, {regime}</text>')
    s.append(f'<text transform="translate(14 {(mt + Hh - mb) / 2}) rotate(-90)" text-anchor="middle">'
             f'{H.escape(metric)}</text>')
    boxes = []        # occupied rectangles: marks and placed labels
    for lat, val, c in pts:
        x, y = X(lat), Y(val)
        boxes.append((x - 7, y - 7, x + 7, y + 7))
    free = lambda b: all(b[2] < o[0] or b[0] > o[2] or b[3] < o[1] or b[1] > o[3] for o in boxes)  # noqa: E731
    marks, labels = [], []
    for lat, val, c in sorted(pts, key=lambda p: (-p[1], p[0])):
        x, y = X(lat), Y(val)
        fill = _fam_var(c["system"])
        tip = (f"{c['label']}: {metric} {val:.3f}, p50 {lat:,.1f} ms, "
               f"{c[regime]['rounds']:.1f} rounds, {c[regime]['kb']:,.0f} KB")
        if SHAPE.get(c["system"]) == "square":
            mark = (f'<rect x="{x - 5:.1f}" y="{y - 5:.1f}" width="10" height="10" rx="2" fill="{fill}" '
                    f'stroke="var(--surface)" stroke-width="2"/>')
        else:
            mark = f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5.5" fill="{fill}" stroke="var(--surface)" stroke-width="2"/>'
        marks.append(f'<g><title>{H.escape(tip)}</title>{mark}'
                     f'<circle cx="{x:.1f}" cy="{y:.1f}" r="12" fill="transparent"/></g>')
        w = 6.0 * len(c["label"]) + 4
        for dx, dy, anchor in ((8, 4, "start"), (8, -8, "start"), (8, 15, "start"), (-8, 4, "end"),
                               (-8, -8, "end"), (-8, 15, "end")):
            x0 = x + dx if anchor == "start" else x + dx - w
            b = (x0, y + dy - 10, x0 + w, y + dy + 2)
            if b[0] < ml or b[2] > W - 2 or b[1] < 0 or b[3] > Hh - mb:
                continue
            if free(b):
                boxes.append(b)
                labels.append(f'<text class="lbl" x="{x + dx:.1f}" y="{y + dy:.1f}" text-anchor="{anchor}">'
                              f'{H.escape(c["label"])}</text>')
                break
    s.extend(marks)
    s.extend(labels)
    s.append("</svg>")
    return "".join(s)


def size_bar(r, W=720):
    parts = sorted(r["sizes"].items(), key=lambda kv: -kv[1])
    tot = sum(v for _, v in parts)
    s = [f'<svg viewBox="0 0 {W} 34" role="img" aria-label="database composition">']
    x = 0.0
    shades = {"fts": "var(--s1)", "dense_graph": "var(--s2)", "dense_ivf": "var(--s2)", "late": "var(--s3)"}
    for k, v in parts:
        w = v / tot * W
        op = "0.55" if k == "dense_ivf" else "1"
        fill = shades.get(k, "var(--axis)")
        s.append(f'<g><title>{H.escape(k)}: {v / 1e6:.2f} MB ({100 * v / tot:.1f} %)</title>'
                 f'<rect x="{x + 1:.1f}" y="2" width="{max(0.5, w - 2):.1f}" height="18" rx="3" fill="{fill}" '
                 f'fill-opacity="{op}"/></g>')
        if w > 60:
            s.append(f'<text x="{x + 5:.1f}" y="32">{H.escape(k)} {v / 1e6:.1f} MB</text>')
        x += w
    s.append("</svg>")
    return "".join(s)


def table(r, spec):
    rows = ["<table><thead><tr><th>Config</th><th>Index MB</th><th>nDCG@10</th><th>Recall@10</th>"
            "<th>AUC@100</th><th>R@10 vs exact</th><th>Rounds</th><th>KB</th>"
            f"<th>p50 ms ({H.escape(spec)})</th><th>p95 ms</th><th>Cold p50 ms</th></tr></thead><tbody>"]
    for c in r["configs"]:
        q = (c.get("quality") or {}).get("all", {})
        w, cd = c.get("warm") or {}, c.get("cold") or {}
        lt = (w.get("latency") or {}).get(spec, {})
        ct = (cd.get("latency") or {}).get(spec, {})
        f = lambda v, d=3: "–" if v is None else f"{v:,.{d}f}"  # noqa: E731
        g = lambda v: "–" if v is None else (f"{v:,.0f}" if v >= 10 else f"{v:.1f}")  # noqa: E731
        rows.append(f"<tr><td>{H.escape(c['label'])}</td><td>{f((c['index_bytes'] or 0) / 1e6, 2)}</td>"
                    f"<td>{f(q.get('ndcg@10'))}</td><td>{f(q.get('recall@10'))}</td><td>{f(q.get('auc@100'))}</td>"
                    f"<td>{f(q.get('r10_vs_exact'))}</td><td>{f(w.get('rounds'), 1)}</td><td>{f(w.get('kb'), 0)}</td>"
                    f"<td>{g(lt.get('p50'))}</td><td>{g(lt.get('p95'))}</td><td>{g(ct.get('p50'))}</td></tr>")
    rows.append("</tbody></table>")
    return "".join(rows)


def html(cfg, results):
    specs = [f"{p},{c}" for c in cfg["concurrency"][::-1] for p in cfg["profiles"]]
    default = "4g,h2"
    out = ["<!doctype html><html lang='en'><head><meta charset='utf-8'>"
           "<meta name='viewport' content='width=device-width,initial-scale=1'>"
           "<title>Benchmark matrix</title><style>", CSS, "</style></head><body>"]
    out.append("<h1>Benchmark matrix</h1>")
    out.append("<p>Retrieval quality against the relevance labels, versus the simulated p50 latency of the same "
               "queries through the WebAssembly SQLite build over HTTP range requests. Each point is one "
               "configuration; hover for rounds and bytes. Colour gives the family (FTS5, dense, late "
               "interaction), squares mark the dense IVF layout. Numbers and method: <code>README.md</code>.</p>")
    out.append("<div class='legend'><span><i class='sw' style='background:var(--s1)'></i>FTS5</span>"
               "<span><i class='sw' style='background:var(--s2)'></i>dense graph</span>"
               "<span><i class='sw sq' style='background:var(--s2)'></i>dense IVF</span>"
               "<span><i class='sw' style='background:var(--s3)'></i>late interaction</span></div>")
    out.append("<p>Network profile: <select id='prof'>" + "".join(
        f"<option{' selected' if s == default else ''}>{H.escape(s)}</option>" for s in specs)
        + "</select> &nbsp; Metric: <select id='met'>" + "".join(
        f"<option{' selected' if m == 'ndcg@10' else ''}>{m}</option>"
        for m in ("ndcg@10", "recall@10", "auc@100")) + "</select></p>")
    for r in results:
        cid = r["corpus"]
        out.append(f"<h2>{H.escape(cid)}</h2>")
        out.append(f"<p>{r['n_docs']:,} documents, {r['n_queries']:,} queries; database {r['db_bytes'] / 1e6:.2f} MB, "
                   "composed of:</p>")
        out.append(size_bar(r))
        for s in specs:
            for m in ("ndcg@10", "recall@10", "auc@100"):
                vis = "" if (s == default and m == "ndcg@10") else " hidden"
                out.append(f"<div class='chart' data-spec='{H.escape(s)}' data-met='{m}'{vis}>"
                           f"<div class='pair'><div>{scatter(r, s, m, 'warm', W=520, Hh=360)}</div>"
                           f"<div>{scatter(r, s, m, 'cold', W=520, Hh=360)}</div></div></div>")
        for s in specs:
            vis = "" if s == default else " hidden"
            out.append(f"<div class='wrap tab' data-spec='{H.escape(s)}'{vis}>{table(r, s)}</div>")
    out.append("""<script>
const P=document.getElementById('prof'),Mt=document.getElementById('met');
function upd(){document.querySelectorAll('.chart').forEach(e=>e.hidden=!(e.dataset.spec===P.value&&e.dataset.met===Mt.value));
document.querySelectorAll('.tab').forEach(e=>e.hidden=e.dataset.spec!==P.value);}
P.onchange=upd;Mt.onchange=upd;
</script>""")
    out.append(f"<p style='font-size:12px'>Generated {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())} by "
               "<code>bench/matrix.py report</code>.</p></body></html>")
    return "".join(out)
