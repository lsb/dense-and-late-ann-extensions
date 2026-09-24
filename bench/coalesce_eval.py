#!/usr/bin/env python3
"""Re-simulate the benchmark matrix's recorded query traces with the VFS's
per-round request budget (wasm/src/httpvfs.c, httpvfs_plan, called through
ctypes from build/native/httpvfs.so, so this is the same code the VFS runs).

Each recorded query (build/matrix/<corpus>.trace.jsonl, one round per VFS
backend call, CPU gaps between rounds as measured) is transformed round by
round and simulated with netsim/simulate.py:

  baseline      the trace as recorded (one request per contiguous range)
  coalesce      ranges merged into at most C requests where the cost model
                (ceil(g/C) * RTT + bytes / bandwidth) says it is faster; RTT
                and bandwidth are the profile's ("oracle", what the VFS's
                estimator converges to) or the fixed defaults (100 ms,
                10 Mbit/s) with --fixed
  multipart     at most C multi-range requests per round, no over-fetch;
                each part costs MULTIPART_PART_BYTES of framing, each
                response MULTIPART_TAIL_BYTES

under <profile>,h1 (browser HTTP/1.1: six requests in service) with C = 6, and
under <profile>,h2 (100 streams) with C = 100 (the VFS default over HTTP/2)
and with C = 6 (HTTP/2 mistaken for HTTP/1.1).

Over-fetched blocks are not fed back into the cache here (the real VFS caches
them, which can save later requests), so coalescing is evaluated slightly
pessimistically; end-to-end runs (bench/coalesce_real.py) include that effect.

  python3 bench/coalesce_eval.py words-10k [--trace-dir DIR] [--out results/coalesce/words-10k.json]
"""
import argparse
import ctypes as C
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "bench"))

from netsim import netmodel as nm  # noqa: E402
from netsim import simulate as sim  # noqa: E402

BS = 4096
MAX_BLOCKS = 512           # the VFS allows cache/2 blocks per round (4 MiB cache)
MULTIPART_PART_BYTES = 100   # "\r\n--boundary\r\nContent-Type: ...\r\nContent-Range: bytes a-b/N\r\n\r\n"
MULTIPART_TAIL_BYTES = 30
DEFAULT_RTT, DEFAULT_KBPS = 100.0, 10000.0

# One representative configuration per system and corpus (the headline rows
# of results/matrix/README.md).
SYSTEMS = {
    "words": {"FTS5": "fts-bm25", "dense graph": "graph-ef64", "dense IVF": "ivf-np64", "late warp": "warp-np8-rr64"},
    "llm": {"FTS5": "fts-bm25-or", "dense graph": "graph-ef64", "dense IVF": "ivf-np128", "late warp": "warp-np8"},
}
PROFILES = ["4g", "lte", "slow-4g", "wifi"]

_LIB = None


def lib():
    global _LIB
    if _LIB is None:
        so = os.environ.get("HTTPVFS_SO", str(REPO / "build" / "native" / "httpvfs.so"))
        L = C.CDLL(so)
        f = L.httpvfs_plan
        f.restype = C.c_int
        f.argtypes = [C.c_int, C.POINTER(C.c_int64), C.POINTER(C.c_int64), C.c_int, C.c_int,
                      C.c_double, C.c_double, C.c_int64, C.c_int, C.c_int, C.POINTER(C.c_int)]
        _LIB = f
    return _LIB


def plan(ranges, max_req, rtt_ms, kbps, multipart, max_parts=100):
    """ranges: sorted [(first_block, end_block)]; returns grp per range."""
    m = len(ranges)
    rs = (C.c_int64 * m)(*[a for a, _ in ranges])
    re = (C.c_int64 * m)(*[b for _, b in ranges])
    grp = (C.c_int * m)()
    lib()(m, rs, re, BS, max_req, rtt_ms, kbps / 8.0, MAX_BLOCKS, 1 if multipart else 0, max_parts, grp)
    return list(grp)


def transform(trace, mode, max_req, rtt_ms=DEFAULT_RTT, kbps=DEFAULT_KBPS):
    """A new trace with every round planned under a budget of max_req."""
    if mode == "baseline":
        return trace, 0
    reads, over = [], 0
    for rnd, rs in trace.by_round():
        rs = sorted(rs, key=lambda r: r.offset)
        # the recorded requests are block-aligned contiguous ranges
        ranges = [(r.offset // BS, (r.offset + r.length + BS - 1) // BS) for r in rs]
        grp = plan(ranges, max_req, rtt_ms, kbps, mode == "multipart")
        groups = {}
        for r, g in zip(rs, grp):
            groups.setdefault(g, []).append(r)
        for g, members in sorted(groups.items()):
            if mode == "multipart":
                n = sum(r.length for r in members)
                if len(members) > 1:
                    n += MULTIPART_PART_BYTES * len(members) + MULTIPART_TAIL_BYTES
                reads.append(sim.Read(members[0].offset, n, rnd, 0.0, None, len(reads)))
            else:
                a = members[0].offset
                b = max(r.offset + r.length for r in members)
                over += (b - a) - sum(r.length for r in members)
                reads.append(sim.Read(a, b - a, rnd, 0.0, None, len(reads)))
    return sim.Trace(reads, dict(trace.cpu_ms), trace.tail_cpu_ms, dict(trace.meta)), over


def record_trace(rec):
    """One round per backend call, CPU gaps from the log (bench/matrix.py)."""
    import matrix
    return matrix.record_trace(rec)


def strategies(profile_name):
    p = nm.preset(profile_name)
    rtt, kbps = p.latency_ms, p.link_kbps
    return [
        # (label, network spec, mode, C, rtt, kbps)
        ("h1 baseline", f"{profile_name},h1", "baseline", 0, 0, 0),
        ("h1 coalesce C=6", f"{profile_name},h1", "coalesce", 6, rtt, kbps),
        ("h1 coalesce C=6 fixed", f"{profile_name},h1", "coalesce", 6, DEFAULT_RTT, DEFAULT_KBPS),
        ("h1 multipart C=6", f"{profile_name},h1", "multipart", 6, rtt, kbps),
        ("h2 baseline", f"{profile_name},h2", "baseline", 0, 0, 0),
        ("h2 coalesce C=100", f"{profile_name},h2", "coalesce", 100, rtt, kbps),
        ("h2 coalesce C=6", f"{profile_name},h2", "coalesce", 6, rtt, kbps),
        ("h2 multipart C=6", f"{profile_name},h2", "multipart", 6, rtt, kbps),
    ]


def run_record(args):
    rec, profiles = args
    tr = record_trace(rec)
    out = {}
    for pn in profiles:
        for label, spec, mode, c, rtt, kbps in strategies(pn):
            t2, over = transform(tr, mode, c, rtt, kbps)
            res = sim.simulate(t2, nm.preset(spec), seed=rec["qi"])
            out[f"{pn}|{label}"] = (res.total_ms, len(t2.reads), t2.total_bytes, over)
    return {"config": rec["config"], "regime": rec["regime"], "qi": rec["qi"],
            "rounds": len(tr.rounds), "requests": len(tr.reads), "bytes": tr.total_bytes, "res": out}


def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return math.nan
    k = (len(xs) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("corpus")
    ap.add_argument("--trace-dir", default=str(REPO / "build" / "matrix"))
    ap.add_argument("--warm", type=int, default=500, help="warm queries per configuration (evenly sampled)")
    ap.add_argument("--cold", type=int, default=200, help="cold queries per configuration")
    ap.add_argument("--profiles", default=",".join(PROFILES))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    fam = "llm" if args.corpus.startswith("llm") else "words"
    systems = SYSTEMS[fam]
    want = {c: s for s, c in systems.items()}
    profiles = args.profiles.split(",")
    recs = {}
    with open(Path(args.trace_dir) / f"{args.corpus}.trace.jsonl") as fh:
        for line in fh:
            if not any(f'"config":"{c}"' in line[:200] for c in want):
                continue
            r = json.loads(line)
            if r.get("error"):
                continue
            recs.setdefault((r["config"], r["regime"]), []).append(r)
    jobs = []
    for (c, regime), rs in recs.items():
        n = args.warm if regime == "warm" else args.cold
        step = max(1, len(rs) // n)
        jobs += [(r, profiles) for r in rs[::step][:n]]
    t = time.time()
    with ProcessPoolExecutor(args.workers) as ex:
        results = list(ex.map(run_record, jobs, chunksize=16))
    print(f"[{args.corpus}] {len(results)} queries x {len(profiles)} profiles x 8 strategies in {time.time() - t:.0f} s",
          file=sys.stderr)
    # Aggregate: p50 / p95 per system, regime, profile, strategy.
    agg = {}
    for r in results:
        sysname = want[r["config"]]
        for key, (ms, nreq, nbytes, over) in r["res"].items():
            pn, label = key.split("|")
            a = agg.setdefault((sysname, r["regime"], pn, label), {"ms": [], "req": [], "bytes": [], "over": []})
            a["ms"].append(ms)
            a["req"].append(nreq)
            a["bytes"].append(nbytes)
            a["over"].append(over)
    rows = []
    for (sysname, regime, pn, label), a in sorted(agg.items()):
        n = len(a["ms"])
        rows.append({"system": sysname, "config": systems[sysname], "regime": regime, "profile": pn, "strategy": label,
                     "n": n, "p50": pct(a["ms"], 0.5), "p95": pct(a["ms"], 0.95), "mean": sum(a["ms"]) / n,
                     "requests": sum(a["req"]) / n, "kb": sum(a["bytes"]) / n / 1024,
                     "overfetch_kb": sum(a["over"]) / n / 1024})
    out = {"corpus": args.corpus, "systems": systems, "profiles": profiles, "rows": rows,
           "notes": {"max_blocks": MAX_BLOCKS, "multipart_part_bytes": MULTIPART_PART_BYTES,
                     "fixed_defaults": [DEFAULT_RTT, DEFAULT_KBPS]}}
    path = Path(args.out or REPO / "results" / "coalesce" / f"{args.corpus}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    print(f"-> {path}", file=sys.stderr)
    print_table(out)


def print_table(out, regime="warm"):
    rows = [r for r in out["rows"] if r["regime"] == regime]
    labels = [s[0] for s in strategies("4g")]
    for pn in out["profiles"]:
        print(f"\n{out['corpus']} {regime} {pn}: p50 / p95 ms")
        print("system".ljust(12) + "".join(l.rjust(24) for l in labels))
        for sysname in out["systems"]:
            cells = []
            for l in labels:
                r = next((x for x in rows if x["system"] == sysname and x["profile"] == pn and x["strategy"] == l), None)
                cells.append(f"{r['p50']:.0f} / {r['p95']:.0f}" if r else "-")
            print(sysname.ljust(12) + "".join(c.rjust(24) for c in cells))


if __name__ == "__main__":
    main()
