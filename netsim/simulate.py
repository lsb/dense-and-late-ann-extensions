#!/usr/bin/env python3
"""Discrete-event simulator for range-request traces over a network profile.

A trace is a list of byte-range reads grouped into dependent rounds: every
read of round k is issued (optionally staggered by ``t_issue``) once all reads
of round k-1 have completed and the round's CPU time has elapsed.  The network
model (latency sampling, FIFO concurrency limit, lazily opened connections,
fluid max-min sharing of the link, optional slow start) is imported from
``netmodel`` and is the same code the range server uses.

Usage:
    python3 netsim/simulate.py TRACE.json --preset 4g
    python3 netsim/simulate.py TRACE.json --preset lte-poor --seeds 200
    python3 netsim/simulate.py TRACE.json --sweep            # all presets
    python3 netsim/simulate.py TRACE.json --sweep --preset-mod h1
"""

from __future__ import annotations

import argparse
import heapq
import itertools
import json
import math
import statistics
import sys
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

try:
    from . import netmodel as nm
except ImportError:  # run as a script
    import netmodel as nm  # type: ignore


# --------------------------------------------------------------------------
# Traces
# --------------------------------------------------------------------------


@dataclass
class Read:
    offset: int
    length: int
    round: int = 0
    t_issue_ms: float = 0.0  # delay after the round starts
    file: str | None = None
    idx: int = 0


@dataclass
class Trace:
    reads: list[Read]
    cpu_ms: dict[int, float] = field(default_factory=dict)  # before round r
    tail_cpu_ms: float = 0.0  # after the last round
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def rounds(self) -> list[int]:
        return sorted({r.round for r in self.reads} | set(self.cpu_ms))

    @property
    def total_bytes(self) -> int:
        return sum(r.length for r in self.reads)

    def by_round(self) -> list[tuple[int, list[Read]]]:
        out: dict[int, list[Read]] = {r: [] for r in self.rounds}
        for rd in self.reads:
            out[rd.round].append(rd)
        return [(r, out[r]) for r in sorted(out)]

    def merged(self, gap: int) -> "Trace":
        """Coalesce reads of the same round and file that are within ``gap``
        bytes of each other into one range (issued at the earliest t_issue)."""
        new: list[Read] = []
        for rnd, reads in self.by_round():
            keyf = lambda r: (r.file or "", r.offset)
            cur: Read | None = None
            for r in sorted(reads, key=keyf):
                if (cur is not None and (cur.file or "") == (r.file or "")
                        and r.offset <= cur.offset + cur.length + gap):
                    end = max(cur.offset + cur.length, r.offset + r.length)
                    cur.length = end - cur.offset
                    cur.t_issue_ms = min(cur.t_issue_ms, r.t_issue_ms)
                else:
                    if cur is not None:
                        new.append(cur)
                    cur = Read(r.offset, r.length, rnd, r.t_issue_ms, r.file)
            if cur is not None:
                new.append(cur)
        for i, r in enumerate(new):
            r.idx = i
        return Trace(new, dict(self.cpu_ms), self.tail_cpu_ms, dict(self.meta))


def _read_from(obj: dict[str, Any], i: int, default_file: str | None) -> Read:
    return Read(
        offset=int(obj["offset"]),
        length=int(obj["length"]),
        round=int(obj.get("round", 0)),
        t_issue_ms=float(obj.get("t_issue", obj.get("t_issue_ms", 0.0)) or 0.0),
        file=obj.get("file", default_file),
        idx=i,
    )


def trace_from_obj(obj: Any) -> Trace:
    """Accepts the documented object form, a bare list of reads, or a list of
    rounds (each a list of reads)."""
    if isinstance(obj, list):
        if obj and isinstance(obj[0], list):
            obj = {"reads": [dict(r, round=k) for k, rnd in enumerate(obj) for r in rnd]}
        else:
            obj = {"reads": obj}
    default_file = obj.get("file")
    reads = [_read_from(r, i, default_file) for i, r in enumerate(obj.get("reads", []))]
    cpu: dict[int, float] = {}
    for i, r in enumerate(obj.get("rounds", []) or []):
        rid = int(r.get("round", i))
        cpu[rid] = float(r.get("cpu_ms", 0.0))
    meta = {k: v for k, v in obj.items() if k not in ("reads", "rounds")}
    return Trace(reads, cpu, float(obj.get("tail_cpu_ms", 0.0)), meta)


def load_trace(path: str) -> Trace:
    with open(path) as fh:
        text = fh.read()
    try:
        return trace_from_obj(json.loads(text))
    except json.JSONDecodeError:
        # JSON lines: each line a read, or {"round": r, "cpu_ms": x} for CPU.
        reads, rounds = [], []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            (reads if "offset" in o else rounds).append(o)
        return trace_from_obj({"reads": reads, "rounds": rounds})


def make_trace(rounds: Sequence[Sequence[int | tuple[int, int]]],
               cpu_ms: Sequence[float] | float = 0.0) -> Trace:
    """Convenience: ``make_trace([[4096]*3, [(0, 65536)]])`` -> Trace.  An int
    is a length (offsets are assigned sequentially); a tuple is (offset, length)."""
    reads, off = [], 0
    for k, rnd in enumerate(rounds):
        for item in rnd:
            if isinstance(item, tuple):
                o, n = item
            else:
                o, n = off, int(item)
            off = o + n
            reads.append(Read(o, n, k, 0.0, None, len(reads)))
    if isinstance(cpu_ms, (int, float)):
        cpu = {k: float(cpu_ms) for k in range(1, len(rounds))} if cpu_ms else {}
    else:
        cpu = {k: float(c) for k, c in enumerate(cpu_ms)}
    return Trace(reads, cpu)


# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------


@dataclass
class ReqResult:
    read: Read
    t_arrive: float
    t_start: float
    t_first_byte: float
    t_finish: float
    latency: float
    conn: int
    new_conn: bool


@dataclass
class SimResult:
    total_s: float
    round_end_s: list[float]
    requests: list[ReqResult]
    profile: nm.Profile
    seed: Any

    @property
    def total_ms(self) -> float:
        return self.total_s * 1000.0

    def summary(self) -> dict[str, Any]:
        return {
            "total_ms": round(self.total_ms, 3),
            "rounds": len(self.round_end_s),
            "requests": len(self.requests),
            "bytes": sum(r.read.length for r in self.requests),
            "connections": len({r.conn for r in self.requests}),
            "max_queue_ms": round(max((r.t_start - r.t_arrive for r in self.requests), default=0) * 1000, 3),
        }


def simulate(trace: Trace, profile: nm.Profile, seed: Any = 0,
             body_overhead: int = 0) -> SimResult:
    """Simulate ``trace`` under ``profile``.  ``body_overhead`` adds bytes to
    each response (e.g. to account for headers); the server does not count
    headers against the bandwidth model, so the default is 0."""
    sampler = nm.RequestSampler(profile, seed)
    link = nm.FluidLink.for_profile(profile, seed=seed, t0=0.0)
    pool = nm.SlotPool(profile.max_concurrent)
    t = 0.0
    results: dict[int, ReqResult] = {}
    round_ends: list[float] = []
    seq = itertools.count()

    for rnd, reads in trace.by_round():
        t += trace.cpu_ms.get(rnd, 0.0) / 1000.0
        round_start = t
        heap: list[tuple[float, int, str, Any]] = []
        for rd in sorted(reads, key=lambda r: (r.t_issue_ms, r.idx)):
            heapq.heappush(heap, (round_start + rd.t_issue_ms / 1000.0, next(seq), "arrive", rd))
        outstanding = len(reads)
        pending: dict[int, dict[str, Any]] = {}

        def start_service(started: list, now: float) -> None:
            for rd, cid, new in started:
                lat = sampler.latency_s()
                cap = sampler.request_cap_Bps()
                pending[rd.idx].update(t_start=now, conn=cid, new=new, lat=lat, cap=cap)
                fb = now + lat + (profile.connect_ms / 1000.0 if new else 0.0)
                heapq.heappush(heap, (fb, next(seq), "first_byte", rd))

        while outstanding:
            t_heap = heap[0][0] if heap else math.inf
            t_link = link.next_event_time()
            if t_link < t_heap:
                link.advance(t_link)
                now = t_link
            else:
                now, _, kind, rd = heapq.heappop(heap)
                link.advance(now)
                if kind == "arrive":
                    pending[rd.idx] = {"t_arrive": now}
                    start_service(pool.arrive(rd), now)
                else:  # first byte: body starts flowing
                    st = pending[rd.idx]
                    st["t_fb"] = now
                    ss = profile.init_cwnd_bytes if profile.slow_start else 0
                    link.add(nm.Flow(rd.idx, rd.length + body_overhead, st["cap"],
                                     ss, profile.ss_rtt_s, user=rd), now)
            for f in link.pop_completed():
                rd = f.user
                st = pending.pop(rd.idx)
                results[rd.idx] = ReqResult(rd, st["t_arrive"], st["t_start"], st["t_fb"],
                                            f.t_done, st["lat"], st["conn"], st["new"])
                outstanding -= 1
                start_service(pool.release(st["conn"]), f.t_done)
            t = max(t, now)
        t = max([t] + [results[r.idx].t_finish for r in reads])
        round_ends.append(t)
    t += trace.tail_cpu_ms / 1000.0
    reqs = [results[r.idx] for r in trace.reads]
    return SimResult(t, round_ends, reqs, profile, seed)


def simulate_many(trace: Trace, profile: nm.Profile, seeds: Iterable[Any]) -> dict[str, Any]:
    totals = sorted(simulate(trace, profile, s).total_ms for s in seeds)
    return distribution(totals)


def _pct(xs: Sequence[float], q: float) -> float:
    if not xs:
        return math.nan
    k = (len(xs) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def distribution(xs: Sequence[float]) -> dict[str, Any]:
    xs = sorted(xs)
    return {
        "n": len(xs),
        "mean": statistics.fmean(xs),
        "stdev": statistics.stdev(xs) if len(xs) > 1 else 0.0,
        "min": xs[0], "p50": _pct(xs, 0.5), "p90": _pct(xs, 0.9),
        "p99": _pct(xs, 0.99), "max": xs[-1],
    }


def lower_bound_ms(trace: Trace, profile: nm.Profile) -> float:
    """Closed-form estimate for fixed latency, no queueing or slow start:
    sum over rounds of cpu + latency + round bytes / min(link, n * cap)."""
    total = trace.tail_cpu_ms
    for rnd, reads in trace.by_round():
        total += trace.cpu_ms.get(rnd, 0.0)
        if not reads:
            continue
        nbytes = sum(r.length for r in reads)
        bw = min(profile.link_Bps, len(reads) * profile.request_Bps)
        total += profile.latency_ms + (nbytes / bw * 1000.0 if bw < math.inf else 0.0)
    return total


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace", help="trace file (JSON or JSON lines), see NOTES.md")
    nm.add_profile_args(ap)
    ap.add_argument("--seed", type=int, default=0, help="first seed")
    ap.add_argument("--seeds", type=int, default=1, help="number of seeds (distribution)")
    ap.add_argument("--sweep", action="store_true", help="run every network preset")
    ap.add_argument("--preset-mod", default="", help="modifiers appended to each preset in --sweep, e.g. h1,jitter")
    ap.add_argument("--merge-gap", type=int, default=-1, help="coalesce reads within a round closer than this many bytes")
    ap.add_argument("--per-request", action="store_true", help="print per-request timings (single seed)")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args(argv)

    trace = load_trace(args.trace)
    if args.merge_gap >= 0:
        trace = trace.merged(args.merge_gap)
    names = [n for n in nm.PRESETS] if args.sweep else [None]
    rows = []
    for name in names:
        if name is None:
            prof = nm.profile_from_args(args)
        else:
            spec = name + ("," + args.preset_mod if args.preset_mod else "")
            args_preset, args.preset = args.preset, spec
            prof = nm.profile_from_args(args)
            args.preset = args_preset
        seeds = range(args.seed, args.seed + args.seeds)
        dist = simulate_many(trace, prof, seeds)
        rows.append({"profile": prof.name, "describe": nm.describe(prof), **dist})
        if args.per_request and name is None:
            res = simulate(trace, prof, args.seed)
            for r in res.requests:
                print(json.dumps({
                    "idx": r.read.idx, "round": r.read.round, "offset": r.read.offset,
                    "length": r.read.length, "conn": r.conn,
                    "arrive_ms": round(r.t_arrive * 1e3, 3), "start_ms": round(r.t_start * 1e3, 3),
                    "first_byte_ms": round(r.t_first_byte * 1e3, 3), "finish_ms": round(r.t_finish * 1e3, 3),
                }))
    if args.json:
        print(json.dumps({"trace": {"reads": len(trace.reads), "rounds": len(trace.rounds),
                                    "bytes": trace.total_bytes}, "results": rows}, indent=1))
        return 0
    print(f"trace: {len(trace.reads)} reads, {len(trace.rounds)} rounds, {trace.total_bytes} bytes")
    hdr = f"{'profile':<22}{'mean ms':>11}{'p50':>11}{'p90':>11}{'p99':>11}  model"
    print(hdr)
    for r in rows:
        print(f"{r['profile']:<22}{r['mean']:>11.1f}{r['p50']:>11.1f}{r['p90']:>11.1f}{r['p99']:>11.1f}  {r['describe']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
