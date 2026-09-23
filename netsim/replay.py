#!/usr/bin/env python3
"""Replay a read trace against a real HTTP server with Range requests.

The client behaves like a browser: requests of one round are issued in
parallel (after their optional ``t_issue`` stagger), a round starts only when
the previous one has completed and its CPU time has elapsed, and at most
``max_conns`` keep-alive connections are used (extra requests wait FIFO).
This is the real-time counterpart of ``simulate.simulate``.

    python3 netsim/replay.py TRACE.json --url http://127.0.0.1:8000/index.db
    python3 netsim/replay.py TRACE.json --url ... --max-conns 6 --repeat 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import urllib.parse
from collections import deque
from dataclasses import dataclass
from typing import Any

try:
    from . import simulate as sim
except ImportError:  # run as a script
    import simulate as sim  # type: ignore


class HTTPError(Exception):
    pass


class Conn:
    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        self.reader, self.writer = reader, writer

    async def request(self, host: str, path: str, headers: dict[str, str] | None = None,
                      method: str = "GET", body: bytes = b""
                      ) -> tuple[int, dict[str, str], bytes, float]:
        """Returns status, headers, body and the time the header block arrived."""
        lines = [f"{method} {path} HTTP/1.1", f"Host: {host}"]
        for k, v in (headers or {}).items():
            lines.append(f"{k}: {v}")
        if body:
            lines.append(f"Content-Length: {len(body)}")
        self.writer.write(("\r\n".join(lines) + "\r\n\r\n").encode("latin-1") + body)
        head = await self.reader.readuntil(b"\r\n\r\n")
        t_head = time.monotonic()
        hl = head.decode("latin-1").split("\r\n")
        status = int(hl[0].split(" ", 2)[1])
        hdrs: dict[str, str] = {}
        for ln in hl[1:]:
            if ln:
                k, v = ln.split(":", 1)
                hdrs[k.strip().lower()] = v.strip()
        n = int(hdrs.get("content-length", "0"))
        data = b"" if method == "HEAD" or status in (204, 304) else await self.reader.readexactly(n)
        return status, hdrs, data, t_head

    def close(self) -> None:
        try:
            self.writer.close()
        except Exception:
            pass


class Client:
    """Keep-alive connection pool with a FIFO limit (0 = unlimited)."""

    def __init__(self, url: str, max_conns: int = 0):
        u = urllib.parse.urlsplit(url)
        self.host, self.port = u.hostname or "127.0.0.1", u.port or 80
        self.path = u.path or "/"
        self.max_conns = max_conns
        self.idle: deque[Conn] = deque()
        self.n = 0
        self.waiters: deque[asyncio.Future] = deque()
        self.opened = 0

    async def _acquire(self) -> Conn:
        if self.idle:
            return self.idle.pop()
        if self.max_conns <= 0 or self.n < self.max_conns:
            self.n += 1
            self.opened += 1
            r, w = await asyncio.open_connection(self.host, self.port, limit=1 << 20)
            return Conn(r, w)
        fut = asyncio.get_running_loop().create_future()
        self.waiters.append(fut)
        return await fut

    def _release(self, c: Conn, ok: bool) -> None:
        if not ok:
            c.close()
            self.n -= 1
            if self.waiters:  # let a waiter open a fresh connection
                w = self.waiters.popleft()
                self.n += 1
                self.opened += 1

                async def _open() -> None:
                    r, wr = await asyncio.open_connection(self.host, self.port, limit=1 << 20)
                    w.set_result(Conn(r, wr))
                asyncio.ensure_future(_open())
            return
        while self.waiters:
            w = self.waiters.popleft()
            if not w.done():
                w.set_result(c)
                return
        self.idle.append(c)

    async def get(self, path: str | None = None, headers: dict[str, str] | None = None,
                  method: str = "GET", body: bytes = b"") -> tuple[int, dict[str, str], bytes, float]:
        c = await self._acquire()
        ok = False
        try:
            res = await c.request(f"{self.host}:{self.port}", path or self.path, headers, method, body)
            ok = res[1].get("connection", "").lower() != "close"
            return res
        finally:
            self._release(c, ok)

    async def range(self, offset: int, length: int, path: str | None = None
                    ) -> tuple[int, dict[str, str], bytes, float]:
        return await self.get(path, {"Range": f"bytes={offset}-{offset + length - 1}"})

    async def close(self) -> None:
        for c in self.idle:
            c.close()
        self.idle.clear()


@dataclass
class ReplayResult:
    total_s: float
    round_end_s: list[float]
    requests: list[dict[str, Any]]

    @property
    def total_ms(self) -> float:
        return self.total_s * 1000.0


async def replay(trace: sim.Trace, url: str, max_conns: int = 0,
                 client: Client | None = None, check: bool = True) -> ReplayResult:
    """Run ``trace`` against ``url`` (the file; a read's ``file`` field, if
    set, replaces the last path component)."""
    own = client is None
    cl = client or Client(url, max_conns)
    base = urllib.parse.urlsplit(url).path
    t0 = time.monotonic()
    rounds: list[float] = []
    reqs: list[dict[str, Any]] = []

    async def one(rd: sim.Read, t_round: float) -> None:
        if rd.t_issue_ms:
            await asyncio.sleep(max(0.0, t_round + rd.t_issue_ms / 1000 - time.monotonic()))
        path = base
        if rd.file:
            path = base.rsplit("/", 1)[0] + "/" + urllib.parse.quote(rd.file)
        t_issue = time.monotonic()
        status, hdrs, data, t_head = await cl.range(rd.offset, rd.length, path)
        t_done = time.monotonic()
        if check and (status != 206 or len(data) != rd.length):
            raise HTTPError(f"read {rd.idx}: status {status}, {len(data)} bytes, wanted {rd.length}")
        reqs.append({"idx": rd.idx, "round": rd.round, "offset": rd.offset, "length": rd.length,
                     "issue_ms": (t_issue - t0) * 1e3, "first_byte_ms": (t_head - t0) * 1e3,
                     "finish_ms": (t_done - t0) * 1e3})

    try:
        for rnd, reads in trace.by_round():
            cpu = trace.cpu_ms.get(rnd, 0.0) / 1000.0
            if cpu:
                await asyncio.sleep(cpu)
            t_round = time.monotonic()
            await asyncio.gather(*(one(rd, t_round) for rd in reads))
            rounds.append(time.monotonic() - t0)
        if trace.tail_cpu_ms:
            await asyncio.sleep(trace.tail_cpu_ms / 1000.0)
    finally:
        if own:
            await cl.close()
    reqs.sort(key=lambda r: r["idx"])
    return ReplayResult(time.monotonic() - t0, rounds, reqs)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace")
    ap.add_argument("--url", required=True, help="URL of the file the trace reads")
    ap.add_argument("--max-conns", type=int, default=6, help="connection limit (0 = unlimited)")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--merge-gap", type=int, default=-1)
    args = ap.parse_args(argv)
    trace = sim.load_trace(args.trace)
    if args.merge_gap >= 0:
        trace = trace.merged(args.merge_gap)
    for i in range(args.repeat):
        res = asyncio.run(replay(trace, args.url, args.max_conns))
        print(json.dumps({"run": i, "total_ms": round(res.total_ms, 3),
                          "round_end_ms": [round(x * 1e3, 3) for x in res.round_end_s]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
