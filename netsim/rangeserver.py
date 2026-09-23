#!/usr/bin/env python3
"""Static-file HTTP/1.1 server with byte ranges, CORS and network shaping.

Serves files from a directory with GET/HEAD/OPTIONS, single and multiple
``Range`` requests (206, 416, multipart/byteranges), ETag/Last-Modified,
keep-alive, CORS, and optional COOP/COEP (``--isolate``).  Every response is
delayed and paced according to a network profile (see ``netmodel.py``), the
same model that ``simulate.py`` evaluates in virtual time.

    python3 netsim/rangeserver.py --dir DIR --port 8000 --preset 4g,h1
    python3 netsim/rangeserver.py --dir DIR --preset none --latency-ms 80 --link-kbps 20000

Control endpoints (never shaped):
    GET  /__netsim/profile          current profile as JSON
    POST /__netsim/profile          {"preset": "3g", "latency_ms": 250, "seed": 1}
                                    (without "preset", fields patch the current profile)
    GET  /__netsim/presets          preset and modifier definitions
    GET  /__netsim/log?since=N      in-memory request log records with id > N
    POST /__netsim/log/clear        clear the in-memory log
    GET  /__netsim/stats            counters

In-process use:
    async with NetSimServer(root, preset("4g")) as srv: ...srv.url...
    with ServerThread(root, preset("4g")) as srv: ...srv.url...
"""

from __future__ import annotations

import argparse
import asyncio
import email.utils
import itertools
import json
import mimetypes
import os
import sys
import threading
import time
import urllib.parse
from collections import deque
from typing import Any

try:
    from . import netmodel as nm
except ImportError:  # run as a script
    import netmodel as nm  # type: ignore

SERVER_NAME = "netsim-rangeserver/1"
CONTROL_PREFIX = "/__netsim/"
MAX_HEADER_BYTES = 64 * 1024
MAX_RANGES = 256
WRITE_CHUNK = 1 << 20
MIN_SHAPED_CHUNK = 16 * 1024

mimetypes.add_type("application/wasm", ".wasm")
mimetypes.add_type("text/javascript", ".mjs")
for _ext in (".db", ".sqlite", ".sqlite3", ".bin", ".idx"):
    mimetypes.add_type("application/octet-stream", _ext)

REASONS = {
    200: "OK", 204: "No Content", 206: "Partial Content", 304: "Not Modified",
    400: "Bad Request", 403: "Forbidden", 404: "Not Found", 405: "Method Not Allowed",
    413: "Payload Too Large", 416: "Range Not Satisfiable", 500: "Internal Server Error",
}


# --------------------------------------------------------------------------
# Range parsing
# --------------------------------------------------------------------------


def parse_range(header: str | None, size: int) -> list[tuple[int, int]] | None:
    """Parse a Range header against a representation of ``size`` bytes.

    Returns None when the header is absent or invalid (serve 200), an empty
    list when no range is satisfiable (serve 416), or a list of inclusive
    (first, last) pairs."""
    if not header:
        return None
    h = header.strip()
    if not h.lower().startswith("bytes="):
        return None
    out: list[tuple[int, int]] = []
    specs = h[6:].split(",")
    if len(specs) > MAX_RANGES:
        return None
    for spec in specs:
        spec = spec.strip()
        if not spec:
            continue
        if "-" not in spec:
            return None
        a, b = (x.strip() for x in spec.split("-", 1))
        if a == "":
            if not b.isdigit():
                return None
            n = int(b)
            if n == 0 or size == 0:
                continue
            out.append((max(0, size - n), size - 1))
            continue
        if not a.isdigit() or (b and not b.isdigit()):
            return None
        first = int(a)
        if b and int(b) < first:
            return None  # syntactically invalid: ignore the header
        if first >= size:
            continue  # unsatisfiable
        last = int(b) if b else size - 1
        out.append((first, min(last, size - 1)))
    return out


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


class FIFOSlots:
    """Admission control: at most ``limit`` requests in service, FIFO queue."""

    def __init__(self, limit: int = 0):
        self.limit = limit
        self.busy = 0
        self.waiters: deque[asyncio.Future] = deque()

    def _free(self) -> bool:
        return self.limit <= 0 or self.busy < self.limit

    async def acquire(self) -> None:
        if self._free() and not self.waiters:
            self.busy += 1
            return
        fut = asyncio.get_running_loop().create_future()
        self.waiters.append(fut)
        try:
            await fut
        except asyncio.CancelledError:
            if fut.done() and not fut.cancelled():
                self.release()
            else:
                try:
                    self.waiters.remove(fut)
                except ValueError:
                    pass
            raise

    def release(self) -> None:
        self.busy -= 1
        self.wake()

    def wake(self) -> None:
        while self.waiters and self._free():
            fut = self.waiters.popleft()
            if not fut.done():
                self.busy += 1
                fut.set_result(None)


class Body:
    """A response body made of in-memory bytes and file segments."""

    __slots__ = ("parts", "length")

    def __init__(self) -> None:
        self.parts: list[tuple[int, Any, int, int]] = []  # (start, bytes|fd, off, len)
        self.length = 0

    def add_bytes(self, b: bytes) -> None:
        if b:
            self.parts.append((self.length, b, 0, len(b)))
            self.length += len(b)

    def add_file(self, fd: int, off: int, n: int) -> None:
        if n > 0:
            self.parts.append((self.length, fd, off, n))
            self.length += n

    def read(self, pos: int, n: int) -> bytes:
        """Bytes [pos, pos+n) of the body."""
        end = pos + n
        chunks = []
        for start, src, off, ln in self.parts:
            s, e = max(pos, start), min(end, start + ln)
            if s >= e:
                continue
            if isinstance(src, bytes):
                chunks.append(src[s - start:e - start])
            else:
                chunks.append(os.pread(src, e - s, off + (s - start)))
        return b"".join(chunks)


class FileCache:
    """Keeps file descriptors open between requests; reopens on change."""

    def __init__(self, max_open: int = 256):
        self.max_open = max_open
        self.fds: dict[str, tuple[int, int, int]] = {}

    def open(self, path: str, st: os.stat_result) -> int:
        ent = self.fds.get(path)
        if ent and ent[1] == st.st_mtime_ns and ent[2] == st.st_size:
            return ent[0]
        if ent:
            os.close(ent[0])
        if len(self.fds) >= self.max_open:
            _, (fd, _, _) = self.fds.popitem()
            os.close(fd)
        fd = os.open(path, os.O_RDONLY)
        self.fds[path] = (fd, st.st_mtime_ns, st.st_size)
        return fd

    def close(self) -> None:
        for fd, _, _ in self.fds.values():
            os.close(fd)
        self.fds.clear()


def _http_date(ts: float | None = None) -> str:
    return email.utils.formatdate(ts, usegmt=True)


# --------------------------------------------------------------------------
# Server
# --------------------------------------------------------------------------


class NetSimServer:
    def __init__(self, root: str, profile: nm.Profile | None = None, *,
                 host: str = "127.0.0.1", port: int = 0, seed: Any = 0,
                 log_path: str | None = None, isolate: bool = False,
                 cache_control: str = "no-store", log_memory: int = 100_000,
                 quiet: bool = True):
        self.root = os.path.realpath(root)
        self.host, self.port = host, port
        self.isolate = isolate
        self.cache_control = cache_control
        self.quiet = quiet
        self.log_path = log_path
        self._log_fh = None
        self.log: deque[dict[str, Any]] = deque(maxlen=log_memory)
        self.files = FileCache()
        self.server: asyncio.base_events.Server | None = None
        self._conn_ids = itertools.count()
        self._req_ids = itertools.count(1)
        self._waiters: set[asyncio.Future] = set()
        self.stats = {"requests": 0, "bytes": 0, "connections": 0}
        self.t0 = time.monotonic()
        self.wall0 = time.time()
        self.seed = seed
        self.slots = FIFOSlots()
        self.link = nm.FluidLink(t0=0.0)
        self.link.on_change = self._wake_writers
        self.set_profile(profile or nm.Profile(name="none"), seed)

    # -- profile -------------------------------------------------------------

    def set_profile(self, profile: nm.Profile, seed: Any = None) -> None:
        """Switch profile at runtime.  In-flight responses keep their sampled
        latency and per-request cap; the link capacity changes immediately."""
        profile.validate()
        if seed is not None:
            self.seed = seed
        self.profile = profile
        self.sampler = nm.RequestSampler(profile, self.seed)
        self.link.reconfigure(profile, seed=self.seed)
        self.slots.limit = profile.max_concurrent
        try:
            asyncio.get_running_loop()
            self.slots.wake()
            self._wake_writers()
        except RuntimeError:
            pass

    def clock(self) -> float:
        return time.monotonic() - self.t0

    # -- lifecycle -------------------------------------------------------------

    async def start(self) -> "NetSimServer":
        if self.log_path:
            self._log_fh = open(self.log_path, "a", buffering=1)
        self.server = await asyncio.start_server(
            self._handle, self.host, self.port, limit=MAX_HEADER_BYTES, backlog=1024)
        self.port = self.server.sockets[0].getsockname()[1]
        return self

    async def stop(self) -> None:
        if self.server is not None:
            self.server.close()
            try:
                await asyncio.wait_for(self.server.wait_closed(), 2.0)
            except asyncio.TimeoutError:
                pass
            self.server = None
        if self._log_fh:
            self._log_fh.close()
            self._log_fh = None
        self.files.close()

    async def __aenter__(self) -> "NetSimServer":
        return await self.start()

    async def __aexit__(self, *exc: Any) -> None:
        await self.stop()

    @property
    def url(self) -> str:
        host = self.host if self.host not in ("0.0.0.0", "::", "") else "127.0.0.1"
        return f"http://{host}:{self.port}"

    # -- waiting on the link ---------------------------------------------------

    def _wake_writers(self) -> None:
        for f in self._waiters:
            if not f.done():
                f.set_result(None)
        self._waiters.clear()

    async def _wait_link(self, timeout: float) -> None:
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        self._waiters.add(fut)
        h = loop.call_later(max(timeout, 0.0), lambda: fut.done() or fut.set_result(None))
        try:
            await fut
        finally:
            h.cancel()
            self._waiters.discard(fut)

    async def _sleep_until(self, t: float) -> None:
        dt = t - self.clock()
        if dt > 0:
            await asyncio.sleep(dt)

    # -- connection handling ---------------------------------------------------

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        cid = next(self._conn_ids)
        self.stats["connections"] += 1
        nreq = 0
        try:
            while True:
                try:
                    head = await reader.readuntil(b"\r\n\r\n")
                except (asyncio.IncompleteReadError, asyncio.LimitOverrunError, ConnectionError):
                    break
                t_arrive = self.clock()
                try:
                    lines = head.decode("latin-1").split("\r\n")
                    method, target, version = lines[0].split(" ", 2)
                    headers: dict[str, str] = {}
                    for ln in lines[1:]:
                        if not ln:
                            continue
                        k, v = ln.split(":", 1)
                        k = k.strip().lower()
                        headers[k] = f"{headers[k]}, {v.strip()}" if k in headers else v.strip()
                except ValueError:
                    writer.write(self._simple(400, b"bad request\n", close=True))
                    break
                conn_hdr = headers.get("connection", "").lower()
                keep = ("close" not in conn_hdr) if version == "HTTP/1.1" else ("keep-alive" in conn_hdr)
                body = b""
                clen = headers.get("content-length")
                if clen:
                    if not clen.isdigit() or int(clen) > 1 << 20:
                        writer.write(self._simple(413, b"too large\n", close=True))
                        break
                    body = await reader.readexactly(int(clen))
                path = urllib.parse.urlsplit(target).path
                if path.startswith(CONTROL_PREFIX):
                    writer.write(self._control(method, target, body, keep))
                    await writer.drain()
                else:
                    await self._serve(writer, cid, nreq == 0, method, target, headers, keep, t_arrive)
                nreq += 1
                if not keep:
                    break
        except (ConnectionError, asyncio.IncompleteReadError):
            pass
        except asyncio.CancelledError:
            raise
        except Exception as e:  # pragma: no cover
            if not self.quiet:
                print(f"netsim: connection {cid}: {e!r}", file=sys.stderr)
        finally:
            try:
                writer.close()
            except Exception:
                pass

    def _common_headers(self) -> list[str]:
        h = [
            f"Server: {SERVER_NAME}",
            f"Date: {_http_date()}",
            "Access-Control-Allow-Origin: *",
            "Access-Control-Expose-Headers: Content-Range, Content-Length, Accept-Ranges, "
            "ETag, Last-Modified, Content-Type, X-Netsim-Request",
            "Timing-Allow-Origin: *",
            "Cross-Origin-Resource-Policy: cross-origin",
        ]
        if self.isolate:
            h += ["Cross-Origin-Opener-Policy: same-origin",
                  "Cross-Origin-Embedder-Policy: require-corp"]
        return h

    def _head_bytes(self, status: int, headers: list[str], keep: bool) -> bytes:
        lines = [f"HTTP/1.1 {status} {REASONS.get(status, 'Unknown')}"]
        lines += self._common_headers()
        lines += headers
        lines.append("Connection: keep-alive" if keep else "Connection: close")
        return ("\r\n".join(lines) + "\r\n\r\n").encode("latin-1")

    def _simple(self, status: int, body: bytes, close: bool = False,
                ctype: str = "text/plain; charset=utf-8", extra: list[str] | None = None) -> bytes:
        hdr = [f"Content-Type: {ctype}", f"Content-Length: {len(body)}", "Cache-Control: no-store"]
        return self._head_bytes(status, hdr + (extra or []), not close) + body

    # -- static files ------------------------------------------------------------

    def _resolve(self, target: str) -> tuple[int, str | None]:
        path = urllib.parse.unquote(urllib.parse.urlsplit(target).path)
        if "\x00" in path:
            return 400, None
        full = os.path.realpath(os.path.join(self.root, path.lstrip("/")))
        if full != self.root and not full.startswith(self.root + os.sep):
            return 403, None
        if os.path.isdir(full):
            full = os.path.join(full, "index.html")
        if not os.path.isfile(full):
            return 404, None
        return 200, full

    def _prepare(self, method: str, target: str, headers: dict[str, str], keep: bool
                 ) -> tuple[int, bytes, Body, dict[str, Any]]:
        """Decide the response: status, header bytes, body, log fields."""
        info: dict[str, Any] = {}
        if method == "OPTIONS":
            req_h = headers.get("access-control-request-headers") or "Range, If-Range, Cache-Control"
            hdr = ["Access-Control-Allow-Methods: GET, HEAD, OPTIONS",
                   f"Access-Control-Allow-Headers: {req_h}",
                   "Access-Control-Max-Age: 86400", "Content-Length: 0"]
            return 204, self._head_bytes(204, hdr, keep), Body(), info
        if method not in ("GET", "HEAD"):
            return 405, self._simple(405, b"method not allowed\n", not keep,
                                     extra=["Allow: GET, HEAD, OPTIONS"]), Body(), info
        status, full = self._resolve(target)
        if full is None:
            msg = {400: b"bad request\n", 403: b"forbidden\n", 404: b"not found\n"}[status]
            return status, self._simple(status, msg, not keep), Body(), info
        st = os.stat(full)
        size = st.st_size
        etag = f'"{st.st_mtime_ns:x}-{size:x}"'
        lastmod = _http_date(st.st_mtime)
        ctype = mimetypes.guess_type(full)[0] or "application/octet-stream"
        base = [f"Accept-Ranges: bytes", f"ETag: {etag}", f"Last-Modified: {lastmod}",
                f"Cache-Control: {self.cache_control}"]
        info["size"] = size
        inm = headers.get("if-none-match")
        if inm and (inm.strip() == "*" or etag in [x.strip() for x in inm.split(",")]):
            return 304, self._head_bytes(304, base, keep), Body(), info
        rng_hdr = headers.get("range")
        if_range = headers.get("if-range")
        if rng_hdr and if_range and if_range.strip() not in (etag, lastmod):
            rng_hdr = None  # validator mismatch: send the whole representation
        ranges = parse_range(rng_hdr, size) if method in ("GET", "HEAD") else None
        body = Body()
        fd = self.files.open(full, st) if method == "GET" else -1
        if ranges is None:
            status = 200
            hdr = base + [f"Content-Type: {ctype}", f"Content-Length: {size}"]
            if fd >= 0:
                body.add_file(fd, 0, size)
            info["ranges"] = [[0, size - 1]] if size else []
        elif not ranges:
            status = 416
            hdr = base + [f"Content-Range: bytes */{size}", "Content-Length: 0"]
            info["ranges"] = []
        elif len(ranges) == 1:
            status = 206
            a, b = ranges[0]
            hdr = base + [f"Content-Type: {ctype}", f"Content-Range: bytes {a}-{b}/{size}",
                          f"Content-Length: {b - a + 1}"]
            if fd >= 0:
                body.add_file(fd, a, b - a + 1)
            info["ranges"] = [[a, b]]
        else:
            status = 206
            boundary = f"netsim{next(self._req_ids):012d}"
            mp = Body()
            for a, b in ranges:
                mp.add_bytes(f"\r\n--{boundary}\r\nContent-Type: {ctype}\r\n"
                             f"Content-Range: bytes {a}-{b}/{size}\r\n\r\n".encode("latin-1"))
                mp.add_file(fd if fd >= 0 else 0, a, b - a + 1)
            mp.add_bytes(f"\r\n--{boundary}--\r\n".encode("latin-1"))
            hdr = base + [f"Content-Type: multipart/byteranges; boundary={boundary}",
                          f"Content-Length: {mp.length}"]
            if fd >= 0:
                body = mp
            info["ranges"] = [list(r) for r in ranges]
        return status, self._head_bytes(status, hdr, keep), body, info

    async def _serve(self, writer: asyncio.StreamWriter, cid: int, first_on_conn: bool,
                     method: str, target: str, headers: dict[str, str], keep: bool,
                     t_arrive: float) -> None:
        rid = next(self._req_ids)
        status, head, body, info = self._prepare(method, target, headers, keep)
        head = head[:-2] + f"X-Netsim-Request: {rid}\r\n\r\n".encode()
        prof = self.profile
        await self.slots.acquire()
        flow = None
        try:
            t_start = self.clock()
            latency = self.sampler.latency_s()
            cap = self.sampler.request_cap_Bps()
            delay = latency + (prof.connect_ms / 1000.0 if first_on_conn else 0.0)
            await self._sleep_until(t_start + delay)
            t_fb = self.clock()
            n = body.length
            if n == 0:
                writer.write(head)
            elif not prof.shapes_bandwidth:
                first = min(n, WRITE_CHUNK)
                writer.write(head + body.read(0, first))
                pos = first
                while pos < n:
                    await writer.drain()
                    k = min(n - pos, WRITE_CHUNK)
                    writer.write(body.read(pos, k))
                    pos += k
            else:
                writer.write(head)
                ss = prof.init_cwnd_bytes if prof.slow_start else 0
                flow = nm.Flow(rid, n, cap, ss, prof.ss_rtt_s)
                self.link.add(flow, t_fb)
                pos = 0
                while pos < n:
                    now = self.clock()
                    self.link.advance(now)
                    allowed = n if flow.done else int(flow.sent)
                    # Write in chunks of at least MIN_SHAPED_CHUNK (or 5 ms
                    # worth of data) so the loop yields between writes.
                    rate = flow.rate if 0 < flow.rate < nm.INF else 0.0
                    goal = min(n, pos + max(MIN_SHAPED_CHUNK, int(rate * 0.005)))
                    if allowed >= goal:
                        while pos < allowed:
                            k = min(allowed - pos, WRITE_CHUNK)
                            writer.write(body.read(pos, k))
                            pos += k
                        await writer.drain()
                        continue
                    eta = self.link.eta(flow, goal)
                    await self._wait_link(min(eta - now, 0.05))
                flow = None
            await writer.drain()
            t_fin = self.clock()
        finally:
            if flow is not None and not flow.done:
                self.link.remove(flow, self.clock())
            self.slots.release()
        self.stats["requests"] += 1
        self.stats["bytes"] += body.length
        rec = {
            "id": rid, "conn": cid, "method": method, "path": target, "status": status,
            "range": headers.get("range"), "ranges": info.get("ranges"),
            "bytes": body.length, "size": info.get("size"),
            "t_arrive_ms": round(t_arrive * 1e3, 3),
            "queue_ms": round((t_start - t_arrive) * 1e3, 3),
            "latency_ms": round(delay * 1e3, 3), "new_conn": first_on_conn,
            "t_first_byte_ms": round(t_fb * 1e3, 3),
            "t_finish_ms": round(t_fin * 1e3, 3),
            "wall_arrive": round(self.wall0 + t_arrive, 6),
            "profile": prof.name,
        }
        self.log.append(rec)
        if self._log_fh:
            self._log_fh.write(json.dumps(rec) + "\n")

    # -- control -------------------------------------------------------------------

    def _control(self, method: str, target: str, body: bytes, keep: bool) -> bytes:
        parts = urllib.parse.urlsplit(target)
        what = parts.path[len(CONTROL_PREFIX):].strip("/")
        q = urllib.parse.parse_qs(parts.query)
        close = not keep

        def ok(obj: Any, status: int = 200) -> bytes:
            return self._simple(status, (json.dumps(obj) + "\n").encode(), close, "application/json")

        try:
            if method == "OPTIONS":
                return self._head_bytes(204, [
                    "Access-Control-Allow-Methods: GET, POST, PUT, OPTIONS",
                    "Access-Control-Allow-Headers: Content-Type",
                    "Content-Length: 0"], keep)
            if what == "profile" and method == "GET":
                return ok({"profile": self.profile.to_dict(), "seed": self.seed,
                           "describe": nm.describe(self.profile)})
            if what == "profile" and method in ("POST", "PUT"):
                obj = json.loads(body or b"{}")
                seed = obj.get("seed")
                prof = nm.profile_from_json(obj, base=self.profile)
                if "preset" not in obj and "name" not in obj:
                    prof.name = self.profile.name + "*"
                self.set_profile(prof, seed=seed if seed is not None else self.seed)
                if obj.get("reset_log"):
                    self.log.clear()
                return ok({"profile": self.profile.to_dict(), "seed": self.seed,
                           "describe": nm.describe(self.profile)})
            if what == "presets":
                return ok({"presets": nm.PRESETS, "modifiers": nm.MODIFIERS})
            if what == "log" and method == "GET":
                since = int(q.get("since", ["0"])[0])
                return ok([r for r in self.log if r["id"] > since])
            if what == "log/clear" and method in ("POST", "PUT"):
                self.log.clear()
                return ok({"cleared": True})
            if what == "stats":
                return ok({**self.stats, "in_service": self.slots.busy,
                           "queued": len(self.slots.waiters),
                           "active_flows": len(self.link.flows),
                           "uptime_s": round(self.clock(), 3)})
            return ok({"error": "unknown control endpoint"}, 404)
        except (ValueError, KeyError, TypeError) as e:
            return ok({"error": str(e)}, 400)


class ServerThread:
    """Runs a NetSimServer on its own event loop in a background thread."""

    def __init__(self, root: str, profile: nm.Profile | None = None, **kw: Any):
        self.srv = NetSimServer(root, profile, **kw)
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self.loop.run_forever, daemon=True)

    def start(self) -> "ServerThread":
        self._thread.start()
        asyncio.run_coroutine_threadsafe(self.srv.start(), self.loop).result(10)
        return self

    def stop(self) -> None:
        asyncio.run_coroutine_threadsafe(self.srv.stop(), self.loop).result(10)
        self.loop.call_soon_threadsafe(self.loop.stop)
        self._thread.join(5)
        self.loop.close()

    def set_profile(self, profile: nm.Profile, seed: Any = None) -> None:
        async def _set() -> None:
            self.srv.set_profile(profile, seed)
        asyncio.run_coroutine_threadsafe(_set(), self.loop).result(10)

    @property
    def url(self) -> str:
        return self.srv.url

    def __enter__(self) -> "ServerThread":
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.stop()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=".", help="directory to serve")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log", default=None, help="append JSON-lines request log to this file")
    ap.add_argument("--isolate", action="store_true", help="send COOP/COEP headers (cross-origin isolation)")
    ap.add_argument("--cache-control", default="no-store")
    ap.add_argument("--verbose", action="store_true")
    nm.add_profile_args(ap)
    args = ap.parse_args(argv)
    prof = nm.profile_from_args(args)

    async def run() -> None:
        srv = NetSimServer(args.dir, prof, host=args.host, port=args.port, seed=args.seed,
                           log_path=args.log, isolate=args.isolate,
                           cache_control=args.cache_control, quiet=not args.verbose)
        await srv.start()
        print(f"netsim: serving {srv.root} at {srv.url} ({prof.name}: {nm.describe(prof)})", flush=True)
        try:
            await asyncio.Event().wait()
        finally:
            await srv.stop()

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
