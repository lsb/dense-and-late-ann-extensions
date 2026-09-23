"""Shared network model for the range server and the simulator.

Both ``rangeserver.py`` (real time) and ``simulate.py`` (virtual time) use the
definitions in this module, so a profile means exactly the same thing in both:

* ``Profile``: a dataclass holding every shaping parameter, plus named presets
  and modifiers that can be combined (``"4g,h1,jitter"``).
* ``RequestSampler``: draws the per-request random quantities (latency, tail
  event, per-request bandwidth factor) from a seeded generator.
* ``FluidLink``: a fluid (max-min fair) model of a shared bottleneck link with
  optional per-flow caps, optional per-request TCP slow start, and optional
  time-varying capacity.  It can be advanced to any time, which the simulator
  does event by event and the server does with the wall clock.
* ``SlotPool``: the connection/concurrency model (FIFO queue, lazily opened
  connections that pay ``connect_ms`` once).

Units: public profile fields use milliseconds and kilobits per second
(1 kbps = 1000 bit/s); internally times are seconds and rates bytes/second.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
import random
import struct
from collections import deque
from dataclasses import dataclass, field, fields
from typing import Any, Callable, Iterable

INF = math.inf

LATENCY_DISTS = ("fixed", "normal", "lognormal", "exponential", "pareto")


# --------------------------------------------------------------------------
# Profiles
# --------------------------------------------------------------------------


@dataclass
class Profile:
    """Network shaping parameters.  Zero means "unlimited"/"off" for rates."""

    name: str = "custom"
    # Per-request latency (time from the request getting a slot to the first
    # response byte).  On a warm keep-alive connection this is one round trip.
    latency_ms: float = 0.0
    latency_dist: str = "fixed"  # one of LATENCY_DISTS
    latency_jitter_ms: float = 0.0  # normal: std-dev; exponential: mean extra
    latency_sigma: float = 0.0  # lognormal: shape (median = latency_ms)
    latency_pareto_alpha: float = 2.5  # pareto: tail index (minimum = latency_ms)
    latency_min_ms: float = 0.0  # clamp applied after sampling
    latency_max_ms: float = INF
    tail_prob: float = 0.0  # with this probability ...
    tail_ms: float = 0.0  # ... add this many ms to the latency
    # Throughput.
    request_kbps: float = 0.0  # per-request (per-flow) cap; 0 = none
    request_kbps_jitter: float = 0.0  # lognormal sigma of a per-request factor
    link_kbps: float = 0.0  # shared bottleneck capacity; 0 = unlimited
    link_jitter: float = 0.0  # lognormal sigma of the capacity per period
    link_jitter_period_ms: float = 100.0
    # Concurrency and connections.
    max_concurrent: int = 0  # requests in service at once; 0 = unlimited
    connect_ms: float = 0.0  # extra delay on the first request of a connection
    # TCP slow start (per request, cold window), off by default.
    slow_start: bool = False
    init_cwnd_bytes: int = 14600  # 10 segments of 1460 bytes (RFC 6928)
    ss_rtt_ms: float = 0.0  # 0 = use latency_ms as the round-trip time

    # -- helpers -----------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        for k, v in d.items():
            if isinstance(v, float) and math.isinf(v):
                d[k] = None if k == "latency_max_ms" else v
        return d

    def replace(self, **kw: Any) -> "Profile":
        return apply_overrides(self, kw)

    @property
    def request_Bps(self) -> float:
        return self.request_kbps * 125.0 if self.request_kbps > 0 else INF

    @property
    def link_Bps(self) -> float:
        return self.link_kbps * 125.0 if self.link_kbps > 0 else INF

    @property
    def shapes_bandwidth(self) -> bool:
        return self.request_kbps > 0 or self.link_kbps > 0 or self.slow_start

    @property
    def ss_rtt_s(self) -> float:
        return (self.ss_rtt_ms or self.latency_ms) / 1000.0

    def validate(self) -> "Profile":
        if self.latency_dist not in LATENCY_DISTS:
            raise ValueError(f"latency_dist must be one of {LATENCY_DISTS}")
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, (int, float)) and not isinstance(v, bool) and v < 0:
                raise ValueError(f"{f.name} must be >= 0")
        if not 0 <= self.tail_prob <= 1:
            raise ValueError("tail_prob must be in [0, 1]")
        if self.latency_dist == "pareto" and self.latency_pareto_alpha <= 0:
            raise ValueError("latency_pareto_alpha must be > 0")
        if self.link_jitter > 0 and self.link_jitter_period_ms <= 0:
            raise ValueError("link_jitter_period_ms must be > 0")
        return self


_FIELD_TYPES = {f.name: f.type for f in fields(Profile)}


def _coerce(name: str, value: Any) -> Any:
    t = _FIELD_TYPES[name]
    if value is None and name == "latency_max_ms":
        return INF
    if t in ("float", float):
        return float(value)
    if t in ("int", int):
        return int(value)
    if t in ("bool", bool):
        if isinstance(value, str):
            return value.lower() in ("1", "true", "yes", "on")
        return bool(value)
    return str(value)


def apply_overrides(p: Profile, overrides: dict[str, Any]) -> Profile:
    d = {f.name: getattr(p, f.name) for f in fields(Profile)}
    for k, v in overrides.items():
        k = k.replace("-", "_")
        if k not in d:
            raise KeyError(f"unknown profile field {k!r}")
        d[k] = _coerce(k, v)
    return Profile(**d).validate()


# Network presets.  latency_ms is the per-request latency on a warm
# connection (about one RTT); link_kbps is the downlink capacity shared by all
# responses.  Sources and caveats are in NOTES.md.
PRESETS: dict[str, dict[str, Any]] = {
    "none": {},
    "lan": {"latency_ms": 1, "link_kbps": 1_000_000},
    "fios": {"latency_ms": 4, "link_kbps": 20_000},  # WebPageTest "FIOS"
    "cable": {"latency_ms": 28, "link_kbps": 5_000},  # WebPageTest "Cable"
    "dsl": {"latency_ms": 50, "link_kbps": 1_500},  # WebPageTest "DSL"
    "wifi": {"latency_ms": 20, "link_kbps": 50_000},
    "5g": {"latency_ms": 40, "link_kbps": 100_000},
    "starlink": {"latency_ms": 45, "link_kbps": 100_000},
    "lte": {"latency_ms": 70, "link_kbps": 12_000},  # WebPageTest "LTE"
    "4g": {"latency_ms": 165, "link_kbps": 8_100},  # Chrome DevTools "Fast 4G"
    "wpt-4g": {"latency_ms": 170, "link_kbps": 9_000},  # WebPageTest "4G"
    "lighthouse": {"latency_ms": 150, "link_kbps": 1_638.4},  # Lighthouse mobile
    "slow-4g": {"latency_ms": 562.5, "link_kbps": 1_440},  # DevTools "Slow 4G"
    "3g": {"latency_ms": 300, "link_kbps": 1_600},  # WebPageTest "3G"
    "slow-3g": {"latency_ms": 2_000, "link_kbps": 400},  # DevTools "Slow 3G"
    "edge": {"latency_ms": 840, "link_kbps": 240},  # WebPageTest "Edge"
    "satellite": {"latency_ms": 600, "link_kbps": 10_000},  # geostationary
    # A probabilistic preset: cell-edge LTE with a heavy latency tail.
    "lte-poor": {
        "latency_ms": 120,
        "latency_dist": "lognormal",
        "latency_sigma": 0.5,
        "latency_max_ms": 5_000,
        "tail_prob": 0.02,
        "tail_ms": 1_000,
        "link_kbps": 2_000,
        "link_jitter": 0.5,
        "link_jitter_period_ms": 200,
    },
}

# Modifiers are presets that only touch a few fields; they are meant to be
# appended to a network preset, e.g. "4g,h1" or "lte,jitter,slowstart".
MODIFIERS: dict[str, dict[str, Any]] = {
    "h1": {"max_concurrent": 6},  # browser HTTP/1.1 per-host connection limit
    "h2": {"max_concurrent": 100},  # typical SETTINGS_MAX_CONCURRENT_STREAMS
    "jitter": {  # mild variability: lognormal latency, rare tail events
        "latency_dist": "lognormal",
        "latency_sigma": 0.25,
        "tail_prob": 0.01,
        "tail_ms": 500,
        "link_jitter": 0.25,
    },
    "slowstart": {"slow_start": True},
    "cold": {"connect_ms_rtts": 2},  # TCP + TLS 1.3 handshakes: 2 RTT
}


def preset(spec: str | None, **overrides: Any) -> Profile:
    """Build a profile from ``"name[,modifier...]"`` plus field overrides."""
    p = Profile(name=spec or "none")
    for part in (spec or "none").split(","):
        part = part.strip()
        if not part:
            continue
        if part in PRESETS:
            p = apply_overrides(p, PRESETS[part])
        elif part in MODIFIERS:
            mod = dict(MODIFIERS[part])
            if "connect_ms_rtts" in mod:
                mod = {"connect_ms": mod.pop("connect_ms_rtts") * p.latency_ms}
            p = apply_overrides(p, mod)
        else:
            raise KeyError(
                f"unknown preset {part!r}; known: "
                + ", ".join(list(PRESETS) + list(MODIFIERS))
            )
    p.name = spec or "none"
    return apply_overrides(p, overrides)


def profile_from_json(obj: dict[str, Any], base: Profile | None = None) -> Profile:
    """``{"preset": "4g,h1", "latency_ms": 100, ...}`` -> Profile.

    Without ``preset`` the overrides apply to ``base`` (default: a blank
    profile), which lets a control request change a single field."""
    obj = dict(obj)
    spec = obj.pop("preset", None)
    obj.pop("seed", None)
    if spec is not None:
        p = preset(spec)
    else:
        p = base or Profile()
    return apply_overrides(p, obj)


# --------------------------------------------------------------------------
# Random quantities
# --------------------------------------------------------------------------


def _hash_uniform(*parts: Any) -> tuple[float, float]:
    """Two uniforms in (0, 1) from a hash of ``parts`` (order-independent RNG)."""
    h = hashlib.blake2b(repr(parts).encode(), digest_size=16).digest()
    a, b = struct.unpack("<QQ", h)
    return ((a >> 11) + 0.5) / 2.0**53, ((b >> 11) + 0.5) / 2.0**53


def _normal_from(u1: float, u2: float) -> float:
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


class RequestSampler:
    """Draws per-request randomness in the order requests obtain a slot."""

    def __init__(self, profile: Profile, seed: int | None = 0):
        self.profile = profile
        self.rng = random.Random(seed)

    def latency_s(self) -> float:
        p, r = self.profile, self.rng
        L = p.latency_ms
        d = p.latency_dist
        if d == "fixed":
            x = L
        elif d == "normal":
            x = r.gauss(L, p.latency_jitter_ms)
        elif d == "lognormal":
            x = L * math.exp(p.latency_sigma * r.gauss(0.0, 1.0))
        elif d == "exponential":
            x = L + (r.expovariate(1.0 / p.latency_jitter_ms) if p.latency_jitter_ms > 0 else 0.0)
        elif d == "pareto":
            x = L * r.paretovariate(p.latency_pareto_alpha)
        else:  # pragma: no cover - validated earlier
            raise ValueError(d)
        x = min(max(x, p.latency_min_ms, 0.0), p.latency_max_ms)
        if p.tail_prob > 0 and r.random() < p.tail_prob:
            x += p.tail_ms
        return x / 1000.0

    def request_cap_Bps(self) -> float:
        p = self.profile
        cap = p.request_Bps
        if p.request_kbps_jitter > 0 and cap < INF:
            cap *= math.exp(p.request_kbps_jitter * self.rng.gauss(0.0, 1.0))
        return cap


def link_capacity_factor(seed: Any, period_index: int, sigma: float) -> float:
    """Lognormal capacity factor for one jitter period, normalised to mean 1.

    It is a pure function of (seed, period), so the capacity trace does not
    depend on the order in which requests happen to arrive."""
    if sigma <= 0:
        return 1.0
    z = _normal_from(*_hash_uniform("link", seed, period_index))
    return math.exp(sigma * z - 0.5 * sigma * sigma)


# --------------------------------------------------------------------------
# Fluid link
# --------------------------------------------------------------------------

_EPS_T = 1e-9


class Flow:
    """One response body travelling over the link."""

    __slots__ = (
        "fid", "size", "cap", "sent", "start", "ss_init", "ss_rtt",
        "rate", "done", "t_done", "user",
    )

    def __init__(self, fid: Any, size: int, cap_Bps: float = INF,
                 ss_init: float = 0.0, ss_rtt: float = 0.0, user: Any = None):
        self.fid = fid
        self.size = float(size)
        self.cap = cap_Bps
        self.sent = 0.0
        self.start = 0.0
        self.ss_init = ss_init  # 0 disables slow start
        self.ss_rtt = ss_rtt
        self.rate = 0.0
        self.done = False
        self.t_done = INF
        self.user = user

    # Slow start as a staircase ceiling on cumulative bytes: by the start of
    # round trip k (k = 0, 1, ...) after the first byte, at most
    # init * (2^(k+1) - 1) bytes may have been sent.
    def _epoch(self, t: float) -> int:
        return int(math.floor((t - self.start) / self.ss_rtt + 1e-7))

    def ceiling(self, t: float) -> float:
        if self.ss_init <= 0 or self.ss_rtt <= 0:
            return INF
        k = self._epoch(t)
        if k >= 60:
            return INF
        return self.ss_init * (2.0 ** (k + 1) - 1.0)

    def next_step(self, t: float) -> float:
        if self.ss_init <= 0 or self.ss_rtt <= 0 or self.ceiling(t) >= self.size:
            return INF
        return self.start + (self._epoch(t) + 1) * self.ss_rtt

    def target(self, t: float) -> float:
        return min(self.size, self.ceiling(t))


class FluidLink:
    """Max-min fair sharing of one bottleneck among active flows.

    Between events every flow moves at a constant rate given by water-filling
    the link capacity over the flows' caps.  Events are flow completions, a
    flow reaching its slow-start ceiling, a slow-start round boundary, and a
    capacity change (when capacity jitter is on)."""

    def __init__(self, capacity_Bps: float = INF, jitter_sigma: float = 0.0,
                 jitter_period_s: float = 0.1, seed: Any = 0, t0: float = 0.0):
        self.capacity = capacity_Bps
        self.jitter_sigma = jitter_sigma
        self.jitter_period = jitter_period_s
        self.seed = seed
        self.t0 = t0
        self.now = t0
        self.flows: dict[Any, Flow] = {}
        self.completed: list[Flow] = []
        self.on_change: Callable[[], None] | None = None
        self._dirty = True

    @classmethod
    def for_profile(cls, p: Profile, seed: Any = 0, t0: float = 0.0) -> "FluidLink":
        return cls(p.link_Bps, p.link_jitter, p.link_jitter_period_ms / 1000.0, seed, t0)

    def reconfigure(self, p: Profile, seed: Any = None) -> None:
        self.advance(self.now)
        self.capacity = p.link_Bps
        self.jitter_sigma = p.link_jitter
        self.jitter_period = p.link_jitter_period_ms / 1000.0
        if seed is not None:
            self.seed = seed
        self._dirty = True

    # -- capacity ----------------------------------------------------------

    def _period(self, t: float) -> int:
        return int(math.floor((t - self.t0) / self.jitter_period + 1e-7))

    def capacity_at(self, t: float) -> float:
        if self.jitter_sigma <= 0 or self.capacity == INF:
            return self.capacity
        return self.capacity * link_capacity_factor(self.seed, self._period(t), self.jitter_sigma)

    def _next_capacity_change(self, t: float) -> float:
        if self.jitter_sigma <= 0 or self.capacity == INF:
            return INF
        return self.t0 + (self._period(t) + 1) * self.jitter_period

    # -- flows -------------------------------------------------------------

    def add(self, flow: Flow, t: float) -> Flow:
        self.advance(t)
        flow.start = self.now
        if flow.size <= 0:
            flow.done, flow.t_done = True, self.now
            self.completed.append(flow)
        else:
            self.flows[flow.fid] = flow
            self._dirty = True
            self.advance(self.now)  # settles flows that finish instantly
        self._notify()
        return flow

    def pop_completed(self) -> list[Flow]:
        c, self.completed = self.completed, []
        return c

    def _notify(self) -> None:
        if self.on_change is not None:
            self.on_change()

    def _compute_rates(self) -> None:
        t = self.now
        active = []
        for f in self.flows.values():
            if f.sent < f.target(t) - 1e-9:
                active.append(f)
            else:
                f.rate = 0.0
        remaining = self.capacity_at(t)
        active.sort(key=lambda f: f.cap)
        n = len(active)
        for i, f in enumerate(active):
            share = remaining / (n - i) if remaining < INF else INF
            if f.cap <= share:
                f.rate = f.cap
                if remaining < INF:
                    remaining -= f.cap
            else:
                for g in active[i:]:
                    g.rate = share
                break
        self._dirty = False

    def next_event_time(self) -> float:
        if self._dirty:
            self._compute_rates()
        t = self.now
        best = self._next_capacity_change(t) if self.flows else INF
        for f in self.flows.values():
            if f.rate > 0:
                rem = f.target(t) - f.sent
                best = min(best, t + (rem / f.rate if f.rate < INF else 0.0))
            best = min(best, f.next_step(t))
        return best

    def eta(self, flow: Flow, nbytes: float) -> float:
        """Time at which ``flow`` reaches ``nbytes`` if rates stay as now."""
        if flow.done or flow.sent >= nbytes:
            return self.now
        if self._dirty:
            self._compute_rates()
        if flow.rate <= 0:
            return min(flow.next_step(self.now), self._next_capacity_change(self.now))
        if flow.rate == INF:
            return self.now
        return self.now + (min(nbytes, flow.target(self.now)) - flow.sent) / flow.rate

    def advance(self, t: float) -> None:
        """Integrate the fluid model up to time ``t`` (never backwards)."""
        if t < self.now:
            t = self.now
        changed = False
        for _ in range(1_000_000):
            if not self.flows:
                self.now = t
                break
            if self._dirty:
                self._compute_rates()
            t_evt = self.next_event_time()
            if t_evt > t + _EPS_T:
                dt = t - self.now
                for f in self.flows.values():
                    if 0 < f.rate < INF:
                        f.sent += f.rate * dt
                self.now = t
                break
            t_evt = max(t_evt, self.now)
            dt = t_evt - self.now
            for f in list(self.flows.values()):
                if f.rate <= 0:
                    continue
                tgt = f.target(self.now)
                if f.rate == INF or self.now + (tgt - f.sent) / f.rate <= t_evt + _EPS_T:
                    f.sent = tgt  # reached its completion or ceiling: snap
                else:
                    f.sent += f.rate * dt
            self.now = t_evt
            for f in list(self.flows.values()):
                if f.sent >= f.size - 1e-6:
                    f.sent = f.size
                    f.done, f.t_done = True, self.now
                    del self.flows[f.fid]
                    self.completed.append(f)
                    changed = True
            self._dirty = True
        else:  # pragma: no cover
            raise RuntimeError("FluidLink.advance did not converge")
        if changed:
            self._notify()


# --------------------------------------------------------------------------
# Connection / concurrency model
# --------------------------------------------------------------------------


class SlotPool:
    """FIFO admission with at most ``limit`` requests in service (0 = no limit).

    Connections are opened lazily: a request that starts service when no idle
    connection exists opens a new one and pays ``connect_ms``.  This mirrors a
    browser, which keeps up to ``limit`` keep-alive connections per host.
    The class is time-free; the simulator drives it with virtual time."""

    def __init__(self, limit: int = 0):
        self.limit = limit
        self.busy = 0
        self.idle_conns: deque[int] = deque()
        self.n_conns = 0
        self.queue: deque[Any] = deque()

    def _can_start(self) -> bool:
        return self.limit <= 0 or self.busy < self.limit

    def _take_conn(self) -> tuple[int, bool]:
        self.busy += 1
        if self.idle_conns:
            return self.idle_conns.pop(), False
        self.n_conns += 1
        return self.n_conns - 1, True

    def arrive(self, item: Any) -> list[tuple[Any, int, bool]]:
        """Returns the items that start now, as (item, conn_id, is_new_conn)."""
        self.queue.append(item)
        return self._drain()

    def release(self, conn_id: int) -> list[tuple[Any, int, bool]]:
        self.busy -= 1
        self.idle_conns.append(conn_id)
        return self._drain()

    def _drain(self) -> list[tuple[Any, int, bool]]:
        started = []
        while self.queue and self._can_start():
            item = self.queue.popleft()
            cid, new = self._take_conn()
            started.append((item, cid, new))
        return started


def describe(p: Profile) -> str:
    bits = [f"latency {p.latency_ms:g} ms ({p.latency_dist})"]
    bits.append(f"link {p.link_kbps:g} kbps" if p.link_kbps else "link unlimited")
    if p.request_kbps:
        bits.append(f"per-request {p.request_kbps:g} kbps")
    if p.max_concurrent:
        bits.append(f"max {p.max_concurrent} concurrent")
    if p.connect_ms:
        bits.append(f"connect {p.connect_ms:g} ms")
    if p.slow_start:
        bits.append("slow start")
    if p.tail_prob:
        bits.append(f"tail {p.tail_prob:g}x+{p.tail_ms:g} ms")
    return ", ".join(bits)


def add_profile_args(ap: Any) -> None:
    """Add ``--preset`` and one ``--field`` flag per profile field to argparse."""
    ap.add_argument("--preset", default="none",
                    help="network preset, optionally with modifiers, e.g. 4g,h1,jitter "
                         f"(presets: {', '.join(PRESETS)}; modifiers: {', '.join(MODIFIERS)})")
    for f in fields(Profile):
        if f.name == "name":
            continue
        flag = "--" + f.name.replace("_", "-")
        if f.type in ("bool", bool):
            ap.add_argument(flag, dest=f.name, default=None,
                            type=lambda s: s.lower() in ("1", "true", "yes", "on"),
                            metavar="BOOL")
        elif f.name == "latency_dist":
            ap.add_argument(flag, dest=f.name, default=None, choices=LATENCY_DISTS)
        else:
            ap.add_argument(flag, dest=f.name, default=None,
                            type=int if f.type in ("int", int) else float)


def profile_from_args(args: Any) -> Profile:
    over = {f.name: getattr(args, f.name) for f in fields(Profile)
            if f.name != "name" and getattr(args, f.name, None) is not None}
    return preset(args.preset, **over)


def iter_presets() -> Iterable[str]:
    return iter(PRESETS)
