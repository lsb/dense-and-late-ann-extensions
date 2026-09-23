"""Tests for netsim: HTTP range semantics, the shared model, and agreement
between the real server (wall-clock) and the simulator (virtual time).

Run with ``python3 -m pytest netsim`` or ``python3 -m unittest netsim.test_netsim``.
Timing tests use generous tolerances because the machine may be loaded.
"""

from __future__ import annotations

import asyncio
import http.client
import json
import os
import shutil
import tempfile
import unittest

try:
    from . import netmodel as nm, rangeserver as rs, replay as rp, simulate as sim
except ImportError:  # run from inside netsim/
    import netmodel as nm  # type: ignore
    import rangeserver as rs  # type: ignore
    import replay as rp  # type: ignore
    import simulate as sim  # type: ignore

SIZE = 2_000_000


def _make_root() -> str:
    d = tempfile.mkdtemp(prefix="netsim-test-")
    data = bytes((i * 7 + (i >> 8)) & 0xFF for i in range(SIZE))
    with open(os.path.join(d, "f.bin"), "wb") as fh:
        fh.write(data)
    with open(os.path.join(d, "empty.bin"), "wb"):
        pass
    os.mkdir(os.path.join(d, "sub"))
    with open(os.path.join(d, "sub", "index.html"), "w") as fh:
        fh.write("<p>hi</p>")
    return d


class RangeSemantics(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = _make_root()
        with open(os.path.join(cls.root, "f.bin"), "rb") as fh:
            cls.data = fh.read()
        cls.logfile = os.path.join(cls.root, "log.jsonl")
        cls.srv = rs.ServerThread(cls.root, nm.Profile(), isolate=True, log_path=cls.logfile).start()
        cls.port = cls.srv.srv.port

    @classmethod
    def tearDownClass(cls) -> None:
        cls.srv.stop()
        shutil.rmtree(cls.root)

    def req(self, method: str, path: str, headers: dict | None = None, body: bytes | None = None,
            conn: http.client.HTTPConnection | None = None):
        c = conn or http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        c.request(method, path, body=body, headers=headers or {})
        r = c.getresponse()
        data = r.read()
        if conn is None:
            c.close()
        return r, data

    def test_full_get(self):
        r, d = self.req("GET", "/f.bin")
        self.assertEqual(r.status, 200)
        self.assertEqual(d, self.data)
        self.assertEqual(r.getheader("Accept-Ranges"), "bytes")
        self.assertEqual(r.getheader("Content-Length"), str(SIZE))
        self.assertTrue(r.getheader("ETag"))
        self.assertTrue(r.getheader("Last-Modified"))
        self.assertEqual(r.getheader("Access-Control-Allow-Origin"), "*")
        self.assertIn("Content-Range", r.getheader("Access-Control-Expose-Headers"))
        self.assertEqual(r.getheader("Cross-Origin-Opener-Policy"), "same-origin")
        self.assertEqual(r.getheader("Cross-Origin-Embedder-Policy"), "require-corp")

    def test_simple_range(self):
        r, d = self.req("GET", "/f.bin", {"Range": "bytes=10-109"})
        self.assertEqual(r.status, 206)
        self.assertEqual(d, self.data[10:110])
        self.assertEqual(r.getheader("Content-Range"), f"bytes 10-109/{SIZE}")
        self.assertEqual(r.getheader("Content-Length"), "100")

    def test_suffix_open_and_clamped(self):
        r, d = self.req("GET", "/f.bin", {"Range": "bytes=-100"})
        self.assertEqual((r.status, d), (206, self.data[-100:]))
        self.assertEqual(r.getheader("Content-Range"), f"bytes {SIZE-100}-{SIZE-1}/{SIZE}")
        r, d = self.req("GET", "/f.bin", {"Range": f"bytes={SIZE-50}-"})
        self.assertEqual((r.status, d), (206, self.data[-50:]))
        r, d = self.req("GET", "/f.bin", {"Range": f"bytes={SIZE-10}-{SIZE+1000}"})
        self.assertEqual((r.status, d), (206, self.data[-10:]))
        r, d = self.req("GET", "/f.bin", {"Range": f"bytes=-{SIZE*2}"})
        self.assertEqual((r.status, len(d)), (206, SIZE))

    def test_unsatisfiable(self):
        r, d = self.req("GET", "/f.bin", {"Range": f"bytes={SIZE}-"})
        self.assertEqual(r.status, 416)
        self.assertEqual(r.getheader("Content-Range"), f"bytes */{SIZE}")
        self.assertEqual(d, b"")
        r, _ = self.req("GET", "/f.bin", {"Range": "bytes=-0"})
        self.assertEqual(r.status, 416)
        r, _ = self.req("GET", "/empty.bin", {"Range": "bytes=0-10"})
        self.assertEqual(r.status, 416)

    def test_invalid_range_ignored(self):
        for h in ("bytes=5-3", "items=0-10", "bytes=abc", "bytes=1-2x"):
            r, d = self.req("GET", "/f.bin", {"Range": h})
            self.assertEqual(r.status, 200, h)
            self.assertEqual(len(d), SIZE)

    def test_multirange(self):
        r, d = self.req("GET", "/f.bin", {"Range": "bytes=0-9,100-199,-5"})
        self.assertEqual(r.status, 206)
        ctype = r.getheader("Content-Type")
        self.assertTrue(ctype.startswith("multipart/byteranges; boundary="))
        self.assertEqual(int(r.getheader("Content-Length")), len(d))
        boundary = ctype.split("boundary=")[1].encode()
        parts = [p for p in d.split(b"--" + boundary) if p.strip() not in (b"", b"--")]
        got = []
        for p in parts:
            head, body = p.split(b"\r\n\r\n", 1)
            got.append((head.split(b"Content-Range: bytes ")[1].split(b"/")[0].decode(), body[:-2]))
        self.assertEqual(got, [("0-9", self.data[0:10]), ("100-199", self.data[100:200]),
                               (f"{SIZE-5}-{SIZE-1}", self.data[-5:])])

    def test_head(self):
        r, d = self.req("HEAD", "/f.bin", {"Range": "bytes=0-99"})
        self.assertEqual((r.status, d), (206, b""))
        self.assertEqual(r.getheader("Content-Length"), "100")
        r, d = self.req("HEAD", "/f.bin")
        self.assertEqual((r.status, r.getheader("Content-Length")), (200, str(SIZE)))

    def test_conditionals(self):
        r, _ = self.req("HEAD", "/f.bin")
        etag = r.getheader("ETag")
        r, d = self.req("GET", "/f.bin", {"Range": "bytes=0-9", "If-Range": etag})
        self.assertEqual((r.status, len(d)), (206, 10))
        r, d = self.req("GET", "/f.bin", {"Range": "bytes=0-9", "If-Range": '"stale"'})
        self.assertEqual((r.status, len(d)), (200, SIZE))
        r, d = self.req("GET", "/f.bin", {"If-None-Match": etag})
        self.assertEqual((r.status, d), (304, b""))

    def test_options_preflight(self):
        r, _ = self.req("OPTIONS", "/f.bin", {"Origin": "http://example.test",
                                                "Access-Control-Request-Method": "GET",
                                                "Access-Control-Request-Headers": "range"})
        self.assertEqual(r.status, 204)
        self.assertEqual(r.getheader("Access-Control-Allow-Origin"), "*")
        self.assertIn("GET", r.getheader("Access-Control-Allow-Methods"))
        self.assertIn("range", r.getheader("Access-Control-Allow-Headers").lower())

    def test_errors_and_paths(self):
        self.assertEqual(self.req("GET", "/nope.bin")[0].status, 404)
        self.assertIn(self.req("GET", "/../../etc/passwd")[0].status, (403, 404))
        self.assertIn(self.req("GET", "/%2e%2e/%2e%2e/etc/passwd")[0].status, (403, 404))
        self.assertEqual(self.req("DELETE", "/f.bin")[0].status, 405)
        r, d = self.req("GET", "/sub/")
        self.assertEqual((r.status, d), (200, b"<p>hi</p>"))
        r, d = self.req("GET", "/f.bin?x=1", {"Range": "bytes=0-0"})
        self.assertEqual((r.status, d), (206, self.data[:1]))

    def test_keepalive_and_log(self):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        ids = []
        for i in range(3):
            r, d = self.req("GET", "/f.bin", {"Range": f"bytes={i*10}-{i*10+9}"}, conn=c)
            self.assertEqual(d, self.data[i * 10:i * 10 + 10])
            ids.append(int(r.getheader("X-Netsim-Request")))
        c.close()
        r, d = self.req("GET", f"/__netsim/log?since={ids[0] - 1}")
        recs = {x["id"]: x for x in json.loads(d)}
        conns = {recs[i]["conn"] for i in ids}
        self.assertEqual(len(conns), 1)
        self.assertEqual([recs[i]["new_conn"] for i in ids], [True, False, False])
        rec = recs[ids[1]]
        for k in ("t_arrive_ms", "queue_ms", "t_first_byte_ms", "t_finish_ms", "bytes", "ranges"):
            self.assertIn(k, rec)
        self.assertEqual(rec["ranges"], [[10, 19]])
        with open(self.logfile) as fh:
            lines = [json.loads(x) for x in fh]
        self.assertIn(ids[2], [x["id"] for x in lines])

    def test_control_profile(self):
        r, d = self.req("POST", "/__netsim/profile", body=json.dumps({"preset": "3g", "seed": 5}).encode(),
                        headers={"Content-Type": "application/json"})
        self.assertEqual(r.status, 200)
        got = json.loads(d)
        self.assertEqual((got["profile"]["latency_ms"], got["seed"]), (300.0, 5))
        r, d = self.req("POST", "/__netsim/profile", body=b'{"latency_ms": 0, "link_kbps": 0}')
        got = json.loads(d)["profile"]
        self.assertEqual((got["latency_ms"], got["link_kbps"]), (0.0, 0.0))
        r, d = self.req("POST", "/__netsim/profile", body=b'{"no_such_field": 1}')
        self.assertEqual(r.status, 400)
        r, d = self.req("POST", "/__netsim/profile", body=b'{"preset": "none"}')
        self.assertEqual(json.loads(d)["profile"]["name"], "none")


class ModelTests(unittest.TestCase):
    def test_presets(self):
        p = nm.preset("4g,h1,cold", latency_ms=100)
        self.assertEqual((p.link_kbps, p.max_concurrent, p.connect_ms, p.latency_ms), (8100, 6, 330, 100))
        with self.assertRaises(KeyError):
            nm.preset("nope")
        for name in nm.PRESETS:
            nm.preset(name)

    def test_water_filling(self):
        t = sim.make_trace([[100_000, 200_000, 300_000, 400_000]])
        r = sim.simulate(t, nm.Profile(request_kbps=4000, link_kbps=8000))
        self.assertEqual([round(x.t_finish, 6) for x in r.requests], [0.4, 0.7, 0.9, 1.1])

    def test_latency_rounds_and_concurrency(self):
        t = sim.make_trace([[4096] * 4] * 3)
        self.assertAlmostEqual(sim.simulate(t, nm.Profile(latency_ms=100)).total_ms, 300)
        t = sim.make_trace([[4096] * 6])
        self.assertAlmostEqual(sim.simulate(t, nm.Profile(latency_ms=50, max_concurrent=2)).total_ms, 150)
        # cold connections: 2 opened, 2 x connect cost only on the first wave
        r = sim.simulate(t, nm.Profile(latency_ms=50, max_concurrent=2, connect_ms=100))
        self.assertAlmostEqual(r.total_ms, 250)
        self.assertEqual(len({x.conn for x in r.requests}), 2)

    def test_cpu_and_lower_bound(self):
        t = sim.make_trace([[1000], [1000], [1000]], cpu_ms=[0, 10, 20])
        p = nm.Profile(latency_ms=40, link_kbps=8000)
        self.assertAlmostEqual(sim.simulate(t, p).total_ms, 3 * 41 + 30)
        self.assertAlmostEqual(sim.lower_bound_ms(t, p), 3 * 41 + 30)

    def test_slow_start(self):
        # One RTT = 100 ms; 3 * init_cwnd needs exactly one extra round trip.
        p = nm.Profile(latency_ms=100, slow_start=True)
        t = sim.make_trace([[3 * 14600]])
        self.assertAlmostEqual(sim.simulate(t, p).total_ms, 200)
        t = sim.make_trace([[14600]])
        self.assertAlmostEqual(sim.simulate(t, p).total_ms, 100)
        t = sim.make_trace([[7 * 14600]])
        self.assertAlmostEqual(sim.simulate(t, p).total_ms, 300)

    def test_seeded_distributions(self):
        p = nm.Profile(latency_ms=100, latency_dist="lognormal", latency_sigma=0.5)
        t = sim.make_trace([[1]])
        a = [sim.simulate(t, p, s).total_ms for s in range(400)]
        b = [sim.simulate(t, p, s).total_ms for s in range(400)]
        self.assertEqual(a, b)
        self.assertAlmostEqual(sorted(a)[200], 100, delta=10)
        p = nm.Profile(latency_ms=100, tail_prob=0.1, tail_ms=1000)
        d = sim.simulate_many(t, p, range(2000))
        self.assertAlmostEqual(d["mean"], 200, delta=25)
        p = nm.Profile(latency_ms=50, latency_dist="pareto", latency_pareto_alpha=2.0)
        d = sim.simulate_many(t, p, range(2000))
        self.assertGreaterEqual(d["min"], 50)
        self.assertAlmostEqual(d["mean"], 100, delta=20)
        p = nm.Profile(latency_ms=50, latency_dist="normal", latency_jitter_ms=100, latency_min_ms=20)
        self.assertGreaterEqual(sim.simulate_many(t, p, range(500))["min"], 20)

    def test_link_jitter_mean(self):
        fs = [nm.link_capacity_factor(3, k, 0.5) for k in range(20000)]
        self.assertAlmostEqual(sum(fs) / len(fs), 1.0, delta=0.03)
        t = sim.make_trace([[2_000_000]])
        d = sim.simulate_many(t, nm.Profile(link_kbps=8000, link_jitter=0.5), range(50))
        self.assertGreater(d["stdev"], 0)

    def test_trace_formats(self):
        obj = {"file": "a.db", "reads": [{"offset": 0, "length": 10, "round": 0},
                                         {"offset": 10, "length": 10, "round": 0},
                                         {"offset": 100, "length": 10, "round": 1, "t_issue": 5}],
               "rounds": [{"round": 1, "cpu_ms": 2.5}]}
        t = sim.trace_from_obj(obj)
        self.assertEqual((len(t.reads), t.cpu_ms, t.reads[2].t_issue_ms, t.reads[0].file), (3, {1: 2.5}, 5.0, "a.db"))
        m = t.merged(0)
        self.assertEqual([(r.offset, r.length, r.round) for r in m.reads], [(0, 20, 0), (100, 10, 1)])
        t2 = sim.trace_from_obj([[{"offset": 0, "length": 1}], [{"offset": 5, "length": 1}]])
        self.assertEqual([r.round for r in t2.reads], [0, 1])
        p = nm.Profile(latency_ms=10)
        self.assertAlmostEqual(sim.simulate(t, p).total_ms, 10 + 2.5 + 5 + 10)


def _close(real_ms: float, sim_ms: float, rel: float = 0.10, abs_ms: float = 40.0) -> bool:
    return sim_ms - 5 <= real_ms <= sim_ms * (1 + rel) + abs_ms


class ServerMatchesSimulator(unittest.TestCase):
    """Replays traces through the real server and compares with simulate()."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.root = _make_root()

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.root)

    def compare(self, profile: nm.Profile, trace: sim.Trace, seed: int = 1, **tol) -> tuple[float, float]:
        async def run() -> float:
            async with rs.NetSimServer(self.root, profile, seed=seed) as srv:
                res = await rp.replay(trace, srv.url + "/f.bin", max_conns=profile.max_concurrent)
                return res.total_ms
        real = asyncio.run(run())
        model = sim.simulate(trace, profile, seed).total_ms
        self.assertTrue(_close(real, model, **tol), f"{profile.name}: real {real:.1f} ms vs sim {model:.1f} ms")
        return real, model

    def test_latency_only(self):
        self.compare(nm.Profile(name="lat", latency_ms=100), sim.make_trace([[4096] * 4] * 3))

    def test_bandwidth_only(self):
        self.compare(nm.Profile(name="bw", link_kbps=8000), sim.make_trace([[500_000]]))

    def test_per_request_cap(self):
        self.compare(nm.Profile(name="cap", request_kbps=4000), sim.make_trace([[100_000, 200_000]]))

    def test_concurrency_limited(self):
        self.compare(nm.Profile(name="conc", latency_ms=50, max_concurrent=2), sim.make_trace([[4096] * 6]))

    def test_shared_bandwidth(self):
        self.compare(nm.Profile(name="share", request_kbps=4000, link_kbps=8000),
                     sim.make_trace([[100_000, 200_000, 300_000, 400_000]]))

    def test_realistic_mix(self):
        tr = sim.make_trace([[300_000], [4096] * 8, [64_000] * 3], cpu_ms=[0, 5, 5])
        self.compare(nm.preset("4g,h1,cold"), tr)
        self.compare(nm.preset("lte,slowstart"), tr)

    def test_same_seed_same_latencies(self):
        # One request per round, so the server draws latencies in the same
        # order as the simulator and the totals agree for the same seed.
        p = nm.preset("none", latency_ms=60, latency_dist="lognormal", latency_sigma=0.6,
                      tail_prob=0.2, tail_ms=100)
        self.compare(p, sim.make_trace([[4096]] * 6), seed=7)

    def test_runtime_profile_switch(self):
        async def run() -> list[float]:
            out = []
            async with rs.NetSimServer(self.root) as srv:
                cl = rp.Client(srv.url + "/f.bin")
                for body in ({"preset": "none", "latency_ms": 120}, {"latency_ms": 30}):
                    st, _, d, _ = await cl.get("/__netsim/profile", method="POST",
                                               body=json.dumps(body).encode())
                    self.assertEqual(st, 200)
                    res = await rp.replay(sim.make_trace([[4096]] * 2), srv.url + "/f.bin", client=cl)
                    out.append(res.total_ms)
                await cl.close()
            return out
        a, b = asyncio.run(run())
        self.assertTrue(_close(a, 240), a)
        self.assertTrue(_close(b, 60), b)


if __name__ == "__main__":
    unittest.main()
