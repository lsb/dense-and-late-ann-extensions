# netsim: network simulation for HTTP range-request workloads

`netsim` provides two ways to estimate how long a sequence of HTTP range reads takes over a given network: a real HTTP server that delays and paces its responses (`rangeserver.py`), and a discrete-event simulator that computes the same timings in virtual time (`simulate.py`). Both import one model (`netmodel.py`), so a profile such as `4g,h1` means the same thing in both. The intended use is an httpvfs-style SQLite database read by a browser, for example a phone fetching database pages over a cellular network.

## Files

| File | Purpose |
|---|---|
| `netmodel.py` | Profiles and presets, latency sampling, the fluid link model, the connection-pool model. Shared by everything else. |
| `rangeserver.py` | Static-file HTTP/1.1 server (asyncio, standard library only) with ranges, CORS, optional COOP/COEP, shaping, request log, control endpoints. |
| `simulate.py` | Discrete-event simulator for read traces; trace loading; command-line sweeps over presets and seeds. |
| `replay.py` | Asyncio HTTP client that replays a trace against a real server with browser-like behaviour (keep-alive pool, FIFO limit, dependent rounds). |
| `test_netsim.py` | Range-semantics tests, model tests, and server-versus-simulator agreement tests. |
| `loadgen.js` | Node.js load generator used to measure server overhead. |

## Usage

```sh
# Serve a directory over a simulated 4G link with the HTTP/1.1 six-connection limit
python3 netsim/rangeserver.py --dir build/ --port 8000 --preset 4g,h1 --log requests.jsonl
# Override individual fields; cross-origin isolation for SharedArrayBuffer
python3 netsim/rangeserver.py --dir build/ --preset lte --latency-ms 90 --tail-prob 0.01 --tail-ms 800 --isolate

# Change the profile without restarting (fields without "preset" patch the current profile)
curl -X POST localhost:8000/__netsim/profile -d '{"preset": "3g,h1", "seed": 2}'
curl -X POST localhost:8000/__netsim/profile -d '{"latency_ms": 250}'
curl localhost:8000/__netsim/log?since=0          # in-memory request log

# Simulate a trace under one profile, 500 seeds, or sweep every preset
python3 netsim/simulate.py trace.json --preset lte-poor --seeds 500
python3 netsim/simulate.py trace.json --sweep --preset-mod h1 --json

# Replay the same trace against the real server
python3 netsim/replay.py trace.json --url http://127.0.0.1:8000/index.db --max-conns 6

# Tests
python3 -m pytest netsim         # or: python3 -m unittest netsim.test_netsim
```

In-process use, for benchmarks:

```python
from netsim import netmodel as nm, rangeserver as rs, simulate as sim, replay as rp
async with rs.NetSimServer("build/", nm.preset("4g,h1"), seed=1) as srv:   # srv.url
    srv.set_profile(nm.preset("3g"))
    res = await rp.replay(sim.load_trace("trace.json"), srv.url + "/index.db", max_conns=6)
with rs.ServerThread("build/", nm.preset("lte")) as srv:                     # from sync code
    ...
sim.simulate(sim.load_trace("trace.json"), nm.preset("4g,h1"), seed=0).total_ms
```

## The model

A request passes through three stages: it waits for a service slot, it waits for its latency, and its body then flows over a link shared with every other body in flight.

### Admission and connections

At most `max_concurrent` requests are in service at once (0 means unlimited). Further requests wait in a single FIFO queue. The limit models the browser's per-host connection limit for HTTP/1.1 (six in Chrome, Firefox and Safari) or an HTTP/2 server's `SETTINGS_MAX_CONCURRENT_STREAMS` (commonly 100). A request holds its slot from the moment it starts service until its last byte has been sent, including its latency.

Connections are opened lazily, as browsers do: a request that starts service while an idle connection exists reuses it; otherwise a new connection is opened and the request pays an extra `connect_ms`. In the server, `connect_ms` is paid by the first request on each TCP connection. The simulator assumes the client opens at most `max_concurrent` connections, which matches a browser but not an arbitrary client.

### Latency

For request *i*, the first byte leaves at

  t<sub>fb,i</sub> = t<sub>start,i</sub> + L<sub>i</sub> + [new connection] · connect_ms,

where t<sub>start,i</sub> = t<sub>arrive,i</sub> + queue wait. The latency L<sub>i</sub> is drawn from a seeded generator when the request starts service:

| `latency_dist` | L (before clamping) | Parameters |
|---|---|---|
| `fixed` | *L*<sub>0</sub> | `latency_ms` = *L*<sub>0</sub> |
| `normal` | *L*<sub>0</sub> + σ*Z* | σ = `latency_jitter_ms` |
| `lognormal` | *L*<sub>0</sub> · exp(σ*Z*), median *L*<sub>0</sub> | σ = `latency_sigma` |
| `exponential` | *L*<sub>0</sub> + Exp(mean *m*) | *m* = `latency_jitter_ms` |
| `pareto` | *L*<sub>0</sub> · *U*<sup>−1/α</sup>, minimum *L*<sub>0</sub>, mean α*L*<sub>0</sub>/(α−1) | α = `latency_pareto_alpha` |

The value is clamped to [`latency_min_ms`, `latency_max_ms`]. Then, with probability `tail_prob`, `tail_ms` is added; this stands in for events the model does not represent directly, such as a retransmission timeout or a radio state change. On a warm keep-alive connection one request costs about one round-trip time, so `latency_ms` in the presets is the RTT.

### Bandwidth: a fluid link with max-min fair sharing

After its first byte, each response body is a *flow* of *S*<sub>i</sub> bytes over one shared bottleneck of capacity *C* (`link_kbps`), with an optional per-flow cap *c*<sub>i</sub> (`request_kbps`, optionally multiplied by a per-request lognormal factor with σ = `request_kbps_jitter`). Between events every active flow moves at a constant rate given by max-min fair sharing (water-filling): with the flows sorted by cap,

  r<sub>i</sub> = min(c<sub>i</sub>, (C − Σ<sub>j capped</sub> r<sub>j</sub>) / (number of flows not yet capped)).

So *n* uncapped parallel requests split the link equally, and a flow finishes at the first time *T* where ∫<sub>t<sub>fb</sub></sub><sup>T</sup> r<sub>i</sub>(t) dt = *S*<sub>i</sub>. Rates are recomputed at every event: a flow starting, a flow finishing, a slow-start step (below), or a capacity change. With a single request and no competition this reduces to

  T = queue wait + L + S / min(C, c).

For a round of *n* equal requests of *S* bytes with fixed latency and no queueing, the round takes L + nS / min(C, n·c); `simulate.lower_bound_ms` sums this over rounds.

**Capacity jitter.** With `link_jitter` σ > 0, time is split into periods of `link_jitter_period_ms` and the capacity in period *k* is *C* · exp(σ*Z*<sub>k</sub> − σ²/2), whose mean is *C*. *Z*<sub>k</sub> is a hash of (seed, *k*), so the capacity trace does not depend on the order in which requests arrive. It models the fluctuating throughput of a cellular link.

**Slow start (optional).** With `slow_start` on, each response starts with a congestion window of `init_cwnd_bytes` (14,600 bytes: ten 1,460-byte segments, the Linux default following RFC 6928) that doubles every round trip (`ss_rtt_ms`, defaulting to `latency_ms`). It is modelled as a ceiling on cumulative bytes: by round trip *k* after the first byte at most init · (2<sup>k+1</sup> − 1) bytes may have been sent, still subject to the link share. Thus 14.6 KB costs no extra round trip, 43.8 KB one, and 102 KB two, on an otherwise unlimited link. The window is reset for every request, which is pessimistic for keep-alive connections (Linux keeps the window unless the connection has been idle longer than the retransmission timeout; `tcp_slow_start_after_idle`). There is no congestion-avoidance phase and no loss.

### Randomness and seeds

All random draws come from the profile's seed. The simulator draws a request's latency and cap in the order requests start service; the server draws them in the order requests obtain a slot, which follows network arrival order. The two therefore produce identical totals for the same seed only when that order is deterministic (for example one request per round); otherwise they agree in distribution. `simulate_many` and `--seeds N` give the mean, standard deviation and percentiles over seeds.

### How the server realises the model

The server runs the same `FluidLink` against the monotonic clock. After a response's latency has elapsed it writes the headers, registers the body as a flow, and repeatedly advances the link to the current time and writes the bytes the model says have been sent, in chunks of at least 16 KB or 5 ms of data. Between chunks it sleeps until the predicted time of the next chunk, and wakes early whenever the set of flows changes. The recorded finish time is when the last byte has been handed to the kernel. With no bandwidth shaping it writes header and body together without the link.

## Presets

`latency_ms` is the per-request latency on a warm connection (about one RTT) and `link_kbps` the downlink capacity shared by all responses. Upload bandwidth is not modelled because range requests are small. Presets can be combined with modifiers and overridden by flags: `--preset lte,h1,jitter --latency-ms 90`.

| Preset | Latency (ms) | Downlink (kbps) | Source / justification |
|---|---:|---:|---|
| `none` | 0 | unlimited | No shaping. |
| `lan` | 1 | 1,000,000 | Gigabit Ethernet on one switch. |
| `fios` | 4 | 20,000 | WebPageTest "FIOS" connectivity profile. |
| `cable` | 28 | 5,000 | WebPageTest "Cable". |
| `dsl` | 50 | 1,500 | WebPageTest "DSL". |
| `wifi` | 20 | 50,000 | Illustrative home Wi-Fi to a nearby CDN edge; not a standard profile. |
| `5g` | 40 | 100,000 | Illustrative. Commercial 5G (mostly non-standalone, mid-band) is typically reported by Ookla and Opensignal with download medians of the order of 100 Mbps or more and latencies of a few tens of ms; the radio link, not the core, dominates latency. |
| `starlink` | 45 | 100,000 | Illustrative low-Earth-orbit service; published measurements report RTTs of roughly 25–60 ms and download rates of the order of 100 Mbps. |
| `lte` | 70 | 12,000 | WebPageTest "LTE". |
| `4g` | 165 | 8,100 | Chrome DevTools "Fast 4G" (9 Mbps × 0.9, 60 ms × 2.75). |
| `wpt-4g` | 170 | 9,000 | WebPageTest "4G". |
| `lighthouse` | 150 | 1,638.4 | Lighthouse mobile simulated throttling ("Slow 4G": 150 ms RTT, 1.6 × 1024 kbps). |
| `slow-4g` | 562.5 | 1,440 | Chrome DevTools "Slow 4G", formerly named "Fast 3G" (1.6 Mbps × 0.9, 150 ms × 3.75). |
| `3g` | 300 | 1,600 | WebPageTest "3G". |
| `slow-3g` | 2,000 | 400 | Chrome DevTools "Slow 3G" (500 kbps × 0.8, 400 ms × 5). |
| `edge` | 840 | 240 | WebPageTest "Edge". |
| `satellite` | 600 | 10,000 | Geostationary satellite. A request and response each go up to and down from 35,786 km, so propagation alone is about 4 × 35,786 km / c ≈ 480 ms; measured RTTs on such services are typically around 600 ms or more. |
| `lte-poor` | 120 (median) | 2,000 (mean) | Probabilistic, illustrative cell-edge LTE: lognormal latency (σ = 0.5, max 5 s), 2% chance of +1 s, capacity jitter σ = 0.5 every 200 ms. |

Chrome DevTools applies its latency once per request rather than per packet, and its numbers include calibration multipliers taken from Lighthouse so that request-level throttling approximates packet-level throttling; that is the same granularity as this model. WebPageTest profiles are packet-level (RTT added to every packet with `netem`/`ipfw`-style shaping), so a WebPageTest RTT maps to one request latency on a warm connection plus extra round trips for handshakes, which is what the `cold` modifier adds. The Chrome values are those in Chromium's `devtools-frontend` (`NetworkManager.ts`) and the WebPageTest values are from its published connectivity profiles; both were checked on 2026-09-23 against `front_end/core/sdk/NetworkManager.ts` in ChromeDevTools/devtools-frontend (main) and `www/settings/connectivity.ini.sample` in catchpoint/WebPageTest (master), and match exactly.

| Modifier | Effect |
|---|---|
| `h1` | `max_concurrent` = 6 (browser HTTP/1.1 per-host limit). |
| `h2` | `max_concurrent` = 100 (typical HTTP/2 stream limit). |
| `jitter` | Lognormal latency σ = 0.25, 1% chance of +500 ms, link capacity jitter σ = 0.25. |
| `slowstart` | Per-request TCP slow start. |
| `cold` | `connect_ms` = 2 × latency: TCP handshake plus TLS 1.3 handshake on each new connection. |

## Trace format

A SQLite VFS (for example a WASM httpvfs implementation) can record its reads in the following JSON format; `simulate.load_trace` and `replay.py` read it.

```json
{
  "version": 1,
  "file": "index.db",
  "file_size": 123456789,
  "page_size": 4096,
  "description": "query 17, cold cache",
  "reads": [
    {"offset": 0,     "length": 4096,  "round": 0},
    {"offset": 8192,  "length": 4096,  "round": 1},
    {"offset": 65536, "length": 32768, "round": 1, "t_issue": 0.4},
    {"offset": 0,     "length": 1024,  "round": 2, "file": "other.db"}
  ],
  "rounds": [
    {"round": 1, "cpu_ms": 0.8},
    {"round": 2, "cpu_ms": 3.5}
  ],
  "tail_cpu_ms": 1.2
}
```

- `reads` (required): one entry per HTTP range request. `offset` and `length` are in bytes; the request is `Range: bytes=offset-(offset+length-1)`.
- `round` (integer, default 0): the dependency level. All reads of round *k* are issued only after every read of round *k*−1 has completed. Rounds are processed in increasing numeric order and need not be contiguous. Reads within one round are issued in parallel.
- `t_issue` (milliseconds, optional, default 0): delay of this read after its round starts, for a client that issues reads of a round in a staggered way. It is relative to the round start, not absolute. A recorder that knows absolute issue times can store `t_issue` = issue time − earliest issue time in the round.
- `file` (optional, top level or per read): the file read; `replay.py` substitutes it for the last path component of the URL. The simulator treats all files as behind the same link.
- `rounds` (optional): CPU time spent before the round is issued (`cpu_ms`), for example decoding pages and choosing the next ones. `tail_cpu_ms` is CPU time after the last round.
- Other top-level keys are kept as metadata.

Two shorthand forms are also accepted: a bare JSON list of reads, and a list of rounds each of which is a list of reads (the round number is the list position). A JSON-lines file, with one read object per line and optional `{"round": r, "cpu_ms": x}` lines, is accepted as well. `simulate.py --merge-gap N` coalesces reads of the same round and file that lie within *N* bytes of each other into one request, to evaluate read coalescing without re-recording.

The simulator's time for a trace is

  T = Σ<sub>rounds k</sub> [ cpu<sub>k</sub> + (completion time of the last read of round k − start of round k) ] + tail_cpu,

with each round's reads simulated through admission, latency and the shared link as above. Connection state (which connections are open) persists across rounds; flows never overlap between rounds, because a round cannot start before the previous one finishes.

## Request log

With `--log FILE` the server appends one JSON object per request (and keeps the last 100,000 in memory, served at `/__netsim/log?since=ID`):

```json
{"id": 12, "conn": 3, "method": "GET", "path": "/index.db", "status": 206,
 "range": "bytes=8192-12287", "ranges": [[8192, 12287]], "bytes": 4096, "size": 123456789,
 "t_arrive_ms": 1532.118, "queue_ms": 0.004, "latency_ms": 165.0, "new_conn": false,
 "t_first_byte_ms": 1697.301, "t_finish_ms": 1701.412, "wall_arrive": 1790204123.51, "profile": "4g,h1"}
```

Times are milliseconds since the server started (monotonic clock); `wall_arrive` is the Unix time of arrival. `latency_ms` includes `connect_ms` when `new_conn` is true. Each response also carries `X-Netsim-Request: <id>` so a client can join its own timings with the log.

## HTTP behaviour

- GET, HEAD, OPTIONS; other methods get 405. HTTP/1.1 keep-alive by default, HTTP/1.0 keep-alive on request.
- `Range: bytes=a-b`, `bytes=a-`, `bytes=-n`; a last position beyond the end is clamped. A syntactically invalid header (for example `bytes=5-3` or a unit other than bytes) is ignored and the whole file is sent with 200, as RFC 9110 requires. If no range is satisfiable (`bytes=SIZE-`, `bytes=-0`, any range of an empty file) the response is 416 with `Content-Range: bytes */SIZE`. Several ranges give a `multipart/byteranges` response; ranges are not merged or reordered.
- `ETag` (from modification time and size), `Last-Modified`, `If-Range` (a mismatch sends the full file), `If-None-Match` (304). `Cache-Control: no-store` by default so a browser benchmark does not hit its HTTP cache; change it with `--cache-control`.
- CORS: `Access-Control-Allow-Origin: *`, `Access-Control-Expose-Headers` with Content-Range, Content-Length, Accept-Ranges, ETag and the request id, and `Timing-Allow-Origin: *` so the Resource Timing API reports detailed cross-origin timings. OPTIONS preflights get 204 with the requested headers allowed. `Cross-Origin-Resource-Policy: cross-origin` is always sent; `--isolate` adds `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp`, which make the page cross-origin isolated so it can use `SharedArrayBuffer`.
- Paths are URL-decoded and resolved inside the served directory (403 outside it); a directory serves its `index.html`. `.wasm` is served as `application/wasm`.
- Control endpoints under `/__netsim/` are never shaped: `profile` (GET, POST), `presets`, `log`, `log/clear`, `stats`.

## Measurements

All measurements were taken on the project machine (4 vCPUs) while an unrelated LLM generation job kept all four CPUs busy (load average about 4.7), so they are conservative.

**Server overhead without shaping** (`--preset none`, 5 MB file in the page cache, Node.js `loadgen.js` with keep-alive connections, random offsets, 4 s per point):

| Workload | Throughput | Latency p50 / p99 |
|---|---:|---:|
| 4 KB ranges, 1 connection | 3,650 requests/s | 0.21 / 0.65 ms |
| 4 KB ranges, 6 connections | 4,890 requests/s | 0.98 / 5.2 ms |
| 4 KB ranges, 32 connections | 6,170 requests/s | 4.8 / 14 ms |
| 1 MB ranges, 1 connection | 600 MB/s | 1.5 / 4.4 ms |
| 1 MB ranges, 4 connections | 610 MB/s | 6.0 / 14 ms |
| Whole 5 MB file, curl | 604 MB/s | |

With the shaping path active but non-binding (`--link-kbps 8000000`, that is 1 GB/s), a single connection achieved 2,960 requests/s of 4 KB (p50 0.27 ms) and four connections 680 MB/s of 1 MB ranges. The fastest shaped preset, `lan`, is 1 ms and 125 MB/s, so the server adds well under one millisecond per request and is not the bottleneck for any preset. The per-request cost is higher than for a C server, and a single asyncio process uses one core; at thousands of requests per second the client, not the server, is usually the limit.

**Server versus simulator.** Traces were replayed with `replay.py` (same process and event loop as the server) and compared with `simulate.simulate`, same seed, three runs each. The "mixed" trace has three rounds: one 300 KB read; eight 4 KB reads; three 64 KB reads, with 5 ms of CPU before rounds 2 and 3.

| Profile and trace | Simulated (ms) | Real, min (ms) | Real, max (ms) |
|---|---:|---:|---:|
| latency 100 ms; 3 rounds of 4 × 4 KB | 300.0 | 303.4 | 309.1 |
| link 8 Mbps; 1 × 500 KB | 500.0 | 501.8 | 502.5 |
| latency 50 ms, max 2 concurrent; 6 × 4 KB | 150.0 | 152.4 | 152.6 |
| 4 Mbps per request, 8 Mbps link; 100, 200, 300, 400 KB in parallel | 1100.0 | 1101.9 | 1102.5 |
| `4g,h1,cold`; mixed | 1675.2 | 1681.6 | 1685.7 |
| `lte,slowstart`; mixed | 764.1 | 770.1 | 771.5 |
| `3g`; mixed | 3533.8 | 3543.8 | 3545.1 |

The real server is consistently 1–3 ms per round slower than the model (timer granularity, the event loop and the Python client), and never faster. The tests accept real = simulated + at most 10% + 40 ms.

**Simulator speed.** A trace of 50 rounds of 20 reads (1,000 requests) takes about 25 ms per simulated run under `4g`, and about 45 ms under `lte-poor` (capacity jitter adds an event every 200 ms), so sweeping every preset over hundreds of seeds takes seconds to minutes.

## Caveats and open issues

- **HTTP/2 is modelled, not spoken.** The server speaks HTTP/1.1 only. HTTP/2 multiplexing is represented by a higher `max_concurrent` (`h2`), without HPACK, framing overhead, stream prioritisation, or the fact that all streams share one TCP connection (and hence one congestion window, and head-of-line blocking on loss). A browser talking to this server over HTTP/1.1 enforces its own limit of six connections per host whatever the profile says; to measure an `h2`-like profile from a real browser, spread requests over several host names (`127.0.0.1`, `localhost`, `127.0.0.2`, …) or rely on the simulator.
- **Latency is per request, not per packet.** Transfer starts at full rate right after the first byte. Without `slow_start`, the model therefore underestimates large transfers on high-RTT links; with it, it overestimates on warm connections (the window is reset per request). There is no packet loss or congestion-avoidance model; tail events approximate retransmission timeouts.
- **Headers are free.** Only body bytes count against bandwidth (about 400 bytes of response headers per request are ignored; `simulate(..., body_overhead=N)` can add them in the simulator only). Request bytes and the uplink are ignored.
- **CORS preflights.** The server applies latency to OPTIONS requests (a preflight is a real round trip), but the simulator does not model preflights. Current versions of the Fetch standard treat a single simple `Range` header as CORS-safelisted, so modern browsers should not preflight plain range GETs; other headers such as `If-Range` or `Cache-Control` do trigger preflights.
- **Connection accounting.** `connect_ms` is charged per TCP connection by the server, and per lazily opened connection (at most `max_concurrent`) by the simulator. The two agree when the client limits its connections to `max_concurrent`, as `replay.py` does when given `--max-conns` equal to the profile's limit. With `max_concurrent` = 0 and a browser, set `h1` in the simulator to match the browser's own limit.
- **Clock alignment.** Capacity-jitter periods start at server start in the server and at trace start in the simulator, so individual runs with `link_jitter` differ even with the same seed; compare distributions.
- **Blocking file reads.** File reads use `os.pread` on the event-loop thread. For files in the page cache this costs microseconds; a cold read from a slow disk would stall all connections briefly.
- **Timer resolution.** The server's sleeps are accurate to about a millisecond; on a heavily loaded machine the tolerance of the timing tests may need to be widened.
