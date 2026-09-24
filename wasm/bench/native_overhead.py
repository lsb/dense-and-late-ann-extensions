"""Native baseline for wasm/bench/overhead.mjs: time the same queries with
Python's sqlite3 module on SQLite 3.53.4 (run with LD_LIBRARY_PATH=build/native),
both on the default VFS and on httpvfs (local-file backend, warm cache).
Prints JSON."""
import json, sqlite3, sys, time

QUERIES = json.loads(sys.argv[2])
db_path = sys.argv[1]
iters, warmup = 200, 20

def bench(conn):
    out = {}
    for name, sql in QUERIES.items():
        for _ in range(warmup):
            conn.execute(sql).fetchall()
        t = []
        for _ in range(iters):
            t0 = time.perf_counter()
            conn.execute(sql).fetchall()
            t.append((time.perf_counter() - t0) * 1e3)
        t.sort()
        out[name] = {"median_ms": t[len(t) // 2], "mean_ms": sum(t) / len(t)}
    return out

res = {"sqlite_version": sqlite3.sqlite_version}
res["native_default_vfs"] = bench(sqlite3.connect(f"file:{db_path}?mode=ro", uri=True))
c = sqlite3.connect(":memory:")
c.enable_load_extension(True)
c.load_extension(sys.argv[3])
res["native_httpvfs"] = bench(sqlite3.connect(f"file:{db_path}?vfs=httpvfs", uri=True))
print(json.dumps(res))
