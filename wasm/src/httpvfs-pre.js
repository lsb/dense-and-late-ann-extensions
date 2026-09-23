// httpvfs-pre.js -- JavaScript backend of the httpvfs VFS (included with
// --pre-js). The C side calls Module.httpvfsFetch with a batch of byte ranges
// and waits (Asyncify or JSPI) until all of them have arrived; the ranges are
// fetched concurrently, so one batch costs one network round trip.
//
// Files are registered by name before sqlite3_open_v2():
//   Module.httpvfsRegister('/httpvfs/1', { url, fetch, headers, maxParallel })

Module.httpvfsFiles = new Map();   // name -> config
Module.httpvfsHandles = [];        // handle -> config (null when closed)

Module.httpvfsRegister = function (name, cfg) {
  Module.httpvfsFiles.set(name, cfg);
};

Module.httpvfsUnregister = function (name) {
  Module.httpvfsFiles.delete(name);
};

Module.httpvfsOpen = function (name) {
  const cfg = Module.httpvfsFiles.get(name);
  if (!cfg) return -1;
  Module.httpvfsHandles.push(cfg);
  return Module.httpvfsHandles.length - 1;
};

Module.httpvfsClose = function (h) {
  Module.httpvfsHandles[h] = null;
};

function httpvfsParseTotal(resp) {
  const cr = resp.headers.get('content-range');
  if (cr) {
    const m = /\/(\d+)\s*$/.exec(cr);
    if (m) return Number(m[1]);
  }
  return -1;
}

async function httpvfsFetchOne(cfg, off, len) {
  const fetchFn = cfg.fetch || globalThis.fetch;
  const headers = Object.assign({}, cfg.headers || {}, {
    Range: `bytes=${off}-${off + len - 1}`,
  });
  const init = { headers };
  if (cfg.cache) init.cache = cfg.cache;
  const resp = await fetchFn(cfg.url, init);
  let buf = new Uint8Array(await resp.arrayBuffer());
  let total = -1;
  if (resp.status === 206) {
    total = httpvfsParseTotal(resp);
  } else if (resp.status === 200) {
    // The server ignored the Range header and sent the whole file.
    total = buf.length;
    if (!cfg.warnedNoRange) {
      cfg.warnedNoRange = true;
      console.warn(`httpvfs: ${cfg.url} does not honour Range requests`);
    }
    buf = buf.subarray(off, off + len);
  } else {
    throw new Error(`httpvfs: HTTP ${resp.status} for ${cfg.url} bytes ${off}+${len}`);
  }
  return { buf, total };
}

// Returns 0 (SQLITE_OK) or 10 (SQLITE_IOERR).
Module.httpvfsFetch = async function (h, n, offPtr, lenPtr, destPtr, t0Ptr, t1Ptr, sizePtr) {
  const cfg = Module.httpvfsHandles[h];
  if (!cfg) return 10;
  const offs = [], lens = [], dests = [];
  for (let i = 0; i < n; i++) {
    offs.push(HEAPF64[(offPtr >> 3) + i]);
    lens.push(HEAP32[(lenPtr >> 2) + i]);
    dests.push(HEAPU32[(destPtr >> 2) + i]);
  }
  const results = new Array(n);
  const t0 = new Array(n), t1 = new Array(n);
  const maxParallel = cfg.maxParallel || n;
  let next = 0, failed = null;
  async function worker() {
    while (next < n && !failed) {
      const i = next++;
      t0[i] = performance.now();
      try {
        results[i] = await httpvfsFetchOne(cfg, offs[i], lens[i]);
      } catch (e) {
        failed = e;
      }
      t1[i] = performance.now();
    }
  }
  const workers = [];
  for (let k = 0; k < Math.min(maxParallel, n); k++) workers.push(worker());
  await Promise.all(workers);
  if (failed) {
    console.error(failed);
    return 10;
  }
  // Memory may have grown while we awaited: always use the current views.
  for (let i = 0; i < n; i++) {
    const { buf, total } = results[i];
    // A response may be short only at the end of the file (the first request
    // is made before the size is known).
    const expect = total >= 0 ? Math.min(lens[i], total - offs[i]) : lens[i];
    if (buf.length !== expect) {
      console.error(`httpvfs: response of ${buf.length} bytes, expected ${expect}, at ${offs[i]}`);
      return 10;
    }
    HEAPU8.set(buf, dests[i]);
    if (buf.length < lens[i]) HEAPU8.fill(0, dests[i] + buf.length, dests[i] + lens[i]);
    HEAPF64[(t0Ptr >> 3) + i] = t0[i];
    HEAPF64[(t1Ptr >> 3) + i] = t1[i];
    if (total >= 0) HEAPF64[sizePtr >> 3] = total;
  }
  if (cfg.onRound) cfg.onRound({ n, offs, lens, t0, t1 });
  return 0;
};

// Synchronous variant (build "sync", no Asyncify): cfg.fetchSync(offs, lens)
// must return [{buf, total}] for the whole batch before returning, e.g. by
// waiting with Atomics.wait while a worker fetches the ranges in parallel.
// Returns 0 (SQLITE_OK) or 10 (SQLITE_IOERR).
Module.httpvfsFetchSync = function (h, n, offPtr, lenPtr, destPtr, t0Ptr, t1Ptr, sizePtr) {
  const cfg = Module.httpvfsHandles[h];
  if (!cfg || !cfg.fetchSync) return 10;
  const offs = [], lens = [];
  for (let i = 0; i < n; i++) {
    offs.push(HEAPF64[(offPtr >> 3) + i]);
    lens.push(HEAP32[(lenPtr >> 2) + i]);
  }
  let results;
  const t0 = performance.now();
  try {
    results = cfg.fetchSync(cfg.url, offs, lens);
  } catch (e) {
    console.error(e);
    return 10;
  }
  const t1 = performance.now();
  for (let i = 0; i < n; i++) {
    const { buf, total } = results[i];
    const dest = HEAPU32[(destPtr >> 2) + i];
    const expect = total >= 0 ? Math.min(lens[i], total - offs[i]) : lens[i];
    if (buf.length !== expect) return 10;
    HEAPU8.set(buf, dest);
    if (buf.length < lens[i]) HEAPU8.fill(0, dest + buf.length, dest + lens[i]);
    HEAPF64[(t0Ptr >> 3) + i] = results[i].t0 ?? t0;
    HEAPF64[(t1Ptr >> 3) + i] = results[i].t1 ?? t1;
    if (total >= 0) HEAPF64[sizePtr >> 3] = total;
  }
  return 0;
};
