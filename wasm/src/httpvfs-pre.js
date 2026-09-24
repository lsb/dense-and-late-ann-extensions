// httpvfs-pre.js -- JavaScript backend of the httpvfs VFS (included with
// --pre-js, after wasm/pkg/multipart.mjs, whose hv* functions it uses). The C
// side calls Module.httpvfsFetch with a batch of byte ranges and waits
// (Asyncify or JSPI) until all of them have arrived; the ranges are fetched
// concurrently, so one batch costs one network round trip.
//
// Files are registered by name before sqlite3_open_v2():
//   Module.httpvfsRegister('/httpvfs/1', { url, fetch, headers, maxParallel })

Module.httpvfsFiles = new Map();   // name -> config
// url -> true / false: whether the server answers multi-range requests
// (shared by every connection to the same file; unknown until the first one).
Module.httpvfsMultipart = new Map();
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

// Exposed for tests.
Module.httpvfsParseMultipart = hvParseMultipart;
Module.httpvfsFetchMultiRange = hvFetchMultiRange;

function httpvfsInit(cfg) {
  // cache: 'no-store' matters: Chromium's HTTP cache takes a per-URL lock,
  // which serialises concurrent range requests for the same file.
  return { headers: cfg.headers || {}, cache: cfg.cache || 'no-store' };
}

function httpvfsNoRange(cfg) {
  return () => {
    if (!cfg.warnedNoRange) {
      cfg.warnedNoRange = true;
      console.warn(`httpvfs: ${cfg.url} does not honour Range requests`);
    }
  };
}

// Fetch the ranges idx (indices into offs/lens) as one request: a plain range
// request for one range, a multi-range request for several (or, once the
// server is known not to support those, one request per range).
async function httpvfsFetchGroup(cfg, idx, offs, lens) {
  const fetchFn = cfg.fetch || globalThis.fetch;
  const init = httpvfsInit(cfg);
  const o = idx.map((i) => offs[i]), l = idx.map((i) => lens[i]);
  if (idx.length > 1 && Module.httpvfsMultipart.get(cfg.url) === false) {
    return Promise.all(o.map((off, k) => hvFetchRange(fetchFn, cfg.url, init, off, l[k], httpvfsNoRange(cfg))));
  }
  const { results, fallback } = await hvFetchMultiRange(fetchFn, cfg.url, init, o, l, httpvfsNoRange(cfg));
  if (idx.length > 1) {
    if (fallback && Module.httpvfsMultipart.get(cfg.url) !== false) {
      Module.httpvfsMultipart.set(cfg.url, false);
      console.warn(`httpvfs: ${cfg.url}: no multi-range support, using single ranges`);
    } else if (!fallback && !Module.httpvfsMultipart.has(cfg.url)) {
      Module.httpvfsMultipart.set(cfg.url, true);
    }
  }
  return results;
}

function httpvfsReadGroups(n, grpPtr) {
  let grp = null;
  if (grpPtr) {
    grp = new Array(n);
    for (let i = 0; i < n; i++) grp[i] = HEAP32[(grpPtr >> 2) + i];
  }
  return hvGroups(n, grp);
}

// Returns 0 (SQLITE_OK) or 10 (SQLITE_IOERR). grpPtr (may be 0): ranges with
// equal consecutive group numbers go into one multi-range request. Sets bit 1
// of *flagsPtr when the server turned out not to support multi-range requests.
Module.httpvfsFetch = async function (h, n, offPtr, lenPtr, destPtr, t0Ptr, t1Ptr, sizePtr, grpPtr, flagsPtr) {
  const cfg = Module.httpvfsHandles[h];
  if (!cfg) return 10;
  const offs = [], lens = [], dests = [];
  for (let i = 0; i < n; i++) {
    offs.push(HEAPF64[(offPtr >> 3) + i]);
    lens.push(HEAP32[(lenPtr >> 2) + i]);
    dests.push(HEAPU32[(destPtr >> 2) + i]);
  }
  const groups = httpvfsReadGroups(n, grpPtr);
  const results = new Array(n);
  const t0 = new Array(n), t1 = new Array(n);
  let failed = null;
  async function one(idx) {
    const ts = performance.now();
    try {
      const got = await httpvfsFetchGroup(cfg, idx, offs, lens);
      idx.forEach((i, k) => { results[i] = got[k]; });
    } catch (e) {
      failed = e;
    }
    const te = performance.now();
    for (const i of idx) { t0[i] = ts; t1[i] = te; }
  }
  // Until the server is known to answer multi-range requests, send one of
  // them alone first: a server that ignores them sends the whole file (the
  // response is aborted after its headers, but only once).
  const queue = groups.slice();
  if (!Module.httpvfsMultipart.has(cfg.url)) {
    const k = queue.findIndex((g) => g.length > 1);
    if (k >= 0) {
      await one(queue[k]);
      queue.splice(k, 1);
    }
  }
  const nreq = queue.length;
  const maxParallel = cfg.maxParallel || nreq;
  let next = 0;
  async function worker() {
    while (next < nreq && !failed) await one(queue[next++]);
  }
  const workers = [];
  for (let k = 0; k < Math.min(maxParallel, nreq); k++) workers.push(worker());
  await Promise.all(workers);
  if (failed) {
    cfg.lastError = failed;   // reported by the JS API with the SQLite error
    return 10;
  }
  if (Module.httpvfsMultipart.get(cfg.url) === false && flagsPtr) HEAP32[flagsPtr >> 2] |= 1;
  // Memory may have grown while we awaited: always use the current views.
  for (let i = 0; i < n; i++) {
    const { buf, total } = results[i];
    // A response may be short only at the end of the file (the first request
    // is made before the size is known).
    const expect = total >= 0 ? Math.min(lens[i], total - offs[i]) : lens[i];
    if (buf.length !== expect) {
      cfg.lastError = new Error(`httpvfs: response of ${buf.length} bytes, expected ${expect}, at ${offs[i]}`);
      return 10;
    }
    HEAPU8.set(buf, dests[i]);
    if (buf.length < lens[i]) HEAPU8.fill(0, dests[i] + buf.length, dests[i] + lens[i]);
    HEAPF64[(t0Ptr >> 3) + i] = t0[i];
    HEAPF64[(t1Ptr >> 3) + i] = t1[i];
    if (total >= 0) HEAPF64[sizePtr >> 3] = total;
  }
  if (cfg.onRound) cfg.onRound({ n, offs, lens, t0, t1, groups });
  return 0;
};

// Synchronous variant (build "sync", no Asyncify): cfg.fetchSync(url, offs,
// lens, groups) must return [{buf, total, t0?, t1?}] for the whole batch
// before returning, e.g. by waiting with Atomics.wait while a worker fetches
// the ranges in parallel; groups ([[i, ...], ...]) are the requests to make.
// It may set fetchSync.multipartFailed. Returns 0 (SQLITE_OK) or 10.
Module.httpvfsFetchSync = function (h, n, offPtr, lenPtr, destPtr, t0Ptr, t1Ptr, sizePtr, grpPtr, flagsPtr) {
  const cfg = Module.httpvfsHandles[h];
  if (!cfg || !cfg.fetchSync) return 10;
  const offs = [], lens = [];
  for (let i = 0; i < n; i++) {
    offs.push(HEAPF64[(offPtr >> 3) + i]);
    lens.push(HEAP32[(lenPtr >> 2) + i]);
  }
  const groups = httpvfsReadGroups(n, grpPtr);
  let results;
  const t0 = performance.now();
  try {
    results = cfg.fetchSync(cfg.url, offs, lens, groups);
  } catch (e) {
    cfg.lastError = e;
    return 10;
  }
  const t1 = performance.now();
  if (results.multipartFailed && flagsPtr) HEAP32[flagsPtr >> 2] |= 1;
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
