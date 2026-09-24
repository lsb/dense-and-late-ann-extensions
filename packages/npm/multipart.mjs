// Range fetching shared by the httpvfs JavaScript backends: single ranges,
// and multi-range requests (Range: bytes=a-b,c-d,... answered with
// multipart/byteranges) with a fallback for servers that do not support them.
//
// This file is an ES module (used by sync-fetch-worker.mjs and the tests) and
// is also compiled into the WebAssembly glue: the build strips the `export`
// keywords and passes it to emcc as a --pre-js. Keep it free of imports and
// of top-level code other than declarations.

const HV_LATIN1 = new TextDecoder('latin1');

/** Parse "bytes a-b/total" (total may be "*"): {start, end, total} or null. */
export function hvParseContentRange(s) {
  const m = /^\s*bytes\s+(\d+)-(\d+)\s*\/\s*(\d+|\*)\s*$/i.exec(s || '');
  if (!m) return null;
  return { start: Number(m[1]), end: Number(m[2]), total: m[3] === '*' ? -1 : Number(m[3]) };
}

/** The boundary of a multipart/byteranges Content-Type, or null. */
export function hvMultipartBoundary(contentType) {
  if (!/^\s*multipart\/byteranges\b/i.test(contentType || '')) return null;
  const m = /;\s*boundary\s*=\s*(?:"([^"]+)"|([^;\s]+))/i.exec(contentType);
  return m ? (m[1] ?? m[2]) : null;
}

function hvIndexOf(hay, needle, from) {
  const n = needle.length, last = hay.length - n, first = needle[0];
  outer: for (let i = Math.max(0, from); i <= last; i++) {
    if (hay[i] !== first) continue;
    for (let j = 1; j < n; j++) if (hay[i + j] !== needle[j]) continue outer;
    return i;
  }
  return -1;
}

/**
 * Split a multipart/byteranges body into its parts:
 * [{start, end, total, data}] (end inclusive; data a view into body). Each
 * part's length is taken from its Content-Range, so binary data that happens
 * to contain the boundary is handled. Throws on a malformed body.
 */
export function hvParseMultipart(body, boundary) {
  const delim = new TextEncoder().encode('--' + boundary);
  const parts = [];
  let pos = hvIndexOf(body, delim, 0);
  if (pos < 0) throw new Error('multipart: boundary not found');
  for (;;) {
    pos += delim.length;
    if (body[pos] === 0x2d && body[pos + 1] === 0x2d) break;   // "--": last one
    // End of the delimiter line (transport padding allowed), then headers.
    const eol = hvIndexOf(body, [0x0a], pos);
    if (eol < 0) throw new Error('multipart: truncated part header');
    const hs = eol + 1;
    let he = hvIndexOf(body, [0x0d, 0x0a, 0x0d, 0x0a], hs), skip = 4;
    const lf = hvIndexOf(body, [0x0a, 0x0a], hs);
    if (lf >= 0 && (he < 0 || lf < he)) { he = lf; skip = 2; }
    if (he < 0) throw new Error('multipart: truncated part header');
    const headers = HV_LATIN1.decode(body.subarray(hs, Math.max(hs, he)));
    let cr = null;
    for (const line of headers.split(/\r?\n/)) {
      const k = line.indexOf(':');
      if (k > 0 && line.slice(0, k).trim().toLowerCase() === 'content-range') cr = hvParseContentRange(line.slice(k + 1));
    }
    if (!cr) throw new Error('multipart: part without a valid Content-Range');
    const ds = he + skip, len = cr.end - cr.start + 1;
    if (ds + len > body.length) throw new Error('multipart: truncated part body');
    parts.push({ start: cr.start, end: cr.end, total: cr.total, data: body.subarray(ds, ds + len) });
    pos = hvIndexOf(body, delim, ds + len);
    if (pos < 0) break;   // tolerate a missing close delimiter
  }
  return parts;
}

function hvRangeInit(init, spec, signal) {
  const headers = Object.assign({}, init.headers || {}, { Range: 'bytes=' + spec });
  const out = Object.assign({}, init, { headers });
  if (signal) out.signal = signal;
  return out;
}

function hvDiscard(resp, ctrl) {
  try { ctrl?.abort(); } catch {}
  try { resp.body?.cancel?.().catch?.(() => {}); } catch {}
}

/**
 * Fetch one byte range: {buf, total} (total = file size, -1 if unknown).
 * A server that ignores Range (200) is tolerated for small files: the slice is
 * cut out of the whole body; onNoRange() is called once per such response.
 */
export async function hvFetchRange(fetchFn, url, init, off, len, onNoRange) {
  const resp = await fetchFn(url, hvRangeInit(init, `${off}-${off + len - 1}`));
  let buf = new Uint8Array(await resp.arrayBuffer());
  let total = -1;
  if (resp.status === 206) {
    const cr = hvParseContentRange(resp.headers.get('content-range'));
    if (cr) {
      total = cr.total;
      if (cr.start !== off) throw new Error(`httpvfs: asked for bytes ${off}+${len}, got ${cr.start}-${cr.end}`);
    }
  } else if (resp.status === 200) {
    total = buf.length;
    if (onNoRange) onNoRange();
    buf = buf.subarray(off, off + len);
  } else {
    throw new Error(`httpvfs: HTTP ${resp.status} for ${url} bytes ${off}+${len}`);
  }
  return { buf, total };
}

/**
 * Fetch several byte ranges with one multi-range request. Returns
 * {results: [{buf, total}] (one per range), fallback} where fallback is true
 * when the server did not answer with every range (it sent 200, one range, or
 * only some ranges): the missing ranges are then fetched one by one, and the
 * caller should stop sending multi-range requests to this server. A 200
 * response is aborted after its headers, so the whole file is not downloaded.
 */
export async function hvFetchMultiRange(fetchFn, url, init, offs, lens, onNoRange) {
  const n = offs.length;
  const results = new Array(n).fill(null);
  let fallback = false;
  if (n === 1) {
    results[0] = await hvFetchRange(fetchFn, url, init, offs[0], lens[0], onNoRange);
    return { results, fallback };
  }
  const spec = offs.map((o, i) => `${o}-${o + lens[i] - 1}`).join(',');
  const ctrl = typeof AbortController === 'function' ? new AbortController() : null;
  const resp = await fetchFn(url, hvRangeInit(init, spec, ctrl?.signal));
  let pieces = [];
  if (resp.status === 206) {
    const body = new Uint8Array(await resp.arrayBuffer());
    const boundary = hvMultipartBoundary(resp.headers.get('content-type'));
    if (boundary) {
      pieces = hvParseMultipart(body, boundary);
    } else {
      const cr = hvParseContentRange(resp.headers.get('content-range'));
      if (!cr || cr.end - cr.start + 1 !== body.length) throw new Error(`httpvfs: bad 206 response to a multi-range request for ${url}`);
      pieces = [{ start: cr.start, end: cr.end, total: cr.total, data: body }];
    }
  } else if (resp.status === 200) {
    hvDiscard(resp, ctrl);   // the whole file: do not download it
    fallback = true;
  } else {
    hvDiscard(resp, ctrl);
    // Some servers refuse multi-range requests (e.g. 416 or 400) rather than
    // ignoring them; fall back to single ranges, which will report real errors.
    fallback = true;
  }
  // Every requested range from the parts that cover it (servers may merge or
  // reorder ranges); a range that ends past the end of the file is short.
  for (let i = 0; i < n; i++) {
    const off = offs[i], len = lens[i];
    for (const p of pieces) {
      const total = p.total;
      const need = total >= 0 ? Math.min(len, total - off) : len;
      if (p.start <= off && p.end >= off + need - 1) {
        results[i] = { buf: p.data.subarray(off - p.start, off - p.start + need), total };
        break;
      }
    }
  }
  const missing = [];
  for (let i = 0; i < n; i++) if (!results[i]) missing.push(i);
  if (missing.length) {
    fallback = true;
    const got = await Promise.all(missing.map((i) => hvFetchRange(fetchFn, url, init, offs[i], lens[i], onNoRange)));
    missing.forEach((i, k) => { results[i] = got[k]; });
  }
  return { results, fallback };
}

/** Consecutive equal entries of grp (or every index alone): [[i, ...], ...]. */
export function hvGroups(n, grp) {
  const out = [];
  for (let i = 0; i < n; i++) {
    if (grp && i > 0 && grp[i] === grp[i - 1]) out[out.length - 1].push(i);
    else out.push([i]);
  }
  return out;
}
