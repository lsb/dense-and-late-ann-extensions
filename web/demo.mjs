// Demo page logic. Everything search-related goes through web/lib/index.mjs.
import { openIndex } from './lib/index.mjs';

const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
const state = { ix: null, busy: false };
window.demo = state;   // for tests and the console

const DB_CANDIDATES = ['../build/web/words-10k.db', '../build/matrix/words-10k.db', '../build/matrix/words-100.db'];
const EXAMPLES = ['pinwheel', 'gossiping', 'player', 'moonlight gemstone', 'wineries Android formatted',
                  'a paragraph about seaports', 'reverse a string in python'];

const fmtMs = (x) => (x == null ? '–' : x < 10 ? x.toFixed(1) : Math.round(x).toLocaleString());
const fmtKB = (b) => (b == null ? '–' : (b / 1024 < 10 ? (b / 1024).toFixed(1) : Math.round(b / 1024).toLocaleString()));
const fmtMB = (b) => (b / 1048576).toFixed(1) + ' MB';
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

// ------------------------------------------------------------------ netsim
async function initNetsim() {
  const sel = $('netProfile');
  try {
    const [pr, cur] = await Promise.all([
      fetch('/__netsim/presets', { cache: 'no-store' }).then((r) => (r.ok ? r.json() : Promise.reject())),
      fetch('/__netsim/profile', { cache: 'no-store' }).then((r) => r.json()),
    ]);
    const names = Object.keys(pr.presets);
    const opts = [];
    for (const n of names) opts.push(n, `${n},h1`);
    const current = cur.profile.name;
    if (!opts.includes(current)) opts.unshift(current);
    sel.innerHTML = opts.map((n) => {
      const p = pr.presets[n.split(',')[0]] || {};
      const d = p.latency_ms != null ? ` (${p.latency_ms} ms, ${p.link_kbps ? (p.link_kbps / 1000).toFixed(1) + ' Mbit/s' : '∞'})` : '';
      return `<option value="${esc(n)}"${n === current ? ' selected' : ''}>${esc(n)}${esc(d)}</option>`;
    }).join('');
    sel.disabled = false;
    $('netStatus').textContent = cur.describe + ' — applies to every file this server sends, models included';
    sel.onchange = async () => {
      const r = await fetch('/__netsim/profile', { method: 'POST', body: JSON.stringify({ preset: sel.value }) });
      const j = await r.json();
      $('netStatus').textContent = r.ok ? j.describe : `error: ${JSON.stringify(j)}`;
    };
  } catch {
    sel.disabled = true;
    $('netStatus').textContent = 'serve with netsim/rangeserver.py to simulate a network';
  }
}

// ------------------------------------------------------------------ database
async function pickDefaultDb() {
  if (params.get('db')) return params.get('db');
  for (const c of DB_CANDIDATES) {
    try {
      const r = await fetch(c, { headers: { Range: 'bytes=0-15' }, cache: 'no-store' });
      if (r.ok) return c;
    } catch { /* next */ }
  }
  return DB_CANDIDATES[0];
}

async function openDb() {
  $('openBtn').disabled = true;
  $('searchBtn').disabled = true;
  $('dbStatus').textContent = 'opening…';
  try {
    if (state.ix) await state.ix.close();
    state.ix = null;
    resetEncoderUi();
    const t0 = performance.now();
    const ix = await openIndex($('dbUrl').value, {
      variant: $('variant').value,
      modelCache: $('modelCache').checked,
    });
    state.ix = ix;
    const s = ix.openStats;
    $('dbStatus').textContent = `${ix.variant} build; ${fmtMB(s.fileSize)} file; indexes: ${ix.indexes.map((i) => i.table).join(', ')}; ` +
      `open ${fmtMs(performance.now() - t0)} ms, ${s.rounds} rounds, ${fmtKB(s.bytes)} KB`;
    const dense = ix.indexes.filter((i) => i.kind === 'dense');
    $('denseTable').innerHTML = dense.map((i) => `<option value="${esc(i.table)}">${esc(i.table)} (${i.layout})</option>`).join('');
    updateKnobs();
    $('searchBtn').disabled = false;
  } catch (e) {
    $('dbStatus').innerHTML = `<span class="err">${esc(e.message)}</span>`;
  } finally {
    $('openBtn').disabled = false;
  }
}

// ------------------------------------------------------------------ encoders
function resetEncoderUi() {
  for (const w of ['minilm', 'lateon']) {
    $(`enc-${w}`).querySelector('.state').textContent = 'not loaded';
    $(`enc-${w}`).querySelector('.bar i').style.width = '0';
  }
}

function loadEncoder(which) {
  const el = $(`enc-${which}`);
  const st = el.querySelector('.state');
  const bar = el.querySelector('.bar i');
  if (st.dataset.done) return state.ix.loadEncoder(which);
  st.textContent = 'loading…';
  const p = state.ix.loadEncoder(which, (ev) => {
    if (ev.kind === 'progress' && ev.total) {
      bar.style.width = `${(100 * ev.loaded / ev.total).toFixed(1)}%`;
      st.textContent = `downloading ${fmtMB(ev.loaded)} / ${fmtMB(ev.total)}`;
    }
  });
  p.then((s) => {
    bar.style.width = '100%';
    st.dataset.done = '1';
    st.textContent = `ready: ${fmtMB(s.modelBytes)} model ${s.modelFromCache ? 'from cache' : `in ${fmtMs(s.modelMs)} ms`}, ` +
      `session ${fmtMs(s.sessionMs)} ms, first run ${fmtMs(s.warmupMs)} ms (total ${fmtMs(s.totalMs)} ms)`;
  }, (e) => { st.innerHTML = `<span class="err">${esc(e.message)}</span>`; });
  return p;
}

// ------------------------------------------------------------------ knobs
function updateKnobs() {
  const ix = state.ix;
  const kinds = new Set(ix ? ix.indexes.map((i) => i.kind) : []);
  for (const fs of document.querySelectorAll('fieldset[data-kind]')) fs.hidden = ix && !kinds.has(fs.dataset.kind);
  const dt = $('denseTable').value;
  const layout = ix?.indexes.find((i) => i.table === dt)?.layout || 'graph';
  for (const s of document.querySelectorAll('fieldset[data-kind="dense"] [data-layout]')) s.hidden = s.dataset.layout !== layout;
  const late = document.querySelector('fieldset[data-kind="late"]');
  const lay = late.querySelector('[name="layout"]').value;
  late.querySelector('[name="rerank"]').closest('label').hidden = lay !== 'warp';
  late.querySelector('[name="approx"]').closest('label').hidden = lay !== 'plaid';
  late.querySelector('[name="ndocs"]').closest('label').hidden = lay !== 'plaid';
}

function knobValues(kind) {
  const fs = document.querySelector(`fieldset[data-kind="${kind}"]`);
  const out = {};
  for (const el of fs.querySelectorAll('[name]')) {
    if (el.closest('[hidden]')) continue;
    out[el.name] = el.value;
  }
  return out;
}

// ------------------------------------------------------------------ search
function highlight(text, query) {
  const words = new Set((query.toLowerCase().match(/[\p{L}\p{N}]+/gu) || []));
  return esc(text).replace(/[\p{L}\p{N}]+/gu, (w) => (words.has(w.toLowerCase()) ? `<mark>${w}</mark>` : w));
}

function column(title, kind) {
  const div = document.createElement('div');
  div.className = 'col';
  div.dataset.kind = kind;
  div.innerHTML = `<h2>${esc(title)}</h2><div class="metrics">searching…</div><div class="meta"></div><ol class="hits"></ol>`;
  $('results').appendChild(div);
  return div;
}

function render(div, r, query) {
  const s = r.stats;
  const ext = r.ext || {};
  div.querySelector('.metrics').innerHTML = [
    ['wall', `${fmtMs(s.wallMs)} ms`], ['enc', `${fmtMs(s.encodeMs)} ms`],
    ['sql', `${fmtMs(s.searchMs)} ms`], ['docs', `${fmtMs(s.docsMs)} ms`],
    ['rounds', `${s.phases.search.rounds} + ${s.phases.docs.rounds}`], ['req', s.requests],
    ['KB', fmtKB(s.bytes)], ['net', `${fmtMs(s.netMs)} ms`],
  ].map(([k, v]) => `<span>${k} <b>${v}</b></span>`).join('');
  const bits = [];
  if (r.kind === 'fts') bits.push(`MATCH ${esc(r.match || '(empty)')}`);
  else {
    bits.push(`${r.query.tokens} tokens${r.kind === 'late' ? ` → ${r.query.vectors} vectors` : ''}; tokenize ${fmtMs(s.tokenizeMs)} ms, model ${fmtMs(s.inferMs)} ms`);
    if (s.encoderLoadWaitMs > 50) bits.push(`waited ${fmtMs(s.encoderLoadWaitMs)} ms for the encoder to load`);
    bits.push(Object.entries(r.params).map(([k, v]) => `${k}=${v}`).join(' '));
  }
  const extKeys = ['rounds', 'pages', 'expanded', 'candidates', 'docs_fetched', 'ms'];
  const extBits = extKeys.filter((k) => ext[k] != null).map((k) => `${k} ${typeof ext[k] === 'number' ? +ext[k].toFixed(2) : ext[k]}`);
  if (extBits.length) bits.push(`extension: ${extBits.join(', ')}`);
  div.querySelector('.meta').innerHTML = bits.join('<br>');
  div.querySelector('.hits').innerHTML = r.rows.length ? r.rows.map((h) =>
    `<li><span class="hit-id">#${h.id} · ${typeof h.score === 'number' ? h.score.toFixed(3) : h.score}</span> ` +
    `<span class="hit-text">${h.text == null ? '' : highlight(h.text.length > 400 ? h.text.slice(0, 400) + '…' : h.text, query)}</span></li>`).join('')
    : '<li>no results</li>';
}

async function runSearch(query, system) {
  const ix = state.ix;
  $('results').innerHTML = '';
  const k = Number($('k').value) || 10;
  const cold = $('cold').checked;
  const jobs = [];
  const want = system === 'all' ? ['fts', 'dense', 'late'] : [system];
  const out = {};
  for (const sys of want) {
    const kinds = new Set(ix.indexes.map((i) => i.kind));
    if (!kinds.has(sys)) continue;
    const opts = { system: sys, k, cold, ...knobValues(sys) };
    if (sys === 'dense') { opts.system = opts.table; delete opts.table; }
    if (sys !== 'fts') loadEncoder(sys === 'dense' ? 'minilm' : 'lateon');
    const title = ix.systems.find((s) => s.id === ix.resolve(opts.system).table).label;
    const div = column(title, sys);
    jobs.push(ix.search(query, opts).then((r) => { out[sys] = r; render(div, r, query); },
      (e) => { out[sys] = { error: e.message }; div.querySelector('.metrics').innerHTML = `<span class="err">${esc(e.message)}</span>`; }));
  }
  await Promise.all(jobs);
  return out;
}

// ------------------------------------------------------------------ wiring
async function main() {
  initNetsim();
  if (params.get('variant')) $('variant').value = params.get('variant');
  $('dbUrl').value = await pickDefaultDb();
  $('openBtn').onclick = openDb;
  $('loadEnc').onclick = () => { if (state.ix) { loadEncoder('minilm'); loadEncoder('lateon'); } };
  $('denseTable').onchange = updateKnobs;
  document.querySelector('fieldset[data-kind="late"] [name="layout"]').onchange = updateKnobs;
  $('examples').innerHTML = 'Try: ' + EXAMPLES.map((e) => `<button type="button">${esc(e)}</button>`).join(' ');
  $('examples').onclick = (e) => { if (e.target.tagName === 'BUTTON') { $('q').value = e.target.textContent; $('searchForm').requestSubmit(); } };
  $('searchForm').onsubmit = async (e) => {
    e.preventDefault();
    if (!state.ix || state.busy) return;
    const q = $('q').value.trim();
    if (!q) return;
    state.busy = true;
    $('searchBtn').disabled = true;
    try { state.last = await runSearch(q, $('system').value); } finally { state.busy = false; $('searchBtn').disabled = false; }
  };
  state.openDb = openDb;
  state.runSearch = runSearch;
  state.ready = openDb();
  await state.ready;
  if (params.get('q')) {
    $('q').value = params.get('q');
    if (params.get('system')) $('system').value = params.get('system');
    $('searchForm').requestSubmit();
  }
}
main();
