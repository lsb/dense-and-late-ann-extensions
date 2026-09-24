/*
** late_plaid.c -- a SQLite virtual table for PLAID-style late-interaction
** (ColBERT / MaxSim) retrieval, designed for read-only databases fetched
** lazily over HTTP range requests ("httpvfs").
**
**   CREATE VIRTUAL TABLE t USING late_plaid(dim=48, nbits=2, centroids=0,
**          layout=both, kmeans_iters=4, kmeans_ppc=256, threads=4);
**   INSERT INTO t(rowid, vectors) VALUES (?, ?);   -- float32 [n_tok x dim] blob
**   INSERT INTO t(t) VALUES ('build');               -- k-means, codec, index
**   INSERT INTO t(t) VALUES ('build_npy VEC.npy OFFSETS.npy');  -- stream from files
**   INSERT INTO t(t) VALUES ('finalize');            -- page hints (after VACUUM)
**   SELECT rowid, score, stats FROM t WHERE t MATCH ?1 AND k = 10
**          [AND nprobe = 4] [AND opts = 'layout=warp impute=1 ...'];
**
** Two storage layouts share the centroids and the residual codec:
**
**   plaid  (doc-major, classic PLAID) -- an IVF (centroid -> bit-packed ids of
**          the documents having a token in that cell) and one row per
**          document with its bit-packed centroid ids and packed residuals.
**          Query: centroid scores (cached table) -> IVF lists (1 round) ->
**          candidate document rows (1 round) -> exact MaxSim.
**   warp   (centroid-major, WARP-like) -- per centroid, the ids and packed
**          residuals of all tokens assigned to it. Query: centroid scores ->
**          posting lists (1 round) -> per query token max over a document's
**          tokens found, missing (document, query token) pairs imputed.
**
** IVF and posting lists are stored as "paged streams": all lists are
** concatenated and cut into rows that each fill exactly one database page,
** so reading any byte range costs whole, fully used pages and never an
** overflow chain. Their leaf page numbers are stored run-length encoded
** ('finalize'), so a query reads them with no b-tree walk at all.
**
** Shadow tables (all "id INTEGER PRIMARY KEY, data BLOB"):
**   <t>_meta    static data: parameters, centroids (float16), codec,
**               list lengths, page hints; loaded once per connection
**   <t>_docs    plaid layout: one row per document (id = document number)
**   <t>_ivf     plaid layout: paged stream of IVF lists
**   <t>_post    warp layout:  paged stream of posting lists
**   <t>_rowids  document number -> user rowid (only if not the identity)
**   <t>_buffer  float16 token vectors inserted before 'build'
**
** See NOTES.md for the design discussion and measurements.
*/
#ifndef _FILE_OFFSET_BITS
#define _FILE_OFFSET_BITS 64
#endif
#include "sqlite3ext.h"
SQLITE_EXTENSION_INIT1

#include <float.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "late_codec.h"
#include "late_common.h"
#include "late_page.h"

#define LT_MAGIC 0x3150544cu   /* "LTP1" */
#define LT_VERSION 2

enum { LAY_PLAID = 1, LAY_WARP = 2 };
enum { COL_VECTORS = 0, COL_SCORE, COL_K, COL_NPROBE, COL_OPTS, COL_STATS, COL_CMD, NCOL };
#define F_MATCH  0x01
#define F_K      0x02
#define F_NPROBE 0x04
#define F_OPTS   0x08
#define F_LIMIT  0x10
#ifndef SQLITE_INDEX_CONSTRAINT_LIMIT
#define SQLITE_INDEX_CONSTRAINT_LIMIT 73
#endif

static double now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

/* ------------------------------------------------------------ configuration */

typedef struct LtCfg {
  int dim, nbits, K, layout, iters, ppc, threads, input_f16, chunk, verbose;
  int coarse, aprobe;          /* two-level centroids: G cells, cells probed per token at build */
  int64_t sample_docs, mem_mb, heldout_max;
  uint64_t seed;
} LtCfg;

static void cfg_default(LtCfg *c) {
  memset(c, 0, sizeof *c);
  c->dim = 48; c->nbits = 2; c->K = 0; c->layout = LAY_PLAID | LAY_WARP;
  c->iters = 4; c->ppc = 256; c->threads = 4; c->seed = 42;
  c->mem_mb = 1024; c->heldout_max = 50000; c->aprobe = 8;
}

static int cfg_parse(LtCfg *c, int argc, const char *const *argv, char **pzErr) {
  cfg_default(c);
  for (int i = 3; i < argc; i++) {
    char key[64], val[256];
    const char *a = argv[i];
    while (*a == ' ') a++;
    const char *eq = strchr(a, '=');
    if (!eq) { *pzErr = sqlite3_mprintf("late_plaid: expected key=value, got '%s'", argv[i]); return SQLITE_ERROR; }
    int kl = (int)(eq - a); while (kl > 0 && a[kl - 1] == ' ') kl--;
    if (kl >= (int)sizeof key) kl = sizeof key - 1;
    memcpy(key, a, kl); key[kl] = 0;
    const char *v = eq + 1; while (*v == ' ' || *v == '\'' || *v == '"') v++;
    snprintf(val, sizeof val, "%s", v);
    int vl = (int)strlen(val);
    while (vl > 0 && (val[vl - 1] == ' ' || val[vl - 1] == '\'' || val[vl - 1] == '"')) val[--vl] = 0;
    long long iv = atoll(val);
    if (!strcmp(key, "dim")) c->dim = (int)iv;
    else if (!strcmp(key, "nbits")) c->nbits = (int)iv;
    else if (!strcmp(key, "centroids") || !strcmp(key, "k")) c->K = (int)iv;
    else if (!strcmp(key, "layout")) {
      if (!strcmp(val, "plaid")) c->layout = LAY_PLAID;
      else if (!strcmp(val, "warp")) c->layout = LAY_WARP;
      else if (!strcmp(val, "both")) c->layout = LAY_PLAID | LAY_WARP;
      else { *pzErr = sqlite3_mprintf("late_plaid: layout must be plaid, warp or both"); return SQLITE_ERROR; }
    }
    else if (!strcmp(key, "kmeans_iters")) c->iters = (int)iv;
    else if (!strcmp(key, "kmeans_ppc")) c->ppc = (int)iv;
    else if (!strcmp(key, "sample_docs")) c->sample_docs = iv;
    else if (!strcmp(key, "seed")) c->seed = (uint64_t)iv;
    else if (!strcmp(key, "threads")) c->threads = (int)iv;
    else if (!strcmp(key, "mem_mb")) c->mem_mb = iv;
    else if (!strcmp(key, "heldout")) c->heldout_max = iv;
    else if (!strcmp(key, "chunk")) c->chunk = (int)iv;
    else if (!strcmp(key, "verbose")) c->verbose = (int)iv;
    else if (!strcmp(key, "coarse")) c->coarse = (int)iv;
    else if (!strcmp(key, "assign_probe")) c->aprobe = (int)iv;
    else if (!strcmp(key, "input")) c->input_f16 = !strcmp(val, "f16");
    else { *pzErr = sqlite3_mprintf("late_plaid: unknown option '%s'", key); return SQLITE_ERROR; }
  }
  if (c->nbits != 1 && c->nbits != 2 && c->nbits != 4) { *pzErr = sqlite3_mprintf("late_plaid: nbits must be 1, 2 or 4"); return SQLITE_ERROR; }
  if (c->dim < 8 || c->dim % 8) { *pzErr = sqlite3_mprintf("late_plaid: dim must be a multiple of 8"); return SQLITE_ERROR; }
  return SQLITE_OK;
}

/* ---------------------------------------------------------------- index */

typedef struct LtIndex {
  int loaded;
  LtCodec codec;
  int K, dim, nbits, layout, kbits, idbits, chunk, rowid_identity;
  int64_t N, T, ivf_entries;
  uint32_t *ivf_cnt, *post_cnt;       /* K each (NULL if the layout is absent) */
  uint64_t *ivf_off, *post_off;       /* K+1 byte offsets in the streams */
  uint32_t *ivf_runs, *post_runs;     /* (first row, first page) pairs */
  int n_ivf_runs, n_post_runs;
  uint32_t root_meta, root_docs, root_ivf, root_post, root_rowids, root_cells;
  int64_t static_bytes;
  int load_rounds;
  /* Two-level centroids (G > 0): only the G coarse centroids are static;
  ** each cell's fine centroids and list lengths form a block of the "cells"
  ** stream, fetched when a query first probes the cell. Fine centroid ids
  ** are cell-major: cell g owns [cell_start[g], cell_start[g+1]). */
  int G;
  float *coarse;                      /* G x dim */
  uint32_t *cell_start;               /* G+1 */
  uint64_t *cell_ivf_base, *cell_post_base, *cell_off;   /* G, G, G+1 */
  float **cell_cent;                  /* G pointers (NULL until loaded) */
  uint32_t *cell_runs; int n_cell_runs;
  int64_t cell_bytes_loaded; int cells_loaded;
} LtIndex;

/* Centroid vector of fine centroid c (its cell must be loaded). */
static const float *cent(const LtIndex *ix, uint32_t c) {
  if (!ix->G) return ix->codec.centroids + (size_t)c * ix->dim;
  int lo = 0, hi = ix->G - 1;
  while (lo < hi) { int mid = (lo + hi + 1) / 2; if (ix->cell_start[mid] <= c) lo = mid; else hi = mid - 1; }
  while (lo + 1 < ix->G && ix->cell_start[lo + 1] <= c) lo++;       /* skip empty cells */
  return ix->cell_cent[lo] ? ix->cell_cent[lo] + (size_t)(c - ix->cell_start[lo]) * ix->dim : NULL;
}

typedef struct LtVtab {
  sqlite3_vtab base;
  sqlite3 *db;
  char *zDb, *zName;
  LtCfg cfg;
  LtIndex ix;
  LpReader rd;
  int rd_state;          /* 0 not opened, 1 ok, -1 unusable */
} LtVtab;

static void ix_free(LtIndex *ix) {
  free(ix->codec.centroids);
  free(ix->ivf_cnt); free(ix->post_cnt); free(ix->ivf_off); free(ix->post_off);
  free(ix->ivf_runs); free(ix->post_runs);
  free(ix->coarse); free(ix->cell_start); free(ix->cell_ivf_base); free(ix->cell_post_base); free(ix->cell_off);
  if (ix->cell_cent) for (int g = 0; g < ix->G; g++) free(ix->cell_cent[g]);
  free(ix->cell_cent); free(ix->cell_runs);
  memset(ix, 0, sizeof *ix);
}

static uint64_t seg_bytes(const LtIndex *ix, int warp, uint32_t cnt) {
  return lt_bits_bytes(cnt, ix->idbits) + (warp ? (uint64_t)cnt * ix->codec.rbytes : 0);
}

static void ix_offsets(LtIndex *ix) {
  if (ix->ivf_cnt) {
    ix->ivf_off[0] = 0;
    for (int c = 0; c < ix->K; c++) ix->ivf_off[c + 1] = ix->ivf_off[c] + seg_bytes(ix, 0, ix->ivf_cnt[c]);
  }
  if (ix->post_cnt) {
    ix->post_off[0] = 0;
    for (int c = 0; c < ix->K; c++) ix->post_off[c + 1] = ix->post_off[c] + seg_bytes(ix, 1, ix->post_cnt[c]);
  }
}

/* Size of cell g's block in the cells stream: fine centroids (float16) and
** the list lengths of each layout. */
static uint64_t cell_block_bytes(const LtIndex *ix, uint32_t kg) {
  return (uint64_t)kg * (2u * ix->dim + ((ix->layout & LAY_PLAID) ? 4 : 0) + ((ix->layout & LAY_WARP) ? 4 : 0));
}

/* Serialise the static data (see LtIndex). Version 2 appends G and the
** cells-stream run count to the version 1 header. */
static void ix_serialize(const LtIndex *ix, LtBuf *b) {
  const LtCodec *cd = &ix->codec;
  lt_buf_u32(b, LT_MAGIC); lt_buf_u32(b, LT_VERSION);
  lt_buf_u32(b, ix->dim); lt_buf_u32(b, ix->nbits); lt_buf_u32(b, ix->K); lt_buf_u32(b, ix->layout);
  lt_buf_u32(b, ix->kbits); lt_buf_u32(b, ix->idbits); lt_buf_u32(b, ix->chunk); lt_buf_u32(b, ix->rowid_identity);
  lt_buf_u64(b, ix->N); lt_buf_u64(b, ix->T); lt_buf_u64(b, ix->ivf_entries);
  lt_buf_add(b, cd->cutoffs, 64); lt_buf_add(b, cd->weights, 64); lt_buf_add(b, &cd->cluster_threshold, 4);
  lt_buf_u32(b, ix->n_ivf_runs); lt_buf_u32(b, ix->n_post_runs);
  lt_buf_u32(b, ix->G); lt_buf_u32(b, ix->n_cell_runs);
  if (ix->G == 0) {
    for (size_t i = 0; i < (size_t)ix->K * ix->dim; i++) { uint16_t h = lt_f2h(cd->centroids[i]); lt_buf_add(b, &h, 2); }
    if (ix->layout & LAY_PLAID) for (int c = 0; c < ix->K; c++) lt_buf_u32(b, ix->ivf_cnt[c]);
    if (ix->layout & LAY_WARP) for (int c = 0; c < ix->K; c++) lt_buf_u32(b, ix->post_cnt[c]);
  } else {
    for (size_t i = 0; i < (size_t)ix->G * ix->dim; i++) { uint16_t h = lt_f2h(ix->coarse[i]); lt_buf_add(b, &h, 2); }
    for (int g = 0; g <= ix->G; g++) lt_buf_u32(b, ix->cell_start[g]);
    if (ix->layout & LAY_PLAID) for (int g = 0; g < ix->G; g++) lt_buf_u64(b, ix->cell_ivf_base[g]);
    if (ix->layout & LAY_WARP) for (int g = 0; g < ix->G; g++) lt_buf_u64(b, ix->cell_post_base[g]);
  }
  for (int i = 0; i < 2 * ix->n_ivf_runs; i++) lt_buf_u32(b, ix->ivf_runs[i]);
  for (int i = 0; i < 2 * ix->n_post_runs; i++) lt_buf_u32(b, ix->post_runs[i]);
  for (int i = 0; i < 2 * ix->n_cell_runs; i++) lt_buf_u32(b, ix->cell_runs[i]);
}

static int read_runs(const uint8_t **q, const uint8_t *end, uint32_t **runs, int n) {
  if (!n) return SQLITE_OK;
  if ((size_t)(end - *q) < 8u * n) return SQLITE_CORRUPT;
  *runs = (uint32_t *)malloc(8 * (size_t)n);
  if (!*runs) return SQLITE_NOMEM;
  for (int i = 0; i < 2 * n; i++, *q += 4) (*runs)[i] = lt_get_u32(*q);
  return SQLITE_OK;
}

static int ix_deserialize(LtIndex *ix, const uint8_t *p, size_t n) {
  if (n < 160 || lt_get_u32(p) != LT_MAGIC) return SQLITE_CORRUPT;
  int ver = (int)lt_get_u32(p + 4);
  if (ver != 1 && ver != 2) return SQLITE_CORRUPT;
  const uint8_t *q = p + 8, *end = p + n;
  ix->dim = lt_get_u32(q); ix->nbits = lt_get_u32(q + 4); ix->K = lt_get_u32(q + 8); ix->layout = lt_get_u32(q + 12);
  ix->kbits = lt_get_u32(q + 16); ix->idbits = lt_get_u32(q + 20); ix->chunk = lt_get_u32(q + 24);
  ix->rowid_identity = lt_get_u32(q + 28); q += 32;
  ix->N = (int64_t)lt_get_u64(q); ix->T = (int64_t)lt_get_u64(q + 8); ix->ivf_entries = (int64_t)lt_get_u64(q + 16); q += 24;
  LtCodec *cd = &ix->codec;
  memcpy(cd->cutoffs, q, 64); memcpy(cd->weights, q + 64, 64); memcpy(&cd->cluster_threshold, q + 128, 4); q += 132;
  ix->n_ivf_runs = lt_get_u32(q); ix->n_post_runs = lt_get_u32(q + 4); q += 8;
  if (ver >= 2) { ix->G = lt_get_u32(q); ix->n_cell_runs = lt_get_u32(q + 4); q += 8; }
  cd->dim = ix->dim; cd->nbits = ix->nbits; cd->K = ix->K; cd->rbytes = ix->dim * ix->nbits / 8;
  lc_init_lut(cd);
  int K = ix->K, G = ix->G;
  if (ix->layout & LAY_PLAID) {
    ix->ivf_cnt = (uint32_t *)calloc(K, 4); ix->ivf_off = (uint64_t *)calloc(K + 1, 8);
    if (!ix->ivf_cnt || !ix->ivf_off) return SQLITE_NOMEM;
  }
  if (ix->layout & LAY_WARP) {
    ix->post_cnt = (uint32_t *)calloc(K, 4); ix->post_off = (uint64_t *)calloc(K + 1, 8);
    if (!ix->post_cnt || !ix->post_off) return SQLITE_NOMEM;
  }
  if (G == 0) {
    size_t need = (size_t)K * ix->dim * 2 + ((ix->layout & LAY_PLAID) ? 4u * K : 0) + ((ix->layout & LAY_WARP) ? 4u * K : 0);
    if ((size_t)(end - q) < need) return SQLITE_CORRUPT;
    cd->centroids = (float *)malloc(sizeof(float) * K * ix->dim);
    if (!cd->centroids) return SQLITE_NOMEM;
    for (size_t i = 0; i < (size_t)K * ix->dim; i++) cd->centroids[i] = lt_h2f((uint16_t)(q[2 * i] | (q[2 * i + 1] << 8)));
    q += (size_t)K * ix->dim * 2;
    if (ix->layout & LAY_PLAID) for (int c = 0; c < K; c++, q += 4) ix->ivf_cnt[c] = lt_get_u32(q);
    if (ix->layout & LAY_WARP) for (int c = 0; c < K; c++, q += 4) ix->post_cnt[c] = lt_get_u32(q);
    ix_offsets(ix);
  } else {
    size_t need = (size_t)G * ix->dim * 2 + 4u * (G + 1) + ((ix->layout & LAY_PLAID) ? 8u * G : 0) + ((ix->layout & LAY_WARP) ? 8u * G : 0);
    if ((size_t)(end - q) < need) return SQLITE_CORRUPT;
    ix->coarse = (float *)malloc(sizeof(float) * G * ix->dim);
    ix->cell_start = (uint32_t *)malloc(4 * (size_t)(G + 1));
    ix->cell_ivf_base = (uint64_t *)calloc(G, 8); ix->cell_post_base = (uint64_t *)calloc(G, 8);
    ix->cell_off = (uint64_t *)malloc(8 * (size_t)(G + 1));
    ix->cell_cent = (float **)calloc(G, sizeof(float *));
    if (!ix->coarse || !ix->cell_start || !ix->cell_ivf_base || !ix->cell_post_base || !ix->cell_off || !ix->cell_cent) return SQLITE_NOMEM;
    for (size_t i = 0; i < (size_t)G * ix->dim; i++) ix->coarse[i] = lt_h2f((uint16_t)(q[2 * i] | (q[2 * i + 1] << 8)));
    q += (size_t)G * ix->dim * 2;
    for (int g = 0; g <= G; g++, q += 4) ix->cell_start[g] = lt_get_u32(q);
    if (ix->layout & LAY_PLAID) for (int g = 0; g < G; g++, q += 8) ix->cell_ivf_base[g] = lt_get_u64(q);
    if (ix->layout & LAY_WARP) for (int g = 0; g < G; g++, q += 8) ix->cell_post_base[g] = lt_get_u64(q);
    ix->cell_off[0] = 0;
    for (int g = 0; g < G; g++) ix->cell_off[g + 1] = ix->cell_off[g] + cell_block_bytes(ix, ix->cell_start[g + 1] - ix->cell_start[g]);
  }
  int rc = read_runs(&q, end, &ix->ivf_runs, ix->n_ivf_runs);
  if (rc == SQLITE_OK) rc = read_runs(&q, end, &ix->post_runs, ix->n_post_runs);
  if (rc == SQLITE_OK) rc = read_runs(&q, end, &ix->cell_runs, ix->n_cell_runs);
  ix->static_bytes = (int64_t)n;
  return rc;
}

/* Install cell g's block (fine centroids and list lengths). */
static int cell_install(LtIndex *ix, int g, const uint8_t *blk) {
  uint32_t c0 = ix->cell_start[g], kg = ix->cell_start[g + 1] - c0;
  float *cc = (float *)malloc(sizeof(float) * (size_t)(kg ? kg : 1) * ix->dim);
  if (!cc) return SQLITE_NOMEM;
  for (size_t i = 0; i < (size_t)kg * ix->dim; i++) cc[i] = lt_h2f((uint16_t)(blk[2 * i] | (blk[2 * i + 1] << 8)));
  const uint8_t *q = blk + (size_t)kg * ix->dim * 2;
  if (ix->layout & LAY_PLAID) {
    uint64_t o = ix->cell_ivf_base[g];
    for (uint32_t j = 0; j < kg; j++, q += 4) { ix->ivf_cnt[c0 + j] = lt_get_u32(q); ix->ivf_off[c0 + j] = o; o += seg_bytes(ix, 0, ix->ivf_cnt[c0 + j]); }
  }
  if (ix->layout & LAY_WARP) {
    uint64_t o = ix->cell_post_base[g];
    for (uint32_t j = 0; j < kg; j++, q += 4) { ix->post_cnt[c0 + j] = lt_get_u32(q); ix->post_off[c0 + j] = o; o += seg_bytes(ix, 1, ix->post_cnt[c0 + j]); }
  }
  ix->cell_cent[g] = cc;
  ix->cells_loaded++;
  ix->cell_bytes_loaded += (int64_t)cell_block_bytes(ix, kg);
  return SQLITE_OK;
}

/* Page hint for stream row index r (rowid r+1) from the run list. */
static uint32_t run_hint(const uint32_t *runs, int n, int64_t r) {
  int lo = 0, hi = n - 1, best = -1;
  while (lo <= hi) {
    int mid = (lo + hi) / 2;
    if ((int64_t)runs[2 * mid] <= r) { best = mid; lo = mid + 1; } else hi = mid - 1;
  }
  if (best < 0) return 0;
  return runs[2 * best + 1] + (uint32_t)(r - runs[2 * best]);
}

/* ------------------------------------------------------------- SQL helpers */

static int exec_fmt(sqlite3 *db, char **pzErr, const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  char *sql = sqlite3_vmprintf(fmt, ap);
  va_end(ap);
  if (!sql) return SQLITE_NOMEM;
  int rc = sqlite3_exec(db, sql, NULL, NULL, pzErr);
  sqlite3_free(sql);
  return rc;
}

static int prep_fmt(sqlite3 *db, sqlite3_stmt **st, const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  char *sql = sqlite3_vmprintf(fmt, ap);
  va_end(ap);
  if (!sql) return SQLITE_NOMEM;
  int rc = sqlite3_prepare_v2(db, sql, -1, st, NULL);
  sqlite3_free(sql);
  return rc;
}

static void set_err(LtVtab *vt, const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  sqlite3_free(vt->base.zErrMsg);
  vt->base.zErrMsg = sqlite3_vmprintf(fmt, ap);
  va_end(ap);
}

static void vlog(LtVtab *vt, const char *fmt, ...) {
  if (!vt->cfg.verbose) return;
  va_list ap;
  va_start(ap, fmt);
  fprintf(stderr, "[late_plaid %.1fs] ", now_ms() / 1000.0);
  vfprintf(stderr, fmt, ap);
  fputc('\n', stderr);
  va_end(ap);
}

static int reader(LtVtab *vt) {
  if (vt->rd_state == 0) vt->rd_state = lpg_open(&vt->rd, vt->db, vt->zDb, 0) == 0 ? 1 : -1;
  return vt->rd_state;
}

/* ------------------------------------------------------------ loading */

typedef struct { LtBuf buf; } MetaCtx;
static int meta_cb(void *ctx, int64_t rowid, uint32_t pgno, const uint8_t *d, int len) {
  (void)rowid; (void)pgno;
  lt_buf_add(&((MetaCtx *)ctx)->buf, d, len);
  return ((MetaCtx *)ctx)->buf.oom ? SQLITE_NOMEM : SQLITE_OK;
}

static int ix_load(LtVtab *vt) {
  LtIndex *ix = &vt->ix;
  if (ix->loaded) return SQLITE_OK;
  char tb[300];
  ix_free(ix);
  snprintf(tb, sizeof tb, "%s_meta", vt->zName); ix->root_meta = lpg_root(vt->db, vt->zDb, tb);
  snprintf(tb, sizeof tb, "%s_docs", vt->zName); ix->root_docs = lpg_root(vt->db, vt->zDb, tb);
  snprintf(tb, sizeof tb, "%s_ivf", vt->zName); ix->root_ivf = lpg_root(vt->db, vt->zDb, tb);
  snprintf(tb, sizeof tb, "%s_post", vt->zName); ix->root_post = lpg_root(vt->db, vt->zDb, tb);
  snprintf(tb, sizeof tb, "%s_rowids", vt->zName); ix->root_rowids = lpg_root(vt->db, vt->zDb, tb);
  snprintf(tb, sizeof tb, "%s_cells", vt->zName); ix->root_cells = lpg_root(vt->db, vt->zDb, tb);
  MetaCtx mc; memset(&mc, 0, sizeof mc);
  int rc = SQLITE_ERROR, r0 = vt->rd.round;
  if (reader(vt) > 0 && ix->root_meta) rc = lpg_scan(&vt->rd, ix->root_meta, meta_cb, &mc);
  if (rc != SQLITE_OK) {           /* direct reads impossible: plain SQL */
    lt_buf_free(&mc.buf);
    sqlite3_stmt *st = NULL;
    rc = prep_fmt(vt->db, &st, "SELECT data FROM \"%w\".\"%w_meta\" ORDER BY id", vt->zDb, vt->zName);
    while (rc == SQLITE_OK && sqlite3_step(st) == SQLITE_ROW)
      lt_buf_add(&mc.buf, sqlite3_column_blob(st, 0), sqlite3_column_bytes(st, 0));
    sqlite3_finalize(st);
  }
  if (rc == SQLITE_OK && mc.buf.n == 0) { set_err(vt, "late_plaid: index not built"); rc = SQLITE_ERROR; }
  else if (rc == SQLITE_OK) rc = ix_deserialize(ix, mc.buf.p, mc.buf.n);
  lt_buf_free(&mc.buf);
  if (rc == SQLITE_OK) { ix->loaded = 1; ix->load_rounds = vt->rd.round - r0; }
  else if (!vt->base.zErrMsg) set_err(vt, "late_plaid: cannot load index (%d)", rc);
  return rc;
}

/* ================================================================ BUILD */

/* Where the document vectors come from. Docs are numbered 0..N-1. */
typedef struct DocSrc {
  int64_t N, T;
  int64_t *off;            /* N+1 token offsets */
  int64_t *rowids;         /* N user rowids, NULL = identity */
  int dim;
  /* npy */
  FILE *f; int64_t data_off; int esize;
  /* buffer table */
  sqlite3_stmt *st;
} DocSrc;

static void src_close(DocSrc *s) {
  if (s->f) fclose(s->f);
  sqlite3_finalize(s->st);
  free(s->off); free(s->rowids);
  memset(s, 0, sizeof *s);
}

/* Read the tokens of docs [d0, d1) as float32 into out. */
static int src_read(DocSrc *s, int64_t d0, int64_t d1, float *out) {
  int64_t n = (s->off[d1] - s->off[d0]) * s->dim;
  if (s->f) {
    if (fseeko(s->f, (off_t)(s->data_off + s->off[d0] * s->dim * s->esize), SEEK_SET)) return SQLITE_IOERR;
    if (s->esize == 4) return fread(out, 4, (size_t)n, s->f) == (size_t)n ? SQLITE_OK : SQLITE_IOERR;
    uint16_t tmp[4096];
    for (int64_t i = 0; i < n;) {
      size_t m = (size_t)(n - i < 4096 ? n - i : 4096);
      if (fread(tmp, 2, m, s->f) != m) return SQLITE_IOERR;
      for (size_t j = 0; j < m; j++) out[i + j] = lt_h2f(tmp[j]);
      i += m;
    }
    return SQLITE_OK;
  }
  /* buffer table: rows in rowid order */
  sqlite3_reset(s->st);
  sqlite3_bind_int64(s->st, 1, s->rowids[d0]);
  sqlite3_bind_int64(s->st, 2, d1 - d0);
  int64_t pos = 0;
  while (sqlite3_step(s->st) == SQLITE_ROW) {
    const uint16_t *v = (const uint16_t *)sqlite3_column_blob(s->st, 0);
    int64_t m = sqlite3_column_bytes(s->st, 0) / 2;
    if (pos + m > n) return SQLITE_CORRUPT;
    for (int64_t j = 0; j < m; j++) out[pos + j] = lt_h2f(v[j]);
    pos += m;
  }
  sqlite3_reset(s->st);
  return pos == n ? SQLITE_OK : SQLITE_CORRUPT;
}

/* Minimal .npy reader: little-endian, C order. */
static int npy_open(const char *path, FILE **pf, char *descr, int64_t *shape, int *ndim, int64_t *data_off) {
  FILE *f = fopen(path, "rb");
  if (!f) return SQLITE_CANTOPEN;
  unsigned char h[12];
  if (fread(h, 1, 10, f) != 10 || memcmp(h, "\x93NUMPY", 6)) { fclose(f); return SQLITE_CORRUPT; }
  uint32_t hl;
  if (h[6] == 1) hl = h[8] | (h[9] << 8);
  else { if (fread(h + 10, 1, 2, f) != 2) { fclose(f); return SQLITE_CORRUPT; } hl = h[8] | (h[9] << 8) | (h[10] << 16) | ((uint32_t)h[11] << 24); }
  char *hdr = (char *)malloc(hl + 1);
  if (!hdr || fread(hdr, 1, hl, f) != hl) { free(hdr); fclose(f); return SQLITE_CORRUPT; }
  hdr[hl] = 0;
  *data_off = ftello(f);
  const char *d = strstr(hdr, "'descr'"), *sh = strstr(hdr, "'shape'");
  if (!d || !sh || strstr(hdr, "'fortran_order': True")) { free(hdr); fclose(f); return SQLITE_CORRUPT; }
  d = strchr(d + 7, '\''); sscanf(d + 1, "%7[^']", descr);
  sh = strchr(sh, '(');
  *ndim = 0;
  for (const char *p = sh + 1; *p && *p != ')' && *ndim < 4;) {
    while (*p == ' ' || *p == ',') p++;
    if (*p == ')' || !*p) break;
    shape[(*ndim)++] = strtoll(p, (char **)&p, 10);
  }
  free(hdr);
  *pf = f;
  return SQLITE_OK;
}

static int src_open_npy(DocSrc *s, const char *vec_path, const char *off_path, int dim, char **perr) {
  memset(s, 0, sizeof *s);
  s->dim = dim;
  char descr[8]; int64_t shape[4]; int nd; int64_t doff;
  FILE *fo = NULL;
  int rc = npy_open(off_path, &fo, descr, shape, &nd, &doff);
  if (rc) { *perr = sqlite3_mprintf("cannot read %s", off_path); return rc; }
  if (nd != 1 || strcmp(descr, "<i8")) { fclose(fo); *perr = sqlite3_mprintf("offsets must be int64 [n+1]"); return SQLITE_ERROR; }
  s->N = shape[0] - 1;
  s->off = (int64_t *)malloc(sizeof(int64_t) * (s->N + 1));
  if (!s->off) { fclose(fo); return SQLITE_NOMEM; }
  fseeko(fo, (off_t)doff, SEEK_SET);
  if (fread(s->off, 8, (size_t)(s->N + 1), fo) != (size_t)(s->N + 1)) { fclose(fo); return SQLITE_IOERR; }
  fclose(fo);
  s->T = s->off[s->N];
  rc = npy_open(vec_path, &s->f, descr, shape, &nd, &s->data_off);
  if (rc) { *perr = sqlite3_mprintf("cannot read %s", vec_path); return rc; }
  if (nd != 2 || shape[1] != dim || shape[0] < s->T) { *perr = sqlite3_mprintf("vectors must be [T, %d]", dim); return SQLITE_ERROR; }
  if (!strcmp(descr, "<f2")) s->esize = 2;
  else if (!strcmp(descr, "<f4")) s->esize = 4;
  else { *perr = sqlite3_mprintf("vectors must be float16 or float32"); return SQLITE_ERROR; }
  return SQLITE_OK;
}

static int src_open_buffer(DocSrc *s, LtVtab *vt) {
  memset(s, 0, sizeof *s);
  s->dim = vt->cfg.dim;
  sqlite3_stmt *st = NULL;
  int rc = prep_fmt(vt->db, &st, "SELECT count(*) FROM \"%w\".\"%w_buffer\"", vt->zDb, vt->zName);
  if (rc) return rc;
  if (sqlite3_step(st) == SQLITE_ROW) s->N = sqlite3_column_int64(st, 0);
  sqlite3_finalize(st);
  s->off = (int64_t *)malloc(sizeof(int64_t) * (s->N + 1));
  s->rowids = (int64_t *)malloc(sizeof(int64_t) * (s->N + 1));
  if (!s->off || !s->rowids) return SQLITE_NOMEM;
  rc = prep_fmt(vt->db, &st, "SELECT id, length(data) FROM \"%w\".\"%w_buffer\" ORDER BY id", vt->zDb, vt->zName);
  if (rc) return rc;
  int64_t i = 0; s->off[0] = 0;
  while (sqlite3_step(st) == SQLITE_ROW && i < s->N) {
    s->rowids[i] = sqlite3_column_int64(st, 0);
    s->off[i + 1] = s->off[i] + sqlite3_column_int64(st, 1) / (2 * s->dim);
    i++;
  }
  sqlite3_finalize(st);
  s->N = i; s->T = s->off[i];
  int ident = 1;
  for (int64_t j = 0; j < s->N && ident; j++) ident = s->rowids[j] == j;
  if (ident) { free(s->rowids); s->rowids = NULL; }
  /* read path needs rowids even if identity */
  if (!s->rowids) {
    s->rowids = (int64_t *)malloc(sizeof(int64_t) * (s->N + 1));
    if (!s->rowids) return SQLITE_NOMEM;
    for (int64_t j = 0; j < s->N; j++) s->rowids[j] = j;
  }
  return prep_fmt(vt->db, &s->st, "SELECT data FROM \"%w\".\"%w_buffer\" WHERE id>=? ORDER BY id LIMIT ?", vt->zDb, vt->zName);
}

/* Writes a byte stream as fixed-size rows (id 1, 2, ...) of one table. */
typedef struct RowWriter {
  sqlite3_stmt *ins;
  uint8_t *buf; int chunk, fill;
  int64_t next_id, bytes;
  int rc;
} RowWriter;

static int rw_open(RowWriter *w, LtVtab *vt, const char *suffix, int chunk) {
  memset(w, 0, sizeof *w);
  w->chunk = chunk; w->next_id = 1;
  w->buf = (uint8_t *)malloc(chunk);
  if (!w->buf) return SQLITE_NOMEM;
  return prep_fmt(vt->db, &w->ins, "INSERT INTO \"%w\".\"%w_%s\"(id, data) VALUES (?, ?)", vt->zDb, vt->zName, suffix);
}
static void rw_flush(RowWriter *w) {
  if (w->rc || w->fill == 0) return;
  sqlite3_bind_int64(w->ins, 1, w->next_id++);
  sqlite3_bind_blob(w->ins, 2, w->buf, w->fill, SQLITE_STATIC);
  if (sqlite3_step(w->ins) != SQLITE_DONE) w->rc = SQLITE_ERROR;
  sqlite3_reset(w->ins);
  w->fill = 0;
}
static void rw_write(RowWriter *w, const uint8_t *p, uint64_t n) {
  while (n > 0 && !w->rc) {
    int m = (int)((uint64_t)(w->chunk - w->fill) < n ? (uint64_t)(w->chunk - w->fill) : n);
    memcpy(w->buf + w->fill, p, m);
    w->fill += m; p += m; n -= m; w->bytes += m;
    if (w->fill == w->chunk) rw_flush(w);
  }
}
static int rw_close(RowWriter *w) {
  rw_flush(w);
  sqlite3_finalize(w->ins);
  free(w->buf);
  return w->rc;
}

/* Encode a block of vectors: codes (argmax dot) and packed residuals. */
typedef struct { const LtCodec *cd; const float *X; const uint32_t *codes; uint8_t *res; } ResCtx;
static void res_range(void *vctx, int64_t lo, int64_t hi, int tid) {
  (void)tid;
  ResCtx *c = (ResCtx *)vctx;
  for (int64_t i = lo; i < hi; i++)
    lc_encode_residual(c->cd, c->X + (size_t)i * c->cd->dim, c->codes[i], c->res + (size_t)i * c->cd->rbytes);
}

static int do_build(LtVtab *vt, DocSrc *src) {
  LtCfg *cfg = &vt->cfg;
  LtIndex *ix = &vt->ix;
  sqlite3 *db = vt->db;
  int rc = SQLITE_OK, dim = cfg->dim;
  double t0 = now_ms();
  float *X = NULL, *blk = NULL;
  uint32_t *codes = NULL, *acodes = NULL, *cursor = NULL;
  uint8_t *res = NULL, *seg = NULL;
  int64_t *perm = NULL, *last = NULL;
  sqlite3_stmt *ins = NULL;
  char *err = NULL;

  if (src->N < 1 || src->T < 2) { set_err(vt, "late_plaid: no documents"); return SQLITE_ERROR; }
  if (src->N >= ((int64_t)1 << 32) - 1) { set_err(vt, "late_plaid: too many documents"); return SQLITE_ERROR; }
  ix_free(ix);
  ix->dim = dim; ix->nbits = cfg->nbits; ix->layout = cfg->layout;
  ix->N = src->N; ix->T = src->T;
  ix->rowid_identity = src->rowids == NULL;
  if (src->rowids) { ix->rowid_identity = 1; for (int64_t i = 0; i < src->N; i++) if (src->rowids[i] != i) { ix->rowid_identity = 0; break; } }

  /* Remove any previous index. */
  static const char *const sfx[] = { "meta", "docs", "ivf", "post", "rowids", "cells" };
  for (int i = 0; i < 6 && rc == SQLITE_OK; i++) rc = exec_fmt(db, &err, "DELETE FROM \"%w\".\"%w_%s\"", vt->zDb, vt->zName, sfx[i]);
  if (rc) { set_err(vt, "%s", err ? err : "late_plaid: cannot clear index"); sqlite3_free(err); return rc; }

  /* Page size decides the stream row size. */
  int pgsz = 4096;
  {
    sqlite3_stmt *st = NULL;
    if (prep_fmt(db, &st, "PRAGMA \"%w\".page_size", vt->zDb) == SQLITE_OK && sqlite3_step(st) == SQLITE_ROW)
      pgsz = sqlite3_column_int(st, 0);
    sqlite3_finalize(st);
  }
  ix->chunk = cfg->chunk > 0 ? cfg->chunk : pgsz - 39;   /* one row fills one page: payload = 4-byte record header + chunk = usable-35, the largest local payload */

  /* ---- number of centroids */
  int64_t T = src->T, N = src->N;
  int K = cfg->K;
  if (K <= 0) { double x = 16.0 * sqrt((double)T); K = 1; while ((double)K * 2 <= x) K *= 2; }
  /* ---- k-means sample (fast-plaid: 1 + 16*sqrt(120 N) documents) */
  int64_t ns = cfg->sample_docs > 0 ? cfg->sample_docs : (int64_t)(1 + floor(16.0 * sqrt(120.0 * (double)N)));
  if (ns > N) ns = N;
  uint64_t rng = cfg->seed;
  perm = (int64_t *)malloc(sizeof(int64_t) * N);
  if (!perm) { rc = SQLITE_NOMEM; goto out; }
  for (int64_t i = 0; i < N; i++) perm[i] = i;
  for (int64_t i = 0; i < ns; i++) { int64_t j = i + (int64_t)lt_rng_below(&rng, (uint64_t)(N - i)); int64_t t = perm[i]; perm[i] = perm[j]; perm[j] = t; }
  int64_t st_tok = 0;
  for (int64_t i = 0; i < ns; i++) st_tok += src->off[perm[i] + 1] - src->off[perm[i]];
  X = (float *)malloc(sizeof(float) * (size_t)st_tok * dim);
  if (!X) { rc = SQLITE_NOMEM; goto out; }
  for (int64_t i = 0, pos = 0; i < ns && rc == SQLITE_OK; i++) {
    rc = src_read(src, perm[i], perm[i] + 1, X + (size_t)pos * dim);
    pos += src->off[perm[i] + 1] - src->off[perm[i]];
  }
  if (rc) goto out;
  if (K > st_tok) { K = 1; while ((int64_t)K * 2 <= st_tok) K *= 2; }
  ix->K = K;
  ix->kbits = lt_ceil_log2((uint64_t)K);
  ix->idbits = lt_ceil_log2((uint64_t)N);
  /* Held-out set: the last tokens of the sample (fast-plaid walks the sample from the end). */
  int64_t nh = (int64_t)llround(fmin(0.05 * (double)st_tok, (double)cfg->heldout_max));
  if (nh < 1) nh = st_tok < 1000 ? st_tok : 1000;
  float *H = (float *)malloc(sizeof(float) * (size_t)nh * dim);
  if (!H) { rc = SQLITE_NOMEM; goto out; }
  memcpy(H, X + (size_t)(st_tok - nh) * dim, sizeof(float) * (size_t)nh * dim);
  /* Training points: at most K * ppc, a random subset. */
  int64_t ntrain = st_tok;
  if (cfg->ppc > 0 && (int64_t)K * cfg->ppc < st_tok) {
    ntrain = (int64_t)K * cfg->ppc;
    for (int64_t i = 0; i < ntrain; i++) {
      int64_t j = i + (int64_t)lt_rng_below(&rng, (uint64_t)(st_tok - i));
      if (j != i) for (int d = 0; d < dim; d++) { float t = X[(size_t)i * dim + d]; X[(size_t)i * dim + d] = X[(size_t)j * dim + d]; X[(size_t)j * dim + d] = t; }
    }
  }
  vlog(vt, "N=%lld T=%lld K=%d sample docs=%lld tokens=%lld train=%lld heldout=%lld", (long long)N, (long long)T, K,
       (long long)ns, (long long)st_tok, (long long)ntrain, (long long)nh);
  LtCodec *cd = &ix->codec;
  cd->dim = dim; cd->nbits = cfg->nbits; cd->K = K; cd->rbytes = dim * cfg->nbits / 8;
  cd->centroids = (float *)malloc(sizeof(float) * (size_t)K * dim);
  if (!cd->centroids) { free(H); rc = SQLITE_NOMEM; goto out; }
  if (lc_kmeans(X, ntrain, K, dim, cfg->iters, cfg->seed, cfg->threads, cd->centroids)) { free(H); rc = SQLITE_NOMEM; goto out; }
  free(X); X = NULL;
  vlog(vt, "k-means done in %.1f s", (now_ms() - t0) / 1000);
  {
    uint32_t *hc = (uint32_t *)malloc(sizeof(uint32_t) * nh);
    if (!hc) { free(H); rc = SQLITE_NOMEM; goto out; }
    lc_assign(H, nh, cd->centroids, NULL, K, dim, hc, NULL, cfg->threads);
    rc = lc_train_codec(cd, H, nh, hc);
    free(hc); free(H);
    if (rc) { rc = SQLITE_NOMEM; goto out; }
  }

  /* ---- encode every document; write doc rows (plaid layout) */
  const int rb = cd->rbytes;
  codes = (uint32_t *)malloc(sizeof(uint32_t) * (size_t)T);
  int keep_res = (cfg->layout & LAY_WARP) && (int64_t)T * rb <= cfg->mem_mb * 1048576;
  if (keep_res) res = (uint8_t *)malloc((size_t)T * rb);
  int64_t blk_tok = 262144;
  blk = (float *)malloc(sizeof(float) * (size_t)blk_tok * dim);
  uint8_t *bres = (uint8_t *)malloc((size_t)blk_tok * rb);
  uint8_t *row = NULL; size_t rowcap = 0;
  if (!codes || (keep_res && !res) || !blk || !bres) { free(bres); rc = SQLITE_NOMEM; goto out; }
  if (cfg->layout & LAY_PLAID) {
    rc = prep_fmt(db, &ins, "INSERT INTO \"%w\".\"%w_docs\"(id, data) VALUES (?, ?)", vt->zDb, vt->zName);
    if (rc) { free(bres); goto out; }
  }
  for (int64_t d0 = 0; d0 < N && rc == SQLITE_OK;) {
    int64_t d1 = d0 + 1;
    while (d1 < N && src->off[d1 + 1] - src->off[d0] <= blk_tok) d1++;
    int64_t t0b = src->off[d0], nt = src->off[d1] - t0b;
    if (nt > blk_tok) {                 /* one huge document */
      float *nb = (float *)realloc(blk, sizeof(float) * (size_t)nt * dim);
      uint8_t *nr = (uint8_t *)realloc(bres, (size_t)nt * rb);
      if (!nb || !nr) { free(nb ? nb : blk); blk = NULL; free(nr ? nr : bres); bres = NULL; rc = SQLITE_NOMEM; break; }
      blk = nb; bres = nr; blk_tok = nt;
    }
    rc = src_read(src, d0, d1, blk);
    if (rc) break;
    lc_assign(blk, nt, cd->centroids, NULL, K, dim, codes + t0b, NULL, cfg->threads);
    ResCtx rcx = { cd, blk, codes + t0b, bres };
    lt_parallel_for(nt, 1024, cfg->threads, res_range, &rcx);
    if (keep_res) memcpy(res + (size_t)t0b * rb, bres, (size_t)nt * rb);
    if (cfg->layout & LAY_PLAID) {
      for (int64_t d = d0; d < d1 && rc == SQLITE_OK; d++) {
        int64_t a = src->off[d] - t0b, n = src->off[d + 1] - src->off[d];
        if (n > 65535) { set_err(vt, "late_plaid: document %lld has more than 65535 tokens", (long long)d); rc = SQLITE_TOOBIG; break; }
        size_t cb = lt_bits_bytes((uint64_t)n, ix->kbits), need = 2 + cb + (size_t)n * rb;
        if (need > rowcap) { rowcap = need * 2; uint8_t *t = (uint8_t *)realloc(row, rowcap); if (!t) { rc = SQLITE_NOMEM; break; } row = t; }
        memset(row, 0, need);
        row[0] = (uint8_t)n; row[1] = (uint8_t)(n >> 8);
        for (int64_t j = 0; j < n; j++) lt_put_bits(row + 2, (uint64_t)j * ix->kbits, ix->kbits, codes[t0b + a + j]);
        memcpy(row + 2 + cb, bres + (size_t)a * rb, (size_t)n * rb);
        sqlite3_bind_int64(ins, 1, d);
        sqlite3_bind_blob(ins, 2, row, (int)need, SQLITE_STATIC);
        if (sqlite3_step(ins) != SQLITE_DONE) { set_err(vt, "%s", sqlite3_errmsg(db)); rc = SQLITE_ERROR; }
        sqlite3_reset(ins);
      }
    }
    if (cfg->verbose && (d1 * 20 / N) != (d0 * 20 / N)) vlog(vt, "encoded %lld/%lld docs (%.1f s)", (long long)d1, (long long)N, (now_ms() - t0) / 1000);
    d0 = d1;
  }
  free(bres); free(row);
  sqlite3_finalize(ins); ins = NULL;
  if (rc) goto out;

  /* ---- IVF: distinct documents per centroid, in increasing order */
  if (cfg->layout & LAY_PLAID) {
    ix->ivf_cnt = (uint32_t *)calloc(K, sizeof(uint32_t));
    ix->ivf_off = (uint64_t *)malloc(sizeof(uint64_t) * (K + 1));
    last = (int64_t *)malloc(sizeof(int64_t) * K);
    if (!ix->ivf_cnt || !ix->ivf_off || !last) { rc = SQLITE_NOMEM; goto out; }
    for (int c = 0; c < K; c++) last[c] = -1;
    for (int64_t d = 0; d < N; d++)
      for (int64_t t = src->off[d]; t < src->off[d + 1]; t++) if (last[codes[t]] != d) { last[codes[t]] = d; ix->ivf_cnt[codes[t]]++; }
    ix->ivf_entries = 0;
    for (int c = 0; c < K; c++) ix->ivf_entries += ix->ivf_cnt[c];
    ix_offsets(ix);
    uint64_t total = ix->ivf_off[K];
    seg = (uint8_t *)calloc(total + LT_BITPAD, 1);
    cursor = (uint32_t *)calloc(K, sizeof(uint32_t));
    if (!seg || !cursor) { rc = SQLITE_NOMEM; goto out; }
    for (int c = 0; c < K; c++) last[c] = -1;
    for (int64_t d = 0; d < N; d++)
      for (int64_t t = src->off[d]; t < src->off[d + 1]; t++) {
        uint32_t c = codes[t];
        if (last[c] == d) continue;
        last[c] = d;
        lt_put_bits(seg + ix->ivf_off[c], (uint64_t)cursor[c]++ * ix->idbits, ix->idbits, (uint32_t)d);
      }
    RowWriter w;
    rc = rw_open(&w, vt, "ivf", ix->chunk);
    if (rc == SQLITE_OK) { rw_write(&w, seg, total); rc = rw_close(&w); }
    free(seg); seg = NULL; free(cursor); cursor = NULL;
    if (rc) goto out;
    vlog(vt, "IVF: %lld entries, %llu bytes (%.1f s)", (long long)ix->ivf_entries, (unsigned long long)total, (now_ms() - t0) / 1000);
  }

  /* ---- posting lists (warp layout), in centroid ranges that fit mem_mb */
  if (cfg->layout & LAY_WARP) {
    ix->post_cnt = (uint32_t *)calloc(K, sizeof(uint32_t));
    ix->post_off = (uint64_t *)malloc(sizeof(uint64_t) * (K + 1));
    cursor = (uint32_t *)calloc(K, sizeof(uint32_t));
    if (!ix->post_cnt || !ix->post_off || !cursor) { rc = SQLITE_NOMEM; goto out; }
    for (int64_t t = 0; t < T; t++) ix->post_cnt[codes[t]]++;
    ix_offsets(ix);
    RowWriter w;
    rc = rw_open(&w, vt, "post", ix->chunk);
    uint64_t budget = (uint64_t)cfg->mem_mb * 1048576;
    int passes = 0;
    for (int c0 = 0; c0 < K && rc == SQLITE_OK;) {
      int c1 = c0 + 1;
      while (c1 < K && ix->post_off[c1 + 1] - ix->post_off[c0] <= budget) c1++;
      uint64_t base = ix->post_off[c0], len = ix->post_off[c1] - base;
      seg = (uint8_t *)calloc(len + LT_BITPAD, 1);
      if (!seg) { rc = SQLITE_NOMEM; break; }
      passes++;
      for (int64_t d0 = 0; d0 < N && rc == SQLITE_OK;) {
        int64_t d1 = N;
        if (!keep_res) {            /* re-read vectors to recompute residuals */
          d1 = d0 + 1;
          while (d1 < N && src->off[d1 + 1] - src->off[d0] <= blk_tok) d1++;
          rc = src_read(src, d0, d1, blk);
          if (rc) break;
        }
        for (int64_t d = d0; d < d1; d++)
          for (int64_t t = src->off[d]; t < src->off[d + 1]; t++) {
            uint32_t c = codes[t];
            if ((int)c < c0 || (int)c >= c1) continue;
            uint8_t *s = seg + (ix->post_off[c] - base);
            uint32_t j = cursor[c]++;
            lt_put_bits(s, (uint64_t)j * ix->idbits, ix->idbits, (uint32_t)d);
            uint8_t *dst = s + lt_bits_bytes(ix->post_cnt[c], ix->idbits) + (size_t)j * rb;
            if (keep_res) memcpy(dst, res + (size_t)t * rb, rb);
            else lc_encode_residual(cd, blk + (size_t)(t - src->off[d0]) * dim, c, dst);
          }
        d0 = d1;
      }
      if (rc == SQLITE_OK) rw_write(&w, seg, len);
      free(seg); seg = NULL;
      c0 = c1;
    }
    int rc2 = rw_close(&w);
    if (rc == SQLITE_OK) rc = rc2;
    free(cursor); cursor = NULL;
    if (rc) goto out;
    vlog(vt, "postings: %llu bytes in %d pass(es) (%.1f s)", (unsigned long long)ix->post_off[K], passes, (now_ms() - t0) / 1000);
  }

  /* ---- rowid map and static data */
  if (!ix->rowid_identity) {
    rc = prep_fmt(db, &ins, "INSERT INTO \"%w\".\"%w_rowids\"(id, data) VALUES (?, ?)", vt->zDb, vt->zName);
    for (int64_t d = 0; d < N && rc == SQLITE_OK; d++) {
      uint8_t b[8]; lt_put_u64(b, (uint64_t)src->rowids[d]);
      sqlite3_bind_int64(ins, 1, d);
      sqlite3_bind_blob(ins, 2, b, 8, SQLITE_TRANSIENT);
      if (sqlite3_step(ins) != SQLITE_DONE) rc = SQLITE_ERROR;
      sqlite3_reset(ins);
    }
    sqlite3_finalize(ins); ins = NULL;
    if (rc) goto out;
  }
  {
    LtBuf b; memset(&b, 0, sizeof b);
    ix_serialize(ix, &b);
    if (b.oom) { lt_buf_free(&b); rc = SQLITE_NOMEM; goto out; }
    RowWriter w;
    rc = rw_open(&w, vt, "meta", ix->chunk);
    if (rc == SQLITE_OK) { rw_write(&w, b.p, b.n); rc = rw_close(&w); }
    lt_buf_free(&b);
  }
  vlog(vt, "build done in %.1f s", (now_ms() - t0) / 1000);

out:
  if (rc && !vt->base.zErrMsg) set_err(vt, "late_plaid: build failed (%d): %s", rc, sqlite3_errmsg(db));
  sqlite3_finalize(ins);
  free(X); free(blk); free(codes); free(acodes); free(cursor); free(res); free(seg); free(perm); free(last);
  ix->loaded = 0;                     /* reload (with fresh roots) on next query */
  if (rc) ix_free(ix);
  return rc;
}

/* 'finalize': record the leaf page of every stream row, run-length encoded. */
typedef struct { uint32_t *runs; int n, cap; int64_t prev_row; uint32_t prev_pg; } RunCtx;
static int run_cb(void *ctx, int64_t rowid, uint32_t pg, const uint8_t *d, int len) {
  (void)d; (void)len;
  RunCtx *r = (RunCtx *)ctx;
  int64_t row = rowid - 1;
  if (r->n > 0 && row == r->prev_row + 1 && pg == r->prev_pg + 1) { r->prev_row = row; r->prev_pg = pg; return SQLITE_OK; }
  if (r->n == r->cap) {
    r->cap = r->cap ? r->cap * 2 : 64;
    uint32_t *t = (uint32_t *)realloc(r->runs, sizeof(uint32_t) * 2 * r->cap);
    if (!t) return SQLITE_NOMEM;
    r->runs = t;
  }
  r->runs[2 * r->n] = (uint32_t)row; r->runs[2 * r->n + 1] = pg; r->n++;
  r->prev_row = row; r->prev_pg = pg;
  return SQLITE_OK;
}

static int do_finalize(LtVtab *vt) {
  LtIndex *ix = &vt->ix;
  ix->loaded = 0;
  lpg_drop_cache(&vt->rd);
  int rc = ix_load(vt);
  if (rc) return rc;
  if (reader(vt) < 0) { set_err(vt, "late_plaid: finalize needs direct page reads (rollback journal mode)"); return SQLITE_ERROR; }
  RunCtx a; memset(&a, 0, sizeof a);
  RunCtx b; memset(&b, 0, sizeof b);
  if (ix->root_ivf && (ix->layout & LAY_PLAID)) rc = lpg_scan(&vt->rd, ix->root_ivf, run_cb, &a);
  if (rc == SQLITE_OK && ix->root_post && (ix->layout & LAY_WARP)) rc = lpg_scan(&vt->rd, ix->root_post, run_cb, &b);
  if (rc) { free(a.runs); free(b.runs); set_err(vt, "late_plaid: finalize scan failed (%d)", rc); return rc; }
  free(ix->ivf_runs); free(ix->post_runs);
  ix->ivf_runs = a.runs; ix->n_ivf_runs = a.n;
  ix->post_runs = b.runs; ix->n_post_runs = b.n;
  LtBuf buf; memset(&buf, 0, sizeof buf);
  ix_serialize(ix, &buf);
  char *err = NULL;
  rc = exec_fmt(vt->db, &err, "DELETE FROM \"%w\".\"%w_meta\"", vt->zDb, vt->zName);
  if (rc == SQLITE_OK) {
    RowWriter w;
    rc = rw_open(&w, vt, "meta", ix->chunk);
    if (rc == SQLITE_OK) { rw_write(&w, buf.p, buf.n); rc = rw_close(&w); }
  }
  sqlite3_free(err);
  lt_buf_free(&buf);
  ix->loaded = 0;
  lpg_drop_cache(&vt->rd);
  return rc;
}

/* ================================================================ QUERY */

typedef struct QOpts {
  int k, nprobe, layout, approx, ndocs, impute, cross, rerank, exact, trace, cprobe;
  float tcs;
} QOpts;
enum { APPROX_CODES = 0, APPROX_IVF = 1 };

static void opts_parse(QOpts *o, const char *s) {
  if (!s) return;
  while (*s) {
    while (*s == ' ' || *s == ',' || *s == ';') s++;
    char key[32]; int kl = 0;
    while (*s && *s != '=' && *s != ' ' && kl < 31) key[kl++] = *s++;
    key[kl] = 0;
    if (*s != '=') { while (*s && *s != ' ') s++; continue; }
    s++;
    char val[32]; int vl = 0;
    while (*s && *s != ' ' && *s != ',' && *s != ';' && vl < 31) val[vl++] = *s++;
    val[vl] = 0;
    int iv = atoi(val);
    if (!strcmp(key, "nprobe")) o->nprobe = iv;
    else if (!strcmp(key, "layout")) o->layout = !strcmp(val, "plaid") ? LAY_PLAID : !strcmp(val, "warp") ? LAY_WARP : 0;
    else if (!strcmp(key, "approx")) o->approx = !strcmp(val, "ivf") ? APPROX_IVF : APPROX_CODES;
    else if (!strcmp(key, "ndocs")) o->ndocs = iv;
    else if (!strcmp(key, "impute")) o->impute = iv;
    else if (!strcmp(key, "cross")) o->cross = iv;
    else if (!strcmp(key, "rerank")) o->rerank = iv;
    else if (!strcmp(key, "exact")) o->exact = iv;
    else if (!strcmp(key, "trace")) o->trace = iv;
    else if (!strcmp(key, "tcs")) o->tcs = (float)atof(val);
    else if (!strcmp(key, "k")) o->k = iv;
    else if (!strcmp(key, "cprobe")) o->cprobe = iv;
  }
}

/* Per-document accumulator: best score so far for each query token. */
typedef struct Acc {
  int nq;
  uint32_t *key; int *slot; int cap;     /* hash docid -> slot (key = docid+1) */
  uint32_t *doc; float *best; uint8_t *own; int n, scap;
} Acc;
/* best[s*nq+i]: max score of query token i over the document's tokens seen
** so far; own[s*nq+i]: one of them came from a centroid that token i itself
** probed. Missing pairs (own = 0) are imputed: max(best, impute[i]). */

static void acc_free(Acc *a) { free(a->key); free(a->slot); free(a->doc); free(a->best); free(a->own); memset(a, 0, sizeof *a); }

static int acc_grow_hash(Acc *a) {
  int ncap = a->cap ? a->cap * 2 : 1024;
  uint32_t *nk = (uint32_t *)calloc(ncap, sizeof(uint32_t));
  int *ns = (int *)malloc(sizeof(int) * ncap);
  if (!nk || !ns) { free(nk); free(ns); return 1; }
  for (int i = 0; i < a->n; i++) {
    uint32_t h = (a->doc[i] * 2654435761u) & (ncap - 1);
    while (nk[h]) h = (h + 1) & (ncap - 1);
    nk[h] = a->doc[i] + 1; ns[h] = i;
  }
  free(a->key); free(a->slot);
  a->key = nk; a->slot = ns; a->cap = ncap;
  return 0;
}

/* Slot of docid, created with all scores -inf if absent; -1 on OOM. */
static int acc_get(Acc *a, uint32_t d) {
  if ((a->n + 1) * 2 > a->cap && acc_grow_hash(a)) return -1;
  uint32_t h = (d * 2654435761u) & (a->cap - 1);
  while (a->key[h]) {
    if (a->key[h] == d + 1) return a->slot[h];
    h = (h + 1) & (a->cap - 1);
  }
  if (a->n == a->scap) {
    int nc = a->scap ? a->scap * 2 : 1024;
    uint32_t *nd = (uint32_t *)realloc(a->doc, sizeof(uint32_t) * nc);
    if (!nd) return -1;
    a->doc = nd;
    float *nb = (float *)realloc(a->best, sizeof(float) * (size_t)nc * a->nq);
    if (!nb) return -1;
    a->best = nb;
    uint8_t *no = (uint8_t *)realloc(a->own, (size_t)nc * a->nq);
    if (!no) return -1;
    a->own = no; a->scap = nc;
  }
  int s = a->n++;
  a->doc[s] = d;
  for (int i = 0; i < a->nq; i++) a->best[(size_t)s * a->nq + i] = -FLT_MAX;
  memset(a->own + (size_t)s * a->nq, 0, a->nq);
  a->key[h] = d + 1; a->slot[h] = s;
  return s;
}

typedef struct Cand { float score; uint32_t doc; } Cand;
static int cmp_cand(const void *a, const void *b) {
  const Cand *x = (const Cand *)a, *y = (const Cand *)b;
  if (x->score != y->score) return x->score > y->score ? -1 : 1;
  return x->doc < y->doc ? -1 : x->doc > y->doc;
}
static int cmp_cand_doc(const void *a, const void *b) {
  const Cand *x = (const Cand *)a, *y = (const Cand *)b;
  return x->doc < y->doc ? -1 : x->doc > y->doc;
}
static int cmp_u32q(const void *a, const void *b) {
  uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b;
  return x < y ? -1 : x > y;
}

typedef struct QStats {
  int nq, probed, lists_bytes_i; int64_t candidates, tokens_decoded, docs_fetched, list_entries, list_bytes, doc_bytes;
  double ms_total, ms_cpu;
  int static_rounds;
  int cells, cell_rounds; int64_t cell_bytes;
} QStats;

typedef struct Query {
  LtVtab *vt; LtIndex *ix; QOpts o; QStats st;
  const float *q; int nq;
  float *S;                    /* nF x nq centroid scores (rows: see srow) */
  int *Fidx;                   /* two-level: K -> row of S or -1 (NULL: row = centroid id) */
  uint32_t *probe;             /* nq x nprobe probed centroid per token */
  float *impute;               /* nq */
  uint32_t *P; int nP;         /* union of probed centroids, sorted */
  uint8_t *Pmask;              /* nP x nq: token i probed P[j] */
  Acc acc;
  Cand *res; int nres;         /* final results (doc, score) */
} Query;

static void q_free(Query *q) {
  free(q->S); free(q->Fidx); free(q->probe); free(q->impute); free(q->P); free(q->Pmask); free(q->res);
  acc_free(&q->acc);
}

static int load_cells(LtVtab *vt, const int *cells, int n);

/* Centroid-score row of fine centroid c, NULL if not scored this query. */
static inline const float *srow(const Query *q, uint32_t c) {
  int r = q->Fidx ? q->Fidx[c] : (int)c;
  return r < 0 ? NULL : q->S + (size_t)r * q->nq;
}

static int q_probe(Query *q) {
  LtIndex *ix = q->ix;
  int K = ix->K, nq = q->nq, np = q->o.nprobe;
  if (np < 1) np = 1;
  /* Candidate centroids F: all of them, or (two-level) the members of the
  ** cprobe best cells of every query token. */
  uint32_t *F = NULL; int nF = K;
  if (ix->G && ix->cells_loaded < ix->G) {
    int G = ix->G, cp = q->o.cprobe < 1 ? 1 : q->o.cprobe > G ? G : q->o.cprobe;
    int *cells = (int *)malloc(sizeof(int) * (size_t)nq * cp);
    float *bs = (float *)malloc(sizeof(float) * cp);
    if (!cells || !bs) { free(cells); free(bs); return SQLITE_NOMEM; }
    int nc = 0;
    for (int i = 0; i < nq; i++) {
      int n = 0, *bi = cells + nc;
      for (int g = 0; g < G; g++) {
        if (ix->cell_start[g + 1] == ix->cell_start[g]) continue;
        float sc = lt_dot(ix->coarse + (size_t)g * ix->dim, q->q + (size_t)i * ix->dim, ix->dim);
        if (n == cp && sc <= bs[n - 1]) continue;
        int j = n < cp ? n++ : cp - 1;
        while (j > 0 && bs[j - 1] < sc) { bs[j] = bs[j - 1]; bi[j] = bi[j - 1]; j--; }
        bs[j] = sc; bi[j] = g;
      }
      nc += n;
    }
    free(bs);
    /* unique cells, ascending */
    for (int i = 1; i < nc; i++) { int v = cells[i], j = i - 1; while (j >= 0 && cells[j] > v) { cells[j + 1] = cells[j]; j--; } cells[j + 1] = v; }
    int u = 0;
    for (int i = 0; i < nc; i++) if (u == 0 || cells[i] != cells[u - 1]) cells[u++] = cells[i];
    q->st.cells = u;
    int r0 = q->vt->rd.round;
    int64_t b0 = ix->cell_bytes_loaded;
    int rc = load_cells(q->vt, cells, u);
    q->st.cell_rounds = q->vt->rd.round - r0;
    q->st.cell_bytes = ix->cell_bytes_loaded - b0;
    if (rc) { free(cells); return rc; }
    nF = 0;
    for (int k = 0; k < u; k++) nF += (int)(ix->cell_start[cells[k] + 1] - ix->cell_start[cells[k]]);
    F = (uint32_t *)malloc(sizeof(uint32_t) * (nF ? nF : 1));
    q->Fidx = (int *)malloc(sizeof(int) * K);
    if (!F || !q->Fidx) { free(cells); free(F); return SQLITE_NOMEM; }
    for (int c = 0; c < K; c++) q->Fidx[c] = -1;
    int m = 0;
    for (int k = 0; k < u; k++)
      for (uint32_t c = ix->cell_start[cells[k]]; c < ix->cell_start[cells[k] + 1]; c++) { q->Fidx[c] = m; F[m++] = c; }
    free(cells);
  }
  if (np > nF) np = nF;
  if (np < 1) return SQLITE_OK;
  q->o.nprobe = np;
  q->S = (float *)malloc(sizeof(float) * (size_t)(nF ? nF : 1) * nq);
  q->probe = (uint32_t *)malloc(sizeof(uint32_t) * (size_t)nq * np);
  q->impute = (float *)malloc(sizeof(float) * nq);
  float *bs = (float *)malloc(sizeof(float) * np);
  if (!q->S || !q->probe || !q->impute || !bs) { free(F); free(bs); return SQLITE_NOMEM; }
  for (int r = 0; r < nF; r++) {
    const float *cv = cent(ix, F ? F[r] : (uint32_t)r);
    for (int i = 0; i < nq; i++) q->S[(size_t)r * nq + i] = lt_dot(cv, q->q + (size_t)i * ix->dim, ix->dim);
  }
  for (int i = 0; i < nq; i++) {
    uint32_t *bi = q->probe + (size_t)i * np;
    int n = 0;
    for (int r = 0; r < nF; r++) {
      float sc = q->S[(size_t)r * nq + i];
      if (n == np && sc <= bs[n - 1]) continue;
      int j = n < np ? n++ : np - 1;
      while (j > 0 && bs[j - 1] < sc) { bs[j] = bs[j - 1]; bi[j] = bi[j - 1]; j--; }
      bs[j] = sc; bi[j] = F ? F[r] : (uint32_t)r;
    }
    q->impute[i] = q->o.impute ? bs[n - 1] : 0.0f;
  }
  free(bs); free(F);
  /* union, optional t_cs pruning */
  uint32_t *all = (uint32_t *)malloc(sizeof(uint32_t) * (size_t)nq * np);
  if (!all) return SQLITE_NOMEM;
  memcpy(all, q->probe, sizeof(uint32_t) * (size_t)nq * np);
  qsort(all, (size_t)nq * np, sizeof(uint32_t), cmp_u32q);
  int u = 0;
  for (int j = 0; j < nq * np; j++) if (u == 0 || all[j] != all[u - 1]) all[u++] = all[j];
  if (q->o.tcs > 0) {
    int v = 0;
    for (int j = 0; j < u; j++) {
      const float *sr = srow(q, all[j]);
      float mx = -FLT_MAX;
      for (int i = 0; i < nq; i++) if (sr[i] > mx) mx = sr[i];
      if (mx >= q->o.tcs) all[v++] = all[j];
    }
    u = v;
  }
  q->P = all; q->nP = u;
  q->Pmask = (uint8_t *)calloc((size_t)(u ? u : 1) * nq, 1);
  if (!q->Pmask) return SQLITE_NOMEM;
  for (int i = 0; i < nq; i++)
    for (int j = 0; j < np; j++) {
      uint32_t c = q->probe[(size_t)i * np + j];
      uint32_t *hit = (uint32_t *)bsearch(&c, q->P, u, sizeof(uint32_t), cmp_u32q);
      if (hit) q->Pmask[(size_t)(hit - q->P) * nq + i] = 1;
    }
  q->st.probed = u;
  return SQLITE_OK;
}

static int cmp_i64q(const void *a, const void *b) {
  int64_t x = *(const int64_t *)a, y = *(const int64_t *)b;
  return x < y ? -1 : x > y;
}

enum { STREAM_IVF = 0, STREAM_POST = 1, STREAM_CELLS = 2 };

/* Fetch byte ranges [start[j], start[j]+len[j]) of a paged stream: all rows
** they touch are read in one round (hinted) or one round per b-tree level.
** On return segs[j] points into *pool (malloc'd, caller frees). */
static int fetch_ranges(LtVtab *vt, int stream, const uint64_t *start, const uint64_t *len, int n,
                        uint8_t **pool, uint8_t **segs) {
  LtIndex *ix = &vt->ix;
  const uint32_t *runs = stream == STREAM_POST ? ix->post_runs : stream == STREAM_IVF ? ix->ivf_runs : ix->cell_runs;
  int nruns = stream == STREAM_POST ? ix->n_post_runs : stream == STREAM_IVF ? ix->n_ivf_runs : ix->n_cell_runs;
  uint32_t root = stream == STREAM_POST ? ix->root_post : stream == STREAM_IVF ? ix->root_ivf : ix->root_cells;
  const char *sfx = stream == STREAM_POST ? "post" : stream == STREAM_IVF ? "ivf" : "cells";
  const int ch = ix->chunk;
  uint64_t total = 0;
  int64_t nrows = 0;
  for (int j = 0; j < n; j++) {
    total += len[j];
    if (len[j]) nrows += (int64_t)((start[j] + len[j] - 1) / ch - start[j] / ch + 1);
  }
  *pool = (uint8_t *)malloc(total + LT_BITPAD);
  int64_t *ids = (int64_t *)malloc(sizeof(int64_t) * (nrows ? nrows : 1));
  LpRow *rows = (LpRow *)calloc(nrows ? nrows : 1, sizeof(LpRow));
  if (!*pool || !ids || !rows) { free(ids); free(rows); return SQLITE_NOMEM; }
  memset(*pool + total, 0, LT_BITPAD);
  int64_t m = 0;
  for (int j = 0; j < n; j++) {
    if (!len[j]) continue;
    for (int64_t r = (int64_t)(start[j] / ch); r <= (int64_t)((start[j] + len[j] - 1) / ch); r++) ids[m++] = r + 1;
  }
  qsort(ids, (size_t)m, sizeof(int64_t), cmp_i64q);
  int64_t u = 0;
  for (int64_t i = 0; i < m; i++) if (u == 0 || ids[i] != ids[u - 1]) ids[u++] = ids[i];
  for (int64_t i = 0; i < u; i++) { rows[i].rowid = ids[i]; rows[i].hint = nruns ? run_hint(runs, nruns, ids[i] - 1) : 0; }
  free(ids);
  char tb[300]; snprintf(tb, sizeof tb, "%s_%s", vt->zName, sfx);
  int rc = lpg_multiget(&vt->rd, tb, root, rows, (int)u);
  uint64_t pos = 0;
  for (int j = 0; j < n && rc == SQLITE_OK; j++) {
    uint64_t a = start[j], b = start[j] + len[j];
    segs[j] = *pool + pos;
    for (uint64_t x = a; x < b;) {
      int64_t r = (int64_t)(x / ch);
      int64_t lo = 0, hi = u - 1, f = -1;
      while (lo <= hi) { int64_t mid = (lo + hi) / 2; if (rows[mid].rowid == r + 1) { f = mid; break; } if (rows[mid].rowid < r + 1) lo = mid + 1; else hi = mid - 1; }
      if (f < 0 || !rows[f].data) { rc = SQLITE_CORRUPT; break; }
      uint64_t inrow = x - (uint64_t)r * ch, take = (uint64_t)((r + 1) * ch) - x;
      if (take > b - x) take = b - x;
      if (inrow + take > (uint64_t)rows[f].len) { rc = SQLITE_CORRUPT; break; }
      memcpy(*pool + pos, rows[f].data + inrow, take);
      pos += take; x += take;
    }
  }
  for (int64_t i = 0; i < u; i++) free(rows[i].data);
  free(rows);
  return rc;
}

/* Make sure the given cells (two-level mode) are loaded: one round. */
static int load_cells(LtVtab *vt, const int *cells, int n) {
  LtIndex *ix = &vt->ix;
  if (!ix->G) return SQLITE_OK;
  uint64_t *st = (uint64_t *)malloc(sizeof(uint64_t) * (n ? n : 1)), *ln = (uint64_t *)malloc(sizeof(uint64_t) * (n ? n : 1));
  int *which = (int *)malloc(sizeof(int) * (n ? n : 1));
  uint8_t **segs = (uint8_t **)malloc(sizeof(uint8_t *) * (n ? n : 1));
  uint8_t *pool = NULL;
  int m = 0, rc = SQLITE_OK;
  if (!st || !ln || !which || !segs) rc = SQLITE_NOMEM;
  for (int i = 0; i < n && rc == SQLITE_OK; i++) {
    int g = cells[i];
    if (ix->cell_cent[g] || ix->cell_start[g + 1] == ix->cell_start[g]) continue;
    int dup = 0;
    for (int k = 0; k < m; k++) if (which[k] == g) dup = 1;
    if (dup) continue;
    which[m] = g; st[m] = ix->cell_off[g]; ln[m] = ix->cell_off[g + 1] - ix->cell_off[g]; m++;
  }
  if (rc == SQLITE_OK && m) rc = fetch_ranges(vt, STREAM_CELLS, st, ln, m, &pool, segs);
  for (int k = 0; k < m && rc == SQLITE_OK; k++) rc = cell_install(ix, which[k], segs[k]);
  free(pool); free(st); free(ln); free(which); free(segs);
  return rc;
}

static int load_all_cells(LtVtab *vt) {
  LtIndex *ix = &vt->ix;
  if (!ix->G || ix->cells_loaded == ix->G) return SQLITE_OK;
  int *all = (int *)malloc(sizeof(int) * ix->G);
  if (!all) return SQLITE_NOMEM;
  int n = 0;
  for (int g = 0; g < ix->G; g++) if (!ix->cell_cent[g]) all[n++] = g;
  /* one call per 4096 cells keeps the duplicate check cheap */
  int rc = SQLITE_OK;
  for (int i = 0; i < n && rc == SQLITE_OK; i += 4096) rc = load_cells(vt, all + i, n - i < 4096 ? n - i : 4096);
  free(all);
  return rc;
}

/* Fetch the list segments of the probed centroids (IVF or postings). */
static int fetch_segments(Query *q, int warp, uint8_t **pool, uint8_t **segs) {
  LtIndex *ix = q->ix;
  uint64_t *st = (uint64_t *)malloc(sizeof(uint64_t) * (q->nP ? q->nP : 1)), *ln = (uint64_t *)malloc(sizeof(uint64_t) * (q->nP ? q->nP : 1));
  if (!st || !ln) { free(st); free(ln); return SQLITE_NOMEM; }
  for (int j = 0; j < q->nP; j++) {
    uint32_t c = q->P[j];
    st[j] = warp ? ix->post_off[c] : ix->ivf_off[c];
    ln[j] = seg_bytes(ix, warp, warp ? ix->post_cnt[c] : ix->ivf_cnt[c]);
    q->st.list_bytes += (int64_t)ln[j];
  }
  int rc = fetch_ranges(q->vt, warp ? STREAM_POST : STREAM_IVF, st, ln, q->nP, pool, segs);
  free(st); free(ln);
  return rc;
}

/* Exact decompressed MaxSim of one doc row. */
static float doc_maxsim(Query *q, const uint8_t *row, int len, float *tmp, float *best) {
  LtIndex *ix = q->ix;
  if (len < 2) return 0;
  int n = row[0] | (row[1] << 8);
  size_t cb = lt_bits_bytes((uint64_t)n, ix->kbits);
  if ((size_t)len < 2 + cb + (size_t)n * ix->codec.rbytes) return 0;
  const uint8_t *res = row + 2 + cb;
  for (int i = 0; i < q->nq; i++) best[i] = -FLT_MAX;
  for (int j = 0; j < n; j++) {
    uint32_t c = lt_get_bits(row + 2, (uint64_t)j * ix->kbits, ix->kbits);
    const float *cv = (int)c < ix->K ? cent(ix, c) : NULL;
    if (!cv) continue;
    lc_decode_with(&ix->codec, cv, res + (size_t)j * ix->codec.rbytes, tmp);
    for (int i = 0; i < q->nq; i++) {
      float s = lt_dot(tmp, q->q + (size_t)i * ix->dim, ix->dim);
      if (s > best[i]) best[i] = s;
    }
  }
  q->st.tokens_decoded += n;
  float sum = 0;
  for (int i = 0; i < q->nq; i++) sum += best[i] > -FLT_MAX ? best[i] : 0;
  return sum;
}

/* Centroid-only approximate score of one doc row. */
static float doc_approx(Query *q, const uint8_t *row, int len, float *best) {
  LtIndex *ix = q->ix;
  if (len < 2) return -FLT_MAX;
  int n = row[0] | (row[1] << 8);
  for (int i = 0; i < q->nq; i++) best[i] = -FLT_MAX;
  for (int j = 0; j < n; j++) {
    uint32_t c = lt_get_bits(row + 2, (uint64_t)j * ix->kbits, ix->kbits);
    const float *s = (int)c < ix->K ? srow(q, c) : NULL;
    if (!s) continue;
    for (int i = 0; i < q->nq; i++) if (s[i] > best[i]) best[i] = s[i];
  }
  float sum = 0;
  for (int i = 0; i < q->nq; i++) sum += best[i] > -FLT_MAX ? best[i] : 0;
  return sum;
}

/* Fetch doc rows for cands (sorted by doc id inside), exact-score them. */
static int rescore_docs(Query *q, Cand *c, int n, LpRow **prows) {
  LtIndex *ix = q->ix;
  qsort(c, n, sizeof(Cand), cmp_cand_doc);
  LpRow *rows = *prows;
  int own = 0;
  if (!rows) {
    rows = (LpRow *)calloc(n ? n : 1, sizeof(LpRow));
    if (!rows) return SQLITE_NOMEM;
    for (int i = 0; i < n; i++) rows[i].rowid = c[i].doc;
    char tb[300]; snprintf(tb, sizeof tb, "%s_docs", q->vt->zName);
    int rc = lpg_multiget(&q->vt->rd, tb, ix->root_docs, rows, n);
    if (rc) { for (int i = 0; i < n; i++) free(rows[i].data); free(rows); return rc; }
    own = 1;
    q->st.docs_fetched += n;
  }
  float tmp[512], *best = (float *)malloc(sizeof(float) * q->nq);
  if (!best) return SQLITE_NOMEM;
  double t = now_ms();
  for (int i = 0; i < n; i++) {
    c[i].score = rows[i].data ? doc_maxsim(q, rows[i].data, rows[i].len, tmp, best) : -FLT_MAX;
    q->st.doc_bytes += rows[i].len;
  }
  q->st.ms_cpu += now_ms() - t;
  free(best);
  if (own) { for (int i = 0; i < n; i++) free(rows[i].data); free(rows); }
  qsort(c, n, sizeof(Cand), cmp_cand);
  return SQLITE_OK;
}

/* Candidates from the accumulator, scored with imputation, best first. */
static Cand *acc_rank(Query *q, int *pn) {
  Acc *a = &q->acc;
  Cand *c = (Cand *)malloc(sizeof(Cand) * (a->n ? a->n : 1));
  if (!c) return NULL;
  for (int s = 0; s < a->n; s++) {
    float sum = 0;
    for (int i = 0; i < q->nq; i++) {
      float b = a->best[(size_t)s * q->nq + i];
      if (!a->own[(size_t)s * q->nq + i] && b < q->impute[i]) b = q->impute[i];
      sum += b;
    }
    c[s].score = sum; c[s].doc = a->doc[s];
  }
  qsort(c, a->n, sizeof(Cand), cmp_cand);
  *pn = a->n;
  return c;
}

static int q_warp(Query *q) {
  LtIndex *ix = q->ix;
  uint8_t *pool = NULL;
  uint8_t **segs = (uint8_t **)malloc(sizeof(uint8_t *) * (q->nP ? q->nP : 1));
  if (!segs) return SQLITE_NOMEM;
  int rc = fetch_segments(q, 1, &pool, segs);
  double t = now_ms();
  float v[512];
  const int nq = q->nq, rb = ix->codec.rbytes;
  int *toks = (int *)malloc(sizeof(int) * nq);
  if (!toks) rc = SQLITE_NOMEM;
  for (int j = 0; j < q->nP && rc == SQLITE_OK; j++) {
    uint32_t c = q->P[j], cnt = ix->post_cnt[c];
    int nt = 0;
    for (int i = 0; i < nq; i++) if (q->o.cross || q->Pmask[(size_t)j * nq + i]) toks[nt++] = i;
    const uint8_t *ids = segs[j], *res = segs[j] + lt_bits_bytes(cnt, ix->idbits);
    const uint8_t *mask = q->Pmask + (size_t)j * nq;
    const float *cv = cent(ix, c);
    q->st.list_entries += cnt;
    for (uint32_t e = 0; e < cnt; e++) {
      uint32_t d = lt_get_bits(ids, (uint64_t)e * ix->idbits, ix->idbits);
      int s = acc_get(&q->acc, d);
      if (s < 0) { rc = SQLITE_NOMEM; break; }
      lc_decode_with(&ix->codec, cv, res + (size_t)e * rb, v);
      float *best = q->acc.best + (size_t)s * nq;
      uint8_t *own = q->acc.own + (size_t)s * nq;
      for (int k = 0; k < nt; k++) {
        int i = toks[k];
        float sc = lt_dot(v, q->q + (size_t)i * ix->dim, ix->dim);
        if (sc > best[i]) best[i] = sc;
        own[i] |= mask[i];
      }
    }
    q->st.tokens_decoded += cnt;
  }
  free(toks); free(segs); free(pool);
  if (rc) return rc;
  int n;
  Cand *c = acc_rank(q, &n);
  if (!c) return SQLITE_NOMEM;
  q->st.candidates = n;
  q->st.ms_cpu += now_ms() - t;
  if (q->o.rerank > 0 && ix->root_docs && (ix->layout & LAY_PLAID)) {
    int m = n < q->o.rerank ? n : q->o.rerank;
    LpRow *none = NULL;
    rc = rescore_docs(q, c, m, &none);
    n = m;
  }
  q->res = c; q->nres = n;
  return rc;
}

static int q_plaid(Query *q) {
  LtIndex *ix = q->ix;
  uint8_t *pool = NULL;
  uint8_t **segs = (uint8_t **)malloc(sizeof(uint8_t *) * (q->nP ? q->nP : 1));
  if (!segs) return SQLITE_NOMEM;
  int rc = fetch_segments(q, 0, &pool, segs);
  double t = now_ms();
  const int nq = q->nq;
  for (int j = 0; j < q->nP && rc == SQLITE_OK; j++) {
    uint32_t c = q->P[j], cnt = ix->ivf_cnt[c];
    const float *sc = srow(q, c);
    const uint8_t *mask = q->Pmask + (size_t)j * nq;
    q->st.list_entries += cnt;
    for (uint32_t e = 0; e < cnt; e++) {
      uint32_t d = lt_get_bits(segs[j], (uint64_t)e * ix->idbits, ix->idbits);
      int s = acc_get(&q->acc, d);
      if (s < 0) { rc = SQLITE_NOMEM; break; }
      float *best = q->acc.best + (size_t)s * nq;
      uint8_t *own = q->acc.own + (size_t)s * nq;
      for (int i = 0; i < nq; i++) { if (sc[i] > best[i]) best[i] = sc[i]; own[i] |= mask[i]; }
    }
  }
  free(segs); free(pool);
  if (rc) return rc;
  int n;
  Cand *c = acc_rank(q, &n);        /* IVF-only approximate score */
  if (!c) return SQLITE_NOMEM;
  q->st.candidates = n;
  q->st.ms_cpu += now_ms() - t;
  int nd = q->o.ndocs > 0 ? q->o.ndocs : 256;
  if (q->o.approx == APPROX_CODES) {
    /* faithful PLAID: codes of every candidate, centroid-interaction score */
    qsort(c, n, sizeof(Cand), cmp_cand_doc);
    LpRow *rows = (LpRow *)calloc(n ? n : 1, sizeof(LpRow));
    if (!rows) { free(c); return SQLITE_NOMEM; }
    for (int i = 0; i < n; i++) rows[i].rowid = c[i].doc;
    char tb[300]; snprintf(tb, sizeof tb, "%s_docs", q->vt->zName);
    rc = lpg_multiget(&q->vt->rd, tb, ix->root_docs, rows, n);
    q->st.docs_fetched += n;
    float *best = (float *)malloc(sizeof(float) * nq);
    t = now_ms();
    for (int i = 0; i < n && rc == SQLITE_OK && best; i++) {
      c[i].score = rows[i].data ? doc_approx(q, rows[i].data, rows[i].len, best) : -FLT_MAX;
      q->st.doc_bytes += rows[i].len;
    }
    free(best);
    /* keep the best nd; their rows are already here */
    int *ord = (int *)malloc(sizeof(int) * (n ? n : 1));
    Cand *cc = (Cand *)malloc(sizeof(Cand) * (n ? n : 1));
    if (!ord || !cc) rc = SQLITE_NOMEM;
    if (rc == SQLITE_OK) {
      memcpy(cc, c, sizeof(Cand) * n);
      for (int i = 0; i < n; i++) { cc[i].doc = (uint32_t)i; }   /* index into rows */
      qsort(cc, n, sizeof(Cand), cmp_cand);
      int m = n < nd ? n : nd;
      float tmp[512], *b2 = (float *)malloc(sizeof(float) * nq);
      for (int i = 0; i < m && b2; i++) {
        int ri = (int)cc[i].doc;
        c[i].doc = (uint32_t)rows[ri].rowid;
        c[i].score = rows[ri].data ? doc_maxsim(q, rows[ri].data, rows[ri].len, tmp, b2) : -FLT_MAX;
      }
      free(b2);
      qsort(c, m, sizeof(Cand), cmp_cand);
      n = m;
    }
    q->st.ms_cpu += now_ms() - t;
    free(ord); free(cc);
    for (int i = 0; i < (int)q->st.candidates; i++) free(rows[i].data);
    free(rows);
  } else {
    int m = n < nd ? n : nd;
    LpRow *none = NULL;
    rc = rescore_docs(q, c, m, &none);
    n = m;
  }
  q->res = c; q->nres = n;
  return rc;
}

/* Brute force: exact decompressed MaxSim over every document. */
typedef struct { Query *q; float *tmp, *best; Cand *c; int n, cap; } ExCtx;
static int ex_doc_cb(void *ctx, int64_t rowid, uint32_t pg, const uint8_t *d, int len) {
  (void)pg;
  ExCtx *e = (ExCtx *)ctx;
  if (e->n == e->cap) {
    e->cap = e->cap ? e->cap * 2 : 1024;
    Cand *t = (Cand *)realloc(e->c, sizeof(Cand) * e->cap);
    if (!t) return SQLITE_NOMEM;
    e->c = t;
  }
  e->c[e->n].doc = (uint32_t)rowid;
  e->c[e->n].score = doc_maxsim(e->q, d, len, e->tmp, e->best);
  e->q->st.doc_bytes += len;
  e->n++;
  return SQLITE_OK;
}
typedef struct { LtBuf b; } StreamCtx;
static int stream_cb(void *ctx, int64_t rowid, uint32_t pg, const uint8_t *d, int len) {
  (void)rowid; (void)pg;
  lt_buf_add(&((StreamCtx *)ctx)->b, d, len);
  return ((StreamCtx *)ctx)->b.oom ? SQLITE_NOMEM : SQLITE_OK;
}

static int q_exact(Query *q) {
  LtIndex *ix = q->ix;
  int rc;
  if ((ix->layout & LAY_PLAID) && ix->root_docs) {
    float tmp[512];
    ExCtx e; memset(&e, 0, sizeof e);
    e.q = q; e.tmp = tmp; e.best = (float *)malloc(sizeof(float) * q->nq);
    if (!e.best) return SQLITE_NOMEM;
    if (reader(q->vt) > 0) rc = lpg_scan(&q->vt->rd, ix->root_docs, ex_doc_cb, &e);
    else rc = SQLITE_ERROR;
    if (rc != SQLITE_OK) {                /* SQL scan */
      e.n = 0; rc = SQLITE_OK;
      sqlite3_stmt *st = NULL;
      rc = prep_fmt(q->vt->db, &st, "SELECT id, data FROM \"%w\".\"%w_docs\"", q->vt->zDb, q->vt->zName);
      while (rc == SQLITE_OK && sqlite3_step(st) == SQLITE_ROW)
        rc = ex_doc_cb(&e, sqlite3_column_int64(st, 0), 0, (const uint8_t *)sqlite3_column_blob(st, 1), sqlite3_column_bytes(st, 1));
      sqlite3_finalize(st);
    }
    free(e.best);
    if (rc) { free(e.c); return rc; }
    qsort(e.c, e.n, sizeof(Cand), cmp_cand);
    q->res = e.c; q->nres = e.n;
    q->st.candidates = e.n;
    return SQLITE_OK;
  }
  /* warp only: every posting list, every token against every query token */
  StreamCtx sc; memset(&sc, 0, sizeof sc);
  rc = reader(q->vt) > 0 ? lpg_scan(&q->vt->rd, ix->root_post, stream_cb, &sc) : SQLITE_ERROR;
  if (rc) { lt_buf_free(&sc.b); return rc; }
  lt_buf_reserve(&sc.b, LT_BITPAD);
  float v[512];
  for (int c = 0; c < ix->K && rc == SQLITE_OK; c++) {
    uint32_t cnt = ix->post_cnt[c];
    const uint8_t *ids = sc.b.p + ix->post_off[c], *res = ids + lt_bits_bytes(cnt, ix->idbits);
    for (uint32_t e = 0; e < cnt; e++) {
      int s = acc_get(&q->acc, lt_get_bits(ids, (uint64_t)e * ix->idbits, ix->idbits));
      if (s < 0) { rc = SQLITE_NOMEM; break; }
      lc_decode_with(&ix->codec, cent(ix, (uint32_t)c), res + (size_t)e * ix->codec.rbytes, v);
      float *best = q->acc.best + (size_t)s * q->nq;
      memset(q->acc.own + (size_t)s * q->nq, 1, q->nq);
      for (int i = 0; i < q->nq; i++) { float x = lt_dot(v, q->q + (size_t)i * ix->dim, ix->dim); if (x > best[i]) best[i] = x; }
    }
  }
  lt_buf_free(&sc.b);
  if (rc) return rc;
  for (int i = 0; i < q->nq; i++) q->impute[i] = 0;
  int n;
  q->res = acc_rank(q, &n);
  if (!q->res) return SQLITE_NOMEM;
  q->nres = n; q->st.candidates = n;
  return SQLITE_OK;
}

/* ------------------------------------------------------------- vtab glue */

typedef struct LtCursor {
  sqlite3_vtab_cursor base;
  int n, i;
  int64_t *rowid;
  float *score;
  char *stats;
} LtCursor;


static int vt_init(sqlite3 *db, void *pAux, int argc, const char *const *argv,
                   sqlite3_vtab **ppVtab, char **pzErr, int create) {
  (void)pAux;
  LtVtab *vt = (LtVtab *)sqlite3_malloc(sizeof(LtVtab));
  if (!vt) return SQLITE_NOMEM;
  memset(vt, 0, sizeof *vt);
  vt->db = db;
  vt->zDb = sqlite3_mprintf("%s", argv[1]);
  vt->zName = sqlite3_mprintf("%s", argv[2]);
  int rc = cfg_parse(&vt->cfg, argc, argv, pzErr);
  if (rc == SQLITE_OK) {
    /* The last hidden column is named after the table, for commands. */
    char *schema = sqlite3_mprintf("CREATE TABLE x(vectors, score, k HIDDEN, nprobe HIDDEN, opts HIDDEN, stats HIDDEN, \"%w\" HIDDEN)", argv[2]);
    rc = sqlite3_declare_vtab(db, schema);
    sqlite3_free(schema);
  }
  if (rc == SQLITE_OK && create) {
    static const char *const sfx[] = { "meta", "docs", "ivf", "post", "rowids", "buffer", "cells" };
    for (int i = 0; i < 7 && rc == SQLITE_OK; i++)
      rc = exec_fmt(db, pzErr, "CREATE TABLE IF NOT EXISTS \"%w\".\"%w_%s\"(id INTEGER PRIMARY KEY, data BLOB)", argv[1], argv[2], sfx[i]);
  }
  if (rc != SQLITE_OK) {
    sqlite3_free(vt->zDb); sqlite3_free(vt->zName); sqlite3_free(vt);
    return rc;
  }
  sqlite3_vtab_config(db, SQLITE_VTAB_INNOCUOUS);
  *ppVtab = &vt->base;
  return SQLITE_OK;
}

static int xCreate(sqlite3 *db, void *a, int argc, const char *const *argv, sqlite3_vtab **pp, char **e) {
  return vt_init(db, a, argc, argv, pp, e, 1);
}
static int xConnect(sqlite3 *db, void *a, int argc, const char *const *argv, sqlite3_vtab **pp, char **e) {
  return vt_init(db, a, argc, argv, pp, e, 0);
}

static int xDisconnect(sqlite3_vtab *p) {
  LtVtab *vt = (LtVtab *)p;
  ix_free(&vt->ix);
  if (vt->rd_state > 0) lpg_close(&vt->rd);
  sqlite3_free(vt->zDb); sqlite3_free(vt->zName);
  sqlite3_free(vt);
  return SQLITE_OK;
}

static int xDestroy(sqlite3_vtab *p) {
  LtVtab *vt = (LtVtab *)p;
  static const char *const sfx[] = { "meta", "docs", "ivf", "post", "rowids", "buffer", "cells" };
  for (int i = 0; i < 7; i++) exec_fmt(vt->db, NULL, "DROP TABLE IF EXISTS \"%w\".\"%w_%s\"", vt->zDb, vt->zName, sfx[i]);
  return xDisconnect(p);
}

static int xBestIndex(sqlite3_vtab *tab, sqlite3_index_info *info) {
  (void)tab;
  int flags = 0, argi[5] = { -1, -1, -1, -1, -1 };
  for (int i = 0; i < info->nConstraint; i++) {
    const struct sqlite3_index_constraint *c = &info->aConstraint[i];
    if (!c->usable) continue;
    if (c->op == SQLITE_INDEX_CONSTRAINT_MATCH && (c->iColumn == COL_VECTORS || c->iColumn == COL_CMD)) { flags |= F_MATCH; argi[0] = i; }
    else if (c->op == SQLITE_INDEX_CONSTRAINT_EQ && c->iColumn == COL_K) { flags |= F_K; argi[1] = i; }
    else if (c->op == SQLITE_INDEX_CONSTRAINT_EQ && c->iColumn == COL_NPROBE) { flags |= F_NPROBE; argi[2] = i; }
    else if (c->op == SQLITE_INDEX_CONSTRAINT_EQ && c->iColumn == COL_OPTS) { flags |= F_OPTS; argi[3] = i; }
    else if (c->op == SQLITE_INDEX_CONSTRAINT_LIMIT) { flags |= F_LIMIT; argi[4] = i; }
  }
  if (!(flags & F_MATCH)) { info->estimatedCost = 1e12; info->idxNum = 0; return SQLITE_OK; }
  int n = 0;
  for (int j = 0; j < 5; j++) if (argi[j] >= 0) {
    info->aConstraintUsage[argi[j]].argvIndex = ++n;
    info->aConstraintUsage[argi[j]].omit = 1;
  }
  info->idxNum = flags;
  info->estimatedCost = 10;
  info->estimatedRows = 10;
  if (info->nOrderBy == 1 && info->aOrderBy[0].iColumn == COL_SCORE && info->aOrderBy[0].desc) info->orderByConsumed = 1;
  return SQLITE_OK;
}

static int xOpen(sqlite3_vtab *p, sqlite3_vtab_cursor **pp) {
  (void)p;
  LtCursor *c = (LtCursor *)sqlite3_malloc(sizeof(LtCursor));
  if (!c) return SQLITE_NOMEM;
  memset(c, 0, sizeof *c);
  *pp = &c->base;
  return SQLITE_OK;
}

static void cur_clear(LtCursor *c) {
  free(c->rowid); free(c->score); sqlite3_free(c->stats);
  c->rowid = NULL; c->score = NULL; c->stats = NULL; c->n = c->i = 0;
}

static int xClose(sqlite3_vtab_cursor *p) {
  cur_clear((LtCursor *)p);
  sqlite3_free(p);
  return SQLITE_OK;
}

static char *stats_json(Query *q, int total_rounds) {
  LpReader *rd = &q->vt->rd;
  const char *lay = q->o.exact ? "exact" : q->o.layout == LAY_WARP ? "warp" : "plaid";
  sqlite3_str *s = sqlite3_str_new(NULL);
  sqlite3_str_appendf(s, "{\"layout\":\"%s\",\"nq\":%d,\"nprobe\":%d,\"probed\":%d,\"candidates\":%lld,"
      "\"list_entries\":%lld,\"list_bytes\":%lld,\"docs_fetched\":%lld,\"doc_bytes\":%lld,\"tokens_decoded\":%lld,"
      "\"rounds\":%d,\"static_rounds\":%d,\"pages\":%d,\"page_size\":%d,\"bytes\":%lld,\"payload_bytes\":%lld,"
      "\"cache_hits\":%lld,\"hint_miss\":%lld,\"fallbacks\":%lld,\"static_bytes\":%lld,\"ms\":%.3f,\"cpu_ms\":%.3f,"
      "\"cells\":%d,\"cell_rounds\":%d,\"cell_bytes\":%lld,\"cells_loaded\":%d",
      lay, q->nq, q->o.nprobe, q->st.probed, (long long)q->st.candidates, (long long)q->st.list_entries,
      (long long)q->st.list_bytes, (long long)q->st.docs_fetched, (long long)q->st.doc_bytes, (long long)q->st.tokens_decoded,
      total_rounds, q->st.static_rounds, rd->tn, rd->pgsz, (long long)rd->tn * rd->pgsz, (long long)rd->payload_bytes,
      (long long)rd->cache_hits, (long long)rd->hint_miss, (long long)rd->fallbacks, (long long)q->ix->static_bytes,
      q->st.ms_total, q->st.ms_cpu, q->st.cells, q->st.cell_rounds, (long long)q->st.cell_bytes, q->ix->cells_loaded);
  if (q->o.trace) {
    sqlite3_str_appendall(s, ",\"trace\":[");
    for (int r = 1; r <= rd->round; r++) {
      sqlite3_str_appendf(s, "%s[", r > 1 ? "," : "");
      int first = 1;
      for (int i = 0; i < rd->tn; i++) if (rd->trd[i] == r) { sqlite3_str_appendf(s, "%s%u", first ? "" : ",", rd->tpg[i]); first = 0; }
      sqlite3_str_appendall(s, "]");
    }
    sqlite3_str_appendall(s, "]");
  }
  sqlite3_str_appendall(s, "}");
  return sqlite3_str_finish(s);
}

static int xFilter(sqlite3_vtab_cursor *pc, int idxNum, const char *idxStr, int argc, sqlite3_value **argv) {
  (void)idxStr; (void)argc;
  LtCursor *cur = (LtCursor *)pc;
  LtVtab *vt = (LtVtab *)pc->pVtab;
  cur_clear(cur);
  if (!(idxNum & F_MATCH)) { set_err(vt, "late_plaid: a MATCH constraint with the query vectors is required"); return SQLITE_ERROR; }
  double t0 = now_ms();
  Query q; memset(&q, 0, sizeof q);
  q.vt = vt; q.ix = &vt->ix;
  q.o.k = 10; q.o.nprobe = 4; q.o.approx = APPROX_CODES; q.o.ndocs = 256; q.o.impute = 1; q.o.cross = 1; q.o.cprobe = 4;
  int ai = 0;
  sqlite3_value *qv = argv[ai++];
  if (idxNum & F_K) q.o.k = sqlite3_value_int(argv[ai++]);
  if (idxNum & F_NPROBE) q.o.nprobe = sqlite3_value_int(argv[ai++]);
  if (idxNum & F_OPTS) opts_parse(&q.o, (const char *)sqlite3_value_text(argv[ai++]));
  if (idxNum & F_LIMIT) { int l = sqlite3_value_int(argv[ai++]); if (l > 0 && (!(idxNum & F_K) || l < q.o.k)) q.o.k = l; }
  if (reader(vt) > 0) lpg_trace_reset(&vt->rd);
  int rc = SQLITE_OK;
  if (!vt->ix.loaded) {
    rc = ix_load(vt);
    q.st.static_rounds = vt->ix.load_rounds;
  }
  if (rc) return rc;
  LtIndex *ix = &vt->ix;
  if (q.o.layout == 0) q.o.layout = (ix->layout & LAY_WARP) ? LAY_WARP : LAY_PLAID;
  if (!(q.o.layout & ix->layout)) { set_err(vt, "late_plaid: layout not built"); return SQLITE_ERROR; }
  int nbytes = sqlite3_value_bytes(qv);
  const float *qf = (const float *)sqlite3_value_blob(qv);
  if (sqlite3_value_type(qv) != SQLITE_BLOB || nbytes == 0 || nbytes % (4 * ix->dim)) {
    set_err(vt, "late_plaid: query must be a float32 blob of [n x %d]", ix->dim);
    return SQLITE_ERROR;
  }
  q.nq = nbytes / (4 * ix->dim);
  if (ix->G && (q.o.exact || q.o.layout == LAY_PLAID || q.o.rerank > 0)) {
    /* decompressing whole documents (or scoring their codes) needs every
    ** centroid: two-level mode then degenerates to the flat table */
    int r0 = vt->rd.round;
    rc = load_all_cells(vt);
    q.st.cell_rounds = vt->rd.round - r0;
    if (rc) return rc;
  }
  float *qcopy = (float *)malloc(nbytes);          /* aligned copy */
  if (!qcopy) return SQLITE_NOMEM;
  memcpy(qcopy, qf, nbytes);
  q.q = qcopy;
  q.st.nq = q.nq;
  q.acc.nq = q.nq;
  rc = q_probe(&q);
  if (rc == SQLITE_OK) {
    if (q.o.exact) rc = q_exact(&q);
    else if (q.o.layout == LAY_WARP) rc = q_warp(&q);
    else rc = q_plaid(&q);
  }
  if (rc == SQLITE_OK) {
    int n = q.nres < q.o.k ? q.nres : q.o.k;
    cur->rowid = (int64_t *)malloc(sizeof(int64_t) * (n ? n : 1));
    cur->score = (float *)malloc(sizeof(float) * (n ? n : 1));
    if (!cur->rowid || !cur->score) rc = SQLITE_NOMEM;
    else {
      cur->n = n;
      for (int i = 0; i < n; i++) { cur->rowid[i] = q.res[i].doc; cur->score[i] = q.res[i].score; }
      if (!ix->rowid_identity && n > 0) {
        LpRow *rows = (LpRow *)calloc(n, sizeof(LpRow));
        int *ord = (int *)malloc(sizeof(int) * n);
        if (!rows || !ord) rc = SQLITE_NOMEM;
        else {
          for (int i = 0; i < n; i++) ord[i] = i;
          for (int i = 1; i < n; i++) { int v = ord[i], j = i - 1; while (j >= 0 && cur->rowid[ord[j]] > cur->rowid[v]) { ord[j + 1] = ord[j]; j--; } ord[j + 1] = v; }
          for (int i = 0; i < n; i++) rows[i].rowid = cur->rowid[ord[i]];
          char tb[300]; snprintf(tb, sizeof tb, "%s_rowids", vt->zName);
          rc = lpg_multiget(&vt->rd, tb, ix->root_rowids, rows, n);
          for (int i = 0; i < n; i++) if (rows[i].data && rows[i].len == 8) cur->rowid[ord[i]] = (int64_t)lt_get_u64(rows[i].data);
          for (int i = 0; i < n; i++) free(rows[i].data);
        }
        free(rows); free(ord);
      }
    }
  }
  q.st.ms_total = now_ms() - t0;
  if (rc == SQLITE_OK) cur->stats = stats_json(&q, vt->rd.round);
  free(qcopy);
  q_free(&q);
  if (rc && !vt->base.zErrMsg) set_err(vt, "late_plaid: query failed (%d)", rc);
  return rc;
}

static int xNext(sqlite3_vtab_cursor *p) { ((LtCursor *)p)->i++; return SQLITE_OK; }
static int xEof(sqlite3_vtab_cursor *p) { LtCursor *c = (LtCursor *)p; return c->i >= c->n; }
static int xRowid(sqlite3_vtab_cursor *p, sqlite3_int64 *r) { LtCursor *c = (LtCursor *)p; *r = c->rowid[c->i]; return SQLITE_OK; }
static int xColumn(sqlite3_vtab_cursor *p, sqlite3_context *ctx, int col) {
  LtCursor *c = (LtCursor *)p;
  if (col == COL_SCORE) sqlite3_result_double(ctx, c->score[c->i]);
  else if (col == COL_STATS) { if (c->stats) sqlite3_result_text(ctx, c->stats, -1, SQLITE_TRANSIENT); }
  else sqlite3_result_null(ctx);
  return SQLITE_OK;
}

static int cmd(LtVtab *vt, const char *z) {
  while (*z == ' ') z++;
  if (!strcmp(z, "build")) {
    DocSrc s;
    int rc = src_open_buffer(&s, vt);
    if (rc == SQLITE_OK) rc = do_build(vt, &s);
    src_close(&s);
    if (rc == SQLITE_OK) rc = exec_fmt(vt->db, NULL, "DELETE FROM \"%w\".\"%w_buffer\"", vt->zDb, vt->zName);
    return rc;
  }
  if (!strncmp(z, "build_npy ", 10)) {
    char a[1024], b[1024];
    if (sscanf(z + 10, "%1023s %1023s", a, b) != 2) { set_err(vt, "late_plaid: build_npy VECTORS.npy OFFSETS.npy"); return SQLITE_ERROR; }
    DocSrc s; char *err = NULL;
    int rc = src_open_npy(&s, a, b, vt->cfg.dim, &err);
    if (rc == SQLITE_OK) rc = do_build(vt, &s);
    else { set_err(vt, "late_plaid: %s", err ? err : "cannot open input"); sqlite3_free(err); }
    src_close(&s);
    return rc;
  }
  if (!strcmp(z, "finalize")) return do_finalize(vt);
  if (!strcmp(z, "drop_cache")) { if (vt->rd_state > 0) lpg_drop_cache(&vt->rd); return SQLITE_OK; }
  if (!strcmp(z, "reload")) { vt->ix.loaded = 0; if (vt->rd_state > 0) lpg_drop_cache(&vt->rd); return SQLITE_OK; }
  set_err(vt, "late_plaid: unknown command '%s'", z);
  return SQLITE_ERROR;
}

static int xUpdate(sqlite3_vtab *p, int argc, sqlite3_value **argv, sqlite3_int64 *pRowid) {
  LtVtab *vt = (LtVtab *)p;
  if (argc == 1) {                       /* DELETE: only from the pre-build buffer */
    return exec_fmt(vt->db, NULL, "DELETE FROM \"%w\".\"%w_buffer\" WHERE id=%lld", vt->zDb, vt->zName,
                    (long long)sqlite3_value_int64(argv[0]));
  }
  if (sqlite3_value_type(argv[0]) != SQLITE_NULL) { set_err(vt, "late_plaid: UPDATE is not supported"); return SQLITE_ERROR; }
  sqlite3_value *vcmd = argv[2 + COL_CMD];
  if (sqlite3_value_type(vcmd) != SQLITE_NULL) return cmd(vt, (const char *)sqlite3_value_text(vcmd));
  sqlite3_value *vv = argv[2 + COL_VECTORS];
  int n = sqlite3_value_bytes(vv), esz = vt->cfg.input_f16 ? 2 : 4;
  if (sqlite3_value_type(vv) != SQLITE_BLOB || n == 0 || n % (esz * vt->cfg.dim)) {
    set_err(vt, "late_plaid: vectors must be a %s blob of [n_tokens x %d]", esz == 2 ? "float16" : "float32", vt->cfg.dim);
    return SQLITE_ERROR;
  }
  int nv = n / esz;
  uint16_t *h = (uint16_t *)malloc(2 * (size_t)nv);
  if (!h) return SQLITE_NOMEM;
  const uint8_t *src = (const uint8_t *)sqlite3_value_blob(vv);
  for (int i = 0; i < nv; i++) {
    if (esz == 2) memcpy(&h[i], src + 2 * i, 2);
    else { float f; memcpy(&f, src + 4 * i, 4); h[i] = lt_f2h(f); }
  }
  sqlite3_stmt *st = NULL;
  int rc = prep_fmt(vt->db, &st, "INSERT INTO \"%w\".\"%w_buffer\"(id, data) VALUES (?, ?)", vt->zDb, vt->zName);
  if (rc == SQLITE_OK) {
    if (sqlite3_value_type(argv[1]) != SQLITE_NULL) sqlite3_bind_int64(st, 1, sqlite3_value_int64(argv[1]));
    sqlite3_bind_blob(st, 2, h, 2 * nv, SQLITE_STATIC);
    rc = sqlite3_step(st) == SQLITE_DONE ? SQLITE_OK : SQLITE_ERROR;
    if (rc) set_err(vt, "%s", sqlite3_errmsg(vt->db));
    *pRowid = sqlite3_last_insert_rowid(vt->db);
  }
  sqlite3_finalize(st);
  free(h);
  return rc;
}

static int xShadowName(const char *z) {
  static const char *const sfx[] = { "meta", "docs", "ivf", "post", "rowids", "buffer", "cells" };
  for (int i = 0; i < 7; i++) if (!sqlite3_stricmp(z, sfx[i])) return 1;
  return 0;
}

static int xFindFunction(sqlite3_vtab *p, int nArg, const char *zName,
                         void (**pxFunc)(sqlite3_context *, int, sqlite3_value **), void **ppArg) {
  (void)p; (void)nArg; (void)zName; (void)pxFunc; (void)ppArg;
  return 0;
}

static sqlite3_module late_module = {
  3, xCreate, xConnect, xBestIndex, xDisconnect, xDestroy, xOpen, xClose, xFilter, xNext, xEof,
  xColumn, xRowid, xUpdate, NULL, NULL, NULL, NULL, xFindFunction, NULL, NULL, NULL, NULL,
  xShadowName, NULL
};

#ifdef _WIN32
__declspec(dllexport)
#endif
int sqlite3_late_init(sqlite3 *db, char **pzErrMsg, const sqlite3_api_routines *pApi) {
  (void)pzErrMsg;
  SQLITE_EXTENSION_INIT2(pApi);
  return sqlite3_create_module_v2(db, "late_plaid", &late_module, NULL, NULL);
}
