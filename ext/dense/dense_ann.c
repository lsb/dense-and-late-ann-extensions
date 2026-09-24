/*
** dense_ann.c — a SQLite virtual table for approximate nearest-neighbour
** search over dense embeddings, designed for read-only databases fetched
** lazily over HTTP range requests.
**
**   CREATE VIRTUAL TABLE v USING dense_ann(dim=384, pq_m=64, M=16,
**                                           ef_construction=200, store_vectors=f16);
**   INSERT INTO v(rowid, embedding) VALUES (?, ?);   -- float32 blob or JSON text
**   INSERT INTO v(v) VALUES ('build');                -- train PQ, build graph
**   INSERT INTO v(v) VALUES ('finalize');             -- store page hints (after VACUUM)
**   SELECT rowid, distance FROM v WHERE embedding MATCH ?1 AND k = 10
**          [AND ef = 64] [AND beam = 4] [AND rerank = 1] [AND exact = 1|2];
**
** Index: an HNSW graph built in memory from the full-precision vectors; only
** layer 0 is persisted, one row per node, and each row carries the PQ codes
** of the node's neighbours ("co-located" layout) so expanding a node is one
** page read. A random sample of high-level HNSW nodes forms the entry set,
** which is loaded once (with the PQ codebook) and scanned in memory to seed
** a beam search on layer 0. See NOTES.md for the design and measurements.
**
** Storage (shadow tables, all "id INTEGER PRIMARY KEY, data BLOB"):
**   <name>_config  (key, value)     parameters and state
**   <name>_blobs   chunked blobs    PQ codebook (float16) and entry set
**   <name>_nodes   one row/node     header, own PQ code, [vector], neighbours
**   <name>_codes   one row/node     PQ codes (layout=separate only)
**   <name>_vectors one row/node     stored vectors (vectors=table only)
**   <name>_rowids  (rowid, node)    user rowid -> node id
**   <name>_buffer  (rowid, vec)     float32 vectors inserted before 'build'
*/
#include "sqlite3ext.h"
SQLITE_EXTENSION_INIT1

#include <float.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "dense_common.h"
#include "hnsw.h"
#include "ivf.h"
#include "pq.h"
#include "rawpage.h"

/* ------------------------------------------------------------ constants */

enum { VT_NONE = 0, VT_INT8 = 1, VT_F16 = 2, VT_F32 = 3 };
enum { LAYOUT_COLOCATED = 0, LAYOUT_SEPARATE = 1, LAYOUT_IVF = 2 };
enum { CENT_F16 = 0, CENT_INT8 = 1, CENT_PQ = 2, CENT_AUTO = 3 };
enum { CB_F16 = 0, CB_INT8 = 1 };
/* Storage format of the index head. 1 (or absent): float16 PQ codebooks.
** 2: the codebook storage type is in config key "codebook" (CB_*). */
#define DENSE_FORMAT_MAX 2
enum { REORDER_AUTO = 0, REORDER_NONE = 1, REORDER_PACK = 2 };
enum { T_NODES = 0, T_CODES = 1, T_VECS = 2, T_BLOBS = 3, T_LISTS = 4, T_COUNT = 5 };
static const char *const TABLE_SUFFIX[T_COUNT] = { "nodes", "codes", "vectors", "blobs", "lists" };

/* Blob chunks are kept below the 4 KiB-page local payload limit (4061 bytes)
** so they never need overflow pages. */
#define CHUNK_BYTES 3800
#define CHUNK_ID_CODEBOOK 1
#define CHUNK_ID_ENTRIES  1000000
#define CHUNK_ID_ROTATION 2000000
#define CHUNK_ID_IVF_HEAD 3000000
#define CHUNK_ID_IVF_CPQ  4000000

/* Node row header: u8 flags, u8 reserved, u16 degree, u32 vector-row page
** hint, i64 user rowid. */
#define NODE_HDR 16
#define NODE_FLAG_DELETED 1

/* Virtual table columns. */
enum {
  COL_EMBEDDING = 0, COL_DISTANCE, COL_K, COL_EF, COL_BEAM, COL_RERANK, COL_EXACT,
  COL_TRACE, COL_NPROBE, COL_RK, COL_STATS, COL_CMD, NCOL
};

/* xBestIndex flags, also the order in which xFilter consumes argv. */
#define F_MATCH  0x001
#define F_K      0x002
#define F_EF     0x004
#define F_BEAM   0x008
#define F_RERANK 0x010
#define F_EXACT  0x020
#define F_TRACE  0x040
#define F_LIMIT  0x080
#define F_ROWID  0x100
#define F_NPROBE 0x200
#define F_RK     0x400

#ifndef SQLITE_INDEX_CONSTRAINT_LIMIT
#define SQLITE_INDEX_CONSTRAINT_LIMIT 73
#endif

/* --------------------------------------------------------- configuration */

typedef struct DenseCfg {
  int dim, m, M, M0, efc;
  int metric;        /* METRIC_* */
  int vtype;         /* VT_*: stored full-precision-ish vectors for rerank */
  int vinline;       /* 1: vector inside node row; 0: separate table */
  int layout;        /* LAYOUT_* */
  int n_entry;       /* entry-set size */
  int64_t train_max; /* max PQ training sample */
  int iters;         /* k-means iterations */
  int opq;           /* OPQ iterations (0 = plain PQ) */
  int64_t seed;
  double alpha;      /* pruning slack for graph construction */
  int reorder;       /* REORDER_* */
  int page_hint;     /* page size assumed when packing nodes (0 = current) */
  int nthreads;
  int ef_default, beam_default, verbose;
  /* IVF layout */
  int nlist;         /* coarse lists (0 = 4 sqrt(n) at build) */
  int ivf_residual;  /* 1: PQ-encode residuals x - centroid (IVFADC); 0: raw vectors */
  int ivf_cent;      /* CENT_*: storage of the coarse centroids in the cached head */
  int cb_type;       /* CB_*: storage of the PQ codebooks in the cached head */
  int format;        /* head format (DENSE_FORMAT_MAX when written by this build) */
  int nprobe_default, rk_default;
} DenseCfg;

/* Byte layout of a node row, derived from the configuration. */
typedef struct RowLayout {
  int size;          /* total bytes (fixed for all rows) */
  int off_code;      /* own PQ code */
  int off_vec;       /* own vector if inline */
  int off_nbr;       /* neighbour entries: u32 id, u32 page [, u32 code page] */
  int nbr_stride;
  int off_nbrcode;   /* neighbour PQ codes (co-located layout only) */
  int vec_bytes;     /* bytes of one stored vector */
} RowLayout;

static int vec_bytes_for(int vtype, int dim) {
  switch (vtype) {
    case VT_INT8: return 4 + dim;       /* float32 scale + int8 values */
    case VT_F16: return 2 * dim;
    case VT_F32: return 4 * dim;
    default: return 0;
  }
}

static void layout_compute(const DenseCfg *c, RowLayout *rl) {
  rl->vec_bytes = vec_bytes_for(c->vtype, c->dim);
  rl->off_code = NODE_HDR;
  rl->off_vec = rl->off_code + c->m;
  rl->off_nbr = rl->off_vec + (c->vinline ? rl->vec_bytes : 0);
  rl->nbr_stride = c->layout == LAYOUT_COLOCATED ? 8 : 12;
  rl->off_nbrcode = rl->off_nbr + c->M0 * rl->nbr_stride;
  rl->size = rl->off_nbrcode + (c->layout == LAYOUT_COLOCATED ? c->M0 * c->m : 0);
}

/* ------------------------------------------------------------ vtab state */

typedef struct DenseVtab {
  sqlite3_vtab base;
  sqlite3 *db;
  char *zDb, *zName;
  DenseCfg cfg;
  RowLayout rl;

  /* persisted state */
  int built;
  int64_t n_nodes, n_deleted;
  uint32_t next_id;
  int hints_valid;
  int n_entry;       /* entries actually stored */

  /* cached index head: codebook and entry set (loaded on first use) */
  int loaded;
  PQ pq;
  uint8_t *entries;  /* n_entry * (8 + m): u32 id, u32 page, code */

  /* IVF head (loaded with the codebook) */
  int nlist;
  float *ivf_cent;   /* decoded centroids, nlist * dim */
  float *ivf_cn;     /* squared norms */
  uint8_t *ivf_head; /* stored head: centroids then directory */
  uint8_t *ivf_dir;  /* nlist * 12 B: u32 nchunks, u32 first page, u32 last page */
  PQ ivf_cpq;        /* PQ of the centroids (ivf_cent=pq only) */

  PageReader pr;
  int pr_state;      /* 0 = untried, 1 = ok, -1 = unavailable */
  int bad_format;    /* config written by a newer build */
  sqlite3_stmt *st_get[T_COUNT];
} DenseVtab;

typedef struct QStats {
  int rounds, pages, fallback, expanded, dist, entry_dist, rerank, rerank_rounds;
  int setup_rounds, setup_pages, raw, ann;
  int64_t bytes, setup_bytes;
  int scanned, lists;
  double ms;
  sqlite3_str *trace;   /* JSON list of per-round page lists, or NULL */
  int ntrace;
} QStats;

typedef struct Result { int64_t rowid; float d; } Result;

typedef struct DenseCursor {
  sqlite3_vtab_cursor base;
  int mode;             /* 0 = eof, 1 = search results, 2 = scan statement */
  Result *res; int nres, pos;
  sqlite3_stmt *scan;
  int scan_eof;
  int64_t k, ef, beam, rerank, exact, trace, nprobe, rk;
  char *stats;
} DenseCursor;

/* --------------------------------------------------------------- helpers */

static double now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6;
}

static void set_err(DenseVtab *vt, const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  sqlite3_free(vt->base.zErrMsg);
  vt->base.zErrMsg = sqlite3_vmprintf(fmt, ap);
  va_end(ap);
}

static int exec_fmt(sqlite3 *db, const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  char *sql = sqlite3_vmprintf(fmt, ap);
  va_end(ap);
  if (!sql) return SQLITE_NOMEM;
  int rc = sqlite3_exec(db, sql, 0, 0, 0);
  sqlite3_free(sql);
  return rc;
}

static int prep_fmt(sqlite3 *db, sqlite3_stmt **st, const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  char *sql = sqlite3_vmprintf(fmt, ap);
  va_end(ap);
  if (!sql) return SQLITE_NOMEM;
  int rc = sqlite3_prepare_v2(db, sql, -1, st, 0);
  sqlite3_free(sql);
  return rc;
}

/* Prepare a statement against one of this table's shadow tables. The format
** receives (schema, table name) as the first two %w arguments. */
#define PREP(vt, st, fmt) prep_fmt((vt)->db, (st), fmt, (vt)->zDb, (vt)->zName)

/* Store a vector in the configured format. */
static void vec_encode(const DenseCfg *c, const float *x, uint8_t *out) {
  int d = c->dim;
  if (c->vtype == VT_F32) {
    for (int i = 0; i < d; i++) { uint32_t u; memcpy(&u, &x[i], 4); dn_wr32(out + 4 * i, u); }
  } else if (c->vtype == VT_F16) {
    for (int i = 0; i < d; i++) dn_wr16(out + 2 * i, dn_f32_to_f16(x[i]));
  } else if (c->vtype == VT_INT8) {
    float mx = 0;
    for (int i = 0; i < d; i++) if (fabsf(x[i]) > mx) mx = fabsf(x[i]);
    float scale = mx > 0 ? mx / 127.0f : 1.0f;
    uint32_t u; memcpy(&u, &scale, 4); dn_wr32(out, u);
    for (int i = 0; i < d; i++) {
      long q = lrintf(x[i] / scale);
      out[4 + i] = (uint8_t)(int8_t)(q > 127 ? 127 : q < -127 ? -127 : q);
    }
  }
}

static void vec_decode(const DenseCfg *c, const uint8_t *in, float *x) {
  int d = c->dim;
  if (c->vtype == VT_F32) {
    for (int i = 0; i < d; i++) { uint32_t u = dn_rd32(in + 4 * i); memcpy(&x[i], &u, 4); }
  } else if (c->vtype == VT_F16) {
    for (int i = 0; i < d; i++) x[i] = dn_f16_to_f32(dn_rd16(in + 2 * i));
  } else if (c->vtype == VT_INT8) {
    float scale; uint32_t u = dn_rd32(in); memcpy(&scale, &u, 4);
    for (int i = 0; i < d; i++) x[i] = scale * (float)(int8_t)in[4 + i];
  }
}

/* Parse a query/insert vector: float32 blob of dim*4 bytes, or JSON-ish
** text "[x, y, ...]". Returns a malloc'd array or NULL (error in *pzErr). */
static float *parse_vector(sqlite3_value *v, int dim, char **pzErr) {
  float *x = (float *)malloc(sizeof(float) * dim);
  if (!x) return NULL;
  if (sqlite3_value_type(v) == SQLITE_BLOB) {
    int n = sqlite3_value_bytes(v);
    if (n != dim * 4) {
      *pzErr = sqlite3_mprintf("dense_ann: expected a %d-byte float32 blob, got %d bytes", dim * 4, n);
      free(x); return NULL;
    }
    memcpy(x, sqlite3_value_blob(v), n);   /* host is little-endian (x86, wasm) */
    return x;
  }
  if (sqlite3_value_type(v) == SQLITE_TEXT) {
    const char *s = (const char *)sqlite3_value_text(v);
    int i = 0;
    while (*s && *s != '[') s++;
    if (*s == '[') s++;
    while (*s && i < dim) {
      char *end;
      double val = strtod(s, &end);
      if (end == s) break;
      x[i++] = (float)val;
      s = end;
      while (*s == ' ' || *s == ',' || *s == '\n' || *s == '\t') s++;
    }
    if (i == dim) return x;
    *pzErr = sqlite3_mprintf("dense_ann: expected %d numbers in JSON vector", dim);
    free(x); return NULL;
  }
  *pzErr = sqlite3_mprintf("dense_ann: vector must be a float32 blob or JSON array");
  free(x);
  return NULL;
}

/* --------------------------------------------------------------- config */

/* Upsert one config value: kind 0 = integer, 1 = blob, 2 = real. */
static int cfg_set(DenseVtab *vt, const char *key, int kind, int64_t ival, double dval,
                   const void *blob, int nblob) {
  sqlite3_stmt *st = 0;
  int rc = PREP(vt, &st, "INSERT OR REPLACE INTO \"%w\".\"%w_config\"(key, value) VALUES (?1, ?2)");
  if (rc) return rc;
  sqlite3_bind_text(st, 1, key, -1, SQLITE_STATIC);
  if (kind == 0) sqlite3_bind_int64(st, 2, ival);
  else if (kind == 1) sqlite3_bind_blob(st, 2, blob, nblob, SQLITE_TRANSIENT);
  else sqlite3_bind_double(st, 2, dval);
  rc = sqlite3_step(st);
  sqlite3_finalize(st);
  return rc == SQLITE_DONE ? SQLITE_OK : rc;
}
#define CFG_INT(vt, k, v)     cfg_set((vt), (k), 0, (int64_t)(v), 0, 0, 0)
#define CFG_DBL(vt, k, v)     cfg_set((vt), (k), 2, 0, (double)(v), 0, 0)
#define CFG_BLOB(vt, k, p, n) cfg_set((vt), (k), 1, 0, 0, (p), (n))

/* Read a BLOB config value into a malloc'd buffer. */
static int cfg_get_blob(DenseVtab *vt, const char *key, uint8_t **out, int *n) {
  sqlite3_stmt *st = 0;
  *out = NULL; *n = 0;
  int rc = PREP(vt, &st, "SELECT value FROM \"%w\".\"%w_config\" WHERE key = ?1");
  if (rc) return rc;
  sqlite3_bind_text(st, 1, key, -1, SQLITE_STATIC);
  if (sqlite3_step(st) == SQLITE_ROW) {
    *n = sqlite3_column_bytes(st, 0);
    *out = (uint8_t *)malloc(*n ? *n : 1);
    if (*n) memcpy(*out, sqlite3_column_blob(st, 0), *n);
  }
  sqlite3_finalize(st);
  return SQLITE_OK;
}

static int save_params(DenseVtab *vt) {
  const DenseCfg *c = &vt->cfg;
  int rc = SQLITE_OK;
#define S(k, v) if (rc == SQLITE_OK) rc = CFG_INT(vt, k, v)
  S("dim", c->dim); S("pq_m", c->m); S("M", c->M); S("M0", c->M0);
  S("ef_construction", c->efc); S("metric", c->metric); S("store_vectors", c->vtype);
  S("vectors_inline", c->vinline); S("layout", c->layout); S("entry_points", c->n_entry);
  S("dnpq_train", c->train_max); S("kmeans_iters", c->iters); S("seed", c->seed); S("opq", c->opq);
  S("reorder", c->reorder); S("page_size_hint", c->page_hint); S("threads", c->nthreads);
  S("ef_search", c->ef_default); S("beam", c->beam_default); S("verbose", c->verbose);
  S("nlist", c->nlist); S("ivf_residual", c->ivf_residual); S("ivf_centroids", c->ivf_cent);
  S("nprobe", c->nprobe_default); S("rerank_k", c->rk_default);
  S("codebook", c->cb_type); S("format", c->format);
#undef S
  if (rc == SQLITE_OK) rc = CFG_DBL(vt, "alpha", c->alpha);
  return rc;
}

static int save_state(DenseVtab *vt) {
  int rc = SQLITE_OK;
#define S(k, v) if (rc == SQLITE_OK) rc = CFG_INT(vt, k, v)
  S("built", vt->built); S("n_nodes", vt->n_nodes); S("n_deleted", vt->n_deleted);
  S("next_id", vt->next_id); S("hints_valid", vt->hints_valid); S("n_entry", vt->n_entry);
  S("row_size", vt->rl.size);
#undef S
  return rc;
}

static int load_config(DenseVtab *vt) {
  sqlite3_stmt *st = 0;
  int rc = PREP(vt, &st, "SELECT key, value FROM \"%w\".\"%w_config\"");
  if (rc) return rc;
  DenseCfg *c = &vt->cfg;
  while (sqlite3_step(st) == SQLITE_ROW) {
    const char *k = (const char *)sqlite3_column_text(st, 0);
    int64_t v = sqlite3_column_int64(st, 1);
    if (!k) continue;
#define G(name, field) else if (strcmp(k, name) == 0) field = v
    if (0) {}
    G("dim", c->dim); G("pq_m", c->m); G("M", c->M); G("M0", c->M0);
    G("ef_construction", c->efc); G("metric", c->metric); G("store_vectors", c->vtype);
    G("vectors_inline", c->vinline); G("layout", c->layout); G("entry_points", c->n_entry);
    G("dnpq_train", c->train_max); G("kmeans_iters", c->iters); G("seed", c->seed); G("opq", c->opq);
    G("reorder", c->reorder); G("page_size_hint", c->page_hint); G("threads", c->nthreads);
    G("ef_search", c->ef_default); G("beam", c->beam_default); G("verbose", c->verbose);
    G("nlist", c->nlist); G("ivf_residual", c->ivf_residual); G("ivf_centroids", c->ivf_cent);
    G("nprobe", c->nprobe_default); G("rerank_k", c->rk_default); G("ivf_nlist", vt->nlist);
    G("codebook", c->cb_type); G("format", c->format);
    G("built", vt->built); G("n_nodes", vt->n_nodes); G("n_deleted", vt->n_deleted);
    G("next_id", vt->next_id); G("hints_valid", vt->hints_valid); G("n_entry", vt->n_entry);
    else if (strcmp(k, "alpha") == 0) c->alpha = sqlite3_column_double(st, 1);
#undef G
  }
  sqlite3_finalize(st);
  layout_compute(c, &vt->rl);
  if (c->format > DENSE_FORMAT_MAX || c->cb_type > CB_INT8) {
    vt->bad_format = c->format;
    return SQLITE_ERROR;
  }
  return SQLITE_OK;
}

/* Parse "key=value" module arguments into cfg. */
static int parse_args(DenseCfg *c, int argc, const char *const *argv, char **pzErr) {
  memset(c, 0, sizeof *c);
  c->M = 16; c->efc = 200; c->metric = METRIC_COSINE; c->vtype = VT_F16; c->vinline = 1;
  c->layout = LAYOUT_COLOCATED; c->n_entry = 1024; c->train_max = 65536; c->iters = 25;
  c->seed = 42; c->alpha = 1.0; c->reorder = REORDER_AUTO; c->nthreads = 4;
  c->ef_default = 64; c->beam_default = 16;
  c->ivf_residual = 1; c->ivf_cent = CENT_AUTO; c->nprobe_default = 32; c->rk_default = 64;
  c->cb_type = CB_INT8; c->format = DENSE_FORMAT_MAX;
  for (int i = 3; i < argc; i++) {
    char key[64], val[64];
    const char *a = argv[i], *eq = strchr(a, '=');
    if (!eq) { *pzErr = sqlite3_mprintf("dense_ann: bad argument '%s' (expected key=value)", a); return SQLITE_ERROR; }
    int kl = 0, vl = 0;
    for (const char *p = a; p < eq && kl < 63; p++) if (*p != ' ' && *p != '\t' && *p != '\n') key[kl++] = *p;
    key[kl] = 0;
    for (const char *p = eq + 1; *p && vl < 63; p++) if (*p != ' ' && *p != '\'' && *p != '"' && *p != '\t' && *p != '\n') val[vl++] = *p;
    val[vl] = 0;
    long long iv = atoll(val);
    if (!strcmp(key, "dim")) c->dim = (int)iv;
    else if (!strcmp(key, "pq_m")) c->m = (int)iv;
    else if (!strcmp(key, "M")) c->M = (int)iv;
    else if (!strcmp(key, "M0")) c->M0 = (int)iv;
    else if (!strcmp(key, "ef_construction")) c->efc = (int)iv;
    else if (!strcmp(key, "ef_search") || !strcmp(key, "ef")) c->ef_default = (int)iv;
    else if (!strcmp(key, "beam")) c->beam_default = (int)iv;
    else if (!strcmp(key, "entry_points")) c->n_entry = (int)iv;
    else if (!strcmp(key, "dnpq_train")) c->train_max = iv;
    else if (!strcmp(key, "kmeans_iters")) c->iters = (int)iv;
    else if (!strcmp(key, "opq")) c->opq = (int)iv;
    else if (!strcmp(key, "seed")) c->seed = iv;
    else if (!strcmp(key, "alpha")) c->alpha = atof(val);
    else if (!strcmp(key, "page_size_hint")) c->page_hint = (int)iv;
    else if (!strcmp(key, "threads")) c->nthreads = (int)iv;
    else if (!strcmp(key, "verbose")) c->verbose = (int)iv;
    else if (!strcmp(key, "nlist")) c->nlist = (int)iv;
    else if (!strcmp(key, "nprobe")) c->nprobe_default = (int)iv;
    else if (!strcmp(key, "rerank_k")) c->rk_default = (int)iv;
    else if (!strcmp(key, "ivf_residual")) c->ivf_residual = (int)iv;
    else if (!strcmp(key, "ivf_centroids")) {
      if (!strcmp(val, "f16")) c->ivf_cent = CENT_F16;
      else if (!strcmp(val, "int8")) c->ivf_cent = CENT_INT8;
      else if (!strcmp(val, "pq")) c->ivf_cent = CENT_PQ;
      else if (!strcmp(val, "auto")) c->ivf_cent = CENT_AUTO;
      else { *pzErr = sqlite3_mprintf("dense_ann: ivf_centroids must be auto, f16, int8 or pq"); return SQLITE_ERROR; }
    }
    else if (!strcmp(key, "codebook")) {
      if (!strcmp(val, "f16")) c->cb_type = CB_F16;
      else if (!strcmp(val, "int8")) c->cb_type = CB_INT8;
      else { *pzErr = sqlite3_mprintf("dense_ann: codebook must be f16 or int8"); return SQLITE_ERROR; }
    }
    else if (!strcmp(key, "metric")) {
      if (!strcmp(val, "cosine")) c->metric = METRIC_COSINE;
      else if (!strcmp(val, "ip")) c->metric = METRIC_IP;
      else if (!strcmp(val, "l2")) c->metric = METRIC_L2;
      else { *pzErr = sqlite3_mprintf("dense_ann: metric must be cosine, ip or l2"); return SQLITE_ERROR; }
    } else if (!strcmp(key, "store_vectors")) {
      if (!strcmp(val, "none")) c->vtype = VT_NONE;
      else if (!strcmp(val, "int8")) c->vtype = VT_INT8;
      else if (!strcmp(val, "f16")) c->vtype = VT_F16;
      else if (!strcmp(val, "f32")) c->vtype = VT_F32;
      else { *pzErr = sqlite3_mprintf("dense_ann: store_vectors must be none, int8, f16 or f32"); return SQLITE_ERROR; }
    } else if (!strcmp(key, "vectors")) {
      if (!strcmp(val, "inline")) c->vinline = 1;
      else if (!strcmp(val, "table")) c->vinline = 0;
      else { *pzErr = sqlite3_mprintf("dense_ann: vectors must be inline or table"); return SQLITE_ERROR; }
    } else if (!strcmp(key, "layout")) {
      if (!strcmp(val, "colocated")) c->layout = LAYOUT_COLOCATED;
      else if (!strcmp(val, "separate")) c->layout = LAYOUT_SEPARATE;
      else if (!strcmp(val, "ivf")) c->layout = LAYOUT_IVF;
      else { *pzErr = sqlite3_mprintf("dense_ann: layout must be colocated, separate or ivf"); return SQLITE_ERROR; }
    } else if (!strcmp(key, "reorder")) {
      if (!strcmp(val, "auto")) c->reorder = REORDER_AUTO;
      else if (!strcmp(val, "none")) c->reorder = REORDER_NONE;
      else if (!strcmp(val, "pack")) c->reorder = REORDER_PACK;
      else { *pzErr = sqlite3_mprintf("dense_ann: reorder must be auto, none or pack"); return SQLITE_ERROR; }
    } else {
      *pzErr = sqlite3_mprintf("dense_ann: unknown parameter '%s'", key);
      return SQLITE_ERROR;
    }
  }
  if (c->dim <= 0) { *pzErr = sqlite3_mprintf("dense_ann: dim=N is required"); return SQLITE_ERROR; }
  if (c->m == 0) c->m = (c->dim % 64 == 0) ? 64 : (c->dim % 32 == 0 ? 32 : 0);
  if (c->m <= 0 || c->dim % c->m) { *pzErr = sqlite3_mprintf("dense_ann: pq_m must divide dim"); return SQLITE_ERROR; }
  if (c->M < 2 || c->M > 128) { *pzErr = sqlite3_mprintf("dense_ann: M must be in [2,128]"); return SQLITE_ERROR; }
  if (c->M0 <= 0) c->M0 = 2 * c->M;
  if (c->M0 > 255) { *pzErr = sqlite3_mprintf("dense_ann: M0 must be <= 255"); return SQLITE_ERROR; }
  if (c->efc < c->M) c->efc = c->M;
  if (c->vtype == VT_NONE) c->vinline = 0;
  if (c->layout == LAYOUT_IVF) c->vinline = 0;   /* IVF keeps vectors in their own table */
  if (c->nprobe_default < 1) c->nprobe_default = 1;
  if (c->rk_default < 1) c->rk_default = 1;
  if (c->ef_default < 1) c->ef_default = 64;
  if (c->beam_default < 1) c->beam_default = 1;
  return SQLITE_OK;
}

/* ------------------------------------------------------ page reading */

/* Page cache for one query: page number -> buffer. Open addressing. */
typedef struct PageCache { uint32_t *keys; uint8_t **bufs; int cap, n; } PageCache;

static uint8_t *pc_get(PageCache *pc, uint32_t pg) {
  if (!pc->cap) return NULL;
  for (uint32_t h = (pg * 2654435761u) & (pc->cap - 1);; h = (h + 1) & (pc->cap - 1)) {
    if (pc->keys[h] == 0) return NULL;
    if (pc->keys[h] == pg) return pc->bufs[h];
  }
}

static void pc_put(PageCache *pc, uint32_t pg, uint8_t *buf) {
  if (2 * (pc->n + 1) > pc->cap) {
    int ncap = pc->cap ? pc->cap * 2 : 64;
    uint32_t *nk = (uint32_t *)calloc(ncap, sizeof(uint32_t));
    uint8_t **nb = (uint8_t **)calloc(ncap, sizeof(uint8_t *));
    for (int i = 0; i < pc->cap; i++) if (pc->keys[i]) {
      uint32_t h = (pc->keys[i] * 2654435761u) & (ncap - 1);
      while (nk[h]) h = (h + 1) & (ncap - 1);
      nk[h] = pc->keys[i]; nb[h] = pc->bufs[i];
    }
    free(pc->keys); free(pc->bufs);
    pc->keys = nk; pc->bufs = nb; pc->cap = ncap;
  }
  uint32_t h = (pg * 2654435761u) & (pc->cap - 1);
  while (pc->keys[h]) h = (h + 1) & (pc->cap - 1);
  pc->keys[h] = pg; pc->bufs[h] = buf; pc->n++;
}

/* Per-call context: page cache, row copies from SQL fallbacks, stats. */
typedef struct QCtx {
  DenseVtab *vt;
  int raw;           /* direct page reads allowed */
  PageCache pc;
  void **allocs; int nalloc, capalloc;
  QStats *st;
  float *vbuf;       /* dim floats of scratch */
} QCtx;

static void qc_init(QCtx *qc, DenseVtab *vt, int raw, QStats *st) {
  memset(qc, 0, sizeof *qc);
  qc->vt = vt; qc->raw = raw; qc->st = st;
  qc->vbuf = (float *)malloc(sizeof(float) * vt->cfg.dim);
}

static void *qc_keep(QCtx *qc, void *p) {
  if (qc->nalloc == qc->capalloc) {
    qc->capalloc = qc->capalloc ? 2 * qc->capalloc : 64;
    qc->allocs = (void **)realloc(qc->allocs, sizeof(void *) * qc->capalloc);
  }
  qc->allocs[qc->nalloc++] = p;
  return p;
}

static void qc_free(QCtx *qc) {
  for (int i = 0; i < qc->pc.cap; i++) if (qc->pc.keys[i]) free(qc->pc.bufs[i]);
  free(qc->pc.keys); free(qc->pc.bufs);
  for (int i = 0; i < qc->nalloc; i++) free(qc->allocs[i]);
  free(qc->allocs);
  free(qc->vbuf);
}

typedef struct RowReq {
  uint32_t id, page;     /* in: row id and page hint (0 = unknown) */
  const uint8_t *data;   /* out: blob, valid until qc_free */
  int len;
} RowReq;

static int ensure_pr(DenseVtab *vt) {
  if (vt->pr_state == 0) vt->pr_state = dnpr_open(&vt->pr, vt->db, vt->zDb) == 0 ? 1 : -1;
  return vt->pr_state == 1;
}

static int cmp_u32(const void *a, const void *b) {
  uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b;
  return x < y ? -1 : x > y;
}

/* Fetch n rows of shadow table `t` in one dependent step ("round"). With
** direct reads, all uncached hinted pages are announced to the VFS together
** (so an HTTP VFS can fetch them in parallel), then read and parsed. Rows
** without a usable hint go through an SQL lookup (counted as fallbacks). */
/* Announce and read pages (skipping ones already cached for this query) as
** one round. Returns the number of pages actually read. */
static int qc_load_pages(QCtx *qc, uint32_t *pages, int n, int *prc) {
  DenseVtab *vt = qc->vt;
  QStats *st = qc->st;
  int m = 0;
  for (int i = 0; i < n; i++) if (pages[i] && !pc_get(&qc->pc, pages[i])) pages[m++] = pages[i];
  qsort(pages, m, sizeof(uint32_t), cmp_u32);
  int u = 0;
  for (int i = 0; i < m; i++) if (u == 0 || pages[i] != pages[u - 1]) pages[u++] = pages[i];
  if (!u) return 0;
  dnpr_prefetch(&vt->pr, pages, u);
  if (st->trace) sqlite3_str_appendf(st->trace, "%s[", st->ntrace++ ? "," : "");
  for (int i = 0; i < u; i++) {
    uint8_t *buf = (uint8_t *)malloc(vt->pr.pgsz);
    if (!buf) { *prc = SQLITE_NOMEM; break; }
    if (dnpr_read(&vt->pr, pages[i], buf) != SQLITE_OK) { free(buf); continue; }
    pc_put(&qc->pc, pages[i], buf);
    st->pages++; st->bytes += vt->pr.pgsz;
    if (st->trace) sqlite3_str_appendf(st->trace, "%s%u", i ? "," : "", pages[i]);
  }
  if (st->trace) sqlite3_str_appendf(st->trace, "]");
  return u;
}

/* Look up row `id` of a table on page `page` in the query page cache. */
static const uint8_t *qc_row_on_page(QCtx *qc, uint32_t page, int64_t id, int *len) {
  const uint8_t *pg, *pl, *blob; int plen, blen;
  if (!page || !(pg = pc_get(&qc->pc, page))) return NULL;
  if (dnpr_leaf_find(&qc->vt->pr, pg, page, id, &pl, &plen) && dnpr_record_blob(pl, plen, 1, &blob, &blen)) {
    *len = blen;
    return blob;
  }
  return NULL;
}

static int q_fetch(QCtx *qc, int t, RowReq *r, int n) {
  DenseVtab *vt = qc->vt;
  QStats *st = qc->st;
  if (n <= 0) return SQLITE_OK;
  int nneed = 0, nfall = 0, rc = SQLITE_OK;
  uint32_t *need = NULL;
  for (int i = 0; i < n; i++) r[i].data = NULL, r[i].len = 0;

  if (qc->raw) {
    need = (uint32_t *)malloc(sizeof(uint32_t) * n);
    for (int i = 0; i < n; i++) if (r[i].page) need[nneed++] = r[i].page;
    nneed = qc_load_pages(qc, need, nneed, &rc);
    for (int i = 0; i < n; i++) {
      const uint8_t *page, *pl, *blob; int len, blen;
      if (!r[i].page || !(page = pc_get(&qc->pc, r[i].page))) continue;
      if (dnpr_leaf_find(&vt->pr, page, r[i].page, r[i].id, &pl, &len) &&
          dnpr_record_blob(pl, len, 1, &blob, &blen)) {
        r[i].data = blob; r[i].len = blen;
      }
    }
    free(need);
  }

  /* SQL fallback for rows not found through a page hint. */
  for (int i = 0; i < n && rc == SQLITE_OK; i++) {
    if (r[i].data) continue;
    if (!vt->st_get[t]) {
      rc = prep_fmt(vt->db, &vt->st_get[t], "SELECT data FROM \"%w\".\"%w_%s\" WHERE id = ?1",
                    vt->zDb, vt->zName, TABLE_SUFFIX[t]);
      if (rc) break;
    }
    sqlite3_stmt *s = vt->st_get[t];
    sqlite3_bind_int64(s, 1, r[i].id);
    if (sqlite3_step(s) == SQLITE_ROW) {
      int len = sqlite3_column_bytes(s, 0);
      uint8_t *copy = (uint8_t *)qc_keep(qc, malloc(len ? len : 1));
      memcpy(copy, sqlite3_column_blob(s, 0), len);
      r[i].data = copy; r[i].len = len;
    }
    rc = sqlite3_reset(s);
    nfall++;
  }
  st->fallback += nfall;
  if (nneed || nfall) st->rounds++;
  return rc;
}

/* ----------------------------------------------------- chunked blobs */

/* Write data as consecutive rows of _blobs starting at first_id. Returns a
** malloc'd list of (u32 id, u32 page hint = 0) pairs in *plist. */
static int write_chunks(DenseVtab *vt, int64_t first_id, const uint8_t *data, int64_t len,
                        uint8_t **plist, int *plen) {
  sqlite3_stmt *st = 0;
  int rc = PREP(vt, &st, "INSERT OR REPLACE INTO \"%w\".\"%w_blobs\"(id, data) VALUES (?1, ?2)");
  if (rc) return rc;
  int nchunks = (int)((len + CHUNK_BYTES - 1) / CHUNK_BYTES);
  uint8_t *list = (uint8_t *)calloc(nchunks ? nchunks : 1, 8);
  for (int i = 0; i < nchunks && rc == SQLITE_OK; i++) {
    int64_t off = (int64_t)i * CHUNK_BYTES;
    int sz = (int)(len - off < CHUNK_BYTES ? len - off : CHUNK_BYTES);
    sqlite3_bind_int64(st, 1, first_id + i);
    sqlite3_bind_blob(st, 2, data + off, sz, SQLITE_STATIC);
    rc = sqlite3_step(st) == SQLITE_DONE ? SQLITE_OK : SQLITE_ERROR;
    sqlite3_reset(st);
    dn_wr32(list + 8 * i, (uint32_t)(first_id + i));
  }
  sqlite3_finalize(st);
  *plist = list; *plen = nchunks * 8;
  return rc;
}

/* ------------------------------------------------ codebook storage */

/* Bytes of a stored PQ codebook of cb_vals values: float16 each, or per
** sub-quantiser a float32 scale followed by its values as int8. */
static size_t cb_bytes(int type, int m, size_t cb_vals) {
  return type == CB_INT8 ? cb_vals + 4 * (size_t)m : 2 * cb_vals;
}

static void cb_store(int type, int m, const float *cent, size_t cb_vals, uint8_t *out) {
  if (type != CB_INT8) { for (size_t j = 0; j < cb_vals; j++) dn_wr16(out + 2 * j, dn_f32_to_f16(cent[j])); return; }
  size_t per = cb_vals / m;
  for (int s = 0; s < m; s++) {
    const float *v = cent + (size_t)s * per;
    float mx = 0;
    for (size_t j = 0; j < per; j++) { float a = fabsf(v[j]); if (a > mx) mx = a; }
    float sc = mx / 127.0f;
    uint8_t *o = out + (size_t)s * (per + 4);
    memcpy(o, &sc, 4);                        /* little-endian hosts only, as elsewhere */
    for (size_t j = 0; j < per; j++) {
      long q = sc > 0 ? lroundf(v[j] / sc) : 0;
      o[4 + j] = (uint8_t)(int8_t)(q > 127 ? 127 : q < -127 ? -127 : q);
    }
  }
}

static void cb_load(int type, int m, const uint8_t *in, size_t cb_vals, float *cent) {
  if (type != CB_INT8) { for (size_t j = 0; j < cb_vals; j++) cent[j] = dn_f16_to_f32(dn_rd16(in + 2 * j)); return; }
  size_t per = cb_vals / m;
  for (int s = 0; s < m; s++) {
    const uint8_t *o = in + (size_t)s * (per + 4);
    float sc; memcpy(&sc, o, 4);
    for (size_t j = 0; j < per; j++) cent[(size_t)s * per + j] = (float)(int8_t)o[4 + j] * sc;
  }
}

/* Round a trained codebook to its stored values (before encoding with it). */
static void cb_round(int type, PQ *pq) {
  size_t n = (size_t)pq->m * PQ_KSUB * pq->dsub;
  if (type != CB_INT8) { dnpq_round_f16(pq); return; }
  uint8_t *tmp = (uint8_t *)malloc(cb_bytes(type, pq->m, n));
  if (!tmp) { dnpq_round_f16(pq); return; }
  cb_store(type, pq->m, pq->cent, n, tmp);
  cb_load(type, pq->m, tmp, n, pq->cent);
  free(tmp);
}

/* Chunked blobs that make up the cached index head, by config key. */
static const char *const HEAD_LISTS[] = {
  "codebook_chunks", "entry_chunks", "rotation_chunks", "ivf_head_chunks", "ivf_cpq_chunks"
};
#define NHEAD 5

static size_t ivf_cent_bytes(const DenseCfg *c);
static int ivf_decode_head(DenseVtab *vt);

/* Load the index head (once per connection): PQ codebook, entry set or IVF
** centroids and directory, optional rotation. All chunks of all parts are
** fetched in a single round. */
static int ensure_loaded(DenseVtab *vt, int raw, QStats *st) {
  if (vt->loaded) return SQLITE_OK;
  const DenseCfg *c = &vt->cfg;
  uint8_t *lists[NHEAD]; int nl[NHEAD], total = 0, rc = SQLITE_OK;
  for (int h = 0; h < NHEAD; h++) {
    lists[h] = NULL; nl[h] = 0;
    if (rc == SQLITE_OK) rc = cfg_get_blob(vt, HEAD_LISTS[h], &lists[h], &nl[h]);
    nl[h] /= 8; total += nl[h];
  }
  RowReq *req = (RowReq *)calloc(total + 1, sizeof(RowReq));
  for (int h = 0, k = 0; h < NHEAD; h++)
    for (int i = 0; i < nl[h]; i++, k++) { req[k].id = dn_rd32(lists[h] + 8 * i); req[k].page = dn_rd32(lists[h] + 8 * i + 4); }

  QStats lst; memset(&lst, 0, sizeof lst);
  QCtx qc; qc_init(&qc, vt, raw && vt->hints_valid && ensure_pr(vt), &lst);
  if (rc == SQLITE_OK) rc = q_fetch(&qc, T_BLOBS, req, total);

  /* Concatenate each part. */
  uint8_t *part[NHEAD]; size_t plen[NHEAD];
  for (int h = 0, k = 0; h < NHEAD; h++) {
    plen[h] = 0;
    for (int i = 0; i < nl[h]; i++) plen[h] += req[k + i].len;
    part[h] = (uint8_t *)malloc(plen[h] ? plen[h] : 1);
    size_t pos = 0;
    for (int i = 0; i < nl[h]; i++, k++) {
      if (!req[k].data) { if (rc == SQLITE_OK) rc = SQLITE_CORRUPT; continue; }
      memcpy(part[h] + pos, req[k].data, req[k].len);
      pos += req[k].len;
    }
  }
  size_t cb_vals = (size_t)c->m * PQ_KSUB * (c->dim / c->m);
  if (rc == SQLITE_OK) {
    dnpq_free(&vt->pq);
    if (dnpq_init(&vt->pq, c->dim, c->m)) rc = SQLITE_NOMEM;
  }
  if (rc == SQLITE_OK && plen[0] != cb_bytes(c->cb_type, c->m, cb_vals)) rc = SQLITE_CORRUPT;
  if (rc == SQLITE_OK) cb_load(c->cb_type, c->m, part[0], cb_vals, vt->pq.cent);
  size_t esz = (size_t)vt->n_entry * (8 + c->m);
  if (rc == SQLITE_OK && plen[1] != esz) rc = SQLITE_CORRUPT;
  if (rc == SQLITE_OK) { free(vt->entries); vt->entries = part[1]; part[1] = NULL; }
  if (rc == SQLITE_OK && plen[2] > 0) {
    size_t nrot = (size_t)c->dim * c->dim;
    if (plen[2] != nrot * 2) rc = SQLITE_CORRUPT;
    else {
      vt->pq.rot = (float *)malloc(sizeof(float) * nrot);
      for (size_t i = 0; i < nrot; i++) vt->pq.rot[i] = dn_f16_to_f32(dn_rd16(part[2] + 2 * i));
    }
  }
  if (rc == SQLITE_OK && c->layout == LAYOUT_IVF) {
    if (plen[3] != (size_t)vt->nlist * (ivf_cent_bytes(c) + 12)) rc = SQLITE_CORRUPT;
    if (rc == SQLITE_OK && c->ivf_cent == CENT_PQ) {
      dnpq_free(&vt->ivf_cpq);
      if (dnpq_init(&vt->ivf_cpq, c->dim, c->m)) rc = SQLITE_NOMEM;
      else if (plen[4] != cb_bytes(c->cb_type, c->m, cb_vals)) rc = SQLITE_CORRUPT;
      else cb_load(c->cb_type, c->m, part[4], cb_vals, vt->ivf_cpq.cent);
    }
    if (rc == SQLITE_OK) {
      free(vt->ivf_head); vt->ivf_head = part[3]; part[3] = NULL;
      rc = ivf_decode_head(vt);
    }
  }
  for (int h = 0; h < NHEAD; h++) { free(part[h]); free(lists[h]); }
  qc_free(&qc);
  free(req);
  if (rc == SQLITE_OK) {
    vt->loaded = 1;
    if (st) { st->setup_rounds += lst.rounds; st->setup_pages += lst.pages; st->setup_bytes += lst.bytes; }
  } else {
    set_err(vt, "dense_ann: could not load the index head (codebook, entry set / centroids)");
  }
  return rc;
}

/* ------------------------------------------------------------- search */

typedef struct Cand {
  float d;
  uint32_t id, page;
  const uint8_t *row;   /* node row once expanded */
  int expanded;
} Cand;

/* Insert into the sorted, bounded candidate list L[0..*n) (capacity ef). */
static void cand_insert(Cand *L, int *n, int ef, float d, uint32_t id, uint32_t page) {
  if (*n == ef && d >= L[ef - 1].d) return;
  int i = (*n < ef) ? (*n)++ : ef - 1;
  while (i > 0 && L[i - 1].d > d) { L[i] = L[i - 1]; i--; }
  L[i].d = d; L[i].id = id; L[i].page = page; L[i].row = NULL; L[i].expanded = 0;
}

/* Visited set of node ids (stores id+1; open addressing). */
typedef struct IdSet { uint32_t *k; int cap, n; } IdSet;

static int idset_add(IdSet *s, uint32_t id) {
  uint32_t key = id + 1;
  if (2 * (s->n + 1) > s->cap) {
    int ncap = s->cap ? s->cap * 2 : 1024;
    uint32_t *nk = (uint32_t *)calloc(ncap, sizeof(uint32_t));
    for (int i = 0; i < s->cap; i++) if (s->k[i]) {
      uint32_t h = (s->k[i] * 2654435761u) & (ncap - 1);
      while (nk[h]) h = (h + 1) & (ncap - 1);
      nk[h] = s->k[i];
    }
    free(s->k); s->k = nk; s->cap = ncap;
  }
  uint32_t h = (key * 2654435761u) & (s->cap - 1);
  while (s->k[h]) {
    if (s->k[h] == key) return 0;
    h = (h + 1) & (s->cap - 1);
  }
  s->k[h] = key; s->n++;
  return 1;
}

/* Beam search on the persisted layer-0 graph using PQ distances.
**
** Seeds: every node in the cached entry set (scored in memory). Each step
** expands the (up to) W closest unexpanded candidates: their rows are
** fetched in one round, and their neighbours are scored from the co-located
** PQ codes (or, in the separate layout, from a second round of code reads).
** Stops when all ef candidates in L have been expanded. */
static int ann_core(DenseVtab *vt, QCtx *qc, const float *tab, const float *qexact, int ef, int W,
                    Cand *L, int *pnL, Cand **pexp, int *pnexp) {
  const RowLayout *rl = &vt->rl;
  const int m = vt->cfg.m, M0 = vt->cfg.M0, estride = 8 + m;
  QStats *st = qc->st;
  IdSet vis; memset(&vis, 0, sizeof vis);
  int nL = 0, rc = SQLITE_OK;
  Cand *exp = NULL;         /* every expanded node, for reranking */
  int nexp = 0, capexp = 0;

  for (int e = 0; e < vt->n_entry; e++) {
    const uint8_t *ent = vt->entries + (size_t)e * estride;
    uint32_t id = dn_rd32(ent);
    idset_add(&vis, id);
    cand_insert(L, &nL, ef, dnpq_adc(tab, ent + 8, m), id, dn_rd32(ent + 4));
    st->entry_dist++;
  }

  RowReq *req = (RowReq *)malloc(sizeof(RowReq) * W);
  int npend_cap = W * M0;
  RowReq *pend = (RowReq *)malloc(sizeof(RowReq) * (npend_cap ? npend_cap : 1));
  uint32_t *pend_page = (uint32_t *)malloc(sizeof(uint32_t) * (npend_cap ? npend_cap : 1));

  for (;;) {
    int nb = 0;
    int bidx[256];
    for (int i = 0; i < nL && nb < W && nb < 256; i++)
      if (!L[i].expanded) { bidx[nb] = i; req[nb].id = L[i].id; req[nb].page = L[i].page; nb++; }
    if (nb == 0) break;
    rc = q_fetch(qc, T_NODES, req, nb);
    if (rc) break;
    /* Mark before inserting anything: inserts shift positions in L. */
    for (int b = 0; b < nb; b++) {
      L[bidx[b]].expanded = 1;
      L[bidx[b]].row = (req[b].data && req[b].len >= rl->size) ? req[b].data : NULL;
      st->expanded++;
      if (qexact && L[bidx[b]].row) {
        /* Exact navigation: the expanded node's own (inline) vector is in
        ** hand, so replace its PQ estimate by the true distance. */
        vec_decode(&vt->cfg, L[bidx[b]].row + rl->off_vec, qc->vbuf);
        L[bidx[b]].d = dn_distance(vt->cfg.metric, qexact, qc->vbuf, vt->cfg.dim);
        st->rerank++;
      }
      if (pexp && L[bidx[b]].row) {
        if (nexp == capexp) { capexp = capexp ? 2 * capexp : 64; exp = (Cand *)realloc(exp, sizeof(Cand) * capexp); }
        exp[nexp++] = L[bidx[b]];
      }
    }
    if (qexact) {   /* restore order after distance updates (insertion sort) */
      for (int i = 1; i < nL; i++) {
        Cand t = L[i]; int j = i - 1;
        while (j >= 0 && L[j].d > t.d) { L[j + 1] = L[j]; j--; }
        L[j + 1] = t;
      }
    }
    int np = 0;
    for (int b = 0; b < nb; b++) {
      const uint8_t *row = req[b].data;
      if (!row || req[b].len < rl->size) continue;
      int deg = dn_rd16(row + 2);
      if (deg > M0) deg = M0;
      for (int j = 0; j < deg; j++) {
        const uint8_t *e = row + rl->off_nbr + j * rl->nbr_stride;
        uint32_t nid = dn_rd32(e);
        if (!idset_add(&vis, nid)) continue;
        if (vt->cfg.layout == LAYOUT_COLOCATED) {
          float d = dnpq_adc(tab, row + rl->off_nbrcode + j * m, m);
          st->dist++;
          cand_insert(L, &nL, ef, d, nid, dn_rd32(e + 4));
        } else if (np < npend_cap) {
          pend[np].id = nid; pend[np].page = dn_rd32(e + 8); pend_page[np] = dn_rd32(e + 4);
          np++;
        }
      }
    }
    if (np > 0) {
      /* Separate layout: a second dependent round for the neighbours' codes. */
      rc = q_fetch(qc, T_CODES, pend, np);
      if (rc) break;
      for (int i = 0; i < np; i++) {
        if (!pend[i].data || pend[i].len < m) continue;
        st->dist++;
        cand_insert(L, &nL, ef, dnpq_adc(tab, pend[i].data, m), pend[i].id, pend_page[i]);
      }
    }
  }
  free(req); free(pend); free(pend_page); free(vis.k);
  *pnL = nL;
  if (pexp) { *pexp = exp; *pnexp = nexp; } else free(exp);
  return rc;
}

static int cmp_result(const void *a, const void *b) {
  float x = ((const Result *)a)->d, y = ((const Result *)b)->d;
  return x < y ? -1 : x > y;
}

/* Approximate search. q must already be normalised for cosine. */
static int ann_search(DenseVtab *vt, const float *q, int k, int ef, int W, int rerank, int raw,
                      Result **pres, int *pn, QStats *st) {
  const DenseCfg *c = &vt->cfg;
  const RowLayout *rl = &vt->rl;
  int rc = ensure_loaded(vt, raw, st);
  if (rc) return rc;
  if (ef < k) ef = k;
  if (W < 1) W = 1;
  if (W > 256) W = 256;
  raw = raw && vt->hints_valid && ensure_pr(vt);
  st->raw = raw;
  st->ann = 1;

  float *tab = (float *)malloc(sizeof(float) * c->m * PQ_KSUB);
  dnpq_adc_table(&vt->pq, q, c->metric, tab);
  QCtx qc; qc_init(&qc, vt, raw, st);
  Cand *L = (Cand *)calloc(ef, sizeof(Cand));
  Cand *exp = NULL;
  int nL = 0, nexp = 0;
  if (rerank && c->vtype == VT_NONE) rerank = 0;
  if (rerank >= 3 && !c->vinline) rerank = 2;
  rc = ann_core(vt, &qc, tab, rerank >= 3 ? q : NULL, ef, W, L, &nL, &exp, &nexp);

  /* Candidates to return: rerank=0/1 use the final list L (PQ order);
  ** rerank=2 uses every expanded node, as DiskANN does. */
  Cand *cand = (rerank >= 2) ? exp : L;
  int ncand = (rerank >= 2) ? nexp : nL;
  Result *res = (Result *)malloc(sizeof(Result) * (ncand ? ncand : 1));
  const uint8_t **rows = (const uint8_t **)malloc(sizeof(uint8_t *) * (ncand ? ncand : 1));
  uint32_t *ids = (uint32_t *)malloc(sizeof(uint32_t) * (ncand ? ncand : 1));
  int nres = 0;
  for (int i = 0; rc == SQLITE_OK && i < ncand; i++) {
    if (!cand[i].row || (cand[i].row[0] & NODE_FLAG_DELETED)) continue;
    rows[nres] = cand[i].row;
    ids[nres] = cand[i].id;
    res[nres].rowid = dn_rd64(cand[i].row + 8);
    res[nres].d = cand[i].d;
    nres++;
  }
  if (rc == SQLITE_OK && rerank && nres > 0) {
    /* Re-score with stored vectors: free if inline, one extra round if not. */
    float *v = (float *)malloc(sizeof(float) * c->dim);
    RowReq *vr = NULL;
    if (!c->vinline) {
      vr = (RowReq *)malloc(sizeof(RowReq) * nres);
      for (int i = 0; i < nres; i++) { vr[i].id = ids[i]; vr[i].page = dn_rd32(rows[i] + 4); }
      int before = st->rounds;
      rc = q_fetch(&qc, T_VECS, vr, nres);
      st->rerank_rounds += st->rounds - before;
    }
    for (int i = 0; i < nres && rc == SQLITE_OK; i++) {
      const uint8_t *vb = c->vinline ? rows[i] + rl->off_vec
                                     : (vr[i].data && vr[i].len >= rl->vec_bytes ? vr[i].data : NULL);
      if (!vb) { res[i].d = FLT_MAX; continue; }
      vec_decode(c, vb, v);
      res[i].d = dn_distance(c->metric, q, v, c->dim);
      st->rerank++;
    }
    qsort(res, nres, sizeof(Result), cmp_result);
    free(v); free(vr);
  }
  free(rows); free(ids); free(exp);
  if (nres > k) nres = k;
  qc_free(&qc);
  free(L); free(tab);
  *pres = res; *pn = nres;
  return rc;
}

/* Bounded top-k insertion for exact search. */
static void topk_insert(Result *R, int *n, int k, int64_t rowid, float d) {
  if (*n == k && d >= R[k - 1].d) return;
  int i = (*n < k) ? (*n)++ : k - 1;
  while (i > 0 && R[i - 1].d > d) { R[i] = R[i - 1]; i--; }
  R[i].rowid = rowid; R[i].d = d;
}

/* Brute force. mode 1: stored vectors (float32 buffer before build);
** mode 2: PQ codes (ADC), i.e. the best any PQ-navigated search can do. */
static int exact_search(DenseVtab *vt, const float *q, int k, int mode, Result **pres, int *pn, QStats *st) {
  const DenseCfg *c = &vt->cfg;
  const RowLayout *rl = &vt->rl;
  sqlite3_stmt *s = 0;
  int rc;
  float *tab = NULL;
  if (!vt->built) {
    rc = PREP(vt, &s, "SELECT rowid, vec FROM \"%w\".\"%w_buffer\"");
    mode = 1;
  } else if (mode == 2 || c->vtype == VT_NONE) {
    if (mode == 1) { set_err(vt, "dense_ann: exact=1 needs store_vectors; use exact=2 (PQ)"); return SQLITE_ERROR; }
    rc = ensure_loaded(vt, 0, st);
    if (rc) return rc;
    tab = (float *)malloc(sizeof(float) * c->m * PQ_KSUB);
    dnpq_adc_table(&vt->pq, q, c->metric, tab);
    rc = prep_fmt(vt->db, &s, "SELECT r.rowid, n.data FROM \"%w\".\"%w_rowids\" r JOIN \"%w\".\"%w_nodes\" n ON n.id = r.node",
                  vt->zDb, vt->zName, vt->zDb, vt->zName);
  } else if (c->vinline) {
    rc = prep_fmt(vt->db, &s, "SELECT r.rowid, n.data FROM \"%w\".\"%w_rowids\" r JOIN \"%w\".\"%w_nodes\" n ON n.id = r.node",
                  vt->zDb, vt->zName, vt->zDb, vt->zName);
  } else {
    rc = prep_fmt(vt->db, &s, "SELECT r.rowid, v.data FROM \"%w\".\"%w_rowids\" r JOIN \"%w\".\"%w_vectors\" v ON v.id = (r.node & 4294967295)",
                  vt->zDb, vt->zName, vt->zDb, vt->zName);
  }
  if (rc) { free(tab); return rc; }
  Result *R = (Result *)malloc(sizeof(Result) * k);
  int n = 0;
  float *v = (float *)malloc(sizeof(float) * c->dim);
  while (sqlite3_step(s) == SQLITE_ROW) {
    int64_t rowid = sqlite3_column_int64(s, 0);
    const uint8_t *b = (const uint8_t *)sqlite3_column_blob(s, 1);
    int len = sqlite3_column_bytes(s, 1);
    float d;
    if (!vt->built) {
      if (len != c->dim * 4) continue;
      memcpy(v, b, len);
      d = dn_distance(c->metric, q, v, c->dim);
    } else if (tab) {
      if (len < rl->size) continue;
      d = dnpq_adc(tab, b + rl->off_code, c->m);
    } else if (c->vinline) {
      if (len < rl->size) continue;
      vec_decode(c, b + rl->off_vec, v);
      d = dn_distance(c->metric, q, v, c->dim);
    } else {
      if (len < rl->vec_bytes) continue;
      vec_decode(c, b, v);
      d = dn_distance(c->metric, q, v, c->dim);
    }
    st->dist++;
    topk_insert(R, &n, k, rowid, d);
  }
  rc = sqlite3_finalize(s);
  free(v); free(tab);
  *pres = R; *pn = n;
  return rc;
}

/* ------------------------------------------------------------- build */

typedef struct { uint8_t level; uint64_t h; uint32_t i; } EntryKey;
static int cmp_entry_key(const void *a, const void *b) {
  const EntryKey *x = (const EntryKey *)a, *y = (const EntryKey *)b;
  if (x->level != y->level) return x->level > y->level ? -1 : 1;
  return x->h < y->h ? -1 : x->h > y->h;
}

/* Nodes per page when rows of `rowsize` bytes are appended to a table with
** page size pgsz (cell = record + payload/rowid varints + 2-byte pointer). */
static int rows_per_page(int rowsize, int pgsz) {
  int cell = rowsize + 4 /* record header */ + 3 /* payload varint */ + 4 /* rowid varint */ + 2;
  int p = (pgsz - 8) / cell;
  return p < 1 ? 1 : p;
}

/* Choose storage order for nodes. With several rows per page, pack each
** page with a node and its graph neighbourhood (BFS within the page), so a
** search that expands nearby nodes touches fewer distinct pages. Seeds are
** taken in global BFS order from the HNSW entry point. perm[new] = old. */
static void order_nodes(const Hnsw *h, int per_page, uint32_t *perm) {
  int64_t n = h->n;
  if (per_page <= 1) {
    for (int64_t i = 0; i < n; i++) perm[i] = (uint32_t)i;
    return;
  }
  uint8_t *seen = (uint8_t *)calloc(n, 1);       /* global BFS */
  uint32_t *bfs = (uint32_t *)malloc(sizeof(uint32_t) * n);
  int64_t head = 0, tail = 0;
  for (int64_t s = -1; tail < n;) {
    uint32_t start;
    if (s < 0) start = h->entry, s = 0;
    else { while (s < n && seen[s]) s++; if (s >= n) break; start = (uint32_t)s; }
    if (seen[start]) continue;
    seen[start] = 1; bfs[tail++] = start;
    while (head < tail) {
      uint32_t u = bfs[head++];
      const uint32_t *l = dnhnsw_list(h, u, 0);
      for (uint32_t j = 1; j <= l[0]; j++) if (!seen[l[j]]) { seen[l[j]] = 1; bfs[tail++] = l[j]; }
    }
  }
  uint8_t *done = seen; memset(done, 0, n);
  int qcap = per_page * (h->M0 + 1) + 16;
  uint32_t *lq = (uint32_t *)malloc(sizeof(uint32_t) * qcap);
  int64_t next = 0;
  for (int64_t si = 0; si < n; si++) {
    uint32_t s = bfs[si];
    if (done[s]) continue;
    int qh = 0, qt = 0;
    lq[qt++] = s;
    while (qh < qt) {
      uint32_t u = lq[qh++];
      if (done[u]) continue;
      done[u] = 1; perm[next++] = u;
      if (next % per_page == 0) break;            /* page full */
      const uint32_t *l = dnhnsw_list(h, u, 0);
      for (uint32_t j = 1; j <= l[0] && qt < qcap; j++) if (!done[l[j]]) lq[qt++] = l[j];
    }
  }
  free(lq); free(bfs); free(seen);
}

static int build_ivf(DenseVtab *vt, const float *X, const int64_t *rowids, int64_t n, double t0, double t_load);

static int do_build(DenseVtab *vt) {
  DenseCfg *c = &vt->cfg;
  const RowLayout *rl = &vt->rl;
  const int dim = c->dim, m = c->m;
  sqlite3_stmt *st = 0;
  int rc;
  double t0 = now_ms();
  if (vt->built) { set_err(vt, "dense_ann: index already built"); return SQLITE_ERROR; }

  /* 1. Load the buffered vectors, then free the buffer so its pages can be reused. */
  int64_t n = 0;
  rc = PREP(vt, &st, "SELECT count(*) FROM \"%w\".\"%w_buffer\"");
  if (rc) return rc;
  if (sqlite3_step(st) == SQLITE_ROW) n = sqlite3_column_int64(st, 0);
  sqlite3_finalize(st); st = 0;
  if (n == 0) { set_err(vt, "dense_ann: nothing to build (no vectors inserted)"); return SQLITE_ERROR; }
  if (n >= 0xFFFFFFF0LL) { set_err(vt, "dense_ann: too many vectors"); return SQLITE_ERROR; }
  float *X = (float *)malloc(sizeof(float) * (size_t)n * dim);
  int64_t *rowids = (int64_t *)malloc(sizeof(int64_t) * n);
  if (!X || !rowids) { free(X); free(rowids); return SQLITE_NOMEM; }
  rc = PREP(vt, &st, "SELECT rowid, vec FROM \"%w\".\"%w_buffer\" ORDER BY rowid");
  int64_t i = 0;
  while (rc == SQLITE_OK && i < n && sqlite3_step(st) == SQLITE_ROW) {
    rowids[i] = sqlite3_column_int64(st, 0);
    if (sqlite3_column_bytes(st, 1) != dim * 4) { rc = SQLITE_CORRUPT; break; }
    memcpy(X + (size_t)i * dim, sqlite3_column_blob(st, 1), dim * 4);
    i++;
  }
  sqlite3_finalize(st); st = 0;
  if (rc == SQLITE_OK && i != n) rc = SQLITE_CORRUPT;
  /* The graph layout frees the buffer first so its pages are reused. The IVF
  ** layout frees it afterwards: reusing scattered freelist pages would break
  ** the contiguous page ranges of the posting lists (VACUUM reclaims it). */
  if (rc == SQLITE_OK && c->layout != LAYOUT_IVF)
    rc = exec_fmt(vt->db, "DELETE FROM \"%w\".\"%w_buffer\"", vt->zDb, vt->zName);
  if (rc) { free(X); free(rowids); set_err(vt, "dense_ann: build: %s", sqlite3_errmsg(vt->db)); return rc; }
  double t_load = now_ms();

  if (c->layout == LAYOUT_IVF) {
    rc = build_ivf(vt, X, rowids, n, t0, t_load);
    free(X); free(rowids);
    if (rc == SQLITE_OK) rc = exec_fmt(vt->db, "DELETE FROM \"%w\".\"%w_buffer\"", vt->zDb, vt->zName);
    return rc;
  }

  /* 2. Train PQ and encode everything. The codebook is rounded to float16
  **    first, because that is what is stored. */
  PQ pq;
  if (dnpq_train(&pq, X, n, dim, m, c->train_max, c->iters, c->opq, (uint64_t)c->seed, c->nthreads)) {
    free(X); free(rowids); return SQLITE_NOMEM;
  }
  cb_round(c->cb_type, &pq);
  uint8_t *codes = (uint8_t *)malloc((size_t)n * m);
  dnpq_encode_many(&pq, X, n, codes, c->nthreads);
  double t_pq = now_ms();
  if (c->verbose) fprintf(stderr, "dense_ann: pq trained+encoded in %.1fs\n", (t_pq - t_load) / 1e3);

  /* 3. HNSW graph over full-precision vectors. */
  Hnsw h; memset(&h, 0, sizeof h);
  h.dim = dim; h.M = c->M; h.M0 = c->M0; h.efc = c->efc; h.metric = c->metric;
  h.alpha = (float)c->alpha; h.seed = (uint64_t)c->seed; h.n = n; h.x = X;
  if (dnhnsw_build(&h, c->nthreads, c->verbose)) { rc = SQLITE_NOMEM; goto done; }
  double t_graph = now_ms();

  /* 4. Storage order. */
  int pgsz = c->page_hint;
  if (pgsz <= 0) {
    rc = prep_fmt(vt->db, &st, "PRAGMA \"%w\".page_size", vt->zDb);
    if (rc == SQLITE_OK && sqlite3_step(st) == SQLITE_ROW) pgsz = sqlite3_column_int(st, 0);
    sqlite3_finalize(st); st = 0;
  }
  int per_page = rows_per_page(rl->size, pgsz > 0 ? pgsz : 4096);
  uint32_t *perm = (uint32_t *)malloc(sizeof(uint32_t) * n);
  uint32_t *inv = (uint32_t *)malloc(sizeof(uint32_t) * n);
  order_nodes(&h, c->reorder == REORDER_NONE ? 1 : per_page, perm);
  for (int64_t j = 0; j < n; j++) inv[perm[j]] = (uint32_t)j;

  /* 5. Write node rows, then the other per-node tables (one table at a time
  **    so each table's pages are contiguous in the file). */
  uint8_t *row = (uint8_t *)malloc(rl->size);
  int64_t deg_sum = 0;
  rc = PREP(vt, &st, "INSERT INTO \"%w\".\"%w_nodes\"(id, data) VALUES (?1, ?2)");
  for (int64_t j = 0; j < n && rc == SQLITE_OK; j++) {
    uint32_t o = perm[j];
    const uint32_t *l = dnhnsw_list(&h, o, 0);
    memset(row, 0, rl->size);
    dn_wr16(row + 2, (uint16_t)l[0]);
    dn_wr64(row + 8, rowids[o]);
    memcpy(row + rl->off_code, codes + (size_t)o * m, m);
    if (c->vinline) vec_encode(c, X + (size_t)o * dim, row + rl->off_vec);
    for (uint32_t t = 0; t < l[0]; t++) {
      dn_wr32(row + rl->off_nbr + t * rl->nbr_stride, inv[l[1 + t]]);
      if (c->layout == LAYOUT_COLOCATED)
        memcpy(row + rl->off_nbrcode + t * m, codes + (size_t)l[1 + t] * m, m);
    }
    deg_sum += l[0];
    sqlite3_bind_int64(st, 1, j);
    sqlite3_bind_blob(st, 2, row, rl->size, SQLITE_STATIC);
    if (sqlite3_step(st) != SQLITE_DONE) rc = SQLITE_ERROR;
    sqlite3_reset(st);
  }
  sqlite3_finalize(st); st = 0;

  if (rc == SQLITE_OK) rc = PREP(vt, &st, "INSERT INTO \"%w\".\"%w_rowids\"(rowid, node) VALUES (?1, ?2)");
  for (int64_t j = 0; j < n && rc == SQLITE_OK; j++) {
    sqlite3_bind_int64(st, 1, rowids[perm[j]]);
    sqlite3_bind_int64(st, 2, j);
    if (sqlite3_step(st) != SQLITE_DONE) rc = SQLITE_ERROR;
    sqlite3_reset(st);
  }
  sqlite3_finalize(st); st = 0;

  if (rc == SQLITE_OK && c->layout == LAYOUT_SEPARATE) {
    rc = PREP(vt, &st, "INSERT INTO \"%w\".\"%w_codes\"(id, data) VALUES (?1, ?2)");
    for (int64_t j = 0; j < n && rc == SQLITE_OK; j++) {
      sqlite3_bind_int64(st, 1, j);
      sqlite3_bind_blob(st, 2, codes + (size_t)perm[j] * m, m, SQLITE_STATIC);
      if (sqlite3_step(st) != SQLITE_DONE) rc = SQLITE_ERROR;
      sqlite3_reset(st);
    }
    sqlite3_finalize(st); st = 0;
  }
  if (rc == SQLITE_OK && c->vtype != VT_NONE && !c->vinline) {
    uint8_t *vb = (uint8_t *)malloc(rl->vec_bytes);
    rc = PREP(vt, &st, "INSERT INTO \"%w\".\"%w_vectors\"(id, data) VALUES (?1, ?2)");
    for (int64_t j = 0; j < n && rc == SQLITE_OK; j++) {
      vec_encode(c, X + (size_t)perm[j] * dim, vb);
      sqlite3_bind_int64(st, 1, j);
      sqlite3_bind_blob(st, 2, vb, rl->vec_bytes, SQLITE_STATIC);
      if (sqlite3_step(st) != SQLITE_DONE) rc = SQLITE_ERROR;
      sqlite3_reset(st);
    }
    sqlite3_finalize(st); st = 0;
    free(vb);
  }

  /* 6. Codebook (float16 or int8) and entry set, as chunked blobs. */
  size_t cb_vals = (size_t)m * PQ_KSUB * pq.dsub, cbn = cb_bytes(c->cb_type, m, cb_vals);
  uint8_t *cb = (uint8_t *)malloc(cbn);
  cb_store(c->cb_type, m, pq.cent, cb_vals, cb);
  uint8_t *list = NULL; int nlist = 0;
  if (rc == SQLITE_OK) rc = write_chunks(vt, CHUNK_ID_CODEBOOK, cb, (int64_t)cbn, &list, &nlist);
  if (rc == SQLITE_OK) rc = CFG_BLOB(vt, "codebook_chunks", list, nlist);
  free(list); list = NULL; free(cb);
  if (pq.rot) {
    size_t nrot = (size_t)dim * dim;
    uint8_t *rb = (uint8_t *)malloc(nrot * 2);
    for (size_t j = 0; j < nrot; j++) dn_wr16(rb + 2 * j, dn_f32_to_f16(pq.rot[j]));
    if (rc == SQLITE_OK) rc = write_chunks(vt, CHUNK_ID_ROTATION, rb, (int64_t)nrot * 2, &list, &nlist);
    if (rc == SQLITE_OK) rc = CFG_BLOB(vt, "rotation_chunks", list, nlist);
    free(list); list = NULL; free(rb);
  }

  int ne = c->n_entry < n ? c->n_entry : (int)n;
  if (ne < 1) ne = 1;
  EntryKey *keys = (EntryKey *)malloc(sizeof(EntryKey) * n);
  for (int64_t j = 0; j < n; j++) {
    keys[j].level = h.level[j]; keys[j].h = dn_hash2((uint64_t)c->seed ^ 0xE17, (uint64_t)j); keys[j].i = (uint32_t)j;
  }
  qsort(keys, n, sizeof(EntryKey), cmp_entry_key);
  /* The HNSW entry point (highest level) comes first. */
  uint8_t *ent = (uint8_t *)calloc((size_t)ne, 8 + m);
  for (int e = 0; e < ne; e++) {
    uint32_t o = keys[e].i;
    dn_wr32(ent + (size_t)e * (8 + m), inv[o]);
    memcpy(ent + (size_t)e * (8 + m) + 8, codes + (size_t)o * m, m);
  }
  free(keys);
  if (rc == SQLITE_OK) rc = write_chunks(vt, CHUNK_ID_ENTRIES, ent, (int64_t)ne * (8 + m), &list, &nlist);
  if (rc == SQLITE_OK) rc = CFG_BLOB(vt, "entry_chunks", list, nlist);
  free(list);

  /* 7. State. The in-memory head is replaced directly. */
  if (rc == SQLITE_OK) {
    vt->built = 1; vt->n_nodes = n; vt->n_deleted = 0; vt->next_id = (uint32_t)n;
    vt->hints_valid = 0; vt->n_entry = ne;
    dnpq_free(&vt->pq); vt->pq = pq; memset(&pq, 0, sizeof pq);
    free(vt->entries); vt->entries = ent; ent = NULL;
    vt->loaded = 1;
    double t_write = now_ms();
    rc = save_state(vt);
    if (rc == SQLITE_OK) rc = CFG_DBL(vt, "build_ms_load", t_load - t0);
    if (rc == SQLITE_OK) rc = CFG_DBL(vt, "build_ms_pq", t_pq - t_load);
    if (rc == SQLITE_OK) rc = CFG_DBL(vt, "build_ms_graph", t_graph - t_pq);
    if (rc == SQLITE_OK) rc = CFG_DBL(vt, "build_ms_write", t_write - t_graph);
    if (rc == SQLITE_OK) rc = CFG_DBL(vt, "avg_degree", (double)deg_sum / n);
    if (rc == SQLITE_OK) rc = CFG_INT(vt, "rows_per_page_hint", per_page);
    if (c->verbose) fprintf(stderr, "dense_ann: wrote index in %.1fs (avg degree %.1f)\n", (t_write - t_graph) / 1e3, (double)deg_sum / n);
  }
  free(ent); free(row); free(perm); free(inv);

done:
  dnhnsw_free(&h);
  dnpq_free(&pq);
  free(codes); free(X); free(rowids);
  if (rc && !vt->base.zErrMsg) set_err(vt, "dense_ann: build failed: %s", sqlite3_errmsg(vt->db));
  return rc;
}

/* ------------------------------------------------------------ finalize */

typedef struct { uint32_t *pages; uint32_t n; int64_t overflow, count; } WalkMap;
static void walk_map_cb(void *ctx, int64_t rowid, uint32_t pgno, int overflow) {
  WalkMap *w = (WalkMap *)ctx;
  w->count++;
  if (overflow) { w->overflow++; return; }
  if (rowid >= 0 && (uint64_t)rowid < w->n) w->pages[rowid] = pgno;
}

typedef struct { uint8_t *lists[NHEAD]; int n[NHEAD]; int64_t overflow; } BlobMap;
static void walk_blob_cb(void *ctx, int64_t rowid, uint32_t pgno, int overflow) {
  BlobMap *b = (BlobMap *)ctx;
  if (overflow) { b->overflow++; return; }
  for (int l = 0; l < NHEAD; l++)
    for (int i = 0; i < b->n[l]; i++)
      if (dn_rd32(b->lists[l] + 8 * i) == (uint32_t)rowid) dn_wr32(b->lists[l] + 8 * i + 4, pgno);
}

static int table_root(DenseVtab *vt, const char *suffix, uint32_t *root);
static int update_chunks(DenseVtab *vt, int64_t first_id, const uint8_t *data, int64_t len);
static int finalize_ivf(DenseVtab *vt);

/* Record the pages of all head chunks in their config lists. */
static int finalize_blob_lists(DenseVtab *vt, int rc) {
  BlobMap bm; memset(&bm, 0, sizeof bm);
  for (int h = 0; h < NHEAD && rc == SQLITE_OK; h++) {
    rc = cfg_get_blob(vt, HEAD_LISTS[h], &bm.lists[h], &bm.n[h]);
    bm.n[h] /= 8;
  }
  uint32_t broot;
  if (rc == SQLITE_OK) rc = table_root(vt, "blobs", &broot);
  if (rc == SQLITE_OK) rc = dnpr_walk_table(&vt->pr, broot, walk_blob_cb, &bm);
  for (int h = 0; h < NHEAD && rc == SQLITE_OK; h++)
    if (bm.n[h]) rc = CFG_BLOB(vt, HEAD_LISTS[h], bm.lists[h], bm.n[h] * 8);
  for (int h = 0; h < NHEAD; h++) free(bm.lists[h]);
  return rc;
}

static int table_root(DenseVtab *vt, const char *suffix, uint32_t *root) {
  sqlite3_stmt *st = 0;
  char *name = sqlite3_mprintf("%s_%s", vt->zName, suffix);
  int rc = prep_fmt(vt->db, &st, "SELECT rootpage FROM \"%w\".sqlite_master WHERE type = 'table' AND name = ?1", vt->zDb);
  *root = 0;
  if (rc == SQLITE_OK) {
    sqlite3_bind_text(st, 1, name, -1, SQLITE_STATIC);
    if (sqlite3_step(st) == SQLITE_ROW) *root = (uint32_t)sqlite3_column_int64(st, 0);
    sqlite3_finalize(st);
  }
  sqlite3_free(name);
  return *root ? SQLITE_OK : SQLITE_ERROR;
}

static int walk_ids(DenseVtab *vt, const char *suffix, WalkMap *w) {
  uint32_t root;
  int rc = table_root(vt, suffix, &root);
  if (rc == SQLITE_OK) rc = dnpr_walk_table(&vt->pr, root, walk_map_cb, w);
  return rc;
}

/* Record, next to every node reference, the page holding the referenced row.
** Reads the committed file directly, so it must run in autocommit mode on a
** rollback-journal database, after any VACUUM / page_size change. Only
** same-size in-place row updates follow, which do not move rows. */
static int do_finalize(DenseVtab *vt) {
  const DenseCfg *c = &vt->cfg;
  const RowLayout *rl = &vt->rl;
  int rc;
  if (!vt->built) { set_err(vt, "dense_ann: finalize: index not built"); return SQLITE_ERROR; }
  if (!sqlite3_get_autocommit(vt->db)) {
    set_err(vt, "dense_ann: finalize must run outside an explicit transaction"); return SQLITE_ERROR;
  }
  vt->pr_state = 0;
  if (!ensure_pr(vt) || vt->pr.wal) {
    set_err(vt, "dense_ann: finalize needs direct page access on a rollback-journal database (not WAL)");
    return SQLITE_ERROR;
  }
  rc = ensure_loaded(vt, 0, NULL);
  if (rc) return rc;

  if (c->layout == LAYOUT_IVF) {
    rc = finalize_ivf(vt);
    if (rc == SQLITE_OK) {
      vt->hints_valid = 1;
      rc = save_state(vt);
      if (rc == SQLITE_OK) rc = CFG_INT(vt, "finalize_page_size", vt->pr.pgsz);
    }
    vt->loaded = 0;
    if (rc && !vt->base.zErrMsg) set_err(vt, "dense_ann: finalize failed: %s", sqlite3_errmsg(vt->db));
    return rc;
  }

  uint32_t n = vt->next_id;
  WalkMap wn = { (uint32_t *)calloc(n ? n : 1, 4), n, 0, 0 };
  WalkMap wc = { NULL, n, 0, 0 }, wv = { NULL, n, 0, 0 };
  rc = walk_ids(vt, "nodes", &wn);
  if (rc == SQLITE_OK && c->layout == LAYOUT_SEPARATE) {
    wc.pages = (uint32_t *)calloc(n ? n : 1, 4);
    rc = walk_ids(vt, "codes", &wc);
  }
  if (rc == SQLITE_OK && c->vtype != VT_NONE && !c->vinline) {
    wv.pages = (uint32_t *)calloc(n ? n : 1, 4);
    rc = walk_ids(vt, "vectors", &wv);
  }
  if (rc == SQLITE_OK && wn.overflow) {
    set_err(vt, "dense_ann: finalize: %lld node rows (%d bytes) need overflow pages at page size %d; "
            "use a larger page size or smaller M / vectors=table", (long long)wn.overflow, rl->size, vt->pr.pgsz);
    rc = SQLITE_ERROR;
  }

  /* Rewrite node rows in id batches with their hints filled in. */
  sqlite3_stmt *sel = 0, *upd = 0;
  if (rc == SQLITE_OK) rc = PREP(vt, &sel, "SELECT id, data FROM \"%w\".\"%w_nodes\" WHERE id >= ?1 AND id < ?2");
  if (rc == SQLITE_OK) rc = PREP(vt, &upd, "UPDATE \"%w\".\"%w_nodes\" SET data = ?2 WHERE id = ?1");
  const int B = 4096;
  uint8_t *batch = (uint8_t *)malloc((size_t)B * rl->size);
  int64_t *bid = (int64_t *)malloc(sizeof(int64_t) * B);
  for (int64_t lo = 0; lo < n && rc == SQLITE_OK; lo += B) {
    int nb = 0;
    sqlite3_bind_int64(sel, 1, lo);
    sqlite3_bind_int64(sel, 2, lo + B);
    while (sqlite3_step(sel) == SQLITE_ROW) {
      if (sqlite3_column_bytes(sel, 1) != rl->size) continue;
      bid[nb] = sqlite3_column_int64(sel, 0);
      memcpy(batch + (size_t)nb * rl->size, sqlite3_column_blob(sel, 1), rl->size);
      nb++;
    }
    sqlite3_reset(sel);
    for (int b = 0; b < nb && rc == SQLITE_OK; b++) {
      uint8_t *row = batch + (size_t)b * rl->size;
      int deg = dn_rd16(row + 2);
      if (wv.pages && bid[b] < n) dn_wr32(row + 4, wv.pages[bid[b]]);
      for (int j = 0; j < deg && j < c->M0; j++) {
        uint8_t *e = row + rl->off_nbr + j * rl->nbr_stride;
        uint32_t nid = dn_rd32(e);
        dn_wr32(e + 4, nid < n ? wn.pages[nid] : 0);
        if (wc.pages) dn_wr32(e + 8, nid < n ? wc.pages[nid] : 0);
      }
      sqlite3_bind_int64(upd, 1, bid[b]);
      sqlite3_bind_blob(upd, 2, row, rl->size, SQLITE_STATIC);
      if (sqlite3_step(upd) != SQLITE_DONE) rc = SQLITE_ERROR;
      sqlite3_reset(upd);
    }
  }
  sqlite3_finalize(sel); sqlite3_finalize(upd);
  free(batch); free(bid);

  /* Entry set: patch page hints and rewrite its chunks (same sizes). */
  uint8_t *list = NULL; int nlist = 0;
  if (rc == SQLITE_OK) {
    int estride = 8 + c->m;
    for (int e = 0; e < vt->n_entry; e++) {
      uint8_t *p = vt->entries + (size_t)e * estride;
      uint32_t id = dn_rd32(p);
      dn_wr32(p + 4, id < n ? wn.pages[id] : 0);
    }
    rc = update_chunks(vt, CHUNK_ID_ENTRIES, vt->entries, (int64_t)vt->n_entry * estride);
    (void)list; (void)nlist;
  }
  /* Blob chunk pages (every part of the head) go into the config. */
  rc = finalize_blob_lists(vt, rc);

  if (rc == SQLITE_OK) {
    vt->hints_valid = 1;
    rc = save_state(vt);
    if (rc == SQLITE_OK) rc = CFG_INT(vt, "finalize_page_size", vt->pr.pgsz);
  }
  free(wn.pages); free(wc.pages); free(wv.pages);
  vt->loaded = 0;   /* reload with hints on next use */
  if (rc && !vt->base.zErrMsg) set_err(vt, "dense_ann: finalize failed: %s", sqlite3_errmsg(vt->db));
  return rc;
}

/* -------------------------------------------- incremental insert / delete */

typedef struct { const float *buf; int dim; } IdxVec;
static const float *idx_vec(void *ctx, uint32_t i) {
  const IdxVec *v = (const IdxVec *)ctx;
  return v->buf + (size_t)i * v->dim;
}

static int row_exists(DenseVtab *vt, const char *fmt, int64_t key) {
  sqlite3_stmt *st = 0;
  int found = 0;
  if (PREP(vt, &st, fmt) == SQLITE_OK) {
    sqlite3_bind_int64(st, 1, key);
    found = sqlite3_step(st) == SQLITE_ROW;
  }
  sqlite3_finalize(st);
  return found;
}

/* Insert one vector into a built index: encode with the existing codebook,
** find neighbours with the same PQ beam search used by queries, link both
** ways, pruning full neighbour lists with the HNSW heuristic on PQ-decoded
** vectors. New rows have no page hints until the next 'finalize'. */
static int insert_built(DenseVtab *vt, int64_t rowid, const float *x) {
  const DenseCfg *c = &vt->cfg;
  const RowLayout *rl = &vt->rl;
  const int m = c->m, dim = c->dim, M0 = c->M0;
  int rc = ensure_loaded(vt, 0, NULL);
  if (rc) return rc;
  if (row_exists(vt, "SELECT 1 FROM \"%w\".\"%w_rowids\" WHERE rowid = ?1", rowid)) {
    set_err(vt, "dense_ann: rowid %lld already exists", (long long)rowid);
    return SQLITE_CONSTRAINT;
  }
  uint8_t *code = (uint8_t *)malloc(m);
  dnpq_encode(&vt->pq, x, code);
  float *tab = (float *)malloc(sizeof(float) * m * PQ_KSUB);
  dnpq_adc_table(&vt->pq, x, c->metric, tab);

  QStats st; memset(&st, 0, sizeof st);
  QCtx qc; qc_init(&qc, vt, 0, &st);
  int ef = c->efc > M0 ? c->efc : M0, nL = 0;
  Cand *L = (Cand *)calloc(ef, sizeof(Cand));
  rc = ann_core(vt, &qc, tab, NULL, ef, 4, L, &nL, NULL, NULL);

  /* Candidates: live expanded nodes, sorted by PQ distance. */
  int nc = 0;
  uint32_t *cidx = (uint32_t *)malloc(sizeof(uint32_t) * (nL + 1));
  float *cd = (float *)malloc(sizeof(float) * (nL + 1));
  float *cv = (float *)malloc(sizeof(float) * (size_t)(nL + 1) * dim);
  int *cpos = (int *)malloc(sizeof(int) * (nL + 1));
  for (int i = 0; i < nL; i++) {
    if (!L[i].row || (L[i].row[0] & NODE_FLAG_DELETED)) continue;
    dnpq_decode(&vt->pq, L[i].row + rl->off_code, cv + (size_t)nc * dim);
    cidx[nc] = (uint32_t)nc; cd[nc] = L[i].d; cpos[nc] = i;
    nc++;
  }
  uint32_t *sel = (uint32_t *)malloc(sizeof(uint32_t) * M0);
  IdxVec iv = { cv, dim };
  int nsel = dnhnsw_select(c->metric, dim, (float)c->alpha, cidx, cd, nc, M0, idx_vec, &iv, sel);

  uint32_t newid = vt->next_id;
  uint8_t *row = (uint8_t *)calloc(1, rl->size);
  dn_wr16(row + 2, (uint16_t)nsel);
  dn_wr64(row + 8, rowid);
  memcpy(row + rl->off_code, code, m);
  if (c->vinline) vec_encode(c, x, row + rl->off_vec);
  for (int j = 0; j < nsel; j++) {
    const Cand *cn = &L[cpos[sel[j]]];
    uint8_t *e = row + rl->off_nbr + j * rl->nbr_stride;
    dn_wr32(e, cn->id); dn_wr32(e + 4, cn->page);
    if (c->layout == LAYOUT_COLOCATED) memcpy(row + rl->off_nbrcode + j * m, cn->row + rl->off_code, m);
  }
  sqlite3_stmt *s = 0;
  if (rc == SQLITE_OK) rc = PREP(vt, &s, "INSERT INTO \"%w\".\"%w_nodes\"(id, data) VALUES (?1, ?2)");
  if (rc == SQLITE_OK) {
    sqlite3_bind_int64(s, 1, newid);
    sqlite3_bind_blob(s, 2, row, rl->size, SQLITE_STATIC);
    rc = sqlite3_step(s) == SQLITE_DONE ? SQLITE_OK : SQLITE_ERROR;
  }
  sqlite3_finalize(s); s = 0;
  if (rc == SQLITE_OK) rc = PREP(vt, &s, "INSERT INTO \"%w\".\"%w_rowids\"(rowid, node) VALUES (?1, ?2)");
  if (rc == SQLITE_OK) {
    sqlite3_bind_int64(s, 1, rowid); sqlite3_bind_int64(s, 2, newid);
    rc = sqlite3_step(s) == SQLITE_DONE ? SQLITE_OK : SQLITE_ERROR;
  }
  sqlite3_finalize(s); s = 0;
  if (rc == SQLITE_OK && c->layout == LAYOUT_SEPARATE) {
    rc = PREP(vt, &s, "INSERT INTO \"%w\".\"%w_codes\"(id, data) VALUES (?1, ?2)");
    if (rc == SQLITE_OK) {
      sqlite3_bind_int64(s, 1, newid); sqlite3_bind_blob(s, 2, code, m, SQLITE_STATIC);
      rc = sqlite3_step(s) == SQLITE_DONE ? SQLITE_OK : SQLITE_ERROR;
    }
    sqlite3_finalize(s); s = 0;
  }
  if (rc == SQLITE_OK && c->vtype != VT_NONE && !c->vinline) {
    uint8_t *vb = (uint8_t *)malloc(rl->vec_bytes);
    vec_encode(c, x, vb);
    rc = PREP(vt, &s, "INSERT INTO \"%w\".\"%w_vectors\"(id, data) VALUES (?1, ?2)");
    if (rc == SQLITE_OK) {
      sqlite3_bind_int64(s, 1, newid); sqlite3_bind_blob(s, 2, vb, rl->vec_bytes, SQLITE_STATIC);
      rc = sqlite3_step(s) == SQLITE_DONE ? SQLITE_OK : SQLITE_ERROR;
    }
    sqlite3_finalize(s); s = 0;
    free(vb);
  }

  /* Back-links. Rows are rewritten at the same size, so they stay in place. */
  if (rc == SQLITE_OK) rc = PREP(vt, &s, "UPDATE \"%w\".\"%w_nodes\" SET data = ?2 WHERE id = ?1");
  uint8_t *nrow = (uint8_t *)malloc(rl->size);
  float *nv = (float *)malloc(sizeof(float) * (size_t)(M0 + 1) * dim);
  float *base = (float *)malloc(sizeof(float) * dim);
  for (int j = 0; j < nsel && rc == SQLITE_OK; j++) {
    const Cand *cn = &L[cpos[sel[j]]];
    memcpy(nrow, cn->row, rl->size);
    int deg = dn_rd16(nrow + 2);
    if (deg < M0) {
      uint8_t *e = nrow + rl->off_nbr + deg * rl->nbr_stride;
      memset(e, 0, rl->nbr_stride);
      dn_wr32(e, newid);
      if (c->layout == LAYOUT_COLOCATED) memcpy(nrow + rl->off_nbrcode + deg * m, code, m);
      dn_wr16(nrow + 2, (uint16_t)(deg + 1));
    } else {
      /* Prune deg+1 candidates back to M0. Neighbour vectors are decoded
      ** from PQ codes (co-located, or fetched from _codes). */
      uint8_t *ncodes = (uint8_t *)malloc((size_t)(deg + 1) * m);
      if (c->layout == LAYOUT_COLOCATED) {
        memcpy(ncodes, nrow + rl->off_nbrcode, (size_t)deg * m);
      } else {
        RowReq *cr = (RowReq *)malloc(sizeof(RowReq) * deg);
        for (int t = 0; t < deg; t++) { cr[t].id = dn_rd32(nrow + rl->off_nbr + t * rl->nbr_stride); cr[t].page = 0; }
        rc = q_fetch(&qc, T_CODES, cr, deg);
        for (int t = 0; t < deg; t++) {
          if (cr[t].data && cr[t].len >= m) memcpy(ncodes + (size_t)t * m, cr[t].data, m);
          else memset(ncodes + (size_t)t * m, 0, m);
        }
        free(cr);
      }
      memcpy(ncodes + (size_t)deg * m, code, m);
      dnpq_decode(&vt->pq, nrow + rl->off_code, base);
      /* Sort candidate slots by distance to the base node. */
      int ncand = deg + 1;
      Result *order = (Result *)malloc(sizeof(Result) * ncand);
      for (int t = 0; t < ncand; t++) {
        dnpq_decode(&vt->pq, ncodes + (size_t)t * m, nv + (size_t)t * dim);
        order[t].rowid = t;
        order[t].d = dn_distance(c->metric, base, nv + (size_t)t * dim, dim);
      }
      qsort(order, ncand, sizeof(Result), cmp_result);
      uint32_t *oid = (uint32_t *)malloc(sizeof(uint32_t) * ncand);
      float *od = (float *)malloc(sizeof(float) * ncand);
      for (int t = 0; t < ncand; t++) { oid[t] = (uint32_t)order[t].rowid; od[t] = order[t].d; }
      uint32_t *keep = (uint32_t *)malloc(sizeof(uint32_t) * M0);
      IdxVec niv = { nv, dim };
      int nkeep = dnhnsw_select(c->metric, dim, (float)c->alpha, oid, od, ncand, M0, idx_vec, &niv, keep);
      /* Rebuild the entry arrays from the kept slots. */
      uint8_t *old = (uint8_t *)malloc(rl->size);
      memcpy(old, nrow, rl->size);
      memset(nrow + rl->off_nbr, 0, rl->size - rl->off_nbr);
      for (int t = 0; t < nkeep; t++) {
        int slot = (int)keep[t];
        uint8_t *e = nrow + rl->off_nbr + t * rl->nbr_stride;
        if (slot < deg) memcpy(e, old + rl->off_nbr + slot * rl->nbr_stride, rl->nbr_stride);
        else dn_wr32(e, newid);
        if (c->layout == LAYOUT_COLOCATED) memcpy(nrow + rl->off_nbrcode + t * m, ncodes + (size_t)slot * m, m);
      }
      dn_wr16(nrow + 2, (uint16_t)nkeep);
      free(old); free(keep); free(oid); free(od); free(order); free(ncodes);
    }
    if (rc == SQLITE_OK) {
      sqlite3_bind_int64(s, 1, cn->id);
      sqlite3_bind_blob(s, 2, nrow, rl->size, SQLITE_STATIC);
      rc = sqlite3_step(s) == SQLITE_DONE ? SQLITE_OK : SQLITE_ERROR;
      sqlite3_reset(s);
    }
  }
  sqlite3_finalize(s);
  free(nrow); free(nv); free(base);

  if (rc == SQLITE_OK) {
    vt->next_id++;
    vt->n_nodes++;
    rc = save_state(vt);
  }
  qc_free(&qc);
  free(row); free(sel); free(cidx); free(cd); free(cv); free(cpos); free(L); free(tab); free(code);
  if (rc && !vt->base.zErrMsg) set_err(vt, "dense_ann: insert failed: %s", sqlite3_errmsg(vt->db));
  return rc;
}

/* Delete: before build, drop from the buffer. After build, the node stays in
** the graph as a routing node, flagged deleted, and leaves the rowid map. */
static int delete_ivf(DenseVtab *vt, int64_t rowid);
static int delete_row(DenseVtab *vt, int64_t rowid) {
  if (!vt->built)
    return exec_fmt(vt->db, "DELETE FROM \"%w\".\"%w_buffer\" WHERE rowid = %lld", vt->zDb, vt->zName, (long long)rowid);
  if (vt->cfg.layout == LAYOUT_IVF) return delete_ivf(vt, rowid);
  sqlite3_stmt *s = 0;
  int64_t node = -1;
  int rc = PREP(vt, &s, "SELECT node FROM \"%w\".\"%w_rowids\" WHERE rowid = ?1");
  if (rc) return rc;
  sqlite3_bind_int64(s, 1, rowid);
  if (sqlite3_step(s) == SQLITE_ROW) node = sqlite3_column_int64(s, 0);
  sqlite3_finalize(s); s = 0;
  if (node < 0) return SQLITE_OK;
  /* Set the deleted flag with a same-size rewrite of the node row. */
  QStats st; memset(&st, 0, sizeof st);
  QCtx qc; qc_init(&qc, vt, 0, &st);
  RowReq r = { (uint32_t)node, 0, NULL, 0 };
  rc = q_fetch(&qc, T_NODES, &r, 1);
  if (rc == SQLITE_OK && r.data && r.len == vt->rl.size) {
    uint8_t *row = (uint8_t *)malloc(r.len);
    memcpy(row, r.data, r.len);
    row[0] |= NODE_FLAG_DELETED;
    rc = PREP(vt, &s, "UPDATE \"%w\".\"%w_nodes\" SET data = ?2 WHERE id = ?1");
    if (rc == SQLITE_OK) {
      sqlite3_bind_int64(s, 1, node);
      sqlite3_bind_blob(s, 2, row, r.len, SQLITE_STATIC);
      rc = sqlite3_step(s) == SQLITE_DONE ? SQLITE_OK : SQLITE_ERROR;
    }
    sqlite3_finalize(s);
    free(row);
  }
  qc_free(&qc);
  if (rc == SQLITE_OK)
    rc = exec_fmt(vt->db, "DELETE FROM \"%w\".\"%w_rowids\" WHERE rowid = %lld", vt->zDb, vt->zName, (long long)rowid);
  if (rc == SQLITE_OK) { vt->n_nodes--; vt->n_deleted++; rc = save_state(vt); }
  return rc;
}

/* ================================================================== IVF
**
** layout=ivf: an IVF-PQ index (IVFADC). The coarse centroids and a list
** directory form the cached head, loaded once with the PQ codebook. Each
** list is stored as consecutive rows of <name>_lists keyed (list << 20 | j),
** each row ("chunk") at most IVF_CHUNK_BYTES so it never overflows a 4 KiB
** page. Because the lists are written list by list, each occupies a
** contiguous page range; the directory records it, so probing nprobe lists
** is one round of nprobe range requests. The top rerank_k candidates are then
** rescored with stored vectors (a second round).
**
** Chunk: u16 count | u16 reserved | u32 first slot | u32 first vector page
**        count x { u32 rowid | u8 page delta | PQ code (m bytes) }
** Entry i refers to vector row (slot) first_slot + i, stored on page
** first_vector_page + delta (delta 0xFF = unknown, 0xFE = deleted).
** Directory entry per list: u32 nchunks | u32 first page | u32 last page.
** The rowid map stores node = (list << 32) | slot.
*/
#define IVF_CHUNK_BYTES 4000
#define IVF_CHUNK_HDR 12
#define IVF_LIST_SHIFT 20
#define IVF_DELTA_UNKNOWN 0xFF
#define IVF_DELTA_DELETED 0xFE

static int ivf_entry_bytes(const DenseCfg *c) { return 5 + c->m; }
static int ivf_chunk_cap(const DenseCfg *c) { return (IVF_CHUNK_BYTES - IVF_CHUNK_HDR) / ivf_entry_bytes(c); }

static size_t ivf_cent_bytes(const DenseCfg *c) {
  if (c->ivf_cent == CENT_PQ) return (size_t)c->m;
  if (c->ivf_cent == CENT_INT8) return (size_t)vec_bytes_for(VT_INT8, c->dim);
  return (size_t)vec_bytes_for(VT_F16, c->dim);
}

static DenseCfg ivf_cent_cfg(const DenseCfg *c) {
  DenseCfg tc = *c;
  tc.vtype = c->ivf_cent == CENT_INT8 ? VT_INT8 : VT_F16;
  return tc;
}

/* Decode centroids from the stored head into vt->ivf_cent / ivf_cn. */
static int ivf_decode_head(DenseVtab *vt) {
  const DenseCfg *c = &vt->cfg;
  int nl = vt->nlist, dim = c->dim;
  size_t cb = ivf_cent_bytes(c);
  DenseCfg tc = ivf_cent_cfg(c);
  free(vt->ivf_cent); free(vt->ivf_cn);
  vt->ivf_cent = (float *)malloc(sizeof(float) * (size_t)nl * dim);
  vt->ivf_cn = (float *)malloc(sizeof(float) * (nl ? nl : 1));
  if (!vt->ivf_cent || !vt->ivf_cn) return SQLITE_NOMEM;
  for (int j = 0; j < nl; j++) {
    const uint8_t *src = vt->ivf_head + (size_t)j * cb;
    float *dst = vt->ivf_cent + (size_t)j * dim;
    if (c->ivf_cent == CENT_PQ) dnpq_decode(&vt->ivf_cpq, src, dst);
    else vec_decode(&tc, src, dst);
    vt->ivf_cn[j] = dn_dot(dst, dst, dim);
  }
  vt->ivf_dir = vt->ivf_head + (size_t)nl * cb;
  return SQLITE_OK;
}

static int ivf_head_len(const DenseVtab *vt) { return (int)((size_t)vt->nlist * (ivf_cent_bytes(&vt->cfg) + 12)); }

/* Overwrite existing chunk rows in place (same sizes as when written). */
static int update_chunks(DenseVtab *vt, int64_t first_id, const uint8_t *data, int64_t len) {
  sqlite3_stmt *st = 0;
  int rc = PREP(vt, &st, "UPDATE \"%w\".\"%w_blobs\" SET data = ?2 WHERE id = ?1");
  int nchunks = (int)((len + CHUNK_BYTES - 1) / CHUNK_BYTES);
  for (int i = 0; i < nchunks && rc == SQLITE_OK; i++) {
    int64_t off = (int64_t)i * CHUNK_BYTES;
    int sz = (int)(len - off < CHUNK_BYTES ? len - off : CHUNK_BYTES);
    sqlite3_bind_int64(st, 1, first_id + i);
    sqlite3_bind_blob(st, 2, data + off, sz, SQLITE_STATIC);
    rc = sqlite3_step(st) == SQLITE_DONE ? SQLITE_OK : SQLITE_ERROR;
    sqlite3_reset(st);
  }
  sqlite3_finalize(st);
  return rc;
}

static int write_named_chunks(DenseVtab *vt, const char *key, int64_t first_id, const uint8_t *data, int64_t len) {
  uint8_t *list = NULL; int nlist = 0;
  int rc = write_chunks(vt, first_id, data, len, &list, &nlist);
  if (rc == SQLITE_OK) rc = CFG_BLOB(vt, key, list, nlist);
  free(list);
  return rc;
}

static int build_ivf(DenseVtab *vt, const float *X, const int64_t *rowids, int64_t n, double t0, double t_load) {
  DenseCfg *c = &vt->cfg;
  const int dim = c->dim, m = c->m;
  int rc = SQLITE_OK;
  sqlite3_stmt *st = 0;
  for (int64_t i = 0; i < n; i++)
    if (rowids[i] < 0 || rowids[i] > 0xFFFFFFFFLL) { set_err(vt, "dense_ann: layout=ivf needs rowids in [0, 2^32)"); return SQLITE_ERROR; }
  int nlist = c->nlist > 0 ? c->nlist : (int)(4 * sqrt((double)n));
  if (nlist < 1) nlist = 1;
  if (nlist > n) nlist = (int)n;
  if (c->ivf_cent == CENT_AUTO) {       /* int8 is lossless in practice; PQ pays off from ~1,000 lists */
    c->ivf_cent = nlist >= 1024 ? CENT_PQ : CENT_INT8;
    rc = CFG_INT(vt, "ivf_centroids", c->ivf_cent);
    if (rc) return rc;
  }

  /* 1. Coarse k-means (10 iterations on a sample of 40 points per list, at least 65,536). */
  float *C = (float *)malloc(sizeof(float) * (size_t)nlist * dim);
  int64_t sample = (int64_t)nlist * 40 > 65536 ? (int64_t)nlist * 40 : 65536;
  dnivf_kmeans(X, n, dim, nlist, sample, 10, (uint64_t)c->seed, c->nthreads, c->verbose, C);

  /* 2. Store centroids in the head format and use the decoded ones from here on. */
  size_t cb = ivf_cent_bytes(c);
  uint8_t *head = (uint8_t *)calloc((size_t)nlist * (cb + 12), 1);
  PQ cpq; memset(&cpq, 0, sizeof cpq);
  if (c->ivf_cent == CENT_PQ) {
    dnpq_train(&cpq, C, nlist, dim, m, nlist, 25, 0, (uint64_t)c->seed ^ 0xC0A5, c->nthreads);
    cb_round(c->cb_type, &cpq);
    dnpq_encode_many(&cpq, C, nlist, head, c->nthreads);   /* codes are contiguous: stride m = cb */
  } else {
    DenseCfg tc = ivf_cent_cfg(c);
    for (int j = 0; j < nlist; j++) vec_encode(&tc, C + (size_t)j * dim, head + (size_t)j * cb);
  }
  free(C);
  vt->nlist = nlist;
  vt->ivf_head = head;
  dnpq_free(&vt->ivf_cpq); vt->ivf_cpq = cpq;
  rc = ivf_decode_head(vt);
  double t_coarse = now_ms();
  if (c->verbose) fprintf(stderr, "dense_ann: coarse quantiser (%d lists) in %.1fs\n", nlist, (t_coarse - t_load) / 1e3);

  /* 3. Assign, form residuals, train PQ on them and encode. */
  int32_t *assign = (int32_t *)malloc(sizeof(int32_t) * n);
  dnivf_assign(X, n, dim, vt->ivf_cent, nlist, c->nthreads, assign);
  const float *Rx = X;
  float *Rbuf = NULL;
  if (c->ivf_residual) {
    Rbuf = (float *)malloc(sizeof(float) * (size_t)n * dim);
    for (int64_t i = 0; i < n; i++) {
      const float *x = X + (size_t)i * dim, *cc = vt->ivf_cent + (size_t)assign[i] * dim;
      float *r = Rbuf + (size_t)i * dim;
      for (int t = 0; t < dim; t++) r[t] = x[t] - cc[t];
    }
    Rx = Rbuf;
  }
  PQ pq;
  if (dnpq_train(&pq, Rx, n, dim, m, c->train_max, c->iters, c->opq, (uint64_t)c->seed, c->nthreads)) {
    free(assign); free(Rbuf); return SQLITE_NOMEM;
  }
  cb_round(c->cb_type, &pq);
  uint8_t *codes = (uint8_t *)malloc((size_t)n * m);
  dnpq_encode_many(&pq, Rx, n, codes, c->nthreads);
  free(Rbuf);
  double t_pq = now_ms();
  if (c->verbose) fprintf(stderr, "dense_ann: pq trained+encoded in %.1fs\n", (t_pq - t_coarse) / 1e3);

  /* 4. Order by (list, rowid): slots are positions in that order. */
  int64_t *start = (int64_t *)calloc((size_t)nlist + 1, sizeof(int64_t));
  for (int64_t i = 0; i < n; i++) start[assign[i] + 1]++;
  for (int j = 0; j < nlist; j++) start[j + 1] += start[j];
  int64_t *fill = (int64_t *)malloc(sizeof(int64_t) * (nlist ? nlist : 1));
  memcpy(fill, start, sizeof(int64_t) * nlist);
  uint32_t *order = (uint32_t *)malloc(sizeof(uint32_t) * n), *slot_of = (uint32_t *)malloc(sizeof(uint32_t) * n);
  for (int64_t i = 0; i < n; i++) { int64_t s = fill[assign[i]]++; order[s] = (uint32_t)i; slot_of[i] = (uint32_t)s; }
  free(fill);

  /* 5. Posting lists, list by list, so each list is a contiguous page range. */
  const int eb = ivf_entry_bytes(c), cap = ivf_chunk_cap(c);
  uint8_t *chunk = (uint8_t *)malloc(IVF_CHUNK_BYTES);
  uint8_t *dir = head + (size_t)nlist * cb;
  int64_t nchunks_total = 0;
  if (rc == SQLITE_OK) rc = PREP(vt, &st, "INSERT INTO \"%w\".\"%w_lists\"(id, data) VALUES (?1, ?2)");
  for (int L = 0; L < nlist && rc == SQLITE_OK; L++) {
    int64_t s0 = start[L], s1 = start[L + 1];
    uint32_t nch = 0;
    for (int64_t s = s0; s < s1 && rc == SQLITE_OK; s += cap, nch++) {
      int cnt = (int)(s1 - s < cap ? s1 - s : cap);
      memset(chunk, 0, IVF_CHUNK_HDR);
      dn_wr16(chunk, (uint16_t)cnt);
      dn_wr32(chunk + 4, (uint32_t)s);
      for (int e = 0; e < cnt; e++) {
        uint32_t o = order[s + e];
        uint8_t *ent = chunk + IVF_CHUNK_HDR + (size_t)e * eb;
        dn_wr32(ent, (uint32_t)rowids[o]);
        ent[4] = IVF_DELTA_UNKNOWN;
        memcpy(ent + 5, codes + (size_t)o * m, m);
      }
      sqlite3_bind_int64(st, 1, ((int64_t)L << IVF_LIST_SHIFT) | nch);
      sqlite3_bind_blob(st, 2, chunk, IVF_CHUNK_HDR + cnt * eb, SQLITE_STATIC);
      if (sqlite3_step(st) != SQLITE_DONE) rc = SQLITE_ERROR;
      sqlite3_reset(st);
    }
    dn_wr32(dir + (size_t)L * 12, nch);
    nchunks_total += nch;
  }
  sqlite3_finalize(st); st = 0;
  free(chunk);

  /* 6. Stored vectors by slot, and the rowid map (in rowid order). */
  if (rc == SQLITE_OK && c->vtype != VT_NONE) {
    uint8_t *vb = (uint8_t *)malloc(vt->rl.vec_bytes);
    rc = PREP(vt, &st, "INSERT INTO \"%w\".\"%w_vectors\"(id, data) VALUES (?1, ?2)");
    for (int64_t s = 0; s < n && rc == SQLITE_OK; s++) {
      vec_encode(c, X + (size_t)order[s] * dim, vb);
      sqlite3_bind_int64(st, 1, s);
      sqlite3_bind_blob(st, 2, vb, vt->rl.vec_bytes, SQLITE_STATIC);
      if (sqlite3_step(st) != SQLITE_DONE) rc = SQLITE_ERROR;
      sqlite3_reset(st);
    }
    sqlite3_finalize(st); st = 0;
    free(vb);
  }
  if (rc == SQLITE_OK) rc = PREP(vt, &st, "INSERT INTO \"%w\".\"%w_rowids\"(rowid, node) VALUES (?1, ?2)");
  for (int64_t i = 0; i < n && rc == SQLITE_OK; i++) {
    sqlite3_bind_int64(st, 1, rowids[i]);
    sqlite3_bind_int64(st, 2, ((int64_t)assign[i] << 32) | slot_of[i]);
    if (sqlite3_step(st) != SQLITE_DONE) rc = SQLITE_ERROR;
    sqlite3_reset(st);
  }
  sqlite3_finalize(st); st = 0;

  /* 7. Head parts: PQ codebook, rotation, centroids + directory, centroid PQ. */
  size_t cb_vals = (size_t)m * PQ_KSUB * pq.dsub, cbn = cb_bytes(c->cb_type, m, cb_vals);
  uint8_t *buf = (uint8_t *)malloc(cb_vals * 2 > (size_t)dim * dim * 2 ? cb_vals * 2 : (size_t)dim * dim * 2);
  cb_store(c->cb_type, m, pq.cent, cb_vals, buf);
  if (rc == SQLITE_OK) rc = write_named_chunks(vt, "codebook_chunks", CHUNK_ID_CODEBOOK, buf, (int64_t)cbn);
  if (rc == SQLITE_OK && pq.rot) {
    for (size_t j = 0; j < (size_t)dim * dim; j++) dn_wr16(buf + 2 * j, dn_f32_to_f16(pq.rot[j]));
    rc = write_named_chunks(vt, "rotation_chunks", CHUNK_ID_ROTATION, buf, (int64_t)dim * dim * 2);
  }
  if (rc == SQLITE_OK && c->ivf_cent == CENT_PQ) {
    cb_store(c->cb_type, m, vt->ivf_cpq.cent, cb_vals, buf);
    rc = write_named_chunks(vt, "ivf_cpq_chunks", CHUNK_ID_IVF_CPQ, buf, (int64_t)cbn);
  }
  free(buf);
  if (rc == SQLITE_OK) rc = write_named_chunks(vt, "ivf_head_chunks", CHUNK_ID_IVF_HEAD, head, ivf_head_len(vt));
  if (rc == SQLITE_OK) rc = CFG_BLOB(vt, "entry_chunks", "", 0);

  if (rc == SQLITE_OK) {
    vt->built = 1; vt->n_nodes = n; vt->n_deleted = 0; vt->next_id = (uint32_t)n;
    vt->hints_valid = 0; vt->n_entry = 0;
    dnpq_free(&vt->pq); vt->pq = pq; memset(&pq, 0, sizeof pq);
    free(vt->entries); vt->entries = NULL;
    vt->loaded = 1;
    double t_write = now_ms();
    rc = save_state(vt);
    if (rc == SQLITE_OK) rc = CFG_INT(vt, "ivf_nlist", nlist);
    if (rc == SQLITE_OK) rc = CFG_INT(vt, "ivf_chunks", nchunks_total);
    if (rc == SQLITE_OK) rc = CFG_INT(vt, "ivf_head_bytes", ivf_head_len(vt));
    if (rc == SQLITE_OK) rc = CFG_DBL(vt, "build_ms_load", t_load - t0);
    if (rc == SQLITE_OK) rc = CFG_DBL(vt, "build_ms_coarse", t_coarse - t_load);
    if (rc == SQLITE_OK) rc = CFG_DBL(vt, "build_ms_pq", t_pq - t_coarse);
    if (rc == SQLITE_OK) rc = CFG_DBL(vt, "build_ms_graph", 0);
    if (rc == SQLITE_OK) rc = CFG_DBL(vt, "build_ms_write", t_write - t_pq);
    if (c->verbose) fprintf(stderr, "dense_ann: wrote ivf index in %.1fs\n", (t_write - t_pq) / 1e3);
  }
  dnpq_free(&pq);
  free(assign); free(codes); free(start); free(order); free(slot_of);
  if (rc && !vt->base.zErrMsg) set_err(vt, "dense_ann: ivf build failed: %s", sqlite3_errmsg(vt->db));
  return rc;
}

typedef struct IvfCand { float d; uint32_t rowid, slot, vpage; } IvfCand;

static void ivfc_insert(IvfCand *R, int *n, int cap, const IvfCand *x) {
  if (*n == cap && x->d >= R[cap - 1].d) return;
  int i = (*n < cap) ? (*n)++ : cap - 1;
  while (i > 0 && R[i - 1].d > x->d) { R[i] = R[i - 1]; i--; }
  R[i] = *x;
}

/* Score every live entry of one chunk into the candidate list. */
static void ivf_scan_chunk(const DenseVtab *vt, const uint8_t *data, int len, const float *tab, float off,
                           IvfCand *R, int *nR, int cap, QStats *st) {
  const int m = vt->cfg.m, eb = ivf_entry_bytes(&vt->cfg);
  if (len < IVF_CHUNK_HDR) return;
  int cnt = dn_rd16(data);
  if (IVF_CHUNK_HDR + cnt * eb > len) cnt = (len - IVF_CHUNK_HDR) / eb;
  uint32_t first_slot = dn_rd32(data + 4), fvp = dn_rd32(data + 8);
  for (int e = 0; e < cnt; e++) {
    const uint8_t *ent = data + IVF_CHUNK_HDR + (size_t)e * eb;
    uint8_t delta = ent[4];
    if (delta == IVF_DELTA_DELETED) continue;
    IvfCand x;
    x.d = dnpq_adc(tab, ent + 5, m) - off;
    st->scanned++;
    if (*nR == cap && x.d >= R[cap - 1].d) continue;
    x.rowid = dn_rd32(ent);
    x.slot = first_slot + (uint32_t)e;
    x.vpage = (fvp && delta < IVF_DELTA_DELETED) ? fvp + delta : 0;
    ivfc_insert(R, nR, cap, &x);
  }
}

/* ADC table and additive offset for list L (see ivf_search). */
static void ivf_list_table(const DenseVtab *vt, const float *q, int L, float *tab, float *off, float *qbuf) {
  const DenseCfg *c = &vt->cfg;
  const float *cent = vt->ivf_cent + (size_t)L * c->dim;
  if (c->metric == METRIC_L2 && c->ivf_residual) {
    for (int t = 0; t < c->dim; t++) qbuf[t] = q[t] - cent[t];
    dnpq_adc_table(&vt->pq, qbuf, c->metric, tab);
    *off = 0;
  } else {
    *off = (c->ivf_residual && c->metric != METRIC_L2) ? dn_dot(q, cent, c->dim) : 0;
  }
}

/* IVF-PQ search. Distances: for cosine/IP with residuals,
**   1 - <q, c + r> = (1 - <q, r>) - <q, c> = ADC(r) - <q, c>,
** so one ADC table serves all lists; for L2 with residuals, a table per list. */
static int ivf_search(DenseVtab *vt, const float *q, int k, int nprobe, int R, int rerank, int raw,
                      Result **pres, int *pn, QStats *st) {
  const DenseCfg *c = &vt->cfg;
  int rc = ensure_loaded(vt, raw, st);
  if (rc) return rc;
  raw = raw && vt->hints_valid && ensure_pr(vt);
  st->raw = raw; st->ann = 1;
  if (nprobe > vt->nlist) nprobe = vt->nlist;
  if (nprobe < 1) nprobe = 1;
  if (R < k) R = k;
  if (c->vtype == VT_NONE || !rerank) R = k;

  int32_t *probe = (int32_t *)malloc(sizeof(int32_t) * nprobe);
  float *pd = (float *)malloc(sizeof(float) * nprobe);
  int np = dnivf_probe(q, c->dim, vt->ivf_cent, vt->ivf_cn, vt->nlist, nprobe, probe, pd);
  st->lists = np;
  st->entry_dist = vt->nlist;

  float *tab = (float *)malloc(sizeof(float) * c->m * PQ_KSUB);
  float *qbuf = (float *)malloc(sizeof(float) * c->dim);
  int per_list_tab = (c->metric == METRIC_L2 && c->ivf_residual);
  if (!per_list_tab) dnpq_adc_table(&vt->pq, q, c->metric, tab);

  QCtx qc; qc_init(&qc, vt, raw, st);
  /* Round 1: every page of every probed list, announced together. */
  if (raw) {
    int npages = 0, cap = 0;
    uint32_t *pages = NULL;
    for (int p = 0; p < np; p++) {
      const uint8_t *d = vt->ivf_dir + (size_t)probe[p] * 12;
      uint32_t first = dn_rd32(d + 4), last = dn_rd32(d + 8);
      if (!first || last < first || last - first > 65536) continue;
      for (uint32_t g = first; g <= last; g++) {
        if (npages == cap) { cap = cap ? 2 * cap : 256; pages = (uint32_t *)realloc(pages, sizeof(uint32_t) * cap); }
        pages[npages++] = g;
      }
    }
    if (qc_load_pages(&qc, pages, npages, &rc) > 0) st->rounds++;
    free(pages);
  }
  IvfCand *cand = (IvfCand *)malloc(sizeof(IvfCand) * R);
  int ncand = 0;
  for (int p = 0; p < np && rc == SQLITE_OK; p++) {
    int L = probe[p];
    const uint8_t *d = vt->ivf_dir + (size_t)L * 12;
    uint32_t nch = dn_rd32(d), first = dn_rd32(d + 4), last = dn_rd32(d + 8), g = first;
    float off;
    ivf_list_table(vt, q, L, tab, &off, qbuf);
    for (uint32_t j = 0; j < nch && rc == SQLITE_OK; j++) {
      int64_t id = ((int64_t)L << IVF_LIST_SHIFT) | j;
      const uint8_t *data = NULL; int len = 0;
      if (raw && first) {
        /* Chunks appear in ascending page order within the list's range. */
        while (g <= last && !(data = qc_row_on_page(&qc, g, id, &len))) g++;
      }
      if (!data) {
        RowReq r = { (uint32_t)id, 0, NULL, 0 };
        rc = q_fetch(&qc, T_LISTS, &r, 1);
        data = r.data; len = r.len;
        if (raw && first) g = first;
      }
      if (data) ivf_scan_chunk(vt, data, len, tab, off, cand, &ncand, R, st);
    }
  }
  st->dist = st->scanned;

  Result *res = (Result *)malloc(sizeof(Result) * (ncand ? ncand : 1));
  for (int i = 0; i < ncand; i++) { res[i].rowid = cand[i].rowid; res[i].d = cand[i].d; }
  if (rc == SQLITE_OK && rerank && c->vtype != VT_NONE && ncand > 0) {
    /* Round 2: the stored vectors of all candidates, in parallel. */
    RowReq *vr = (RowReq *)malloc(sizeof(RowReq) * ncand);
    for (int i = 0; i < ncand; i++) { vr[i].id = cand[i].slot; vr[i].page = cand[i].vpage; }
    int before = st->rounds;
    rc = q_fetch(&qc, T_VECS, vr, ncand);
    st->rerank_rounds += st->rounds - before;
    for (int i = 0; i < ncand && rc == SQLITE_OK; i++) {
      if (!vr[i].data || vr[i].len < vt->rl.vec_bytes) { res[i].d = FLT_MAX; continue; }
      vec_decode(c, vr[i].data, qbuf);
      res[i].d = dn_distance(c->metric, q, qbuf, c->dim);
      st->rerank++;
    }
    qsort(res, ncand, sizeof(Result), cmp_result);
    free(vr);
  }
  int nres = ncand < k ? ncand : k;
  qc_free(&qc);
  free(cand); free(tab); free(qbuf); free(probe); free(pd);
  *pres = res; *pn = nres;
  return rc;
}

/* Brute force over all PQ codes of an IVF index (exact=2). */
static int ivf_exact_pq(DenseVtab *vt, const float *q, int k, Result **pres, int *pn, QStats *st) {
  const DenseCfg *c = &vt->cfg;
  int rc = ensure_loaded(vt, 0, st);
  if (rc) return rc;
  float *tab = (float *)malloc(sizeof(float) * c->m * PQ_KSUB);
  float *qbuf = (float *)malloc(sizeof(float) * c->dim);
  if (!(c->metric == METRIC_L2 && c->ivf_residual)) dnpq_adc_table(&vt->pq, q, c->metric, tab);
  IvfCand *cand = (IvfCand *)malloc(sizeof(IvfCand) * k);
  int ncand = 0, lastL = -1;
  float off = 0;
  sqlite3_stmt *s = 0;
  rc = PREP(vt, &s, "SELECT id, data FROM \"%w\".\"%w_lists\" ORDER BY id");
  while (rc == SQLITE_OK && sqlite3_step(s) == SQLITE_ROW) {
    int L = (int)(sqlite3_column_int64(s, 0) >> IVF_LIST_SHIFT);
    if (L != lastL) { ivf_list_table(vt, q, L, tab, &off, qbuf); lastL = L; }
    ivf_scan_chunk(vt, (const uint8_t *)sqlite3_column_blob(s, 1), sqlite3_column_bytes(s, 1), tab, off, cand, &ncand, k, st);
  }
  if (s) rc = sqlite3_finalize(s);
  Result *res = (Result *)malloc(sizeof(Result) * (k ? k : 1));
  for (int i = 0; i < ncand; i++) { res[i].rowid = cand[i].rowid; res[i].d = cand[i].d; }
  st->dist = st->scanned;
  free(cand); free(tab); free(qbuf);
  *pres = res; *pn = ncand;
  return rc;
}

typedef struct { uint8_t *dir; int nlist; uint32_t *prev; uint8_t *bad; } ListWalk;
static void walk_list_cb(void *ctx, int64_t rowid, uint32_t pgno, int overflow) {
  ListWalk *w = (ListWalk *)ctx;
  int L = (int)(rowid >> IVF_LIST_SHIFT);
  uint32_t j = (uint32_t)(rowid & ((1 << IVF_LIST_SHIFT) - 1));
  if (L < 0 || L >= w->nlist) return;
  uint8_t *d = w->dir + (size_t)L * 12;
  if (overflow) w->bad[L] = 1;
  if (j == 0) dn_wr32(d + 4, pgno);
  else if (pgno < w->prev[L]) w->bad[L] = 1;
  w->prev[L] = pgno;
  dn_wr32(d + 8, pgno);
}

/* finalize for IVF: page range of each list (directory), vector pages of
** each chunk's slots. Walks happen before any write (they read the file). */
static int finalize_ivf(DenseVtab *vt) {
  const DenseCfg *c = &vt->cfg;
  const int eb = ivf_entry_bytes(c);
  uint32_t n = vt->next_id;
  int rc = SQLITE_OK;
  WalkMap wv = { NULL, n, 0, 0 };
  if (c->vtype != VT_NONE) {
    wv.pages = (uint32_t *)calloc(n ? n : 1, 4);
    rc = walk_ids(vt, "vectors", &wv);
  }
  ListWalk lw = { vt->ivf_dir, vt->nlist, (uint32_t *)calloc(vt->nlist, 4), (uint8_t *)calloc(vt->nlist, 1) };
  uint32_t root;
  if (rc == SQLITE_OK) rc = table_root(vt, "lists", &root);
  if (rc == SQLITE_OK) rc = dnpr_walk_table(&vt->pr, root, walk_list_cb, &lw);
  int nbad = 0;
  for (int L = 0; L < vt->nlist; L++)
    if (lw.bad[L]) { nbad++; dn_wr32(vt->ivf_dir + (size_t)L * 12 + 4, 0); dn_wr32(vt->ivf_dir + (size_t)L * 12 + 8, 0); }
  if (rc == SQLITE_OK) rc = finalize_blob_lists(vt, rc);

  /* Patch vector-page hints into the chunks (same-size rewrites). */
  sqlite3_stmt *sel = 0, *upd = 0;
  if (rc == SQLITE_OK) rc = PREP(vt, &sel, "SELECT id, data FROM \"%w\".\"%w_lists\" WHERE id >= ?1 AND id < ?2");
  if (rc == SQLITE_OK) rc = PREP(vt, &upd, "UPDATE \"%w\".\"%w_lists\" SET data = ?2 WHERE id = ?1");
  uint8_t *buf = (uint8_t *)malloc(IVF_CHUNK_BYTES + 64);
  for (int L = 0; L < vt->nlist && rc == SQLITE_OK && wv.pages; L++) {
    sqlite3_bind_int64(sel, 1, (int64_t)L << IVF_LIST_SHIFT);
    sqlite3_bind_int64(sel, 2, (int64_t)(L + 1) << IVF_LIST_SHIFT);
    /* Collect the list's chunks first; the updates happen after the scan. */
    int64_t *ids = NULL; uint8_t **rows = NULL; int *lens = NULL; int nr = 0, capr = 0;
    while (sqlite3_step(sel) == SQLITE_ROW) {
      if (nr == capr) {
        capr = capr ? 2 * capr : 16;
        ids = (int64_t *)realloc(ids, sizeof(int64_t) * capr);
        rows = (uint8_t **)realloc(rows, sizeof(uint8_t *) * capr);
        lens = (int *)realloc(lens, sizeof(int) * capr);
      }
      lens[nr] = sqlite3_column_bytes(sel, 1);
      rows[nr] = (uint8_t *)malloc(lens[nr] ? lens[nr] : 1);
      memcpy(rows[nr], sqlite3_column_blob(sel, 1), lens[nr]);
      ids[nr++] = sqlite3_column_int64(sel, 0);
    }
    sqlite3_reset(sel);
    for (int r = 0; r < nr; r++) {
      uint8_t *data = rows[r];
      if (lens[r] >= IVF_CHUNK_HDR && rc == SQLITE_OK) {
        int cnt = dn_rd16(data);
        if (IVF_CHUNK_HDR + cnt * eb > lens[r]) cnt = (lens[r] - IVF_CHUNK_HDR) / eb;
        uint32_t fs = dn_rd32(data + 4);
        uint32_t fvp = fs < n ? wv.pages[fs] : 0;
        dn_wr32(data + 8, fvp);
        for (int e = 0; e < cnt; e++) {
          uint8_t *ent = data + IVF_CHUNK_HDR + (size_t)e * eb;
          if (ent[4] == IVF_DELTA_DELETED) continue;
          uint32_t s = fs + (uint32_t)e, pg = s < n ? wv.pages[s] : 0;
          ent[4] = (fvp && pg >= fvp && pg - fvp < IVF_DELTA_DELETED) ? (uint8_t)(pg - fvp) : IVF_DELTA_UNKNOWN;
        }
        sqlite3_bind_int64(upd, 1, ids[r]);
        sqlite3_bind_blob(upd, 2, data, lens[r], SQLITE_STATIC);
        if (sqlite3_step(upd) != SQLITE_DONE) rc = SQLITE_ERROR;
        sqlite3_reset(upd);
      }
      free(rows[r]);
    }
    free(ids); free(rows); free(lens);
  }
  sqlite3_finalize(sel); sqlite3_finalize(upd);
  free(buf);
  if (rc == SQLITE_OK) rc = update_chunks(vt, CHUNK_ID_IVF_HEAD, vt->ivf_head, ivf_head_len(vt));
  if (rc == SQLITE_OK) rc = CFG_INT(vt, "ivf_lists_without_range", nbad);
  free(wv.pages); free(lw.prev); free(lw.bad);
  return rc;
}

/* Incremental insert into a built IVF index: the vector goes to its nearest
** list as a new one-entry chunk (keeping slot numbering consecutive within
** each chunk). Such chunks are found through SQL until the next 'finalize';
** many inserts fragment lists, so rebuild after large changes. */
static int insert_ivf(DenseVtab *vt, int64_t rowid, const float *x) {
  const DenseCfg *c = &vt->cfg;
  int rc = ensure_loaded(vt, 0, NULL);
  if (rc) return rc;
  if (rowid < 0 || rowid > 0xFFFFFFFFLL) { set_err(vt, "dense_ann: layout=ivf needs rowids in [0, 2^32)"); return SQLITE_CONSTRAINT; }
  if (row_exists(vt, "SELECT 1 FROM \"%w\".\"%w_rowids\" WHERE rowid = ?1", rowid)) {
    set_err(vt, "dense_ann: rowid %lld already exists", (long long)rowid);
    return SQLITE_CONSTRAINT;
  }
  int32_t L; float dL;
  dnivf_probe(x, c->dim, vt->ivf_cent, vt->ivf_cn, vt->nlist, 1, &L, &dL);
  float *r = (float *)malloc(sizeof(float) * c->dim);
  const float *cent = vt->ivf_cent + (size_t)L * c->dim;
  for (int t = 0; t < c->dim; t++) r[t] = c->ivf_residual ? x[t] - cent[t] : x[t];
  int eb = ivf_entry_bytes(c);
  uint8_t *chunk = (uint8_t *)calloc(IVF_CHUNK_HDR + eb, 1);
  uint32_t slot = vt->next_id;
  dn_wr16(chunk, 1);
  dn_wr32(chunk + 4, slot);
  dn_wr32(chunk + IVF_CHUNK_HDR, (uint32_t)rowid);
  chunk[IVF_CHUNK_HDR + 4] = IVF_DELTA_UNKNOWN;
  dnpq_encode(&vt->pq, r, chunk + IVF_CHUNK_HDR + 5);
  free(r);
  uint8_t *d = vt->ivf_dir + (size_t)L * 12;
  uint32_t nch = dn_rd32(d);
  sqlite3_stmt *s = 0;
  rc = PREP(vt, &s, "INSERT INTO \"%w\".\"%w_lists\"(id, data) VALUES (?1, ?2)");
  if (rc == SQLITE_OK) {
    sqlite3_bind_int64(s, 1, ((int64_t)L << IVF_LIST_SHIFT) | nch);
    sqlite3_bind_blob(s, 2, chunk, IVF_CHUNK_HDR + eb, SQLITE_STATIC);
    rc = sqlite3_step(s) == SQLITE_DONE ? SQLITE_OK : SQLITE_ERROR;
  }
  sqlite3_finalize(s); s = 0;
  free(chunk);
  if (rc == SQLITE_OK && c->vtype != VT_NONE) {
    uint8_t *vb = (uint8_t *)malloc(vt->rl.vec_bytes);
    vec_encode(c, x, vb);
    rc = PREP(vt, &s, "INSERT INTO \"%w\".\"%w_vectors\"(id, data) VALUES (?1, ?2)");
    if (rc == SQLITE_OK) {
      sqlite3_bind_int64(s, 1, slot); sqlite3_bind_blob(s, 2, vb, vt->rl.vec_bytes, SQLITE_STATIC);
      rc = sqlite3_step(s) == SQLITE_DONE ? SQLITE_OK : SQLITE_ERROR;
    }
    sqlite3_finalize(s); s = 0;
    free(vb);
  }
  if (rc == SQLITE_OK) rc = PREP(vt, &s, "INSERT INTO \"%w\".\"%w_rowids\"(rowid, node) VALUES (?1, ?2)");
  if (rc == SQLITE_OK) {
    sqlite3_bind_int64(s, 1, rowid); sqlite3_bind_int64(s, 2, ((int64_t)L << 32) | slot);
    rc = sqlite3_step(s) == SQLITE_DONE ? SQLITE_OK : SQLITE_ERROR;
  }
  sqlite3_finalize(s);
  if (rc == SQLITE_OK) {
    dn_wr32(d, nch + 1);
    rc = update_chunks(vt, CHUNK_ID_IVF_HEAD, vt->ivf_head, ivf_head_len(vt));
  }
  if (rc == SQLITE_OK) { vt->next_id++; vt->n_nodes++; rc = save_state(vt); }
  if (rc && !vt->base.zErrMsg) set_err(vt, "dense_ann: insert failed: %s", sqlite3_errmsg(vt->db));
  return rc;
}

/* Delete from a built IVF index: mark the entry deleted in its chunk. */
static int delete_ivf(DenseVtab *vt, int64_t rowid) {
  const DenseCfg *c = &vt->cfg;
  sqlite3_stmt *s = 0;
  int64_t node = -1;
  int rc = PREP(vt, &s, "SELECT node FROM \"%w\".\"%w_rowids\" WHERE rowid = ?1");
  if (rc) return rc;
  sqlite3_bind_int64(s, 1, rowid);
  if (sqlite3_step(s) == SQLITE_ROW) node = sqlite3_column_int64(s, 0);
  sqlite3_finalize(s); s = 0;
  if (node < 0) return SQLITE_OK;
  int L = (int)(node >> 32);
  uint32_t slot = (uint32_t)(node & 0xFFFFFFFF);
  int eb = ivf_entry_bytes(c), done = 0;
  rc = PREP(vt, &s, "SELECT id, data FROM \"%w\".\"%w_lists\" WHERE id >= ?1 AND id < ?2");
  int64_t hit = -1; uint8_t *row = NULL; int len = 0;
  if (rc == SQLITE_OK) {
    sqlite3_bind_int64(s, 1, (int64_t)L << IVF_LIST_SHIFT);
    sqlite3_bind_int64(s, 2, (int64_t)(L + 1) << IVF_LIST_SHIFT);
    while (!done && sqlite3_step(s) == SQLITE_ROW) {
      const uint8_t *data = (const uint8_t *)sqlite3_column_blob(s, 1);
      int dl = sqlite3_column_bytes(s, 1);
      if (dl < IVF_CHUNK_HDR) continue;
      uint32_t fs = dn_rd32(data + 4), cnt = dn_rd16(data);
      if (slot < fs || slot >= fs + cnt) continue;
      len = dl; row = (uint8_t *)malloc(len); memcpy(row, data, len);
      hit = sqlite3_column_int64(s, 0); done = 1;
    }
  }
  sqlite3_finalize(s); s = 0;
  if (row) {
    uint8_t *ent = row + IVF_CHUNK_HDR + (size_t)(slot - dn_rd32(row + 4)) * eb;
    if (IVF_CHUNK_HDR + (size_t)(slot - dn_rd32(row + 4) + 1) * eb <= (size_t)len && dn_rd32(ent) == (uint32_t)rowid) {
      ent[4] = IVF_DELTA_DELETED;
      rc = PREP(vt, &s, "UPDATE \"%w\".\"%w_lists\" SET data = ?2 WHERE id = ?1");
      if (rc == SQLITE_OK) {
        sqlite3_bind_int64(s, 1, hit); sqlite3_bind_blob(s, 2, row, len, SQLITE_STATIC);
        rc = sqlite3_step(s) == SQLITE_DONE ? SQLITE_OK : SQLITE_ERROR;
      }
      sqlite3_finalize(s);
    }
    free(row);
  }
  if (rc == SQLITE_OK)
    rc = exec_fmt(vt->db, "DELETE FROM \"%w\".\"%w_rowids\" WHERE rowid = %lld", vt->zDb, vt->zName, (long long)rowid);
  if (rc == SQLITE_OK) { vt->n_nodes--; vt->n_deleted++; rc = save_state(vt); }
  return rc;
}

/* ------------------------------------------------------ vtab methods */

static int dense_connect_common(sqlite3 *db, int argc, const char *const *argv, sqlite3_vtab **ppVtab,
                                char **pzErr, int create) {
  DenseVtab *vt = (DenseVtab *)sqlite3_malloc(sizeof(DenseVtab));
  if (!vt) return SQLITE_NOMEM;
  memset(vt, 0, sizeof *vt);
  vt->db = db;
  vt->zDb = sqlite3_mprintf("%s", argv[1]);
  vt->zName = sqlite3_mprintf("%s", argv[2]);
  int rc = SQLITE_OK;
  if (create) {
    rc = parse_args(&vt->cfg, argc, argv, pzErr);
    if (rc == SQLITE_OK) {
      layout_compute(&vt->cfg, &vt->rl);
      const char *tpl[] = {
        "CREATE TABLE \"%w\".\"%w_config\"(key TEXT PRIMARY KEY, value)",
        "CREATE TABLE \"%w\".\"%w_blobs\"(id INTEGER PRIMARY KEY, data BLOB)",
        "CREATE TABLE \"%w\".\"%w_nodes\"(id INTEGER PRIMARY KEY, data BLOB)",
        "CREATE TABLE \"%w\".\"%w_codes\"(id INTEGER PRIMARY KEY, data BLOB)",
        "CREATE TABLE \"%w\".\"%w_vectors\"(id INTEGER PRIMARY KEY, data BLOB)",
        "CREATE TABLE \"%w\".\"%w_rowids\"(rowid INTEGER PRIMARY KEY, node INTEGER)",
        "CREATE TABLE \"%w\".\"%w_buffer\"(rowid INTEGER PRIMARY KEY, vec BLOB)",
        "CREATE TABLE \"%w\".\"%w_lists\"(id INTEGER PRIMARY KEY, data BLOB)",
      };
      for (size_t i = 0; i < sizeof tpl / sizeof tpl[0] && rc == SQLITE_OK; i++)
        rc = exec_fmt(db, tpl[i], vt->zDb, vt->zName);
      if (rc == SQLITE_OK) rc = save_params(vt);
      if (rc == SQLITE_OK) rc = save_state(vt);
      if (rc) *pzErr = sqlite3_mprintf("dense_ann: %s", sqlite3_errmsg(db));
    }
  } else {
    rc = load_config(vt);
    if (rc && vt->bad_format)
      *pzErr = sqlite3_mprintf("dense_ann: index format %d is not supported by this build (it reads formats 1-%d); "
                               "rebuild the index or upgrade the extension", vt->bad_format, DENSE_FORMAT_MAX);
    else if (rc) *pzErr = sqlite3_mprintf("dense_ann: cannot read config: %s", sqlite3_errmsg(db));
  }
  if (rc == SQLITE_OK) {
    char *schema = sqlite3_mprintf(
        "CREATE TABLE x(embedding, distance, k HIDDEN, ef HIDDEN, beam HIDDEN, rerank HIDDEN, "
        "exact HIDDEN, trace HIDDEN, nprobe HIDDEN, rerank_k HIDDEN, stats HIDDEN, \"%w\" HIDDEN)", argv[2]);
    rc = sqlite3_declare_vtab(db, schema);
    sqlite3_free(schema);
  }
  if (rc) {
    sqlite3_free(vt->zDb); sqlite3_free(vt->zName); sqlite3_free(vt);
    return rc;
  }
  *ppVtab = &vt->base;
  return SQLITE_OK;
}

static int dense_create(sqlite3 *db, void *pAux, int argc, const char *const *argv, sqlite3_vtab **pp, char **pzErr) {
  (void)pAux;
  return dense_connect_common(db, argc, argv, pp, pzErr, 1);
}

static int dense_connect(sqlite3 *db, void *pAux, int argc, const char *const *argv, sqlite3_vtab **pp, char **pzErr) {
  (void)pAux;
  return dense_connect_common(db, argc, argv, pp, pzErr, 0);
}

static int dense_disconnect(sqlite3_vtab *pVtab) {
  DenseVtab *vt = (DenseVtab *)pVtab;
  for (int t = 0; t < T_COUNT; t++) sqlite3_finalize(vt->st_get[t]);
  dnpq_free(&vt->pq);
  dnpq_free(&vt->ivf_cpq);
  free(vt->entries);
  free(vt->ivf_cent); free(vt->ivf_cn); free(vt->ivf_head);
  sqlite3_free(vt->zDb); sqlite3_free(vt->zName);
  sqlite3_free(vt);
  return SQLITE_OK;
}

static int dense_destroy(sqlite3_vtab *pVtab) {
  DenseVtab *vt = (DenseVtab *)pVtab;
  static const char *const sfx[] = { "config", "blobs", "nodes", "codes", "vectors", "rowids", "buffer", "lists" };
  int rc = SQLITE_OK;
  for (size_t i = 0; i < sizeof sfx / sizeof sfx[0] && rc == SQLITE_OK; i++)
    rc = exec_fmt(vt->db, "DROP TABLE IF EXISTS \"%w\".\"%w_%s\"", vt->zDb, vt->zName, sfx[i]);
  if (rc == SQLITE_OK) dense_disconnect(pVtab);
  return rc;
}

static int dense_best_index(sqlite3_vtab *pVtab, sqlite3_index_info *info) {
  (void)pVtab;
  int idx[11];
  for (int i = 0; i < 11; i++) idx[i] = -1;
  for (int i = 0; i < info->nConstraint; i++) {
    const struct sqlite3_index_constraint *c = &info->aConstraint[i];
    if (!c->usable) continue;
    if (c->iColumn == COL_EMBEDDING && c->op == SQLITE_INDEX_CONSTRAINT_MATCH) { if (idx[0] < 0) idx[0] = i; }
    else if (c->op == SQLITE_INDEX_CONSTRAINT_LIMIT) { if (idx[7] < 0) idx[7] = i; }
    else if (c->op == SQLITE_INDEX_CONSTRAINT_EQ) {
      int slot = -1;
      switch (c->iColumn) {
        case COL_K: slot = 1; break;
        case COL_EF: slot = 2; break;
        case COL_BEAM: slot = 3; break;
        case COL_RERANK: slot = 4; break;
        case COL_EXACT: slot = 5; break;
        case COL_TRACE: slot = 6; break;
        case COL_NPROBE: slot = 9; break;
        case COL_RK: slot = 10; break;
        case -1: slot = 8; break;
      }
      if (slot >= 0 && idx[slot] < 0) idx[slot] = i;
    }
  }
  int flags = 0, argn = 1;
  if (idx[0] >= 0) {
    static const int bits[11] = { F_MATCH, F_K, F_EF, F_BEAM, F_RERANK, F_EXACT, F_TRACE, F_LIMIT, 0, F_NPROBE, F_RK };
    for (int s = 0; s < 11; s++) {
      if (s == 8) continue;
      if (idx[s] < 0) continue;
      flags |= bits[s];
      info->aConstraintUsage[idx[s]].argvIndex = argn++;
      info->aConstraintUsage[idx[s]].omit = 1;
    }
    info->estimatedCost = 10.0;
    info->estimatedRows = 10;
    if (info->nOrderBy == 1 && info->aOrderBy[0].iColumn == COL_DISTANCE && !info->aOrderBy[0].desc)
      info->orderByConsumed = 1;
  } else if (idx[8] >= 0) {
    flags = F_ROWID;
    info->aConstraintUsage[idx[8]].argvIndex = 1;
    info->aConstraintUsage[idx[8]].omit = 1;
    info->estimatedCost = 1.0;
    info->estimatedRows = 1;
    info->idxFlags = SQLITE_INDEX_SCAN_UNIQUE;
  } else {
    info->estimatedCost = 1e7;
  }
  info->idxNum = flags;
  return SQLITE_OK;
}

static int dense_open(sqlite3_vtab *pVtab, sqlite3_vtab_cursor **pp) {
  (void)pVtab;
  DenseCursor *cur = (DenseCursor *)sqlite3_malloc(sizeof(DenseCursor));
  if (!cur) return SQLITE_NOMEM;
  memset(cur, 0, sizeof *cur);
  *pp = &cur->base;
  return SQLITE_OK;
}

static void cursor_reset(DenseCursor *cur) {
  free(cur->res); cur->res = NULL; cur->nres = cur->pos = 0;
  sqlite3_finalize(cur->scan); cur->scan = NULL;
  sqlite3_free(cur->stats); cur->stats = NULL;
  cur->mode = 0;
}

static int dense_close(sqlite3_vtab_cursor *pCur) {
  DenseCursor *cur = (DenseCursor *)pCur;
  cursor_reset(cur);
  sqlite3_free(cur);
  return SQLITE_OK;
}

static char *stats_json(const QStats *st) {
  return sqlite3_mprintf(
      "{\"ann\":%d,\"raw\":%d,\"rounds\":%d,\"pages\":%d,\"bytes\":%lld,\"fallback\":%d,"
      "\"expanded\":%d,\"dist\":%d,\"entry_dist\":%d,\"rerank\":%d,\"rerank_rounds\":%d,"
      "\"setup_rounds\":%d,\"setup_pages\":%d,\"setup_bytes\":%lld,\"lists\":%d,\"scanned\":%d,\"ms\":%.3f",
      st->ann, st->raw, st->rounds, st->pages, (long long)st->bytes, st->fallback, st->expanded,
      st->dist, st->entry_dist, st->rerank, st->rerank_rounds, st->setup_rounds, st->setup_pages,
      (long long)st->setup_bytes, st->lists, st->scanned, st->ms);
}

static int dense_filter(sqlite3_vtab_cursor *pCur, int idxNum, const char *idxStr, int argc, sqlite3_value **argv) {
  DenseCursor *cur = (DenseCursor *)pCur;
  DenseVtab *vt = (DenseVtab *)pCur->pVtab;
  (void)idxStr; (void)argc;
  cursor_reset(cur);
  int rc = SQLITE_OK;

  if (idxNum & F_MATCH) {
    int a = 0;
    sqlite3_value *qv = argv[a++];
    cur->k = 10; cur->ef = vt->cfg.ef_default; cur->beam = vt->cfg.beam_default;
    cur->rerank = vt->cfg.vtype != VT_NONE ? 2 : 0;   /* rerank everything expanded when vectors exist */
    cur->exact = 0; cur->trace = 0;
    int have_k = 0;
    if (idxNum & F_K) { cur->k = sqlite3_value_int64(argv[a++]); have_k = 1; }
    if (idxNum & F_EF) cur->ef = sqlite3_value_int64(argv[a++]);
    if (idxNum & F_BEAM) cur->beam = sqlite3_value_int64(argv[a++]);
    if (idxNum & F_RERANK) cur->rerank = sqlite3_value_int64(argv[a++]);
    if (idxNum & F_EXACT) cur->exact = sqlite3_value_int64(argv[a++]);
    if (idxNum & F_TRACE) cur->trace = sqlite3_value_int64(argv[a++]);
    if (idxNum & F_LIMIT) { int64_t lim = sqlite3_value_int64(argv[a++]); if (!have_k || lim < cur->k) cur->k = lim; }
    cur->nprobe = vt->cfg.nprobe_default; cur->rk = vt->cfg.rk_default;
    if (idxNum & F_NPROBE) cur->nprobe = sqlite3_value_int64(argv[a++]);
    if (idxNum & F_RK) cur->rk = sqlite3_value_int64(argv[a++]);
    if (cur->rk > 100000) cur->rk = 100000;
    if (cur->k < 1) { cur->mode = 0; return SQLITE_OK; }
    if (cur->k > 100000) cur->k = 100000;
    if (cur->ef > 100000) cur->ef = 100000;

    if (sqlite3_value_type(qv) == SQLITE_TEXT && !sqlite3_stricmp((const char *)sqlite3_value_text(qv), "warm")) {
      /* MATCH 'warm': load the index head (codebook, entry set or IVF
      ** centroids) now, so that the first real query does not pay for it. */
      QStats st; memset(&st, 0, sizeof st);
      double t0 = now_ms();
      if (vt->built) rc = ensure_loaded(vt, sqlite3_get_autocommit(vt->db), &st);
      st.ms = now_ms() - t0;
      char *js = stats_json(&st);
      cur->stats = sqlite3_mprintf("%s,\"warm\":1}", js);
      sqlite3_free(js);
      cur->mode = 1;              /* no rows */
      return rc;
    }
    char *err = NULL;
    float *q = parse_vector(qv, vt->cfg.dim, &err);
    if (!q) { sqlite3_free(vt->base.zErrMsg); vt->base.zErrMsg = err; return SQLITE_ERROR; }
    if (vt->cfg.metric == METRIC_COSINE) dn_normalize(q, vt->cfg.dim);

    /* Auto-build on first query; if that is impossible (e.g. read-only
    ** database) fall back to brute force over the buffer. */
    if (!vt->built && !cur->exact) {
      if (do_build(vt) != SQLITE_OK) {
        sqlite3_free(vt->base.zErrMsg); vt->base.zErrMsg = NULL;
        cur->exact = 1;
      }
    }
    QStats st; memset(&st, 0, sizeof st);
    if (cur->trace) st.trace = sqlite3_str_new(vt->db);
    double t0 = now_ms();
    int raw = sqlite3_get_autocommit(vt->db);
    if (cur->exact == 2 && vt->built && vt->cfg.layout == LAYOUT_IVF)
      rc = ivf_exact_pq(vt, q, (int)cur->k, &cur->res, &cur->nres, &st);
    else if (cur->exact) rc = exact_search(vt, q, (int)cur->k, (int)cur->exact, &cur->res, &cur->nres, &st);
    else if (vt->cfg.layout == LAYOUT_IVF)
      rc = ivf_search(vt, q, (int)cur->k, (int)cur->nprobe, (int)cur->rk, (int)cur->rerank, raw, &cur->res, &cur->nres, &st);
    else rc = ann_search(vt, q, (int)cur->k, (int)cur->ef, (int)cur->beam, (int)cur->rerank, raw, &cur->res, &cur->nres, &st);
    st.ms = now_ms() - t0;
    free(q);
    char *js = stats_json(&st);
    if (st.trace) {
      char *tr = sqlite3_str_finish(st.trace);
      cur->stats = sqlite3_mprintf("%s,\"trace\":[%s]}", js, tr ? tr : "");
      sqlite3_free(tr);
    } else {
      cur->stats = sqlite3_mprintf("%s}", js);
    }
    sqlite3_free(js);
    cur->mode = 1;
    return rc;
  }

  /* Point lookup or full scan over rowids. */
  const char *tbl = vt->built ? "rowids" : "buffer";
  if (idxNum & F_ROWID) {
    rc = prep_fmt(vt->db, &cur->scan, "SELECT rowid FROM \"%w\".\"%w_%s\" WHERE rowid = ?1", vt->zDb, vt->zName, tbl);
    if (rc == SQLITE_OK) sqlite3_bind_value(cur->scan, 1, argv[0]);
  } else {
    rc = prep_fmt(vt->db, &cur->scan, "SELECT rowid FROM \"%w\".\"%w_%s\" ORDER BY rowid", vt->zDb, vt->zName, tbl);
  }
  if (rc) return rc;
  cur->mode = 2;
  cur->scan_eof = sqlite3_step(cur->scan) != SQLITE_ROW;
  return SQLITE_OK;
}

static int dense_next(sqlite3_vtab_cursor *pCur) {
  DenseCursor *cur = (DenseCursor *)pCur;
  if (cur->mode == 1) cur->pos++;
  else if (cur->mode == 2) cur->scan_eof = sqlite3_step(cur->scan) != SQLITE_ROW;
  return SQLITE_OK;
}

static int dense_eof(sqlite3_vtab_cursor *pCur) {
  DenseCursor *cur = (DenseCursor *)pCur;
  if (cur->mode == 1) return cur->pos >= cur->nres;
  if (cur->mode == 2) return cur->scan_eof;
  return 1;
}

static int dense_rowid(sqlite3_vtab_cursor *pCur, sqlite3_int64 *pRowid) {
  DenseCursor *cur = (DenseCursor *)pCur;
  if (cur->mode == 1) *pRowid = cur->res[cur->pos].rowid;
  else *pRowid = sqlite3_column_int64(cur->scan, 0);
  return SQLITE_OK;
}

/* The stored vector of rowid as a float32 blob (NULL if not stored). */
static void result_embedding(DenseVtab *vt, sqlite3_context *ctx, int64_t rowid) {
  const DenseCfg *c = &vt->cfg;
  sqlite3_stmt *s = 0;
  if (!vt->built) {
    if (PREP(vt, &s, "SELECT vec FROM \"%w\".\"%w_buffer\" WHERE rowid = ?1") == SQLITE_OK) {
      sqlite3_bind_int64(s, 1, rowid);
      if (sqlite3_step(s) == SQLITE_ROW) sqlite3_result_value(ctx, sqlite3_column_value(s, 0));
    }
    sqlite3_finalize(s);
    return;
  }
  if (c->vtype == VT_NONE) return;
  int rc = c->vinline
    ? prep_fmt(vt->db, &s, "SELECT n.data FROM \"%w\".\"%w_rowids\" r JOIN \"%w\".\"%w_nodes\" n ON n.id = r.node WHERE r.rowid = ?1",
               vt->zDb, vt->zName, vt->zDb, vt->zName)
    : prep_fmt(vt->db, &s, "SELECT v.data FROM \"%w\".\"%w_rowids\" r JOIN \"%w\".\"%w_vectors\" v ON v.id = (r.node & 4294967295) WHERE r.rowid = ?1",
               vt->zDb, vt->zName, vt->zDb, vt->zName);
  if (rc == SQLITE_OK) {
    sqlite3_bind_int64(s, 1, rowid);
    if (sqlite3_step(s) == SQLITE_ROW) {
      const uint8_t *b = (const uint8_t *)sqlite3_column_blob(s, 0);
      int len = sqlite3_column_bytes(s, 0);
      int need = c->vinline ? vt->rl.off_vec + vt->rl.vec_bytes : vt->rl.vec_bytes;
      if (len >= need) {
        float *v = (float *)malloc(sizeof(float) * c->dim);
        vec_decode(c, c->vinline ? b + vt->rl.off_vec : b, v);
        sqlite3_result_blob(ctx, v, c->dim * 4, SQLITE_TRANSIENT);
        free(v);
      }
    }
  }
  sqlite3_finalize(s);
}

static int dense_column(sqlite3_vtab_cursor *pCur, sqlite3_context *ctx, int i) {
  DenseCursor *cur = (DenseCursor *)pCur;
  DenseVtab *vt = (DenseVtab *)pCur->pVtab;
  if (cur->mode == 2) {
    if (i == COL_EMBEDDING) result_embedding(vt, ctx, sqlite3_column_int64(cur->scan, 0));
    return SQLITE_OK;
  }
  switch (i) {
    case COL_DISTANCE: sqlite3_result_double(ctx, cur->res[cur->pos].d); break;
    case COL_K: sqlite3_result_int64(ctx, cur->k); break;
    case COL_EF: sqlite3_result_int64(ctx, cur->ef); break;
    case COL_BEAM: sqlite3_result_int64(ctx, cur->beam); break;
    case COL_RERANK: sqlite3_result_int64(ctx, cur->rerank); break;
    case COL_EXACT: sqlite3_result_int64(ctx, cur->exact); break;
    case COL_TRACE: sqlite3_result_int64(ctx, cur->trace); break;
    case COL_NPROBE: sqlite3_result_int64(ctx, cur->nprobe); break;
    case COL_RK: sqlite3_result_int64(ctx, cur->rk); break;
    case COL_STATS: if (cur->stats) sqlite3_result_text(ctx, cur->stats, -1, SQLITE_TRANSIENT); break;
    default: break;   /* embedding is not returned by searches */
  }
  return SQLITE_OK;
}

static int dense_update(sqlite3_vtab *pVtab, int argc, sqlite3_value **argv, sqlite3_int64 *pRowid) {
  DenseVtab *vt = (DenseVtab *)pVtab;
  const DenseCfg *c = &vt->cfg;
  int rc;
  if (argc == 1) return delete_row(vt, sqlite3_value_int64(argv[0]));

  /* Commands: INSERT INTO v(v) VALUES ('build' | 'finalize'). */
  sqlite3_value *cmd = argv[2 + COL_CMD];
  if (sqlite3_value_type(argv[0]) == SQLITE_NULL && sqlite3_value_type(cmd) != SQLITE_NULL) {
    const char *z = (const char *)sqlite3_value_text(cmd);
    if (z && !strcmp(z, "build")) return do_build(vt);
    if (z && !strcmp(z, "finalize")) return do_finalize(vt);
    set_err(vt, "dense_ann: unknown command '%s'", z ? z : "");
    return SQLITE_ERROR;
  }

  /* UPDATE: delete the old row, then insert the new one. */
  if (sqlite3_value_type(argv[0]) != SQLITE_NULL) {
    rc = delete_row(vt, sqlite3_value_int64(argv[0]));
    if (rc) return rc;
  }
  if (sqlite3_value_type(argv[1]) == SQLITE_NULL) {
    set_err(vt, "dense_ann: an explicit rowid is required");
    return SQLITE_CONSTRAINT;
  }
  int64_t rowid = sqlite3_value_int64(argv[1]);
  char *err = NULL;
  float *x = parse_vector(argv[2 + COL_EMBEDDING], c->dim, &err);
  if (!x) { sqlite3_free(vt->base.zErrMsg); vt->base.zErrMsg = err; return SQLITE_ERROR; }
  if (c->metric == METRIC_COSINE) dn_normalize(x, c->dim);
  if (!vt->built) {
    sqlite3_stmt *s = 0;
    rc = PREP(vt, &s, "INSERT INTO \"%w\".\"%w_buffer\"(rowid, vec) VALUES (?1, ?2)");
    if (rc == SQLITE_OK) {
      sqlite3_bind_int64(s, 1, rowid);
      sqlite3_bind_blob(s, 2, x, c->dim * 4, SQLITE_STATIC);
      rc = sqlite3_step(s);
      rc = (rc == SQLITE_DONE) ? SQLITE_OK : rc;
      if (rc) set_err(vt, "dense_ann: %s", sqlite3_errmsg(vt->db));
    }
    sqlite3_finalize(s);
  } else if (c->layout == LAYOUT_IVF) {
    rc = insert_ivf(vt, rowid, x);
  } else {
    rc = insert_built(vt, rowid, x);
  }
  free(x);
  *pRowid = rowid;
  return rc;
}

static int dense_shadow_name(const char *zName) {
  static const char *const sfx[] = { "config", "blobs", "nodes", "codes", "vectors", "rowids", "buffer", "lists" };
  for (size_t i = 0; i < sizeof sfx / sizeof sfx[0]; i++) if (!sqlite3_stricmp(zName, sfx[i])) return 1;
  return 0;
}

static sqlite3_module dense_module = {
  3,                  /* iVersion */
  dense_create, dense_connect, dense_best_index, dense_disconnect, dense_destroy,
  dense_open, dense_close, dense_filter, dense_next, dense_eof, dense_column, dense_rowid,
  dense_update,
  0, 0, 0, 0,         /* xBegin, xSync, xCommit, xRollback */
  0, 0,               /* xFindFunction, xRename */
  0, 0, 0,            /* xSavepoint, xRelease, xRollbackTo */
  dense_shadow_name,
#if SQLITE_VERSION_NUMBER >= 3044000
  0                   /* xIntegrity */
#endif
};

#ifdef _WIN32
__declspec(dllexport)
#endif
int sqlite3_denseann_init(sqlite3 *db, char **pzErrMsg, const sqlite3_api_routines *pApi) {
  SQLITE_EXTENSION_INIT2(pApi);
  (void)pzErrMsg;
  return sqlite3_create_module_v2(db, "dense_ann", &dense_module, 0, 0);
}
