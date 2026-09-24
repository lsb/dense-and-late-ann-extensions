/*
** hnsw.c — HNSW construction (Malkov & Yashunin 2016), following the
** structure of hnswlib: per-node locks for concurrent inserts, a global lock
** for the entry point, the neighbour-selection heuristic, and at most M
** links on upper layers and M0 (normally 2M) on layer 0.
*/
#include "hnsw.h"
#include "dense_common.h"

#include <float.h>
#include <stdio.h>
#include <time.h>
#include <alloca.h>

#define HNSW_MAX_LEVEL 16

/* ----------------------------------------------------------------- heap */
/* A binary min-heap of (distance, id). A max-heap is the same heap storing
** negated distances. */
typedef struct { float d; uint32_t id; } HItem;
typedef struct { HItem *a; int n, cap; } Heap;

static void heap_push(Heap *h, float d, uint32_t id) {
  if (h->n == h->cap) {
    h->cap = h->cap ? h->cap * 2 : 64;
    h->a = (HItem *)realloc(h->a, sizeof(HItem) * h->cap);
  }
  int i = h->n++;
  while (i > 0) {
    int p = (i - 1) / 2;
    if (h->a[p].d <= d) break;
    h->a[i] = h->a[p]; i = p;
  }
  h->a[i].d = d; h->a[i].id = id;
}

static HItem heap_pop(Heap *h) {
  HItem top = h->a[0], last = h->a[--h->n];
  int i = 0;
  for (;;) {
    int l = 2 * i + 1, r = l + 1, m = i;
    float md = last.d;
    if (l < h->n && h->a[l].d < md) { m = l; md = h->a[l].d; }
    if (r < h->n && h->a[r].d < md) { m = r; }
    if (m == i) break;
    h->a[i] = h->a[m]; i = m;
  }
  if (h->n > 0) h->a[i] = last;
  return top;
}

/* ------------------------------------------------------------- locking */

typedef struct {
  uint32_t *vis;   /* visited tags, one per node */
  uint32_t tag;
  Heap cand, res;
  HItem *tmp; int tmpcap;
} Scratch;

typedef struct {
#ifdef DENSE_ANN_THREADS
  pthread_mutex_t *node;
  pthread_mutex_t global;
#endif
  Scratch *scratch;
  int nthreads, verbose;
  int64_t done;
  double t0;
} Priv;

#ifdef DENSE_ANN_THREADS
#define LOCK_NODE(p, i)   pthread_mutex_lock(&(p)->node[i])
#define UNLOCK_NODE(p, i) pthread_mutex_unlock(&(p)->node[i])
#define LOCK_GLOBAL(p)    pthread_mutex_lock(&(p)->global)
#define UNLOCK_GLOBAL(p)  pthread_mutex_unlock(&(p)->global)
#else
#define LOCK_NODE(p, i)   ((void)0)
#define UNLOCK_NODE(p, i) ((void)0)
#define LOCK_GLOBAL(p)    ((void)0)
#define UNLOCK_GLOBAL(p)  ((void)0)
#endif

static inline float hdist(const Hnsw *h, uint32_t a, const float *q) {
  return dn_distance(h->metric, h->x + (size_t)a * h->dim, q, h->dim);
}

/* ------------------------------------------------ neighbour selection */

int dnhnsw_select(int metric, int dim, float alpha, const uint32_t *cand, const float *dist,
                int ncand, int cap, dnhnsw_vec_fn vec, void *vctx, uint32_t *out) {
  int nout = 0;
  for (int i = 0; i < ncand && nout < cap; i++) {
    const float *vc = vec(vctx, cand[i]);
    int keep = 1;
    for (int j = 0; j < nout; j++) {
      /* Drop c if an already-kept neighbour r is (alpha times) closer to c
      ** than the base point is: the edge base->r->c covers it. */
      if (alpha * dn_distance(metric, vc, vec(vctx, out[j]), dim) <= dist[i]) { keep = 0; break; }
    }
    if (keep) out[nout++] = cand[i];
  }
  return nout;
}

static const float *vec_of(void *ctx, uint32_t id) {
  const Hnsw *h = (const Hnsw *)ctx;
  return h->x + (size_t)id * h->dim;
}

/* Sort HItems by ascending distance (insertion sort is fine for <= ~500). */
static void sort_items(HItem *a, int n) {
  for (int i = 1; i < n; i++) {
    HItem t = a[i]; int j = i - 1;
    while (j >= 0 && a[j].d > t.d) { a[j + 1] = a[j]; j--; }
    a[j + 1] = t;
  }
}

/* ------------------------------------------------------- layer search */

/* Beam search on one layer from entry ep. Leaves the ef best results in
** s->tmp sorted ascending and returns their number. */
static int search_layer(Hnsw *h, Priv *p, Scratch *s, const float *q, uint32_t ep, float epd,
                        int ef, int lv) {
  int cap = (lv == 0 ? h->M0 : h->M);
  uint32_t *buf = (uint32_t *)alloca(sizeof(uint32_t) * (cap + 1));
  if (++s->tag == 0) { memset(s->vis, 0, sizeof(uint32_t) * h->n); s->tag = 1; }
  s->cand.n = s->res.n = 0;
  heap_push(&s->cand, epd, ep);
  heap_push(&s->res, -epd, ep);
  s->vis[ep] = s->tag;
  while (s->cand.n > 0) {
    HItem c = heap_pop(&s->cand);
    if (c.d > -s->res.a[0].d && s->res.n >= ef) break;
    LOCK_NODE(p, c.id);
    uint32_t *l = dnhnsw_list(h, c.id, lv);
    memcpy(buf, l, sizeof(uint32_t) * (l[0] + 1));
    UNLOCK_NODE(p, c.id);
    for (uint32_t k = 1; k <= buf[0]; k++) {
      uint32_t nb = buf[k];
      if (s->vis[nb] == s->tag) continue;
      s->vis[nb] = s->tag;
      float d = hdist(h, nb, q);
      if (s->res.n < ef || d < -s->res.a[0].d) {
        heap_push(&s->cand, d, nb);
        heap_push(&s->res, -d, nb);
        if (s->res.n > ef) heap_pop(&s->res);
      }
    }
  }
  if (s->tmpcap < s->res.n) {
    s->tmpcap = s->res.n;
    s->tmp = (HItem *)realloc(s->tmp, sizeof(HItem) * s->tmpcap);
  }
  int n = s->res.n;
  for (int i = n - 1; i >= 0; i--) { HItem t = heap_pop(&s->res); t.d = -t.d; s->tmp[i] = t; }
  return n;
}

/* Add `node` to the neighbour list of `s` at level lv, pruning if full. */
static void link_back(Hnsw *h, Priv *p, uint32_t s, uint32_t node, int lv) {
  int cap = (lv == 0 ? h->M0 : h->M);
  LOCK_NODE(p, s);
  uint32_t *l = dnhnsw_list(h, s, lv);
  if ((int)l[0] < cap) {
    l[1 + l[0]] = node; l[0]++;
  } else {
    int nc = cap + 1;
    HItem *it = (HItem *)alloca(sizeof(HItem) * nc);
    const float *vs = h->x + (size_t)s * h->dim;
    for (int k = 0; k < cap; k++) { it[k].id = l[1 + k]; it[k].d = hdist(h, it[k].id, vs); }
    it[cap].id = node; it[cap].d = hdist(h, node, vs);
    sort_items(it, nc);
    uint32_t *ids = (uint32_t *)alloca(sizeof(uint32_t) * nc);
    float *ds = (float *)alloca(sizeof(float) * nc);
    for (int k = 0; k < nc; k++) { ids[k] = it[k].id; ds[k] = it[k].d; }
    l[0] = (uint32_t)dnhnsw_select(h->metric, h->dim, h->alpha, ids, ds, nc, cap, vec_of, h, l + 1);
  }
  UNLOCK_NODE(p, s);
}

static void insert_node(Hnsw *h, Priv *p, Scratch *s, uint32_t i) {
  const float *q = h->x + (size_t)i * h->dim;
  int L = h->level[i];

  LOCK_GLOBAL(p);
  int maxl = h->maxlevel;
  uint32_t ep = h->entry;
  int hold_global = (L > maxl);
  if (!hold_global) UNLOCK_GLOBAL(p);

  float epd = hdist(h, ep, q);
  /* Greedy descent through the layers above L. */
  for (int lv = maxl; lv > L; lv--) {
    int changed = 1;
    while (changed) {
      changed = 0;
      LOCK_NODE(p, ep);
      uint32_t *l = dnhnsw_list(h, ep, lv);
      uint32_t cnt = l[0];
      uint32_t buf[256];
      if (cnt > 256) cnt = 256;
      memcpy(buf, l + 1, sizeof(uint32_t) * cnt);
      UNLOCK_NODE(p, ep);
      for (uint32_t k = 0; k < cnt; k++) {
        float d = hdist(h, buf[k], q);
        if (d < epd) { epd = d; ep = buf[k]; changed = 1; }
      }
    }
  }
  /* Connect on each layer from min(L, maxl) down to 0. */
  for (int lv = (L < maxl ? L : maxl); lv >= 0; lv--) {
    int cap = (lv == 0 ? h->M0 : h->M);
    int nres = search_layer(h, p, s, q, ep, epd, h->efc, lv);
    uint32_t *ids = (uint32_t *)alloca(sizeof(uint32_t) * nres);
    float *ds = (float *)alloca(sizeof(float) * nres);
    for (int k = 0; k < nres; k++) { ids[k] = s->tmp[k].id; ds[k] = s->tmp[k].d; }
    uint32_t *sel = (uint32_t *)alloca(sizeof(uint32_t) * cap);
    int nsel = dnhnsw_select(h->metric, h->dim, h->alpha, ids, ds, nres, cap, vec_of, h, sel);
    LOCK_NODE(p, i);
    uint32_t *l = dnhnsw_list(h, i, lv);
    memcpy(l + 1, sel, sizeof(uint32_t) * nsel);
    l[0] = (uint32_t)nsel;
    UNLOCK_NODE(p, i);
    for (int k = 0; k < nsel; k++) link_back(h, p, sel[k], i, lv);
    ep = ids[0]; epd = ds[0];
  }
  if (hold_global) {
    h->entry = i; h->maxlevel = L;
    UNLOCK_GLOBAL(p);
  }
}

static double now_sec(void) {
  struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void insert_task(void *ctx, int64_t i, int thread) {
  Hnsw *h = (Hnsw *)ctx; Priv *p = (Priv *)h->priv;
  if (i == 0) return;   /* node 0 is the initial entry point */
  insert_node(h, p, &p->scratch[thread], (uint32_t)i);
  if (p->verbose) {
    int64_t d = __atomic_add_fetch(&p->done, 1, __ATOMIC_RELAXED);
    if (d % 50000 == 0)
      fprintf(stderr, "dense_ann: hnsw %lld/%lld nodes, %.1fs\n", (long long)d, (long long)h->n, now_sec() - p->t0);
  }
}

/* dn_parallel_for counts from 0; this shifts task i to node off+i. */
typedef struct { Hnsw *h; int64_t off; } OffsetCtx;
static void insert_offset_task(void *vctx, int64_t i, int thread) {
  OffsetCtx *c = (OffsetCtx *)vctx;
  insert_task(c->h, c->off + i, thread);
}

int dnhnsw_build(Hnsw *h, int nthreads, int verbose) {
  if (nthreads < 1) nthreads = 1;
#ifndef DENSE_ANN_THREADS
  nthreads = 1;
#endif
  int64_t n = h->n;
  if (h->alpha <= 0) h->alpha = 1.0f;
  h->level = (uint8_t *)calloc(n ? n : 1, 1);
  h->l0 = (uint32_t *)calloc((size_t)(n ? n : 1) * (h->M0 + 1), sizeof(uint32_t));
  h->up = (uint32_t **)calloc(n ? n : 1, sizeof(uint32_t *));
  if (!h->level || !h->l0 || !h->up) return 1;
  if (n == 0) return 0;

  /* Levels depend only on (seed, i), not on thread scheduling. */
  double mL = 1.0 / log((double)h->M);
  for (int64_t i = 0; i < n; i++) {
    double u = ((dn_hash2(h->seed, (uint64_t)i) >> 11) + 1) * (1.0 / 9007199254740993.0);
    int L = (int)floor(-log(u) * mL);
    if (L > HNSW_MAX_LEVEL) L = HNSW_MAX_LEVEL;
    h->level[i] = (uint8_t)L;
    if (L > 0) h->up[i] = (uint32_t *)calloc((size_t)L * (h->M + 1), sizeof(uint32_t));
  }
  h->entry = 0; h->maxlevel = h->level[0];

  Priv *p = (Priv *)calloc(1, sizeof(Priv));
  p->nthreads = nthreads; p->verbose = verbose; p->t0 = now_sec();
  h->priv = p;
#ifdef DENSE_ANN_THREADS
  p->node = (pthread_mutex_t *)malloc(sizeof(pthread_mutex_t) * n);
  for (int64_t i = 0; i < n; i++) pthread_mutex_init(&p->node[i], NULL);
  pthread_mutex_init(&p->global, NULL);
#endif
  p->scratch = (Scratch *)calloc(nthreads, sizeof(Scratch));
  for (int t = 0; t < nthreads; t++) p->scratch[t].vis = (uint32_t *)calloc(n, sizeof(uint32_t));

  /* Insert the first few thousand nodes serially so the concurrent phase
  ** starts from a connected graph, then the rest in parallel. */
  int64_t serial = n < 2000 ? n : 2000;
  for (int64_t i = 1; i < serial; i++) insert_node(h, p, &p->scratch[0], (uint32_t)i);
  p->done = serial;
  if (n > serial) {
    OffsetCtx ctx = { h, serial };
    dn_parallel_for(n - serial, nthreads, 16, insert_offset_task, &ctx);
  }

  for (int t = 0; t < nthreads; t++) {
    free(p->scratch[t].vis); free(p->scratch[t].cand.a); free(p->scratch[t].res.a); free(p->scratch[t].tmp);
  }
  free(p->scratch);
#ifdef DENSE_ANN_THREADS
  for (int64_t i = 0; i < n; i++) pthread_mutex_destroy(&p->node[i]);
  pthread_mutex_destroy(&p->global);
  free(p->node);
#endif
  if (verbose) fprintf(stderr, "dense_ann: hnsw built %lld nodes in %.1fs\n", (long long)n, now_sec() - p->t0);
  free(p);
  h->priv = NULL;
  return 0;
}

void dnhnsw_free(Hnsw *h) {
  if (h->up) for (int64_t i = 0; i < h->n; i++) free(h->up[i]);
  free(h->up); free(h->l0); free(h->level);
  h->up = NULL; h->l0 = NULL; h->level = NULL;
}
