/*
** pq.c — product quantiser training, encoding and ADC tables.
*/
#include "pq.h"
#include "dense_common.h"

#include <float.h>
#include <stdio.h>

int pq_init(PQ *pq, int dim, int m) {
  memset(pq, 0, sizeof *pq);
  if (m <= 0 || dim % m != 0) return 1;
  pq->dim = dim; pq->m = m; pq->dsub = dim / m;
  pq->cent = (float *)calloc((size_t)m * PQ_KSUB * pq->dsub, sizeof(float));
  return pq->cent ? 0 : 1;
}

void pq_free(PQ *pq) {
  free(pq->cent);
  memset(pq, 0, sizeof *pq);
}

void pq_round_f16(PQ *pq) {
  size_t n = (size_t)pq->m * PQ_KSUB * pq->dsub;
  for (size_t i = 0; i < n; i++) pq->cent[i] = dn_f16_to_f32(dn_f32_to_f16(pq->cent[i]));
}

/* ------------------------------------------------------------ k-means */

/* Index of the nearest of k centroids c[k*d] to point p[d]. */
static int nearest(const float *p, const float *c, int k, int d, float *dist_out) {
  int best = 0; float bd = FLT_MAX;
  for (int j = 0; j < k; j++) {
    float s = dn_l2sq(p, c + (size_t)j * d, d);
    if (s < bd) { bd = s; best = j; }
  }
  if (dist_out) *dist_out = bd;
  return best;
}

/* Lloyd's k-means for one subspace: data x[n*d] -> centroids c[k*d].
** k-means++ initialisation, deterministic given seed. */
static void kmeans(const float *x, int64_t n, int d, int k, int iters, uint64_t seed, float *c) {
  uint64_t rng = seed;
  if (n <= k) {
    /* Fewer points than centroids: copy points, duplicate the rest. */
    for (int j = 0; j < k; j++) memcpy(c + (size_t)j * d, x + (size_t)(n ? j % n : 0) * d, sizeof(float) * d);
    return;
  }
  float *mind = (float *)malloc(sizeof(float) * n);
  int *assign = (int *)malloc(sizeof(int) * n);
  int64_t *cnt = (int64_t *)malloc(sizeof(int64_t) * k);
  double *sum = (double *)malloc(sizeof(double) * (size_t)k * d);

  /* k-means++ seeding. */
  int64_t first = (int64_t)(dn_rng_uniform(&rng) * n);
  memcpy(c, x + first * d, sizeof(float) * d);
  for (int64_t i = 0; i < n; i++) mind[i] = dn_l2sq(x + i * d, c, d);
  for (int j = 1; j < k; j++) {
    double tot = 0;
    for (int64_t i = 0; i < n; i++) tot += mind[i];
    double r = dn_rng_uniform(&rng) * tot, acc = 0;
    int64_t pick = n - 1;
    for (int64_t i = 0; i < n; i++) { acc += mind[i]; if (acc >= r) { pick = i; break; } }
    memcpy(c + (size_t)j * d, x + pick * d, sizeof(float) * d);
    for (int64_t i = 0; i < n; i++) {
      float s = dn_l2sq(x + i * d, c + (size_t)j * d, d);
      if (s < mind[i]) mind[i] = s;
    }
  }

  for (int it = 0; it < iters; it++) {
    memset(cnt, 0, sizeof(int64_t) * k);
    memset(sum, 0, sizeof(double) * (size_t)k * d);
    int64_t changed = 0;
    for (int64_t i = 0; i < n; i++) {
      int a = nearest(x + i * d, c, k, d, NULL);
      if (it == 0 || a != assign[i]) changed++;
      assign[i] = a;
      cnt[a]++;
      for (int t = 0; t < d; t++) sum[(size_t)a * d + t] += x[i * d + t];
    }
    for (int j = 0; j < k; j++) {
      if (cnt[j] > 0) {
        for (int t = 0; t < d; t++) c[(size_t)j * d + t] = (float)(sum[(size_t)j * d + t] / cnt[j]);
      } else {
        /* Empty cluster: split the largest one by copying its centroid and
        ** nudging both copies apart slightly. */
        int big = 0;
        for (int t = 1; t < k; t++) if (cnt[t] > cnt[big]) big = t;
        for (int t = 0; t < d; t++) {
          float v = c[(size_t)big * d + t], eps = 1e-4f * (float)(dn_rng_uniform(&rng) - 0.5);
          c[(size_t)j * d + t] = v * (1 + eps) + eps;
          c[(size_t)big * d + t] = v * (1 - eps) - eps;
        }
        cnt[j] = cnt[big] / 2; cnt[big] -= cnt[j];
      }
    }
    if (it > 0 && changed == 0) break;
  }
  free(mind); free(assign); free(cnt); free(sum);
}

typedef struct {
  const float *x; int64_t n, ns; int dim, dsub; int64_t *idx;
  int iters; uint64_t seed; float *cent;
} TrainCtx;

static void train_subspace(void *vctx, int64_t j, int thread) {
  TrainCtx *t = (TrainCtx *)vctx; (void)thread;
  int d = t->dsub;
  float *sub = (float *)malloc(sizeof(float) * (size_t)t->ns * d);
  for (int64_t i = 0; i < t->ns; i++)
    memcpy(sub + i * d, t->x + t->idx[i] * t->dim + j * d, sizeof(float) * d);
  kmeans(sub, t->ns, d, PQ_KSUB, t->iters, dn_hash2(t->seed, (uint64_t)j),
         t->cent + (size_t)j * PQ_KSUB * d);
  free(sub);
}

int pq_train(PQ *pq, const float *x, int64_t n, int dim, int m,
             int64_t max_train, int iters, uint64_t seed, int nthreads) {
  if (pq_init(pq, dim, m)) return 1;
  /* Deterministic sample: Fisher-Yates prefix of a shuffled index list. */
  int64_t ns = (max_train > 0 && n > max_train) ? max_train : n;
  int64_t *idx = (int64_t *)malloc(sizeof(int64_t) * n);
  for (int64_t i = 0; i < n; i++) idx[i] = i;
  uint64_t rng = seed ^ 0x5eedULL;
  for (int64_t i = 0; i < ns; i++) {
    int64_t j = i + (int64_t)(dn_rng_next(&rng) % (uint64_t)(n - i));
    int64_t tmp = idx[i]; idx[i] = idx[j]; idx[j] = tmp;
  }
  TrainCtx t = { x, n, ns, dim, pq->dsub, idx, iters, seed, pq->cent };
  dn_parallel_for(m, nthreads, 1, train_subspace, &t);
  free(idx);
  return 0;
}

/* ------------------------------------------------- encode / decode */

void pq_encode(const PQ *pq, const float *x, uint8_t *code) {
  for (int j = 0; j < pq->m; j++)
    code[j] = (uint8_t)nearest(x + j * pq->dsub, pq->cent + (size_t)j * PQ_KSUB * pq->dsub,
                               PQ_KSUB, pq->dsub, NULL);
}

typedef struct { const PQ *pq; const float *x; uint8_t *codes; } EncCtx;

static void encode_one(void *vctx, int64_t i, int thread) {
  EncCtx *e = (EncCtx *)vctx; (void)thread;
  pq_encode(e->pq, e->x + i * e->pq->dim, e->codes + i * e->pq->m);
}

void pq_encode_many(const PQ *pq, const float *x, int64_t n, uint8_t *codes, int nthreads) {
  EncCtx e = { pq, x, codes };
  dn_parallel_for(n, nthreads, 256, encode_one, &e);
}

void pq_decode(const PQ *pq, const uint8_t *code, float *out) {
  for (int j = 0; j < pq->m; j++)
    memcpy(out + j * pq->dsub, pq->cent + ((size_t)j * PQ_KSUB + code[j]) * pq->dsub,
           sizeof(float) * pq->dsub);
}

void pq_adc_table(const PQ *pq, const float *q, int metric, float *tab) {
  int d = pq->dsub;
  for (int j = 0; j < pq->m; j++) {
    const float *qs = q + j * d, *cs = pq->cent + (size_t)j * PQ_KSUB * d;
    float *row = tab + j * PQ_KSUB;
    for (int c = 0; c < PQ_KSUB; c++) {
      if (metric == METRIC_L2) row[c] = dn_l2sq(qs, cs + c * d, d);
      else row[c] = -dn_dot(qs, cs + c * d, d);
    }
  }
  if (metric != METRIC_L2)
    for (int c = 0; c < PQ_KSUB; c++) tab[c] += 1.0f;   /* fold "1 -" into subspace 0 */
}
