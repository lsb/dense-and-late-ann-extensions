/*
** pq.c — product quantiser training, encoding and ADC tables.
*/
#include "pq.h"
#include "dense_common.h"

#include <float.h>
#include <stdio.h>

int dnpq_init(PQ *pq, int dim, int m) {
  memset(pq, 0, sizeof *pq);
  if (m <= 0 || dim % m != 0) return 1;
  pq->dim = dim; pq->m = m; pq->dsub = dim / m;
  pq->cent = (float *)calloc((size_t)m * PQ_KSUB * pq->dsub, sizeof(float));
  return pq->cent ? 0 : 1;
}

void dnpq_free(PQ *pq) {
  free(pq->cent);
  free(pq->rot);
  memset(pq, 0, sizeof *pq);
}

void dnpq_round_f16(PQ *pq) {
  size_t n = (size_t)pq->m * PQ_KSUB * pq->dsub;
  for (size_t i = 0; i < n; i++) pq->cent[i] = dn_f16_to_f32(dn_f32_to_f16(pq->cent[i]));
}

/* ------------------------------------------------------------ k-means */

/* Nearest-centroid search for short subvectors. Centroids are passed
** transposed, cT[t*k + j] = c_j[t], with squared norms cn[j], so the inner
** loop runs over centroids and vectorises well even for d = 6:
**   argmin_j ||x - c_j||^2 = argmin_j (||c_j||^2 - 2 <x, c_j>). */
static int nearest_t(const float *x, const float *cT, const float *cn, int k, int d, float *acc) {
  for (int j = 0; j < k; j++) acc[j] = cn[j];
  for (int t = 0; t < d; t++) {
    float xt = -2.0f * x[t];
    const float *row = cT + (size_t)t * k;
    for (int j = 0; j < k; j++) acc[j] += xt * row[j];
  }
  int best = 0; float bd = acc[0];
  for (int j = 1; j < k; j++) if (acc[j] < bd) { bd = acc[j]; best = j; }
  return best;
}

/* Build the transposed copy and norms used by nearest_t. */
static void transpose_centroids(const float *c, int k, int d, float *cT, float *cn) {
  for (int j = 0; j < k; j++) {
    float s = 0;
    for (int t = 0; t < d; t++) { float v = c[(size_t)j * d + t]; cT[(size_t)t * k + j] = v; s += v * v; }
    cn[j] = s;
  }
}

/* Lloyd's k-means for one subspace: data x[n*d] -> centroids c[k*d].
** k-means++ initialisation, deterministic given seed. */
static void kmeans(const float *x, int64_t n, int d, int k, int iters, uint64_t seed, float *c, int warm) {
  uint64_t rng = seed;
  if (n <= k && !warm) {
    /* Fewer points than centroids: copy points, duplicate the rest. */
    for (int j = 0; j < k; j++) memcpy(c + (size_t)j * d, x + (size_t)(n ? j % n : 0) * d, sizeof(float) * d);
    return;
  }
  float *mind = (float *)malloc(sizeof(float) * n);
  int *assign = (int *)malloc(sizeof(int) * n);
  int64_t *cnt = (int64_t *)malloc(sizeof(int64_t) * k);
  double *sum = (double *)malloc(sizeof(double) * (size_t)k * d);
  float *cT = (float *)malloc(sizeof(float) * (size_t)k * d);
  float *cn = (float *)malloc(sizeof(float) * k), *acc = (float *)malloc(sizeof(float) * k);

  if (!warm) {
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
  }

  for (int it = 0; it < iters; it++) {
    memset(cnt, 0, sizeof(int64_t) * k);
    memset(sum, 0, sizeof(double) * (size_t)k * d);
    int64_t changed = 0;
    transpose_centroids(c, k, d, cT, cn);
    for (int64_t i = 0; i < n; i++) {
      int a = nearest_t(x + i * d, cT, cn, k, d, acc);
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
  free(mind); free(assign); free(cnt); free(sum); free(cT); free(cn); free(acc);
}

typedef struct {
  const float *s; int64_t ns; int dim, dsub;   /* training sample, ns x dim */
  int iters, warm; uint64_t seed; float *cent;
} TrainCtx;

static void train_subspace(void *vctx, int64_t j, int thread) {
  TrainCtx *t = (TrainCtx *)vctx; (void)thread;
  int d = t->dsub;
  float *sub = (float *)malloc(sizeof(float) * (size_t)t->ns * d);
  for (int64_t i = 0; i < t->ns; i++)
    memcpy(sub + i * d, t->s + i * t->dim + j * d, sizeof(float) * d);
  kmeans(sub, t->ns, d, PQ_KSUB, t->iters, dn_hash2(t->seed, (uint64_t)j),
         t->cent + (size_t)j * PQ_KSUB * d, t->warm);
  free(sub);
}

/* y = x R for a row vector x (R is dim x dim, row-major). */
static void rotate(const float *R, const float *x, float *y, int d) {
  for (int j = 0; j < d; j++) y[j] = 0;
  for (int i = 0; i < d; i++) {
    float xi = x[i];
    const float *row = R + (size_t)i * d;
    for (int j = 0; j < d; j++) y[j] += xi * row[j];
  }
}

typedef struct { const float *R, *x; float *y; int d; } RotCtx;
static void rotate_task(void *vctx, int64_t i, int thread) {
  RotCtx *c = (RotCtx *)vctx; (void)thread;
  rotate(c->R, c->x + i * c->d, c->y + i * c->d, c->d);
}

/* R = U V^T where M = U S V^T: the orthogonal matrix closest to M
** (orthogonal Procrustes). One-sided Jacobi SVD (Hestenes) in double:
** rotate column pairs of A = M until orthogonal, accumulating V; then the
** columns of A are U scaled by the singular values. */
static void polar_orthogonal(const double *M, int d, float *R) {
  double *A = (double *)malloc(sizeof(double) * (size_t)d * d);   /* column-major */
  double *V = (double *)calloc((size_t)d * d, sizeof(double));
  for (int i = 0; i < d; i++)
    for (int j = 0; j < d; j++) A[(size_t)j * d + i] = M[(size_t)i * d + j];
  for (int i = 0; i < d; i++) V[(size_t)i * d + i] = 1;
  for (int sweep = 0; sweep < 30; sweep++) {
    double off = 0;
    for (int p = 0; p < d - 1; p++) {
      for (int q = p + 1; q < d; q++) {
        double *ap = A + (size_t)p * d, *aq = A + (size_t)q * d;
        double al = 0, be = 0, ga = 0;
        for (int k = 0; k < d; k++) { al += ap[k] * ap[k]; be += aq[k] * aq[k]; ga += ap[k] * aq[k]; }
        if (fabs(ga) <= 1e-15 * sqrt(al * be) || ga == 0) continue;
        double r = fabs(ga) / sqrt(al * be);
        if (r > off) off = r;
        double zeta = (be - al) / (2 * ga);
        double t = (zeta >= 0 ? 1.0 : -1.0) / (fabs(zeta) + sqrt(1 + zeta * zeta));
        double c = 1 / sqrt(1 + t * t), s = c * t;
        double *vp = V + (size_t)p * d, *vq = V + (size_t)q * d;
        for (int k = 0; k < d; k++) {
          double x = ap[k], y = aq[k];
          ap[k] = c * x - s * y; aq[k] = s * x + c * y;
          x = vp[k]; y = vq[k];
          vp[k] = c * x - s * y; vq[k] = s * x + c * y;
        }
      }
    }
    if (off < 1e-10) break;
  }
  /* U columns = A columns / norms; R = U V^T. */
  for (int j = 0; j < d; j++) {
    double *a = A + (size_t)j * d, nrm = 0;
    for (int k = 0; k < d; k++) nrm += a[k] * a[k];
    nrm = nrm > 0 ? 1 / sqrt(nrm) : 0;
    for (int k = 0; k < d; k++) a[k] *= nrm;
  }
  for (int i = 0; i < d; i++)
    for (int j = 0; j < d; j++) {
      double s = 0;
      for (int k = 0; k < d; k++) s += A[(size_t)k * d + i] * V[(size_t)k * d + j];
      R[(size_t)i * d + j] = (float)s;
    }
  free(A); free(V);
}

typedef struct { const float *S, *Y; double *M; int64_t n; int d; } GramCtx;
static void gram_row(void *vctx, int64_t i, int thread) {   /* M[i,:] = sum_r S[r,i] Y[r,:] */
  GramCtx *g = (GramCtx *)vctx; (void)thread;
  double *row = g->M + (size_t)i * g->d;
  for (int j = 0; j < g->d; j++) row[j] = 0;
  float *acc = (float *)calloc(g->d, sizeof(float));
  for (int64_t r = 0; r < g->n; r++) {
    float s = g->S[r * g->d + i];
    const float *y = g->Y + r * g->d;
    for (int j = 0; j < g->d; j++) acc[j] += s * y[j];
    if ((r & 1023) == 1023) { for (int j = 0; j < g->d; j++) { row[j] += acc[j]; acc[j] = 0; } }
  }
  for (int j = 0; j < g->d; j++) row[j] += acc[j];
  free(acc);
}

int dnpq_train(PQ *pq, const float *x, int64_t n, int dim, int m,
             int64_t max_train, int iters, int opq_iters, uint64_t seed, int nthreads) {
  if (dnpq_init(pq, dim, m)) return 1;
  /* Deterministic sample: Fisher-Yates prefix of a shuffled index list. */
  int64_t ns = (max_train > 0 && n > max_train) ? max_train : n;
  int64_t *idx = (int64_t *)malloc(sizeof(int64_t) * n);
  for (int64_t i = 0; i < n; i++) idx[i] = i;
  uint64_t rng = seed ^ 0x5eedULL;
  for (int64_t i = 0; i < ns; i++) {
    int64_t j = i + (int64_t)(dn_rng_next(&rng) % (uint64_t)(n - i));
    int64_t tmp = idx[i]; idx[i] = idx[j]; idx[j] = tmp;
  }
  float *S = (float *)malloc(sizeof(float) * (size_t)ns * dim);
  for (int64_t i = 0; i < ns; i++) memcpy(S + i * dim, x + idx[i] * dim, sizeof(float) * dim);
  free(idx);
  float *Y = S;
  int warm = 0;

  if (opq_iters > 0) {
    /* OPQ (Ge et al. 2013, non-parametric): alternate a few k-means
    ** iterations on the rotated sample with a Procrustes update of R that
    ** best maps the sample onto its reconstructions. Uses a subsample. */
    int64_t no = ns < 20000 ? ns : 20000;
    pq->rot = (float *)calloc((size_t)dim * dim, sizeof(float));
    for (int i = 0; i < dim; i++) pq->rot[(size_t)i * dim + i] = 1;
    Y = (float *)malloc(sizeof(float) * (size_t)ns * dim);
    float *Yh = (float *)malloc(sizeof(float) * (size_t)no * dim);
    double *M = (double *)malloc(sizeof(double) * (size_t)dim * dim);
    uint8_t *code = (uint8_t *)malloc(m);
    PQ plain = *pq; plain.rot = NULL;           /* codebook without rotation */
    for (int it = 0; it < opq_iters; it++) {
      RotCtx rc = { pq->rot, S, Y, dim };
      dn_parallel_for(no, nthreads, 256, rotate_task, &rc);
      TrainCtx t = { Y, no, dim, pq->dsub, 4, it > 0, seed + (uint64_t)it, pq->cent };
      dn_parallel_for(m, nthreads, 1, train_subspace, &t);
      for (int64_t i = 0; i < no; i++) {
        dnpq_encode(&plain, Y + i * dim, code);
        dnpq_decode(&plain, code, Yh + i * dim);
      }
      GramCtx g = { S, Yh, M, no, dim };
      dn_parallel_for(dim, nthreads, 1, gram_row, &g);
      polar_orthogonal(M, dim, pq->rot);
    }
    /* The rotation is stored as float16; use exactly that. */
    for (size_t i = 0; i < (size_t)dim * dim; i++) pq->rot[i] = dn_f16_to_f32(dn_f32_to_f16(pq->rot[i]));
    RotCtx rc = { pq->rot, S, Y, dim };
    dn_parallel_for(ns, nthreads, 256, rotate_task, &rc);
    free(Yh); free(M); free(code);
    warm = 1;
  }
  TrainCtx t = { Y, ns, dim, pq->dsub, iters, warm, seed, pq->cent };
  dn_parallel_for(m, nthreads, 1, train_subspace, &t);
  if (Y != S) free(Y);
  free(S);
  return 0;
}

/* ------------------------------------------------- encode / decode */

/* Encoding uses transposed centroids; they are built on the fly per call
** for single vectors and once per batch in dnpq_encode_many. */
static void encode_t(const PQ *pq, const float *cT, const float *cn, const float *x, uint8_t *code) {
  float acc[PQ_KSUB];
  for (int j = 0; j < pq->m; j++)
    code[j] = (uint8_t)nearest_t(x + j * pq->dsub, cT + (size_t)j * PQ_KSUB * pq->dsub,
                                 cn + (size_t)j * PQ_KSUB, PQ_KSUB, pq->dsub, acc);
}

static void make_transposed(const PQ *pq, float **pcT, float **pcn) {
  float *cT = (float *)malloc(sizeof(float) * (size_t)pq->m * PQ_KSUB * pq->dsub);
  float *cn = (float *)malloc(sizeof(float) * (size_t)pq->m * PQ_KSUB);
  for (int j = 0; j < pq->m; j++)
    transpose_centroids(pq->cent + (size_t)j * PQ_KSUB * pq->dsub, PQ_KSUB, pq->dsub,
                        cT + (size_t)j * PQ_KSUB * pq->dsub, cn + (size_t)j * PQ_KSUB);
  *pcT = cT; *pcn = cn;
}

void dnpq_encode(const PQ *pq, const float *x, uint8_t *code) {
  float *cT, *cn, *y = NULL;
  make_transposed(pq, &cT, &cn);
  if (pq->rot) { y = (float *)malloc(sizeof(float) * pq->dim); rotate(pq->rot, x, y, pq->dim); x = y; }
  encode_t(pq, cT, cn, x, code);
  free(cT); free(cn); free(y);
}

typedef struct { const PQ *pq; const float *x; uint8_t *codes; const float *cT, *cn; } EncCtx;

static void encode_one(void *vctx, int64_t i, int thread) {
  EncCtx *e = (EncCtx *)vctx; (void)thread;
  const float *x = e->x + i * e->pq->dim;
  float y[4096];
  if (e->pq->rot && e->pq->dim <= 4096) { rotate(e->pq->rot, x, y, e->pq->dim); x = y; }
  encode_t(e->pq, e->cT, e->cn, x, e->codes + i * e->pq->m);
}

void dnpq_encode_many(const PQ *pq, const float *x, int64_t n, uint8_t *codes, int nthreads) {
  float *cT, *cn;
  make_transposed(pq, &cT, &cn);
  EncCtx e = { pq, x, codes, cT, cn };
  dn_parallel_for(n, nthreads, 256, encode_one, &e);
  free(cT); free(cn);
}

void dnpq_decode(const PQ *pq, const uint8_t *code, float *out) {
  for (int j = 0; j < pq->m; j++)
    memcpy(out + j * pq->dsub, pq->cent + ((size_t)j * PQ_KSUB + code[j]) * pq->dsub,
           sizeof(float) * pq->dsub);
}

void dnpq_adc_table(const PQ *pq, const float *q, int metric, float *tab) {
  int d = pq->dsub;
  float *y = NULL;
  if (pq->rot) { y = (float *)malloc(sizeof(float) * pq->dim); rotate(pq->rot, q, y, pq->dim); q = y; }
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
  free(y);
}
