/*
** late_codec.c -- see late_codec.h.
*/
#include "late_codec.h"
#include "late_common.h"

#include <float.h>
#include <stdlib.h>
#include <string.h>

void lc_init_lut(LtCodec *c) {
  int nb = c->nbits, per = 8 / nb;
  for (int v = 0; v < 256; v++) {
    for (int k = 0; k < per; k++) {
      int b = 0;
      for (int t = 0; t < nb; t++) b |= ((v >> (7 - (k * nb + t))) & 1) << t;
      c->lut[v][k] = c->weights[b];
    }
    for (int k = per; k < 8; k++) c->lut[v][k] = 0;
  }
}

void lc_decode(const LtCodec *c, uint32_t code, const uint8_t *res, float *out) {
  lc_decode_with(c, c->centroids + (size_t)code * c->dim, res, out);
}

void lc_decode_with(const LtCodec *c, const float *cen, const uint8_t *res, float *out) {
  int per = 8 / c->nbits, d = 0;
  for (int j = 0; j < c->rbytes; j++) {
    const float *w = c->lut[res[j]];
    for (int k = 0; k < per; k++, d++) out[d] = cen[d] + w[k];
  }
  lt_normalize(out, c->dim);
}

void lc_encode_residual(const LtCodec *c, const float *e, uint32_t code, uint8_t *out) {
  const float *cen = c->centroids + (size_t)code * c->dim;
  int nb = c->nbits, ncut = (1 << nb) - 1;
  memset(out, 0, c->rbytes);
  for (int d = 0; d < c->dim; d++) {
    float r = e[d] - cen[d];
    int b = 0;
    while (b < ncut && c->cutoffs[b] < r) b++;     /* torch.bucketize(right=False) */
    for (int t = 0; t < nb; t++) {
      if ((b >> t) & 1) {
        int p = d * nb + t;
        out[p >> 3] |= (uint8_t)(1u << (7 - (p & 7)));
      }
    }
  }
}

/* ------------------------------------------------------------ assignment */

#define LC_CB 64      /* centroids per block */
#define LC_PB 4       /* points per micro-kernel */

typedef struct {
  const float *X; int64_t n; int dim, K, nblk;
  const float *CT;       /* nblk * dim * LC_CB, transposed blocks */
  const float *B;        /* nblk * LC_CB biases, -inf for padding */
  uint32_t *out; float *score;
} AssignCtx;

static void assign_range(void *vctx, int64_t lo, int64_t hi, int tid) {
  (void)tid;
  AssignCtx *a = (AssignCtx *)vctx;
  const int dim = a->dim;
  float acc[LC_PB][LC_CB];
  for (int64_t p = lo; p < hi; p += LC_PB) {
    int np = (int)(hi - p < LC_PB ? hi - p : LC_PB);
    const float *x[LC_PB];
    for (int i = 0; i < LC_PB; i++) x[i] = a->X + (size_t)(p + (i < np ? i : 0)) * dim;
    float best[LC_PB]; uint32_t bi[LC_PB];
    for (int i = 0; i < LC_PB; i++) { best[i] = -FLT_MAX; bi[i] = 0; }
    for (int b = 0; b < a->nblk; b++) {
      const float *ct = a->CT + (size_t)b * dim * LC_CB;
      const float *bias = a->B + (size_t)b * LC_CB;
      for (int i = 0; i < LC_PB; i++) memcpy(acc[i], bias, sizeof(float) * LC_CB);
      for (int d = 0; d < dim; d++) {
        const float *row = ct + (size_t)d * LC_CB;
        float x0 = x[0][d], x1 = x[1][d], x2 = x[2][d], x3 = x[3][d];
        for (int j = 0; j < LC_CB; j++) {
          float r = row[j];
          acc[0][j] += x0 * r; acc[1][j] += x1 * r; acc[2][j] += x2 * r; acc[3][j] += x3 * r;
        }
      }
      for (int i = 0; i < np; i++) {
        float bv = best[i]; uint32_t bj = bi[i];
        for (int j = 0; j < LC_CB; j++) if (acc[i][j] > bv) { bv = acc[i][j]; bj = (uint32_t)(b * LC_CB + j); }
        best[i] = bv; bi[i] = bj;
      }
    }
    for (int i = 0; i < np; i++) {
      a->out[p + i] = bi[i];
      if (a->score) a->score[p + i] = best[i];
    }
  }
}

void lc_assign(const float *X, int64_t n, const float *C, const float *bias, int K, int dim,
               uint32_t *out, float *out_score, int nthreads) {
  AssignCtx a;
  a.X = X; a.n = n; a.dim = dim; a.K = K; a.out = out; a.score = out_score;
  a.nblk = (K + LC_CB - 1) / LC_CB;
  float *CT = (float *)calloc((size_t)a.nblk * dim * LC_CB, sizeof(float));
  float *B = (float *)malloc((size_t)a.nblk * LC_CB * sizeof(float));
  if (!CT || !B) { free(CT); free(B); for (int64_t i = 0; i < n; i++) out[i] = 0; return; }
  for (int c = 0; c < a.nblk * LC_CB; c++) {
    int b = c / LC_CB, j = c % LC_CB;
    if (c < K) {
      for (int d = 0; d < dim; d++) CT[((size_t)b * dim + d) * LC_CB + j] = C[(size_t)c * dim + d];
      B[c] = bias ? bias[c] : 0.0f;
    } else B[c] = -FLT_MAX / 4;
  }
  a.CT = CT; a.B = B;
  lt_parallel_for(n, 256, nthreads, assign_range, &a);
  free(CT); free(B);
}

typedef struct {
  const float *X, *coarse, *C; const uint32_t *start; int G, aprobe, dim; uint32_t *out;
} HierCtx;

static void hier_range(void *vctx, int64_t lo, int64_t hi, int tid) {
  (void)tid;
  HierCtx *h = (HierCtx *)vctx;
  int ap = h->aprobe < 1 ? 1 : h->aprobe > h->G ? h->G : h->aprobe;
  float bs[64]; int bi[64];
  if (ap > 64) ap = 64;
  for (int64_t p = lo; p < hi; p++) {
    const float *x = h->X + (size_t)p * h->dim;
    int n = 0;
    for (int g = 0; g < h->G; g++) {
      if (h->start[g + 1] == h->start[g]) continue;
      float s = lt_dot(x, h->coarse + (size_t)g * h->dim, h->dim);
      if (n == ap && s <= bs[n - 1]) continue;
      int j = n < ap ? n++ : ap - 1;
      while (j > 0 && bs[j - 1] < s) { bs[j] = bs[j - 1]; bi[j] = bi[j - 1]; j--; }
      bs[j] = s; bi[j] = g;
    }
    float best = -FLT_MAX; uint32_t arg = 0;
    for (int k = 0; k < n; k++)
      for (uint32_t c = h->start[bi[k]]; c < h->start[bi[k] + 1]; c++) {
        float s = lt_dot(x, h->C + (size_t)c * h->dim, h->dim);
        if (s > best) { best = s; arg = c; }
      }
    h->out[p] = arg;
  }
}

void lc_assign_hier(const float *X, int64_t n, const float *coarse, int G, const float *C,
                    const uint32_t *cell_start, int aprobe, int dim, uint32_t *out, int nthreads) {
  HierCtx h = { X, coarse, C, cell_start, G, aprobe, dim, out };
  lt_parallel_for(n, 512, nthreads, hier_range, &h);
}

/* --------------------------------------------------------------- k-means */

int lc_kmeans(const float *X, int64_t n, int K, int dim, int iters, uint64_t seed,
              int nthreads, float *C) {
  if (n < K || K < 1) return 1;
  uint64_t rng = seed;
  /* K distinct random points: partial Fisher-Yates over the indices. */
  int64_t *idx = (int64_t *)malloc(sizeof(int64_t) * n);
  uint32_t *asg = (uint32_t *)malloc(sizeof(uint32_t) * n);
  float *bias = (float *)malloc(sizeof(float) * K);
  double *sum = (double *)malloc(sizeof(double) * (size_t)K * dim);
  int64_t *cnt = (int64_t *)malloc(sizeof(int64_t) * K);
  if (!idx || !asg || !bias || !sum || !cnt) { free(idx); free(asg); free(bias); free(sum); free(cnt); return 2; }
  for (int64_t i = 0; i < n; i++) idx[i] = i;
  for (int k = 0; k < K; k++) {
    int64_t j = k + (int64_t)lt_rng_below(&rng, (uint64_t)(n - k));
    int64_t t = idx[k]; idx[k] = idx[j]; idx[j] = t;
    memcpy(C + (size_t)k * dim, X + (size_t)idx[k] * dim, sizeof(float) * dim);
  }
  for (int it = 0; it < iters; it++) {
    for (int k = 0; k < K; k++) bias[k] = -0.5f * lt_dot(C + (size_t)k * dim, C + (size_t)k * dim, dim);
    lc_assign(X, n, C, bias, K, dim, asg, NULL, nthreads);
    memset(sum, 0, sizeof(double) * (size_t)K * dim);
    memset(cnt, 0, sizeof(int64_t) * K);
    for (int64_t i = 0; i < n; i++) {
      double *s = sum + (size_t)asg[i] * dim;
      const float *x = X + (size_t)i * dim;
      for (int d = 0; d < dim; d++) s[d] += x[d];
      cnt[asg[i]]++;
    }
    double shift = 0;
    for (int k = 0; k < K; k++) {
      float *c = C + (size_t)k * dim;
      if (cnt[k] == 0) {                 /* re-seed an empty cluster */
        const float *x = X + (size_t)lt_rng_below(&rng, (uint64_t)n) * dim;
        for (int d = 0; d < dim; d++) { shift += fabs((double)x[d] - c[d]); c[d] = x[d]; }
      } else {
        for (int d = 0; d < dim; d++) {
          float v = (float)(sum[(size_t)k * dim + d] / (double)cnt[k]);
          shift += fabs((double)v - c[d]);
          c[d] = v;
        }
      }
    }
    if (shift < 1e-8) break;
  }
  for (int k = 0; k < K; k++) {
    float *c = C + (size_t)k * dim;
    lt_normalize(c, dim);
    for (int d = 0; d < dim; d++) c[d] = lt_h2f(lt_f2h(c[d]));
  }
  free(idx); free(asg); free(bias); free(sum); free(cnt);
  return 0;
}

/* ----------------------------------------------------------------- codec */

static int cmp_float(const void *a, const void *b) {
  float x = *(const float *)a, y = *(const float *)b;
  return x < y ? -1 : x > y;
}

/* numpy.quantile(method="linear") over sorted v[0..n). */
static float quantile_sorted(const float *v, int64_t n, double q) {
  if (n <= 0) return 0;
  double p = q * (double)(n - 1);
  int64_t lo = (int64_t)floor(p), hi = (int64_t)ceil(p);
  return (float)(v[lo] + (p - (double)lo) * ((double)v[hi] - v[lo]));
}

int lc_train_codec(LtCodec *c, const float *X, int64_t n, const uint32_t *codes) {
  int dim = c->dim, B = 1 << c->nbits;
  float *r = (float *)malloc(sizeof(float) * (size_t)n * dim);
  float *norms = (float *)malloc(sizeof(float) * (size_t)(n > 0 ? n : 1));
  if (!r || !norms) { free(r); free(norms); return 1; }
  for (int64_t i = 0; i < n; i++) {
    const float *cen = c->centroids + (size_t)codes[i] * dim;
    double s = 0;
    for (int d = 0; d < dim; d++) {
      float v = X[(size_t)i * dim + d] - cen[d];
      r[(size_t)i * dim + d] = v; s += (double)v * v;
    }
    norms[i] = (float)sqrt(s);
  }
  qsort(norms, (size_t)n, sizeof(float), cmp_float);
  c->cluster_threshold = quantile_sorted(norms, n, 0.75);
  qsort(r, (size_t)n * dim, sizeof(float), cmp_float);
  for (int i = 1; i < B; i++) c->cutoffs[i - 1] = quantile_sorted(r, n * dim, (double)i / B);
  for (int i = 0; i < B; i++) c->weights[i] = quantile_sorted(r, n * dim, (i + 0.5) / B);
  free(r); free(norms);
  c->rbytes = dim * c->nbits / 8;
  lc_init_lut(c);
  return 0;
}
