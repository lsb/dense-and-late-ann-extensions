/*
** ivf.c — coarse k-means and assignment for the IVF layout. See ivf.h.
*/
#include "ivf.h"
#include "dense_common.h"

#include <float.h>
#include <stdio.h>

/* Nearest centroid for 4 points at once: every centroid row is loaded once
** and dotted with 4 points (argmin of ||c||^2 - 2<x,c>). */
static void nearest4(const float *x0, const float *x1, const float *x2, const float *x3, int d,
                     const float *c, const float *cn, int k, int32_t *out, float *dout) {
  float best[4] = { FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX };
  int32_t arg[4] = { 0, 0, 0, 0 };
  for (int j = 0; j < k; j++) {
    const float *cj = c + (size_t)j * d;
    float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    for (int t = 0; t < d; t++) {
      float v = cj[t];
      s0 += v * x0[t]; s1 += v * x1[t]; s2 += v * x2[t]; s3 += v * x3[t];
    }
    float e0 = cn[j] - 2 * s0, e1 = cn[j] - 2 * s1, e2 = cn[j] - 2 * s2, e3 = cn[j] - 2 * s3;
    if (e0 < best[0]) { best[0] = e0; arg[0] = j; }
    if (e1 < best[1]) { best[1] = e1; arg[1] = j; }
    if (e2 < best[2]) { best[2] = e2; arg[2] = j; }
    if (e3 < best[3]) { best[3] = e3; arg[3] = j; }
  }
  for (int i = 0; i < 4; i++) { out[i] = arg[i]; if (dout) dout[i] = best[i]; }
}

typedef struct { const float *x; int64_t n; int d; const float *c, *cn; int k; int32_t *assign; } AssignCtx;

/* Task i handles points 64*i .. 64*i+63. */
static void assign_task(void *vctx, int64_t blk, int thread) {
  AssignCtx *a = (AssignCtx *)vctx; (void)thread;
  int64_t s = blk * 64, e = s + 64 < a->n ? s + 64 : a->n;
  for (int64_t i = s; i < e; i += 4) {
    const float *p[4];
    int32_t out[4];
    for (int t = 0; t < 4; t++) p[t] = a->x + (size_t)(i + t < e ? i + t : i) * a->d;
    nearest4(p[0], p[1], p[2], p[3], a->d, a->c, a->cn, a->k, out, NULL);
    for (int t = 0; t < 4 && i + t < e; t++) a->assign[i + t] = out[t];
  }
}

static void norms(const float *c, int k, int d, float *cn) {
  for (int j = 0; j < k; j++) cn[j] = dn_dot(c + (size_t)j * d, c + (size_t)j * d, d);
}

void dnivf_assign(const float *x, int64_t n, int d, const float *c, int k, int nthreads, int32_t *assign) {
  float *cn = (float *)malloc(sizeof(float) * k);
  norms(c, k, d, cn);
  AssignCtx a = { x, n, d, c, cn, k, assign };
  dn_parallel_for((n + 63) / 64, nthreads, 4, assign_task, &a);
  free(cn);
}

int dnivf_kmeans(const float *x, int64_t n, int d, int k, int64_t max_train, int iters,
                 uint64_t seed, int nthreads, int verbose, float *c) {
  if (k <= 0 || n <= 0) return 1;
  /* Deterministic sample (partial Fisher-Yates). */
  int64_t ns = (max_train > 0 && n > max_train) ? max_train : n;
  int64_t *idx = (int64_t *)malloc(sizeof(int64_t) * n);
  for (int64_t i = 0; i < n; i++) idx[i] = i;
  uint64_t rng = seed ^ 0x1f5eedULL;
  for (int64_t i = 0; i < ns; i++) {
    int64_t j = i + (int64_t)(dn_rng_next(&rng) % (uint64_t)(n - i));
    int64_t t = idx[i]; idx[i] = idx[j]; idx[j] = t;
  }
  float *s = (float *)malloc(sizeof(float) * (size_t)ns * d);
  for (int64_t i = 0; i < ns; i++) memcpy(s + (size_t)i * d, x + (size_t)idx[i] * d, sizeof(float) * d);
  free(idx);
  /* Init: the first k sample points (the sample is already shuffled). */
  for (int j = 0; j < k; j++) memcpy(c + (size_t)j * d, s + (size_t)(j % ns) * d, sizeof(float) * d);

  int32_t *assign = (int32_t *)malloc(sizeof(int32_t) * ns);
  double *sum = (double *)malloc(sizeof(double) * (size_t)k * d);
  int64_t *cnt = (int64_t *)malloc(sizeof(int64_t) * k);
  for (int it = 0; it < iters; it++) {
    dnivf_assign(s, ns, d, c, k, nthreads, assign);
    memset(sum, 0, sizeof(double) * (size_t)k * d);
    memset(cnt, 0, sizeof(int64_t) * k);
    for (int64_t i = 0; i < ns; i++) {
      int a = assign[i];
      cnt[a]++;
      const float *p = s + (size_t)i * d;
      double *q = sum + (size_t)a * d;
      for (int t = 0; t < d; t++) q[t] += p[t];
    }
    int empty = 0;
    for (int j = 0; j < k; j++) {
      if (cnt[j] > 0) {
        for (int t = 0; t < d; t++) c[(size_t)j * d + t] = (float)(sum[(size_t)j * d + t] / cnt[j]);
      } else {
        /* Empty list: restart it at a random sample point. */
        empty++;
        int64_t r = (int64_t)(dn_rng_next(&rng) % (uint64_t)ns);
        memcpy(c + (size_t)j * d, s + (size_t)r * d, sizeof(float) * d);
      }
    }
    if (verbose) fprintf(stderr, "dense_ann: coarse k-means iter %d/%d (%d empty)\n", it + 1, iters, empty);
  }
  free(assign); free(sum); free(cnt); free(s);
  return 0;
}

int dnivf_probe(const float *q, int d, const float *c, const float *cn, int k, int np, int32_t *out,
                float *dist_out) {
  if (np > k) np = k;
  /* Bounded insertion into a sorted list; np is small (<= a few hundred). */
  int n = 0;
  for (int j = 0; j < k; j++) {
    float e = cn[j] - 2 * dn_dot(q, c + (size_t)j * d, d);
    if (n == np && e >= dist_out[np - 1]) continue;
    int i = n < np ? n++ : np - 1;
    while (i > 0 && dist_out[i - 1] > e) { dist_out[i] = dist_out[i - 1]; out[i] = out[i - 1]; i--; }
    dist_out[i] = e; out[i] = j;
  }
  return n;
}
