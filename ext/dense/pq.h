/*
** pq.h — product quantisation with 8-bit sub-codes (256 centroids per
** subspace), trained with k-means++ / Lloyd iterations, and asymmetric
** distance computation (ADC) through per-query lookup tables.
*/
#ifndef DENSE_PQ_H
#define DENSE_PQ_H

#include <stdint.h>

#define PQ_KSUB 256

typedef struct PQ {
  int dim;      /* full dimension */
  int m;        /* number of subspaces (= bytes per code) */
  int dsub;     /* dim / m */
  float *cent;  /* centroids, laid out [m][PQ_KSUB][dsub] */
  float *rot;   /* optional OPQ rotation, dim x dim row-major (y = x R); NULL = none */
} PQ;

/* Train on n vectors x[n*dim] (row-major). At most max_train vectors are
** used, chosen with a deterministic shuffle from seed. With opq_iters > 0 an
** OPQ rotation is learned first (stored in pq->rot). Returns 0 on success. */
int pq_train(PQ *pq, const float *x, int64_t n, int dim, int m,
             int64_t max_train, int iters, int opq_iters, uint64_t seed, int nthreads);

/* Allocate an untrained PQ (centroids zeroed), e.g. before loading. */
int pq_init(PQ *pq, int dim, int m);
void pq_free(PQ *pq);

/* Round every centroid through float16, so that the in-memory codebook equals
** the one stored on disk (which is float16). Call before encoding. */
void pq_round_f16(PQ *pq);

void pq_encode(const PQ *pq, const float *x, uint8_t *code);
void pq_encode_many(const PQ *pq, const float *x, int64_t n, uint8_t *codes, int nthreads);
/* Reconstruction in the rotated space (distances are unchanged by R). */
void pq_decode(const PQ *pq, const uint8_t *code, float *out);

/* Build the ADC table tab[m*256] for query q, such that
**   distance(q, decode(code)) == pq_adc(tab, code, m)
** for the given metric (COSINE/IP: 1 - <q,x>; L2: squared distance); q is
** rotated first when the PQ has an OPQ rotation. The
** constant 1 of the cosine/IP distance is folded into subspace 0. */
void pq_adc_table(const PQ *pq, const float *q, int metric, float *tab);

static inline float pq_adc(const float *tab, const uint8_t *code, int m) {
  float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
  int j = 0;
  for (; j + 4 <= m; j += 4) {
    s0 += tab[(j + 0) * PQ_KSUB + code[j + 0]];
    s1 += tab[(j + 1) * PQ_KSUB + code[j + 1]];
    s2 += tab[(j + 2) * PQ_KSUB + code[j + 2]];
    s3 += tab[(j + 3) * PQ_KSUB + code[j + 3]];
  }
  for (; j < m; j++) s0 += tab[j * PQ_KSUB + code[j]];
  return (s0 + s1) + (s2 + s3);
}

#endif
