/*
** late_codec.h -- k-means and the PLAID residual codec (fast-plaid
** semantics; see docs/plaid.md "Residual codec" and "Encoding a vector").
*/
#ifndef LATE_CODEC_H
#define LATE_CODEC_H

#include <stdint.h>

typedef struct LtCodec {
  int dim, nbits, K;
  int rbytes;                 /* packed residual bytes per vector: dim*nbits/8 */
  float *centroids;           /* K*dim, unit, float16-rounded (owned) */
  float cutoffs[16];          /* 2^nbits - 1 used */
  float weights[16];          /* 2^nbits used */
  float cluster_threshold;    /* 0.75 quantile of held-out residual norms */
  float lut[256][8];          /* byte value -> bucket weights of its 8/nbits dims */
} LtCodec;

/* Fill lut[][] from weights[] (call after setting nbits and weights). */
void lc_init_lut(LtCodec *c);

/* Reconstruct a vector: normalize(centroid[code] + weights[buckets]). */
void lc_decode(const LtCodec *c, uint32_t code, const uint8_t *res, float *out);

/* Pack the residual of e against centroid[code] (fast-plaid bit order). */
void lc_encode_residual(const LtCodec *c, const float *e, uint32_t code, uint8_t *out);

/* For each of the n rows of X, the index of the centroid maximising
** x.c + bias[c] (bias NULL means 0). With bias = -|c|^2/2 this is the
** nearest centroid in Euclidean distance. out_score may be NULL. */
void lc_assign(const float *X, int64_t n, const float *C, const float *bias, int K, int dim,
               uint32_t *out, float *out_score, int nthreads);

/* Two-level assignment: the aprobe best coarse cells (by dot product with
** the G coarse centroids), then the best fine centroid inside them. Fine
** centroids are grouped by cell: cell g owns [cell_start[g], cell_start[g+1]). */
void lc_assign_hier(const float *X, int64_t n, const float *coarse, int G, const float *C,
                    const uint32_t *cell_start, int aprobe, int dim, uint32_t *out, int nthreads);

/* Decode against an explicit centroid vector. */
void lc_decode_with(const LtCodec *c, const float *cen, const uint8_t *res, float *out);

/* Lloyd's k-means (fast-plaid FastKMeans): K distinct random sample points
** as the initial centroids, `iters` iterations, empty clusters re-seeded with
** random points, then L2-normalised and rounded to float16. C_out is K*dim.
** Returns 0 on success. */
int lc_kmeans(const float *X, int64_t n, int K, int dim, int iters, uint64_t seed,
              int nthreads, float *C_out);

/* Bucket cutoffs/weights and cluster_threshold from held-out vectors (codes
** already assigned). c->centroids, dim, nbits and K must be set. */
int lc_train_codec(LtCodec *c, const float *X, int64_t n, const uint32_t *codes);

#endif
