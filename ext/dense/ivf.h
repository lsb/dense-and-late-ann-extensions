/*
** ivf.h — coarse quantiser for the IVF layout of dense_ann: k-means over
** full vectors and nearest-centroid assignment, with a blocked dot-product
** kernel (several points share each centroid load).
*/
#ifndef DENSE_IVF_H
#define DENSE_IVF_H

#include <stdint.h>

/* Train k centroids c[k*d] with Lloyd's algorithm on (a deterministic sample
** of at most max_train of) x[n*d]. k-means++-lite init: random distinct
** points. Returns 0 on success. */
int dnivf_kmeans(const float *x, int64_t n, int d, int k, int64_t max_train, int iters,
                 uint64_t seed, int nthreads, int verbose, float *c);

/* assign[i] = argmin_j ||x_i - c_j||^2 for all n points. */
void dnivf_assign(const float *x, int64_t n, int d, const float *c, int k, int nthreads, int32_t *assign);

/* Indices of the `np` centroids closest (L2) to q, closest first.
** cn[j] = ||c_j||^2 must be precomputed. Returns the number written. */
int dnivf_probe(const float *q, int d, const float *c, const float *cn, int k, int np, int32_t *out,
                float *dist_out);

#endif
