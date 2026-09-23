/*
** hnsw.h — in-memory HNSW graph construction over full-precision vectors.
**
** Only used at index build time. The persisted index keeps layer 0 (the
** full graph) plus a sample of high-level nodes as a cached entry set; see
** dense_ann.c and NOTES.md for why the upper layers are not persisted.
*/
#ifndef DENSE_HNSW_H
#define DENSE_HNSW_H

#include <stdint.h>

typedef struct Hnsw {
  /* parameters (set before hnsw_build) */
  int dim, M, M0, efc, metric;
  float alpha;          /* neighbour-pruning slack; 1.0 = classic HNSW heuristic */
  uint64_t seed;
  int64_t n;
  const float *x;       /* n*dim vectors, normalised if metric is cosine */

  /* result */
  uint8_t *level;       /* level[i] */
  uint32_t *l0;         /* layer 0 lists: n * (1+M0) u32, [count, ids...] */
  uint32_t **up;        /* up[i]: level[i] lists of (1+M) u32, or NULL */
  uint32_t entry;
  int maxlevel;

  void *priv;           /* locks and scratch (internal) */
} Hnsw;

/* Build the graph over all n vectors. Returns 0 on success. When verbose,
** progress is printed to stderr. */
int hnsw_build(Hnsw *h, int nthreads, int verbose);
void hnsw_free(Hnsw *h);

static inline uint32_t *hnsw_list(const Hnsw *h, uint32_t i, int lv) {
  return lv == 0 ? h->l0 + (size_t)i * (h->M0 + 1) : h->up[i] + (size_t)(lv - 1) * (h->M + 1);
}

/* Select at most cap neighbours out of cand ids (sorted by ascending distance
** dist[] to the base point) with the HNSW/Vamana pruning rule. vec(i) must
** return the full vector of id i. Writes the chosen ids to out and returns
** their number. Exposed so the SQLite-side incremental insert can reuse it. */
typedef const float *(*hnsw_vec_fn)(void *ctx, uint32_t id);
int hnsw_select(int metric, int dim, float alpha, const uint32_t *cand, const float *dist,
                int ncand, int cap, hnsw_vec_fn vec, void *vctx, uint32_t *out);

#endif
