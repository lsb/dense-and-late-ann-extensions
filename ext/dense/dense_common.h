/*
** dense_common.h — small shared helpers for the dense_ann extension:
** deterministic RNG, vector distances, float16 conversion, little-endian
** load/store, and an optional thread pool "parallel for".
**
** Everything here is header-only (static inline) so that each translation
** unit stays self-contained and the whole extension can also be compiled as
** a single unity build (e.g. with Emscripten).
*/
#ifndef DENSE_COMMON_H
#define DENSE_COMMON_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef DENSE_ANN_THREADS
#include <pthread.h>
#endif

/* Distance metrics. For COSINE, vectors are L2-normalised on insert and on
** query, and the distance is 1 - <a,b>. IP uses 1 - <a,b> without
** normalisation; L2 is the squared Euclidean distance. */
enum { METRIC_COSINE = 0, METRIC_IP = 1, METRIC_L2 = 2 };

/* ---------------------------------------------------------------- RNG */

/* SplitMix64: tiny, fast, deterministic across platforms. */
static inline uint64_t dn_rng_next(uint64_t *s) {
  uint64_t z = (*s += 0x9E3779B97F4A7C15ull);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

/* Uniform double in [0,1). */
static inline double dn_rng_uniform(uint64_t *s) {
  return (double)(dn_rng_next(s) >> 11) * (1.0 / 9007199254740992.0);
}

/* A stateless hash of (seed, i), used where a per-item random value must not
** depend on processing order (e.g. HNSW levels under multithreading). */
static inline uint64_t dn_hash2(uint64_t seed, uint64_t i) {
  uint64_t s = seed ^ (i * 0xD1B54A32D192ED03ull);
  return dn_rng_next(&s);
}

/* ---------------------------------------------------------- distances */

/* Written with several independent accumulators so that compilers
** vectorise the loop without -ffast-math. */
static inline float dn_dot(const float *a, const float *b, int n) {
  float s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0, s7 = 0;
  int i = 0;
  for (; i + 8 <= n; i += 8) {
    s0 += a[i] * b[i];         s1 += a[i + 1] * b[i + 1];
    s2 += a[i + 2] * b[i + 2]; s3 += a[i + 3] * b[i + 3];
    s4 += a[i + 4] * b[i + 4]; s5 += a[i + 5] * b[i + 5];
    s6 += a[i + 6] * b[i + 6]; s7 += a[i + 7] * b[i + 7];
  }
  for (; i < n; i++) s0 += a[i] * b[i];
  return ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7));
}

static inline float dn_l2sq(const float *a, const float *b, int n) {
  float s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0, s7 = 0;
  int i = 0;
  for (; i + 8 <= n; i += 8) {
    float d0 = a[i] - b[i], d1 = a[i + 1] - b[i + 1];
    float d2 = a[i + 2] - b[i + 2], d3 = a[i + 3] - b[i + 3];
    float d4 = a[i + 4] - b[i + 4], d5 = a[i + 5] - b[i + 5];
    float d6 = a[i + 6] - b[i + 6], d7 = a[i + 7] - b[i + 7];
    s0 += d0 * d0; s1 += d1 * d1; s2 += d2 * d2; s3 += d3 * d3;
    s4 += d4 * d4; s5 += d5 * d5; s6 += d6 * d6; s7 += d7 * d7;
  }
  for (; i < n; i++) { float d = a[i] - b[i]; s0 += d * d; }
  return ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7));
}

static inline float dn_distance(int metric, const float *a, const float *b, int n) {
  if (metric == METRIC_L2) return dn_l2sq(a, b, n);
  return 1.0f - dn_dot(a, b, n);
}

/* Normalise in place; leaves an all-zero vector unchanged. */
static inline void dn_normalize(float *x, int n) {
  float s = dn_dot(x, x, n);
  if (s > 0) {
    float inv = 1.0f / sqrtf(s);
    for (int i = 0; i < n; i++) x[i] *= inv;
  }
}

/* ------------------------------------------------------------ float16 */

static inline uint16_t dn_f32_to_f16(float f) {
  uint32_t x; memcpy(&x, &f, 4);
  uint32_t sign = (x >> 16) & 0x8000u;
  int32_t exp = (int32_t)((x >> 23) & 0xff) - 127 + 15;
  uint32_t mant = x & 0x7fffffu;
  if (((x >> 23) & 0xff) == 0xff) /* inf / nan */
    return (uint16_t)(sign | 0x7c00u | (mant ? 0x200u : 0));
  if (exp >= 31) return (uint16_t)(sign | 0x7c00u); /* overflow -> inf */
  if (exp <= 0) {                                   /* subnormal or zero */
    if (exp < -10) return (uint16_t)sign;
    mant |= 0x800000u;
    uint32_t shift = (uint32_t)(14 - exp);
    uint32_t h = mant >> shift;
    uint32_t rem = mant & ((1u << shift) - 1), half = 1u << (shift - 1);
    if (rem > half || (rem == half && (h & 1))) h++;
    return (uint16_t)(sign | h);
  }
  uint32_t h = sign | ((uint32_t)exp << 10) | (mant >> 13);
  uint32_t rem = mant & 0x1fffu;                    /* round to nearest even */
  if (rem > 0x1000u || (rem == 0x1000u && (h & 1))) h++;
  return (uint16_t)h;
}

static inline float dn_f16_to_f32(uint16_t h) {
  uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1f, mant = h & 0x3ffu, x;
  if (exp == 0) {
    if (mant == 0) x = sign;
    else {                                            /* subnormal */
      int e = -1;
      do { e++; mant <<= 1; } while (!(mant & 0x400u));
      x = sign | ((uint32_t)(127 - 15 - e) << 23) | ((mant & 0x3ffu) << 13);
    }
  } else if (exp == 31) x = sign | 0x7f800000u | (mant << 13);
  else x = sign | ((exp + 127 - 15) << 23) | (mant << 13);
  float f; memcpy(&f, &x, 4);
  return f;
}

/* ------------------------------------------- little-endian load/store */
/* Our own blob formats are little-endian (x86 and wasm are both LE, but we
** go through bytes anyway so unaligned access is never an issue). SQLite's
** on-disk page structures are big-endian; see rawpage.c. */

static inline uint16_t dn_rd16(const uint8_t *p) { return (uint16_t)(p[0] | (p[1] << 8)); }
static inline uint32_t dn_rd32(const uint8_t *p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
static inline int64_t dn_rd64(const uint8_t *p) {
  return (int64_t)((uint64_t)dn_rd32(p) | ((uint64_t)dn_rd32(p + 4) << 32));
}
static inline void dn_wr16(uint8_t *p, uint16_t v) { p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); }
static inline void dn_wr32(uint8_t *p, uint32_t v) {
  p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); p[2] = (uint8_t)(v >> 16); p[3] = (uint8_t)(v >> 24);
}
static inline void dn_wr64(uint8_t *p, int64_t v) {
  dn_wr32(p, (uint32_t)(uint64_t)v); dn_wr32(p + 4, (uint32_t)((uint64_t)v >> 32));
}

/* ------------------------------------------------------ parallel for */

/* Run fn(ctx, i, thread_index) for i in [0,n). With DENSE_ANN_THREADS and
** nthreads > 1, work is handed out dynamically in chunks from an atomic
** counter; otherwise it is a plain loop. */
typedef void (*dn_task_fn)(void *ctx, int64_t i, int thread);

typedef struct dn_par_state {
  dn_task_fn fn; void *ctx; int64_t n, chunk;
  int64_t next;          /* shared counter, updated atomically */
  int thread;            /* per-thread copy only */
  struct dn_par_state *shared;
} dn_par_state;

#ifdef DENSE_ANN_THREADS
static void *dn_par_worker(void *arg) {
  dn_par_state *me = (dn_par_state *)arg, *sh = me->shared;
  for (;;) {
    int64_t start = __atomic_fetch_add(&sh->next, sh->chunk, __ATOMIC_RELAXED);
    if (start >= sh->n) break;
    int64_t end = start + sh->chunk < sh->n ? start + sh->chunk : sh->n;
    for (int64_t i = start; i < end; i++) sh->fn(sh->ctx, i, me->thread);
  }
  return NULL;
}
#endif

static inline void dn_parallel_for(int64_t n, int nthreads, int64_t chunk, dn_task_fn fn, void *ctx) {
#ifdef DENSE_ANN_THREADS
  if (nthreads > 1 && n > 1) {
    dn_par_state sh; memset(&sh, 0, sizeof sh);
    sh.fn = fn; sh.ctx = ctx; sh.n = n; sh.chunk = chunk < 1 ? 1 : chunk; sh.next = 0;
    pthread_t *th = (pthread_t *)malloc(sizeof(pthread_t) * nthreads);
    dn_par_state *st = (dn_par_state *)malloc(sizeof(dn_par_state) * nthreads);
    for (int t = 0; t < nthreads; t++) {
      st[t] = sh; st[t].thread = t; st[t].shared = &sh;
      pthread_create(&th[t], NULL, dn_par_worker, &st[t]);
    }
    for (int t = 0; t < nthreads; t++) pthread_join(th[t], NULL);
    free(th); free(st);
    return;
  }
#endif
  (void)nthreads; (void)chunk;
  for (int64_t i = 0; i < n; i++) fn(ctx, i, 0);
}

#endif /* DENSE_COMMON_H */
