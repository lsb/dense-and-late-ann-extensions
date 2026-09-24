/*
** late_common.h -- small header-only helpers shared by the late_plaid
** extension: float16 conversion, deterministic RNG, dot products, bit
** packing, little-endian load/store, a growable byte buffer and an optional
** pthread "parallel for" used only while building an index.
**
** Everything is static inline so that each translation unit is
** self-contained and nothing clashes when several extensions are linked
** statically into one WebAssembly module.
*/
#ifndef LATE_COMMON_H
#define LATE_COMMON_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef LATE_THREADS
#include <pthread.h>
#endif

/* ------------------------------------------------------------- float16 */

static inline float lt_h2f(uint16_t h) {
  uint32_t sign = (uint32_t)(h & 0x8000) << 16;
  uint32_t exp = (h >> 10) & 0x1f, man = h & 0x3ff, bits;
  if (exp == 0) {
    if (man == 0) bits = sign;
    else {                                   /* subnormal */
      int e = -1;
      do { e++; man <<= 1; } while (!(man & 0x400));
      bits = sign | ((uint32_t)(127 - 15 - e) << 23) | ((man & 0x3ff) << 13);
    }
  } else if (exp == 31) bits = sign | 0x7f800000u | (man << 13);
  else bits = sign | ((exp + 112) << 23) | (man << 13);
  float f; memcpy(&f, &bits, 4); return f;
}

/* Round to nearest even, like numpy's astype(float16). */
static inline uint16_t lt_f2h(float f) {
  uint32_t x; memcpy(&x, &f, 4);
  uint32_t sign = (x >> 16) & 0x8000;
  int32_t exp = (int32_t)((x >> 23) & 0xff) - 127 + 15;
  uint32_t man = x & 0x7fffff;
  if (((x >> 23) & 0xff) == 0xff) return (uint16_t)(sign | 0x7c00 | (man ? 0x200 : 0));
  if (exp >= 31) return (uint16_t)(sign | 0x7c00);
  if (exp <= 0) {
    if (exp < -10) return (uint16_t)sign;
    man |= 0x800000;
    int shift = 14 - exp;
    uint32_t h = man >> shift, rem = man & ((1u << shift) - 1), half = 1u << (shift - 1);
    if (rem > half || (rem == half && (h & 1))) h++;
    return (uint16_t)(sign | h);
  }
  uint32_t h = ((uint32_t)exp << 10) | (man >> 13), rem = man & 0x1fff;
  if (rem > 0x1000 || (rem == 0x1000 && (h & 1))) h++;
  return (uint16_t)(sign | h);
}

/* ----------------------------------------------------------------- RNG */

static inline uint64_t lt_rng_next(uint64_t *s) {     /* SplitMix64 */
  uint64_t z = (*s += 0x9E3779B97F4A7C15ull);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}
static inline uint64_t lt_rng_below(uint64_t *s, uint64_t n) {
  return n ? lt_rng_next(s) % n : 0;
}

/* --------------------------------------------------------------- math */

static inline float lt_dot(const float *a, const float *b, int n) {
  float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
  int i = 0;
  for (; i + 4 <= n; i += 4) {
    s0 += a[i] * b[i]; s1 += a[i + 1] * b[i + 1];
    s2 += a[i + 2] * b[i + 2]; s3 += a[i + 3] * b[i + 3];
  }
  for (; i < n; i++) s0 += a[i] * b[i];
  return (s0 + s1) + (s2 + s3);
}

static inline void lt_normalize(float *v, int n) {
  double s = 0;
  for (int i = 0; i < n; i++) s += (double)v[i] * v[i];
  float inv = (float)(1.0 / (sqrt(s) > 1e-12 ? sqrt(s) : 1e-12));
  for (int i = 0; i < n; i++) v[i] *= inv;
}

static inline int lt_ceil_log2(uint64_t n) {        /* bits to store 0..n-1 */
  int b = 0;
  while (((uint64_t)1 << b) < n) b++;
  return b ? b : 1;
}

/* ---------------------------------------------------------- bit packing */
/* Generic little-endian bit streams (bit i of the stream is bit i%8 of byte
** i/8). Used for centroid ids and document ids; the residual codes use
** fast-plaid's own order (see late_codec.h). Readers may touch up to 8
** bytes past the last field, so buffers are padded by LT_BITPAD. */
#define LT_BITPAD 8

static inline uint32_t lt_get_bits(const uint8_t *buf, uint64_t pos, int nb) {
  const uint8_t *p = buf + (pos >> 3);
  uint64_t w = 0;
  int need = (int)((pos & 7) + nb + 7) >> 3;
  for (int i = 0; i < need; i++) w |= (uint64_t)p[i] << (8 * i);
  return (uint32_t)((w >> (pos & 7)) & ((nb >= 32) ? 0xffffffffu : ((1u << nb) - 1)));
}

static inline void lt_put_bits(uint8_t *buf, uint64_t pos, int nb, uint32_t v) {
  for (int i = 0; i < nb; i++, pos++) {
    if ((v >> i) & 1) buf[pos >> 3] |= (uint8_t)(1u << (pos & 7));
    else buf[pos >> 3] &= (uint8_t)~(1u << (pos & 7));
  }
}

static inline uint64_t lt_bits_bytes(uint64_t n, int nb) { return (n * (uint64_t)nb + 7) >> 3; }

/* ------------------------------------------------------- little endian */

static inline void lt_put_u32(uint8_t *p, uint32_t v) {
  p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); p[2] = (uint8_t)(v >> 16); p[3] = (uint8_t)(v >> 24);
}
static inline uint32_t lt_get_u32(const uint8_t *p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
static inline void lt_put_u64(uint8_t *p, uint64_t v) {
  lt_put_u32(p, (uint32_t)v); lt_put_u32(p + 4, (uint32_t)(v >> 32));
}
static inline uint64_t lt_get_u64(const uint8_t *p) {
  return (uint64_t)lt_get_u32(p) | ((uint64_t)lt_get_u32(p + 4) << 32);
}

/* ------------------------------------------------------- byte buffer */

typedef struct LtBuf { uint8_t *p; size_t n, cap; int oom; } LtBuf;

static inline int lt_buf_reserve(LtBuf *b, size_t extra) {
  if (b->oom) return 1;
  if (b->n + extra <= b->cap) return 0;
  size_t cap = b->cap ? b->cap : 256;
  while (cap < b->n + extra) cap *= 2;
  uint8_t *p = (uint8_t *)realloc(b->p, cap);
  if (!p) { b->oom = 1; return 1; }
  b->p = p; b->cap = cap;
  return 0;
}
static inline void lt_buf_add(LtBuf *b, const void *src, size_t n) {
  if (lt_buf_reserve(b, n)) return;
  memcpy(b->p + b->n, src, n); b->n += n;
}
static inline void lt_buf_u32(LtBuf *b, uint32_t v) { uint8_t t[4]; lt_put_u32(t, v); lt_buf_add(b, t, 4); }
static inline void lt_buf_u64(LtBuf *b, uint64_t v) { uint8_t t[8]; lt_put_u64(t, v); lt_buf_add(b, t, 8); }
static inline void lt_buf_free(LtBuf *b) { free(b->p); memset(b, 0, sizeof *b); }

/* ---------------------------------------------------------- parallel for */

typedef void (*lt_range_fn)(void *ctx, int64_t lo, int64_t hi, int tid);

#ifdef LATE_THREADS
typedef struct { lt_range_fn fn; void *ctx; int64_t n, chunk; int64_t next; int tid;
                 pthread_mutex_t mu; } LtPar;
typedef struct { LtPar *par; int tid; } LtParArg;
static void *lt_par_worker(void *arg) {
  LtParArg *a = (LtParArg *)arg; LtPar *p = a->par;
  for (;;) {
    pthread_mutex_lock(&p->mu);
    int64_t lo = p->next; p->next += p->chunk;
    pthread_mutex_unlock(&p->mu);
    if (lo >= p->n) break;
    int64_t hi = lo + p->chunk < p->n ? lo + p->chunk : p->n;
    p->fn(p->ctx, lo, hi, a->tid);
  }
  return NULL;
}
#endif

/* Calls fn(ctx, lo, hi, tid) over [0, n) in chunks, on up to nthreads
** threads (tid < nthreads). Single-threaded without LATE_THREADS. */
static inline void lt_parallel_for(int64_t n, int64_t chunk, int nthreads, lt_range_fn fn, void *ctx) {
  if (chunk < 1) chunk = 1;
#ifdef LATE_THREADS
  if (nthreads > 1 && n > chunk) {
    if (nthreads > 64) nthreads = 64;
    LtPar p; p.fn = fn; p.ctx = ctx; p.n = n; p.chunk = chunk; p.next = 0;
    pthread_mutex_init(&p.mu, NULL);
    pthread_t th[64]; LtParArg args[64];
    int started = 0;
    for (int t = 1; t < nthreads; t++) {
      args[t].par = &p; args[t].tid = t;
      if (pthread_create(&th[t], NULL, lt_par_worker, &args[t]) != 0) break;
      started = t;
    }
    args[0].par = &p; args[0].tid = 0;
    lt_par_worker(&args[0]);
    for (int t = 1; t <= started; t++) pthread_join(th[t], NULL);
    pthread_mutex_destroy(&p.mu);
    return;
  }
#endif
  (void)nthreads;
  for (int64_t lo = 0; lo < n; lo += chunk) fn(ctx, lo, lo + chunk < n ? lo + chunk : n, 0);
}

#endif /* LATE_COMMON_H */
