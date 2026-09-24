/*
** httpvfs.c -- a read-only SQLite VFS for databases fetched by byte range.
**
** The VFS keeps an LRU cache of fixed-size blocks. A read that misses the
** cache blocks on the "backend" for one round, which fetches any number of
** byte ranges concurrently:
**
**   - WebAssembly: the backend is JavaScript (httpvfs-pre.js) issuing HTTP
**     range requests with fetch(); the C code waits for them through
**     Emscripten's Asyncify (or JSPI), so a batch of ranges really is fetched
**     in parallel.
**   - Native: the backend reads a local file with pread() and can sleep for a
**     simulated round-trip time per round, so extensions can be developed and
**     their round counts measured without a browser.
**
** Extensions reduce the number of rounds with the prefetch and speculation
** calls declared in httpvfs.h (private file-control opcodes).
**
** Every backend request is logged as {offset, length, round, t_start, t_end}.
** "round" is the number of the backend call; all ranges of one prefetch batch
** share a round.
**
** Options come from URI parameters on the database name, for example
**   file:/path/db.sqlite?vfs=httpvfs&cache_kb=8192&block=4096
**   cache_kb      VFS cache budget in KiB; "auto" or 0 (the default) sizes it
**                 from the file: clamp(file size / 64, cache_min_kb, 64 MiB)
**   cache_min_kb  lower bound of the automatic budget (default 4096)
**   cache_max_kb  hard limit in KiB while prefetched blocks are pinned
**                 (default 4 x the budget)
**   block         cache block (and minimum request) size in bytes (default 4096)
**   readahead_kb  maximum sequential readahead in KiB, 0 disables (default 1024)
**   gap_kb        merge ranges of one batch whose gap is at most this (default 0)
**   latency_ms    native backend only: simulated round-trip time per round
**   log_max       maximum log entries kept (default 1000000)
**   max_req       request budget per round: when a round needs more requests
**                 than this, nearby ranges are merged (or, with multipart=1,
**                 sent as multi-range requests); 0 disables (default 0)
**   rtt_ms        estimated request latency for the merge cost model (100)
**   bw_kbps       estimated bandwidth in kbit/s for the cost model (10000)
**   multipart     1: ranges may be combined into multi-range requests (0)
**   net_auto      1: re-estimate rtt_ms and bw_kbps from observed rounds (1)
**
** Built three ways from this one file:
**   - static, with -DSQLITE_CORE (WASM build and the native CLI); call
**     httpvfs_register() once, e.g. from SQLITE_EXTRA_INIT;
**   - as a loadable extension (httpvfs.so), entry point sqlite3_httpvfs_init.
*/
#include "sqlite3ext.h"
SQLITE_EXTENSION_INIT1
#include "httpvfs.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifdef __EMSCRIPTEN__
# include <emscripten.h>
#else
# include <fcntl.h>
# include <unistd.h>
# include <time.h>
# include <sys/stat.h>
#endif

#define HV_VFS_NAME "httpvfs"

/* ------------------------------------------------------------------------ */
/* Backends                                                                  */

typedef struct HvBackend HvBackend;
struct HvBackend {
  /* Fetch n ranges concurrently into dest[]; one call is one round. Fill
  ** t0[i], t1[i] (milliseconds). *pSize receives the file size if the
  ** backend learns it (it is -1 on entry when unknown). If grp is not NULL,
  ** ranges with equal grp[i] (grp is nondecreasing) should be requested
  ** together as one multi-range request; a backend that cannot do that, or
  ** finds that the server does not, fetches them separately and sets
  ** HV_FLAG_NO_MULTIPART in *pFlags. Returns SQLITE_OK or an error code. */
  int (*xFetch)(HvBackend*, int n, const double *off, const int *len,
                unsigned char **dest, double *t0, double *t1,
                sqlite3_int64 *pSize, const int *grp, int *pFlags);
  void (*xClose)(HvBackend*);
};

#define HV_FLAG_NO_MULTIPART 1   /* server ignored or refused multi-range */

#ifdef __EMSCRIPTEN__

#ifdef HV_SYNC_JS
/* Synchronous backend (no Asyncify/JSPI): Module.httpvfsFetchSync must
** block until the batch is in memory, e.g. a worker doing parallel fetches
** while this thread waits on a SharedArrayBuffer with Atomics.wait. */
EM_JS(int, hv_js_fetch, (int h, int n, const double *off, const int *len,
      unsigned char **dest, double *t0, double *t1, double *pSize,
      const int *grp, int *pFlags), {
  return Module.httpvfsFetchSync(h, n, off, len, dest, t0, t1, pSize, grp, pFlags);
});
#else
EM_ASYNC_JS(int, hv_js_fetch, (int h, int n, const double *off, const int *len,
            unsigned char **dest, double *t0, double *t1, double *pSize,
            const int *grp, int *pFlags), {
  return await Module.httpvfsFetch(h, n, off, len, dest, t0, t1, pSize, grp, pFlags);
});
#endif
EM_JS(int, hv_js_open, (const char *zName), {
  return Module.httpvfsOpen(UTF8ToString(zName));
});
EM_JS(void, hv_js_close, (int h), { Module.httpvfsClose(h); });
EM_JS(double, hv_now, (void), { return performance.now(); });

typedef struct HvJsBackend { HvBackend base; int h; } HvJsBackend;

static int hvJsFetch(HvBackend *p, int n, const double *off, const int *len,
                     unsigned char **dest, double *t0, double *t1,
                     sqlite3_int64 *pSize, const int *grp, int *pFlags){
  double sz = (double)*pSize;
  int rc = hv_js_fetch(((HvJsBackend*)p)->h, n, off, len, dest, t0, t1, &sz,
                       grp, pFlags);
  *pSize = (sqlite3_int64)sz;
  return rc;
}
static void hvJsClose(HvBackend *p){
  hv_js_close(((HvJsBackend*)p)->h);
  sqlite3_free(p);
}

#else /* native */

static double hv_now(void){
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec*1e3 + ts.tv_nsec/1e6;
}

typedef struct HvFileBackend {
  HvBackend base;
  int fd;
  double latency_ms;
} HvFileBackend;

static int hvFileFetch(HvBackend *p, int n, const double *off, const int *len,
                       unsigned char **dest, double *t0, double *t1,
                       sqlite3_int64 *pSize, const int *grp, int *pFlags){
  HvFileBackend *f = (HvFileBackend*)p;
  double start = hv_now(), end;
  int i;
  (void)pSize; (void)grp; (void)pFlags;   /* pread: groups cost nothing */
  if( f->latency_ms>0 ){
    struct timespec ts;
    ts.tv_sec = (time_t)(f->latency_ms/1000);
    ts.tv_nsec = (long)((f->latency_ms - ts.tv_sec*1000.0)*1e6);
    nanosleep(&ts, 0);
  }
  for(i=0; i<n; i++){
    ssize_t got = pread(f->fd, dest[i], (size_t)len[i], (off_t)off[i]);
    if( got!=len[i] ) return SQLITE_IOERR_READ;
  }
  end = hv_now();
  for(i=0; i<n; i++){ t0[i] = start; t1[i] = end; }
  return SQLITE_OK;
}
static void hvFileClose(HvBackend *p){
  close(((HvFileBackend*)p)->fd);
  sqlite3_free(p);
}
#endif

/* ------------------------------------------------------------------------ */
/* Block cache: LRU, with pinned prefetch batches                            */

/*
** The cache holds fixed-size blocks in slots. A slot is free, in the LRU
** list (unpinned), or in the pinned list. The blocks an extension asks for
** in a prefetch or speculation batch are pinned until they are first read
** ("consumed"): eviction never takes a pinned block while an unpinned one is
** left, so a batch larger than the cache still arrives in one round and is
** then read from the cache, instead of evicting itself and being fetched
** again block by block.
**
** cap is the budget. While blocks are pinned the cache may exceed it, up to
** hard blocks; the unpinned part then keeps at least cap/4 blocks (recently
** read ones), and the rest of the budget goes to the batch. As the pinned
** blocks are consumed the cache shrinks back to cap. Only when the cache
** would exceed hard with nothing unpinned left is the oldest pinned block
** evicted (counted in pin_evictions). Data lives in chunks of HV_CHUNK
** blocks allocated on demand; hvCacheShrink() moves the blocks back below
** the budget and frees the chunks above it.
*/
#define HV_CHUNK_SHIFT 8
#define HV_CHUNK (1<<HV_CHUNK_SHIFT)
#define HV_FREE 0
#define HV_LRU  1
#define HV_PIN  2

typedef struct HvList { int head, tail, n; } HvList;

typedef struct HvCache {
  int bs;                  /* block size */
  int cap;                 /* budget in blocks */
  int hard;                /* hard limit in blocks (>= cap) */
  int n;                   /* live blocks */
  int nslot;               /* slots in use or free (0..nslot-1) */
  int aslot;               /* slots allocated in the per-slot arrays */
  int nchunk;              /* data chunks allocated */
  unsigned char **chunk;   /* chunk[i]: HV_CHUNK*bs bytes */
  sqlite3_int64 *key;      /* block index of each slot */
  int *prev, *next;        /* list links (LRU or pinned list) */
  int *hnext;              /* hash chains; the free list for free slots */
  unsigned char *state;    /* HV_FREE, HV_LRU or HV_PIN */
  HvList lru, pin;         /* heads are the most recent */
  int freehead;
  int *bucket;
  unsigned nbucket;        /* power of two */
  int peak;                /* high-water mark of n since the last reset */
  sqlite3_int64 pin_evictions;
} HvCache;

static unsigned hvHash(sqlite3_int64 k, unsigned nb){
  sqlite3_uint64 x = (sqlite3_uint64)k * 0x9E3779B97F4A7C15ull;
  return (unsigned)(x>>32) & (nb-1);
}

static unsigned char *hvSlotData(HvCache *c, int s){
  return c->chunk[s>>HV_CHUNK_SHIFT] + (size_t)(s & (HV_CHUNK-1))*c->bs;
}

static int hvCacheInit(HvCache *c, int bs, int cap, int hard){
  int i;
  memset(c, 0, sizeof(*c));
  c->bs = bs; c->cap = cap; c->hard = hard<cap ? cap : hard;
  c->lru.head = c->lru.tail = c->pin.head = c->pin.tail = -1;
  c->freehead = -1;
  c->nbucket = 64;
  while( c->nbucket < (unsigned)cap*2 ) c->nbucket <<= 1;
  c->bucket = sqlite3_malloc64(sizeof(int)*c->nbucket);
  if( !c->bucket ) return SQLITE_NOMEM;
  for(i=0; i<(int)c->nbucket; i++) c->bucket[i] = -1;
  return SQLITE_OK;
}

static void hvCacheFree(HvCache *c){
  int i;
  for(i=0; i<c->nchunk; i++) sqlite3_free(c->chunk[i]);
  sqlite3_free(c->chunk); sqlite3_free(c->key); sqlite3_free(c->prev);
  sqlite3_free(c->next); sqlite3_free(c->hnext); sqlite3_free(c->state);
  sqlite3_free(c->bucket);
  memset(c, 0, sizeof(*c));
}

static HvList *hvListOf(HvCache *c, int s){
  return c->state[s]==HV_PIN ? &c->pin : &c->lru;
}
static void hvListUnlink(HvCache *c, HvList *L, int s){
  if( c->prev[s]>=0 ) c->next[c->prev[s]] = c->next[s]; else L->head = c->next[s];
  if( c->next[s]>=0 ) c->prev[c->next[s]] = c->prev[s]; else L->tail = c->prev[s];
  L->n--;
}
static void hvListPushFront(HvCache *c, HvList *L, int s){
  c->prev[s] = -1; c->next[s] = L->head;
  if( L->head>=0 ) c->prev[L->head] = s;
  L->head = s;
  if( L->tail<0 ) L->tail = s;
  L->n++;
}
static void hvListPushBack(HvCache *c, HvList *L, int s){
  c->next[s] = -1; c->prev[s] = L->tail;
  if( L->tail>=0 ) c->next[L->tail] = s;
  L->tail = s;
  if( L->head<0 ) L->head = s;
  L->n++;
}
/* Move slot s to the front (or back) of the list of state st. */
static void hvSlotMoveTo(HvCache *c, int s, int st, int front){
  HvList *to = st==HV_PIN ? &c->pin : &c->lru;
  hvListUnlink(c, hvListOf(c, s), s);
  c->state[s] = (unsigned char)st;
  if( front ) hvListPushFront(c, to, s); else hvListPushBack(c, to, s);
}

/* Return the slot holding block k, or -1. With touch, the block counts as
** read: it becomes the most recently used and, if pinned, is unpinned. */
static int hvCacheGet(HvCache *c, sqlite3_int64 k, int touch){
  int s = c->bucket[hvHash(k, c->nbucket)];
  while( s>=0 && c->key[s]!=k ) s = c->hnext[s];
  if( s>=0 && touch && (c->state[s]==HV_PIN || c->lru.head!=s) ){
    hvSlotMoveTo(c, s, HV_LRU, 1);
  }
  return s;
}

static void hvEvict(HvCache *c, int s){
  int *pp = &c->bucket[hvHash(c->key[s], c->nbucket)];
  if( c->state[s]==HV_PIN ) c->pin_evictions++;
  while( *pp!=s ) pp = &c->hnext[*pp];
  *pp = c->hnext[s];
  hvListUnlink(c, hvListOf(c, s), s);
  c->state[s] = HV_FREE;
  c->key[s] = -1;
  c->hnext[s] = c->freehead;
  c->freehead = s;
  c->n--;
}

/*
** Evict so that `extra` more blocks fit: unpinned blocks beyond the budget
** go first (keeping a floor of cap/4 unpinned blocks while pinned blocks
** exist or are being added), then anything beyond the hard limit, pinned
** blocks (oldest first) only when no unpinned block is left.
*/
static void hvCacheTrim(HvCache *c, int extra, int pinning){
  int floor = (c->pin.n>0 || pinning) ? c->cap/4 : 0;
  while( c->n+extra > c->cap && c->lru.n > floor ) hvEvict(c, c->lru.tail);
  while( c->n+extra > c->hard && c->n>0 ){
    hvEvict(c, c->lru.n>0 ? c->lru.tail : c->pin.tail);
  }
}

/* Rebuild the hash table with nb buckets (and the free list, which shares
** the hnext links). */
static int hvCacheRehash(HvCache *c, unsigned nb){
  int *nbkt = sqlite3_malloc64(sizeof(int)*nb), s;
  unsigned i;
  if( !nbkt ) return SQLITE_NOMEM;
  for(i=0; i<nb; i++) nbkt[i] = -1;
  c->freehead = -1;
  for(s=c->nslot-1; s>=0; s--){
    if( c->state[s]==HV_FREE ){
      c->hnext[s] = c->freehead; c->freehead = s;
    }else{
      unsigned h = hvHash(c->key[s], nb);
      c->hnext[s] = nbkt[h];
      nbkt[h] = s;
    }
  }
  sqlite3_free(c->bucket);
  c->bucket = nbkt; c->nbucket = nb;
  return SQLITE_OK;
}

/* A free slot (growing the arrays and data chunks as needed), or -1 when
** out of memory. The slot is not linked anywhere yet. */
static int hvSlotAlloc(HvCache *c){
  int s;
  if( c->freehead>=0 ){
    s = c->freehead;
    c->freehead = c->hnext[s];
    return s;
  }
  s = c->nslot;
  if( s>=c->aslot ){
    int na = c->aslot ? c->aslot*2 : HV_CHUNK;
    sqlite3_int64 *k;
    int *p;
    unsigned char *st;
    if( !(k = sqlite3_realloc64(c->key, sizeof(sqlite3_int64)*na)) ) return -1;
    c->key = k;
    if( !(p = sqlite3_realloc64(c->prev, sizeof(int)*na)) ) return -1;
    c->prev = p;
    if( !(p = sqlite3_realloc64(c->next, sizeof(int)*na)) ) return -1;
    c->next = p;
    if( !(p = sqlite3_realloc64(c->hnext, sizeof(int)*na)) ) return -1;
    c->hnext = p;
    if( !(st = sqlite3_realloc64(c->state, na)) ) return -1;
    c->state = st;
    c->aslot = na;
  }
  if( (s>>HV_CHUNK_SHIFT) >= c->nchunk ){
    unsigned char **ch = sqlite3_realloc64(c->chunk, sizeof(*ch)*(c->nchunk+1));
    if( !ch ) return -1;
    c->chunk = ch;
    ch[c->nchunk] = sqlite3_malloc64((sqlite3_uint64)HV_CHUNK*c->bs);
    if( !ch[c->nchunk] ) return -1;
    c->nchunk++;
  }
  c->state[s] = HV_FREE;
  c->key[s] = -1;
  c->nslot++;
  if( (unsigned)c->nslot*2 > c->nbucket ){
    c->hnext[s] = -1;
    hvCacheRehash(c, c->nbucket*2);   /* on failure the old table stays valid */
    if( c->freehead==s ) c->freehead = c->hnext[s];   /* s is taken */
  }
  return s;
}

/* Insert block k with nsrc bytes from src (zero-padded to the block size),
** pinned if pin. A block already cached is only (re)pinned. */
static void hvCachePut(HvCache *c, sqlite3_int64 k, const unsigned char *src,
                       int nsrc, int pin){
  int s = hvCacheGet(c, k, 0);
  unsigned h;
  if( s>=0 ){
    if( pin ) hvSlotMoveTo(c, s, HV_PIN, 1);
    return;
  }
  hvCacheTrim(c, 1, pin);
  s = hvSlotAlloc(c);
  if( s<0 ){
    /* Out of memory: reuse the least valuable block, or do not cache. */
    if( c->n==0 ) return;
    hvEvict(c, c->lru.n>0 ? c->lru.tail : c->pin.tail);
    s = hvSlotAlloc(c);
    if( s<0 ) return;
  }
  c->key[s] = k;
  h = hvHash(k, c->nbucket);
  c->hnext[s] = c->bucket[h];
  c->bucket[h] = s;
  c->state[s] = pin ? HV_PIN : HV_LRU;
  hvListPushFront(c, pin ? &c->pin : &c->lru, s);
  c->n++;
  if( c->n>c->peak ) c->peak = c->n;
  memcpy(hvSlotData(c, s), src, nsrc);
  if( nsrc<c->bs ) memset(hvSlotData(c, s) + nsrc, 0, c->bs-nsrc);
}

/*
** Unpin every pinned block (a new batch starts, or the statement ended).
** The blocks stay warm: they go to the front of the LRU list, in their
** order, since a batch prefetched by one statement (httpvfs_warm) may be
** meant for the next one.
*/
static void hvCacheUnpinAll(HvCache *c){
  while( c->pin.tail>=0 ) hvSlotMoveTo(c, c->pin.tail, HV_LRU, 1);
}

/* Move the block in slot s to slot d (free, and not on the free list). */
static void hvSlotRelocate(HvCache *c, int s, int d){
  int *pp = &c->bucket[hvHash(c->key[s], c->nbucket)];
  HvList *L = hvListOf(c, s);
  while( *pp!=s ) pp = &c->hnext[*pp];
  *pp = d;
  c->hnext[d] = c->hnext[s];
  c->key[d] = c->key[s];
  c->state[d] = c->state[s];
  c->prev[d] = c->prev[s];
  c->next[d] = c->next[s];
  if( c->prev[s]>=0 ) c->next[c->prev[s]] = d; else L->head = d;
  if( c->next[s]>=0 ) c->prev[c->next[s]] = d; else L->tail = d;
  memcpy(hvSlotData(c, d), hvSlotData(c, s), c->bs);
  c->state[s] = HV_FREE;
  c->key[s] = -1;
}

/*
** Give back the memory above the budget: move the blocks held in slots past
** the budget's chunks into free slots below them, then free those chunks.
** Does nothing unless n fits below (call hvCacheTrim first).
*/
static void hvCacheShrink(HvCache *c){
  int keep = (c->cap + HV_CHUNK - 1) >> HV_CHUNK_SHIFT;
  int limit = keep << HV_CHUNK_SHIFT, s, d = 0, i;
  if( c->nslot<=limit || c->n>limit ) return;
  for(s=limit; s<c->nslot; s++){
    if( c->state[s]==HV_FREE ) continue;
    while( c->state[d]!=HV_FREE ) d++;     /* stays below limit: n <= limit */
    hvSlotRelocate(c, s, d);
  }
  for(i=keep; i<c->nchunk; i++) sqlite3_free(c->chunk[i]);
  c->nchunk = keep;
  c->nslot = limit;
  c->freehead = -1;
  for(s=limit-1; s>=0; s--){
    if( c->state[s]==HV_FREE ){ c->hnext[s] = c->freehead; c->freehead = s; }
  }
}

static void hvCacheClear(HvCache *c){
  int i, keep = (c->cap + HV_CHUNK - 1) >> HV_CHUNK_SHIFT;
  c->n = 0; c->nslot = 0; c->freehead = -1;
  c->lru.head = c->lru.tail = c->pin.head = c->pin.tail = -1;
  c->lru.n = c->pin.n = 0;
  for(i=0; i<(int)c->nbucket; i++) c->bucket[i] = -1;
  for(i=keep; i<c->nchunk; i++) sqlite3_free(c->chunk[i]);
  if( c->nchunk>keep ) c->nchunk = keep;
}

/* End of a statement: unpin what was never read, trim to the budget and
** free the memory above it. */
static void hvCacheRelease(HvCache *c){
  hvCacheUnpinAll(c);
  hvCacheTrim(c, 0, 0);
  hvCacheShrink(c);
}

/* Change the budget (and hard limit) of a cache without pinned blocks. */
static void hvCacheSetBudget(HvCache *c, int cap, int hard){
  c->cap = cap;
  c->hard = hard<cap ? cap : hard;
  hvCacheRelease(c);
}

/* ------------------------------------------------------------------------ */
/* Request planning under a per-round request budget                         */

/*
** One round needs m disjoint block ranges [rs[i], re[i]) (sorted, in blocks
** of bs bytes). Over HTTP/1.1 a browser runs at most six requests per host
** at once, so m > 6 requests take about ceil(m/6) latencies. This planner
** assigns the ranges to requests, grp[i] = request number (nondecreasing),
** and returns the number of requests:
**
**   multipart == 0 (coalescing): the ranges of one request are merged into
**     one contiguous range, so the blocks between them are fetched too. The
**     cost model is
**         T(g) = ceil(g / max_req) * rtt_ms + bytes(g) / bw_Bpms,
**     where bytes(g) is the smallest total span of g contiguous requests:
**     cut at the g-1 largest gaps. Only g = m and multiples of max_req can
**     be optimal (between them the number of waves is constant and more
**     requests fetch fewer bytes), so all candidates are evaluated exactly.
**     The total span may not exceed max_blocks (> 0; unless g = m).
**   multipart != 0: every range is fetched exactly; the ranges are split
**     in offset order into min(max_req, m) requests of similar byte size,
**     with at most max_parts ranges per request.
**
** max_req <= 0 or m <= max_req: one request per range. Pure function;
** exported so that the tests and the trace re-simulation
** (bench/coalesce_eval.py) use exactly this code.
*/
typedef struct HvGap { sqlite3_int64 len; int idx; } HvGap;
static int hvGapCmp(const void *a, const void *b){
  const HvGap *x = (const HvGap*)a, *y = (const HvGap*)b;
  if( x->len!=y->len ) return x->len > y->len ? -1 : 1;   /* larger first */
  return x->idx - y->idx;
}

int httpvfs_plan(int m, const sqlite3_int64 *rs, const sqlite3_int64 *re,
                 int bs, int max_req, double rtt_ms, double bw_Bpms,
                 sqlite3_int64 max_blocks, int multipart, int max_parts,
                 int *grp){
  int i, g;
  if( m<=0 ) return 0;
  if( max_req<=0 || m<=max_req ){
    for(i=0; i<m; i++) grp[i] = i;
    return m;
  }
  if( multipart ){
    double total = 0, acc = 0;
    int k = 0, cnt = 0;
    if( max_parts<=0 ) max_parts = m;
    g = max_req;
    if( (m + max_parts - 1)/max_parts > g ) g = (m + max_parts - 1)/max_parts;
    for(i=0; i<m; i++) total += (double)(re[i]-rs[i]);
    for(i=0; i<m; i++){
      /* Close the current request when it is full, when it has its share
      ** of the bytes and the rest still fits into the remaining requests,
      ** or when every remaining request needs at least one range. */
      if( cnt>0 && ( cnt>=max_parts
                  || (k<g-1 && ((acc >= total*(k+1)/g
                                 && m-i <= (sqlite3_int64)(g-1-k)*max_parts)
                                || m-i <= g-1-k)) ) ){
        k++; cnt = 0;
      }
      grp[i] = k;
      acc += (double)(re[i]-rs[i]);
      cnt++;
    }
    return k+1;
  }else{
    HvGap *gap;
    sqlite3_int64 span = re[m-1] - rs[0], cutsum = 0, bytes;
    char *cut;
    int best_g = m, c;
    double best_t, t;
    gap = (HvGap*)malloc(sizeof(HvGap)*(size_t)m + (size_t)m);
    if( !gap ){ for(i=0; i<m; i++) grp[i] = i; return m; }
    cut = (char*)(gap + m);
    for(i=0; i<m-1; i++){ gap[i].len = rs[i+1]-re[i]; gap[i].idx = i; }
    qsort(gap, (size_t)(m-1), sizeof(HvGap), hvGapCmp);
    if( rtt_ms<0 ) rtt_ms = 0;
    if( bw_Bpms<=0 ) bw_Bpms = 1e-9;
    /* g = m: every gap cut. */
    for(i=0; i<m-1; i++) cutsum += gap[i].len;
    bytes = (span - cutsum) * (sqlite3_int64)bs;
    best_t = (double)((m + max_req - 1)/max_req) * rtt_ms + (double)bytes/bw_Bpms;
    /* g = c*max_req < m: cut the g-1 largest gaps. */
    cutsum = 0;
    for(c=1, i=0; c*max_req < m; c++){
      g = c*max_req;
      while( i<g-1 ){ cutsum += gap[i].len; i++; }
      if( max_blocks>0 && span - cutsum > max_blocks ) continue;
      bytes = (span - cutsum) * (sqlite3_int64)bs;
      t = (double)c * rtt_ms + (double)bytes/bw_Bpms;
      if( t < best_t ){ best_t = t; best_g = g; }
    }
    memset(cut, 0, (size_t)m);
    for(i=0; i<best_g-1; i++) cut[gap[i].idx] = 1;
    for(g=0, i=0; i<m; i++){
      grp[i] = g;
      if( i<m-1 && cut[i] ) g++;
    }
    free(gap);
    return best_g;
  }
}

/*
** Network parameters of one file: the request budget, the cost model's
** latency and bandwidth, and a small estimator that refits them from the
** rounds actually observed (only rounds of at most max_req requests, which
** run without client-side queueing): T = rtt + bytes/bw.
*/
#define HV_NET_WIN 64
typedef struct HvNet {
  int max_req;                /* request budget per round (0: unlimited) */
  int multipart;              /* 1: may send multi-range requests */
  int max_parts;              /* ranges per multi-range request */
  int autoest;                /* refit rtt/bw from observations */
  double rtt_ms;              /* estimated request latency */
  double bw_Bpms;             /* estimated bandwidth, bytes per ms */
  double wB[HV_NET_WIN], wT[HV_NET_WIN];   /* observed rounds: bytes, ms */
  int wn, wi;
  sqlite3_int64 overfetch;    /* bytes fetched that were not asked for */
  sqlite3_int64 planned_rounds;   /* rounds the planner changed */
  sqlite3_int64 mp_requests;  /* multi-range requests sent */
  int mp_failed;              /* the server does not do multi-range */
} HvNet;

static void hvNetResetCounters(HvNet *nt){
  nt->overfetch = 0; nt->planned_rounds = 0; nt->mp_requests = 0;
}

static int hvDblCmp(const void *a, const void *b){
  double x = *(const double*)a, y = *(const double*)b;
  return x<y ? -1 : x>y;
}
static int hvSampleCmp(const void *a, const void *b){
  return hvDblCmp(a, b);   /* by bytes, the first member */
}
static double hvMedian(double *v, int n){
  qsort(v, n, sizeof(double), hvDblCmp);
  return n%2 ? v[n/2] : 0.5*(v[n/2-1] + v[n/2]);
}

/*
** Record one round (B bytes in T ms by n requests) and refit rtt and bw.
** The samples are split at the median byte count; the bandwidth is the slope
** between the medians of the two halves (when their sizes differ by 2x or
** more, otherwise the bandwidth cannot be told apart from the latency and
** keeps its value), and the latency is a low quartile of T - B/bw over the
** smaller half, but at least half of that half's low-quartile T.
*/
static void hvNetObserve(HvNet *nt, int n, double B, double T){
  double s[HV_NET_WIN][2], v[HV_NET_WIN], mBlo, mBhi, mTlo, mThi, L, W, Tq;
  int i, h, m;
  if( !nt->autoest || !(T>0) || !(B>0) ) return;
  if( nt->max_req>0 && n>nt->max_req ) return;   /* queued: not a clean sample */
  nt->wB[nt->wi] = B; nt->wT[nt->wi] = T;
  nt->wi = (nt->wi+1) % HV_NET_WIN;
  if( nt->wn<HV_NET_WIN ) nt->wn++;
  m = nt->wn;
  if( m<4 ) return;
  for(i=0; i<m; i++){ s[i][0] = nt->wB[i]; s[i][1] = nt->wT[i]; }
  qsort(s, m, sizeof(s[0]), hvSampleCmp);
  h = m/2;
  for(i=0; i<h; i++) v[i] = s[i][0];
  mBlo = hvMedian(v, h);
  for(i=0; i<h; i++) v[i] = s[i][1];
  mTlo = hvMedian(v, h);
  Tq = v[h/4];
  for(i=h; i<m; i++) v[i-h] = s[i][0];
  mBhi = hvMedian(v, m-h);
  for(i=h; i<m; i++) v[i-h] = s[i][1];
  mThi = hvMedian(v, m-h);
  W = nt->bw_Bpms;
  if( mBhi >= 2*mBlo && mThi > mTlo ) W = (mBhi - mBlo)/(mThi - mTlo);
  if( W<1 ) W = 1;             /* 8 kbit/s */
  if( W>1e6 ) W = 1e6;         /* 8 Gbit/s */
  for(i=0; i<h; i++) v[i] = s[i][1] - s[i][0]/W;
  qsort(v, h, sizeof(double), hvDblCmp);
  L = v[h/4];
  if( L<0.5*Tq ) L = 0.5*Tq;
  if( L<0.1 ) L = 0.1;
  nt->rtt_ms = L; nt->bw_Bpms = W;
}

/* ------------------------------------------------------------------------ */
/* File                                                                      */

typedef struct HvLogEntry {
  double off;
  int len;
  int round;
  int req;                    /* request number within the round */
  double t0, t1;
} HvLogEntry;

typedef struct HvFile {
  sqlite3_file base;
  HvBackend *be;
  sqlite3_int64 size;         /* file size in bytes */
  HvCache cache;
  int ra_max;                 /* max readahead in blocks */
  int ra;                     /* current readahead in blocks */
  int gap;                    /* merge gap in blocks */
  sqlite3_int64 next_seq;     /* block after the previous read */
  int speculating;
  sqlite3_int64 *spec;        /* blocks missed while speculating */
  int nspec, aspec;
  HvLogEntry *log;
  int nlog, alog, log_max;
  HttpvfsStats st;
  HvNet net;
  int cache_auto;             /* budget from the file size */
  sqlite3_int64 cache_min_kb; /* lower bound of the automatic budget */
  sqlite3_int64 cache_max_kb; /* hard limit (0: 4 x budget) */
} HvFile;

typedef struct HvGlobal {
  sqlite3_vfs *pOrig;
} HvGlobal;
static HvGlobal hvg;

static int hvI64Cmp(const void *a, const void *b){
  sqlite3_int64 x = *(const sqlite3_int64*)a, y = *(const sqlite3_int64*)b;
  return x<y ? -1 : x>y;
}

static sqlite3_int64 hvNumBlocks(HvFile *f){
  return (f->size + f->cache.bs - 1) / f->cache.bs;
}

static void hvLog(HvFile *f, double off, int len, int round, int req,
                  double t0, double t1){
  if( f->nlog>=f->log_max ) return;
  if( f->nlog>=f->alog ){
    int na = f->alog ? f->alog*2 : 256;
    HvLogEntry *p;
    if( na>f->log_max ) na = f->log_max;
    p = sqlite3_realloc64(f->log, sizeof(HvLogEntry)*na);
    if( !p ) return;
    f->log = p; f->alog = na;
  }
  f->log[f->nlog].off = off;
  f->log[f->nlog].len = len;
  f->log[f->nlog].round = round;
  f->log[f->nlog].req = req;
  f->log[f->nlog].t0 = t0;
  f->log[f->nlog].t1 = t1;
  f->nlog++;
}

/*
** Fetch the sorted, distinct, uncached blocks blk[0..n-1] in one round and
** insert them into the cache. Blocks closer than the merge gap share a
** request (the blocks in the gap are fetched and cached too). If the round
** then needs more requests than net.max_req, httpvfs_plan() merges nearby
** ranges (again caching the blocks in between) or groups them into
** multi-range requests. If out is not NULL, the bytes [out_off,
** out_off+out_len) are also copied into it straight from the fetched data,
** so that a read larger than the cache still works. With pin, the blocks
** of blk[] are pinned in the cache until they are read (a prefetch or
** speculation batch); the extra blocks fetched in gaps never are.
*/
static int hvWanted(const sqlite3_int64 *blk, int n, sqlite3_int64 b){
  int lo = 0, hi = n-1;
  while( lo<=hi ){
    int mid = (lo+hi)/2;
    if( blk[mid]==b ) return 1;
    if( blk[mid]<b ) lo = mid+1; else hi = mid-1;
  }
  return 0;
}

static int hvFetchBlocks(HvFile *f, const sqlite3_int64 *blk, int n,
                         unsigned char *out, sqlite3_int64 out_off,
                         int out_len, int pin){
  int bs = f->cache.bs;
  int nr = 0, i, j, k, rc, pass, flags = 0, nreq;
  double *off, *t0, *t1;
  int *len, *grp;
  unsigned char **dest;
  sqlite3_int64 *first, *rend;
  sqlite3_int64 size = f->size, B = 0;
  const int *pgrp = 0;
  double tmin, tmax;

  if( n<=0 ) return SQLITE_OK;
  off = sqlite3_malloc64(sizeof(double)*n*3);
  len = sqlite3_malloc64(sizeof(int)*n*2);
  dest = sqlite3_malloc64(sizeof(unsigned char*)*n);
  first = sqlite3_malloc64(sizeof(sqlite3_int64)*n*2);
  if( !off || !len || !dest || !first ){ rc = SQLITE_NOMEM; goto done; }
  memset(dest, 0, sizeof(unsigned char*)*n);
  t0 = off + n; t1 = off + 2*n;
  grp = len + n;
  rend = first + n;

  /* Coalesce into ranges of blocks [first, rend). */
  for(i=0; i<n; i=j){
    sqlite3_int64 last = blk[i];
    for(j=i+1; j<n && blk[j]-last-1 <= f->gap; j++) last = blk[j];
    first[nr] = blk[i];
    rend[nr] = last+1;
    grp[nr] = nr;
    nr++;
  }

  /* More requests than the budget: plan merges or multi-range requests. */
  if( f->net.max_req>0 && nr>f->net.max_req ){
    int mp = f->net.multipart && !f->net.mp_failed;
    /* Over-fetched gap blocks are not pinned: at most half the budget. */
    sqlite3_int64 maxb = n + f->cache.cap/2;
    int g = httpvfs_plan(nr, first, rend, bs, f->net.max_req, f->net.rtt_ms,
                         f->net.bw_Bpms, maxb, mp, f->net.max_parts, grp);
    if( g<nr ){
      f->net.planned_rounds++;
      if( mp ){
        pgrp = grp;
      }else{
        for(k=0, i=0; i<nr; i=j){
          for(j=i+1; j<nr && grp[j]==grp[i]; j++){}
          first[k] = first[i]; rend[k] = rend[j-1]; grp[k] = k;
          k++;
        }
        nr = k;
      }
    }
  }

  /* Byte ranges (the last one clipped to the file size). */
  for(k=0, i=0; i<nr; i++){
    sqlite3_int64 end = rend[i]*bs;
    if( size>=0 && end>size ) end = size;
    if( end - first[i]*bs <= 0 ) continue;
    first[k] = first[i];
    grp[k] = grp[i];
    off[k] = (double)(first[i]*bs);
    len[k] = (int)(end - first[i]*bs);
    dest[k] = sqlite3_malloc64(len[k]);
    if( !dest[k] ){ rc = SQLITE_NOMEM; goto done; }
    k++;
  }
  nr = k;
  if( nr==0 ){ rc = SQLITE_OK; goto done; }

  f->st.rounds++;
  rc = f->be->xFetch(f->be, nr, off, len, dest, t0, t1, &size, pgrp, &flags);
  if( rc!=SQLITE_OK ) goto done;
  if( f->size<0 ) f->size = size;
  if( flags & HV_FLAG_NO_MULTIPART ){
    f->net.mp_failed = 1;
    pgrp = 0;              /* the backend fetched every range separately */
  }

  nreq = 0;
  tmin = t0[0]; tmax = t1[0];
  for(i=0; i<nr; i++){
    int req = pgrp ? pgrp[i] : i;
    if( i==0 || req!=(pgrp ? pgrp[i-1] : i-1) ){
      nreq++;
      if( pgrp && i+1<nr && pgrp[i+1]==req ) f->net.mp_requests++;
    }
    if( t0[i]<tmin ) tmin = t0[i];
    if( t1[i]>tmax ) tmax = t1[i];
    f->st.bytes += len[i];
    B += len[i];
    hvLog(f, off[i], len[i], (int)f->st.rounds, req, t0[i], t1[i]);
    if( out ){
      /* Copy the overlap of this range with the output window. */
      sqlite3_int64 a0 = (sqlite3_int64)off[i], a1 = a0 + len[i];
      sqlite3_int64 o0 = out_off, o1 = out_off + out_len;
      sqlite3_int64 lo = a0>o0 ? a0 : o0, hi = a1<o1 ? a1 : o1;
      if( lo<hi ) memcpy(out + (lo-o0), dest[i] + (lo-a0), (size_t)(hi-lo));
    }
  }
  f->st.requests += nreq;
  f->st.net_ms += tmax - tmin;
  hvNetObserve(&f->net, nreq, (double)B, tmax - tmin);

  /* Into the cache: blocks nobody asked for first, then the requested ones,
  ** so that the requested blocks are the most recently used (and, for a
  ** prefetch batch, pinned). */
  for(pass=0; pass<2; pass++){
    for(i=0; i<nr; i++){
      sqlite3_int64 b;
      int kk;
      for(kk=0, b=first[i]; kk<len[i]; kk+=bs, b++){
        int m = len[i]-kk < bs ? len[i]-kk : bs;
        if( hvWanted(blk, n, b)!=pass ) continue;
        if( pass==0 ) f->net.overfetch += m;
        if( hvCacheGet(&f->cache, b, 0)<0 ){
          hvCachePut(&f->cache, b, dest[i]+kk, m, pass==1 && pin);
        }
      }
    }
  }

done:
  if( dest ) for(i=0; i<n; i++) sqlite3_free(dest[i]);
  sqlite3_free(off); sqlite3_free(len); sqlite3_free(dest); sqlite3_free(first);
  return rc;
}

static int hvSpecRecord(HvFile *f, sqlite3_int64 b){
  if( f->nspec>=f->aspec ){
    int na = f->aspec ? f->aspec*2 : 64;
    sqlite3_int64 *p = sqlite3_realloc64(f->spec, sizeof(sqlite3_int64)*na);
    if( !p ) return SQLITE_NOMEM;
    f->spec = p; f->aspec = na;
  }
  f->spec[f->nspec++] = b;
  return SQLITE_OK;
}

/* Sort, deduplicate and drop cached blocks; returns the new count. With
** pin, the cached blocks dropped are pinned (they belong to the batch). */
static int hvUncached(HvFile *f, sqlite3_int64 *b, int n, int pin){
  int i, m = 0, s;
  qsort(b, n, sizeof(*b), hvI64Cmp);
  for(i=0; i<n; i++){
    if( i>0 && b[i]==b[i-1] ) continue;
    if( b[i]<0 || (f->size>=0 && b[i]>=hvNumBlocks(f)) ) continue;
    if( (s = hvCacheGet(&f->cache, b[i], 0))>=0 ){
      if( pin ) hvSlotMoveTo(&f->cache, s, HV_PIN, 1);
      continue;
    }
    b[m++] = b[i];
  }
  return m;
}

/*
** Fetch a prefetch or speculation batch: blocks b[0..n-1] (any order,
** duplicates allowed) in one round, all of them pinned until read. Pins
** left over from an earlier batch are dropped first (the extensions read a
** batch before they announce the next one). A batch that does not fit
** under the hard limit is cut; the rest is fetched on demand. Returns the
** number of blocks fetched in *pn.
*/
static int hvFetchBatch(HvFile *f, sqlite3_int64 *b, int n, int *pn){
  int rc = SQLITE_OK, room;
  hvCacheUnpinAll(&f->cache);
  n = hvUncached(f, b, n, 1);
  room = f->cache.hard - f->cache.pin.n;
  if( n>room ) n = room>0 ? room : 0;
  if( n>0 ) rc = hvFetchBlocks(f, b, n, 0, 0, 0, 1);
  *pn = rc==SQLITE_OK ? n : 0;
  return rc;
}

static int hvRead(sqlite3_file *pFile, void *zBuf, int iAmt,
                  sqlite3_int64 iOfst){
  HvFile *f = (HvFile*)pFile;
  int bs = f->cache.bs;
  unsigned char *out = (unsigned char*)zBuf;
  sqlite3_int64 b0, b1, b, nblocks;
  sqlite3_int64 *miss = 0;
  int nmiss = 0, rc = SQLITE_OK, avail;

  f->st.reads++;
  if( iOfst>=f->size ){
    memset(zBuf, 0, iAmt);
    return SQLITE_IOERR_SHORT_READ;
  }
  avail = (iOfst+iAmt > f->size) ? (int)(f->size - iOfst) : iAmt;
  b0 = iOfst / bs;
  b1 = (iOfst + avail - 1) / bs;
  nblocks = hvNumBlocks(f);

  /* Pass 1: copy cached blocks, collect misses. */
  for(b=b0; b<=b1; b++){
    int s = hvCacheGet(&f->cache, b, 1);
    sqlite3_int64 lo = b*bs > iOfst ? b*bs : iOfst;
    sqlite3_int64 hi = (b+1)*bs < iOfst+avail ? (b+1)*bs : iOfst+avail;
    if( s>=0 ){
      f->st.cache_hits++;
      memcpy(out + (lo-iOfst), hvSlotData(&f->cache, s) + (lo - b*bs),
             (size_t)(hi-lo));
    }else if( f->speculating && b>0 ){
      /* Block 0 (the header and page 1) is always fetched for real. */
      f->st.spec_misses++;
      memset(out + (lo-iOfst), 0, (size_t)(hi-lo));
      rc = hvSpecRecord(f, b);
      if( rc ) return rc;
    }else{
      if( !miss ){
        miss = sqlite3_malloc64(sizeof(sqlite3_int64)*(b1-b0+1+f->ra_max+1));
        if( !miss ) return SQLITE_NOMEM;
      }
      miss[nmiss++] = b;
    }
  }

  /* Blocks read here may have been pinned: back towards the budget. */
  if( f->cache.n > f->cache.cap ) hvCacheTrim(&f->cache, 0, 0);

  if( nmiss>0 ){
    f->st.cache_misses += nmiss;
    /* Sequential readahead: grow while reads keep following each other. */
    if( f->ra_max>0 ){
      if( b0==f->next_seq ){
        f->ra = f->ra ? f->ra*2 : 1;
        if( f->ra>f->ra_max ) f->ra = f->ra_max;
      }else{
        f->ra = 0;
      }
      for(b=b1+1; b<=b1+f->ra && b<nblocks; b++){
        if( hvCacheGet(&f->cache, b, 0)>=0 ) break;
        miss[nmiss++] = b;
      }
    }
    rc = hvFetchBlocks(f, miss, nmiss, out, iOfst, avail, 0);
    sqlite3_free(miss);
    if( rc!=SQLITE_OK ) return SQLITE_IOERR_READ;
  }
  f->next_seq = b1+1;

  /* A WAL-mode database is read as a rollback-journal one (read-only). */
  if( iOfst<=18 && iOfst+avail>=20 && out[18-iOfst]==2 && out[19-iOfst]==2 ){
    out[18-iOfst] = 1; out[19-iOfst] = 1;
  }
  if( avail<iAmt ){
    memset(out+avail, 0, iAmt-avail);
    return SQLITE_IOERR_SHORT_READ;
  }
  return SQLITE_OK;
}

static int hvPrefetchBlocks(HvFile *f, sqlite3_int64 *b, int n){
  int rc;
  f->st.prefetch_calls++;
  rc = hvFetchBatch(f, b, n, &n);
  f->st.prefetch_blocks += n;
  return rc;
}

static int hvPrefetchRanges(HvFile *f, const sqlite3_int64 *off,
                            const int *len, int n){
  int bs = f->cache.bs, i, nb = 0, rc;
  sqlite3_int64 *b, total = 0;
  for(i=0; i<n; i++){
    if( len[i]<=0 || off[i]<0 ) continue;
    total += (off[i]+len[i]-1)/bs - off[i]/bs + 1;
  }
  if( total==0 ){ f->st.prefetch_calls++; return SQLITE_OK; }
  b = sqlite3_malloc64(sizeof(sqlite3_int64)*total);
  if( !b ) return SQLITE_NOMEM;
  for(i=0; i<n; i++){
    sqlite3_int64 k;
    if( len[i]<=0 || off[i]<0 ) continue;
    for(k=off[i]/bs; k<=(off[i]+len[i]-1)/bs; k++) b[nb++] = k;
  }
  rc = hvPrefetchBlocks(f, b, nb);
  sqlite3_free(b);
  return rc;
}

static int hvDbPageSize(HvFile *f){
  unsigned char h[2];
  int ps;
  if( hvRead(&f->base, h, 2, 16)!=SQLITE_OK ) return 0;
  ps = (h[0]<<8) | h[1];
  return ps==1 ? 65536 : ps;
}

static void hvResetStats(HvFile *f){
  memset(&f->st, 0, sizeof(f->st));
  hvNetResetCounters(&f->net);
  f->nlog = 0;
  f->cache.peak = f->cache.n;
  f->cache.pin_evictions = 0;
}

static int hvFileControl(sqlite3_file *pFile, int op, void *pArg){
  HvFile *f = (HvFile*)pFile;
  switch( op ){
    case HTTPVFS_FCNTL_PREFETCH: {
      HttpvfsRanges *r = (HttpvfsRanges*)pArg;
      return hvPrefetchRanges(f, r->offsets, r->lengths, r->n);
    }
    case HTTPVFS_FCNTL_PREFETCH_PAGES: {
      HttpvfsPages *p = (HttpvfsPages*)pArg;
      int ps = p->page_size ? p->page_size : hvDbPageSize(f);
      int bs = f->cache.bs, i, nb = 0, rc;
      int per = ps>bs ? ps/bs : 1;
      sqlite3_int64 *b;
      if( ps<=0 ) return SQLITE_IOERR;
      b = sqlite3_malloc64(sizeof(sqlite3_int64)*(p->n*per + 1));
      if( !b ) return SQLITE_NOMEM;
      for(i=0; i<p->n; i++){
        sqlite3_int64 o = (sqlite3_int64)(p->pgnos[i]-1)*ps;
        int k;
        for(k=0; k<per; k++) b[nb++] = (o + (sqlite3_int64)k*bs)/bs;
      }
      rc = hvPrefetchBlocks(f, b, nb);
      sqlite3_free(b);
      return rc;
    }
    case HTTPVFS_FCNTL_STATS: {
      HttpvfsStats *s = (HttpvfsStats*)pArg;
      *s = f->st;
      s->file_size = f->size;
      s->block_size = f->cache.bs;
      s->cache_blocks = f->cache.cap;
      s->cached_blocks = f->cache.n;
      s->cache_max_blocks = f->cache.hard;
      s->pinned_blocks = f->cache.pin.n;
      s->peak_blocks = f->cache.peak;
      s->pin_evictions = f->cache.pin_evictions;
      return SQLITE_OK;
    }
    case HTTPVFS_FCNTL_RESET_STATS: {
      hvResetStats(f);
      return SQLITE_OK;
    }
    case HTTPVFS_FCNTL_RELEASE: {
      hvCacheRelease(&f->cache);
      return SQLITE_OK;
    }
    case HTTPVFS_FCNTL_SPECULATE: {
      int *op2 = (int*)pArg;
      if( *op2==1 ){
        f->speculating = 1;
        f->nspec = 0;
        return SQLITE_OK;
      }else{
        int n = 0, rc = SQLITE_OK;
        f->speculating = 0;
        if( *op2==0 ){
          rc = hvFetchBatch(f, f->spec, f->nspec, &n);
          f->st.prefetch_blocks += n;
        }
        f->nspec = 0;
        *op2 = n;
        return rc;
      }
    }
    case 0x44414e01: {
      /* DENSE_ANN_FCNTL_PREFETCH from ext/dense/rawpage.h:
      ** struct { int n; int len; const sqlite3_int64 *offsets; } */
      struct { int n; int len; const sqlite3_int64 *offsets; } *r = pArg;
      int i, rc;
      int *lens = sqlite3_malloc64(sizeof(int)*(r->n>0 ? r->n : 1));
      if( !lens ) return SQLITE_NOMEM;
      for(i=0; i<r->n; i++) lens[i] = r->len;
      rc = hvPrefetchRanges(f, r->offsets, lens, r->n);
      sqlite3_free(lens);
      return rc;
    }
    case HTTPVFS_FCNTL_PAGE_SIZE: {
      *(int*)pArg = f->cache.bs;
      return SQLITE_OK;
    }
    case SQLITE_FCNTL_VFSNAME: {
      *(char**)pArg = sqlite3_mprintf("%s", HV_VFS_NAME);
      return SQLITE_OK;
    }
    case SQLITE_FCNTL_PRAGMA: {
      /* PRAGMA httpvfs_stats; PRAGMA httpvfs_reset; */
      char **a = (char**)pArg;
      if( sqlite3_stricmp(a[1], "httpvfs_stats")==0 ){
        HttpvfsStats *s = &f->st;
        a[0] = sqlite3_mprintf(
          "requests=%lld bytes=%lld rounds=%lld reads=%lld hits=%lld "
          "misses=%lld prefetch_calls=%lld prefetch_blocks=%lld net_ms=%.1f "
          "cached_blocks=%d overfetch=%lld planned_rounds=%lld "
          "multipart_requests=%lld max_req=%d rtt_ms=%.1f bw_kbps=%.0f "
          "cache_blocks=%d cache_max_blocks=%d pinned_blocks=%d "
          "peak_blocks=%d pin_evictions=%lld",
          s->requests, s->bytes, s->rounds, s->reads, s->cache_hits,
          s->cache_misses, s->prefetch_calls, s->prefetch_blocks, s->net_ms,
          f->cache.n, f->net.overfetch, f->net.planned_rounds,
          f->net.mp_requests, f->net.max_req, f->net.rtt_ms,
          f->net.bw_Bpms*8.0, f->cache.cap, f->cache.hard, f->cache.pin.n,
          f->cache.peak, f->cache.pin_evictions);
        return SQLITE_OK;
      }
      if( sqlite3_stricmp(a[1], "httpvfs_release")==0 ){
        hvCacheRelease(&f->cache);
        a[0] = sqlite3_mprintf("ok");
        return SQLITE_OK;
      }
      if( sqlite3_stricmp(a[1], "httpvfs_reset")==0 ){
        if( a[2] && sqlite3_stricmp(a[2], "cache")==0 ) hvCacheClear(&f->cache);
        hvResetStats(f);
        a[0] = sqlite3_mprintf("ok");   /* NULL would be a NULL column name */
        return SQLITE_OK;
      }
      return SQLITE_NOTFOUND;
    }
  }
  return SQLITE_NOTFOUND;
}

static int hvClose(sqlite3_file *pFile){
  HvFile *f = (HvFile*)pFile;
  if( f->be ) f->be->xClose(f->be);
  hvCacheFree(&f->cache);
  sqlite3_free(f->spec);
  sqlite3_free(f->log);
  return SQLITE_OK;
}
static int hvWrite(sqlite3_file *p, const void *z, int n, sqlite3_int64 o){
  (void)p; (void)z; (void)n; (void)o; return SQLITE_READONLY;
}
static int hvTruncate(sqlite3_file *p, sqlite3_int64 s){
  (void)p; (void)s; return SQLITE_READONLY;
}
static int hvSync(sqlite3_file *p, int fl){ (void)p; (void)fl; return SQLITE_OK; }
static int hvFileSize(sqlite3_file *p, sqlite3_int64 *pSize){
  *pSize = ((HvFile*)p)->size; return SQLITE_OK;
}
static int hvLock(sqlite3_file *p, int l){ (void)p; (void)l; return SQLITE_OK; }
static int hvCheckReservedLock(sqlite3_file *p, int *pOut){
  (void)p; *pOut = 0; return SQLITE_OK;
}
static int hvSectorSize(sqlite3_file *p){ (void)p; return 4096; }
static int hvDeviceCharacteristics(sqlite3_file *p){
  (void)p; return SQLITE_IOCAP_IMMUTABLE;
}

static const sqlite3_io_methods hvIoMethods = {
  1, hvClose, hvRead, hvWrite, hvTruncate, hvSync, hvFileSize, hvLock, hvLock,
  hvCheckReservedLock, hvFileControl, hvSectorSize, hvDeviceCharacteristics,
  0, 0, 0, 0, 0, 0
};

/*
** Default cache budget: 1/64 of the file, at least 4 MiB and at most
** 64 MiB. Small databases (10k documents) keep the old 4 MiB; at 1M
** documents (0.7-4 GB per index) this gives 11-64 MiB, enough for a
** query's working set plus the pages shared between queries, while a
** browser tab's memory stays bounded.
*/
#define HV_AUTO_MIN_KB  4096
#define HV_AUTO_MAX_KB  65536
#define HV_AUTO_DIV     64
static sqlite3_int64 hvAutoKb(sqlite3_int64 size, sqlite3_int64 min_kb){
  sqlite3_int64 kb = size/HV_AUTO_DIV/1024;
  if( kb>HV_AUTO_MAX_KB ) kb = HV_AUTO_MAX_KB;
  if( kb<min_kb ) kb = min_kb;
  return kb;
}
/* Budget and hard limit in blocks for a budget of cache_kb. */
static void hvCacheSize(HvFile *f, sqlite3_int64 cache_kb, int bs,
                        int *pCap, int *pHard){
  sqlite3_int64 cap = (cache_kb*1024)/bs, hard;
  if( cap<16 ) cap = 16;
  if( cap>(1<<26) ) cap = 1<<26;
  hard = f->cache_max_kb>0 ? (f->cache_max_kb*1024)/bs : 4*cap;
  if( hard<cap ) hard = cap;
  if( hard>(1<<28) ) hard = 1<<28;
  *pCap = (int)cap; *pHard = (int)hard;
}

static int hvOpen(sqlite3_vfs *pVfs, const char *zName, sqlite3_file *pFile,
                  int flags, int *pOutFlags){
  HvFile *f = (HvFile*)pFile;
  int rc, bs, ra_kb, gap_kb;
  sqlite3_int64 cache_kb;
  (void)pVfs;
  if( !(flags & SQLITE_OPEN_MAIN_DB) || zName==0 ){
    /* Temp files, journals: hand to the default VFS in the same storage. */
    return hvg.pOrig->xOpen(hvg.pOrig, zName, pFile, flags, pOutFlags);
  }
  memset(f, 0, sizeof(*f));
  {
    const char *z = sqlite3_uri_parameter(zName, "cache_kb");
    cache_kb = z ? sqlite3_uri_int64(zName, "cache_kb", 0) : 0;
    f->cache_auto = cache_kb<=0;
    f->cache_min_kb = sqlite3_uri_int64(zName, "cache_min_kb", HV_AUTO_MIN_KB);
    if( f->cache_auto ) cache_kb = f->cache_min_kb;
    f->cache_max_kb = sqlite3_uri_int64(zName, "cache_max_kb", 0);
  }
  bs = (int)sqlite3_uri_int64(zName, "block", 4096);
  ra_kb = (int)sqlite3_uri_int64(zName, "readahead_kb", 1024);
  gap_kb = (int)sqlite3_uri_int64(zName, "gap_kb", 0);
  if( bs<512 || (bs & (bs-1)) ) return SQLITE_CANTOPEN;
  f->log_max = (int)sqlite3_uri_int64(zName, "log_max", 1000000);
  f->ra_max = (int)(((sqlite3_int64)ra_kb*1024)/bs);
  f->gap = (int)(((sqlite3_int64)gap_kb*1024)/bs);
  f->next_seq = -1;
  f->net.max_req = (int)sqlite3_uri_int64(zName, "max_req", 0);
  f->net.multipart = (int)sqlite3_uri_int64(zName, "multipart", 0);
  f->net.max_parts = (int)sqlite3_uri_int64(zName, "max_parts", 100);
  f->net.autoest = (int)sqlite3_uri_int64(zName, "net_auto", 1);
  {
    const char *z = sqlite3_uri_parameter(zName, "rtt_ms");
    f->net.rtt_ms = z ? atof(z) : 100.0;
    z = sqlite3_uri_parameter(zName, "bw_kbps");
    f->net.bw_Bpms = (z ? atof(z) : 10000.0) / 8.0;   /* kbit/s -> bytes/ms */
    if( f->net.bw_Bpms<=0 ) f->net.bw_Bpms = 1;
  }
  {
    int cap, hard;
    hvCacheSize(f, cache_kb, bs, &cap, &hard);
    rc = hvCacheInit(&f->cache, bs, cap, hard);
    if( rc ){ hvCacheFree(&f->cache); return rc; }
  }

#ifdef __EMSCRIPTEN__
  {
    HvJsBackend *be;
    int h = hv_js_open(zName);
    if( h<0 ){ hvCacheFree(&f->cache); return SQLITE_CANTOPEN; }
    be = sqlite3_malloc(sizeof(*be));
    if( !be ){ hv_js_close(h); hvCacheFree(&f->cache); return SQLITE_NOMEM; }
    be->base.xFetch = hvJsFetch;
    be->base.xClose = hvJsClose;
    be->h = h;
    f->be = &be->base;
    f->size = -1;
  }
#else
  {
    HvFileBackend *be;
    struct stat stbuf;
    int fd = open(zName, O_RDONLY);
    if( fd<0 ){ hvCacheFree(&f->cache); return SQLITE_CANTOPEN; }
    if( fstat(fd, &stbuf) ){ close(fd); hvCacheFree(&f->cache); return SQLITE_CANTOPEN; }
    be = sqlite3_malloc(sizeof(*be));
    if( !be ){ close(fd); hvCacheFree(&f->cache); return SQLITE_NOMEM; }
    be->base.xFetch = hvFileFetch;
    be->base.xClose = hvFileClose;
    be->fd = fd;
    {
      const char *z = sqlite3_uri_parameter(zName, "latency_ms");
      be->latency_ms = z ? atof(z) : 0.0;
    }
    f->be = &be->base;
    f->size = stbuf.st_size;
  }
#endif

  f->base.pMethods = &hvIoMethods;
  if( f->size<0 ){
    /* Learn the size from the first request, which also brings in the
    ** header block that SQLite is about to read. */
    sqlite3_int64 b = 0;
    rc = hvFetchBlocks(f, &b, 1, 0, 0, 0, 0);
    if( rc!=SQLITE_OK || f->size<0 ){
      hvClose(pFile);
      f->base.pMethods = 0;
      return SQLITE_CANTOPEN;
    }
  }
  if( f->cache_auto ){
    int cap, hard;
    hvCacheSize(f, hvAutoKb(f->size, f->cache_min_kb), bs, &cap, &hard);
    hvCacheSetBudget(&f->cache, cap, hard);
  }
  if( pOutFlags ) *pOutFlags = SQLITE_OPEN_READONLY | SQLITE_OPEN_MAIN_DB;
  return SQLITE_OK;
}

static int hvDelete(sqlite3_vfs *v, const char *z, int s){
  (void)v; return hvg.pOrig->xDelete(hvg.pOrig, z, s);
}
static int hvAccess(sqlite3_vfs *v, const char *z, int fl, int *pOut){
  (void)v; return hvg.pOrig->xAccess(hvg.pOrig, z, fl, pOut);
}
static int hvFullPathname(sqlite3_vfs *v, const char *z, int n, char *zOut){
  (void)v; return hvg.pOrig->xFullPathname(hvg.pOrig, z, n, zOut);
}
static void *hvDlOpen(sqlite3_vfs *v, const char *z){
  (void)v; return hvg.pOrig->xDlOpen(hvg.pOrig, z);
}
static void hvDlError(sqlite3_vfs *v, int n, char *z){
  (void)v; hvg.pOrig->xDlError(hvg.pOrig, n, z);
}
static void (*hvDlSym(sqlite3_vfs *v, void *p, const char *z))(void){
  (void)v; return hvg.pOrig->xDlSym(hvg.pOrig, p, z);
}
static void hvDlClose(sqlite3_vfs *v, void *p){
  (void)v; hvg.pOrig->xDlClose(hvg.pOrig, p);
}
static int hvRandomness(sqlite3_vfs *v, int n, char *z){
  (void)v; return hvg.pOrig->xRandomness(hvg.pOrig, n, z);
}
static int hvSleep(sqlite3_vfs *v, int us){
  (void)v; return hvg.pOrig->xSleep(hvg.pOrig, us);
}
static int hvCurrentTime(sqlite3_vfs *v, double *p){
  (void)v; return hvg.pOrig->xCurrentTime(hvg.pOrig, p);
}
static int hvGetLastError(sqlite3_vfs *v, int n, char *z){
  (void)v; return hvg.pOrig->xGetLastError ? hvg.pOrig->xGetLastError(hvg.pOrig, n, z) : 0;
}
static int hvCurrentTimeInt64(sqlite3_vfs *v, sqlite3_int64 *p){
  (void)v; return hvg.pOrig->xCurrentTimeInt64(hvg.pOrig, p);
}

static sqlite3_vfs hvVfs = {
  2, 0, 1024, 0, HV_VFS_NAME, 0,
  hvOpen, hvDelete, hvAccess, hvFullPathname, hvDlOpen, hvDlError, hvDlSym,
  hvDlClose, hvRandomness, hvSleep, hvCurrentTime, hvGetLastError,
  hvCurrentTimeInt64, 0, 0, 0
};

/* Register the "httpvfs" VFS (not as the default). Idempotent. */
int httpvfs_register(void){
  if( sqlite3_vfs_find(HV_VFS_NAME) ) return SQLITE_OK;
  hvg.pOrig = sqlite3_vfs_find(0);
  if( !hvg.pOrig ) return SQLITE_ERROR;
  hvVfs.szOsFile = hvg.pOrig->szOsFile > (int)sizeof(HvFile)
                 ? hvg.pOrig->szOsFile : (int)sizeof(HvFile);
  hvVfs.mxPathname = hvg.pOrig->mxPathname;
  return sqlite3_vfs_register(&hvVfs, 0);
}

/* Access to the request log of a database's main file (for the JS API). */
static HvFile *hvMainFile(sqlite3 *db){
  sqlite3_file *p = 0;
  if( sqlite3_file_control(db, "main", SQLITE_FCNTL_FILE_POINTER, &p) ) return 0;
  if( !p || p->pMethods!=&hvIoMethods ) return 0;
  return (HvFile*)p;
}
int httpvfs_log_count(sqlite3 *db){
  HvFile *f = hvMainFile(db);
  return f ? f->nlog : 0;
}
/* Entry i as 6 doubles: offset, length, round, t_start, t_end, request
** (ranges of one multi-range request share it within their round). */
int httpvfs_log_entry(sqlite3 *db, int i, double *out6){
  HvFile *f = hvMainFile(db);
  if( !f || i<0 || i>=f->nlog ) return SQLITE_RANGE;
  out6[0] = f->log[i].off;
  out6[1] = f->log[i].len;
  out6[2] = f->log[i].round;
  out6[3] = f->log[i].t0;
  out6[4] = f->log[i].t1;
  out6[5] = f->log[i].req;
  return SQLITE_OK;
}
double httpvfs_now(void){ return hv_now(); }

#if !defined(SQLITE_CORE)
#ifdef _WIN32
__declspec(dllexport)
#endif
int sqlite3_httpvfs_init(sqlite3 *db, char **pzErrMsg,
                         const sqlite3_api_routines *pApi){
  int rc;
  (void)db; (void)pzErrMsg;
  SQLITE_EXTENSION_INIT2(pApi);
  rc = httpvfs_register();
  return rc==SQLITE_OK ? SQLITE_OK_LOAD_PERMANENTLY : rc;
}
#endif

/*
** Helpers for the JavaScript API: stats as a flat array of doubles, in the
** order of HttpvfsStats, and bulk copy of the request log.
*/
int httpvfs_stats_array(sqlite3 *db, double *out){
  HttpvfsStats s;
  int rc = sqlite3_file_control(db, "main", HTTPVFS_FCNTL_STATS, &s);
  if( rc ) return rc;
  out[0] = (double)s.requests;     out[1] = (double)s.bytes;
  out[2] = (double)s.rounds;       out[3] = (double)s.reads;
  out[4] = (double)s.cache_hits;   out[5] = (double)s.cache_misses;
  out[6] = (double)s.prefetch_calls; out[7] = (double)s.prefetch_blocks;
  out[8] = (double)s.spec_misses;  out[9] = (double)s.file_size;
  out[10] = s.block_size;          out[11] = s.cache_blocks;
  out[12] = s.cached_blocks;       out[13] = s.net_ms;
  out[14] = s.cache_max_blocks;    out[15] = s.pinned_blocks;
  out[16] = s.peak_blocks;         out[17] = (double)s.pin_evictions;
  return SQLITE_OK;
}
int httpvfs_reset(sqlite3 *db, int clear_cache){
  HvFile *f = hvMainFile(db);
  if( !f ) return SQLITE_NOTFOUND;
  if( clear_cache ) hvCacheClear(&f->cache);
  hvResetStats(f);
  f->next_seq = -1;
  f->ra = 0;
  return SQLITE_OK;
}
/* End of a statement (JavaScript API): unpin and shrink to the budget. */
int httpvfs_release_db(sqlite3 *db){
  HvFile *f = hvMainFile(db);
  if( !f ) return SQLITE_NOTFOUND;
  hvCacheRelease(&f->cache);
  return SQLITE_OK;
}
/* Copy up to n log entries starting at i0, 6 doubles each (as
** httpvfs_log_entry); returns the count. */
int httpvfs_log_copy(sqlite3 *db, int i0, int n, double *out){
  HvFile *f = hvMainFile(db);
  int i;
  if( !f || i0<0 ) return 0;
  if( i0+n>f->nlog ) n = f->nlog - i0;
  for(i=0; i<n; i++){
    const HvLogEntry *e = &f->log[i0+i];
    out[6*i] = e->off; out[6*i+1] = e->len; out[6*i+2] = e->round;
    out[6*i+3] = e->t0; out[6*i+4] = e->t1; out[6*i+5] = e->req;
  }
  return n<0 ? 0 : n;
}

/*
** Network parameters (JavaScript API). Arguments below 0 leave the current
** value: max_req (0 = no request budget), rtt_ms, bw_kbps, multipart (0/1),
** net_auto (0/1: refit rtt and bandwidth from observed rounds), max_parts.
*/
int httpvfs_net_config(sqlite3 *db, int max_req, double rtt_ms, double bw_kbps,
                       int multipart, int autoest, int max_parts){
  HvFile *f = hvMainFile(db);
  if( !f ) return SQLITE_NOTFOUND;
  if( max_req>=0 ) f->net.max_req = max_req;
  if( rtt_ms>=0 ) f->net.rtt_ms = rtt_ms;
  if( bw_kbps>0 ) f->net.bw_Bpms = bw_kbps/8.0;
  if( multipart>=0 ){ f->net.multipart = multipart; f->net.mp_failed = 0; }
  if( autoest>=0 ) f->net.autoest = autoest;
  if( max_parts>0 ) f->net.max_parts = max_parts;
  if( rtt_ms>=0 || bw_kbps>0 ) f->net.wn = f->net.wi = 0;   /* restart the fit */
  return SQLITE_OK;
}
/* Network state as 10 doubles: max_req, rtt_ms, bw_kbps, multipart,
** net_auto, overfetch bytes, planned rounds, multi-range requests,
** multipart_failed, observed rounds in the estimator window. */
int httpvfs_net_state(sqlite3 *db, double *out){
  HvFile *f = hvMainFile(db);
  if( !f ) return SQLITE_NOTFOUND;
  out[0] = f->net.max_req;   out[1] = f->net.rtt_ms;
  out[2] = f->net.bw_Bpms*8.0; out[3] = f->net.multipart;
  out[4] = f->net.autoest;   out[5] = (double)f->net.overfetch;
  out[6] = (double)f->net.planned_rounds; out[7] = (double)f->net.mp_requests;
  out[8] = f->net.mp_failed; out[9] = f->net.wn;
  return SQLITE_OK;
}
