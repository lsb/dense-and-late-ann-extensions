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
**   cache_kb      VFS cache size in KiB (default 4096)
**   block         cache block (and minimum request) size in bytes (default 4096)
**   readahead_kb  maximum sequential readahead in KiB, 0 disables (default 1024)
**   gap_kb        merge ranges of one batch whose gap is at most this (default 0)
**   latency_ms    native backend only: simulated round-trip time per round
**   log_max       maximum log entries kept (default 1000000)
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
  ** backend learns it (it is -1 on entry when unknown). Returns SQLITE_OK or
  ** an SQLite error code. */
  int (*xFetch)(HvBackend*, int n, const double *off, const int *len,
                unsigned char **dest, double *t0, double *t1,
                sqlite3_int64 *pSize);
  void (*xClose)(HvBackend*);
};

#ifdef __EMSCRIPTEN__

#ifdef HV_SYNC_JS
/* Synchronous backend (no Asyncify/JSPI): Module.httpvfsFetchSync must
** block until the batch is in memory, e.g. a worker doing parallel fetches
** while this thread waits on a SharedArrayBuffer with Atomics.wait. */
EM_JS(int, hv_js_fetch, (int h, int n, const double *off, const int *len,
      unsigned char **dest, double *t0, double *t1, double *pSize), {
  return Module.httpvfsFetchSync(h, n, off, len, dest, t0, t1, pSize);
});
#else
EM_ASYNC_JS(int, hv_js_fetch, (int h, int n, const double *off, const int *len,
            unsigned char **dest, double *t0, double *t1, double *pSize), {
  return await Module.httpvfsFetch(h, n, off, len, dest, t0, t1, pSize);
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
                     sqlite3_int64 *pSize){
  double sz = (double)*pSize;
  int rc = hv_js_fetch(((HvJsBackend*)p)->h, n, off, len, dest, t0, t1, &sz);
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
                       sqlite3_int64 *pSize){
  HvFileBackend *f = (HvFileBackend*)p;
  double start = hv_now(), end;
  int i;
  (void)pSize;
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
/* LRU block cache                                                           */

typedef struct HvCache {
  int bs;                  /* block size */
  int cap;                 /* capacity in blocks */
  int n;                   /* blocks in use (slots 0..n-1) */
  unsigned char *data;     /* cap*bs bytes */
  sqlite3_int64 *key;      /* block index of each slot */
  int *prev, *next;        /* LRU list, head is most recently used */
  int head, tail;
  int *hnext;              /* hash chains */
  int *bucket;
  unsigned nbucket;        /* power of two */
} HvCache;

static unsigned hvHash(sqlite3_int64 k, unsigned nb){
  sqlite3_uint64 x = (sqlite3_uint64)k * 0x9E3779B97F4A7C15ull;
  return (unsigned)(x>>32) & (nb-1);
}

static int hvCacheInit(HvCache *c, int bs, int cap){
  int i;
  memset(c, 0, sizeof(*c));
  c->bs = bs; c->cap = cap; c->head = c->tail = -1;
  c->nbucket = 1;
  while( c->nbucket < (unsigned)cap*2 ) c->nbucket <<= 1;
  c->data = sqlite3_malloc64((sqlite3_uint64)cap*bs);
  c->key = sqlite3_malloc64(sizeof(sqlite3_int64)*cap);
  c->prev = sqlite3_malloc64(sizeof(int)*cap);
  c->next = sqlite3_malloc64(sizeof(int)*cap);
  c->hnext = sqlite3_malloc64(sizeof(int)*cap);
  c->bucket = sqlite3_malloc64(sizeof(int)*c->nbucket);
  if( !c->data || !c->key || !c->prev || !c->next || !c->hnext || !c->bucket ){
    return SQLITE_NOMEM;
  }
  for(i=0; i<(int)c->nbucket; i++) c->bucket[i] = -1;
  return SQLITE_OK;
}

static void hvCacheFree(HvCache *c){
  sqlite3_free(c->data); sqlite3_free(c->key); sqlite3_free(c->prev);
  sqlite3_free(c->next); sqlite3_free(c->hnext); sqlite3_free(c->bucket);
  memset(c, 0, sizeof(*c));
}

static void hvLruUnlink(HvCache *c, int s){
  if( c->prev[s]>=0 ) c->next[c->prev[s]] = c->next[s]; else c->head = c->next[s];
  if( c->next[s]>=0 ) c->prev[c->next[s]] = c->prev[s]; else c->tail = c->prev[s];
}
static void hvLruPushFront(HvCache *c, int s){
  c->prev[s] = -1; c->next[s] = c->head;
  if( c->head>=0 ) c->prev[c->head] = s;
  c->head = s;
  if( c->tail<0 ) c->tail = s;
}

/* Return the slot holding block k (and mark it recently used), or -1. */
static int hvCacheGet(HvCache *c, sqlite3_int64 k, int touch){
  int s = c->bucket[hvHash(k, c->nbucket)];
  while( s>=0 && c->key[s]!=k ) s = c->hnext[s];
  if( s>=0 && touch && c->head!=s ){ hvLruUnlink(c, s); hvLruPushFront(c, s); }
  return s;
}

/* Insert block k with bs bytes from src (zero-padded to the block size). */
static void hvCachePut(HvCache *c, sqlite3_int64 k, const unsigned char *src,
                       int nsrc){
  int s = hvCacheGet(c, k, 1);
  if( s<0 ){
    unsigned h;
    if( c->n < c->cap ){
      s = c->n++;
    }else{
      int *pp;
      s = c->tail;
      hvLruUnlink(c, s);
      pp = &c->bucket[hvHash(c->key[s], c->nbucket)];
      while( *pp!=s ) pp = &c->hnext[*pp];
      *pp = c->hnext[s];
    }
    c->key[s] = k;
    h = hvHash(k, c->nbucket);
    c->hnext[s] = c->bucket[h];
    c->bucket[h] = s;
    hvLruPushFront(c, s);
  }
  memcpy(c->data + (size_t)s*c->bs, src, nsrc);
  if( nsrc<c->bs ) memset(c->data + (size_t)s*c->bs + nsrc, 0, c->bs-nsrc);
}

static void hvCacheClear(HvCache *c){
  int i;
  c->n = 0; c->head = c->tail = -1;
  for(i=0; i<(int)c->nbucket; i++) c->bucket[i] = -1;
}

/* ------------------------------------------------------------------------ */
/* File                                                                      */

typedef struct HvLogEntry {
  double off;
  int len;
  int round;
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

static void hvLog(HvFile *f, double off, int len, int round, double t0,
                  double t1){
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
  f->log[f->nlog].t0 = t0;
  f->log[f->nlog].t1 = t1;
  f->nlog++;
}

/*
** Fetch the sorted, distinct, uncached blocks blk[0..n-1] in one round and
** insert them into the cache. Blocks closer than the merge gap share a
** request (the blocks in the gap are fetched and cached too). If out is not
** NULL, the bytes [out_off, out_off+out_len) are also copied into it straight
** from the fetched data, so that a read larger than the cache still works.
*/
static int hvFetchBlocks(HvFile *f, const sqlite3_int64 *blk, int n,
                         unsigned char *out, sqlite3_int64 out_off,
                         int out_len){
  int bs = f->cache.bs;
  int nr = 0, i, j, rc;
  double *off, *t0, *t1;
  int *len;
  unsigned char **dest;
  sqlite3_int64 *first;
  sqlite3_int64 size = f->size;
  double tmin, tmax;

  if( n<=0 ) return SQLITE_OK;
  off = sqlite3_malloc64(sizeof(double)*n*3);
  len = sqlite3_malloc64(sizeof(int)*n);
  dest = sqlite3_malloc64(sizeof(unsigned char*)*n);
  first = sqlite3_malloc64(sizeof(sqlite3_int64)*n);
  if( !off || !len || !dest || !first ){ rc = SQLITE_NOMEM; goto done; }
  memset(dest, 0, sizeof(unsigned char*)*n);
  t0 = off + n; t1 = off + 2*n;

  /* Coalesce into ranges. */
  for(i=0; i<n; i=j){
    sqlite3_int64 last = blk[i];
    for(j=i+1; j<n && blk[j]-last-1 <= f->gap; j++) last = blk[j];
    first[nr] = blk[i];
    off[nr] = (double)(blk[i]*bs);
    {
      sqlite3_int64 end = (last+1)*bs;
      if( size>=0 && end>size ) end = size;
      len[nr] = (int)(end - blk[i]*bs);
    }
    if( len[nr]<=0 ){ continue; }
    dest[nr] = sqlite3_malloc64(len[nr]);
    if( !dest[nr] ){ rc = SQLITE_NOMEM; goto done; }
    nr++;
  }
  if( nr==0 ){ rc = SQLITE_OK; goto done; }

  f->st.rounds++;
  rc = f->be->xFetch(f->be, nr, off, len, dest, t0, t1, &size);
  if( rc!=SQLITE_OK ) goto done;
  if( f->size<0 ) f->size = size;

  tmin = t0[0]; tmax = t1[0];
  for(i=0; i<nr; i++){
    sqlite3_int64 b;
    int k;
    if( t0[i]<tmin ) tmin = t0[i];
    if( t1[i]>tmax ) tmax = t1[i];
    f->st.requests++;
    f->st.bytes += len[i];
    hvLog(f, off[i], len[i], (int)f->st.rounds, t0[i], t1[i]);
    for(k=0, b=first[i]; k<len[i]; k+=bs, b++){
      int m = len[i]-k < bs ? len[i]-k : bs;
      if( hvCacheGet(&f->cache, b, 0)<0 ) hvCachePut(&f->cache, b, dest[i]+k, m);
    }
    if( out ){
      /* Copy the overlap of this range with the output window. */
      sqlite3_int64 a0 = (sqlite3_int64)off[i], a1 = a0 + len[i];
      sqlite3_int64 o0 = out_off, o1 = out_off + out_len;
      sqlite3_int64 lo = a0>o0 ? a0 : o0, hi = a1<o1 ? a1 : o1;
      if( lo<hi ) memcpy(out + (lo-o0), dest[i] + (lo-a0), (size_t)(hi-lo));
    }
  }
  f->st.net_ms += tmax - tmin;

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

/* Sort, deduplicate and drop cached blocks; returns the new count. */
static int hvUncached(HvFile *f, sqlite3_int64 *b, int n){
  int i, m = 0;
  qsort(b, n, sizeof(*b), hvI64Cmp);
  for(i=0; i<n; i++){
    if( i>0 && b[i]==b[i-1] ) continue;
    if( b[i]<0 || (f->size>=0 && b[i]>=hvNumBlocks(f)) ) continue;
    if( hvCacheGet(&f->cache, b[i], 0)>=0 ) continue;
    b[m++] = b[i];
  }
  return m;
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
      memcpy(out + (lo-iOfst), f->cache.data + (size_t)s*bs + (lo - b*bs),
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
    rc = hvFetchBlocks(f, miss, nmiss, out, iOfst, avail);
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
  n = hvUncached(f, b, n);
  /* Never fetch more than the cache can hold in one batch. */
  if( n > f->cache.cap/2 ) n = f->cache.cap/2;
  if( n==0 ) return SQLITE_OK;
  rc = hvFetchBlocks(f, b, n, 0, 0, 0);
  if( rc==SQLITE_OK ) f->st.prefetch_blocks += n;
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
      return SQLITE_OK;
    }
    case HTTPVFS_FCNTL_RESET_STATS: {
      memset(&f->st, 0, sizeof(f->st));
      f->nlog = 0;
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
          n = hvUncached(f, f->spec, f->nspec);
          if( n > f->cache.cap/2 ) n = f->cache.cap/2;
          if( n>0 ) rc = hvFetchBlocks(f, f->spec, n, 0, 0, 0);
          if( rc==SQLITE_OK ) f->st.prefetch_blocks += n;
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
          "cached_blocks=%d",
          s->requests, s->bytes, s->rounds, s->reads, s->cache_hits,
          s->cache_misses, s->prefetch_calls, s->prefetch_blocks, s->net_ms,
          f->cache.n);
        return SQLITE_OK;
      }
      if( sqlite3_stricmp(a[1], "httpvfs_reset")==0 ){
        memset(&f->st, 0, sizeof(f->st));
        f->nlog = 0;
        if( a[2] && sqlite3_stricmp(a[2], "cache")==0 ) hvCacheClear(&f->cache);
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

static int hvOpen(sqlite3_vfs *pVfs, const char *zName, sqlite3_file *pFile,
                  int flags, int *pOutFlags){
  HvFile *f = (HvFile*)pFile;
  int rc, cache_kb, bs, ra_kb, gap_kb;
  (void)pVfs;
  if( !(flags & SQLITE_OPEN_MAIN_DB) || zName==0 ){
    /* Temp files, journals: hand to the default VFS in the same storage. */
    return hvg.pOrig->xOpen(hvg.pOrig, zName, pFile, flags, pOutFlags);
  }
  memset(f, 0, sizeof(*f));
  cache_kb = (int)sqlite3_uri_int64(zName, "cache_kb", 4096);
  bs = (int)sqlite3_uri_int64(zName, "block", 4096);
  ra_kb = (int)sqlite3_uri_int64(zName, "readahead_kb", 1024);
  gap_kb = (int)sqlite3_uri_int64(zName, "gap_kb", 0);
  if( bs<512 || (bs & (bs-1)) ) return SQLITE_CANTOPEN;
  f->log_max = (int)sqlite3_uri_int64(zName, "log_max", 1000000);
  f->ra_max = (int)(((sqlite3_int64)ra_kb*1024)/bs);
  f->gap = (int)(((sqlite3_int64)gap_kb*1024)/bs);
  f->next_seq = -1;
  {
    sqlite3_int64 cap = ((sqlite3_int64)cache_kb*1024)/bs;
    if( cap<16 ) cap = 16;
    rc = hvCacheInit(&f->cache, bs, (int)cap);
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
    rc = hvFetchBlocks(f, &b, 1, 0, 0, 0);
    if( rc!=SQLITE_OK || f->size<0 ){
      hvClose(pFile);
      f->base.pMethods = 0;
      return SQLITE_CANTOPEN;
    }
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
/* Entry i as 5 doubles: offset, length, round, t_start, t_end. */
int httpvfs_log_entry(sqlite3 *db, int i, double *out5){
  HvFile *f = hvMainFile(db);
  if( !f || i<0 || i>=f->nlog ) return SQLITE_RANGE;
  out5[0] = f->log[i].off;
  out5[1] = f->log[i].len;
  out5[2] = f->log[i].round;
  out5[3] = f->log[i].t0;
  out5[4] = f->log[i].t1;
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
  return SQLITE_OK;
}
int httpvfs_reset(sqlite3 *db, int clear_cache){
  HvFile *f = hvMainFile(db);
  if( !f ) return SQLITE_NOTFOUND;
  memset(&f->st, 0, sizeof(f->st));
  f->nlog = 0;
  f->next_seq = -1;
  f->ra = 0;
  if( clear_cache ) hvCacheClear(&f->cache);
  return SQLITE_OK;
}
/* Copy up to n log entries starting at i0, 5 doubles each; returns count. */
int httpvfs_log_copy(sqlite3 *db, int i0, int n, double *out){
  HvFile *f = hvMainFile(db);
  int i;
  if( !f || i0<0 ) return 0;
  if( i0+n>f->nlog ) n = f->nlog - i0;
  for(i=0; i<n; i++){
    const HvLogEntry *e = &f->log[i0+i];
    out[5*i] = e->off; out[5*i+1] = e->len; out[5*i+2] = e->round;
    out[5*i+3] = e->t0; out[5*i+4] = e->t1;
  }
  return n<0 ? 0 : n;
}
