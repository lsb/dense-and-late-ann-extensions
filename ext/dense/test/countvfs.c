/*
** countvfs.c — a pass-through VFS that counts reads of main database files.
** Test-only. Loading the extension registers VFS "countvfs" (wrapping the
** default VFS) and two SQL functions:
**
**   countvfs_reset()   clear counters
**   countvfs_stats()   JSON {"reads":N,"bytes":N,"distinct":N,"prefetch_calls":N,
**                            "prefetch_pages":N}
**
** "distinct" counts distinct file offsets read since the last reset, i.e.
** pages that a cold HTTP VFS would have had to fetch. The shim also answers
** the dense_ann prefetch file control, counting announced pages.
*/
#include "sqlite3ext.h"
SQLITE_EXTENSION_INIT1

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define DENSE_ANN_FCNTL_PREFETCH 0x44414e01
typedef struct { int n; int len; const sqlite3_int64 *offsets; } prefetch_req;

typedef struct CountFile {
  sqlite3_file base;
  int is_main;
  sqlite3_file *real;   /* follows this struct in memory */
} CountFile;

static sqlite3_vfs count_vfs;
static sqlite3_vfs *base_vfs;
static struct {
  long long reads, bytes, prefetch_calls, prefetch_pages;
  sqlite3_int64 *offs; int n, cap;   /* distinct offsets, open addressing (stored +1) */
} g;

static void note_offset(sqlite3_int64 off) {
  sqlite3_int64 key = off + 1;
  if (2 * (g.n + 1) > g.cap) {
    int ncap = g.cap ? 2 * g.cap : 4096;
    sqlite3_int64 *nk = (sqlite3_int64 *)calloc(ncap, sizeof *nk);
    for (int i = 0; i < g.cap; i++) if (g.offs[i]) {
      uint64_t h = ((uint64_t)g.offs[i] * 0x9E3779B97F4A7C15ull) >> 20;
      while (nk[h & (ncap - 1)]) h++;
      nk[h & (ncap - 1)] = g.offs[i];
    }
    free(g.offs); g.offs = nk; g.cap = ncap;
  }
  uint64_t h = ((uint64_t)key * 0x9E3779B97F4A7C15ull) >> 20;
  for (;; h++) {
    sqlite3_int64 *slot = &g.offs[h & (g.cap - 1)];
    if (*slot == key) return;
    if (!*slot) { *slot = key; g.n++; return; }
  }
}

#define REAL(f) (((CountFile *)(f))->real)

static int cClose(sqlite3_file *f) {
  int rc = REAL(f)->pMethods ? REAL(f)->pMethods->xClose(REAL(f)) : SQLITE_OK;
  return rc;
}
static int cRead(sqlite3_file *f, void *buf, int amt, sqlite3_int64 off) {
  CountFile *p = (CountFile *)f;
  if (p->is_main) { g.reads++; g.bytes += amt; note_offset(off); }
  return p->real->pMethods->xRead(p->real, buf, amt, off);
}
static int cWrite(sqlite3_file *f, const void *b, int a, sqlite3_int64 o) { return REAL(f)->pMethods->xWrite(REAL(f), b, a, o); }
static int cTruncate(sqlite3_file *f, sqlite3_int64 s) { return REAL(f)->pMethods->xTruncate(REAL(f), s); }
static int cSync(sqlite3_file *f, int fl) { return REAL(f)->pMethods->xSync(REAL(f), fl); }
static int cFileSize(sqlite3_file *f, sqlite3_int64 *s) { return REAL(f)->pMethods->xFileSize(REAL(f), s); }
static int cLock(sqlite3_file *f, int l) { return REAL(f)->pMethods->xLock(REAL(f), l); }
static int cUnlock(sqlite3_file *f, int l) { return REAL(f)->pMethods->xUnlock(REAL(f), l); }
static int cCheckReserved(sqlite3_file *f, int *r) { return REAL(f)->pMethods->xCheckReservedLock(REAL(f), r); }
static int cFileControl(sqlite3_file *f, int op, void *arg) {
  if (op == DENSE_ANN_FCNTL_PREFETCH) {
    const prefetch_req *r = (const prefetch_req *)arg;
    g.prefetch_calls++; g.prefetch_pages += r->n;
    return SQLITE_OK;
  }
  return REAL(f)->pMethods->xFileControl(REAL(f), op, arg);
}
static int cSectorSize(sqlite3_file *f) { return REAL(f)->pMethods->xSectorSize(REAL(f)); }
static int cDevChar(sqlite3_file *f) { return REAL(f)->pMethods->xDeviceCharacteristics(REAL(f)); }
static int cShmMap(sqlite3_file *f, int a, int b, int c, void volatile **d) { return REAL(f)->pMethods->xShmMap(REAL(f), a, b, c, d); }
static int cShmLock(sqlite3_file *f, int a, int b, int c) { return REAL(f)->pMethods->xShmLock(REAL(f), a, b, c); }
static void cShmBarrier(sqlite3_file *f) { REAL(f)->pMethods->xShmBarrier(REAL(f)); }
static int cShmUnmap(sqlite3_file *f, int d) { return REAL(f)->pMethods->xShmUnmap(REAL(f), d); }
static int cFetch(sqlite3_file *f, sqlite3_int64 o, int a, void **pp) { *pp = 0; (void)f; (void)o; (void)a; return SQLITE_OK; }
static int cUnfetch(sqlite3_file *f, sqlite3_int64 o, void *p) { (void)f; (void)o; (void)p; return SQLITE_OK; }

static const sqlite3_io_methods count_io = {
  3, cClose, cRead, cWrite, cTruncate, cSync, cFileSize, cLock, cUnlock, cCheckReserved,
  cFileControl, cSectorSize, cDevChar, cShmMap, cShmLock, cShmBarrier, cShmUnmap, cFetch, cUnfetch
};

static int cOpen(sqlite3_vfs *vfs, const char *name, sqlite3_file *f, int flags, int *outFlags) {
  CountFile *p = (CountFile *)f;
  (void)vfs;
  memset(p, 0, sizeof *p);
  p->real = (sqlite3_file *)&p[1];
  p->is_main = (flags & SQLITE_OPEN_MAIN_DB) != 0;
  int rc = base_vfs->xOpen(base_vfs, name, p->real, flags, outFlags);
  p->base.pMethods = (rc == SQLITE_OK && p->real->pMethods) ? &count_io : 0;
  return rc;
}
static int cDelete(sqlite3_vfs *v, const char *n, int s) { (void)v; return base_vfs->xDelete(base_vfs, n, s); }
static int cAccess(sqlite3_vfs *v, const char *n, int f, int *r) { (void)v; return base_vfs->xAccess(base_vfs, n, f, r); }
static int cFullPath(sqlite3_vfs *v, const char *n, int o, char *out) { (void)v; return base_vfs->xFullPathname(base_vfs, n, o, out); }
static void *cDlOpen(sqlite3_vfs *v, const char *n) { (void)v; return base_vfs->xDlOpen(base_vfs, n); }
static void cDlError(sqlite3_vfs *v, int n, char *m) { (void)v; base_vfs->xDlError(base_vfs, n, m); }
static void (*cDlSym(sqlite3_vfs *v, void *h, const char *s))(void) { (void)v; return base_vfs->xDlSym(base_vfs, h, s); }
static void cDlClose(sqlite3_vfs *v, void *h) { (void)v; base_vfs->xDlClose(base_vfs, h); }
static int cRandomness(sqlite3_vfs *v, int n, char *o) { (void)v; return base_vfs->xRandomness(base_vfs, n, o); }
static int cSleep(sqlite3_vfs *v, int us) { (void)v; return base_vfs->xSleep(base_vfs, us); }
static int cCurrentTime(sqlite3_vfs *v, double *t) { (void)v; return base_vfs->xCurrentTime(base_vfs, t); }
static int cGetLastError(sqlite3_vfs *v, int n, char *m) { (void)v; return base_vfs->xGetLastError ? base_vfs->xGetLastError(base_vfs, n, m) : 0; }
static int cCurrentTimeInt64(sqlite3_vfs *v, sqlite3_int64 *t) { (void)v; return base_vfs->xCurrentTimeInt64(base_vfs, t); }

static void fn_reset(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
  (void)argc; (void)argv;
  g.reads = g.bytes = g.prefetch_calls = g.prefetch_pages = 0;
  if (g.offs) memset(g.offs, 0, sizeof(*g.offs) * g.cap);
  g.n = 0;
  sqlite3_result_null(ctx);
}

static void fn_stats(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
  (void)argc; (void)argv;
  char *s = sqlite3_mprintf("{\"reads\":%lld,\"bytes\":%lld,\"distinct\":%d,\"prefetch_calls\":%lld,\"prefetch_pages\":%lld}",
                            g.reads, g.bytes, g.n, g.prefetch_calls, g.prefetch_pages);
  sqlite3_result_text(ctx, s, -1, sqlite3_free);
}

#ifdef _WIN32
__declspec(dllexport)
#endif
int sqlite3_countvfs_init(sqlite3 *db, char **pzErr, const sqlite3_api_routines *pApi) {
  SQLITE_EXTENSION_INIT2(pApi);
  (void)pzErr;
  if (!base_vfs) {
    base_vfs = sqlite3_vfs_find(0);
    count_vfs.iVersion = 2;
    count_vfs.szOsFile = (int)sizeof(CountFile) + base_vfs->szOsFile;
    count_vfs.mxPathname = base_vfs->mxPathname;
    count_vfs.zName = "countvfs";
    count_vfs.xOpen = cOpen; count_vfs.xDelete = cDelete; count_vfs.xAccess = cAccess;
    count_vfs.xFullPathname = cFullPath; count_vfs.xDlOpen = cDlOpen; count_vfs.xDlError = cDlError;
    count_vfs.xDlSym = cDlSym; count_vfs.xDlClose = cDlClose; count_vfs.xRandomness = cRandomness;
    count_vfs.xSleep = cSleep; count_vfs.xCurrentTime = cCurrentTime; count_vfs.xGetLastError = cGetLastError;
    count_vfs.xCurrentTimeInt64 = cCurrentTimeInt64;
    sqlite3_vfs_register(&count_vfs, 0);
  }
  int rc = sqlite3_create_function(db, "countvfs_reset", 0, SQLITE_UTF8, 0, fn_reset, 0, 0);
  if (rc == SQLITE_OK) rc = sqlite3_create_function(db, "countvfs_stats", 0, SQLITE_UTF8, 0, fn_stats, 0, 0);
  return rc == SQLITE_OK ? SQLITE_OK_LOAD_PERMANENTLY : rc;
}
