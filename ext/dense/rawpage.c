/*
** rawpage.c — see rawpage.h. The on-disk format references are from
** https://www.sqlite.org/fileformat2.html (sections 1.6 "B-tree Pages" and
** 2.1 "Record Format"). All multi-byte integers here are big-endian.
*/
#include "rawpage.h"
SQLITE_EXTENSION_INIT3

#include <stdlib.h>
#include <string.h>

static dense_ann_prefetch_fn g_prefetch_fn = NULL;
static void *g_prefetch_ctx = NULL;

void dense_ann_set_prefetch_hook(dense_ann_prefetch_fn fn, void *ctx) {
  g_prefetch_fn = fn;
  g_prefetch_ctx = ctx;
}

static uint32_t get2(const uint8_t *p) { return ((uint32_t)p[0] << 8) | p[1]; }
static uint32_t get4(const uint8_t *p) {
  return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) | ((uint32_t)p[2] << 8) | p[3];
}

/* SQLite varint: 1-9 bytes, big-endian 7-bit groups, 9th byte has 8 bits.
** Returns the number of bytes consumed. */
static int get_varint(const uint8_t *p, const uint8_t *end, uint64_t *v) {
  uint64_t x = 0;
  for (int i = 0; i < 9; i++) {
    if (p + i >= end) { *v = 0; return 0; }
    if (i == 8) { x = (x << 8) | p[i]; *v = x; return 9; }
    x = (x << 7) | (p[i] & 0x7f);
    if (!(p[i] & 0x80)) { *v = x; return i + 1; }
  }
  *v = x;
  return 9;
}

int pr_open(PageReader *pr, sqlite3 *db, const char *zDb) {
  memset(pr, 0, sizeof *pr);
  pr->db = db; pr->zDb = zDb;
  sqlite3_file *fd = NULL;
  if (sqlite3_file_control(db, zDb, SQLITE_FCNTL_FILE_POINTER, &fd) != SQLITE_OK || !fd || !fd->pMethods)
    return 1;
  uint8_t hdr[100];
  if (fd->pMethods->xRead(fd, hdr, 100, 0) != SQLITE_OK) return 1;
  if (memcmp(hdr, "SQLite format 3", 16) != 0) return 1;
  uint32_t ps = get2(hdr + 16);
  pr->pgsz = ps == 1 ? 65536 : (int)ps;
  pr->usable = pr->pgsz - hdr[20];
  pr->wal = (hdr[18] == 2 || hdr[19] == 2);
  pr->fd = fd;
  pr->ok = pr->pgsz >= 512 && !pr->wal;
  return pr->ok ? 0 : 1;
}

int pr_read(PageReader *pr, uint32_t pgno, uint8_t *buf) {
  if (!pr->ok || pgno == 0) return SQLITE_ERROR;
  return pr->fd->pMethods->xRead(pr->fd, buf, pr->pgsz, (sqlite3_int64)(pgno - 1) * pr->pgsz);
}

static int cmp_i64(const void *a, const void *b) {
  sqlite3_int64 x = *(const sqlite3_int64 *)a, y = *(const sqlite3_int64 *)b;
  return x < y ? -1 : x > y;
}

void pr_prefetch(PageReader *pr, const uint32_t *pgnos, int n) {
  if (!pr->ok || n <= 0) return;
  sqlite3_int64 *off = (sqlite3_int64 *)malloc(sizeof(sqlite3_int64) * n);
  if (!off) return;
  int m = 0;
  for (int i = 0; i < n; i++) if (pgnos[i]) off[m++] = (sqlite3_int64)(pgnos[i] - 1) * pr->pgsz;
  qsort(off, m, sizeof *off, cmp_i64);
  int u = 0;
  for (int i = 0; i < m; i++) if (u == 0 || off[i] != off[u - 1]) off[u++] = off[i];
  dense_ann_prefetch_req req = { u, pr->pgsz, off };
  if (u > 0) {
    if (g_prefetch_fn) g_prefetch_fn(g_prefetch_ctx, pr->db, pr->zDb, &req);
    else sqlite3_file_control(pr->db, pr->zDb, DENSE_ANN_FCNTL_PREFETCH, &req);
  }
  free(off);
}

/* Parse leaf cell i: returns rowid, payload pointer and length, and whether
** the payload is entirely local. Returns 0 if malformed. */
static int leaf_cell(const PageReader *pr, const uint8_t *page, int hdroff, int i,
                     int64_t *rowid, const uint8_t **payload, int *plen, int *local) {
  const uint8_t *end = page + pr->usable;
  uint32_t cp = get2(page + hdroff + 8 + 2 * i);
  if (cp >= (uint32_t)pr->usable) return 0;
  const uint8_t *c = page + cp;
  uint64_t P, R;
  int a = get_varint(c, end, &P); if (!a) return 0;
  int b = get_varint(c + a, end, &R); if (!b) return 0;
  *rowid = (int64_t)R;
  *payload = c + a + b;
  *plen = (int)P;
  *local = (P <= (uint64_t)(pr->usable - 35)) && (c + a + b + P <= end);
  return 1;
}

int pr_leaf_find(const PageReader *pr, const uint8_t *page, uint32_t pgno, int64_t rowid,
                 const uint8_t **payload, int *plen) {
  int hdroff = (pgno == 1) ? 100 : 0;
  if (page[hdroff] != 0x0d) return 0;          /* not a table leaf */
  int n = (int)get2(page + hdroff + 3);
  if (8 + 2 * n > pr->usable) return 0;
  int lo = 0, hi = n - 1;
  while (lo <= hi) {                            /* cells are sorted by rowid */
    int mid = (lo + hi) / 2, local;
    int64_t r; const uint8_t *pl; int len;
    if (!leaf_cell(pr, page, hdroff, mid, &r, &pl, &len, &local)) return 0;
    if (r == rowid) {
      if (!local) return 0;
      *payload = pl; *plen = len;
      return 1;
    }
    if (r < rowid) lo = mid + 1; else hi = mid - 1;
  }
  return 0;
}

static int serial_size(uint64_t t) {
  static const int fixed[] = { 0, 1, 2, 3, 4, 6, 8, 8, 0, 0, 0, 0 };
  if (t < 12) return fixed[t];
  return (int)((t - 12) / 2);
}

int pr_record_blob(const uint8_t *rec, int len, int col, const uint8_t **p, int *n) {
  const uint8_t *end = rec + len;
  uint64_t hsz;
  int k = get_varint(rec, end, &hsz);
  if (!k || hsz > (uint64_t)len) return 0;
  const uint8_t *h = rec + k, *hend = rec + hsz;
  uint64_t off = hsz;
  for (int c = 0; h < hend; c++) {
    uint64_t t;
    int a = get_varint(h, hend, &t); if (!a) return 0;
    h += a;
    int sz = serial_size(t);
    if (c == col) {
      if (t < 12 || off + sz > (uint64_t)len) return 0;
      *p = rec + off; *n = sz;
      return 1;
    }
    off += sz;
  }
  return 0;
}

static int walk(PageReader *pr, uint32_t pgno, pr_walk_cb cb, void *ctx, int depth) {
  if (depth > 20) return SQLITE_CORRUPT;
  uint8_t *page = (uint8_t *)malloc(pr->pgsz);
  if (!page) return SQLITE_NOMEM;
  int rc = pr_read(pr, pgno, page);
  int hdroff = (pgno == 1) ? 100 : 0;
  if (rc == SQLITE_OK) {
    int n = (int)get2(page + hdroff + 3);
    if (page[hdroff] == 0x0d) {
      for (int i = 0; i < n && rc == SQLITE_OK; i++) {
        int64_t r; const uint8_t *pl; int len, local;
        if (!leaf_cell(pr, page, hdroff, i, &r, &pl, &len, &local)) { rc = SQLITE_CORRUPT; break; }
        cb(ctx, r, pgno, !local);
      }
    } else if (page[hdroff] == 0x05) {
      for (int i = 0; i < n && rc == SQLITE_OK; i++) {
        uint32_t cp = get2(page + hdroff + 12 + 2 * i);
        rc = walk(pr, get4(page + cp), cb, ctx, depth + 1);
      }
      if (rc == SQLITE_OK) rc = walk(pr, get4(page + hdroff + 8), cb, ctx, depth + 1);
    } else {
      rc = SQLITE_CORRUPT;
    }
  }
  free(page);
  return rc;
}

int pr_walk_table(PageReader *pr, uint32_t root, pr_walk_cb cb, void *ctx) {
  if (!pr->ok) return SQLITE_ERROR;
  return walk(pr, root, cb, ctx, 0);
}
