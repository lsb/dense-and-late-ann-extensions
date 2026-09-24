/*
** late_page.c -- see late_page.h.
*/
#include "sqlite3ext.h"
SQLITE_EXTENSION_INIT3
#include "httpvfs.h"
#include "late_page.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t g2(const uint8_t *p) { return ((uint32_t)p[0] << 8) | p[1]; }
static uint32_t g4(const uint8_t *p) {
  return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) | ((uint32_t)p[2] << 8) | p[3];
}
static int gvarint(const uint8_t *p, const uint8_t *end, uint64_t *v) {
  uint64_t x = 0;
  for (int i = 0; i < 9; i++) {
    if (p + i >= end) { *v = 0; return 0; }
    if (i == 8) { *v = (x << 8) | p[i]; return 9; }
    x = (x << 7) | (p[i] & 0x7f);
    if (!(p[i] & 0x80)) { *v = x; return i + 1; }
  }
  *v = x; return 9;
}

uint32_t lpg_root(sqlite3 *db, const char *zDb, const char *zTable) {
  char *sql = sqlite3_mprintf("SELECT rootpage FROM \"%w\".sqlite_schema WHERE type='table' AND name=%Q",
                              zDb, zTable);
  sqlite3_stmt *st = NULL;
  uint32_t root = 0;
  if (sql && sqlite3_prepare_v2(db, sql, -1, &st, NULL) == SQLITE_OK && sqlite3_step(st) == SQLITE_ROW)
    root = (uint32_t)sqlite3_column_int64(st, 0);
  sqlite3_finalize(st);
  sqlite3_free(sql);
  return root;
}

int lpg_open(LpReader *r, sqlite3 *db, const char *zDb, int max_cache_pages) {
  memset(r, 0, sizeof *r);
  r->db = db;
  snprintf(r->zDb, sizeof r->zDb, "%s", zDb);
  r->cmax = max_cache_pages > 0 ? max_cache_pages : 16384;
  sqlite3_file *fd = NULL;
  if (sqlite3_file_control(db, zDb, SQLITE_FCNTL_FILE_POINTER, &fd) != SQLITE_OK || !fd || !fd->pMethods)
    return 1;
  uint8_t hdr[100];
  if (fd->pMethods->xRead(fd, hdr, 100, 0) != SQLITE_OK) return 1;
  if (memcmp(hdr, "SQLite format 3", 16) != 0) return 1;
  uint32_t ps = g2(hdr + 16);
  r->pgsz = ps == 1 ? 65536 : (int)ps;
  r->usable = r->pgsz - hdr[20];
  r->fd = fd;
  r->ok = r->pgsz >= 512 && hdr[18] != 2 && hdr[19] != 2;   /* not WAL */
  return r->ok ? 0 : 1;
}

void lpg_drop_cache(LpReader *r) {
  for (int i = 0; i < r->ccap; i++) free(r->cval ? r->cval[i] : NULL);
  free(r->ckey); free(r->cval);
  r->ckey = NULL; r->cval = NULL; r->ccap = r->ccount = 0;
}

void lpg_close(LpReader *r) {
  lpg_drop_cache(r);
  free(r->tpg); free(r->trd);
  r->tpg = NULL; r->trd = NULL; r->tn = r->tcap = 0;
}

void lpg_trace_reset(LpReader *r) {
  r->tn = 0; r->round = 0; r->cache_hits = 0; r->hint_miss = 0; r->fallbacks = 0; r->payload_bytes = 0;
}

static uint8_t *cache_get(LpReader *r, uint32_t pg) {
  if (!r->ccap) return NULL;
  for (uint32_t h = (pg * 2654435761u) & (r->ccap - 1);; h = (h + 1) & (r->ccap - 1)) {
    if (r->ckey[h] == 0) return NULL;
    if (r->ckey[h] == pg) return r->cval[h];
  }
}

static void cache_put(LpReader *r, uint32_t pg, const uint8_t *page) {
  if (r->ccount >= r->cmax || cache_get(r, pg)) return;
  if ((r->ccount + 1) * 2 > r->ccap) {
    int ncap = r->ccap ? r->ccap * 2 : 256;
    uint32_t *nk = (uint32_t *)calloc(ncap, sizeof(uint32_t));
    uint8_t **nv = (uint8_t **)calloc(ncap, sizeof(uint8_t *));
    if (!nk || !nv) { free(nk); free(nv); return; }
    for (int i = 0; i < r->ccap; i++) if (r->ckey[i]) {
      uint32_t h = (r->ckey[i] * 2654435761u) & (ncap - 1);
      while (nk[h]) h = (h + 1) & (ncap - 1);
      nk[h] = r->ckey[i]; nv[h] = r->cval[i];
    }
    free(r->ckey); free(r->cval);
    r->ckey = nk; r->cval = nv; r->ccap = ncap;
  }
  uint8_t *copy = (uint8_t *)malloc(r->pgsz);
  if (!copy) return;
  memcpy(copy, page, r->pgsz);
  uint32_t h = (pg * 2654435761u) & (r->ccap - 1);
  while (r->ckey[h]) h = (h + 1) & (r->ccap - 1);
  r->ckey[h] = pg; r->cval[h] = copy; r->ccount++;
}

static void trace_add(LpReader *r, uint32_t pg) {
  if (r->tn == r->tcap) {
    int nc = r->tcap ? r->tcap * 2 : 256;
    uint32_t *a = (uint32_t *)realloc(r->tpg, nc * sizeof(uint32_t));
    if (!a) return;
    r->tpg = a;
    int *b = (int *)realloc(r->trd, nc * sizeof(int));
    if (!b) return;
    r->trd = b; r->tcap = nc;
  }
  r->tpg[r->tn] = pg; r->trd[r->tn] = r->round; r->tn++;
}

static int cmp_u32(const void *a, const void *b) {
  uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b;
  return x < y ? -1 : x > y;
}

/* Reads a set of pages as one round: pages found in the interior cache are
** returned from it; the rest are announced to the VFS together, then read.
** pgs[] is sorted and de-duplicated in place; bufs[i] receives either a
** cache pointer (owned[i]=0) or a malloc'd page (owned[i]=1). */
static int read_round(LpReader *r, uint32_t *pgs, int *pn, uint8_t **bufs, char *owned) {
  int n = *pn;
  qsort(pgs, n, sizeof(uint32_t), cmp_u32);
  int u = 0;
  for (int i = 0; i < n; i++) if (u == 0 || pgs[i] != pgs[u - 1]) pgs[u++] = pgs[i];
  *pn = n = u;
  uint32_t *need = (uint32_t *)malloc(sizeof(uint32_t) * (n ? n : 1));
  if (!need) return SQLITE_NOMEM;
  int nn = 0;
  for (int i = 0; i < n; i++) {
    uint8_t *c = cache_get(r, pgs[i]);
    if (c) { bufs[i] = c; owned[i] = 0; r->cache_hits++; }
    else { bufs[i] = NULL; need[nn++] = pgs[i]; }
  }
  int rc = SQLITE_OK;
  if (nn) {
    r->round++;
    httpvfs_prefetch_pages(r->db, need, nn);        /* no-op off httpvfs */
    for (int i = 0; i < n && rc == SQLITE_OK; i++) {
      if (bufs[i]) continue;
      bufs[i] = (uint8_t *)malloc(r->pgsz); owned[i] = 1;
      if (!bufs[i]) { rc = SQLITE_NOMEM; break; }
      rc = r->fd->pMethods->xRead(r->fd, bufs[i], r->pgsz, (sqlite3_int64)(pgs[i] - 1) * r->pgsz);
      trace_add(r, pgs[i]);
    }
  }
  free(need);
  return rc;
}

/* Leaf cell i: rowid, payload and whether it is entirely local. */
static int leaf_cell(const LpReader *r, const uint8_t *page, int ho, int i,
                     int64_t *rowid, const uint8_t **pl, int *plen, int *local) {
  const uint8_t *end = page + r->usable;
  uint32_t cp = g2(page + ho + 8 + 2 * i);
  if (cp >= (uint32_t)r->usable) return 0;
  const uint8_t *c = page + cp;
  uint64_t P, R;
  int a = gvarint(c, end, &P); if (!a) return 0;
  int b = gvarint(c + a, end, &R); if (!b) return 0;
  *rowid = (int64_t)R; *pl = c + a + b; *plen = (int)P;
  *local = P <= (uint64_t)(r->usable - 35) && c + a + b + P <= end;
  return 1;
}

static int serial_size(uint64_t t) {
  static const int fixed[] = { 0, 1, 2, 3, 4, 6, 8, 8, 0, 0, 0, 0 };
  return t < 12 ? fixed[t] : (int)((t - 12) / 2);
}

/* The first BLOB column of a record. */
static int record_blob(const uint8_t *rec, int len, const uint8_t **p, int *n) {
  const uint8_t *end = rec + len;
  uint64_t hsz;
  int k = gvarint(rec, end, &hsz);
  if (!k || hsz > (uint64_t)len) return 0;
  const uint8_t *h = rec + k, *hend = rec + hsz;
  uint64_t off = hsz;
  while (h < hend) {
    uint64_t t;
    int a = gvarint(h, hend, &t); if (!a) return 0;
    h += a;
    int sz = serial_size(t);
    if (t >= 12 && !(t & 1)) {
      if (off + sz > (uint64_t)len) return 0;
      *p = rec + off; *n = sz; return 1;
    }
    off += sz;
  }
  return 0;
}

/* Find rowid in a table leaf. 1 = found local, -1 = found but overflows,
** 0 = absent or not a table leaf. */
static int leaf_find(LpReader *r, const uint8_t *page, uint32_t pg, int64_t rowid,
                     const uint8_t **data, int *len) {
  int ho = pg == 1 ? 100 : 0;
  if (page[ho] != 0x0d) return 0;
  int n = (int)g2(page + ho + 3), lo = 0, hi = n - 1;
  while (lo <= hi) {
    int mid = (lo + hi) / 2, local, plen;
    int64_t rr; const uint8_t *pl;
    if (!leaf_cell(r, page, ho, mid, &rr, &pl, &plen, &local)) return 0;
    if (rr == rowid) {
      if (!local) return -1;
      return record_blob(pl, plen, data, len) ? 1 : -1;
    }
    if (rr < rowid) lo = mid + 1; else hi = mid - 1;
  }
  return 0;
}

static int copy_out(LpReader *r, LpRow *row, const uint8_t *d, int n) {
  row->data = (uint8_t *)malloc(n + 8);     /* +8: bit readers may overrun */
  if (!row->data) return SQLITE_NOMEM;
  memcpy(row->data, d, n);
  memset(row->data + n, 0, 8);
  row->len = n;
  r->payload_bytes += n;
  return SQLITE_OK;
}

static int sql_fetch(LpReader *r, const char *zTable, LpRow *row) {
  char *sql = sqlite3_mprintf("SELECT data FROM \"%w\".\"%w\" WHERE id=?", r->zDb, zTable);
  sqlite3_stmt *st = NULL;
  int rc = sql ? sqlite3_prepare_v2(r->db, sql, -1, &st, NULL) : SQLITE_NOMEM;
  sqlite3_free(sql);
  if (rc != SQLITE_OK) return rc;
  sqlite3_bind_int64(st, 1, row->rowid);
  if (sqlite3_step(st) == SQLITE_ROW) {
    rc = copy_out(r, row, (const uint8_t *)sqlite3_column_blob(st, 0), sqlite3_column_bytes(st, 0));
    r->fallbacks++;
  }
  sqlite3_finalize(st);
  return rc;
}

typedef struct Node { uint32_t pg; int lo, hi; } Node;   /* rows [lo, hi) */

int lpg_multiget(LpReader *r, const char *zTable, uint32_t root, LpRow *rows, int n) {
  int rc = SQLITE_OK;
  for (int i = 0; i < n; i++) { rows[i].data = NULL; rows[i].len = 0; }
  if (n == 0) return SQLITE_OK;
  char *todo = (char *)calloc(n, 1);       /* 1: resolve by walk, 2: by SQL */
  Node *cur = (Node *)malloc(sizeof(Node) * n), *nxt = (Node *)malloc(sizeof(Node) * n);
  uint32_t *pgs = (uint32_t *)malloc(sizeof(uint32_t) * n);
  uint8_t **bufs = (uint8_t **)malloc(sizeof(uint8_t *) * n);
  char *owned = (char *)malloc(n);
  int *map = (int *)malloc(sizeof(int) * n);
  if (!todo || !cur || !nxt || !pgs || !bufs || !owned || !map) { rc = SQLITE_NOMEM; goto done; }
  if (!r->ok) { for (int i = 0; i < n; i++) todo[i] = 2; goto sql; }

  /* 1. Hinted rows: read their leaves in one round. */
  int np = 0;
  for (int i = 0; i < n; i++) if (rows[i].hint) pgs[np++] = rows[i].hint; else todo[i] = 1;
  if (np) {
    rc = read_round(r, pgs, &np, bufs, owned);
    for (int i = 0; i < n && rc == SQLITE_OK; i++) {
      if (!rows[i].hint) continue;
      uint32_t *hit = (uint32_t *)bsearch(&rows[i].hint, pgs, np, sizeof(uint32_t), cmp_u32);
      const uint8_t *d; int len;
      int f = hit ? leaf_find(r, bufs[hit - pgs], rows[i].hint, rows[i].rowid, &d, &len) : 0;
      if (f == 1) rc = copy_out(r, &rows[i], d, len);
      else { todo[i] = f < 0 ? 2 : 1; if (f == 0) r->hint_miss++; }
    }
    for (int i = 0; i < np; i++) if (owned[i]) free(bufs[i]);
    if (rc) goto done;
  }

  /* 2. Level-by-level walk for the rest. idx[] lists rows still to find. */
  int m = 0;
  for (int i = 0; i < n; i++) if (todo[i] == 1) map[m++] = i;
  int ncur = 0;
  if (m) { cur[0].pg = root; cur[0].lo = 0; cur[0].hi = m; ncur = 1; }
  for (int depth = 0; ncur > 0 && depth < 24; depth++) {
    np = ncur;
    for (int i = 0; i < ncur; i++) pgs[i] = cur[i].pg;
    rc = read_round(r, pgs, &np, bufs, owned);
    if (rc) break;
    int nnxt = 0;
    for (int ci = 0; ci < ncur && rc == SQLITE_OK; ci++) {
      Node nd = cur[ci];
      int pi = (int)((uint32_t *)bsearch(&nd.pg, pgs, np, sizeof(uint32_t), cmp_u32) - pgs);
      const uint8_t *page = bufs[pi];
      int ho = nd.pg == 1 ? 100 : 0;
      if (page[ho] == 0x0d) {
        for (int k = nd.lo; k < nd.hi && rc == SQLITE_OK; k++) {
          LpRow *row = &rows[map[k]];
          const uint8_t *d; int len;
          int f = leaf_find(r, page, nd.pg, row->rowid, &d, &len);
          if (f == 1) rc = copy_out(r, row, d, len);
          else if (f < 0) todo[map[k]] = 2;
          else todo[map[k]] = 0;           /* absent */
        }
      } else if (page[ho] == 0x05) {
        if (owned[pi]) cache_put(r, nd.pg, page);
        int nc = (int)g2(page + ho + 3), k = nd.lo;
        for (int c = 0; c <= nc && k < nd.hi; c++) {
          uint32_t child; int64_t key = INT64_MAX;
          if (c < nc) {
            const uint8_t *cell = page + g2(page + ho + 12 + 2 * c);
            uint64_t kk;
            child = g4(cell);
            gvarint(cell + 4, page + r->usable, &kk);
            key = (int64_t)kk;
          } else child = g4(page + ho + 8);
          int s = k;
          while (k < nd.hi && rows[map[k]].rowid <= key) k++;
          if (k > s) { nxt[nnxt].pg = child; nxt[nnxt].lo = s; nxt[nnxt].hi = k; nnxt++; }
        }
      } else {
        for (int k = nd.lo; k < nd.hi; k++) todo[map[k]] = 2;   /* unexpected: use SQL */
      }
    }
    for (int i = 0; i < np; i++) if (owned[i]) free(bufs[i]);
    Node *t = cur; cur = nxt; nxt = t; ncur = nnxt;
  }
  if (rc) goto done;

sql:
  for (int i = 0; i < n && rc == SQLITE_OK; i++) if (todo[i] == 2 && !rows[i].data) rc = sql_fetch(r, zTable, &rows[i]);
done:
  free(todo); free(cur); free(nxt); free(pgs); free(bufs); free(owned); free(map);
  return rc;
}

int lpg_scan(LpReader *r, uint32_t root, lpg_scan_cb cb, void *ctx) {
  if (!r->ok) return SQLITE_ERROR;
  int cap = 64, ncur = 1, rc = SQLITE_OK;
  uint32_t *cur = (uint32_t *)malloc(sizeof(uint32_t) * cap);
  if (!cur) return SQLITE_NOMEM;
  cur[0] = root;
  for (int depth = 0; ncur > 0 && depth < 24 && rc == SQLITE_OK; depth++) {
    /* Keep the level's order (it is the key order); read_round sorts a copy. */
    uint32_t *pgs = (uint32_t *)malloc(sizeof(uint32_t) * ncur);
    uint8_t **bufs = (uint8_t **)malloc(sizeof(uint8_t *) * ncur);
    char *owned = (char *)malloc(ncur);
    int ncap = 64, nn = 0;
    uint32_t *nxt = (uint32_t *)malloc(sizeof(uint32_t) * ncap);
    if (!pgs || !bufs || !owned || !nxt) { free(pgs); free(bufs); free(owned); free(nxt); rc = SQLITE_NOMEM; break; }
    memcpy(pgs, cur, sizeof(uint32_t) * ncur);
    int np = ncur;
    rc = read_round(r, pgs, &np, bufs, owned);
    for (int i = 0; i < ncur && rc == SQLITE_OK; i++) {
      int pi = (int)((uint32_t *)bsearch(&cur[i], pgs, np, sizeof(uint32_t), cmp_u32) - pgs);
      const uint8_t *page = bufs[pi];
      int ho = cur[i] == 1 ? 100 : 0, nc = (int)g2(page + ho + 3);
      if (page[ho] == 0x0d) {
        for (int c = 0; c < nc && rc == SQLITE_OK; c++) {
          int64_t rowid; const uint8_t *pl, *d; int plen, local, len;
          if (!leaf_cell(r, page, ho, c, &rowid, &pl, &plen, &local)) { rc = SQLITE_CORRUPT; break; }
          if (!local || !record_blob(pl, plen, &d, &len)) { rc = SQLITE_NOTFOUND; break; }
          r->payload_bytes += len;
          rc = cb(ctx, rowid, cur[i], d, len);
        }
      } else if (page[ho] == 0x05) {
        if (owned[pi]) cache_put(r, cur[i], page);
        for (int c = 0; c <= nc; c++) {
          if (nn == ncap) { ncap *= 2; uint32_t *t = (uint32_t *)realloc(nxt, sizeof(uint32_t) * ncap); if (!t) { rc = SQLITE_NOMEM; break; } nxt = t; }
          nxt[nn++] = c < nc ? g4(page + g2(page + ho + 12 + 2 * c)) : g4(page + ho + 8);
        }
      } else rc = SQLITE_CORRUPT;
    }
    for (int i = 0; i < np; i++) if (owned[i]) free(bufs[i]);
    free(pgs); free(bufs); free(owned);
    free(cur); cur = nxt; ncur = nn;
  }
  free(cur);
  return rc;
}
