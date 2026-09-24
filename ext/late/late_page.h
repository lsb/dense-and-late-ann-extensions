/*
** late_page.h -- batched, round-counting reads of SQLite table b-trees.
**
** On an HTTP-range VFS every page that is not cached costs a network round
** trip, and an ordinary rowid lookup walks root -> interior -> leaf one page
** at a time. The functions here read b-tree pages directly (xRead on the
** database file) and fetch a whole *level* of pages at once: before reading
** the pages of one level they are announced to the VFS with
** httpvfs_prefetch_pages() (wasm/src/httpvfs.h), which fetches them in one
** parallel round. A lookup of n rows therefore costs one round per b-tree
** level instead of n per level. Interior pages are cached in the reader, so
** after warm-up a lookup costs a single round (the leaves). If a leaf page
** number is already known ("hint"), even that walk is skipped.
**
** The reader also records every page it had to read, grouped by round, so
** that a benchmark can replay the exact access pattern in a network
** simulator (netsim/).
**
** Format reference: https://www.sqlite.org/fileformat2.html sections 1.6
** (b-tree pages) and 2.1 (record format).
*/
#ifndef LATE_PAGE_H
#define LATE_PAGE_H

#include <stdint.h>
#include "sqlite3ext.h"

typedef struct LpRow {
  int64_t rowid;       /* in */
  uint32_t hint;       /* in: leaf page believed to hold the row, 0 = unknown */
  uint8_t *data;       /* out: malloc'd copy of the first column (a BLOB), or NULL */
  int len;             /* out */
} LpRow;

typedef struct LpReader {
  sqlite3 *db;
  char zDb[64];
  sqlite3_file *fd;
  int pgsz, usable;
  int ok;              /* direct reads possible (rollback journal, readable header) */
  /* interior-page cache: open addressing on page number */
  uint32_t *ckey; uint8_t **cval; int ccap, ccount, cmax;
  /* trace of the current query */
  uint32_t *tpg; int *trd; int tn, tcap;
  int round;           /* rounds so far */
  int64_t cache_hits;  /* interior pages served from the reader's cache */
  int64_t hint_miss;   /* hinted rows not found on their hinted page */
  int64_t fallbacks;   /* rows read through SQL (overflow pages, bad hints) */
  int64_t payload_bytes; /* bytes of row payload delivered */
} LpReader;

/* Returns 0 if direct reads are possible. max_cache_pages bounds the
** interior cache (0 = default 16384). */
int lpg_open(LpReader *r, sqlite3 *db, const char *zDb, int max_cache_pages);
void lpg_close(LpReader *r);
void lpg_drop_cache(LpReader *r);
void lpg_trace_reset(LpReader *r);

/* Fetch rows (sorted by rowid, distinct) of the rowid table zTable (root
** page `root`), whose rows are (id INTEGER PRIMARY KEY, data BLOB). The
** payload of column `data` is copied into rows[i].data. Rows that cannot be
** read directly are read with SQL. Returns SQLITE_OK or an error. */
int lpg_multiget(LpReader *r, const char *zTable, uint32_t root, LpRow *rows, int n);

/* Visit every row of the table in rowid order, reading the tree one level
** per round. cb gets the leaf page number and the data column. Returns
** SQLITE_OK, or SQLITE_NOTFOUND if a row spills to overflow pages. */
typedef int (*lpg_scan_cb)(void *ctx, int64_t rowid, uint32_t pgno, const uint8_t *data, int len);
int lpg_scan(LpReader *r, uint32_t root, lpg_scan_cb cb, void *ctx);

/* Root page of a table, via sqlite_schema. 0 if absent. */
uint32_t lpg_root(sqlite3 *db, const char *zDb, const char *zTable);

#endif
