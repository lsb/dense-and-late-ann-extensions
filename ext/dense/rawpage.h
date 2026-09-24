/*
** rawpage.h — direct reads of SQLite b-tree pages, bypassing the b-tree
** layer, plus the prefetch hook for HTTP VFSes.
**
** Why: on an HTTP-range VFS every page SQLite reads is a network request, and
** a normal lookup by rowid first walks the interior pages of the table's
** b-tree (root -> interior -> leaf), which are sequential dependent requests.
** The index therefore stores, next to every node reference, the page number
** of the leaf page holding that node's row ("page hint"). At query time we
** read that leaf page directly with xRead, parse the cell ourselves, and can
** ask the VFS to fetch all the pages of a search step in parallel first.
** A hint is always validated (page type, rowid present, no overflow); if it
** is stale we fall back to an ordinary SQL lookup.
*/
#ifndef DENSE_RAWPAGE_H
#define DENSE_RAWPAGE_H

#include <stdint.h>
#include "sqlite3ext.h"

/* ---------------------------------------------------------------------
** Prefetch interface (for the HTTP VFS).
**
** Before each search step the extension announces the pages it is about to
** read, all of which are independent. A VFS that can fetch in parallel should
** start all of them and return when they are cached; the subsequent xRead
** calls are then served from cache. Three ways to hook in, tried in order:
**
**  1. A function pointer registered with dense_ann_set_prefetch_hook().
**  2. The project's httpvfs file control HTTPVFS_FCNTL_PREFETCH_PAGES
**     (wasm/src/httpvfs.h; the struct is mirrored here so this extension
**     has no build dependency on the VFS).
**  3. If that returns SQLITE_NOTFOUND, DENSE_ANN_FCNTL_PREFETCH with byte
**     offsets (used by the test VFS in test/countvfs.c).
** Unknown opcodes return SQLITE_NOTFOUND, so natively this is a no-op.
** ------------------------------------------------------------------- */
#define HTTPVFS_FCNTL_PREFETCH_PAGES_DN 0x48560002   /* = HTTPVFS_FCNTL_PREFETCH_PAGES */
typedef struct dense_httpvfs_pages {                 /* = HttpvfsPages */
  const unsigned int *pgnos;
  int n;
  int page_size;
} dense_httpvfs_pages;

#define DENSE_ANN_FCNTL_PREFETCH 0x44414e01   /* "DAN\1" */

typedef struct dense_ann_prefetch_req {
  int n;                          /* number of ranges */
  int len;                        /* bytes per range (= page size) */
  const sqlite3_int64 *offsets;   /* file offsets, ascending, distinct */
} dense_ann_prefetch_req;

typedef void (*dense_ann_prefetch_fn)(void *ctx, sqlite3 *db, const char *zDb,
                                      const dense_ann_prefetch_req *req);
void dense_ann_set_prefetch_hook(dense_ann_prefetch_fn fn, void *ctx);

/* --------------------------------------------------------------------- */

typedef struct PageReader {
  sqlite3 *db;
  const char *zDb;
  sqlite3_file *fd;
  int pgsz;       /* page size */
  int usable;     /* page size minus reserved bytes */
  int ok;         /* direct reads possible */
  int wal;        /* database header says WAL mode: the file may be stale */
} PageReader;

/* Set up direct reads of database zDb. Returns 0 if possible. */
int dnpr_open(PageReader *pr, sqlite3 *db, const char *zDb);

/* Read page pgno (1-based) into buf[pgsz]. Returns SQLITE_OK on success. */
int dnpr_read(PageReader *pr, uint32_t pgno, uint8_t *buf);

/* Announce the pages about to be read (see above). pgnos need not be sorted. */
void dnpr_prefetch(PageReader *pr, const uint32_t *pgnos, int n);

/* In table-leaf page `page` (number pgno), find the cell with the given rowid.
** On success sets *payload and *plen to the record and returns 1. Returns 0 if
** the page is not a table leaf, the rowid is absent, or the payload spills
** to overflow pages. */
int dnpr_leaf_find(const PageReader *pr, const uint8_t *page, uint32_t pgno, int64_t rowid,
                 const uint8_t **payload, int *plen);

/* Column `col` of a record, which must be a BLOB or TEXT. Returns 1 on success. */
int dnpr_record_blob(const uint8_t *rec, int len, int col, const uint8_t **p, int *n);

/* Walk every leaf cell of the table b-tree rooted at `root`, calling
** cb(ctx, rowid, leaf_pgno, overflows). Returns SQLITE_OK on success. */
typedef void (*dnpr_walk_cb)(void *ctx, int64_t rowid, uint32_t pgno, int overflow);
int dnpr_walk_table(PageReader *pr, uint32_t root, dnpr_walk_cb cb, void *ctx);

#endif
