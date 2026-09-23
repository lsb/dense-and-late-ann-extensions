/*
** httpvfs.h -- interface that SQLite extensions use to batch network reads.
**
** The "httpvfs" VFS serves a read-only database from a remote file (HTTP range
** requests in WebAssembly, a local file with simulated latency natively). Each
** time SQLite reads a block that is not in the VFS page cache, execution blocks
** for one network round trip. Extensions whose reads are data-dependent (graph
** neighbours, IVF lists) can cut the number of round trips by telling the VFS
** in advance what they will read, so that many ranges are fetched in parallel
** in a single round.
**
** Everything here goes through sqlite3_file_control() with private opcodes, so
** an extension has no link-time dependency on the VFS: it works as a loadable
** extension, statically linked into the WASM build, or against an ordinary
** database file (where every call is a harmless no-op returning
** SQLITE_NOTFOUND).
**
** Include this after sqlite3.h or sqlite3ext.h.
*/
#ifndef HTTPVFS_H
#define HTTPVFS_H

#define HTTPVFS_FCNTL_PREFETCH        0x48560001  /* HttpvfsRanges* */
#define HTTPVFS_FCNTL_PREFETCH_PAGES  0x48560002  /* HttpvfsPages* */
#define HTTPVFS_FCNTL_STATS           0x48560003  /* HttpvfsStats* (out) */
#define HTTPVFS_FCNTL_RESET_STATS     0x48560004  /* NULL */
#define HTTPVFS_FCNTL_SPECULATE       0x48560005  /* int* in: 1 begin, 0 end+fetch, -1 end+discard; out: #blocks */
#define HTTPVFS_FCNTL_PAGE_SIZE       0x48560006  /* int* (out): VFS block size */

typedef struct HttpvfsRanges {
  const sqlite3_int64 *offsets;   /* byte offsets in the database file */
  const int *lengths;             /* byte lengths */
  int n;
} HttpvfsRanges;

typedef struct HttpvfsPages {
  const unsigned int *pgnos;      /* SQLite page numbers, 1-based */
  int n;
  int page_size;                  /* database page size (0: ask SQLite) */
} HttpvfsPages;

typedef struct HttpvfsStats {
  sqlite3_int64 requests;         /* HTTP (or simulated) range requests */
  sqlite3_int64 bytes;            /* bytes fetched */
  sqlite3_int64 rounds;           /* times execution blocked on the network */
  sqlite3_int64 reads;            /* xRead calls from SQLite */
  sqlite3_int64 cache_hits;       /* blocks served from the VFS cache */
  sqlite3_int64 cache_misses;     /* blocks that had to be fetched on demand */
  sqlite3_int64 prefetch_calls;   /* prefetch requests from extensions */
  sqlite3_int64 prefetch_blocks;  /* blocks fetched by prefetch */
  sqlite3_int64 spec_misses;      /* blocks recorded during speculation */
  sqlite3_int64 file_size;
  int block_size;                 /* VFS cache block size in bytes */
  int cache_blocks;               /* capacity of the VFS cache, in blocks */
  int cached_blocks;              /* blocks currently cached */
  double net_ms;                  /* wall time spent blocked on the network */
} HttpvfsStats;

/*
** Fetch the given byte ranges of the "main" database into the VFS cache,
** in parallel, as one round. Ranges already cached cost nothing. Returns
** SQLITE_OK, SQLITE_NOTFOUND if the database is not on httpvfs, or an error.
*/
static inline int httpvfs_prefetch(sqlite3 *db, const sqlite3_int64 *offsets,
                                   const int *lengths, int n){
  HttpvfsRanges r;
  r.offsets = offsets; r.lengths = lengths; r.n = n;
  return sqlite3_file_control(db, "main", HTTPVFS_FCNTL_PREFETCH, &r);
}

/* Same, by SQLite page number (page_size 0 means "use the database's"). */
static inline int httpvfs_prefetch_pages(sqlite3 *db, const unsigned int *pgnos,
                                         int n){
  HttpvfsPages p;
  p.pgnos = pgnos; p.n = n; p.page_size = 0;
  return sqlite3_file_control(db, "main", HTTPVFS_FCNTL_PREFETCH_PAGES, &p);
}

static inline int httpvfs_stats(sqlite3 *db, HttpvfsStats *out){
  return sqlite3_file_control(db, "main", HTTPVFS_FCNTL_STATS, out);
}

static inline int httpvfs_reset_stats(sqlite3 *db){
  return sqlite3_file_control(db, "main", HTTPVFS_FCNTL_RESET_STATS, 0);
}

/*
** Speculative batching, for reads whose byte offsets the extension cannot
** know (rows of an ordinary table found through a B-tree):
**
**   do {
**     httpvfs_speculate_begin(db);
**     ... run the lookups (e.g. SELECT ... WHERE rowid=? for 16 ids),
**         ignoring their results and errors: while speculating, uncached
**         blocks read as zeros and are only recorded ...
**     ... reset/finalize those statements ...
**   } while( httpvfs_speculate_end(db) > 0 );
**   ... now run the lookups for real; they hit the cache ...
**
** Each pass fetches every recorded miss in one parallel round, so N lookups
** cost about one round per uncached B-tree level instead of N per level.
** httpvfs_speculate_end() returns the number of blocks fetched (0 when the
** pass found nothing new), or a negative SQLite error code. It also calls
** sqlite3_db_release_memory() so that the zero-filled pages SQLite read
** during the pass leave SQLite's own page cache. Statements that were
** stepped during the pass must be reset before httpvfs_speculate_end().
*/
static inline int httpvfs_speculate_begin(sqlite3 *db){
  int op = 1;
  return sqlite3_file_control(db, "main", HTTPVFS_FCNTL_SPECULATE, &op);
}

static inline int httpvfs_speculate_end(sqlite3 *db){
  int op = 0, rc;
  rc = sqlite3_file_control(db, "main", HTTPVFS_FCNTL_SPECULATE, &op);
  sqlite3_db_release_memory(db);
  if( rc!=SQLITE_OK ) return rc==SQLITE_NOTFOUND ? 0 : -rc;
  return op;
}

#endif /* HTTPVFS_H */
