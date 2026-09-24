/*
** demo_ext.c -- placeholder extension that proves the build and registration
** mechanism, and shows how an extension uses the httpvfs batching calls.
**
**   hello(X)                     'hello, X' (or 'hello, world')
**   httpvfs_warm(SQL, IDS)       run SQL (one '?' parameter) once per integer
**                                in the comma-separated list IDS, using
**                                speculative batching so that all rows are
**                                fetched in about one round per uncached
**                                B-tree level; returns the number of passes
**                                that fetched something
**   httpvfs_prefetch_pages(P, N) prefetch database pages P..P+N-1 as one
**                                batch (pinned in the VFS cache until
**                                read); returns the SQLite result code
**   httpvfs_stats()              JSON text of the VFS counters for "main"
**
** Loadable name: demo_ext.so, entry point sqlite3_demoext_init.
*/
#include "sqlite3ext.h"
SQLITE_EXTENSION_INIT1
#include "httpvfs.h"
#include <stdlib.h>

static void helloFunc(sqlite3_context *ctx, int argc, sqlite3_value **argv){
  const char *z = argc>0 ? (const char*)sqlite3_value_text(argv[0]) : 0;
  sqlite3_result_text(ctx, sqlite3_mprintf("hello, %s", z ? z : "world"), -1,
                      sqlite3_free);
}

static void warmFunc(sqlite3_context *ctx, int argc, sqlite3_value **argv){
  sqlite3 *db = sqlite3_context_db_handle(ctx);
  const char *zSql = (const char*)sqlite3_value_text(argv[0]);
  const char *zIds = (const char*)sqlite3_value_text(argv[1]);
  sqlite3_stmt *st = 0;
  sqlite3_int64 *ids = 0;
  int nid = 0, aid = 0, passes = 0, rc;
  const char *p;
  (void)argc;
  if( !zSql || !zIds ){ sqlite3_result_null(ctx); return; }
  for(p=zIds; *p; ){
    char *e;
    sqlite3_int64 v = strtoll(p, &e, 10);
    if( e==p ){ p++; continue; }
    if( nid>=aid ){
      aid = aid ? aid*2 : 16;
      ids = sqlite3_realloc64(ids, sizeof(*ids)*aid);
      if( !ids ){ sqlite3_result_error_nomem(ctx); return; }
    }
    ids[nid++] = v;
    p = e;
  }
  rc = sqlite3_prepare_v2(db, zSql, -1, &st, 0);
  if( rc ){
    sqlite3_result_error(ctx, sqlite3_errmsg(db), -1);
    sqlite3_free(ids);
    return;
  }
  while( passes<16 ){
    int i, n;
    if( httpvfs_speculate_begin(db)!=SQLITE_OK ) break;  /* not on httpvfs */
    for(i=0; i<nid; i++){
      sqlite3_bind_int64(st, 1, ids[i]);
      while( sqlite3_step(st)==SQLITE_ROW ){}
      sqlite3_reset(st);                   /* errors are expected here */
    }
    n = httpvfs_speculate_end(db);
    if( n<=0 ) break;
    passes++;
  }
  sqlite3_finalize(st);
  sqlite3_free(ids);
  sqlite3_result_int(ctx, passes);
}

static void prefetchPagesFunc(sqlite3_context *ctx, int argc,
                              sqlite3_value **argv){
  sqlite3_int64 first = sqlite3_value_int64(argv[0]);
  int n = sqlite3_value_int(argv[1]), i, rc;
  unsigned int *pg;
  (void)argc;
  if( first<1 || n<0 || n>(1<<24) ){ sqlite3_result_error_code(ctx, SQLITE_RANGE); return; }
  pg = sqlite3_malloc64(sizeof(*pg)*(n>0 ? n : 1));
  if( !pg ){ sqlite3_result_error_nomem(ctx); return; }
  for(i=0; i<n; i++) pg[i] = (unsigned int)(first+i);
  rc = httpvfs_prefetch_pages(sqlite3_context_db_handle(ctx), pg, n);
  sqlite3_free(pg);
  sqlite3_result_int(ctx, rc);
}

static void statsFunc(sqlite3_context *ctx, int argc, sqlite3_value **argv){
  HttpvfsStats s;
  (void)argc; (void)argv;
  if( httpvfs_stats(sqlite3_context_db_handle(ctx), &s)!=SQLITE_OK ){
    sqlite3_result_null(ctx);
    return;
  }
  sqlite3_result_text(ctx, sqlite3_mprintf(
    "{\"requests\":%lld,\"bytes\":%lld,\"rounds\":%lld,\"reads\":%lld,"
    "\"cache_hits\":%lld,\"cache_misses\":%lld,\"prefetch_calls\":%lld,"
    "\"prefetch_blocks\":%lld,\"net_ms\":%.3f,\"cache_blocks\":%d,"
    "\"cache_max_blocks\":%d,\"cached_blocks\":%d,\"pinned_blocks\":%d,"
    "\"peak_blocks\":%d,\"pin_evictions\":%lld}",
    s.requests, s.bytes, s.rounds, s.reads, s.cache_hits, s.cache_misses,
    s.prefetch_calls, s.prefetch_blocks, s.net_ms, s.cache_blocks,
    s.cache_max_blocks, s.cached_blocks, s.pinned_blocks, s.peak_blocks,
    s.pin_evictions), -1, sqlite3_free);
}

#ifdef _WIN32
__declspec(dllexport)
#endif
int sqlite3_demoext_init(sqlite3 *db, char **pzErrMsg,
                         const sqlite3_api_routines *pApi){
  int rc;
  (void)pzErrMsg;
  SQLITE_EXTENSION_INIT2(pApi);
  rc = sqlite3_create_function(db, "hello", 0, SQLITE_UTF8|SQLITE_DETERMINISTIC,
                               0, helloFunc, 0, 0);
  if( rc==SQLITE_OK ) rc = sqlite3_create_function(db, "hello", 1,
                               SQLITE_UTF8|SQLITE_DETERMINISTIC, 0, helloFunc, 0, 0);
  if( rc==SQLITE_OK ) rc = sqlite3_create_function(db, "httpvfs_warm", 2,
                               SQLITE_UTF8, 0, warmFunc, 0, 0);
  if( rc==SQLITE_OK ) rc = sqlite3_create_function(db, "httpvfs_prefetch_pages", 2,
                               SQLITE_UTF8, 0, prefetchPagesFunc, 0, 0);
  if( rc==SQLITE_OK ) rc = sqlite3_create_function(db, "httpvfs_stats", 0,
                               SQLITE_UTF8, 0, statsFunc, 0, 0);
  return rc;
}
