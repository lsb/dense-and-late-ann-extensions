/*
** fts5rank.c -- FTS5 ranking functions that do not read per-document lengths.
**
** FTS5's built-in bm25() calls xColumnSize() for every matching row, which
** reads one row of the %_docsize table per match. Over an HTTP range-request
** VFS those rows are scattered (one 4 KiB page per match, fetched one at a
** time), so a single-word query on 1M documents costs ~700 sequential round
** trips. The functions here compute the same formula without that read:
**
**   bm25c(fts [, w0, w1, ...])
**       bm25 with every document's length taken to be the average length
**       (equivalently: bm25 with b = 0). IDF, term frequencies, k1 = 1.2 and
**       the per-column weights are exactly as in bm25(). Reads only the
**       doclists (already read by MATCH) and the averages record.
**
**   bm25dl(fts, D [, w0, w1, ...])
**       bm25 with the document length D supplied by the caller: an integer or
**       real token count, or a BLOB in the %_docsize format (one varint per
**       column; the sum is used). With D = (SELECT sz FROM fts_docsize WHERE
**       id = fts.rowid) this reproduces bm25() exactly (used by the tests);
**       with a length taken from a compact side table it gives bm25 at the
**       cost of that table instead of %_docsize. NULL D means the average.
**
** Both return -score like bm25(), so ORDER BY rank (ascending) is best first:
**   SELECT rowid FROM fts WHERE fts MATCH ?1 AND rank MATCH 'bm25c()'
**   ORDER BY rank LIMIT 10;
**
** Entry point sqlite3_fts5rank_init; loadable name fts5rank.so. See
** docs/fts5-httpvfs.md.
*/
#include "sqlite3ext.h"
SQLITE_EXTENSION_INIT1
#include <math.h>
#include <string.h>

typedef struct Fts5RankData Fts5RankData;
struct Fts5RankData {
  int nPhrase;
  double avgdl;
  double *aIDF;
  double *aFreq;
};

static int fts5rankCountCb(const Fts5ExtensionApi *pApi, Fts5Context *pFts,
                           void *pUserData){
  (void)pApi; (void)pFts;
  (*(sqlite3_int64*)pUserData)++;
  return SQLITE_OK;
}

/* Same per-query data as fts5Bm25GetData() in SQLite (IDF floor 1e-6). */
static int fts5rankGetData(const Fts5ExtensionApi *pApi, Fts5Context *pFts,
                           Fts5RankData **pp){
  int rc = SQLITE_OK;
  Fts5RankData *p = (Fts5RankData*)pApi->xGetAuxdata(pFts, 0);
  if( p==0 ){
    int i, nPhrase = pApi->xPhraseCount(pFts);
    sqlite3_int64 nRow = 0, nToken = 0;
    p = (Fts5RankData*)sqlite3_malloc64(sizeof(*p) + nPhrase*2*sizeof(double));
    if( p==0 ) return SQLITE_NOMEM;
    memset(p, 0, sizeof(*p) + nPhrase*2*sizeof(double));
    p->nPhrase = nPhrase;
    p->aIDF = (double*)&p[1];
    p->aFreq = &p->aIDF[nPhrase];
    rc = pApi->xRowCount(pFts, &nRow);
    if( rc==SQLITE_OK ) rc = pApi->xColumnTotalSize(pFts, -1, &nToken);
    if( rc==SQLITE_OK ) p->avgdl = nRow>0 ? (double)nToken / (double)nRow : 1.0;
    for(i=0; rc==SQLITE_OK && i<nPhrase; i++){
      sqlite3_int64 nHit = 0;
      rc = pApi->xQueryPhrase(pFts, i, (void*)&nHit, fts5rankCountCb);
      if( rc==SQLITE_OK ){
        double idf = log((nRow - nHit + 0.5) / (nHit + 0.5));
        if( idf<=0.0 ) idf = 1e-6;
        p->aIDF[i] = idf;
      }
    }
    if( rc==SQLITE_OK ) rc = pApi->xSetAuxdata(pFts, p, sqlite3_free);
    else sqlite3_free(p);
    if( rc!=SQLITE_OK ) p = 0;
  }
  *pp = p;
  return rc;
}

/* Sum of the varints of a %_docsize blob. */
static double fts5rankDocsizeBlob(const unsigned char *a, int n){
  double tot = 0;
  int i = 0;
  while( i<n ){
    sqlite3_uint64 v = 0;
    int shift = 0;
    /* SQLite varint (fts5GetVarint32 format): 7 bits per byte, high bit set
    ** on all but the last; values here are small (< 2^28). */
    while( i<n ){
      unsigned char c = a[i++];
      v = (v<<7) | (c & 0x7f);
      if( !(c & 0x80) ) break;
      if( ++shift>8 ) break;
    }
    tot += (double)v;
  }
  return tot;
}

static void fts5rankScore(const Fts5ExtensionApi *pApi, Fts5Context *pFts,
                          sqlite3_context *pCtx, int nWeight,
                          sqlite3_value **apWeight, int haveD, double D){
  const double k1 = 1.2, b = 0.75;
  Fts5RankData *pData = 0;
  int rc, i, nInst = 0;
  double score = 0.0;
  rc = fts5rankGetData(pApi, pFts, &pData);
  if( rc==SQLITE_OK ){
    memset(pData->aFreq, 0, sizeof(double)*pData->nPhrase);
    rc = pApi->xInstCount(pFts, &nInst);
  }
  for(i=0; rc==SQLITE_OK && i<nInst; i++){
    int ip, ic, io;
    rc = pApi->xInst(pFts, i, &ip, &ic, &io);
    if( rc==SQLITE_OK ){
      pData->aFreq[ip] += (nWeight > ic) ? sqlite3_value_double(apWeight[ic]) : 1.0;
    }
  }
  if( rc!=SQLITE_OK ){
    sqlite3_result_error_code(pCtx, rc);
    return;
  }
  {
    double norm = haveD ? k1 * (1 - b + b * D / pData->avgdl) : k1;
    for(i=0; i<pData->nPhrase; i++){
      double f = pData->aFreq[i];
      score += pData->aIDF[i] * ((f * (k1 + 1.0)) / (f + norm));  /* as bm25() */
    }
  }
  sqlite3_result_double(pCtx, -1.0 * score);
}

static void bm25cFunc(const Fts5ExtensionApi *pApi, Fts5Context *pFts,
                      sqlite3_context *pCtx, int nVal, sqlite3_value **apVal){
  fts5rankScore(pApi, pFts, pCtx, nVal, apVal, 0, 0.0);
}

static void bm25dlFunc(const Fts5ExtensionApi *pApi, Fts5Context *pFts,
                       sqlite3_context *pCtx, int nVal, sqlite3_value **apVal){
  double D = 0.0;
  int haveD = 0;
  if( nVal<1 ){
    sqlite3_result_error(pCtx, "bm25dl: missing document length argument", -1);
    return;
  }
  switch( sqlite3_value_type(apVal[0]) ){
    case SQLITE_INTEGER: case SQLITE_FLOAT:
      D = sqlite3_value_double(apVal[0]); haveD = 1; break;
    case SQLITE_BLOB:
      D = fts5rankDocsizeBlob((const unsigned char*)sqlite3_value_blob(apVal[0]),
                              sqlite3_value_bytes(apVal[0]));
      haveD = 1; break;
    default: break;  /* NULL: average length */
  }
  fts5rankScore(pApi, pFts, pCtx, nVal-1, apVal+1, haveD, D);
}

static fts5_api *fts5rankApi(sqlite3 *db){
  fts5_api *pRet = 0;
  sqlite3_stmt *pStmt = 0;
  if( sqlite3_prepare_v2(db, "SELECT fts5(?1)", -1, &pStmt, 0)==SQLITE_OK ){
    sqlite3_bind_pointer(pStmt, 1, (void*)&pRet, "fts5_api_ptr", 0);
    sqlite3_step(pStmt);
  }
  sqlite3_finalize(pStmt);
  return pRet;
}

#ifdef _WIN32
__declspec(dllexport)
#endif
int sqlite3_fts5rank_init(sqlite3 *db, char **pzErrMsg,
                          const sqlite3_api_routines *pApi){
  fts5_api *p;
  int rc;
  SQLITE_EXTENSION_INIT2(pApi);
  (void)pzErrMsg;
  p = fts5rankApi(db);
  if( p==0 || p->iVersion<2 ) return SQLITE_OK;   /* no FTS5: nothing to add */
  rc = p->xCreateFunction(p, "bm25c", 0, bm25cFunc, 0);
  if( rc==SQLITE_OK ) rc = p->xCreateFunction(p, "bm25dl", 0, bm25dlFunc, 0);
  return rc;
}
