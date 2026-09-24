import type { Database, VfsStats } from '../index.js';

export interface IndexInfo {
  table: string;
  kind: 'fts' | 'dense' | 'late';
  module: 'fts5' | 'dense_ann' | 'late_plaid';
  layout?: string;
  params?: Record<string, string>;
  content?: string | null;
  contentRowid?: string;
}

export interface DocsInfo { table: string; key: string; column: string | null }

/** Query parameters: dense graph ef, beam, rerank; dense IVF nprobe, rerank_k, rerank;
 *  late nprobe, layout, stoplist, rerank, approx, ndocs, …; FTS mode 'or' | 'and',
 *  rank (FTS_RANKS; default 'bm25c'), prefetchCap (bm25-prefetch, 4000),
 *  rerankDepth (bm25-rerank, 50). */
export type SearchParams = Record<string, string | number>;

export interface SearchRow { id: number; score: number; text?: string | null }

export interface SearchResult {
  table: string;
  kind: IndexInfo['kind'];
  layout?: string;
  sql: string;
  sqlArgs?: unknown[];
  match?: string;
  params?: SearchParams;
  rows: SearchRow[];
  /** The extension's own per-query statistics (parsed JSON), if any. */
  ext: unknown;
  stats: {
    searchMs?: number; docsMs?: number; rounds?: number; requests?: number; bytes?: number; netMs?: number;
    phases?: { search: Partial<VfsStats>; docs: Partial<VfsStats> };
    [key: string]: unknown;
  };
}

export interface SearchRequest {
  /** 'fts' | 'dense' | 'late' (default 'fts'), or use `table`. */
  system?: string;
  table?: string;
  /** Query text (FTS5). */
  text?: string;
  /** A ready FTS5 MATCH expression, used instead of `text`. */
  match?: string;
  /** Query vector(s): [384] for MiniLM (dense), [n × 48] for LateOn (late). */
  vector?: Float32Array;
  k?: number;
  params?: SearchParams;
  /** Empty the VFS and SQLite page caches first. */
  cold?: boolean;
  /** Fetch the text of the hits (default true). */
  fetchDocs?: boolean;
}

export declare const DEFAULT_PARAMS: Record<string, SearchParams>;
/** FTS5 ranking methods (params.rank): 'bm25c' (default: bm25 at the average
 *  document length, no per-document reads), 'bm25' (FTS5's built-in, one
 *  round trip per match over HTTP), 'bm25-prefetch' (exact bm25, lengths
 *  fetched in one batch), 'bm25-rerank' (bm25c's top 50 re-scored exactly),
 *  'none' (rowid order). See docs/fts5-httpvfs.md. */
export declare const FTS_RANKS: readonly ['bm25c', 'bm25', 'bm25-prefetch', 'bm25-rerank', 'none'];
export declare function ftsQuery(text: string, mode?: 'or' | 'and'): string;
export declare function discover(db: Database): Promise<{ indexes: IndexInfo[]; docs: DocsInfo | null }>;
export declare function buildSql(ix: IndexInfo, k: number, params: SearchParams): { sql: string; extra?: unknown[] };

export declare class SearchDb {
  static open(db: Database): Promise<SearchDb>;
  db: Database;
  indexes: IndexInfo[];
  docs: DocsInfo | null;
  resolve(system: string): IndexInfo;
  fetchDocs(ids: number[]): Promise<Map<number, string>>;
  /** Fetch the %_docsize rows of these documents in one batch (httpvfs_warm). */
  warmDocsizes(ix: IndexInfo, ids: number[]): Promise<boolean>;
  search(req: SearchRequest): Promise<SearchResult>;
  close(): Promise<void>;
}
