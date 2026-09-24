// Types for the SQLite runtime (wasm/pkg/index.mjs): read-only SQLite over
// HTTP range requests, with FTS5, dense_ann and late_plaid linked in.

export type Variant = 'auto' | 'asyncify' | 'jspi' | 'sync';
export type SqlValue = null | number | bigint | string | Uint8Array;
export type Params = SqlValue[] | Record<string, SqlValue>;

export interface OpenOptions {
  /** VFS block cache size in bytes (default 4 MiB). */
  pageCacheBytes?: number;
  /** Cache block and minimum request size (default 4096; use the page size or a multiple). */
  blockSize?: number;
  /** Maximum sequential readahead in bytes (default 1 MiB; 0 disables). */
  readaheadBytes?: number;
  /** Merge ranges of one batch separated by at most this many bytes (default 0). */
  coalesceGapBytes?: number;
  /** Concurrent requests per round (default unlimited). */
  maxParallel?: number;
  /**
   * Request budget per round: a round needing more range requests is merged
   * into at most this many (nearby ranges joined when the cost model says it
   * is faster) or, with `multipart`, sent as multi-range requests. 'auto'
   * (default): 6 in browsers over HTTP/1.x or an unknown protocol, 100 over
   * HTTP/2 and HTTP/3, 0 (off) in Node. 0 disables.
   */
  maxRequests?: number | 'auto';
  /** Cost-model latency in ms (default 100); refitted when netAutoEstimate. */
  rttMs?: number;
  /** Cost-model bandwidth in kbit/s (default 10000); refitted when netAutoEstimate. */
  bandwidthKbps?: number;
  /** Refit rttMs and bandwidthKbps from observed rounds (default true). */
  netAutoEstimate?: boolean;
  /** Use multi-range requests (multipart/byteranges) when over budget (default false; falls back automatically). */
  multipart?: boolean;
  /** Ranges per multi-range request (default 100). */
  maxRangesPerRequest?: number;
  /** Extra request headers. */
  headers?: Record<string, string>;
  /** fetch implementation (default globalThis.fetch). */
  fetch?: typeof fetch;
  /** fetch() cache mode (default 'no-store'). */
  httpCache?: RequestCache;
  /** SQLite's own page cache in KiB (default 2048). */
  sqliteCacheKiB?: number;
  /** WebAssembly build: 'auto' (default) picks 'jspi' where supported, else 'asyncify'. */
  variant?: Variant;
  /** Sync variant only: blocking fetch of several ranges (default: a worker from sync-fetch.mjs). */
  fetchSync?: (url: string, offsets: number[], lengths: number[], groups?: number[][]) =>
    { buf: Uint8Array; total: number }[] & { multipartFailed?: boolean };
  /** Extra arguments for the Emscripten module factory. */
  moduleArgs?: Record<string, unknown>;
}

/** VFS counters (cumulative since open or resetStats). */
export interface VfsStats {
  requests: number; bytes: number; rounds: number; reads: number;
  cacheHits: number; cacheMisses: number; prefetchCalls: number; prefetchBlocks: number;
  specMisses: number; fileSize: number; blockSize: number; cacheBlocks: number;
  cachedBlocks: number; netMs: number;
}

/** One range; the ranges of one multi-range request share `req` within their round. */
export interface LogEntry { offset: number; length: number; round: number; tStart: number; tEnd: number; req: number }

/** Request planner state (counters since open or resetStats). */
export interface NetState {
  protocol: string; maxRequests: number; rttMs: number; bandwidthKbps: number;
  multipart: boolean; netAutoEstimate: boolean; overfetchBytes: number;
  plannedRounds: number; multipartRequests: number; multipartFailed: boolean; observedRounds: number;
}

export declare class SqliteError extends Error {
  code: number;
  constructor(message: string, code: number);
}

export declare class Database {
  readonly url: string;
  readonly variant: Exclude<Variant, 'auto'>;
  /** Run one or more statements, discarding results. */
  exec(sql: string): Promise<void>;
  /** Run one statement; rows as arrays. */
  queryRaw(sql: string, params?: Params): Promise<{ columns: string[]; rows: SqlValue[][] }>;
  /** Run one statement; rows as objects keyed by column name. */
  query<T = Record<string, SqlValue>>(sql: string, params?: Params): Promise<T[]>;
  stats(): VfsStats | null;
  log(): LogEntry[];
  /** Request planner state. */
  netState(): NetState | null;
  /** Change request planning options (omitted ones keep their value). */
  setNetOptions(opts: Pick<OpenOptions, 'maxRequests' | 'rttMs' | 'bandwidthKbps' | 'netAutoEstimate' | 'multipart' | 'maxRangesPerRequest'>): Promise<void>;
  resetStats(opts?: { clearCache?: boolean }): Promise<void>;
  close(): Promise<void>;
}

/** Open a read-only database served at url (the server must support HTTP Range). */
export declare function open(url: string | URL, opts?: OpenOptions): Promise<Database>;
/** 'jspi' where WebAssembly.Suspending exists, else 'asyncify'. */
export declare function autoVariant(): 'jspi' | 'asyncify';
/** Load (once) and return the Emscripten module of a variant. */
export declare function loadModule(variant?: Variant, moduleArgs?: Record<string, unknown>): Promise<unknown>;
