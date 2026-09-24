import type { OpenOptions, Variant, VfsStats, LogEntry } from '../index.js';
import type { IndexInfo, SearchParams, SearchResult } from './search-core.js';

export type EncoderName = 'minilm' | 'lateon';

export interface OpenIndexOptions extends Pick<OpenOptions, 'pageCacheBytes' | 'blockSize' | 'readaheadBytes' | 'maxParallel' |
  'maxRequests' | 'multipart' | 'rttMs' | 'bandwidthKbps' | 'netAutoEstimate'> {
  variant?: Exclude<Variant, 'sync'>;
  /** URL of the SQLite module (default: this package's index.mjs). */
  sqliteModule?: string;
  /** Directory URL of onnxruntime-web's dist/ (default: next to this package in node_modules). */
  ortBase?: string;
  /** URL of @huggingface/tokenizers' dist/tokenizers.min.mjs (default: next to this package in node_modules). */
  tokenizersModule?: string;
  /** Base URL for model paths (default: the page's own directory). */
  modelBase?: string;
  /** Model and tokenizer paths, relative to modelBase. */
  models?: Partial<Record<EncoderName, { model?: string; tokenizer?: string }>>;
  /** Start loading these encoders at once. */
  preload?: EncoderName[];
  /** Keep model files in Cache Storage (default true). */
  modelCache?: boolean;
  /** ONNX Runtime threads (needs cross-origin isolation; default 1). */
  encoderThreads?: number;
}

export interface SystemInfo {
  id: string; kind: IndexInfo['kind']; layout?: string; table: string; label: string;
  encoder: EncoderName | null; defaults: SearchParams;
}

export interface EncodeResult {
  ids: number[]; vectors: Float32Array; n: number; dim: number;
  tokenizeMs: number; inferMs: number; ms: number;
}

export interface IndexSearchResult extends SearchResult {
  system: string;
  label: string;
  query: { text: string; tokens?: number; vectors?: number; ids?: number[] };
  /** The query vector(s): [dim] (dense) or [n × dim] (late). */
  embedding?: Float32Array;
}

export declare const DEFAULT_PARAMS: Record<string, SearchParams>;
export declare const MODEL_FILES: Record<EncoderName, { model: string; tokenizer: string }>;

export declare class SearchIndex {
  url: string;
  variant: string;
  indexes: IndexInfo[];
  systems: SystemInfo[];
  openStats: Record<string, number>;
  loadEncoder(which: EncoderName, onProgress?: (p: { url: string; loaded: number; total: number }) => void): Promise<Record<string, unknown>>;
  encode(text: string, which: EncoderName): Promise<EncodeResult>;
  resolve(system: string): IndexInfo;
  search(text: string, opts?: { system?: string; k?: number; cold?: boolean; fetchDocs?: boolean } & SearchParams): Promise<IndexSearchResult>;
  vfsStats(): Promise<VfsStats>;
  vfsLog(): Promise<LogEntry[]>;
  resetStats(opts?: { clearCache?: boolean }): Promise<VfsStats>;
  query(sql: string, params?: unknown[]): Promise<{ columns: string[]; rows: unknown[][] }>;
  close(): Promise<void>;
}

/** Open a search database in a browser (uses module Workers). */
export declare function openIndex(url: string, opts?: OpenIndexOptions): Promise<SearchIndex>;
