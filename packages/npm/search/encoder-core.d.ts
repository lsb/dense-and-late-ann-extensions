export declare const MODELS: Record<'minilm' | 'lateon', { onnx: string; tokenizer: string; dim: number }>;

export declare class Encoder {
  /**
   * @param ort       the onnxruntime-web (or onnxruntime-node) namespace
   * @param Tokenizer the Tokenizer class of @huggingface/tokenizers
   * @param which     'minilm' | 'lateon'
   * @param modelBytes the .onnx file
   * @param tokenizerJson the parsed tokenizer.json
   */
  static create(ort: unknown, Tokenizer: unknown, which: 'minilm' | 'lateon', modelBytes: Uint8Array,
                tokenizerJson: unknown, sessionOptions?: Record<string, unknown>): Promise<Encoder>;
  which: 'minilm' | 'lateon';
  tokenizerMs: number;
  sessionMs: number;
  /** Encode one query: MiniLM gives one unit vector of 384; LateOn gives n unit vectors of 48. */
  encode(text: string): Promise<{ ids: number[]; vectors: Float32Array; n: number; dim: number; tokenizeMs: number; inferMs: number }>;
}
