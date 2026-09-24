"""Dense encoder for all-MiniLM-L6-v2 (sentence-transformers semantics) via ONNX Runtime.

Semantics reproduced from sentence-transformers' ``all-MiniLM-L6-v2``:

* BERT WordPiece tokenizer with lower-casing (done by the tokenizer's
  ``BertNormalizer``), ``[CLS] ... [SEP]``, truncation to ``max_seq_length`` = 256
  word pieces *including* the two special tokens.  The ``tokenizer.json`` shipped
  with the model says 128 / fixed padding (Hugging Face tokenizer default); that is
  overridden here, because sentence-transformers sets ``max_seq_length = 256``.
* Mean pooling of ``last_hidden_state`` over the attention mask.
* L2 normalisation.

Usage::

    from enc.minilm import MiniLM
    m = MiniLM()
    X = m.encode(["some text", ...])   # float32 [n, 384], unit rows
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

REPO = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = REPO / "models" / "minilm-l6-v2" / "model_w8.onnx"  # weight-only int8; see scripts/dequantize_activations.py
DEFAULT_TOKENIZER = REPO / "models" / "minilm-l6-v2" / "tokenizer.json"
MAX_SEQ_LENGTH = 256
DIM = 384


def default_threads() -> int:
    return int(os.environ.get("ENC_THREADS", "2"))


class MiniLM:
    def __init__(self, model_path=DEFAULT_MODEL, tokenizer_path=DEFAULT_TOKENIZER,
                 threads: int | None = None, max_seq_length: int = MAX_SEQ_LENGTH):
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.tokenizer.no_padding()
        self.tokenizer.enable_truncation(max_length=max_seq_length)
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads or default_threads()
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")  # do not spin on a shared CPU
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(model_path), so, providers=["CPUExecutionProvider"])
        self.input_names = {i.name for i in self.session.get_inputs()}

    def tokenize(self, texts):
        # sentence-transformers strips surrounding whitespace before tokenizing.
        return self.tokenizer.encode_batch([str(t).strip() for t in texts], add_special_tokens=True)

    def _run(self, encs) -> np.ndarray:
        n = len(encs)
        L = max(len(e.ids) for e in encs)
        ids = np.zeros((n, L), np.int64)
        mask = np.zeros((n, L), np.int64)
        for i, e in enumerate(encs):
            ids[i, :len(e.ids)] = e.ids
            mask[i, :len(e.ids)] = 1
        feeds = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self.input_names:
            feeds["token_type_ids"] = np.zeros_like(ids)
        h = self.session.run(None, feeds)[0]  # [n, L, 384]
        m = mask[:, :, None].astype(np.float32)
        pooled = (h * m).sum(1) / np.clip(m.sum(1), 1e-9, None)
        norm = np.linalg.norm(pooled, axis=1, keepdims=True)
        return (pooled / np.clip(norm, 1e-12, None)).astype(np.float32)

    def encode(self, texts, batch_size: int = 64) -> np.ndarray:
        """Encode texts to float32 [n, 384] unit vectors (order preserved).

        Texts are sorted by token length before batching to minimise padding;
        padding does not change results because pooling uses the attention mask.
        """
        texts = list(texts)
        if not texts:
            return np.zeros((0, DIM), np.float32)
        encs = self.tokenize(texts)
        order = np.argsort([len(e.ids) for e in encs], kind="stable")
        out = np.empty((len(texts), DIM), np.float32)
        for s in range(0, len(order), batch_size):
            idx = order[s:s + batch_size]
            out[idx] = self._run([encs[i] for i in idx])
        return out


if __name__ == "__main__":
    import sys
    m = MiniLM()
    v = m.encode(sys.argv[1:] or ["hello world"])
    print(v.shape, v[:, :6])
