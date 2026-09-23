"""Late-interaction (ColBERT) encoder for LateOn-Code-edge via ONNX Runtime.

Reproduces PyLate's ``ColBERT.tokenize`` / ``ColBERT.encode`` and next-plaid-onnx's
``prepare_batch_*`` / ``encode_prepared_batch_with_session`` for this model.  The
settings come from ``enc/lateon_onnx_config.json``, a byte-exact reconstruction of the
model's ``onnx_config.json`` (its SHA-256 matches the hash pinned for the Hugging Face
revision whose ``model_int8.onnx`` and ``tokenizer.json`` are identical to ours; see
``enc/NOTES.md``):

* Text is stripped and lower-cased (``do_lower_case = true``).
* Tokenised with ``[CLS] ... [SEP]`` and no padding/truncation from ``tokenizer.json``.
* Truncated to ``length - 1`` tokens keeping the final ``[SEP]``, then the prefix token
  (``[Q] `` = 50368 for queries, ``[D] `` = 50369 for documents) is inserted at
  position 1, right after ``[CLS]``.  ``length`` is 256 for queries, 2048 for documents.
* No query expansion (``do_query_expansion = false``): queries are not padded with
  ``[MASK]``; padding positions get attention 0 and are dropped from the output.
* Documents drop output vectors whose input id is in the skiplist (the 32 ASCII
  punctuation characters of Python's ``string.punctuation``, looked up as whole tokens;
  BPE tokens with a leading space such as ``Ġ(`` are *not* in the skiplist).  Skiplist
  tokens are still attended to by the encoder; only their output vectors are removed.
  Queries keep every token (``[CLS]``, ``[Q] ``, ..., ``[SEP]``).
* The ONNX graph already ends with the two dense projections (256 -> 512 -> 48) and a
  per-token L2 normalisation (``ReduceL2`` / ``Clip`` / ``Div``), so outputs are unit
  vectors and are not re-normalised here.

Score: ``MaxSim(q, d) = sum_i max_j <q_i, d_j>``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

REPO = Path(__file__).resolve().parent.parent
MODEL_DIR = REPO / "models" / "lateon-code-edge"
CONFIG_PATH = Path(__file__).resolve().parent / "lateon_onnx_config.json"
DIM = 48


def default_threads() -> int:
    return int(os.environ.get("ENC_THREADS", "2"))


class LateOn:
    def __init__(self, model_path=MODEL_DIR / "model_int8.onnx",
                 tokenizer_path=MODEL_DIR / "tokenizer.json", config_path=CONFIG_PATH,
                 threads: int | None = None, **overrides):
        cfg = json.loads(Path(config_path).read_text())
        cfg.update(overrides)
        self.cfg = cfg
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        # tokenizer.json carries a stale truncation (2047) and [MASK] batch padding
        # from training; PyLate re-sets both per call, next-plaid-onnx clears them.
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        self.query_prefix_id = cfg.get("query_prefix_id") or self.tokenizer.token_to_id(cfg["query_prefix"])
        self.document_prefix_id = cfg.get("document_prefix_id") or self.tokenizer.token_to_id(cfg["document_prefix"])
        self.pad_id = cfg["pad_token_id"]
        self.mask_id = cfg["mask_token_id"]
        self.skiplist_ids = np.array(sorted(
            {i for w in cfg["skiplist_words"] if (i := self.tokenizer.token_to_id(w)) is not None}),
            np.int64)
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads or default_threads()
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(model_path), so, providers=["CPUExecutionProvider"])
        self.input_names = {i.name for i in self.session.get_inputs()}

    # ------------------------------------------------------------------ tokenisation
    def token_ids(self, texts, is_query: bool):
        """Return the list of model input id sequences (prefix inserted, truncated)."""
        cfg = self.cfg
        L = cfg["query_length"] if is_query else cfg["document_length"]
        prefix = self.query_prefix_id if is_query else self.document_prefix_id
        texts = [str(t).strip() for t in texts]
        if cfg.get("do_lower_case"):
            texts = [t.lower() for t in texts]
        out = []
        for e in self.tokenizer.encode_batch(texts, add_special_tokens=True):
            ids = e.ids
            if len(ids) > L - 1:          # HF truncation to L-1 keeps [SEP] at the end
                ids = ids[:L - 2] + ids[-1:]
            out.append([ids[0], prefix] + ids[1:])
        return out

    # ------------------------------------------------------------------ inference
    def _run(self, seqs, is_query: bool):
        cfg = self.cfg
        n = len(seqs)
        expand = is_query and cfg.get("do_query_expansion")
        L = cfg["query_length"] if expand else max(len(s) for s in seqs)
        ids = np.full((n, L), self.mask_id if expand else self.pad_id, np.int64)
        att = np.zeros((n, L), np.int64)
        for i, s in enumerate(seqs):
            ids[i, :len(s)] = s
            att[i, :len(s)] = 1
        if expand and cfg.get("attend_to_expansion_tokens"):
            att[:] = 1
        feeds = {"input_ids": ids, "attention_mask": att}
        if "token_type_ids" in self.input_names:
            feeds["token_type_ids"] = np.zeros_like(ids)
        y = self.session.run(None, feeds)[0]
        res = []
        for i, s in enumerate(seqs):
            if expand:
                res.append(y[i])                       # all query_length vectors
            elif is_query:
                res.append(y[i, :len(s)])
            else:
                keep = ~np.isin(np.asarray(s), self.skiplist_ids)
                res.append(y[i, :len(s)][keep])
        return res

    def encode(self, texts, is_query: bool, batch_size: int = 32):
        """Encode texts; returns a list of float32 [n_tokens_i, 48] arrays (input order)."""
        texts = list(texts)
        seqs = self.token_ids(texts, is_query)
        order = np.argsort([len(s) for s in seqs], kind="stable")
        out = [None] * len(texts)
        for b in range(0, len(order), batch_size):
            idx = order[b:b + batch_size]
            for i, v in zip(idx, self._run([seqs[i] for i in idx], is_query)):
                out[i] = np.ascontiguousarray(v, dtype=np.float32)
        return out

    def encode_queries(self, texts, batch_size: int = 32):
        return self.encode(texts, True, batch_size)

    def encode_documents(self, texts, batch_size: int = 32):
        return self.encode(texts, False, batch_size)

    def encode_documents_flat(self, texts, batch_size: int = 32):
        """Return (vectors float32 [T, 48], doclens int64 [n])."""
        embs = self.encode_documents(texts, batch_size)
        lens = np.array([len(e) for e in embs], np.int64)
        flat = np.concatenate(embs) if embs else np.zeros((0, DIM), np.float32)
        return flat, lens


def maxsim(q: np.ndarray, d: np.ndarray) -> float:
    return float((q @ d.T).max(1).sum())


if __name__ == "__main__":
    import sys
    m = LateOn()
    q = m.encode_queries([sys.argv[1] if len(sys.argv) > 1 else "hello"])[0]
    print(q.shape, np.linalg.norm(q, axis=1)[:4])
