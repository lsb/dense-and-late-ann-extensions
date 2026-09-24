#!/usr/bin/env python3
"""Pure-Python reader of the late_plaid storage format (reference/debugging).

Parses the static data in <t>_meta, decodes document rows of <t>_docs and
posting lists of <t>_post with fast-plaid's residual bit order, and computes
decompressed MaxSim with NumPy. Used by the tests to check the C code.
"""
import sqlite3
import struct

import numpy as np

LAY_PLAID, LAY_WARP = 1, 2


def _stream(con, table):
    return b"".join(r[0] for r in con.execute(f"SELECT data FROM {table} ORDER BY id"))


class Index:
    def __init__(self, con, name="t"):
        self.name = name
        blob = _stream(con, f"{name}_meta")
        magic, ver = struct.unpack_from("<II", blob, 0)
        assert magic == 0x3150544C and ver in (1, 2)
        (self.dim, self.nbits, self.K, self.layout, self.kbits, self.idbits, self.chunk,
         self.rowid_identity) = struct.unpack_from("<8I", blob, 8)
        self.N, self.T, self.ivf_entries = struct.unpack_from("<3Q", blob, 40)
        p = 64
        self.cutoffs = np.frombuffer(blob, np.float32, 16, p)[: (1 << self.nbits) - 1]
        self.weights = np.frombuffer(blob, np.float32, 16, p + 64)[: 1 << self.nbits]
        p += 132
        n_ivf_runs, n_post_runs = struct.unpack_from("<II", blob, p)
        p += 8
        self.G = 0
        if ver >= 2:
            self.G, _ = struct.unpack_from("<II", blob, p)
            p += 8
        K, D = self.K, self.dim
        self.ivf_cnt = self.post_cnt = None
        if self.G == 0:
            self.centroids = np.frombuffer(blob, np.float16, K * D, p).astype(np.float32).reshape(K, D)
            p += 2 * K * D
            if self.layout & LAY_PLAID:
                self.ivf_cnt = np.frombuffer(blob, np.uint32, K, p); p += 4 * K
            if self.layout & LAY_WARP:
                self.post_cnt = np.frombuffer(blob, np.uint32, K, p); p += 4 * K
        else:
            G = self.G
            p += 2 * G * D
            start = np.frombuffer(blob, np.uint32, G + 1, p); p += 4 * (G + 1)
            cells = _stream(con, f"{name}_cells")
            cen, ivf, post, q = [], [], [], 0
            for g in range(G):
                kg = int(start[g + 1] - start[g])
                cen.append(np.frombuffer(cells, np.float16, kg * D, q).astype(np.float32).reshape(kg, D)); q += 2 * kg * D
                if self.layout & LAY_PLAID:
                    ivf.append(np.frombuffer(cells, np.uint32, kg, q)); q += 4 * kg
                if self.layout & LAY_WARP:
                    post.append(np.frombuffer(cells, np.uint32, kg, q)); q += 4 * kg
            self.centroids = np.concatenate(cen)
            if ivf:
                self.ivf_cnt = np.concatenate(ivf)
            if post:
                self.post_cnt = np.concatenate(post)
        self.rbytes = D * self.nbits // 8
        # byte value -> bucket of each of its 8/nbits dimensions (fast-plaid order:
        # bucket bits LSB first, bit stream packed MSB first)
        per = 8 // self.nbits
        self.byte_buckets = np.zeros((256, per), np.int64)
        for v in range(256):
            for k in range(per):
                b = 0
                for t in range(self.nbits):
                    b |= ((v >> (7 - (k * self.nbits + t))) & 1) << t
                self.byte_buckets[v, k] = b

    def decode(self, codes, residuals):
        """codes [n], residuals uint8 [n, rbytes] -> unit vectors [n, dim]."""
        b = self.byte_buckets[residuals].reshape(len(codes), -1)
        v = self.centroids[codes] + self.weights[b]
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)

    @staticmethod
    def _bits(buf, n, nb):
        bits = np.unpackbits(np.frombuffer(buf, np.uint8), bitorder="little")[: n * nb]
        return (bits.reshape(n, nb).astype(np.int64) << np.arange(nb)).sum(1)

    def docs(self, con):
        """{doc id: unit vectors [n_tok, dim]} from the plaid layout."""
        out = {}
        for i, row in con.execute(f"SELECT id, data FROM {self.name}_docs"):
            n = row[0] | (row[1] << 8)
            cb = (n * self.kbits + 7) // 8
            codes = self._bits(row[2:2 + cb], n, self.kbits)
            res = np.frombuffer(row, np.uint8, n * self.rbytes, 2 + cb).reshape(n, self.rbytes)
            out[i] = self.decode(codes, res)
        return out

    def postings(self, con):
        """[(centroid, doc ids, unit vectors)] from the warp layout."""
        s = _stream(con, f"{self.name}_post")
        out, p = [], 0
        for c, n in enumerate(self.post_cnt):
            n = int(n)
            ib = (n * self.idbits + 7) // 8
            ids = self._bits(s[p:p + ib], n, self.idbits)
            res = np.frombuffer(s, np.uint8, n * self.rbytes, p + ib).reshape(n, self.rbytes)
            out.append((c, ids, self.decode(np.full(n, c), res)))
            p += ib + n * self.rbytes
        return out


def maxsim_all(docs, q):
    return {d: float((v @ q.T).max(0).sum()) for d, v in docs.items()}


if __name__ == "__main__":
    import sys
    con = sqlite3.connect(sys.argv[1])
    ix = Index(con)
    print({k: getattr(ix, k) for k in ("dim", "nbits", "K", "G", "N", "T", "kbits", "idbits", "layout")})
    print("cutoffs", ix.cutoffs, "weights", ix.weights)
