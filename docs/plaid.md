# PLAID as implemented in fast-plaid and next-plaid

**PLAID** (Performance-optimized Late Interaction Driver; Santhanam et al., 2022) is an index and search procedure for ColBERT-style late-interaction retrieval. Every document token vector is replaced by the id of its nearest *centroid* plus a few bits of quantized *residual*; an inverted file (IVF) maps each centroid to the documents that contain a token assigned to it. A query first selects centroids, gathers candidate documents from their IVF lists, ranks the candidates cheaply using centroid scores alone, and only then decompresses the residuals of a short list for exact MaxSim scoring.

This page specifies PLAID as implemented in LightOn's [fast-plaid](https://github.com/lightonai/fast-plaid) (Rust on libtorch, Python front end; commit `df2be46e`, 2026-09-10) and the differences in [next-plaid](https://github.com/lightonai/next-plaid) (pure Rust on ndarray, used by `colgrep`; commit `00e26aae`, 2026-08-18). File references are relative to `/home/user/ref/`. The final section translates the design into requirements for a SQLite extension that reads its database over HTTP range requests. It is the specification for this project's C implementation.

Notation: *N* documents, *T* stored token vectors in total, dimension *D* (48 for LateOn-Code-edge), *K* centroids, `nbits` ∈ {1, 2, 4} residual bits per dimension, a query with *Q* token vectors. All vectors are L2-normalized, so a dot product is a cosine similarity. MaxSim: score(q, d) = Σ_{i<Q} max_{j<|d|} q_i · d_j.

## Index construction

### Entry points and defaults

`FastPlaid.create` (`fast-plaid/python/fast_plaid/search/fast_plaid.py:672-786`) computes centroids in Python, then calls the Rust `create_index` (`fast-plaid/rust/index/create.rs:206-585`).

| Parameter | fast-plaid default | next-plaid default | Meaning |
|---|---|---|---|
| `nbits` | 4 (`fast_plaid.py:677`) | 4 (`next-plaid/src/index.rs:170`) | residual bits per dimension |
| `kmeans_niters` | 4 (`fast_plaid.py:675`) | 4 (`index.rs:173`) | Lloyd iterations |
| `max_points_per_centroid` | 256 (`fast_plaid.py:676`) | 256 (`index.rs:174`) | caps k-means training points at K·256 |
| `n_samples_kmeans` | `None` → formula below | `None` → formula below | documents sampled for k-means |
| `batch_size` | 25,000 documents per chunk (`fast_plaid.py:679`) | 50,000 (`index.rs:171`) | documents per on-disk chunk; in fast-plaid also the token batch for compression |
| `seed` | 42 | 42 | sampling and k-means seed |
| `start_from_scratch` | 1000 at `create`, 999 at `update` | 999 (`next-plaid/src/lib.rs:49`) | small indexes keep raw embeddings and are rebuilt on update |

PyLate's `PLAID` index wraps fast-plaid with `nbits=4`, `n_ivf_probe=8`, `n_full_scores=8192` (`pylate/pylate/indexes/plaid.py:129-145`). colgrep uses next-plaid defaults except `n_full_scores = 8192` (`next-plaid/colgrep/src/index/mod.rs:817`) and document token pooling with `pool_factor = 2`.

### Number of centroids

The document sample for k-means has

  n_samples = min(1 + ⌊16 · √(120 · N)⌋, N)

documents, chosen by a seeded random permutation (`fast_plaid.py:111-126`; next-plaid `kmeans.rs:273-290`). For N = 10,000 this is the whole corpus; for N = 1,000,000 it is 17,528 documents.

The number of centroids is

  K = 2^⌊log₂(16 · √T̂)⌋

where T̂ = (tokens in the sample / n_samples) · N is the estimated total number of token vectors (`fast_plaid.py:154-162`; next-plaid `kmeans.rs:303-309`). K is then capped at the number of sample tokens (`fast_plaid.py:168`). The Rust side recomputes the same formula with the exact average document length and stores it as `num_partitions` in `metadata.json` (`create.rs:292-294`, `create.rs:569-577`); it only sizes the IVF length array (`bincount` minimum length, `create.rs:542`). If the two estimates straddle a power of two, the metadata disagrees with the real centroid count; an implementation should take K from `centroids.npy`.

| Corpus (LateOn-Code-edge) | T | 16·√T | K |
|---|---|---|---|
| words-100 | 11,086 | 1,685 | 1,024 |
| words-10k | 1,113,741 | 16,885 | 16,384 = 2¹⁴ |
| LLM paragraphs, 10k (58.7 tokens/doc) | ≈ 587,000 | 12,260 | 8,192 = 2¹³ |
| words-1m (111.4 tokens/doc) | ≈ 111.4 M | 168,900 | 131,072 = 2¹⁷ |
| 1M LLM-style documents | ≈ 58.7 M | 122,600 | 65,536 = 2¹⁶ |

The 17-bit centroid ids in Omar Khattab's measurement (see below) are this formula at about 100 M tokens.

### k-means

fast-plaid's `FastKMeans` (`fast-plaid/python/fast_plaid/search/kmeans.py:61-265`) is plain Lloyd's algorithm with squared Euclidean distance:

1. If the sample holds more than K·`max_points_per_centroid` tokens, a random subset of exactly K·256 tokens is used (`kmeans.py:116-122`). At K = 2¹⁷ that is 33.5 M tokens.
2. Initialization: K distinct random sample points (`kmeans.py:130-131`), not k-means++.
3. `kmeans_niters` = 4 iterations (the function default of 25 is overridden, `fast_plaid.py:170-178`), stopping early if the summed centroid shift falls below 1e-8.
4. Empty clusters are re-seeded with random sample points (`kmeans.py:201-209`).
5. The centroids are L2-normalized and stored as float16 (`fast_plaid.py:187-190`). next-plaid does the same (`kmeans.rs:414-421`).

Because the vectors and centroids are unit vectors, nearest centroid by Euclidean distance and by dot product coincide after normalization; assignment during encoding uses `argmax(e · c)` (`create.rs:148-170`).

### Residual codec

The codec is trained on a held-out set of token vectors (`create.rs:222-398`; next-plaid `index.rs:270-362`):

1. **Held-out set.** heldout_size = round(min(0.05 · tokens in the sample, 50,000)). fast-plaid walks the sampled documents from the *end* of the sample list, taking whole documents and, for the last one, its final rows (`create.rs:255-282`). next-plaid takes the *first* rows of each document instead (`index.rs:296-310`). Both use every held-out token in fp16/fp32.
2. **Codes and residuals.** code(e) = argmax_c e · c over the K centroids; residual r = e − c_code(e), computed in float32.
3. **`cluster_threshold`** = the 0.75 quantile of ‖r‖₂ over the held-out set (`create.rs:333-339`). It is used only by updates (centroid expansion).
4. **`avg_residual`** = mean |r_d| per dimension (`create.rs:341-344`). It is saved but *not used* by search.
5. **Bucket cutoffs.** Flatten all residual components r_d of the held-out set into one list. With B = 2^nbits buckets, cutoff_i = quantile(i / B) for i = 1 … B−1 (`create.rs:346-357`).
6. **Bucket weights.** weight_i = quantile((i + 0.5) / B) for i = 0 … B−1 (`create.rs:359-364`): the median of each bucket's quantile range.
7. Quantiles are linearly interpolated order statistics: with the values sorted, position p = q·(n−1), value = x_⌊p⌋ + (p − ⌊p⌋)·(x_⌈p⌉ − x_⌊p⌋) (fast-plaid `rust/search/tensor.rs:18-34`, next-plaid `utils.rs:94-148`), i.e. NumPy's default `linear` method.

The cutoffs and weights are one global scalar table shared by all dimensions and all centroids: for nbits = 1 a single cutoff (the median residual component) and two weights (the 25th and 75th percentiles); for nbits = 2, three cutoffs and four weights; for nbits = 4, fifteen and sixteen.

### Encoding a vector

For each document token vector e (`create.rs:404-428`; next-plaid `codec.rs:356-400`):

1. code = argmax_c e · c.
2. r = e − c_code.
3. For each dimension d: bucket_d = number of cutoffs strictly less than r_d (`torch.bucketize` with `right=False`; a value equal to a cutoff falls in the lower bucket). bucket_d ∈ [0, B).
4. **Packing.** Emit the bits of each bucket least-significant bit first, dimensions in order, giving a stream of D·nbits bits; pack the stream into bytes most-significant bit first (`numpy.packbits` order; `create.rs:176-184`). Equivalently, bit t of bucket_d is stored in byte ⌊(d·nbits + t)/8⌋ at bit position 7 − ((d·nbits + t) mod 8). Each vector takes D·nbits/8 bytes: 6, 12 or 24 bytes at D = 48. The layout was checked by simulating fast-plaid's pack and unpack tables (for example nbits = 2, buckets [3,2,2,1,1,0,0,0] pack to bytes `11010110 10000000`).

Decoding (`search.rs:268-326`): the codec builds a 256-entry table that reverses the bit order inside each nbits-wide field (`residual_codec.rs:80-110`) and a `[256, 8/nbits]` table that splits a byte into its fields, most significant field first (`residual_codec.rs:116-142`); the composition maps each packed byte to the bucket indices of its 8/nbits dimensions. The reconstructed vector is

  ê = normalize(c_code + (w_{bucket_0}, …, w_{bucket_{D−1}}))

computed in float16 on the device, with L2 normalization (norm clamped at 1e-12). Normalization matters: next-plaid measured that skipping it costs up to 0.17 NDCG@10 at nbits = 1 (`next-plaid/src/residual_lut.rs:27-30`).

### On-disk layout (fast-plaid)

Documents are processed in chunks of `batch_size` documents; chunk *k* writes (`create.rs:476-490`):

| File | Type and shape | Contents |
|---|---|---|
| `centroids.npy` | float16 [K, D] | unit centroids |
| `bucket_cutoffs.npy` | float32 [B−1] | loaded as float16 for search (`rust/search/load.rs:156-159`) |
| `bucket_weights.npy` | float32 [B] | loaded as float16 |
| `avg_residual.npy` | float32 [D] | unused at search |
| `cluster_threshold.npy` | float32 scalar | for updates |
| `{k}.codes.npy` | int64 [tokens in chunk] | centroid id per token, documents concatenated in order |
| `{k}.residuals.npy` | uint8 [tokens in chunk, D·nbits/8] | packed residuals |
| `doclens.{k}.json` | int list | tokens per document |
| `{k}.metadata.json` | JSON | `num_documents`, `num_embeddings`, `embedding_offset` |
| `ivf.npy` | int64 [total IVF entries] | document ids, grouped by centroid |
| `ivf_lengths.npy` | int32 [K] | entries per centroid |
| `metadata.json` | JSON | `num_chunks`, `nbits`, `num_partitions`, `num_embeddings`, `avg_doclen`, `num_documents` |
| `plan.json` | JSON | `nbits`, `num_chunks` |

At load time the chunks are concatenated into one codes array and one residuals array addressed through per-document offsets (a "strided tensor", `rust/search/tensor.rs:315-376`). Document ids are dense, 0 … N−1, in insertion order. next-plaid uses the same files plus memory-mapped merged copies (`merged_codes.npy`, `merged_residuals.npy`) and, for new indexes, a per-token float32 inverse reconstruction norm (`{k}.inv_norms.npy`, `merged_inv_norms.npy`; `next-plaid/src/index.rs:38`, `index.rs:2477`).

### Inverted file

`optimize_ivf` (`create.rs:55-132`, built at `create.rs:527-558`): sort all token codes; map each token to its document id; within each centroid's run, keep the **sorted, de-duplicated document ids**. The IVF therefore lists, for each centroid, every document with at least one token assigned to it, once. Its size is the number of distinct (centroid, document) pairs, at most T.

## Search pipeline

`FastPlaid.search` (`fast_plaid.py:1196-1251`) forwards to the Rust `search` (`fast-plaid/rust/search/search.rs:744-1022`), one query at a time.

| Parameter | fast-plaid | next-plaid | Role |
|---|---|---|---|
| `top_k` | 10 | 10 | results returned |
| `n_ivf_probe` | 8 | 8 | centroids probed *per query token* |
| `n_full_scores` | 4096 | 4096 (colgrep and PyLate: 8192) | candidates kept after approximate scoring |
| documents decompressed | max(n_full_scores/4, 1) = 1024 | max(n_full_scores/4, top_k) = 1024 | exact MaxSim short list |
| `centroid_score_threshold` (t_cs) | none | 0.4 (`next-plaid/src/search.rs:49-51`) | drop probed centroids whose best query-token score is below t_cs |
| `batch_size` | `"auto"`: documents per scoring chunk sized from a 256 MiB host workspace (`fast_plaid.py:196-237`) | 2000 queries per batch (`search.rs:56`) | memory/throughput only; never changes results |
| `centroid_batch_size` | — | 100,000 (`search.rs:45-47`) | above this K, probing uses a batched, sparse path |

### Stage 1: centroid scores

S = C · Qᵀ, a [K, Q] matrix of query-token/centroid similarities, float16 in fast-plaid (`search.rs:767`), float32 in next-plaid (`next-plaid/src/search.rs:968`). It is computed once and reused by stages 2 and 3. It requires the whole centroid table.

### Stage 2: IVF probing and candidate generation

For every query token i, take the `n_ivf_probe` centroids with the highest S[·, i] (`search.rs:802-813`). The probed set is the union over query tokens, de-duplicated (`search.rs:815-816`); up to Q·8 cells. next-plaid then drops any probed centroid c with max_i S[c, i] < t_cs (`next-plaid/src/search.rs:1063-1072`). The candidates are the union of the probed cells' IVF lists, sorted and de-duplicated (`search.rs:818-824`; next-plaid uses a bitmap over document ids, `index.rs:1420-1446`).

### Stage 3: approximate scoring (centroid interaction)

For every candidate document d with codes (k_1 … k_|d|), the approximate score replaces each token by its centroid:

  approx(q, d) = Σ_i max_j S[k_j, i]

(`search.rs:848-863`, reduction in `colbert_score_reduce`, `search.rs:656-670`; padding rows are masked with −9999). This reads the codes of **every** candidate, but no residuals. next-plaid quantizes S to uint8 with one global scale for this stage (`next-plaid/src/search.rs:651-700`, `search.rs:1100-1124`), and has an equivalent sparse path that only scores the centroids actually referenced by candidates when K exceeds `centroid_batch_size` (`next-plaid/src/search.rs:1152-1225`).

### Stage 4: short-listing

Keep the `n_full_scores` best candidates by approximate score, then the best max(n_full_scores/4, 1) of those (`search.rs:886-906`; next-plaid `search.rs:1127-1144`, with max(·, top_k)). Because the first cut is sorted, the effective rule in both implementations is simply "decompress the top n_full_scores/4 candidates by approximate score"; the value of `n_full_scores` matters only through that quotient.

### Stage 5: exact scoring

For each short-listed document, read its codes and packed residuals, decompress and normalize every token (formula above), compute MaxSim against the full-precision query, and return the `top_k` highest (`search.rs:929-977`).

next-plaid avoids decompression by default with an asymmetric lookup-table kernel (`next-plaid/src/residual_lut.rs:1-30`):

  q_i · ê_j = inv_norm_j · (S[k_j, i] + Σ_d q_{i,d} · w_{bucket_{j,d}})

where S is reused from stage 1, the residual term is an int8 query times an int8-quantized weight table (one fused 256-entry table per byte, factored into nibble tables for SIMD), and inv_norm_j = 1/‖c_{k_j} + w_{bucket_j}‖ is precomputed per token at index time. Measured cost versus the float path: under 0.002 NDCG@10.

### Relation to the PLAID paper

The paper's pipeline has four stages: candidate generation (`nprobe` cells per query token), *pruned* centroid interaction (only centroids with score ≥ t_cs contribute), full centroid interaction on the survivors, and residual decompression. fast-plaid keeps stages 1, 3 and 4 and drops centroid pruning altogether; next-plaid applies t_cs only to the choice of probed cells, not inside the approximate score. For reference, the paper's settings were, from memory of Santhanam et al. (2022) and therefore to be checked: k = 10: nprobe 1, t_cs 0.5, ndocs 256; k = 100: nprobe 2, t_cs 0.45, ndocs 1024; k = 1000: nprobe 4, t_cs 0.4, ndocs 4096 — much smaller than fast-plaid's `n_ivf_probe = 8`.

## Updates, appends and deletes

fast-plaid `update` (`fast_plaid.py:883-958`, `python/fast_plaid/search/update.py:238-495`, `rust/index/update.rs:30-473`); next-plaid mirrors it (`next-plaid/src/update.rs`, `index.rs:1706-1920`):

1. **Small index: rebuild.** If the index holds ≤ `start_from_scratch` (999) documents and still has its raw embeddings (`embeddings.npy`, written by `create` for indexes of ≤ 1000 documents), old and new embeddings are concatenated and the index is rebuilt from scratch, including new centroids and a new codec (`update.py:349-385`).
2. **Buffered append.** Otherwise new documents are appended immediately using the *existing* centroids and codec, and their raw embeddings are also kept in `buffer.npy` (`update.py:476-495`).
3. **Centroid expansion.** When buffered plus new documents reach `buffer_size` (100 documents), the buffered documents are deleted from the index, and all their tokens plus the new ones are checked against the existing centroids: tokens whose squared distance to the nearest centroid exceeds `cluster_threshold²` are outliers. If there are any, k = 4·⌈outliers/256⌉ new centroids are trained on them with the same k-means and appended to `centroids.npy` (`update.py:97-233`, `k_update` at line 194). The documents are then re-added (`update.py:418-474`).
4. **Encoding appended documents** uses the frozen bucket cutoffs and weights (`update.rs:141-160`); only the centroid set grows. With expansion, `cluster_threshold` becomes the token-weighted average of the old value and the 0.75 quantile of the new residual norms (`update.rs:277-297`).
5. **Chunks.** New documents go into the last chunk if it holds fewer than 2000 documents, otherwise into new chunks of 25,000 (`update.rs:78-115`).
6. **IVF merge.** New (centroid, document) pairs are appended to the end of each centroid's list; lists stay sorted because new document ids are larger than old ones (`update.rs:318-444`).
7. **Deletes** (`rust/index/delete.rs`) rewrite the affected chunks without the deleted documents' tokens and rebuild the whole IVF. Document ids are compacted, so the ids of later documents shift down.

## Implications for a SQLite/httpvfs implementation

A browser client reads the database lazily over HTTP range requests, typically in SQLite pages of 4 KiB, with request latency (tens of milliseconds) dominating bandwidth. What matters is therefore (a) how many bytes each query must read, (b) how many *dependent* round trips it needs, and (c) whether the reads are contiguous.

### Storage per token at D = 48

| Component | nbits = 1 | nbits = 2 | nbits = 4 | Notes |
|---|---|---|---|---|
| Packed residual | 6 B | 12 B | 24 B | D·nbits/8 |
| Centroid id, as fast-plaid stores it | 8 B | 8 B | 8 B | int64 |
| Centroid id, bit-packed | 2.125 B | 2.125 B | 2.125 B | 17 bits for K = 2¹⁷; 14 bits for words-10k |
| Inverse norm (next-plaid LUT path) | 0–4 B | 0–4 B | 0–4 B | float32 in next-plaid; float16 or a log-scaled byte would do; recomputable from code + residual |
| IVF entry, as fast-plaid stores it | ≤ 8 B | ≤ 8 B | ≤ 8 B | int64 per distinct (centroid, document) pair |
| IVF entry, bit-packed | ≤ 2.5 B | ≤ 2.5 B | ≤ 2.5 B | 20 bits for N = 10⁶; delta coding of sorted ids needs ≈ log₂(N·K/entries) + 2 ≈ 12 bits |
| **Total, packed, no norm** | **≈ 10.6 B** | **≈ 16.6 B** | **≈ 28.6 B** | versus 96 B for float16 vectors |

Fixed per-index data: the centroid table, K·D·2 bytes in float16 (1.5 MiB at K = 2¹⁴, 6 MiB at 2¹⁶, 12 MiB at 2¹⁷), and the codec tables (a few bytes).

Projected index sizes (packed layout, nbits = 1 / 2 / 4, IVF entries counted as one per token, which is an upper bound):

| Corpus | Tokens | nbits = 1 | nbits = 2 | nbits = 4 |
|---|---|---|---|---|
| words-10k | 1.11 M | 12 MB | 19 MB | 32 MB |
| words-1m | 111.4 M | 1.18 GB | 1.85 GB | 3.19 GB |
| 1M LLM-style 50-word documents | 58.7 M | 0.62 GB | 0.97 GB | 1.68 GB |

### What each query reads

The stages form a strict dependency chain; each arrow is at least one network round trip unless the data is cached:

1. **Encode the query** locally (ONNX). LateOn-Code-edge queries have no `[MASK]` expansion, so Q = number of word pieces + 3 (typically 6–15).
2. **Centroid table** (whole, K·D·2 bytes). Needed before anything else. It is the same for every query, so it should be fetched once per session and cached, but it is the single largest read for a first query: 12 MiB at K = 2¹⁷. Reducing it: store it as int8 (half the size, small score error), choose a smaller K than the formula (for example 2¹⁵ for 1M documents, at the cost of longer IVF lists), or add a coarse second level (centroids of centroids) so that only the relevant slice of the table is fetched.
3. **IVF lists** of the probed cells: up to Q·`n_ivf_probe` lists, each of about (IVF entries / K) document ids: 65 on average for words-10k (measured) and about 800 for words-1m (list length grows as √T because K does). These are independent reads that can be issued in parallel; they are contiguous if the IVF is stored centroid-major.
4. **Codes of every candidate document** for approximate scoring. This is the expensive stage over HTTP: with thousands to tens of thousands of candidates, each needing its own row, a faithful implementation reads megabytes scattered over the whole document table (for words-1m, roughly 50,000 candidates × 111 tokens × 17 bits ≈ 12 MB). This read depends on stage 3.
5. **Residuals (and codes) of the short list**, n_full_scores/4 documents: 1024 × 111 tokens × 6 B ≈ 0.7 MB at nbits = 1 (2.7 MB at nbits = 4), again scattered. Depends on stage 4's ranking.

So a faithful PLAID query costs three dependent round trips after the centroid table (IVF → codes → residuals), and stages 4 and 5 read one row per document.

**Measured on words-10k.** A NumPy simulation of index construction (K = 16,384, 4 k-means iterations on all 1.11 M LateOn-Code-edge token vectors, codes by maximum dot product) and of stages 1–2, with 50 queries of three random words taken from a random document (9.9 query vectors on average):

| `n_ivf_probe` | Cells probed (t_cs none / 0.4 / 0.5) | Candidates, mean (max) | Source document among candidates |
|---|---|---|---|
| 1 | 8.8 / 8.8 / 8.4 | 420 (756) | 70 % |
| 2 | 17.8 / 17.7 / 16.5 | 824 (1,527) | 90 % |
| 4 | 35.1 / 34.8 / 31.9 | 1,557 (2,694) | 92 % |
| 8 | 69.4 / 68.4 / 61.5 | 2,794 (4,649) | 92 % |

Other measurements from the same run: 1,067,606 distinct (centroid, document) pairs, 0.959 per token, so the IVF is nearly as long as the token array for this corpus; IVF lists average 65 documents (median 51, maximum 646); the 0.75 quantile of residual norms (`cluster_threshold`) is 0.42. The t_cs = 0.4 threshold removes almost nothing for LateOn-Code-edge, whose query–centroid similarities are high. At the default `n_ivf_probe = 8`, a query touches 28 % of all documents in stage 3, which confirms that the approximate-scoring read is the bottleneck. (Exact brute-force MaxSim ranks the source document first for only 48 % of these deliberately hard, non-semantic queries; they measure candidate generation, not model quality.)

### Recommendations for the C extension

1. **Faithful baseline first.** Implement exactly the pipeline above (fast-plaid semantics, next-plaid's t_cs = 0.4 as an option, decompress-and-normalize for exact scoring), with the parameters exposed: `n_ivf_probe`, `n_full_scores` (or directly the number decompressed), `t_cs`, `top_k`. Validate against a Python reference on words-10k before optimizing.
2. **Bit-pack everything.** Store codes as ⌈log₂ K⌉-bit fields and document ids as ⌈log₂ N⌉-bit fields or delta-coded runs, as in Omar Khattab's measurement. Keep residuals in fast-plaid's packing so the decode tables are identical.
3. **Co-locate per-document data.** Store each document's codes and residuals (and optionally inverse norms) in one row, so stages 4 and 5 read the same rows; the stage-5 residuals then often arrive with the stage-4 codes if rows are small enough to be read whole. For 50-word documents a row is 111 × (2.1 + 6) ≈ 0.9 KB at nbits = 1, so four documents share a 4 KiB page.
4. **Order documents for locality.** Candidates for one query are scattered by document id. Renumbering documents so that documents with similar tokens are adjacent (for example by clustering their mean vectors, or by their most frequent centroid) turns many page reads into fewer contiguous range reads. Since fast-plaid's document ids are only positions, this needs an id-mapping column.
5. **Consider IVF-only approximate scoring.** Stage 4 exists only to read codes of candidates. If the IVF posting for (centroid c, document d) is extended to carry the positions or simply the fact of membership, the approximate score restricted to probed centroids, Σ_i max_{c ∈ probed ∩ codes(d)} S[c, i], can be computed from the IVF lists alone, removing one dependent round trip. Unprobed centroids contribute nothing, which is close in spirit to the paper's pruned centroid interaction (low-scoring centroids ignored). The WARP engine (XTR-style retrieval with imputation of missing similarities; PyLate `pylate/indexes/warp.py`) goes further and stores residuals centroid-major, so a query reads only the probed cells' token records and needs a single dependent round trip after the centroid table. This is the most promising layout for httpvfs and should be evaluated against the faithful baseline.
6. **Smaller search parameters.** fast-plaid's defaults are tuned for a GPU and a large corpus. For a browser, `n_full_scores` of 256–1024 (64–256 documents decompressed) and `n_ivf_probe` of 2–4 are the natural starting points to sweep.
7. **Keep the normalization.** Whether decompressing or scoring with next-plaid's lookup tables, divide by the reconstructed norm; at nbits = 1 this is worth up to 0.17 NDCG@10.

### What Omar Khattab's measurement suggests

In Tom Aarsen's Sentence Transformers multi-vector training post, Omar Khattab measured fast-plaid on a 200,000-passage medical corpus (about 878 tokens per passage, about 176 M token vectors, 128 dimensions) at nbits = 1 with 17-bit centroid ids and 18-bit document ids instead of 64-bit integers, plus naive document-side token pruning:

| Configuration | Vectors kept | Index | NDCG@10 |
|---|---|---|---|
| float16 embeddings (exact search) | 100 % | 45 GB | 0.9139 (= 0.8984 + 0.0155) |
| 1-bit PLAID | 100 % | 3.37 GB | 0.8984 |
| 1-bit PLAID + pruning | 65 % | 2.23 GB | 0.8830 |
| 1-bit PLAID + pruning | 42 % | 1.45 GB | 0.8642 |

The 3.37 GB is about 19–20 bytes per vector: 16 B of residual, 2.1 B of code and about 2 B of IVF entry. This confirms that once integers are bit-packed, **residual bits dominate**, that nbits = 1 costs little quality at 128 dimensions (−0.016 NDCG@10), and that token pruning composes with quantization at a steeper quality cost. At 48 dimensions the balance shifts: the residual is only 6 B, so codes and IVF entries are about 40 % of the index and bit-packing them matters more, and a 1-bit residual carries less information per vector, so nbits = 2 deserves a comparison. next-plaid's warning that *sign-binarized* storage (not 1-bit residuals) degrades sharply at 48 dimensions (`next-plaid/colgrep/README.md:336-341`) is a separate scheme, but it is a hint that low-dimensional models are sensitive to aggressive quantization. For short documents such as ours, pruning has less to remove: the skiplist already drops about 10 % of LLM-paragraph tokens and none of the random-word tokens.
