# Ranked FTS5 queries over httpvfs

FTS5's `bm25()` makes ranked full-text search expensive over HTTP range requests: on the 1M-document corpus a single-word query reads about 690 pages, one round trip each (RESEARCH_LOG.md, "FTS5 baseline"). This document compares ways to avoid that, measured in rounds, bytes, simulated 4G/LTE latency and retrieval quality, and describes what the client now does.

**Result.** The client ranks with **`bm25c()`**, a new FTS5 ranking function in `ext/fts5rank`: the bm25 formula with every document at the average length, so it reads no per-document lengths. On words-1m a cold single-word query drops from 685 rounds and 2.9 MB (115 s on simulated 4G, 50 s on LTE) to 10 rounds and 39 KB (1.7 s and 0.7 s), and quality is unchanged. On the corpora here, whose documents have almost equal lengths, `bm25c` and `bm25` give identical scores. On an LLM corpus with variable-length documents (the untruncated paragraphs), `bm25c` is within 0.01 nDCG@10 of `bm25`. Exact bm25 is still available at low cost: `bm25-rerank` re-scores bm25c's top 50 with their stored lengths (its top 10 agrees with bm25's at 0.975–1.00), and `bm25-prefetch` fetches every matching length in one batch. `rank: 'bm25'` restores the old behaviour.

## Why bm25 is expensive

`bm25()` (SQLite's `fts5Bm25Function`) needs three things:

- the IDF of each phrase, from the phrase's doclist (`xQueryPhrase`), which the MATCH has read anyway;
- the average document length, from the averages record (`%_data` row 1);
- the length of the current document, from `xColumnSize`, which reads that document's `%_docsize` row.

The last one is the problem. `ORDER BY rank` computes the score of every matching document, so a query that matches *n* documents reads *n* `%_docsize` rows. On words-1m the rows of a word's ~670 documents lie on ~670 different pages of the 2,450-page table. The VFS is synchronous, so each page is its own round trip. At 10k documents the whole `%_docsize` table is only 25 pages, and readahead hides most of the cost.

## Options

| | Option | Reads per query (cold) | Exact bm25? | Verdict |
|---|---|---|---|---|
| — | `bm25` (FTS5 built-in) | one round per matching document (words-1m: 685 rounds, 2.9 MB) | yes | the baseline |
| (a) | **`bm25-prefetch`**: an unranked MATCH lists the rowids (doclist only), `httpvfs_warm` fetches their `%_docsize` rows by speculative batching (one round per uncached B-tree level), then `bm25()` runs on a warm cache | +1 doclist pass, +3 rounds; still one page per match (words-1m: 13 rounds but 2.7 MB, 5.7 s on 4g,h2, 19.7 s on 4g,h1) | yes | implemented; exact, but bytes grow with the number of matches |
| (b) | read the whole `%_docsize` once per session | words-1m: 77 rounds, 9.6 MB (23 s on 4G, 12 s on LTE) and a 10 MB cache; 10k: 8 rounds, 100–156 KB | yes | fine at 10k, too big at 1M; prefetching only the interior pages saves ≤ 2 rounds per lookup and none of the per-match leaf rounds |
| (c) | **`bm25c()`**: bm25 with every document at the average length (b = 0); no `%_docsize` reads | the MATCH's own doclist pages + 2–4 rounds (words-1m: 10 rounds, 39 KB) | no; exact when all documents have the same length | **default** |
| (c′) | **`bm25-rerank`**: bm25c's top 50, their 50 `%_docsize` rows in one batch, exact bm25 (`bm25dl()`) on those candidates | bm25c + 3 rounds + ≤ 50 pages (words-1m: 13 rounds, 244 KB) | top 10 = bm25 for 97.5–100 % of results | implemented; the "near-exact" switch |
| (c″) | lengths in a compact side table (1 byte per document, clustered by rowid) | 1 MB at 1M documents, 4,000 lengths per page; a 670-match query still touches ~230 of its 245 pages | yes, up to quantisation | not built: it only pays if prefetched whole (1 MB per session), and the rerank variant is cheaper; `bm25dl(fts, D)` accepts any length source if needed |
| (d) | FTS5 options | `columnsize=0` removes `%_docsize`, so `bm25()` re-tokenises each matching document from the content table (worse); `detail=column/none` shrink doclists but make every ranking function re-tokenise the matching documents | – | see "FTS5 options" below |
| (d′) | page size | rounds unchanged (one per match); bytes per match scale with the page size (64 KiB pages: ~10 MB per word query at 1M) | – | no help; smaller pages would only make (a) cheaper in bytes |
| (e) | top-k early termination | FTS5 doclists are in rowid order, so the top k by score needs the whole doclist; an impact-ordered side index would save doclist bytes only for very common terms (here a whole cold bm25c query, doclists included, reads 30–90 KB) | – | not worth it for these corpora |

## Measurements

Method (`bench/fts5_rank.py`, `bench/fts5_rank.mjs`):

- **Quality** is measured natively (Python `sqlite3` on `build/native`, `fts5rank.so`) on every query of each set, with k = 10 for nDCG@10 and MRR@10 and k = 100 for AUC. Queries are the matrix's: each word quoted, joined with OR (the client default). "top-10 = bm25" is the overlap of a method's top 10 with bm25's.
- **Costs** go end to end through the product path. The WASM build (Asyncify, Node 22), with `SearchDb.search` from `web/lib/search-core.mjs` and a 16 MiB block cache (the web client's default), runs against `netsim/rangeserver.py` (unshaped). Each query's request log is recorded and simulated with `netsim/simulate.py` on `4g,h2` and `lte,h2` (and h1).
  - *Cold*: the VFS and page caches are emptied before each query; the connection and its parsed schema stay.
  - *Warm*: one connection per method and query kind answers the kind's queries in order.
  - 50 cold and 200 warm queries per kind. Latencies are medians.
- **Databases**:
  - `build/matrix/words-10k.db` and `llm-10k.db`: the matrix databases, with all indexes, 4 KiB pages.
  - words-1m: FTS only, `tools/build_db.py words-1m --split-only --indexes fts`, 690 MiB (docs 485 MiB, FTS 205 MiB).
  - **llm-10k-full**: the same 10,000 LLM paragraphs untruncated (`data/llm/paragraphs-10k.jsonl`: 11–205 words, mean 103, sd 19), built only to test length normalisation. The corpus files proper have 50 words per document (words-*) or 11–60 words (llm-10k, sd 1.5).

### Quality (OR queries, all queries)

| Corpus | Query kind (n) | bm25 nDCG@10 / MRR / AUC@100 | bm25c | bm25-rerank | none (rowid order) | top 10 = bm25: bm25c / rerank |
|---|---|---|---|---|---|---|
| words-10k | word (1,000) | 0.9874 / 0.9910 / 1.0000 | 0.9874 / 0.9910 / 1.0000 | 0.9874 / 0.9910 / 1.0000 | 0.9874 / 0.9910 / 1.0000 | 1.000 / 1.000 |
| words-10k | known (1,000) | 1.0000 / 1.0000 / 1.0000 | 1.0000 / 1.0000 / 1.0000 | 1.0000 / 1.0000 / 1.0000 | 0.3080 / 0.2262 / 0.9991 | 1.000 / 1.000 |
| words-1m | word (1,000) | 0.9848 / 0.9940 / 0.5735 | 0.9848 / 0.9940 / 0.5735 | 0.9848 / 0.9940 / 0.5735 | 0.9836 / 0.9925 / 0.5735 | 1.000 / 1.000 |
| words-1m | known (1,000) | 1.0000 / 1.0000 / 1.0000 | 1.0000 / 1.0000 / 1.0000 | 1.0000 / 1.0000 / 1.0000 | 0.0035 / 0.0027 / 0.5264 | 1.000 / 1.000 |
| llm-10k | word (10,000) | 0.7450 / 0.7314 / 0.9021 | 0.7461 / 0.7329 / 0.9026 | 0.7450 / 0.7314 / 0.9026 | 0.7202 / 0.7021 / 0.9021 | 0.972 / 1.000 |
| llm-10k | llmq (10,000) | 0.1404 / 0.1290 / 0.6523 | 0.1408 / 0.1297 / 0.6526 | 0.1404 / 0.1290 / 0.6526 | 0.0166 / 0.0134 / 0.5513 | 0.738 / 0.994 |
| llm-10k-full | word (10,000) | 0.7842 / 0.7757 / 0.9096 | 0.7759 / 0.7660 / 0.9092 | 0.7842 / 0.7757 / 0.9093 | 0.7078 / 0.6872 / 0.9068 | 0.951 / 0.997 |
| llm-10k-full | llmq (10,000) | 0.2080 / 0.1941 / 0.6697 | 0.2079 / 0.1941 / 0.6690 | 0.2081 / 0.1941 / 0.6691 | 0.0143 / 0.0118 / 0.5394 | 0.660 / 0.975 |

`bm25-prefetch` returns exactly bm25's results and is not listed. The "top 10 = bm25" column is the overlap with bm25's top 10.

- **words-\*.** Every document has 50 tokens, so `bm25c` gives the same scores as `bm25` and all metrics are identical. At 1M the word queries have about 670 relevant documents each, so recall@10 is 0.015 and AUC@100 is 0.57 for every method.
- **llm-10k.** Documents have 11–60 tokens. `bm25c` is +0.001 nDCG on word queries and +0.0004 on llmq, which is noise; its top 10 differs from bm25's mostly by reordering near-ties.
- **llm-10k-full.** Documents have 11–205 words. Length normalisation now matters a little: `bm25c` loses 0.008 nDCG@10 on word queries and nothing on llmq. `bm25-rerank` recovers bm25 exactly (0.7842) with a 0.997 top-10 overlap.

The *none* rows (rowid order) are there for scale. On the words-* single-word queries rowid order is as good as bm25, because relevance there is binary (every document containing the word is relevant). On known-item OR queries it collapses (nDCG 0.003 at 1M), and on the LLM sets it loses 0.02–0.13.

**Caveat on query subsets.** Ties are broken by rowid, and in the LLM query sets query *i* is about document *i*. A subset of the first 1,000 queries therefore favours any method with many ties. On that subset `bm25c` (many ties) scores 0.787 against bm25's 0.757 on llm-10k word queries, and rowid order 0.781. Over all 10,000 queries (the table above) the effect averages out. The matrix uses the first 1,000, so compare FTS rows there with this in mind.

### Fetch costs (OR queries; rounds, KB, simulated median on 4g,h2 / lte,h2)

| Corpus | Query kind | Regime | bm25 | bm25-prefetch | bm25-rerank | bm25c | none |
|---|---|---|---|---|---|---|---|
| words-10k | known | cold | 22.6 r, 113 KB, 3.91 / 1.69 s | 12.8 r, 101 KB, 2.25 / 0.98 s | 12.8 r, 101 KB, 2.25 / 0.98 s | 10.8 r, 43 KB, 1.86 / 0.80 s | 9.5 r, 38 KB, 1.69 / 0.73 s |
| words-10k | known | warm | 1.8 r, 7 KB, 0.34 / 0.15 s | 1.8 r, 7 KB, 0.34 / 0.15 s | 1.8 r, 7 KB, 0.34 / 0.15 s | 1.8 r, 7 KB, 0.34 / 0.15 s | 1.8 r, 7 KB, 0.34 / 0.15 s |
| words-10k | word | cold | 15.3 r, 64 KB, 2.54 / 1.10 s | 9.6 r, 61 KB, 1.71 / 0.74 s | 9.6 r, 61 KB, 1.71 / 0.74 s | 7.6 r, 31 KB, 1.35 / 0.58 s | 6.0 r, 24 KB, 1.01 / 0.44 s |
| words-10k | word | warm | 0.9 r, 4 KB, 0.17 / 0.07 s | 0.9 r, 4 KB, 0.17 / 0.07 s | 0.9 r, 4 KB, 0.17 / 0.07 s | 0.8 r, 3 KB, 0.17 / 0.07 s | 0.8 r, 3 KB, 0.17 / 0.07 s |
| words-1m | known | cold | 1091.5 r, 7,641 KB, 190.95 / 82.98 s | 20.3 r, 6,134 KB, 9.60 / 5.63 s | 20.3 r, 241 KB, 3.64 / 1.61 s | 17.3 r, 69 KB, 2.96 / 1.28 s | 13.4 r, 54 KB, 2.20 / 0.95 s |
| words-1m | known | warm | 9.3 r, 39 KB, 1.02 / 0.45 s | 6.8 r, 47 KB, 1.04 / 0.46 s | 7.7 r, 38 KB, 1.20 / 0.52 s | 6.8 r, 27 KB, 1.02 / 0.44 s | 4.8 r, 19 KB, 0.68 / 0.29 s |
| words-1m | word | cold | 684.9 r, 2,957 KB, 114.87 / 49.50 s | 12.7 r, 2,750 KB, 5.66 / 3.03 s | 12.7 r, 244 KB, 2.39 / 1.08 s | 9.7 r, 39 KB, 1.69 / 0.73 s | 7.0 r, 28 KB, 1.18 / 0.51 s |
| words-1m | word | warm | 11.4 r, 49 KB, 0.51 / 0.22 s | 3.0 r, 50 KB, 0.52 / 0.23 s | 2.9 r, 17 KB, 0.51 / 0.22 s | 2.9 r, 14 KB, 0.51 / 0.22 s | 2.2 r, 9 KB, 0.51 / 0.22 s |
| llm-10k | llmq | cold | 20.5 r, 203 KB, 3.21 / 1.40 s | 18.6 r, 172 KB, 2.29 / 1.01 s | 15.6 r, 140 KB, 2.27 / 0.99 s | 13.6 r, 71 KB, 1.86 / 0.80 s | 9.5 r, 38 KB, 1.52 / 0.66 s |
| llm-10k | llmq | warm | 1.2 r, 6 KB, 0.17 / 0.07 s | 1.3 r, 6 KB, 0.17 / 0.07 s | 1.3 r, 6 KB, 0.17 / 0.07 s | 1.2 r, 6 KB, 0.17 / 0.07 s | 1.1 r, 4 KB, 0.17 / 0.07 s |
| llm-10k | word | cold | 9.7 r, 42 KB, 1.69 / 0.73 s | 9.1 r, 40 KB, 1.69 / 0.73 s | 9.1 r, 40 KB, 1.69 / 0.73 s | 7.4 r, 30 KB, 1.35 / 0.58 s | 6.0 r, 24 KB, 1.01 / 0.44 s |
| llm-10k | word | warm | 0.8 r, 3 KB, 0.17 / 0.07 s | 0.7 r, 3 KB, 0.17 / 0.07 s | 0.7 r, 3 KB, 0.17 / 0.07 s | 0.7 r, 3 KB, 0.17 / 0.07 s | 0.7 r, 3 KB, 0.17 / 0.07 s |
| llm-10k-full | llmq | cold | 22.0 r, 176 KB, 3.35 / 1.45 s | 20.6 r, 174 KB, 2.46 / 1.09 s | 16.9 r, 159 KB, 2.45 / 1.07 s | 14.9 r, 87 KB, 2.03 / 0.88 s | 10.3 r, 41 KB, 1.69 / 0.73 s |
| llm-10k-full | llmq | warm | 1.6 r, 8 KB, 0.17 / 0.08 s | 1.7 r, 8 KB, 0.17 / 0.08 s | 1.6 r, 8 KB, 0.17 / 0.08 s | 1.6 r, 8 KB, 0.17 / 0.07 s | 1.3 r, 5 KB, 0.17 / 0.07 s |
| llm-10k-full | word | cold | 10.3 r, 43 KB, 1.69 / 0.73 s | 9.3 r, 43 KB, 1.69 / 0.73 s | 9.3 r, 42 KB, 1.69 / 0.73 s | 7.5 r, 30 KB, 1.35 / 0.58 s | 6.0 r, 24 KB, 1.01 / 0.44 s |
| llm-10k-full | word | warm | 0.9 r, 4 KB, 0.17 / 0.07 s | 0.8 r, 4 KB, 0.17 / 0.07 s | 0.8 r, 4 KB, 0.17 / 0.07 s | 0.8 r, 3 KB, 0.17 / 0.07 s | 0.8 r, 3 KB, 0.17 / 0.07 s |

Simulated medians of the recorded request logs, `4g,h2` / `lte,h2`, 50 cold and 200 warm queries per kind. On h1 (six connections) only `bm25-prefetch` at 1M differs much: its ~670 page requests take 19.7 s on `4g,h1` against 5.7 s on h2.

Reading the whole `fts_docsize` table once (option b), per session:

| Corpus | Rounds | Bytes | 4g,h2 / lte,h2 |
|---|---|---|---|
| words-10k, llm-10k (the matrix databases) | 8 | 156 KB | 1.5 / 0.7 s |
| llm-10k-full | 8 | 100 KB | 1.4 / 0.7 s |
| words-1m | 77 | 9.6 MB | 23.0 / 12.4 s |

**Reading the tables.**

- At 10k all methods are cheap. `bm25` costs 2–9 more rounds than `bm25c` cold, 0.3–1.4 s on 4G. Warm sessions cache the 25-page `fts_docsize` table, and every method costs the same.
- At 1M `bm25` is unusable over the network: 685 rounds and 115 s cold on 4G for a word query, and 1,092 rounds and 191 s for a three-word OR query. Warm, a session gradually caches the 10 MB table (11 rounds on average after 200 queries, p95 24).
- `bm25c` costs 2–4 rounds more than an unranked query and 1/70 of bm25's bytes.
- `bm25-rerank` adds 3 rounds and ~200 KB to bm25c.
- `bm25-prefetch` has few rounds but still transfers one page per match (2.7–6.1 MB at 1M).
- The remaining cold cost of every method, 7–17 rounds, is FTS5's own chain of dependent lookups (see "Open points").

### End-to-end validation

The cost numbers above already come from the product path: `SearchDb.search` in the WASM build, in Node, against `netsim/rangeserver.py`. Two further checks:

- **Same results as native.** For all 10,000 recorded queries on words-1m and llm-10k-full (every method, cold and warm), the WASM result ids equal those of the same SQL run natively.
- **Real shaped network.** `bench/fts5_rank.py cost … --preset P` runs the same queries against the range server shaped with `4g,h2` or `lte,h2`, on 10 cold word queries on words-1m (`bm25` 2 queries):

| Method | Real median | Simulated from the same logs |
|---|---|---|
| `none`, 4g,h2 | 1.20 s | 1.18 s |
| `bm25c`, 4g,h2 | 1.71 s | 1.69 s |
| `bm25-rerank`, 4g,h2 | 2.43 s | 2.40 s |
| `bm25-prefetch`, 4g,h2 | 5.05 s | 5.65 s |
| `bm25c`, lte,h2 | 0.71 s | 0.70 s |
| `bm25`, lte,h2 | 51.1 s | 49.7 s |

## Implementation

**`ext/fts5rank/fts5rank.c`** (entry point `sqlite3_fts5rank_init`; compiled into the WASM builds and the native CLI like every `ext/*` directory; loadable `build/native/ext/fts5rank.so`) registers two FTS5 auxiliary functions through `fts5_api.xCreateFunction`:

- `bm25c(fts [, w0, …])`: bm25 with the average length. It uses the same IDF (including its 1e-6 floor), k1 = 1.2 and column weights as `bm25()`, and returns −score. Use it as `WHERE fts MATCH ?1 AND rank MATCH 'bm25c()' ORDER BY rank`.
- `bm25dl(fts, D [, w0, …])`: bm25 with a caller-supplied length D, either a number or a `%_docsize` blob (its varints are summed). NULL means the average length. With `D = (SELECT sz FROM fts_docsize WHERE id = fts.rowid)` it equals `bm25()` bit for bit (tested).

**`web/lib/search-core.mjs`**: FTS searches take `params.rank` (`FTS_RANKS`):

| `rank` | What runs |
|---|---|
| `bm25c` (default) | one query with `rank MATCH 'bm25c()'` |
| `bm25` | FTS5's `bm25()`, as before |
| `bm25-prefetch` | `SELECT rowid … MATCH ?1 LIMIT cap+1`, then `httpvfs_warm('SELECT sz FROM <fts>_docsize WHERE id = ?', ids)` (skipped above `prefetchCap`, 4,000, which would overflow the block cache), then `bm25()` |
| `bm25-rerank` | bm25c top `rerankDepth` (50), `httpvfs_warm` of their `%_docsize` rows, then `SELECT … CASE WHEN rowid IN (…) THEN bm25dl(fts, (SELECT sz …)) END AS score … WHERE score IS NOT NULL ORDER BY score LIMIT k`. The candidate test is in the select list, not in WHERE, so that FTS5 does not get one `xFilter` per rowid; that form was 15 ms per query natively. |
| `none` | rowid order |

A database opened by a SQLite build without the functions (`no such function: bm25c`) falls back to `bm25` once and remembers it. `search({match})` takes a ready MATCH expression, which the benchmark uses. The demo page has a "rank" selector, and `bench/matrix_config.json` has an `fts-bm25c-or` configuration for future matrix runs.

The Python CLI (`dense-late-ann search`) reads local files, where `bm25()` costs 1.3 ms per query even at 1M (bm25c 0.3 ms), so it is unchanged. The Python package does not ship `fts5rank.so`.

**Tests.** `web/test/search-core.test.mjs` is part of `make test-node`. It builds a 30,000-document test database with variable lengths and checks:

- `bm25dl` with the stored length equals `bm25` exactly;
- every method returns what its SQL returns natively;
- `bm25-prefetch` equals `bm25`, scores included;
- `bm25-rerank` equals bm25 restricted to bm25c's top 50;
- on a word in about 40 documents, cold `bm25` needs at least 15 more rounds than `bm25c`, and prefetch and rerank at most 4 more;
- the fallback without `bm25c` works.

`web/test/native_search.py` loads `fts5rank.so`, so the browser test's native comparison covers the new default.

## FTS5 options (option d)

`bench/fts5_options.py` rebuilds the FTS5 table with each option set, on the first 300 queries per kind, OR form. It counts the distinct pages a cold connection reads, natively (APSW VFS shim, no readahead). The nDCG values are on this subset, so they are only for comparing options.

| Corpus | Option | FTS size | bm25 pages: word / multi-word | bm25c pages: word / multi-word |
|---|---|---|---|---|
| llm-10k-full | default (`detail=full`, `columnsize=1`) | 3,088 KB | 11.9 / 42.8 | 8.7 / 20.6 |
| | `columnsize=0` | 2,992 KB | 22.1 / 1,005 | 8.7 / 20.6 |
| | `detail=column` | 2,848 KB | 25.0 / 1,006 | 21.8 / 984 |
| | `detail=none` | 1,252 KB | 23.5 / 1,000 | 20.4 / 978 |
| words-10k | default | 2,488 KB | 16.3 / 26.6 | 8.6 / 12.0 |
| | `columnsize=0` | 2,396 KB | 19.4 / 34.7 | 8.6 / 12.0 |
| | `detail=column` | 2,488 KB | 27.2 / 49.2 | 19.4 / 34.7 |
| | `detail=none` | 1,496 KB | 27.1 / 48.7 | 19.4 / 34.2 |

- `columnsize=0` drops `%_docsize` (only 3 % of the index). `bm25()` then tokenises each matching document from the content table instead, which is worse: 1,005 pages for an LLM OR query. `bm25c` does not care.
- `detail=column` and `detail=none` shrink the doclists (`detail=none` halves the index). But FTS5 then finds term instances for `xInstCount`/`xInst` by re-tokenising the matching documents, so *any* ranking function reads every matching document's text: ~1,000 pages. Phrase queries (such as `"what's"`, two tokens) are also rejected under `detail=column`/`none` (10 errors).
- A prefix index (`prefix=…`) only matters for `word*` queries, which the client does not issue.
- Page size changes bytes, not rounds. `bm25` needs one page per match whatever the page size, so larger pages make it worse (at 1M, 64 KiB pages would mean ~150 × 64 KiB ≈ 10 MB per word query), and smaller pages would only make `bm25-prefetch` cheaper in bytes.

So none of the table options helps ranking over httpvfs; the default (`detail=full`) with `bm25c` is the best combination.

## Open points

- The cold FTS5 query itself is a chain of 6–10 dependent reads: page 1, `%_config`, the structure record, the `%_idx` lookup, then the leaves. The doclists are not the cost. All of these are cached in a warm session, but the first query could batch the fixed ones: the structure pages could be prefetched at open, as the dense and late extensions load their static data.
- The quality comparison uses corpora whose document lengths vary little. On real web text with long and short documents, length normalisation matters more, and `bm25-rerank` would be the safer default. It costs about 3 extra rounds and up to 50 pages.
