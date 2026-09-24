# dense_ann: notes

A SQLite virtual table for approximate nearest-neighbour search over dense embeddings (all-MiniLM-L6-v2, 384 dimensions, cosine), designed for read-only databases fetched lazily over HTTP range requests. The whole query path is single-threaded C99 with no dependencies beyond SQLite and libm; pthreads are used only to build the index natively (`-DDENSE_ANN_THREADS`).

(RESULTS_PLACEHOLDER)
