#!/bin/sh
# Graph vs IVF-PQ on the real 1M MiniLM corpus (data/emb/words-1m.minilm.npy).
# About 1-1.5 h with 2 threads; needs ~6 GB of disk at peak. Run from anywhere.
set -e
cd "$(dirname "$0")/../.."
make -C ext/dense
B="python3 ext/dense/bench.py --data words-1m"
$B --sweep graphw --params "threads=2, verbose=1" --tag words1m-graph
rm -f ext/dense/build/words1m-graph.db
$B --sweep ivfw --params "layout=ivf, threads=2, ivf_centroids=pq, verbose=1" --tag words1m-ivf
$B --sweep ivfw --params "layout=ivf, threads=2, ivf_centroids=pq, nlist=1000" --tag words1m-ivf-n1000
rm -f ext/dense/build/words1m-ivf.db ext/dense/build/words1m-ivf-n1000.db
python3 ext/dense/compare.py words1m-graph:words1m-ivf:words1m-ivf-n1000
