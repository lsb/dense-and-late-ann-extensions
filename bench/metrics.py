"""Retrieval quality metrics for ranked result lists.

All functions take ``ranked`` (list of doc ids, best first, as returned by a
system — possibly truncated to k) and ``relevant`` (set of doc ids).

AUC is the ROC area over the whole collection of ``n_docs`` documents: the
probability that a random relevant document is ranked above a random
non-relevant one. Documents a system did not return are treated as tied
below every returned document (a tie counts one half).
"""
import math


def recall_at(ranked, relevant, k):
    if not relevant:
        return float("nan")
    return len(set(ranked[:k]) & relevant) / len(relevant)


def success_at(ranked, relevant, k):
    return float(any(d in relevant for d in ranked[:k]))


def mrr_at(ranked, relevant, k):
    for i, d in enumerate(ranked[:k]):
        if d in relevant:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at(ranked, relevant, k):
    dcg = sum(1.0 / math.log2(i + 2) for i, d in enumerate(ranked[:k]) if d in relevant)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(relevant))))
    return dcg / ideal if ideal else float("nan")


def auc(ranked, relevant, n_docs):
    n_rel = len(relevant)
    n_non = n_docs - n_rel
    if n_rel == 0 or n_non == 0:
        return float("nan")
    # A returned relevant doc beats every non-relevant doc not ranked above it.
    wins = 0.0
    non_above = 0
    returned_rel = 0
    for d in ranked:
        if d in relevant:
            wins += n_non - non_above
            returned_rel += 1
        else:
            non_above += 1
    unreturned_rel = n_rel - returned_rel
    unreturned_non = n_non - non_above
    wins += unreturned_rel * unreturned_non * 0.5
    return wins / (n_rel * n_non)


def summarize(rows):
    """Mean of each numeric key over a list of dicts, ignoring NaN."""
    keys = [k for k, v in rows[0].items() if isinstance(v, (int, float))]
    out = {}
    for k in keys:
        vals = [r[k] for r in rows if not (isinstance(r[k], float) and math.isnan(r[k]))]
        out[k] = sum(vals) / len(vals) if vals else float("nan")
    return out


def percentile(xs, p):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    i = (len(xs) - 1) * p / 100
    lo, hi = math.floor(i), math.ceil(i)
    return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)
