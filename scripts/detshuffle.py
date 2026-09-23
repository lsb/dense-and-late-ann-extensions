"""Deterministic, portable shuffling.

A Fisher-Yates shuffle driven by SplitMix64. The algorithm is small enough to
reimplement exactly in any language (see RESEARCH_LOG.md), unlike GNU shuf or
Python's random module, whose outputs depend on the implementation version.
"""
MASK = (1 << 64) - 1


def splitmix64(seed):
    x = seed & MASK
    while True:
        x = (x + 0x9E3779B97F4A7C15) & MASK
        z = x
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
        yield z ^ (z >> 31)


def shuffled(items, seed):
    """Return a new list: Fisher-Yates from the end, j = r mod (i+1)."""
    out = list(items)
    rng = splitmix64(seed)
    for i in range(len(out) - 1, 0, -1):
        j = next(rng) % (i + 1)
        out[i], out[j] = out[j], out[i]
    return out
