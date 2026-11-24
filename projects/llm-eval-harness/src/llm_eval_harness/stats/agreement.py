"""Agreement statistics: Pearson and Spearman correlation.

Pure numpy (already a repo dependency); no scipy. Spearman uses
midranks so tied scores - common on a 1..5 judge scale - are handled
correctly (the naive ``D / sqrt(n(n^2-1))`` formula assumes no ties).
"""

from __future__ import annotations

import numpy as np


def _validated_pairs(xs: object, ys: object) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
        raise ValueError("xs and ys must be 1-D sequences of equal length")
    if x.size < 2:
        raise ValueError("need at least 2 paired observations")
    return x, y


def pearson(xs: object, ys: object) -> float:
    """Pearson product-moment correlation coefficient in [-1, 1]."""
    x, y = _validated_pairs(xs, ys)
    if x.std(ddof=1) == 0 or y.std(ddof=1) == 0:
        raise ValueError("zero variance: correlation undefined")
    return float(np.corrcoef(x, y)[0, 1])


def _midranks(values: np.ndarray) -> np.ndarray:
    """Midranks (1-based, ties averaged) of a 1-D array."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(1, len(values) + 1, dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(sorted_values):
        end = start
        while (
            end + 1 < len(sorted_values)
            and sorted_values[end + 1] == sorted_values[start]
        ):
            end += 1
        if end > start:
            mean = ranks[order[start : end + 1]].mean()
            ranks[order[start : end + 1]] = mean
        start = end + 1
    return ranks


def spearman(xs: object, ys: object) -> float:
    """Spearman rank correlation (Pearson on midranks)."""
    x, y = _validated_pairs(xs, ys)
    return pearson(_midranks(x), _midranks(y))
