"""Agreement statistics: Pearson, Spearman, and Cohen's family of kappa.

Pure numpy (already a repo dependency); no scipy. Spearman uses
midranks so tied scores - common on a 1..5 judge scale - are handled
correctly (the naive ``D / sqrt(n(n^2-1))`` formula assumes no ties).
Kappa is by hand from the confusion matrix: unweighted (2x2 up to kxk)
and quadratic-weighted for the ordered 5x5 judge scale.
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


def _contingency(a: object, b: object) -> tuple[np.ndarray, list]:
    """Confusion matrix (rows = a, cols = b) over the observed levels."""
    if len(a) != len(b):
        raise ValueError("a and b must have equal length")
    n = len(a)
    if n < 1:
        raise ValueError("need at least 1 paired observation")
    levels = sorted(set(a) | set(b))
    pos = {v: i for i, v in enumerate(levels)}
    obs = np.zeros((len(levels), len(levels)), dtype=float)
    for x, y in zip(a, b, strict=True):
        if x not in pos or y not in pos:
            raise ValueError("unobserved level outside the category set")
        obs[pos[x], pos[y]] += 1.0
    return obs, levels


def cohen_kappa(a: object, b: object) -> float:
    """Cohen's kappa (2x2 up to kxk) for categorical agreement.

    (po - pe) / (1 - pe) from the confusion matrix; raises on the
    degenerate case where chance agreement saturates (1 - pe = 0).
    """
    obs, levels = _contingency(a, b)
    if len(levels) < 2:
        raise ValueError("degenerate margins: kappa undefined")
    n = obs.sum()
    po = obs.diagonal().sum() / n
    pe = (obs.sum(axis=1) @ obs.sum(axis=0)) / (n * n)
    if abs(1.0 - pe) < 1e-12:
        raise ValueError("degenerate margins: kappa undefined")
    return float((po - pe) / (1.0 - pe))


def quadratic_weighted_kappa(a: object, b: object, k: int = 5) -> float:
    """Quadratic weighted kappa on an ordered 1..k scale.

    1 - sum(w*O) / sum(w*E) with weights (i - j)^2 / (k - 1)^2; raises
    on the degenerate case where the expected weighted sum is 0.
    """
    if not (isinstance(k, int) and k >= 2):
        raise ValueError("k must be an integer >= 2")
    levels = list(range(1, k + 1))
    pos = {v: i for i, v in enumerate(levels)}
    if len(a) != len(b) or len(a) < 1:
        raise ValueError("a and b must be equal-length sequences, non-empty")
    obs = np.zeros((k, k), dtype=float)
    for x, y in zip(a, b, strict=True):
        if x not in pos or y not in pos:
            raise ValueError("scores must lie within 1..k")
        obs[pos[x], pos[y]] += 1.0
    n = obs.sum()
    i = np.arange(k)
    w = ((i[:, None] - i[None, :]) / (k - 1)) ** 2
    expected = np.outer(obs.sum(axis=1), obs.sum(axis=0)) / n
    denominator = float((w * expected).sum())
    if denominator < 1e-12:
        raise ValueError("degenerate margins: weighted kappa undefined")
    return float(1.0 - float((w * obs).sum()) / denominator)
