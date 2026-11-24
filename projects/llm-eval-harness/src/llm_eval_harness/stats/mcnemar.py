"""McNemar's test for paired binary outcomes, exact (no scipy).

Cells: ``b`` = (A pass, B fail), ``c`` = (A fail, B pass) - the
discordant pairs. Under H0 each discordant pair is a fair coin flip,
so B ~ Binomial(n = b + c, p = 0.5).

- Exact two-sided p: ``2 * P(X <= min(b, c))`` capped at 1.
- CI: Clopper-Pearson on the flip probability p, mapped to the
  difference scale ``(b - c) / (b + c) = 2p - 1``; inverted exactly by
  bisection on the binomial CDF built from ``math.comb`` (the CDF sum
  is truncated to its smaller tail, so it stays exact for moderate
  n - thousands of discordant pairs are comfortably in range).
- ``b + c == 0``: no discordant pairs - p = 1.0, no CI.
"""

from __future__ import annotations

import math

ALPHA_DEFAULT = 0.05
_BISECTIONS = 200


def _binom_cdf_leq(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Binomial(n, p), summed on the smaller tail."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    if k > n // 2:  # P_p(X <= k) = 1 - P_{1-p}(X <= n - 1 - k)
        q = 1.0 - p
        return 1.0 - _binom_cdf_leq(n - 1 - k, n, q)
    return sum(math.comb(n, i) * p**i * (1.0 - p) ** (n - i) for i in range(k + 1))


def _cp_solve(cdf_target: float, k: int, n: int) -> float:
    """Solve _binom_cdf_leq(k, n, p) = cdf_target for p in [0, 1].

    For 0 <= k < n the CDF is continuous and strictly DEcreasing in p
    (larger p shifts the binomial up), so bisection moves the upper end
    down whenever the CDF is still below the target.
    """
    lo, hi = 0.0, 1.0
    for _ in range(_BISECTIONS):
        mid = (lo + hi) / 2.0
        if _binom_cdf_leq(k, n, mid) < cdf_target:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2.0


def mcnemar_pvalue(b: int, c: int) -> float:
    """Exact two-sided McNemar p from the discordant cell counts."""
    if b < 0 or c < 0:
        raise ValueError("cell counts must be non-negative")
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2.0 * _binom_cdf_leq(min(b, c), n, 0.5))


def mcnemar_ci(
    b: int, c: int, alpha: float = ALPHA_DEFAULT
) -> tuple[float, float] | None:
    """Exact Clopper-Pearson CI on (b - c) / (b + c); None if b + c == 0.

    Lower: the p with P_p(X >= b) = alpha/2, i.e. P_p(X <= b - 1) = 1 - alpha/2
    (0 when b = 0). Upper: the p with P_p(X <= b) = alpha/2 (1 when c = 0).
    """
    if b < 0 or c < 0:
        raise ValueError("cell counts must be non-negative")
    n = b + c
    if n == 0:
        return None
    lo_p = 0.0 if b == 0 else _cp_solve(1.0 - alpha / 2.0, b - 1, n)
    hi_p = 1.0 if c == 0 else _cp_solve(alpha / 2.0, b, n)
    return (2.0 * lo_p - 1.0, 2.0 * hi_p - 1.0)


def mcnemar(a: object, b: object) -> dict:
    """Paired binary outcomes (equal-length 0/1 or bool sequences).

    Returns the discordant cells, the exact two-sided p, the (b - c) /
    (b + c) difference with its exact CI, and a note when there are no
    discordant pairs.
    """
    if len(a) != len(b):
        raise ValueError("a and b must have equal length")
    if len(a) < 1:
        raise ValueError("need at least 1 paired outcome")
    for v in list(a) + list(b):
        if v not in (0, 1):
            raise ValueError("outcomes must be 0 or 1")
    b_cell = sum(1 for x, y in zip(a, b, strict=True) if x == 1 and y == 0)
    c_cell = sum(1 for x, y in zip(a, b, strict=True) if x == 0 and y == 1)
    discordant = b_cell + c_cell
    return {
        "n_pairs": int(len(a)),
        "b": b_cell,  # A pass / B fail
        "c": c_cell,  # A fail / B pass
        "discordant": discordant,
        "diff": (b_cell - c_cell) / discordant if discordant else None,
        "p": mcnemar_pvalue(b_cell, c_cell),
        "ci": list(mcnemar_ci(b_cell, c_cell)) if discordant else None,
        "note": "no discordant pairs" if discordant == 0 else None,
    }
