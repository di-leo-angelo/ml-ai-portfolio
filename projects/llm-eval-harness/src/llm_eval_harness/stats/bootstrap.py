"""Bootstrap confidence intervals (percentile method).

Pure: numpy in, JSON-safe dict out, no I/O, no scipy. Deterministic
under ``seed`` (default 1337, overridable); ``b`` resamples default to
10000 per PLAN.md section 6. Three variants:

- ``bootstrap_ci`` - plain: resample task-level values with
  replacement (overall means / rates);
- ``stratified_bootstrap_ci`` - resample within strata (e.g. subject),
  preserving each stratum's size (per-stratum tables);
- ``paired_bootstrap_ci`` - resample paired indices, CI on the mean of
  the per-task differences (condition comparisons).
"""

from __future__ import annotations

import numpy as np

B_DEFAULT = 10_000
SEED_DEFAULT = 1337
ALPHA_DEFAULT = 0.05


def _as_finite_1d(values: object, name: str = "values") -> np.ndarray:
    x = np.asarray(values, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"{name} must be a 1-D sequence")
    if x.size < 2:
        raise ValueError(f"{name} needs at least 2 observations")
    if not np.isfinite(x).all():
        raise ValueError(f"{name} must contain only finite values")
    return x


def _check_b(b: int) -> None:
    if b < 2:
        raise ValueError("b (number of resamples) must be >= 2")


def percentile_ci(samples: object, alpha: float = ALPHA_DEFAULT) -> tuple[float, float]:
    """Percentile CI of a 1-D sample of bootstrap replicates."""
    s = np.asarray(samples, dtype=float)
    if s.ndim != 1 or s.size < 2:
        raise ValueError("samples must be a 1-D sequence of replicates")
    lo, hi = np.percentile(s, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    return float(lo), float(hi)


def bootstrap_ci(
    values: object,
    b: int = B_DEFAULT,
    seed: int = SEED_DEFAULT,
    alpha: float = ALPHA_DEFAULT,
) -> dict:
    """Plain bootstrap: percentile CI on the mean of ``values``."""
    x = _as_finite_1d(values)
    _check_b(b)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(b, x.size))
    samples = x[idx].mean(axis=1)
    lo, hi = percentile_ci(samples, alpha)
    return {
        "statistic": "mean",
        "value": float(x.mean()),
        "ci": [lo, hi],
        "n": int(x.size),
        "b": int(b),
        "seed": int(seed),
        "alpha": float(alpha),
    }


def stratified_bootstrap_ci(
    values: object,
    strata: object,
    b: int = B_DEFAULT,
    seed: int = SEED_DEFAULT,
    alpha: float = ALPHA_DEFAULT,
) -> dict:
    """Stratified bootstrap: resample within each stratum, keeping sizes."""
    x = _as_finite_1d(values)
    s = np.asarray(strata)
    if s.ndim != 1 or s.shape[0] != x.size:
        raise ValueError("strata must be a 1-D sequence aligned with values")
    _check_b(b)
    try:
        labels = sorted(set(s.tolist()))
        parts = []
        rng = np.random.default_rng(seed)
        for label in labels:
            idx0 = np.nonzero(s == label)[0]
            drawn = rng.integers(0, idx0.size, size=(b, idx0.size))
            # size-weighted so the replicate is the POOLED mean, not the
            # mean of the stratum means
            parts.append(x[idx0][drawn].mean(axis=1) * (idx0.size / x.size))
        samples = np.sum(np.stack(parts), axis=0)
    except TypeError as exc:  # mixed, non-comparable stratum labels
        raise ValueError("strata labels must be comparable (same type)") from exc
    lo, hi = percentile_ci(samples, alpha)
    return {
        "statistic": "mean",
        "value": float(x.mean()),
        "ci": [lo, hi],
        "n": int(x.size),
        "strata": {str(label): int(sum(s == label)) for label in labels},
        "b": int(b),
        "seed": int(seed),
        "alpha": float(alpha),
    }


def paired_bootstrap_ci(
    x: object,
    y: object,
    b: int = B_DEFAULT,
    seed: int = SEED_DEFAULT,
    alpha: float = ALPHA_DEFAULT,
) -> dict:
    """Paired bootstrap: resample pair indices, CI on the mean of (x - y)."""
    xa = _as_finite_1d(x, "x")
    ya = _as_finite_1d(y, "y")
    if xa.shape != ya.shape:
        raise ValueError("x and y must have equal length")
    _check_b(b)
    diff = xa - ya
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diff.size, size=(b, diff.size))
    samples = diff[idx].mean(axis=1)
    lo, hi = percentile_ci(samples, alpha)
    return {
        "statistic": "mean_diff",
        "value": float(diff.mean()),
        "ci": [lo, hi],
        "n": int(diff.size),
        "b": int(b),
        "seed": int(seed),
        "alpha": float(alpha),
    }
