"""Unit tests for the agreement statistics (hand-computed references)."""

from __future__ import annotations

import pytest
from llm_eval_harness.stats.agreement import pearson, spearman


def test_pearson_perfect() -> None:
    assert pearson([1, 2, 3, 4], [2, 4, 6, 8]) == pytest.approx(1.0)
    assert pearson([1, 2, 3, 4], [8, 6, 4, 2]) == pytest.approx(-1.0)


def test_pearson_known_zero() -> None:
    # dx = [-1.5, -0.5, 0.5, 1.5], dy = [0.5, -1.5, 1.5, -0.5] -> dot 0
    assert pearson([1, 2, 3, 4], [3, 1, 4, 2]) == pytest.approx(0.0)


def test_spearman_perfect_and_tied() -> None:
    assert spearman([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]) == pytest.approx(-1.0)
    # Tie case: x midranks [1, 2, 3.5, 3.5, 5], y midranks [2, 1, 3.5, 3.5, 5]
    # -> hand-computed Pearson on those ranks (mean 3.0 each): dot 8.5,
    # each sum of squared deviations 9.5 -> r = 8.5 / 9.5 = 17/19.
    got = spearman([1, 2, 3, 3, 5], [3, 1, 4, 4, 5])
    assert got == pytest.approx(8.5 / 9.5, abs=1e-9)
    # The naive no-ties D/sqrt(n(n^2-1)) formula gives 0.90 on this data,
    # not 0.895 - exactly what midranks exist to fix.


def test_agreement_errors() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        pearson([1], [2])
    with pytest.raises(ValueError, match="equal length"):
        pearson([1, 2], [1])
    with pytest.raises(ValueError, match="zero variance"):
        pearson([1, 1, 1], [1, 2, 3])
