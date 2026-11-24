"""Unit tests for the M3 statistics layer (hand-computed references).

Covers bootstrap CIs (percentile method, plain / stratified / paired),
exact McNemar p-values and Clopper-Pearson CIs, Cohen's kappa and
quadratic weighted kappa, and Spearman on tied ranks.
"""

from __future__ import annotations

import numpy as np
import pytest
from llm_eval_harness.stats.agreement import (
    cohen_kappa,
    quadratic_weighted_kappa,
    spearman,
)
from llm_eval_harness.stats.bootstrap import (
    B_DEFAULT,
    SEED_DEFAULT,
    bootstrap_ci,
    paired_bootstrap_ci,
    percentile_ci,
    stratified_bootstrap_ci,
)
from llm_eval_harness.stats.mcnemar import mcnemar, mcnemar_ci, mcnemar_pvalue

# --- bootstrap: percentile method ------------------------------------------


def test_percentile_ci_hand_computed() -> None:
    # 101 samples 1..101: numpy's linear-interpolated percentiles land
    # exactly halfway: 2.5% -> position 2.5 -> 3.5; 97.5% -> 98.5.
    assert percentile_ci(list(range(1, 102))) == pytest.approx((3.5, 98.5))
    assert percentile_ci([1.0, 2.0, 3.0], alpha=0.0) == pytest.approx((1.0, 3.0))


def test_bootstrap_deterministic_and_defaults() -> None:
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    a = bootstrap_ci(data, b=200, seed=7)
    b = bootstrap_ci(data, b=200, seed=7)
    c = bootstrap_ci(data, b=200, seed=8)
    assert a == b
    assert a["ci"] != c["ci"]
    assert a["value"] == pytest.approx(3.0)
    d = bootstrap_ci(data)
    assert d["b"] == B_DEFAULT == 10_000
    assert d["seed"] == SEED_DEFAULT == 1337


def test_bootstrap_constant_collapses() -> None:
    res = bootstrap_ci([3.0, 3.0, 3.0], b=50)
    assert res["ci"] == [3.0, 3.0]


def test_bootstrap_coverage_known_distribution() -> None:
    # The percentile CI of a N(5, 1) sample should contain the true mean
    # in ~95% of trials; 50 deterministic trials, loose bound.
    hits = 0
    for trial in range(50):
        rng = np.random.default_rng(10_000 + trial)
        sample = 5.0 + rng.standard_normal(40)
        res = bootstrap_ci(sample, b=500, seed=trial)
        if res["ci"][0] <= 5.0 <= res["ci"][1]:
            hits += 1
    assert hits >= 42


def test_bootstrap_stratified_hand_computed() -> None:
    # Constant within strata: every replicate is (3*1 + 2*3) / 5 = 1.8.
    res = stratified_bootstrap_ci([1, 1, 1, 3, 3], ["a", "a", "a", "b", "b"], b=50)
    assert res["ci"] == pytest.approx([1.8, 1.8])
    assert res["strata"] == {"a": 3, "b": 2}


def test_bootstrap_stratified_preserves_sizes() -> None:
    # Strata {1, 2} and {10, 12}: replicate means are (s_a + s_b) / 2 with
    # s_a in {1, 1.5, 2}, s_b in {10, 11, 12} -> 7 possible values.
    res = stratified_bootstrap_ci([1, 2, 10, 12], ["a", "a", "b", "b"], b=300)
    allowed = {5.5, 5.75, 6.0, 6.25, 6.5, 6.75, 7.0}
    lo, hi = res["ci"]
    assert lo <= 6.25 <= hi
    assert any(abs(v - lo) < 1e-9 for v in allowed)
    assert any(abs(v - hi) < 1e-9 for v in allowed)


def test_paired_bootstrap_constant_difference() -> None:
    x = [2.0, 3.0, 4.0, 5.0, 6.0]
    y = [0.0, 1.0, 2.0, 3.0, 4.0]
    res = paired_bootstrap_ci(x, y, b=50)
    assert res["value"] == pytest.approx(2.0)
    assert res["ci"] == [2.0, 2.0]


def test_paired_bootstrap_small() -> None:
    # Diffs [-1, 1]: replicate means in {-1, 0, 1}.
    res = paired_bootstrap_ci([1, 2], [2, 1], b=100)
    assert res["value"] == pytest.approx(0.0)
    assert res["ci"][0] <= 0.0 <= res["ci"][1]
    # Replicate means of diffs [-1, 1] are exactly {-1, 0, 1}.
    for v in res["ci"]:
        assert v in (-1.0, 0.0, 1.0)


def test_bootstrap_errors() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        bootstrap_ci([1.0])
    with pytest.raises(ValueError, match="finite"):
        bootstrap_ci([1.0, float("nan")])
    with pytest.raises(ValueError, match="must be >= 2"):
        bootstrap_ci([1.0, 2.0], b=1)
    with pytest.raises(ValueError, match="equal length"):
        paired_bootstrap_ci([1.0, 2.0, 3.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="aligned"):
        stratified_bootstrap_ci([1.0, 2.0], ["a"])
    with pytest.raises(ValueError, match="1-D"):
        bootstrap_ci([[1.0, 2.0]])


# --- McNemar ----------------------------------------------------------------


def test_mcnemar_pvalue_hand_computed() -> None:
    # 2 * P(X <= min(b, c)) under Binomial(b + c, 0.5), capped at 1.
    assert mcnemar_pvalue(2, 0) == pytest.approx(0.5)  # 2 * 0.25
    assert mcnemar_pvalue(3, 1) == pytest.approx(0.625)  # 2 * 5/16
    assert mcnemar_pvalue(5, 5) == pytest.approx(1.0)
    assert mcnemar_pvalue(0, 0) == pytest.approx(1.0)
    assert mcnemar_pvalue(3, 1) == mcnemar_pvalue(1, 3)


def test_mcnemar_ci_hand_computed() -> None:
    # b = 2, c = 0: lower bound solves p^2 = 0.025 -> 2*sqrt(0.025) - 1;
    # c = 0 pins the upper bound at 1.
    lo, hi = mcnemar_ci(2, 0)
    assert lo == pytest.approx(2.0 * 0.025**0.5 - 1.0, abs=1e-9)
    assert hi == pytest.approx(1.0)
    assert mcnemar_ci(0, 0) is None
    # Symmetry: flipping the cells negates the interval.
    a_lo, a_hi = mcnemar_ci(3, 1)
    b_lo, b_hi = mcnemar_ci(1, 3)
    assert a_lo == pytest.approx(-b_hi, abs=1e-9)
    assert a_hi == pytest.approx(-b_lo, abs=1e-9)
    assert a_lo < 0.5 < a_hi  # the difference (3-1)/4 = 0.5 is covered


def test_mcnemar_from_pairs() -> None:
    # Pairs: (1,1) (1,0) (0,0) (0,0) (1,1) (0,1) (1,0) -> b = 2, c = 1.
    a = [1, 1, 0, 0, 1, 0, 1]
    b = [1, 0, 0, 0, 1, 1, 0]
    res = mcnemar(a, b)
    assert res["n_pairs"] == 7
    assert res["b"] == 2 and res["c"] == 1
    assert res["diff"] == pytest.approx(1.0 / 3.0)
    assert res["p"] == pytest.approx(1.0)  # 2 * 4/8 capped
    # Upper bound: Beta(4, 1) CDF is p^3 -> 0.975^(1/3), exact.
    assert res["ci"][1] == pytest.approx(2.0 * 0.975 ** (1.0 / 3.0) - 1.0, abs=1e-9)
    # Lower bound: solve 3p^2 - 2p^3 = 0.025 (bisection by hand) -> p ~ 0.0943.
    assert res["ci"][0] == pytest.approx(2.0 * 0.0943 - 1.0, abs=2e-4)
    assert res["ci"][0] < res["diff"] < res["ci"][1]
    assert res["note"] is None


def test_mcnemar_no_discordant_pairs() -> None:
    res = mcnemar([1, 1, 0, 0], [1, 1, 0, 0])
    assert res["p"] == 1.0
    assert res["ci"] is None
    assert res["diff"] is None
    assert res["note"] == "no discordant pairs"


def test_mcnemar_errors() -> None:
    with pytest.raises(ValueError, match="equal length"):
        mcnemar([1, 0], [1])
    with pytest.raises(ValueError, match="0 or 1"):
        mcnemar([1, 2], [0, 1])
    with pytest.raises(ValueError, match="non-negative"):
        mcnemar_pvalue(-1, 0)
    with pytest.raises(ValueError, match="at least 1"):
        mcnemar([], [])


# --- kappa -------------------------------------------------------------------


def test_cohen_kappa_hand_computed() -> None:
    # a = [1, 1, 0, 0, 1, 0], b = [1, 0, 0, 0, 1, 1]:
    # po = 4/6 = 2/3; marginals (3, 3) both ways -> pe = (9 + 9) / 36 = 1/2
    # -> kappa = (2/3 - 1/2) / (1/2) = 1/3.
    assert cohen_kappa([1, 1, 0, 0, 1, 0], [1, 0, 0, 0, 1, 1]) == pytest.approx(
        1.0 / 3.0
    )


def test_cohen_kappa_perfect_and_reversed() -> None:
    assert cohen_kappa([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) == pytest.approx(1.0)
    # Reversed 4 levels: no pair agrees (po = 0), marginals (1,1,1,1) both
    # ways -> pe = 4/16 = 1/4 -> kappa = (0 - 1/4) / (3/4) = -1/3. (An odd
    # reversal keeps the middle pair agreed, which would dilute po.)
    assert cohen_kappa([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0 / 3.0)


def test_quadratic_weighted_kappa_perfect_and_reversed() -> None:
    assert quadratic_weighted_kappa([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) == pytest.approx(
        1.0
    )
    # Reversed 5x5: sum(w*O) = 1 + 1/4 + 0 + 1/4 + 1 = 2.5 (counts);
    # expected is 1/5 everywhere -> sum(w*E) = (100/16) / 5 = 1.25
    # -> kappa = 1 - 2.5 / 1.25 = -1.
    assert quadratic_weighted_kappa([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]) == pytest.approx(
        -1.0
    )


def test_quadratic_weighted_kappa_partial() -> None:
    # a = [1, 2, 3], b = [1, 3, 3], k = 3: sum(w*O) = 0 + 1/4 + 0 = 0.25;
    # rows (1, 1, 1), cols (1, 0, 2) -> sum(w*E) = 2/3 + 1/12 + 1/6 + 1/3
    # = 5/4 -> kappa = 1 - 0.25 / 1.25 = 0.8.
    got = quadratic_weighted_kappa([1, 2, 3], [1, 3, 3], k=3)
    assert got == pytest.approx(0.8)


def test_kappa_errors() -> None:
    with pytest.raises(ValueError, match="degenerate"):
        cohen_kappa([1, 1, 1], [1, 1, 1])
    with pytest.raises(ValueError, match="degenerate"):
        quadratic_weighted_kappa([1, 1], [1, 1], k=5)
    with pytest.raises(ValueError, match="1..k"):
        quadratic_weighted_kappa([1, 6], [1, 2], k=5)
    with pytest.raises(ValueError, match="integer >= 2"):
        quadratic_weighted_kappa([1, 2], [1, 2], k=1)
    with pytest.raises(ValueError, match="equal length"):
        cohen_kappa([1, 0], [1])


# --- Spearman on tied ranks ---------------------------------------------------


def test_spearman_tied_ranks_zero() -> None:
    # x = [1, 1, 2, 2] -> midranks [1.5, 1.5, 3.5, 3.5];
    # y = [1, 2, 1, 2] -> midranks [1.5, 3.5, 1.5, 3.5];
    # mean 2.5 both, dx*dy sums to 1 - 1 - 1 + 1 = 0 -> r = 0.
    assert spearman([1, 1, 2, 2], [1, 2, 1, 2]) == pytest.approx(0.0, abs=1e-12)
