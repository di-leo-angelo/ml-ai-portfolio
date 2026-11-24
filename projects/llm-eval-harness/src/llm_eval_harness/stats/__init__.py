"""Statistics layer: agreement and inference, pure CPU.

M2 landed the agreement functions the port validation needs (Pearson /
Spearman correlation of P0 judge scores against P1's legacy
``judge_scores.jsonl``); M3 adds bootstrap CIs (percentile, plain /
stratified / paired), exact McNemar with Clopper-Pearson CI, and
Cohen's / quadratic-weighted kappa. No scipy, no sklearn (PLAN.md
section 6).
"""

from __future__ import annotations

from .agreement import (
    cohen_kappa,
    pearson,
    quadratic_weighted_kappa,
    spearman,
)
from .bootstrap import (
    bootstrap_ci,
    paired_bootstrap_ci,
    percentile_ci,
    stratified_bootstrap_ci,
)
from .mcnemar import mcnemar, mcnemar_ci, mcnemar_pvalue

__all__ = [
    "bootstrap_ci",
    "cohen_kappa",
    "mcnemar",
    "mcnemar_ci",
    "mcnemar_pvalue",
    "paired_bootstrap_ci",
    "pearson",
    "percentile_ci",
    "quadratic_weighted_kappa",
    "spearman",
    "stratified_bootstrap_ci",
]
