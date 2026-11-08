"""Statistics layer: agreement and inference, pure CPU.

M2 lands the agreement functions the port validation needs (Pearson /
Spearman correlation of P0 judge scores against P1's legacy
``judge_scores.jsonl``); M3 adds bootstrap CIs, McNemar, and seed-run
variance (PLAN.md section 4).
"""

from __future__ import annotations

from .agreement import pearson, spearman

__all__ = ["pearson", "spearman"]
