"""Format grader: word budget + step-scaffolding heuristics.

Port of P1's ``scripts/auto_grade.py`` ``format_metrics``: whitespace
word count, step scaffolding (a line starting with ``Step N`` / ``N.`` /
``N)``), and the ~350-word tutor cap.
"""

from __future__ import annotations

import re
from typing import Any

from ..tasks import Task
from ._registry import register

#: P1's tutor word cap.
WORD_CAP = 350

_STEP_RE = re.compile(r"(?im)^\s*(step\s*\d+|\d+[.):])")


@register("format")
def format_metrics(task: Task, output: str) -> dict[str, Any]:
    """Deterministic format metrics for one output (task unused)."""
    words = len(output.split())
    return {
        "word_count": words,
        "step_scaffolding": bool(_STEP_RE.search(output)),
        "within_word_cap": words <= WORD_CAP,
    }
