"""Exact-match grader: final-answer equivalence for math subjects.

Port of P1's ``scripts/auto_grade.py`` (``extract_final_answer`` /
``answers_match``), kept verbatim on the regexes and constants so P1's
published 45 % exact-match rate stays reproducible: the last fraction
(preferred) or number in the last 500 characters of the text, compared
with exact fraction arithmetic (``3/36 == 1/12``). Non-math subjects
get ``exact_match: null``; the extracted answers are still recorded for
inspection (P1's behavior).
"""

from __future__ import annotations

import re
from fractions import Fraction
from typing import Any

from ..tasks import Task
from ._registry import register

#: P1's math subjects - exact match applies only to these (P1 constant).
MATH_SUBJECTS: frozenset[str] = frozenset(
    {"probability", "algebra", "calculus", "physics", "computer_science"}
)

FRACTION = re.compile(r"(-?\d+)\s*[/÷]\s*(\d+)")
NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def extract_final_answer(text: str) -> Fraction | None:
    """Last fraction (preferred) or last number in the tail of the text."""
    tail = text[-500:]
    fracs = FRACTION.findall(tail)
    if fracs:
        num, den = fracs[-1]
        if int(den) != 0:
            return Fraction(int(num), int(den))
        return None
    nums = NUMBER.findall(tail)
    if nums:
        try:
            return Fraction(nums[-1])
        except (ValueError, ZeroDivisionError):
            return None
    return None


def answers_match(pred: Fraction | None, ref: Fraction | None) -> bool | None:
    """True/False when both answers are extractable, else None."""
    if pred is None or ref is None:
        return None  # not gradable by exact match
    return abs(float(pred) - float(ref)) < 1e-9


@register("exact_match")
def exact_match(task: Task, output: str) -> dict[str, Any]:
    """Grade one output against the task's reference answer."""
    is_math = task.subject is not None and task.subject in MATH_SUBJECTS
    ref = extract_final_answer(task.reference) if task.reference else None
    pred = extract_final_answer(output)
    return {
        "math": is_math,
        "exact_match": answers_match(pred, ref) if is_math else None,
        "pred_answer": None if pred is None else str(pred),
        "ref_answer": None if ref is None else str(ref),
    }
