"""Deterministic graders (M2) and the LLM-as-judge.

Graders are pure CPU functions over (task, output) pairs, registered by
name (see :mod:`_registry`); the judge (see :mod:`judge`) is a backend
call with a rubric-anchored prompt, a parse ladder, and a resume-safe
``judge_scores.jsonl``.
"""

from __future__ import annotations

from . import exact_match, format  # noqa: F401 - registration side effects
from ._registry import REGISTRY, Grader, grade_record, register

__all__ = ["Grader", "REGISTRY", "exact_match", "format", "grade_record", "register"]
