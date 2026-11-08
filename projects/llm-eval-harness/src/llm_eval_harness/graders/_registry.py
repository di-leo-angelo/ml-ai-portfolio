"""Grader registry: pure ``(task, output) -> metric fields`` functions.

Deterministic graders run CPU-side over every record of
``generations.jsonl``; they are registered by name and ``grade_record``
merges their outputs. Landed in M2 (PLAN.md section 4): exact-match for
math subjects and the format heuristics, both ported from P1's
``scripts/auto_grade.py``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..tasks import Task

Grader = Callable[[Task, str], dict[str, Any]]

REGISTRY: dict[str, Grader] = {}


def register(name: str) -> Callable[[Grader], Grader]:
    """Decorator: register a grader under ``name``."""

    def decorator(func: Grader) -> Grader:
        REGISTRY[name] = func
        return func

    return decorator


def grade_record(task: Task, output: str) -> dict[str, Any]:
    """Run every registered grader on one (task, output) pair."""
    merged: dict[str, Any] = {}
    for grader in REGISTRY.values():
        merged.update(grader(task, output))
    return merged
