"""Unit tests for the deterministic graders (M2), incl. P1 port fidelity.

The P1 fixture (``fixtures/p1_grades_sample.jsonl``) holds real rows from
P1's ``graded.jsonl`` / ``raw_outputs.jsonl`` / ``test.jsonl`` with the
P1 metric values; the grader port must reproduce them exactly.
"""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

from llm_eval_harness.graders import REGISTRY, grade_record
from llm_eval_harness.graders.exact_match import (
    answers_match,
    extract_final_answer,
)
from llm_eval_harness.graders.format import WORD_CAP, format_metrics
from llm_eval_harness.tasks import Task

FIXTURE = Path(__file__).parent / "fixtures" / "p1_grades_sample.jsonl"


def _task(
    prompt: str = "Solve", reference: str = "1/2", subject: str = "algebra"
) -> Task:
    return Task(prompt=prompt, reference=reference, id="t1", subject=subject)


def test_registry_has_both_graders() -> None:
    assert set(REGISTRY) == {"exact_match", "format"}


def test_port_matches_p1_grades() -> None:
    rows = [
        json.loads(line)
        for line in FIXTURE.read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) >= 10
    for row in rows:
        task = Task(
            prompt=row["prompt"],
            reference=row["reference"],
            id="t1",
            subject=row["subject"],
            difficulty=row["difficulty"],
        )
        got = grade_record(task, row["output"])
        p1 = row["p1"]
        assert got["math"] == p1["math"], row["key"]
        assert got["exact_match"] == p1["exact_match"], row["key"]
        assert got["pred_answer"] == p1["pred_answer"], row["key"]
        assert got["ref_answer"] == p1["ref_answer"], row["key"]
        assert got["word_count"] == p1["word_count"], row["key"]
        assert got["step_scaffolding"] == p1["step_scaffolding"], row["key"]
        assert got["within_word_cap"] == p1["within_word_cap"], row["key"]


# --- extract_final_answer (P1 port) ------------------------------------------


def test_extract_final_answer_last_fraction_wins() -> None:
    assert (
        extract_final_answer("... so 3/36, which reduces to 1/12")
        == Fraction(1, 12)
    )


def test_extract_final_answer_last_number() -> None:
    assert extract_final_answer("the answer is 42") == Fraction(42)
    assert extract_final_answer("42 then 43") == Fraction(43)


def test_extract_final_answer_no_number() -> None:
    assert extract_final_answer("no numeric answer here") is None


def test_extract_final_answer_tail_window() -> None:
    long = "1 \n" * 400 + "x" * 400 + " final 7"
    assert extract_final_answer(long) == Fraction(7)
    # a number only in the head (beyond the last 500 chars) is ignored
    assert extract_final_answer("5 " + "x" * 600) is None


def test_extract_final_answer_division_by_zero() -> None:
    assert extract_final_answer("answer 3/0") is None


# --- answers_match (P1 port) --------------------------------------------------


def test_answers_match_equivalent_fractions() -> None:
    assert answers_match(Fraction(3, 36), Fraction(1, 12)) is True
    assert answers_match(Fraction(1), Fraction(2)) is False
    assert answers_match(None, Fraction(1)) is None
    assert answers_match(Fraction(1), None) is None


# --- exact_match grader -------------------------------------------------------


def test_non_math_gets_null_exact() -> None:
    got = grade_record(_task(subject="world_geography"), "answer: 5")
    assert got["math"] is False
    assert got["exact_match"] is None
    assert got["pred_answer"] == "5"


def test_math_exact_true_and_false() -> None:
    assert grade_record(_task(reference="1/2"), "so 1/2")["exact_match"] is True
    assert grade_record(_task(reference="1/2"), "so 1/3")["exact_match"] is False
    # float-vs-fraction equivalence through exact arithmetic
    assert grade_record(_task(reference="0.5"), "so 1/2")["exact_match"] is True


# --- format grader (P1 port) ---------------------------------------------------


def test_format_step_scaffolding() -> None:
    assert format_metrics(_task(), "Step 1: think\nStep 2: answer")["step_scaffolding"]
    assert format_metrics(_task(), "1. first\n2. second")["step_scaffolding"]
    assert format_metrics(_task(), "2) a\n3) b")["step_scaffolding"]
    assert not format_metrics(_task(), "no structure here")["step_scaffolding"]
    # "Step" mid-line must not count (anchored at line start)
    assert not format_metrics(_task(), "the word Step 1 appears inline")[
        "step_scaffolding"
    ]


def test_format_word_cap_boundary() -> None:
    assert format_metrics(_task(), " ".join(["a"] * 40))["word_count"] == 40
    at_cap = " ".join("w" * 3 for _ in range(WORD_CAP))
    over_cap = " ".join("w" * 3 for _ in range(WORD_CAP + 1))
    assert format_metrics(_task(), at_cap)["within_word_cap"] is True
    assert format_metrics(_task(), over_cap)["within_word_cap"] is False
