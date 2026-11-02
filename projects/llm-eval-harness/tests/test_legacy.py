"""Unit tests for the P1 legacy importer (no generated P1 data required)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from llm_eval_harness.legacy import SOURCE_TAG, import_jsonl, import_record

P1_SEEDS = (
    Path(__file__).resolve().parents[1]
    / ".."
    / "qwen-qlora"
    / "data"
    / "samples"
    / "seed_prompts.jsonl"
)


def _write(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "p1.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def test_import_chat_record() -> None:
    task = import_record(
        {
            "system": "tutor",
            "user": "Solve x",
            "assistant": "x = 1",
            "subject": "algebra",
            "difficulty": "beginner",
        },
        line_no=7,
    )
    assert task.id == "p1_0007"
    assert task.prompt == "Solve x"
    assert task.reference == "x = 1"
    assert task.subject == "algebra"
    assert task.source == f"{SOURCE_TAG}:chat"


def test_import_seed_record() -> None:
    task = import_record(
        {"subject": "physics", "difficulty": "beginner", "question": "Why blue?"},
        line_no=2,
    )
    assert task.id == "p1seed_0002"
    assert task.prompt == "Why blue?"
    assert task.reference is None
    assert task.source == f"{SOURCE_TAG}:seed"


def test_import_jsonl_mixed(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        [
            {
                "system": "s",
                "user": "q1",
                "assistant": "a1",
                "subject": "m",
                "difficulty": "b",
            },
            {"question": "q2", "subject": "c", "difficulty": "i"},
        ],
    )
    tasks = import_jsonl(path)
    assert [t.id for t in tasks] == ["p1_0001", "p1seed_0002"]


def test_unrecognised_record_raises() -> None:
    with pytest.raises(ValueError, match="unrecognised"):
        import_record({"foo": "bar"}, line_no=1)


def test_import_real_seed_file() -> None:
    if not P1_SEEDS.is_file():
        pytest.skip("P1 seed file not present")
    tasks = import_jsonl(P1_SEEDS)
    assert len(tasks) == 10
    assert all(t.source == f"{SOURCE_TAG}:seed" for t in tasks)
    subjects = {t.subject for t in tasks}
    assert len(subjects) == 10
