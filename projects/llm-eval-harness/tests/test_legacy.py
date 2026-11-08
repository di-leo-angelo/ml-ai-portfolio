"""Unit tests for the P1 legacy importer (no generated P1 data required)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from llm_eval_harness.legacy import (
    SOURCE_TAG,
    import_generations,
    import_jsonl,
    import_record,
    p1_key_to_task_id,
)

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


# --- P1 recorded generations (M2) ----------------------------------------------


def test_p1_key_to_task_id() -> None:
    assert p1_key_to_task_id("0:lora") == ("p1_0001", "lora")
    assert p1_key_to_task_id("10:base") == ("p1_0011", "base")
    with pytest.raises(ValueError, match="unrecognised"):
        p1_key_to_task_id("x:lora")
    with pytest.raises(ValueError, match="unrecognised"):
        p1_key_to_task_id("3")


def test_import_generations(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        [
            {
                "key": "0:lora",
                "sample_id": 0,
                "subject": "m",
                "difficulty": "b",
                "condition": "lora",
                "wall_s": 17.0,
                "output": "A",
            },
            {
                "key": "1:base",
                "sample_id": 1,
                "subject": "m",
                "difficulty": "b",
                "condition": "base",
                "output": "B",
            },
        ],
    )
    records = import_generations(path)
    assert [r["task_id"] for r in records] == ["p1_0001", "p1_0002"]
    assert [r["condition"] for r in records] == ["lora", "base"]
    assert records[0]["model"] == "Qwen3.8-27B+LoRA"
    assert records[1]["model"] == "Qwen3.8-27B"
    assert records[0]["backend"] == "imported"
    assert records[0]["source"] == f"{SOURCE_TAG}:raw_outputs"
    assert records[0]["seed"] == 0
    assert records[0]["wall_s"] == 17.0
    assert "wall_s" not in records[1]
    assert records[0]["output"] == "A"


def test_import_generations_key_sample_mismatch_raises(tmp_path: Path) -> None:
    rows = [
        {
            "key": "2:lora",
            "sample_id": 0,
            "condition": "lora",
            "output": "A",
        }
    ]
    with pytest.raises(ValueError, match="disagrees"):
        import_generations(_write(tmp_path, rows))


def test_import_generations_unknown_condition_raises(tmp_path: Path) -> None:
    rows = [
        {
            "key": "0:chatgpt",
            "sample_id": 0,
            "condition": "chatgpt",
            "output": "A",
        }
    ]
    with pytest.raises(ValueError, match="unknown P1 condition"):
        import_generations(_write(tmp_path, rows))


def test_import_generations_missing_field_raises(tmp_path: Path) -> None:
    rows = [{"key": "0:lora", "sample_id": 0, "condition": "lora"}]
    with pytest.raises(ValueError, match="missing"):
        import_generations(_write(tmp_path, rows))
