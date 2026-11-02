"""Unit tests for the task spec (JSONL load + validation)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from llm_eval_harness.tasks import Task, load_tasks

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def _write(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "tasks.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def test_load_valid_tasks(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        [
            {"id": "t1", "prompt": "What is 2+2?"},
            {"prompt": "Explain gravity.", "subject": "physics"},
        ],
    )
    tasks = load_tasks(path)
    assert len(tasks) == 2
    assert tasks[0].id == "t1"
    assert tasks[1].id == "task_0002"
    assert tasks[1].subject == "physics"
    assert tasks[0].system is None


def test_derived_id_uses_line_number(tmp_path: Path) -> None:
    path = _write(tmp_path, [{"prompt": "a"}, {"prompt": "b"}])
    tasks = load_tasks(path)
    assert tasks[0].id == "task_0001"
    assert tasks[1].id == "task_0002"


def test_missing_prompt_raises(tmp_path: Path) -> None:
    path = _write(tmp_path, [{"reference": "x"}])
    with pytest.raises(ValueError, match="missing required"):
        load_tasks(path)


def test_unknown_field_raises(tmp_path: Path) -> None:
    path = _write(tmp_path, [{"prompt": "hi", "bogus": 1}])
    with pytest.raises(ValueError, match="unknown field"):
        load_tasks(path)


def test_duplicate_id_raises(tmp_path: Path) -> None:
    path = _write(tmp_path, [{"id": "x", "prompt": "a"}, {"id": "x", "prompt": "b"}])
    with pytest.raises(ValueError, match="duplicate task id"):
        load_tasks(path)


def test_blank_prompt_rejected_by_loader(tmp_path: Path) -> None:
    path = _write(tmp_path, [{"prompt": "   "}, {"prompt": "ok"}])
    with pytest.raises(ValueError, match="missing required"):
        load_tasks(path)


def test_task_rejects_blank_prompt() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        Task(prompt="   ", id="t1")


def test_tags_coerced_to_tuple(tmp_path: Path) -> None:
    path = _write(tmp_path, [{"prompt": "hi", "tags": ["a", "b"]}])
    task = load_tasks(path)[0]
    assert task.tags == ("a", "b")
    assert isinstance(task.tags, tuple)


def test_example_tasks_load() -> None:
    path = CONFIG_DIR / "example_tasks.jsonl"
    tasks = load_tasks(path)
    assert tasks, "example task file should not be empty"
    for task in tasks:
        assert task.prompt.strip()


def test_roundtrip() -> None:
    task = Task(prompt="hi", id="t1", reference="42", tags=["a", "b"])
    raw = task.to_dict()
    assert raw["reference"] == "42"
    assert raw["tags"] == ["a", "b"]
    assert task.has_reference
