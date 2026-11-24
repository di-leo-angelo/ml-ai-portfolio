"""End-to-end tests for the CPU grading pipeline (no network, no models)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from llm_eval_harness.runner import run_grading

_TOML = """
name = "unit-grade"
data = "TASKS_PLACEHOLDER"
temperature = 0.0
max_tokens = 512

[conditions.base]
seed = 0

[conditions.base.backend]
kind = "imported"
model = "test-model"

[judge]
dimensions = ["correctness", "tone"]

[judge.backend]
kind = "api"
base_url = "http://localhost:1/v1"
model = "judge-model"
api_key_env = "UNSET_TEST_KEY"
"""


def _setup(tmp_path: Path) -> Path:
    tasks = [
        {
            "id": "t1",
            "prompt": "Solve x",
            "reference": "x = 3/4",
            "subject": "algebra",
            "difficulty": "beginner",
        },
        {
            "id": "t2",
            "prompt": "Capital of France?",
            "reference": "Paris",
            "subject": "world_geography",
            "difficulty": "beginner",
        },
    ]
    tasks_path = tmp_path / "tasks.jsonl"
    tasks_path.write_text(
        "\n".join(json.dumps(task) for task in tasks) + "\n", encoding="utf-8"
    )
    generations = [
        {
            "task_id": "t1",
            "condition": "base",
            "seed": 0,
            "output": "Step 1: isolate x. So 3/4",
            "model": "test-model",
        },
        {
            "task_id": "t2",
            "condition": "base",
            "seed": 0,
            "output": "The answer is Paris.",
            "model": "test-model",
        },
    ]
    gen_path = tmp_path / "generations.jsonl"
    gen_path.write_text(
        "\n".join(json.dumps(record) for record in generations) + "\n",
        encoding="utf-8",
    )
    config = tmp_path / "run.toml"
    config.write_text(
        _TOML.replace('"TASKS_PLACEHOLDER"', f'"{tasks_path.as_posix()}"'),
        encoding="utf-8",
    )
    return config


def test_run_grading_writes_merged_rows(tmp_path: Path) -> None:
    config = _setup(tmp_path)
    assert (
        run_grading(
            config,
            generations=tmp_path / "generations.jsonl",
            output_dir=tmp_path,
        )
        == 0
    )
    rows = [
        json.loads(line)
        for line in (tmp_path / "grades.jsonl").read_text().splitlines()
    ]
    assert [r["task_id"] for r in rows] == ["t1", "t2"]
    assert rows[0]["math"] is True
    assert rows[0]["exact_match"] is True
    assert rows[0]["pred_answer"] == "3/4"
    assert rows[0]["step_scaffolding"] is True
    assert rows[0]["word_count"] > 0
    assert rows[1]["math"] is False
    assert rows[1]["exact_match"] is None
    assert rows[1]["model"] == "test-model"


def test_run_grading_unknown_task_raises(tmp_path: Path) -> None:
    config = _setup(tmp_path)
    path = tmp_path / "generations.jsonl"
    line = path.read_text(encoding="utf-8").splitlines()[0]
    path.write_text(
        line.replace('"task_id": "t1"', '"task_id": "zz"') + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="unknown task"):
        run_grading(config, generations=path, output_dir=tmp_path)


def test_run_grading_missing_file_raises(tmp_path: Path) -> None:
    config = _setup(tmp_path)
    with pytest.raises(FileNotFoundError, match="generations file not found"):
        run_grading(
            config,
            generations=tmp_path / "nope.jsonl",
            output_dir=tmp_path,
        )
