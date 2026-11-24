"""Unit tests for the M5 run report (summary.json + report.md)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from llm_eval_harness.report import (
    build_summary,
    read_jsonl_tolerant,
    write_run_report,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _fixture_run(root: Path) -> Path:
    run_dir = root / "reports" / "fixture-run"
    run_dir.mkdir(parents=True)
    _write_jsonl(
        run_dir / "generations.jsonl",
        [
            {"task_id": f"t0{i}", "condition": "lora", "seed": 0, "output": "x"}
            for i in range(1, 7)
        ],
    )
    _write_jsonl(
        run_dir / "grades.jsonl",
        [
            {
                "task_id": "t01",
                "condition": "lora",
                "math": True,
                "exact_match": True,
                "word_count": 100,
            },
            {
                "task_id": "t02",
                "condition": "lora",
                "math": True,
                "exact_match": False,
                "word_count": 110,
            },
            {
                "task_id": "t03",
                "condition": "lora",
                "math": True,
                "exact_match": True,
                "word_count": 120,
            },
            # t04 abstained: exact_match null
            {
                "task_id": "t04",
                "condition": "lora",
                "math": True,
                "exact_match": None,
                "word_count": 130,
            },
            {
                "task_id": "t05",
                "condition": "lora",
                "math": False,
                "exact_match": None,
                "word_count": 140,
            },
            {
                "task_id": "t06",
                "condition": "lora",
                "math": False,
                "exact_match": None,
                "word_count": 150,
            },
        ],
    )
    _write_jsonl(
        run_dir / "judge_scores.jsonl",
        [
            {
                "key": f"t0{i}:lora",
                "correctness": c,
                "format": 3,
                "parse": "regex" if i == 4 else "json",
                "task_id": f"t0{i}",
                "condition": "lora",
                "judge_model": "J",
                "rubric_version": "1",
                "ts": "2026-11-24T00:00:00+00:00",
            }
            for i, c in zip(range(1, 6), [5, 4, 3, 5, 4], strict=True)
        ],
    )
    return run_dir


def test_build_summary_values_hand_computed(tmp_path) -> None:
    run_dir = _fixture_run(tmp_path)
    summary = build_summary(run_dir)

    # meta contract
    assert summary["meta"]["schema"] == 1
    assert summary["meta"]["run_id"] == "fixture-run"
    assert summary["meta"]["n_generations"] == 6
    assert summary["meta"]["n_graded"] == 6
    assert summary["meta"]["n_judged"] == 5
    assert summary["meta"]["judge_model"] == "J"

    m = summary["metrics"]
    # pass_rate: 2 of 3 answered (t04 abstention excluded from the base)
    assert m["exact_match.pass_rate"]["value"] == pytest.approx(2.0 / 3.0, abs=1e-6)
    assert m["exact_match.pass_rate"]["n"] == 3
    assert len(m["exact_match.pass_rate"]["ci"]) == 2
    # abstention_rate: 1 of 4 math rows
    assert m["exact_match.abstention_rate"]["value"] == pytest.approx(0.25)
    assert m["exact_match.abstention_rate"]["n"] == 4
    # p95 of 100..150: 0.95 * 5 = 4.75 -> 140 + 0.75 * 10 = 147.5
    assert m["format.word_count.p95"]["value"] == pytest.approx(147.5)
    assert m["format.word_count.p95"]["n"] == 6
    # coverage: 5 judged pairs of 6 generated
    assert m["judge.coverage"]["value"] == pytest.approx(5.0 / 6.0, abs=1e-6)
    assert m["judge.coverage"]["n"] == 6
    # parse_json_rate: 4 of 5 rows at the JSON stage
    assert m["judge.parse_json_rate"]["value"] == pytest.approx(0.8)
    # correctness mean of [5, 4, 3, 5, 4] with a bootstrap CI
    assert m["judge.correctness.mean"]["value"] == pytest.approx(4.2)
    assert m["judge.correctness.mean"]["n"] == 5
    lo, hi = m["judge.correctness.mean"]["ci"]
    assert lo < 4.2 < hi
    # format is constant -> the CI collapses to the constant
    assert m["judge.format.mean"]["value"] == pytest.approx(3.0)
    assert m["judge.format.mean"]["ci"] == [3.0, 3.0]


def test_build_summary_judge_only_run(tmp_path) -> None:
    run_dir = tmp_path / "reports" / "judge-only"
    run_dir.mkdir(parents=True)
    _write_jsonl(
        run_dir / "judge_scores.jsonl",
        [
            {
                "key": "t01:lora",
                "correctness": 5,
                "parse": "json",
                "task_id": "t01",
                "condition": "lora",
            },
            {
                "key": "t02:lora",
                "correctness": 3,
                "parse": "json",
                "task_id": "t02",
                "condition": "lora",
            },
        ],
    )
    summary = build_summary(run_dir)
    # nothing to be judged against -> coverage 1.0 by definition
    assert summary["metrics"]["judge.coverage"]["value"] == 1.0
    # dimension detection without a config
    assert "judge.correctness.mean" in summary["metrics"]
    assert "judge.format.mean" not in summary["metrics"]
    assert summary["meta"]["judge_model"] is None


def test_build_summary_empty_run_raises(tmp_path) -> None:
    empty = tmp_path / "reports" / "empty"
    empty.mkdir(parents=True)
    with pytest.raises(ValueError, match="no run outputs"):
        build_summary(empty)


def test_write_run_report_files(tmp_path) -> None:
    run_dir = _fixture_run(tmp_path)
    summary = write_run_report(run_dir)
    on_disk = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert on_disk["metrics"] == summary["metrics"]
    md = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "# Run report: fixture-run" in md
    assert "| metric | value | 95% CI | n |" in md
    assert "exact_match.pass_rate" in md
    assert "Parse coverage" in md
    assert "## Exact match by condition" in md
    assert "| lora | 2 | 3 | 1 | 4 |" in md  # passed / answered / abstained / math
    assert "## Judge means by condition" in md
    assert "## Provenance" in md


def test_read_jsonl_tolerant_torn_tail(tmp_path) -> None:
    good = tmp_path / "good.jsonl"
    good.write_text('{"a": 1}\n{"a": 2}\n', encoding="utf-8")
    assert len(read_jsonl_tolerant(good, "good")) == 2

    torn = tmp_path / "torn.jsonl"
    torn.write_text('{"a": 1}\n{"a": 2\n', encoding="utf-8")
    rows = read_jsonl_tolerant(torn, "torn")
    assert [r["a"] for r in rows] == [1]

    broken = tmp_path / "broken.jsonl"
    broken.write_text('{"a": 1}\n{bad}\n{"a": 3}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="malformed line 2"):
        read_jsonl_tolerant(broken, "broken")

    with pytest.raises(FileNotFoundError, match="not found"):
        read_jsonl_tolerant(tmp_path / "missing.jsonl", "missing")
