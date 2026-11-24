"""Unit tests for the M5 CI gate (PLAN section 7)."""

from __future__ import annotations

import copy
import json

import pytest
from llm_eval_harness.gate import (
    evaluate,
    load_gate_config,
    render_gate_md,
    run_gate,
)

SHA_A = "a" * 64
SHA_B = "b" * 64

PASS_CFG = """
[metrics."exact_match.pass_rate"]
min_delta = -0.02

[metrics."format.word_count.p95"]
max_delta = 40

[metrics."judge.correctness.mean"]
check = false
min_delta = 0.0
"""

COVERAGE_GUARD_CFG = (
    "[guards]\njudge_coverage_min = 0.95\n\n"
    '[metrics."judge.correctness.mean"]\ncheck = false\n'
)


def make_summary(
    run_id: str = "new-run",
    coverage: float | None = 0.9,
    sha: str = SHA_A,
    **metric_overrides: float,
) -> dict:
    metrics: dict = {
        "exact_match.pass_rate": {
            "value": 0.666667,
            "ci": [0.333333, 1.0],
            "n": 3,
        },
        "exact_match.abstention_rate": {"value": 0.25, "n": 4},
        "format.word_count.p95": {"value": 147.5, "n": 6},
        "judge.parse_json_rate": {"value": 0.8, "n": 5},
        "judge.correctness.mean": {"value": 4.2, "ci": [3.4, 5.0], "n": 5},
    }
    if coverage is not None:
        metrics["judge.coverage"] = {"value": coverage, "n": 6}
    for name, value in metric_overrides.items():
        metrics[name]["value"] = value
    return {
        "meta": {
            "schema": 1,
            "run_id": run_id,
            "generated_at": "2026-11-24T00:00:00+00:00",
            "tasks_sha256": sha,
        },
        "metrics": metrics,
    }


def _cfg(tmp_path, text: str) -> str:
    path = tmp_path / "gate.toml"
    path.write_text(text, encoding="utf-8")
    return str(path)


def _evaluate(new: dict, base: dict, cfg_text: str, tmp_path) -> dict:
    return evaluate(new, base, load_gate_config(_cfg(tmp_path, cfg_text)))


def test_gate_passes_against_its_own_baseline(tmp_path) -> None:
    new = make_summary()
    base = make_summary(run_id="baseline-run")
    result = _evaluate(new, base, PASS_CFG, tmp_path)
    assert result["exit_code"] == 0
    assert result["passed"] is True
    assert result["errors"] == []
    assert result["warnings"] == []
    by_name = {m["name"]: m for m in result["metrics"]}
    assert by_name["exact_match.pass_rate"]["passed"] is True
    assert by_name["format.word_count.p95"]["passed"] is True
    # the check=false dimension is carried but not evaluated
    assert by_name["judge.correctness.mean"]["checked"] is False
    assert by_name["judge.correctness.mean"]["passed"] is None
    assert by_name["judge.correctness.mean"]["delta"] == pytest.approx(0.0)


def test_planted_regression_fails(tmp_path) -> None:
    # The baseline passed 0.05 more; min_delta = -0.02 -> regression.
    new = make_summary()
    base = make_summary(run_id="baseline-run")
    base["metrics"]["exact_match.pass_rate"]["value"] = 0.716667
    result = _evaluate(new, base, PASS_CFG, tmp_path)
    assert result["exit_code"] == 1
    assert result["passed"] is False
    failed = [m for m in result["metrics"] if m["passed"] is False]
    assert [m["name"] for m in failed] == ["exact_match.pass_rate"]
    assert "min_delta" in failed[0]["reasons"][0]


def test_ci_low_min_fails_when_lower_bound_drops(tmp_path) -> None:
    cfg = '[metrics."exact_match.pass_rate"]\nmin_delta = 0.0\nci_low_min = 0.5\n'
    new = make_summary()  # CI lower bound 0.333333 < 0.5
    base = make_summary(run_id="baseline-run")
    result = _evaluate(new, base, cfg, tmp_path)
    assert result["exit_code"] == 1
    failed = [m for m in result["metrics"] if m["passed"] is False]
    assert failed and "ci_low_min" in failed[0]["reasons"][0]


def test_max_delta_catches_length_gaming(tmp_path) -> None:
    cfg = '[metrics."format.word_count.p95"]\nmax_delta = 40\n'
    new = make_summary()  # 147.5
    base = make_summary(run_id="baseline-run")
    base["metrics"]["format.word_count.p95"]["value"] = 100.0  # delta +47.5
    result = _evaluate(new, base, cfg, tmp_path)
    assert result["exit_code"] == 1
    failed = [m for m in result["metrics"] if m["passed"] is False]
    assert failed and "max_delta" in failed[0]["reasons"][0]


def test_coverage_guard_triggers_at_90_percent(tmp_path) -> None:
    new = make_summary(coverage=0.9)
    base = make_summary(run_id="baseline-run", coverage=0.9)
    result = _evaluate(new, base, COVERAGE_GUARD_CFG, tmp_path)
    # section 7: coverage below the guard is an error (exit 2)
    assert result["exit_code"] == 2
    assert any("judge coverage 0.9000" in e for e in result["errors"])
    assert result["guards"]["judge_coverage_min"]["passed"] is False


def test_coverage_guard_requires_coverage_metric(tmp_path) -> None:
    new = make_summary(coverage=None)  # no judge.coverage at all
    base = make_summary(run_id="baseline-run", coverage=None)
    result = _evaluate(new, base, COVERAGE_GUARD_CFG, tmp_path)
    assert result["exit_code"] == 2
    assert any("no judge.coverage" in e for e in result["errors"])


def test_missing_metric_is_an_error(tmp_path) -> None:
    cfg = '[metrics."judge.tone.mean"]\nmin_delta = 0.0\n'
    result = _evaluate(
        make_summary(), make_summary(run_id="baseline-run"), cfg, tmp_path
    )
    assert result["exit_code"] == 2
    assert any("judge.tone.mean" in e for e in result["errors"])


def test_task_hash_mismatch_warns_but_passes(tmp_path) -> None:
    new = make_summary(sha=SHA_A)
    base = make_summary(run_id="baseline-run", sha=SHA_B)
    result = _evaluate(new, base, PASS_CFG, tmp_path)
    assert result["exit_code"] == 0
    assert any("SHA-256 MISMATCH" in w for w in result["warnings"])


def test_check_false_metrics_are_never_enforced(tmp_path) -> None:
    cfg = '[metrics."judge.correctness.mean"]\ncheck = false\nmin_delta = 0.0\n'
    new = make_summary()
    base = make_summary(run_id="baseline-run")
    base["metrics"]["judge.correctness.mean"]["value"] = 1.0  # delta +3.2
    result = _evaluate(new, base, cfg, tmp_path)
    assert result["exit_code"] == 0
    entry = result["metrics"][0]
    assert entry["checked"] is False and entry["passed"] is None
    assert entry["delta"] == pytest.approx(3.2)


def test_unknown_schema_refuses_to_compare(tmp_path) -> None:
    new = make_summary()
    base = make_summary(run_id="baseline-run")
    base["meta"]["schema"] = 99
    result = _evaluate(new, base, PASS_CFG, tmp_path)
    assert result["exit_code"] == 2
    assert any("unknown summary schema" in e for e in result["errors"])


def test_gate_config_validation(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="gate config not found"):
        load_gate_config(tmp_path / "nope.toml")
    with pytest.raises(ValueError, match="unknown top-level"):
        load_gate_config(_cfg(tmp_path, "[bogus]\nx = 1\n"))
    with pytest.raises(ValueError, match="unknown key"):
        load_gate_config(_cfg(tmp_path, '[metrics."a.b"]\nbogus = 1\n'))
    with pytest.raises(ValueError, match="must be a number"):
        load_gate_config(_cfg(tmp_path, '[metrics."a.b"]\nmin_delta = "x"\n'))
    with pytest.raises(ValueError, match="checks nothing"):
        load_gate_config(_cfg(tmp_path, "# empty\n"))
    cfg = load_gate_config(_cfg(tmp_path, PASS_CFG))
    assert cfg["metrics"]["exact_match.pass_rate"]["min_delta"] == pytest.approx(-0.02)


def test_render_gate_md_contents(tmp_path) -> None:
    new = make_summary()
    base = make_summary(run_id="baseline-run")
    base["metrics"]["exact_match.pass_rate"]["value"] = 0.716667
    result = _evaluate(new, base, PASS_CFG, tmp_path)
    md = render_gate_md(result, tmp_path / "b.json", tmp_path / "g.toml")
    assert "# Gate report: new-run" in md
    assert "REGRESSION" in md
    assert "min_delta=-0.02" in md
    assert "report-only" in md  # the check=false row
    assert "must not drive the gate" in md


def _write_summary_file(run_dir, summary: dict) -> None:
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")


def test_run_gate_cli_happy_path(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    run_dir = tmp_path / "reports" / "fixture-run"
    run_dir.mkdir(parents=True)
    new = make_summary()
    _write_summary_file(run_dir, new)
    base = make_summary(run_id="baseline-run")
    baseline_file = tmp_path / "baseline.json"
    baseline_file.write_text(json.dumps(base), encoding="utf-8")
    cfg = _cfg(tmp_path, PASS_CFG)

    code = run_gate("fixture-run", baseline_file, cfg)
    assert code == 0
    gate_json = json.loads((run_dir / "gate.json").read_text(encoding="utf-8"))
    assert gate_json["exit_code"] == 0 and gate_json["passed"] is True
    assert (run_dir / "gate_report.md").is_file()


def test_run_gate_cli_missing_inputs(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    cfg = _cfg(tmp_path, PASS_CFG)
    # no summary at all
    assert run_gate("ghost-run", tmp_path / "b.json", cfg) == 2
    # summary exists but the baseline file is missing
    run_dir = tmp_path / "reports" / "fixture-run"
    run_dir.mkdir(parents=True)
    _write_summary_file(run_dir, make_summary())
    assert run_gate("fixture-run", tmp_path / "missing.json", cfg) == 2
    # a corrupt baseline is an error, not a crash
    bad = tmp_path / "bad.json"
    bad.write_text("{nope", encoding="utf-8")
    assert run_gate("fixture-run", bad, cfg) == 2


def test_run_gate_writes_outputs_to_output_dir(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    run_dir = tmp_path / "reports" / "fixture-run"
    run_dir.mkdir(parents=True)
    _write_summary_file(run_dir, make_summary())
    base = make_summary(run_id="baseline-run")
    baseline_file = tmp_path / "baseline.json"
    baseline_file.write_text(json.dumps(base), encoding="utf-8")
    cfg = _cfg(tmp_path, PASS_CFG)
    out = tmp_path / "elsewhere"
    assert run_gate("fixture-run", baseline_file, cfg, output_dir=out) == 0
    assert (out / "gate.json").is_file() and (out / "gate_report.md").is_file()


def test_evaluate_is_pure_no_mutation(tmp_path) -> None:
    new = make_summary()
    base = make_summary(run_id="baseline-run")
    snapshot = copy.deepcopy((new, base))
    _evaluate(new, base, PASS_CFG, tmp_path)
    assert (new, base) == snapshot
