"""Unit tests for M4 judge calibration (PLAN section 5) and tolerance kappa."""

from __future__ import annotations

import pytest
from llm_eval_harness.calibration import calibrate, render_report
from llm_eval_harness.stats.agreement import tolerance_kappa


def _judge(dims: dict, start: int = 1) -> list[dict]:
    rows = []
    for i, score in enumerate(dims):
        rows.append(
            {
                "task_id": f"t{start + i:02d}",
                "condition": "lora",
                "judge_model": "J",
                "rubric_version": "1",
                **{dim: val for dim, val in score.items()},
            }
        )
    return rows


def _labels(human: dict, start: int = 1) -> list[dict]:
    return [
        {
            "task_id": f"t{start + i:02d}",
            "condition": "lora",
            "scores": {dim: val for dim, val in score.items()},
            "pass": True,
            "notes": "",
            "annotator": "self",
        }
        for i, score in enumerate(human)
    ]


def test_tolerance_kappa_hand_computed() -> None:
    # Perfect agreement -> 1.0 (po = 1, pe = 0.52 on 5 levels).
    assert tolerance_kappa([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) == pytest.approx(1.0)
    # Shifted by one: po = 4/5; pe = 0.2 * (0.4 + 0.6 + 0.6 + 0.6 + 0.4) = 0.52
    # -> (0.8 - 0.52) / (1 - 0.52) = 7/12.
    assert tolerance_kappa([1, 2, 3, 4, 5], [2, 3, 4, 5, 1]) == pytest.approx(
        7.0 / 12.0
    )
    # Every pair within one: po = 1 -> kappa 1.0 (pe = 0.4375 for 4 levels).
    assert tolerance_kappa([1, 2, 3, 4], [2, 3, 4, 5]) == pytest.approx(1.0)
    # Degenerate: all mass in one level -> pe = 1.
    with pytest.raises(ValueError, match="degenerate"):
        tolerance_kappa([1, 1, 1], [1, 1, 1])
    with pytest.raises(ValueError, match=">= 1"):
        tolerance_kappa([1, 2], [1, 2], tolerance=0)


def test_calibrate_perfect_agreement_is_gate_worthy() -> None:
    judge = _judge([{"correctness": v} for v in [1, 2, 3, 4, 5]])
    labels = _labels([{"correctness": v} for v in [1, 2, 3, 4, 5]])
    res = calibrate(judge, labels)
    assert res["joined"] == 5 and res["labels_unmatched"] == 0
    entry = res["dimensions"]["correctness"]
    assert entry["spearman"] == pytest.approx(1.0)
    assert entry["pearson"] == pytest.approx(1.0)
    assert entry["kappa_strict"] == pytest.approx(1.0)
    assert entry["kappa_within1"] == pytest.approx(1.0)
    assert entry["kappa_weighted"] == pytest.approx(1.0)
    assert entry["bias"]["mean"] == 0.0 and entry["bias"]["ci"] == [0.0, 0.0]
    assert entry["verdict"] == "gate-worthy"
    diag = [
        c
        for i, row in enumerate(entry["confusion"]["counts"])
        for j, c in enumerate(row)
        if i == j
    ]
    assert diag == [1, 1, 1, 1, 1]


def test_calibrate_leniency_fails_bias_bar() -> None:
    # The judge is systematically +1: perfect monotonic agreement, but
    # bias = 1.0 > 0.5 -> report-only.
    judge = _judge([{"correctness": v} for v in [2, 3, 4, 5]])
    labels = _labels([{"correctness": v} for v in [1, 2, 3, 4]])
    res = calibrate(judge, labels)
    entry = res["dimensions"]["correctness"]
    assert entry["spearman"] == pytest.approx(1.0)
    # po = 0; pe = 3 * (1/4)(1/4) = 3/16 -> (0 - 3/16) / (13/16).
    # calibrate rounds metrics to 4 dp, hence abs tolerance.
    assert entry["kappa_strict"] == pytest.approx(-3.0 / 13.0, abs=1e-4)
    assert entry["kappa_within1"] == pytest.approx(1.0)
    assert entry["bias"]["mean"] == pytest.approx(1.0)
    assert entry["verdict"] == "report-only"


def test_calibrate_reversal_fails_spearman_bar() -> None:
    judge = _judge([{"correctness": v} for v in [5, 4, 3, 2, 1]])
    labels = _labels([{"correctness": v} for v in [1, 2, 3, 4, 5]])
    entry = calibrate(judge, labels)["dimensions"]["correctness"]
    assert entry["spearman"] == pytest.approx(-1.0)
    assert entry["bias"]["mean"] == pytest.approx(0.0)
    assert entry["verdict"] == "report-only"


def test_calibrate_unmatched_rows_and_unscored_dimension() -> None:
    judge = _judge([{"correctness": v} for v in [1, 2, 3, 4, 5]])
    labels = _labels([{"correctness": v, "format": 4} for v in [1, 2, 3, 4, 5]])
    labels.append(
        {
            "task_id": "t99",
            "condition": "lora",
            "scores": {"correctness": 4},
            "pass": True,
            "notes": "",
            "annotator": "self",
        }
    )
    res = calibrate(judge, labels)
    assert res["labels_total"] == 6
    assert res["joined"] == 5
    assert res["labels_unmatched"] == 1
    # "format" is labeled but never judged -> not enough joined rows.
    assert res["dimensions"]["format"]["n"] == 0
    assert res["dimensions"]["format"]["verdict"] == "report-only"


def test_calibrate_zero_variance_is_undefined() -> None:
    judge = _judge([{"correctness": 5} for _ in range(5)])
    labels = _labels([{"correctness": v} for v in [1, 2, 3, 4, 5]])
    entry = calibrate(judge, labels)["dimensions"]["correctness"]
    assert entry["spearman"] is None
    assert entry["pearson"] is None
    # Judge is a constant 5: every kappa is 0. Cohen: po = 1/5 ((5, 5)
    # only), pe = 5/25 = 0.2. Tolerance: po = 2/5, pe = 2 * (1/5)(5/5)
    # = 0.4. Weighted: all pairs (h, 5) -> sum(w*O) = sum(w*E) = 30/16.
    assert entry["kappa_strict"] == pytest.approx(0.0)
    assert entry["kappa_within1"] == pytest.approx(0.0)
    assert entry["kappa_weighted"] == pytest.approx(0.0)
    assert entry["bias"]["mean"] == pytest.approx(2.0)  # mean(5 - h) over h = 1..5
    assert entry["verdict"] == "report-only"


def test_calibrate_label_validation() -> None:
    judge = _judge([{"correctness": 3} for _ in range(2)])
    with pytest.raises(ValueError, match="not in 1\\.\\.5"):
        calibrate(judge, _labels([{"correctness": 6}, {"correctness": 3}]))
    with pytest.raises(ValueError, match="scores"):
        calibrate(judge, [{"task_id": "t01", "condition": "lora", "scores": {}}])


def test_render_report_contains_verdicts_and_matrix() -> None:
    judge = _judge([{"correctness": v} for v in [2, 3, 4, 5]])
    labels = _labels([{"correctness": v} for v in [1, 2, 3, 4]])
    res = calibrate(judge, labels)
    meta = {
        "run": "demo",
        "ts": "2026-11-24T00:00:00Z",
        "labels_path": "labels.jsonl",
        "annotators": ["self"],
        "judge_model": "J",
        "rubric_version": "1",
    }
    report = render_report(res, meta)
    assert "# Judge calibration: demo" in report
    assert "**report-only**" in report
    assert "| human \\ judge |" in report
    assert "must not drive the CI gate" in report
    assert "spearman >= 0.6 and |bias| <= 0.5" in report
