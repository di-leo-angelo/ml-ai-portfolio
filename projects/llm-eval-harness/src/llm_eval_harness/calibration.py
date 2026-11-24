"""M4 judge calibration: join human labels to a run's judge scores.

Pure - no I/O. ``calibrate`` takes the judge rows (the run's
``judge_scores.jsonl`` dicts) and the label rows (PLAN section 3.4
dicts: ``task_id`` / ``condition`` / ``scores`` / ``pass`` / ``notes``
/ ``annotator``), joins them on ``(task_id, condition)``, and scores
every label dimension per PLAN section 5:

- Spearman / Pearson (monotonic vs linear agreement);
- Cohen's kappa (strict), tolerance kappa (|judge - human| <= 1),
  quadratic weighted kappa;
- judge bias = mean(judge - human) with a bootstrap CI;
- the 5x5 confusion matrix (rows human, columns judge).

Decision rule (documented in the report): a dimension is *gate-worthy*
iff Spearman >= 0.6 and |bias| <= 0.5; otherwise *report-only*. An
undefined Spearman (fewer than 2 rows, zero variance) is report-only.
"""

from __future__ import annotations

from typing import Any

from .stats.agreement import (
    cohen_kappa,
    pearson,
    quadratic_weighted_kappa,
    spearman,
    tolerance_kappa,
)
from .stats.bootstrap import bootstrap_ci

SCALE: tuple[int, ...] = (1, 2, 3, 4, 5)
SPEARMAN_MIN = 0.6
BIAS_MAX = 0.5
RULE = f"spearman >= {SPEARMAN_MIN} and |bias| <= {BIAS_MAX}"


def _validate_labels(labels: list[dict[str, Any]]) -> None:
    for index, row in enumerate(labels):
        if not isinstance(row, dict) or "task_id" not in row or "condition" not in row:
            raise ValueError(f"label row {index + 1}: need task_id and condition")
        scores = row.get("scores")
        if not isinstance(scores, dict) or not scores:
            raise ValueError(f"label row {index + 1}: need a non-empty scores dict")
        for dim, value in scores.items():
            if value not in SCALE:
                raise ValueError(
                    f"label row {index + 1} ({row['task_id']}): "
                    f"{dim}={value!r} is not in 1..5"
                )


def _judge_index(judge_rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict]:
    index: dict[tuple[str, str], dict] = {}
    for row in judge_rows:
        index[(str(row.get("task_id")), str(row.get("condition")))] = row
    return index


def _metric_or_none(func: Any, xs: list[int], ys: list[int]) -> float | None:
    try:
        return round(float(func(xs, ys)), 4)
    except ValueError:
        return None


def calibrate(
    judge_rows: list[dict[str, Any]], labels: list[dict[str, Any]]
) -> dict[str, Any]:
    """Join on (task_id, condition) and score every label dimension."""
    _validate_labels(labels)
    jindex = _judge_index(judge_rows)
    joined = sum(
        1 for row in labels if (str(row["task_id"]), str(row["condition"])) in jindex
    )
    dimensions = sorted({d for row in labels for d in row["scores"]})
    per_dim: dict[str, dict[str, Any]] = {}
    for dim in dimensions:
        human: list[int] = []
        judge: list[int] = []
        for row in labels:
            jrow = jindex.get((str(row["task_id"]), str(row["condition"])))
            if jrow is None or dim not in row["scores"] or dim not in jrow:
                continue
            hv, jv = row["scores"][dim], jrow[dim]
            if hv in SCALE and jv in SCALE:
                human.append(hv)
                judge.append(jv)
        n = len(human)
        entry: dict[str, Any] = {"n": n}
        if n < 2:
            entry["verdict"] = "report-only"
            entry["note"] = "not enough joined rows for this dimension"
            per_dim[dim] = entry
            continue

        entry["spearman"] = _metric_or_none(spearman, human, judge)
        entry["pearson"] = _metric_or_none(pearson, human, judge)
        entry["kappa_strict"] = _metric_or_none(cohen_kappa, human, judge)
        entry["kappa_within1"] = _metric_or_none(tolerance_kappa, human, judge)
        entry["kappa_weighted"] = _metric_or_none(
            quadratic_weighted_kappa, human, judge
        )

        diffs = [j - h for h, j in zip(human, judge, strict=True)]
        bias = bootstrap_ci(diffs)
        entry["bias"] = {
            "mean": round(bias["value"], 4),
            "ci": [round(v, 4) for v in bias["ci"]],
            "seed": bias["seed"],
            "b": bias["b"],
        }

        counts = [[0] * len(SCALE) for _ in SCALE]
        for h, j in zip(human, judge, strict=True):
            counts[h - 1][j - 1] += 1
        entry["confusion"] = {
            "human": list(SCALE),  # rows
            "judge": list(SCALE),  # columns
            "counts": counts,
        }

        spear = entry["spearman"]
        gate = (
            spear is not None
            and spear >= SPEARMAN_MIN
            and abs(entry["bias"]["mean"]) <= BIAS_MAX
        )
        entry["verdict"] = "gate-worthy" if gate else "report-only"
        per_dim[dim] = entry

    return {
        "schema": 1,
        "labels_total": len(labels),
        "joined": joined,
        "labels_unmatched": len(labels) - joined,
        "judge_rows": len(judge_rows),
        "dimensions": per_dim,
        "rule": RULE,
        "scale": list(SCALE),
    }


def render_report(result: dict[str, Any], meta: dict[str, Any]) -> str:
    """Markdown report for ``reports/calibration/<run>/report.md``."""
    lines: list[str] = [
        f"# Judge calibration: {meta.get('run', 'run')}",
        "",
        f"Generated {meta.get('ts', '')} by llm-eval-harness "
        f"(schema {result['schema']}).",
        "",
        f"- Judge: {meta.get('judge_model', 'unknown')} "
        f"(rubric {meta.get('rubric_version', '?')})",
        f"- Human labels: {meta.get('labels_path', '?')} - "
        f"{result['labels_total']} rows, annotators: "
        f"{', '.join(meta.get('annotators', ['?']))}",
        f"- Joined on (task_id, condition): {result['joined']} of "
        f"{result['labels_total']} labels ({result['labels_unmatched']} "
        f"unmatched), vs {result['judge_rows']} judge rows.",
        f"- **Decision rule:** a dimension is gate-worthy iff "
        f"{result['rule']}; otherwise report-only. Undefined Spearman "
        "(n < 2 or zero variance) is report-only.",
        "",
        "| dimension | n | Spearman | Pearson | kappa strict | "
        "kappa (±1) | kappa weighted | bias 95% CI | verdict |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for dim, entry in result["dimensions"].items():

        def fmt(value: Any) -> str:
            return "undefined" if value is None else f"{value:.3f}"

        bias = entry.get("bias")
        bias_s = (
            f"{bias['mean']:+.3f} [{bias['ci'][0]:+.3f}, {bias['ci'][1]:+.3f}]"
            if bias
            else "n/a"
        )
        lines.append(
            f"| {dim} | {entry['n']} | {fmt(entry.get('spearman'))} | "
            f"{fmt(entry.get('pearson'))} | {fmt(entry.get('kappa_strict'))} | "
            f"{fmt(entry.get('kappa_within1'))} | {fmt(entry.get('kappa_weighted'))} | "
            f"{bias_s} | **{entry['verdict']}** |"
        )
    for dim, entry in result["dimensions"].items():
        if "note" in entry:
            lines += ["", f"**{dim}:** {entry['note']}."]
            continue
        conf = entry["confusion"]
        header = "| human \\ judge | " + " | ".join(map(str, conf["judge"])) + " |"
        sep = "| --- " * (len(conf["judge"]) + 1) + "|"
        lines += ["", f"## {dim}", "", header, sep]
        for i, h in enumerate(conf["human"]):
            cells = " | ".join(str(c) for c in conf["counts"][i])
            lines.append(f"| {h} | {cells} |")
        spear = entry.get("spearman")
        reasons: list[str] = []
        if spear is None:
            reasons.append("Spearman undefined")
        elif spear < SPEARMAN_MIN:
            reasons.append(f"Spearman {spear:.3f} < {SPEARMAN_MIN}")
        if abs(entry["bias"]["mean"]) > BIAS_MAX:
            reasons.append(f"|bias| {abs(entry['bias']['mean']):.3f} > {BIAS_MAX}")
        if entry["verdict"] == "gate-worthy":
            lines += [
                "",
                f"Verdict: **gate-worthy** - {spear:.3f} Spearman and "
                f"bias {entry['bias']['mean']:+.3f} both clear the bar.",
            ]
        else:
            why = "; ".join(reasons) if reasons else "undefined metrics"
            lines += [
                "",
                f"Verdict: **report-only** - {why}. This dimension may "
                "appear in reports but must not drive the CI gate or a "
                "go/no-go decision.",
            ]
    lines += [
        "",
        "## Reading this report",
        "",
        "Rows are human scores, columns judge scores; the diagonal is "
        "exact agreement. Mass stacked on the 5 column with a positive "
        "bias is the judge's ceiling saturation (it cannot tell a 4 from "
        "a 5), which is what the weighted kappa and the (±1) kappa "
        "separate from genuine disagreement. The gate (M5) may only use "
        "gate-worthy dimensions.",
    ]
    return "\n".join(lines) + "\n"
