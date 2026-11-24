"""Run report: ``summary.json`` + ``report.md`` (M5).

``summary.json`` is the machine-readable run summary and the *gate
contract* (PLAN section 7): provenance under ``meta`` (schema version,
run id, task-file SHA-256, git commit, judge model) and a flat
``metrics`` table keyed by dotted names:

- ``exact_match.pass_rate``    - true / answered (abstentions excluded);
- ``exact_match.abstention_rate`` - abstentions / math rows (the
  "refuse everything" guardrail);
- ``format.word_count.p95``    - 95th percentile of graded word counts;
- ``judge.coverage``           - judged (task, condition) pairs /
  generated pairs (the parse-failure guard input);
- ``judge.parse_json_rate``    - share of judge rows parsed at the
  JSON stage of the ladder (not the regex fallback);
- ``judge.<dim>.mean``         - judge dimension means, with a
  bootstrap 95% CI (B = 10,000, seed 1337, section 6).

Headline numbers in ``report.md`` carry their CI and n, and coverage
is printed next to every judge metric (section 6). The gate (M5)
compares the ``metrics`` table of a new summary against a committed
baseline; the per-condition breakdowns live only in ``report.md``.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__
from .config import RunConfig
from .stats import bootstrap_ci

SCHEMA = 1
# Fields a judge row carries that are not rubric dimension scores.
_JUDGE_RESERVED = frozenset(
    {"key", "parse", "task_id", "condition", "model", "judge_model", "ts"}
)


def read_jsonl_tolerant(path: str | Path, what: str) -> list[dict[str, Any]]:
    """JSONL reader with the stores' torn-tail tolerance.

    A malformed *final* line is a crash mid-write and is dropped; any
    other malformed line fails loudly.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"{what} not found: {file_path}")
    lines = [
        line
        for line in file_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    records: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                continue  # torn tail from a crash mid-write
            raise ValueError(f"malformed line {index + 1} in {file_path}") from None
    return records


def _maybe_read(path: Path, what: str) -> list[dict[str, Any]] | None:
    if not path.is_file():
        return None
    return read_jsonl_tolerant(path, what)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> str | None:
    """Current repo HEAD, or None when not in a git worktree."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def _mode(rows: list[dict[str, Any]], field: str) -> str | None:
    values = sorted({str(r[field]) for r in rows if r.get(field) is not None})
    return ", ".join(values) if values else None


def _p95(values: list[int | float]) -> float:
    """95th percentile, linear interpolation (the stats layer's convention)."""
    return float(np.percentile(np.asarray(values, dtype=float), 95.0))


def _detect_dimensions(
    judge_rows: list[dict[str, Any]], config: RunConfig | None
) -> list[str]:
    """Rubric dimensions, in rubric order when a config is available."""
    if config is not None:
        return list(config.judge.rubric().names())
    dims: list[str] = []
    for row in judge_rows:
        for key, value in row.items():
            if (
                key not in _JUDGE_RESERVED
                and key not in dims
                and isinstance(value, int)
                and 1 <= value <= 5
            ):
                dims.append(key)
    return sorted(dims)


def build_summary(
    run_dir: str | Path,
    config: RunConfig | None = None,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Compute the run summary (pure; no files written).

    ``config`` (loaded run TOML) is optional provenance: with it, the
    task-file hash and the rubric-ordered dimension list are known;
    without it they fall back to detection (or null). ``config_path``
    is the TOML path, recorded in ``meta.config``.
    """
    directory = Path(run_dir)
    generations = _maybe_read(directory / "generations.jsonl", "generations")
    grades = _maybe_read(directory / "grades.jsonl", "grades")
    judge = _maybe_read(directory / "judge_scores.jsonl", "judge scores")
    if generations is None and grades is None and judge is None:
        raise ValueError(
            f"no run outputs (generations.jsonl / grades.jsonl / "
            f"judge_scores.jsonl) in {directory}"
        )

    metrics: dict[str, Any] = {}

    # --- deterministic graders ----------------------------------------
    math_rows = [r for r in grades or [] if r.get("math")]
    if grades:
        answered = [r for r in math_rows if r.get("exact_match") in (True, False)]
        if answered:
            ci = bootstrap_ci(
                [1.0 if r["exact_match"] is True else 0.0 for r in answered]
            )
            metrics["exact_match.pass_rate"] = {
                "value": round(ci["value"], 6),
                "ci": [round(v, 6) for v in ci["ci"]],
                "n": int(len(answered)),
            }
        if math_rows:
            abstained = sum(1 for r in math_rows if r.get("exact_match") is None)
            metrics["exact_match.abstention_rate"] = {
                "value": round(abstained / len(math_rows), 6),
                "n": int(len(math_rows)),
            }
        word_counts = [
            r["word_count"] for r in grades if isinstance(r.get("word_count"), int)
        ]
        if word_counts:
            metrics["format.word_count.p95"] = {
                "value": _p95(word_counts),
                "n": int(len(word_counts)),
            }

    # --- judge ---------------------------------------------------------
    if judge:
        dimensions = _detect_dimensions(judge, config)
        if generations is not None:
            expected = len(generations)
            judged_pairs = {
                (str(r.get("task_id")), str(r.get("condition"))) for r in judge
            }
            generated_pairs = {
                (str(r.get("task_id")), str(r.get("condition"))) for r in generations
            }
            coverage = len(judged_pairs & generated_pairs) / expected
        else:
            coverage = 1.0  # nothing to be judged against
        metrics["judge.coverage"] = {
            "value": round(coverage, 6),
            "n": int(len(generations)) if generations is not None else len(judge),
        }
        json_rows = sum(1 for r in judge if r.get("parse") == "json")
        metrics["judge.parse_json_rate"] = {
            "value": round(json_rows / len(judge), 6),
            "n": int(len(judge)),
        }
        for dim in dimensions:
            values = [r[dim] for r in judge if isinstance(r.get(dim), int)]
            if len(values) >= 2:
                ci = bootstrap_ci(values)
                metrics[f"judge.{dim}.mean"] = {
                    "value": round(ci["value"], 6),
                    "ci": [round(v, 6) for v in ci["ci"]],
                    "n": int(len(values)),
                }

    tasks_file: str | None = None
    tasks_sha: str | None = None
    if config is not None and config.data:
        tasks_file = config.data
        if Path(config.data).is_file():
            tasks_sha = _sha256(Path(config.data))

    return {
        "meta": {
            "schema": SCHEMA,
            "run_id": directory.name,
            "generated_at": datetime.now(UTC).isoformat(),
            "harness_version": __version__,
            "config": str(config_path) if config_path is not None else None,
            "tasks_file": tasks_file,
            "tasks_sha256": tasks_sha,
            "git_commit": _git_commit(),
            "judge_model": _mode(judge, "judge_model") if judge else None,
            "rubric_version": _mode(judge, "rubric_version") if judge else None,
            "n_generations": len(generations) if generations is not None else None,
            "n_graded": len(grades) if grades is not None else None,
            "n_judged": len(judge) if judge is not None else None,
        },
        "metrics": metrics,
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def render_report_md(summary: dict[str, Any], run_dir: str | Path) -> str:
    """Human report: headline table (CI + n everywhere) plus the
    per-condition breakdowns that ``summary.json`` deliberately omits."""
    directory = Path(run_dir)
    grades = _maybe_read(directory / "grades.jsonl", "grades") or []
    judge = _maybe_read(directory / "judge_scores.jsonl", "judge scores") or []
    meta = summary["meta"]
    metrics = summary["metrics"]

    lines: list[str] = [
        f"# Run report: {meta['run_id']}",
        "",
        f"Generated {meta['generated_at']} by llm-eval-harness "
        f"{meta['harness_version']} (summary schema {meta['schema']}).",
        "",
        "| metric | value | 95% CI | n |",
        "| --- | --- | --- | --- |",
    ]
    for name, entry in metrics.items():
        ci = entry.get("ci")
        ci_s = f"[{ci[0]:.4f}, {ci[1]:.4f}]" if ci is not None else "-"
        lines.append(f"| {name} | {_fmt(entry['value'])} | {ci_s} | {entry['n']} |")

    lines += [
        "",
        f"Parse coverage: {_fmt(metrics.get('judge.coverage', {}).get('value'))} "
        f"(printed next to every judge metric per section 6).",
        "",
    ]

    # --- per-condition breakdowns --------------------------------------
    if grades:
        lines += [
            "## Exact match by condition",
            "",
            "| condition | passed | answered | abstained | math rows |",
            "| --- | --- | --- | --- | --- |",
        ]
        for cond in sorted({str(r["condition"]) for r in grades}):
            rows = [r for r in grades if r.get("condition") == cond and r.get("math")]
            answered = [r for r in rows if r.get("exact_match") in (True, False)]
            passed = sum(1 for r in answered if r["exact_match"] is True)
            abstained = sum(1 for r in rows if r.get("exact_match") is None)
            lines.append(
                f"| {cond} | {passed} | {len(answered)} | {abstained} | {len(rows)} |"
            )

    if judge:
        dims = _detect_dimensions(judge, None)
        if dims:
            lines += [
                "",
                "## Judge means by condition",
                "",
                "| condition | " + " | ".join(dims) + " | n |",
                "| --- " * (len(dims) + 2) + "|",
            ]
            for cond in sorted({str(r["condition"]) for r in judge}):
                rows = [r for r in judge if r.get("condition") == cond]
                cells: list[str] = []
                for d in dims:
                    vals = [r[d] for r in rows if isinstance(r.get(d), int)]
                    cells.append(_fmt(round(sum(vals) / len(vals), 3)) if vals else "-")
                lines.append(f"| {cond} | " + " | ".join(cells) + f" | {len(rows)} |")

    lines += [
        "",
        "## Provenance",
        "",
        f"- Tasks file: {meta['tasks_file'] or '(unknown)'}"
        + (f" (sha256 {meta['tasks_sha256'][:12]}...)" if meta["tasks_sha256"] else ""),
        f"- Git commit: {meta['git_commit'] or '(unknown)'}",
        f"- Judge: {meta['judge_model'] or '(none)'}"
        + (f" (rubric {meta['rubric_version']})" if meta["rubric_version"] else ""),
        f"- Rows: {meta['n_generations'] or 0} generations, "
        f"{meta['n_graded'] or 0} graded, {meta['n_judged'] or 0} judged.",
        "",
    ]
    return "\n".join(lines)


def write_run_report(
    run_dir: str | Path,
    config: RunConfig | None = None,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build the summary and write ``summary.json`` + ``report.md``."""
    directory = Path(run_dir)
    summary = build_summary(directory, config, config_path)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (directory / "report.md").write_text(
        render_report_md(summary, directory), encoding="utf-8"
    )
    return summary
