"""Command line interface: ``python -m llm_eval_harness <command>``.

M1 commands: ``generate`` (drive generation for a run config) and
``import-tasks`` (convert a P1 split/seed file to the harness schema).
M2 commands: ``import-generations`` (P1 raw_outputs -> harness
generations), ``grade`` (deterministic graders, CPU-only), ``judge``
(LLM-as-judge, resume-safe, ``--against`` correlation vs a P1 legacy
file), ``pairwise`` (position-swapped pairwise smoke). M4 adds
``calibrate`` (judge vs human labels, PLAN section 5). M5 adds
``report`` (summary.json + report.md, the gate contract) and ``gate``
(baseline comparison, PLAN section 7).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .calibration import calibrate, render_report
from .config import ConfigError, load_run_config
from .gate import run_gate
from .legacy import import_generations, import_jsonl
from .report import read_jsonl_tolerant, write_run_report
from .runner import run_generation, run_grading, run_judging, run_pairwise


def _cmd_import_tasks(args: argparse.Namespace) -> int:
    tasks = import_jsonl(args.input)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps(task.to_dict(), ensure_ascii=False) + "\n")
    print(f"imported {len(tasks)} tasks -> {output}")
    return 0


def _cmd_import_generations(args: argparse.Namespace) -> int:
    records = import_generations(args.input)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    by_condition: dict[str, int] = {}
    for record in records:
        by_condition[str(record["condition"])] = (
            by_condition.get(str(record["condition"]), 0) + 1
        )
    print(
        f"imported {len(records)} generations "
        f"({', '.join(f'{k}: {v}' for k, v in sorted(by_condition.items()))}) "
        f"-> {output}"
    )
    return 0


def _cmd_calibrate(args: argparse.Namespace) -> int:
    """M4: join human labels to the run's judge scores (PLAN section 5)."""
    run = Path(args.run)
    judge_path = (
        run / "judge_scores.jsonl"
        if run.is_dir()
        else Path("reports") / run / "judge_scores.jsonl"
    )
    try:
        judge_rows = read_jsonl_tolerant(judge_path, "judge scores")
        label_rows = read_jsonl_tolerant(Path(args.labels), "labels")
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not judge_rows or not label_rows:
        print("error: judge scores or labels are empty", file=sys.stderr)
        return 2
    try:
        result = calibrate(judge_rows, label_rows)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if result["joined"] < 2:
        print(
            f"error: only {result['joined']} label row(s) joined to the "
            "judge store - need at least 2 (check task_id/condition overlap)",
            file=sys.stderr,
        )
        return 2

    def mode(rows: list[dict[str, Any]], field: str) -> str:
        return ", ".join(
            sorted({str(r[field]) for r in rows if r.get(field) is not None})
        )

    meta = {
        "run": run.name,
        "ts": datetime.now(UTC).isoformat(),
        "labels_path": str(Path(args.labels).resolve()),
        "annotators": sorted({str(r.get("annotator", "?")) for r in label_rows}),
        "judge_model": mode(judge_rows, "judge_model"),
        "rubric_version": mode(judge_rows, "rubric_version"),
    }
    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("reports/calibration") / judge_path.parent.name
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "agreement.json").write_text(
        json.dumps({**meta, **result}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (out_dir / "report.md").write_text(render_report(result, meta), encoding="utf-8")

    print(f"calibration: {result['joined']}/{result['labels_total']} labels joined")
    for dim, entry in result["dimensions"].items():
        spear = entry.get("spearman")
        bias = entry.get("bias", {}).get("mean")
        print(
            f"  {dim}: n={entry['n']} spearman="
            f"{spear if spear is not None else 'undefined'} "
            f"bias={f'{bias:+.3f}' if bias is not None else 'n/a'} "
            f"-> {entry['verdict']}"
        )
    print(f"Output: {out_dir / 'agreement.json'}, {out_dir / 'report.md'}")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    """M5: write the run's ``summary.json`` + ``report.md`` (PLAN 6-7)."""
    run = Path(args.run)
    run_dir = run if run.is_dir() else Path("reports") / run
    if not run_dir.is_dir():
        print(f"error: run directory not found: {run_dir}", file=sys.stderr)
        return 2
    config = None
    config_path: str | None = None
    if args.config:
        try:
            config = load_run_config(args.config)
            config_path = args.config
        except ConfigError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
    try:
        summary = write_run_report(run_dir, config, config_path)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    meta = summary["meta"]
    print(f"report: {meta['run_id']} - {len(summary['metrics'])} metrics")
    for name, entry in summary["metrics"].items():
        ci = entry.get("ci")
        ci_s = f"  95% CI [{ci[0]:.4f}, {ci[1]:.4f}]" if ci else ""
        print(f"  {name}: {entry['value']}{ci_s} (n={entry['n']})")
    print(f"Output: {run_dir / 'summary.json'}, {run_dir / 'report.md'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llm_eval_harness")
    sub = parser.add_subparsers(dest="command", required=True)

    generate = sub.add_parser(
        "generate", help="generate outputs for every condition of a run config"
    )
    generate.add_argument("--config", required=True, help="path to a run TOML")
    generate.add_argument(
        "--env-file",
        default=None,
        help="dotenv file for API keys (e.g. projects/qwen-qlora/.env)",
    )
    generate.add_argument(
        "--limit", type=int, default=0, help="generate for at most N tasks (0 = all)"
    )
    generate.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="parallel API requests for api/runpod backends (>= 1)",
    )
    generate.add_argument(
        "--output-dir", default=None, help="override the config's output_dir"
    )

    import_tasks = sub.add_parser(
        "import-tasks", help="convert a P1 split/seed file to a tasks JSONL"
    )
    import_tasks.add_argument(
        "--input", required=True, help="P1 JSONL (chat split or seed prompts)"
    )
    import_tasks.add_argument(
        "--output",
        required=True,
        help="output tasks JSONL (e.g. data/legacy/p1_test.jsonl)",
    )

    import_generations_cmd = sub.add_parser(
        "import-generations",
        help="convert P1's raw_outputs.jsonl to harness generations",
    )
    import_generations_cmd.add_argument(
        "--input",
        required=True,
        help="P1 raw_outputs.jsonl (key/sample_id/condition/output records)",
    )
    import_generations_cmd.add_argument(
        "--output",
        required=True,
        help="output generations JSONL (the run's output_dir/generations.jsonl)",
    )

    grade = sub.add_parser(
        "grade", help="run the deterministic graders over generations (CPU-only)"
    )
    grade.add_argument("--config", required=True, help="path to a run TOML")
    grade.add_argument(
        "--generations",
        default=None,
        help="generations JSONL (default: output_dir/generations.jsonl)",
    )
    grade.add_argument(
        "--output-dir", default=None, help="override the config's output_dir"
    )

    judge = sub.add_parser("judge", help="LLM-as-judge the generations (resume-safe)")
    judge.add_argument("--config", required=True, help="path to a run TOML")
    judge.add_argument(
        "--generations",
        default=None,
        help="generations JSONL (default: output_dir/generations.jsonl)",
    )
    judge.add_argument(
        "--output-dir", default=None, help="override the config's output_dir"
    )
    judge.add_argument(
        "--env-file",
        default=None,
        help="dotenv file for API keys (e.g. projects/qwen-qlora/.env)",
    )
    judge.add_argument(
        "--limit", type=int, default=0, help="judge at most N samples (0 = all)"
    )
    judge.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="parallel judge requests for api/runpod backends (>= 1)",
    )
    judge.add_argument(
        "--against",
        default=None,
        help="P1-style legacy judge_scores.jsonl: print Pearson/Spearman "
        "per dimension after judging (M2 port validation)",
    )

    pairwise = sub.add_parser(
        "pairwise",
        help="position-swapped pairwise judge between two conditions",
    )
    pairwise.add_argument("--config", required=True, help="path to a run TOML")
    pairwise.add_argument(
        "--condition-a", required=True, help="first condition name (Answer A)"
    )
    pairwise.add_argument(
        "--condition-b", required=True, help="second condition name (Answer B)"
    )
    pairwise.add_argument(
        "--generations",
        default=None,
        help="generations JSONL (default: output_dir/generations.jsonl)",
    )
    pairwise.add_argument(
        "--output-dir", default=None, help="override the config's output_dir"
    )
    pairwise.add_argument(
        "--env-file",
        default=None,
        help="dotenv file for API keys (e.g. projects/qwen-qlora/.env)",
    )
    pairwise.add_argument(
        "--limit", type=int, default=0, help="compare at most N pairs (0 = all)"
    )
    pairwise.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="parallel judge requests for api/runpod backends (>= 1)",
    )
    calibrate_cmd = sub.add_parser(
        "calibrate",
        help="calibrate the judge against human labels (M4, PLAN section 5)",
    )
    calibrate_cmd.add_argument(
        "--run",
        required=True,
        help="run name under reports/ (or a path to the run directory)",
    )
    calibrate_cmd.add_argument(
        "--labels",
        required=True,
        help="human labels JSONL (PLAN section 3.4; see tools/LABELING.md)",
    )
    calibrate_cmd.add_argument(
        "--output-dir",
        default=None,
        help="output directory (default: reports/calibration/<run>)",
    )
    report_cmd = sub.add_parser(
        "report",
        help="write the run's summary.json + report.md (M5, PLAN sections 6-7)",
    )
    report_cmd.add_argument(
        "--run",
        required=True,
        help="run name under reports/ (or a path to the run directory)",
    )
    report_cmd.add_argument(
        "--config",
        default=None,
        help="the run's TOML (provenance: task-file hash, dimension order)",
    )
    gate_cmd = sub.add_parser(
        "gate",
        help="gate the run's summary against a committed baseline (M5, PLAN 7)",
    )
    gate_cmd.add_argument(
        "--run",
        required=True,
        help="run name under reports/ (or a path to the run dir / summary.json)",
    )
    gate_cmd.add_argument(
        "--baseline",
        required=True,
        help="committed baseline summary.json (e.g. baselines/tutor.json)",
    )
    gate_cmd.add_argument(
        "--config",
        required=True,
        help="gate thresholds TOML (e.g. configs/gate.toml)",
    )
    gate_cmd.add_argument(
        "--output-dir",
        default=None,
        help="output directory (default: the run directory)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "generate":
        return run_generation(
            args.config,
            limit=args.limit,
            concurrency=args.concurrency,
            env_file=args.env_file,
            output_dir=args.output_dir,
        )
    if args.command == "import-tasks":
        return _cmd_import_tasks(args)
    if args.command == "import-generations":
        return _cmd_import_generations(args)
    if args.command == "grade":
        return run_grading(
            args.config,
            generations=args.generations,
            output_dir=args.output_dir,
        )
    if args.command == "judge":
        return run_judging(
            args.config,
            generations=args.generations,
            output_dir=args.output_dir,
            env_file=args.env_file,
            limit=args.limit,
            concurrency=args.concurrency,
            against=args.against,
        )
    if args.command == "pairwise":
        return run_pairwise(
            args.config,
            args.condition_a,
            args.condition_b,
            generations=args.generations,
            output_dir=args.output_dir,
            env_file=args.env_file,
            limit=args.limit,
            concurrency=args.concurrency,
        )
    if args.command == "calibrate":
        return _cmd_calibrate(args)
    if args.command == "report":
        return _cmd_report(args)
    if args.command == "gate":
        return run_gate(
            args.run,
            args.baseline,
            args.config,
            output_dir=args.output_dir,
        )
    return 2  # pragma: no cover - argparse enforces a known command


if __name__ == "__main__":
    raise SystemExit(main())
