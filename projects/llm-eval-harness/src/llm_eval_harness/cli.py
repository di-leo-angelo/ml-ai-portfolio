"""Command line interface: ``python -m llm_eval_harness <command>``.

M1 commands: ``generate`` (drive generation for a run config) and
``import-tasks`` (convert a P1 split/seed file to the harness schema).
M2 commands: ``import-generations`` (P1 raw_outputs -> harness
generations), ``grade`` (deterministic graders, CPU-only), ``judge``
(LLM-as-judge, resume-safe, ``--against`` correlation vs a P1 legacy
file), ``pairwise`` (position-swapped pairwise smoke). Later milestones
add calibrate / report / gate (PLAN.md).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .legacy import import_generations, import_jsonl
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
    return 2  # pragma: no cover - argparse enforces a known command


if __name__ == "__main__":
    raise SystemExit(main())
