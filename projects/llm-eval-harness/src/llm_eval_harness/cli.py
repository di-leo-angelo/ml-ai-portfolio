"""Command line interface: ``python -m llm_eval_harness <command>``.

M1 commands: ``generate`` (drive generation for a run config) and
``import-tasks`` (convert a P1 split/seed file to the harness schema).
Later milestones add grade / judge / calibrate / report / gate (PLAN.md).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .legacy import import_jsonl
from .runner import run_generation


def _cmd_import_tasks(args: argparse.Namespace) -> int:
    tasks = import_jsonl(args.input)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps(task.to_dict(), ensure_ascii=False) + "\n")
    print(f"imported {len(tasks)} tasks -> {output}")
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
    return 2  # pragma: no cover - argparse enforces a known command


if __name__ == "__main__":
    raise SystemExit(main())
