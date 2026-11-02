"""Run orchestration for ``generate``: config + tasks in, outputs out.

Resume-safe end to end: generation is keyed by ``(task_id, condition,
seed)`` against ``generations.jsonl``, so re-invoking a killed run with
the same config generates only what is missing.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import __version__
from .backends import GenerationError, GenerationJob, GenerationStore, build_backend
from .config import RunConfig, apply_env_file, load_run_config
from .legacy import import_jsonl
from .tasks import Task, load_tasks


def load_run_tasks(path: str | Path) -> list[Task]:
    """Load a tasks file, accepting the harness schema or P1 shapes.

    The first non-empty line decides: a ``prompt`` field means the harness
    task schema; P1 chat (``user``) or seed (``question``) records go
    through the legacy importer.
    """
    task_path = Path(path)
    if not task_path.is_file():
        raise FileNotFoundError(f"tasks file not found: {task_path}")
    first_raw: dict[str, Any] | None = None
    for line in task_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        first_raw = json.loads(line)
        break
    if first_raw is None:
        raise ValueError(f"no tasks found in {task_path}")
    if "prompt" in first_raw:
        return load_tasks(task_path)
    if "user" in first_raw or "question" in first_raw:
        return import_jsonl(task_path)
    raise ValueError(
        f"unrecognised task file {task_path}: first line has neither "
        "'prompt' nor 'user'/'question'"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str | None:
    """Best-effort short git commit (None outside a repo or on failure)."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        return result.stdout.strip() or None
    except Exception:  # noqa: BLE001 - the manifest must never block a run
        return None


def _backend_echo(backend: Any) -> dict[str, Any]:
    """Plain-dict echo of a backend config (api-key env NAME, never a key)."""
    return {f.name: getattr(backend, f.name) for f in dataclasses.fields(backend)}


def write_manifest(
    out_dir: Path, config: RunConfig, tasks: list[Task], data_path: Path
) -> Path:
    """Write manifest.json (config echo + data hash + provenance)."""
    manifest = {
        "schema": 1,
        "name": config.name,
        "ts": datetime.now(UTC).isoformat(),
        "harness_version": __version__,
        "rubric_version": config.rubric_version,
        "git_commit": _git_commit(),
        "data": {
            "path": config.data,
            "sha256": _sha256(data_path),
            "tasks": len(tasks),
        },
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "conditions": {
            condition.name: {
                "seed": condition.seed,
                "backend": _backend_echo(condition.backend),
            }
            for condition in config.conditions
        },
        "judge": {
            "dimensions": list(config.judge.dimension_names),
            "temperature": config.judge.temperature,
            "max_tokens": config.judge.max_tokens,
            "backend": _backend_echo(config.judge.backend),
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def run_generation(
    config_path: str | Path,
    limit: int = 0,
    concurrency: int = 1,
    env_file: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> int:
    """Drive generation for every condition of a run config.

    Returns an exit code: 0 when everything is (or already was)
    generated, 1 when some jobs failed after retries (re-running resumes
    the missing ones).
    """
    if env_file is not None:
        apply_env_file(env_file)
    config = load_run_config(config_path)
    tasks = load_run_tasks(config.data)
    if limit > 0:
        tasks = tasks[:limit]
    out_dir = Path(output_dir) if output_dir else Path(config.output_dir)
    write_manifest(out_dir, config, tasks, Path(config.data))

    store = GenerationStore(out_dir / "generations.jsonl")
    done = store.existing_keys()
    generated = skipped = failed = 0
    for condition in config.conditions:
        backend = build_backend(condition.name, condition.backend, concurrency)
        jobs = [
            GenerationJob(
                task_id=task.id,
                prompt=task.prompt,
                system=task.system,
                seed=condition.seed,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )
            for task in tasks
        ]
        pending = [j for j in jobs if (j.task_id, condition.name, j.seed) not in done]
        skipped += len(jobs) - len(pending)
        if not pending:
            print(
                f"[{condition.name}] all {len(jobs)} present - nothing to do",
                flush=True,
            )
            continue
        print(
            f"[{condition.name}] generating {len(pending)}/{len(jobs)} "
            f"(skipping {len(jobs) - len(pending)} already present)",
            flush=True,
        )
        completed = 0
        try:
            for record in backend.generate(pending):
                store.append(record)
                completed += 1
        except GenerationError as exc:
            failed += len(pending) - completed
            print(
                f"[{condition.name}] STOPPED after {completed}/{len(pending)}: "
                f"{exc} (re-run to resume)",
                flush=True,
            )
        generated += completed
    print(
        f"done: {generated} generated, {skipped} skipped, {failed} failed. "
        f"Output: {out_dir / 'generations.jsonl'}",
        flush=True,
    )
    return 1 if failed else 0
