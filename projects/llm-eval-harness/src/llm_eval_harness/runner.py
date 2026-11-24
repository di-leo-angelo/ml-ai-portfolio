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
from .config import (
    RunConfig,
    apply_env_file,
    judge_model_string,
    load_run_config,
)
from .graders import grade_record
from .graders.judge import (
    JudgeScoreStore,
    judge_model_warnings,
    pairwise_compare,
)
from .legacy import import_jsonl, p1_key_to_task_id
from .stats.agreement import pearson, spearman
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
            **(
                {"dimension_anchors": dict(config.judge.dimension_specs)}
                if config.judge.dimension_specs is not None
                else {}
            ),
            "batch_size": config.judge.batch_size,
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


# --- M2: grading, judging, pairwise ------------------------------------------


def read_generations(path: Path) -> list[dict[str, Any]]:
    """Load a generations file (tolerates a torn final line, like the store)."""
    if not path.is_file():
        raise FileNotFoundError(
            f"generations file not found: {path} (run `generate` or "
            "`import-generations` first)"
        )
    lines = [
        line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    records: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                continue  # torn tail from a crash mid-write
            raise ValueError(f"malformed line {index + 1} in {path}") from None
    return records


def run_grading(
    config_path: str | Path,
    generations: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> int:
    """Run every registered deterministic grader over ``generations.jsonl``.

    Pure CPU (no backends, no network): reads the generations + tasks,
    writes ``grades.jsonl`` recomputed whole - P1's auto_grade.py does
    the same. Returns 0.
    """
    config = load_run_config(config_path)
    tasks = load_run_tasks(config.data)
    tasks_by_id = {task.id: task for task in tasks}
    out_dir = Path(output_dir) if output_dir else Path(config.output_dir)
    gen_path = Path(generations) if generations else out_dir / "generations.jsonl"
    records = read_generations(gen_path)
    rows: list[dict[str, Any]] = []
    for record in records:
        task_id = str(record.get("task_id", ""))
        if task_id not in tasks_by_id:
            raise ValueError(
                f"generation references unknown task {task_id!r} (not in {config.data})"
            )
        task = tasks_by_id[task_id]
        row: dict[str, Any] = {
            "task_id": task_id,
            "condition": record.get("condition"),
            "seed": record.get("seed"),
            "model": record.get("model"),
        }
        row.update(grade_record(task, str(record.get("output", ""))))
        rows.append(row)
    out_path = out_dir / "grades.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    for condition in sorted({str(r["condition"]) for r in rows}):
        rows_c = [r for r in rows if r["condition"] == condition]
        math_rows = [r for r in rows_c if r.get("math")]
        n_match = sum(1 for r in math_rows if r.get("exact_match") is True)
        exact = (
            f"{100 * n_match / len(math_rows):.0f}% ({n_match}/{len(math_rows)})"
            if math_rows
            else "-"
        )
        steps = f"{100 * sum(r['step_scaffolding'] for r in rows_c) / len(rows_c):.0f}%"
        cap = f"{100 * sum(r['within_word_cap'] for r in rows_c) / len(rows_c):.0f}%"
        print(
            f"[{condition}] n={len(rows_c)} exact={exact} steps={steps} under-cap={cap}"
        )
    print(f"graded {len(rows)} records -> {out_path}")
    return 0


def _correlation_report(
    legacy_path: str | Path, out_dir: Path, config: RunConfig
) -> None:
    """Pearson/Spearman of P0 judge scores vs a P1-style legacy file.

    P1 rows: ``{"key": "10:base", "correctness": 5, ...}``; keys map
    onto harness (task_id, condition) pairs via
    ``legacy.p1_key_to_task_id``. This is the M2 port-validation check
    (acceptance: >= 0.8 per dimension). A dimension with fewer than 2
    overlapping rows, or with zero variance (correlation undefined -
    possible on a tiny ``--limit`` smoke), is reported as such instead
    of raising.
    """
    legacy: dict[str, dict[str, Any]] = {}
    for line in Path(legacy_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        legacy[str(raw["key"])] = raw
    judge_path = out_dir / "judge_scores.jsonl"
    if not judge_path.is_file():
        print(f"no judge_scores.jsonl in {out_dir} to compare")
        return
    ours: dict[str, dict[str, Any]] = {}
    for line in judge_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            raw = json.loads(line)
            ours[str(raw["key"])] = raw
    mapped = {
        f"{p1_key_to_task_id(key)[0]}:{p1_key_to_task_id(key)[1]}": key
        for key in legacy
    }
    print(f"correlation vs P1 legacy {legacy_path}")
    for dim in config.judge.dimension_names:
        xs: list[float] = []
        ys: list[float] = []
        for key, row in ours.items():
            legacy_key = mapped.get(key)
            if legacy_key is None or dim not in row or dim not in legacy[legacy_key]:
                continue
            xs.append(float(row[dim]))
            ys.append(float(legacy[legacy_key][dim]))
        if len(xs) < 2:
            print(f"  {dim}: n={len(xs)} - not enough overlap")
            continue
        try:
            print(
                f"  {dim}: n={len(xs)} pearson={pearson(xs, ys):.3f} "
                f"spearman={spearman(xs, ys):.3f}"
            )
        except ValueError as exc:
            # Zero-variance dimension (constant scores) makes the
            # correlation undefined; report it instead of crashing the
            # run after a fully successful judging pass.
            print(f"  {dim}: n={len(xs)} - undefined ({exc})")


def run_judging(
    config_path: str | Path,
    generations: str | Path | None = None,
    output_dir: str | Path | None = None,
    env_file: str | Path | None = None,
    limit: int = 0,
    concurrency: int = 1,
    against: str | Path | None = None,
) -> int:
    """Judge ``generations.jsonl`` with the config's judge backend (M2).

    Resume-safe via ``judge_scores.jsonl``; parse coverage is counted
    per batch and reported. Returns 0 on success, 1 when batches failed
    after retries, 2 when the judge==testee guard trips. ``against``
    adds a Pearson/Spearman report vs a P1 legacy file.
    """
    if env_file is not None:
        apply_env_file(env_file)
    config = load_run_config(config_path)
    for warning in config.judge_warnings:
        print(warning, flush=True)
    tasks = load_run_tasks(config.data)
    tasks_by_id = {task.id: task for task in tasks}
    out_dir = Path(output_dir) if output_dir else Path(config.output_dir)
    gen_path = Path(generations) if generations else out_dir / "generations.jsonl"
    records = read_generations(gen_path)

    # Data-driven guard: the models that actually produced the outputs.
    judge_model = judge_model_string(config.judge.backend) or ""
    testee_models = {str(record.get("model") or "") for record in records} - {""}
    try:
        for warning in judge_model_warnings(judge_model, testee_models):
            print(warning, flush=True)
    except ValueError as exc:
        print(f"refusing to judge: {exc}", flush=True)
        return 2

    backend = build_backend("judge", config.judge.backend, concurrency)
    from .graders.judge import judge_pending  # noqa: PLC0415 - late import

    exit_code = judge_pending(
        backend, config.judge, records, tasks_by_id, out_dir, limit
    )
    if against is not None:
        _correlation_report(against, out_dir, config)
    return exit_code


def run_pairwise(
    config_path: str | Path,
    condition_a: str,
    condition_b: str,
    generations: str | Path | None = None,
    output_dir: str | Path | None = None,
    env_file: str | Path | None = None,
    limit: int = 0,
    concurrency: int = 1,
) -> int:
    """Position-swapped pairwise judge between two conditions (M2 smoke).

    Pairs every task present in both conditions' generations, judges it
    twice ((A,B) then (B,A)), and writes a resume-safe ``pairwise.jsonl``
    (keyed by task_id). A preference is recorded only when both
    directions agree. Returns 0.
    """
    if condition_a == condition_b:
        raise ValueError("condition_a and condition_b must differ")
    if env_file is not None:
        apply_env_file(env_file)
    config = load_run_config(config_path)
    for warning in config.judge_warnings:
        print(warning, flush=True)
    names = {condition.name for condition in config.conditions}
    missing = {condition_a, condition_b} - names
    if missing:
        raise ValueError(f"unknown condition(s) {sorted(missing)} in config")
    tasks = load_run_tasks(config.data)
    tasks_by_id = {task.id: task for task in tasks}
    out_dir = Path(output_dir) if output_dir else Path(config.output_dir)
    gen_path = Path(generations) if generations else out_dir / "generations.jsonl"
    records = read_generations(gen_path)

    by_condition: dict[str, dict[str, dict[str, Any]]] = {}
    for record in records:
        condition = str(record.get("condition"))
        by_condition.setdefault(condition, {})[str(record["task_id"])] = record
    if condition_a not in by_condition or condition_b not in by_condition:
        raise ValueError(
            f"generations {gen_path} lack condition {condition_a!r} or {condition_b!r}"
        )
    pairs = [
        task_id
        for task_id in by_condition[condition_a]
        if task_id in by_condition[condition_b] and task_id in tasks_by_id
    ]
    store = JudgeScoreStore(out_dir / "pairwise.jsonl")
    done = store.existing_keys()
    todo = [task_id for task_id in pairs if task_id not in done]
    if limit > 0:
        todo = todo[:limit]
    if not todo:
        print(f"nothing to compare ({len(done)} already in pairwise.jsonl)")
        return 0
    rubric = config.judge.rubric()
    backend = build_backend("judge", config.judge.backend, concurrency)
    counts: dict[str, int] = {}
    for index, task_id in enumerate(todo, 1):
        task = tasks_by_id[task_id]
        result = pairwise_compare(
            backend,
            rubric,
            task.prompt,
            task.reference,
            by_condition[condition_a][task_id].get("output", ""),
            by_condition[condition_b][task_id].get("output", ""),
            temperature=config.judge.temperature,
            max_tokens=config.judge.max_tokens,
        )
        row = {
            "key": task_id,
            "task_id": task_id,
            "condition_a": condition_a,
            "condition_b": condition_b,
            "preference": result["preference"],
            "forward": result["forward"],
            "backward": result["backward"],
            "judge_model": backend.model,
        }
        store.append_row(row)
        counts[result["preference"]] = counts.get(result["preference"], 0) + 1
        print(f"[{index}/{len(todo)}] {task_id}: {result['preference']}")
    print(
        f"done: {dict(sorted(counts.items()))} over {len(todo)} pairs "
        f"({len(pairs) - len(todo)} skipped). Output: {out_dir / 'pairwise.jsonl'}"
    )
    return 0
