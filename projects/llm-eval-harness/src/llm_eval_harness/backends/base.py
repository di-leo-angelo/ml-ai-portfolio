"""Backend building blocks: jobs, records, and the resume-safe store.

The generation output (``generations.jsonl``) is append-only and keyed by
``(task_id, condition, seed)``: on re-run, keys already present are
skipped, so a killed run resumes exactly where it stopped (P1's
judge_run.py pattern). Every record is flushed (and fsynced) immediately
after being written, so a crash can at worst leave a torn final line,
which the reader detects and drops.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class GenerationError(RuntimeError):
    """A generation job failed after all retries."""


@dataclass(frozen=True)
class GenerationJob:
    """One (task, condition) pair to generate output for."""

    task_id: str
    prompt: str
    seed: int
    temperature: float
    max_tokens: int
    system: str | None = None

    def messages(self) -> list[dict[str, str]]:
        """OpenAI chat messages for this job (system optional)."""
        messages: list[dict[str, str]] = []
        if self.system:
            messages.append({"role": "system", "content": self.system})
        messages.append({"role": "user", "content": self.prompt})
        return messages


@dataclass
class GenerationRecord:
    """One completed generation (one line of generations.jsonl)."""

    task_id: str
    condition: str
    seed: int
    output: str
    backend: str
    model: str
    tokens_in: int | None = None
    tokens_out: int | None = None
    wall_s: float | None = None
    cost_usd: float | None = None

    @property
    def key(self) -> tuple[str, str, int]:
        return (self.task_id, self.condition, self.seed)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "task_id": self.task_id,
            "condition": self.condition,
            "seed": self.seed,
            "output": self.output,
            "backend": self.backend,
            "model": self.model,
        }
        for name in ("tokens_in", "tokens_out", "wall_s", "cost_usd"):
            value = getattr(self, name)
            if value is not None:
                payload[name] = value
        return payload


class GenerationStore:
    """Append-only generations.jsonl with resume-safe key tracking."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def existing_keys(self) -> set[tuple[str, str, int]]:
        """Resume keys already present (empty set when the file is absent).

        A malformed final line is treated as a torn tail from a crash
        mid-write and dropped; any other malformed line fails loudly.
        """
        if not self.path.is_file():
            return set()
        lines = [
            line
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        keys: set[tuple[str, str, int]] = set()
        for index, line in enumerate(lines):
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                if index == len(lines) - 1:
                    continue  # torn tail: drop, keep resuming
                raise ValueError(f"malformed line {index + 1} in {self.path}") from None
            try:
                keys.add((str(raw["task_id"]), str(raw["condition"]), int(raw["seed"])))
            except (KeyError, TypeError, ValueError):
                raise ValueError(
                    f"missing key field(s) on line {index + 1} in {self.path}"
                ) from None
        return keys

    def append(self, record: GenerationRecord) -> None:
        """Append one record and flush it to disk immediately."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
