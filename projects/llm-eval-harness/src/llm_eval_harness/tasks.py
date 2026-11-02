"""Task spec: one eval item the harness generates output for and scores.

Task files are JSONL. Each line is one task:

    {"id": "task_0001", "prompt": "...", "system": "...",
     "reference": "...", "subject": "algebra", "difficulty": "beginner",
     "tags": ["algebra"]}

``reference`` is optional (open-ended tasks may have none); ``id`` is
optional and derived as ``task_`` + zero-padded line number when absent.
The schema is intentionally small so P2 (DPO pairs) and P3 (GRPO
verifiable rewards) can reuse the same file with extra fields.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REQUIRED_FIELDS: tuple[str, ...] = ("prompt",)
OPTIONAL_FIELDS: frozenset[str] = frozenset(
    {
        "id",
        "system",
        "reference",
        "subject",
        "difficulty",
        "tags",
        "source",
    }
)


@dataclass
class Task:
    """One eval item."""

    prompt: str
    system: str | None = None
    reference: str | None = None
    id: str = ""
    subject: str | None = None
    difficulty: str | None = None
    tags: tuple[str, ...] = ()
    source: str | None = None

    def __post_init__(self) -> None:
        if not self.prompt.strip():
            raise ValueError("task prompt must be non-empty")
        if not self.id:
            raise ValueError("task id must be set (the loader fills it)")
        self.tags = tuple(self.tags)

    @property
    def has_reference(self) -> bool:
        return bool(self.reference and self.reference.strip())

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"id": self.id, "prompt": self.prompt}
        for name in ("system", "reference", "subject", "difficulty", "source"):
            value = getattr(self, name)
            if value is not None:
                payload[name] = value
        if self.tags:
            payload["tags"] = list(self.tags)
        return payload


def _coerce_tags(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value)
    raise ValueError(f"tags must be a string or list, got {type(value).__name__}")


def task_from_dict(raw: dict[str, Any], line_no: int) -> Task:
    """Validate one raw JSON object into a :class:`Task`."""
    missing = [key for key in REQUIRED_FIELDS if not str(raw.get(key, "")).strip()]
    if missing:
        raise ValueError(f"line {line_no}: missing required field(s) {missing}")
    unknown = set(raw) - set(REQUIRED_FIELDS) - set(OPTIONAL_FIELDS)
    if unknown:
        raise ValueError(f"line {line_no}: unknown field(s) {sorted(unknown)}")
    task_id = str(raw.get("id") or "").strip() or f"task_{line_no:04d}"
    return Task(
        prompt=str(raw["prompt"]),
        system=raw.get("system"),
        reference=raw.get("reference"),
        id=task_id,
        subject=raw.get("subject"),
        difficulty=raw.get("difficulty"),
        tags=_coerce_tags(raw.get("tags")),
        source=raw.get("source"),
    )


def load_tasks(path: str | Path) -> list[Task]:
    """Load and validate a JSONL task file."""
    text = Path(path).read_text(encoding="utf-8")
    tasks: list[Task] = []
    seen: set[str] = set()
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"line {line_no}: invalid JSON ({exc})") from exc
        if not isinstance(raw, dict):
            raise ValueError(f"line {line_no}: expected a JSON object")
        task = task_from_dict(raw, line_no)
        if task.id in seen:
            raise ValueError(f"line {line_no}: duplicate task id {task.id!r}")
        seen.add(task.id)
        tasks.append(task)
    if not tasks:
        raise ValueError(f"no tasks found in {path}")
    return tasks
