"""Import P1 (qwen-qlora) eval sets into harness task specs.

P1's split files (``data/test.jsonl`` etc.) are JSONL of chat-format
records:

    {"system": "...", "user": "...", "assistant": "...",
     "subject": "algebra", "difficulty": "beginner"}

The 230-row ``test.jsonl`` is the corpus every later project is measured
against; this importer turns it into harness tasks with a stable
``source`` marker so results trace back to the original split.

The committed 10-row ``samples/seed_prompts.jsonl`` uses a different
shape (``question`` instead of ``user``/``assistant``); it is handled
too, with no reference.
"""

from __future__ import annotations

import json
from pathlib import Path

from .tasks import Task, task_from_dict

SOURCE_TAG = "qwen-qlora"

_CHAT_KEYS: tuple[str, ...] = ("system", "user", "assistant")


def _detect_shape(raw: dict) -> str:
    if all(key in raw for key in _CHAT_KEYS):
        return "chat"
    if "question" in raw:
        return "seed"
    raise ValueError(
        "unrecognised P1 record: expected chat fields "
        f"{list(_CHAT_KEYS)} or a 'question' field"
    )


def import_record(raw: dict, line_no: int) -> Task:
    """Convert one P1 record (chat split or seed prompt) into a Task."""
    shape = _detect_shape(raw)
    if shape == "chat":
        return task_from_dict(
            {
                "id": f"p1_{line_no:04d}",
                "prompt": str(raw["user"]),
                "system": raw.get("system"),
                "reference": raw.get("assistant"),
                "subject": raw.get("subject"),
                "difficulty": raw.get("difficulty"),
                "source": f"{SOURCE_TAG}:chat",
            },
            line_no,
        )
    return task_from_dict(
        {
            "id": f"p1seed_{line_no:04d}",
            "prompt": str(raw["question"]),
            "subject": raw.get("subject"),
            "difficulty": raw.get("difficulty"),
            "source": f"{SOURCE_TAG}:seed",
        },
        line_no,
    )


def import_jsonl(path: str | Path) -> list[Task]:
    """Import a P1 JSONL file (chat split or seed prompts) into tasks."""
    text = Path(path).read_text(encoding="utf-8")
    tasks: list[Task] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        raw = json.loads(line)
        tasks.append(import_record(raw, line_no))
    if not tasks:
        raise ValueError(f"no importable records in {path}")
    return tasks
