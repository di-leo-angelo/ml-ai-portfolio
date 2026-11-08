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
import re
from pathlib import Path
from typing import Any

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


# --- P1 recorded generations (M2 dogfood) -----------------------------------

#: P1 model provenance for its recorded conditions (P1: Qwen3.8-27B,
#: base vs +LoRA adapter, greedy decoding - the seed is meaningless, 0).
_P1_MODELS: dict[str, str] = {
    "lora": "Qwen3.8-27B+LoRA",
    "base": "Qwen3.8-27B",
}

_P1_KEY_RE = re.compile(r"(\d+):([A-Za-z0-9_-]+)")


def p1_key_to_task_id(key: str) -> tuple[str, str]:
    """P1's ``"N:condition"`` key -> (harness task id, condition).

    P1's ``sample_id`` is a 0-based index into ``data/test.jsonl``; the
    task importer assigns ``p1_0001`` to the first line, so sample N
    maps to ``p1_{N+1:04d}``.
    """
    match = _P1_KEY_RE.fullmatch(str(key).strip())
    if not match:
        raise ValueError(f"unrecognised P1 key: {key!r}")
    sample_id = int(match.group(1))
    return f"p1_{sample_id + 1:04d}", match.group(2)


def import_generations(path: str | Path) -> list[dict[str, Any]]:
    """Import P1's ``raw_outputs.jsonl`` into harness generation records.

    P1 records: ``{"key": "0:lora", "sample_id": 0, "subject",
    "difficulty", "condition": "lora", "wall_s": 17.0, "output": ...}``.
    Each becomes a harness record carrying the same output text under
    the (task_id, condition, seed) resume key, with the P1 model as
    ``model`` and a ``source`` marker for provenance.
    """
    records: list[dict[str, Any]] = []
    text = Path(path).read_text(encoding="utf-8")
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        raw = json.loads(line)
        missing = [
            key
            for key in ("key", "sample_id", "condition", "output")
            if key not in raw
        ]
        if missing:
            raise ValueError(f"line {line_no}: missing P1 field(s) {missing}")
        condition = str(raw["condition"])
        if condition not in _P1_MODELS:
            raise ValueError(
                f"line {line_no}: unknown P1 condition {condition!r} "
                f"(expected one of {sorted(_P1_MODELS)})"
            )
        task_id, _ = p1_key_to_task_id(str(raw["key"]))
        if int(raw["sample_id"]) + 1 != int(task_id.split("_")[1]):
            raise ValueError(
                f"line {line_no}: key {str(raw['key'])!r} disagrees with "
                f"sample_id {raw['sample_id']}"
            )
        record: dict[str, Any] = {
            "task_id": task_id,
            "condition": condition,
            "seed": 0,
            "output": str(raw["output"]),
            "backend": "imported",
            "model": _P1_MODELS[condition],
            "source": f"{SOURCE_TAG}:raw_outputs",
        }
        if raw.get("wall_s") is not None:
            record["wall_s"] = float(raw["wall_s"])
        records.append(record)
    if not records:
        raise ValueError(f"no importable generation records in {path}")
    return records
