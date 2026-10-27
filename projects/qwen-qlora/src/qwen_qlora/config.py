"""Shared constants and dataset configuration for the Qwen3 QLoRA project."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data"

#: Base model (Apache-2.0). 27.8B multimodal hybrid; fits in 24 GB at 4-bit.
DEFAULT_MODEL = "Qwen/Qwen3.8-27B"


@dataclass(frozen=True)
class DatasetConfig:
    """Declarative specification of the synthetic dataset.

    The dataset is generated from ``seed_file`` by a teacher model
    (``generator_model``); nothing else about its content is hand-written.
    This class is the single source of truth for the generation recipe.
    """

    seed_file: str
    generator_model: str
    subjects: tuple[str, ...]
    samples_per_subject: int
    difficulties: tuple[str, ...]
    split_ratio: tuple[float, float, float]
    max_chars: int


DATASET_CONFIG = DatasetConfig(
    seed_file="samples/seed_prompts.jsonl",
    generator_model="qwen-3.8-27b-nvfp4",
    subjects=(
        "algebra",
        "calculus",
        "probability",
        "chemistry",
        "physics",
        "biology",
        "history",
        "world_geography",
        "english_literature",
        "computer_science",
    ),
    samples_per_subject=150,
    difficulties=("beginner", "intermediate", "advanced"),
    split_ratio=(0.75, 0.10, 0.15),
    max_chars=1400,
)


def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file into a list of dicts, skipping blank lines."""
    records: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(records: list[dict], path: Path) -> None:
    """Atomically write records to a JSONL file (one JSON object per line).

    The content is written to a temporary file in the same directory and
    moved into place with ``os.replace`` (plus an fsync), so a process that
    dies mid-save can never truncate or corrupt the existing file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def hash_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file's bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
