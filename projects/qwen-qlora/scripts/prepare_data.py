"""Validate, split, and fingerprint the generated Q&A dataset.

Reads data/qa_raw.jsonl, splits it into train/dev/test stratified by subject
(75/10/15, fixed seed), writes the split files, and records sizes and
SHA-256 hashes in data/split.json so any dataset can be fingerprinted.

Usage (from the repository root):

    uv run python projects/qwen-qlora/scripts/prepare_data.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_FILE = PROJECT_DIR / "data" / "qa_raw.jsonl"
SPLIT_FILE = PROJECT_DIR / "data" / "split.json"
TRAIN_FILE = PROJECT_DIR / "data" / "train.jsonl"
DEV_FILE = PROJECT_DIR / "data" / "dev.jsonl"
TEST_FILE = PROJECT_DIR / "data" / "test.jsonl"

sys.path.insert(0, str(PROJECT_DIR / "src"))

from qwen_qlora.config import (  # noqa: E402
    DATASET_CONFIG,
    hash_file,
    load_jsonl,
    save_jsonl,
)
from qwen_qlora.data import split_balanced  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate, split, and fingerprint the Q&A dataset."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--min-samples",
        type=int,
        default=500,
        help="Abort if fewer than this many samples (default: %(default)s).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not RAW_FILE.exists():
        raise SystemExit(f"{RAW_FILE} not found - run scripts/generate_data.py first.")

    records = load_jsonl(RAW_FILE)
    if len(records) < args.min_samples:
        raise SystemExit(
            f"Only {len(records):,} samples (< {args.min_samples:,}); "
            "generate more before preparing splits."
        )

    by_subject: dict[str, int] = {}
    for record in records:
        by_subject[record["subject"]] = by_subject.get(record["subject"], 0) + 1

    train, dev, test = split_balanced(
        records, DATASET_CONFIG.split_ratio, seed=args.seed
    )
    save_jsonl(train, TRAIN_FILE)
    save_jsonl(dev, DEV_FILE)
    save_jsonl(test, TEST_FILE)

    manifest = {
        "seed": args.seed,
        "split_ratio": list(DATASET_CONFIG.split_ratio),
        "sizes": {"train": len(train), "dev": len(dev), "test": len(test)},
        "samples_by_subject": dict(sorted(by_subject.items())),
        "generator_model": DATASET_CONFIG.generator_model,
        "hashes": {
            "raw": hash_file(RAW_FILE),
            "train": hash_file(TRAIN_FILE),
            "dev": hash_file(DEV_FILE),
            "test": hash_file(TEST_FILE),
        },
    }
    SPLIT_FILE.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Train: {len(train):,} | Dev: {len(dev):,} | Test: {len(test):,}")
    print(f"Manifest written to {SPLIT_FILE.name}")


if __name__ == "__main__":
    main()
