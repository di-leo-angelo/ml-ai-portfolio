"""Unit tests for the Qwen3 QLoRA data pipeline (no model weights needed)."""

from __future__ import annotations

from pathlib import Path

import pytest
from qwen_qlora.config import (
    DATASET_CONFIG,
    PROJECT_DIR,
    hash_file,
    load_jsonl,
    save_jsonl,
)
from qwen_qlora.data import (
    SYSTEM_PROMPT,
    build_texts,
    build_user_prompt,
    normalize_record,
    split_balanced,
)
from qwen_qlora.env import parse_extra_headers


class _FakeTokenizer:
    """Minimal stand-in: renders messages without loading a real tokenizer."""

    def apply_chat_template(
        self, messages: list[dict], tokenize: bool = False, **kwargs
    ) -> str:
        return " | ".join(message["content"] for message in messages)


@pytest.fixture()
def tiny_records() -> list[dict]:
    records: list[dict] = []
    for subject in ("algebra", "chemistry"):
        for index in range(10):
            records.append(
                {
                    "system": SYSTEM_PROMPT,
                    "user": f"Question {subject}-{index}",
                    "assistant": f"Answer {subject}-{index}",
                    "subject": subject,
                    "difficulty": "beginner",
                }
            )
    return records


def test_seed_prompts_cover_all_subjects() -> None:
    seed_file = PROJECT_DIR / "data" / "samples" / "seed_prompts.jsonl"
    seeds = load_jsonl(seed_file)
    assert {seed["subject"] for seed in seeds} == set(DATASET_CONFIG.subjects)
    for seed in seeds:
        assert seed["question"].strip()
        assert seed["difficulty"] in DATASET_CONFIG.difficulties


def test_jsonl_roundtrip(tmp_path: Path) -> None:
    records = [{"a": 1}, {"b": "text"}]
    path = tmp_path / "out.jsonl"
    save_jsonl(records, path)
    assert load_jsonl(path) == records


def test_hash_file_is_stable(tmp_path: Path) -> None:
    path = tmp_path / "data.jsonl"
    path.write_text("hello\nworld\n", encoding="utf-8")
    digest = hash_file(path)
    assert len(digest) == 64
    assert hash_file(path) == digest


def test_build_user_prompt_includes_recipe() -> None:
    prompt = build_user_prompt("Solve 2x = 4", "algebra", "beginner")
    for fragment in (
        "Subject: algebra",
        "Difficulty: beginner",
        "Solve 2x = 4",
        '"question"',
        '"answer"',
    ):
        assert fragment in prompt


def test_normalize_record_accepts_valid_input() -> None:
    raw = {
        "question": "What is 2 + 2?",
        "answer": "2 + 2 = 4.",
        "subject": "algebra",
        "difficulty": "beginner",
    }
    record = normalize_record(raw, max_chars=200)
    assert record is not None
    assert record["system"] == SYSTEM_PROMPT
    assert record["user"] == "What is 2 + 2?"
    assert record["assistant"] == "2 + 2 = 4."


def test_normalize_record_strips_scaffolding_prefix() -> None:
    raw = {
        "question": "What is mitosis?",
        "answer": "ASSISTANT: Mitosis is cell division.",
        "subject": "biology",
        "difficulty": "beginner",
    }
    record = normalize_record(raw, max_chars=200)
    assert record is not None
    assert record["assistant"].startswith("Mitosis")


@pytest.mark.parametrize(
    "raw",
    [
        {"question": "", "answer": "A", "subject": "algebra", "difficulty": "b"},
        {"question": "Q", "answer": "", "subject": "algebra", "difficulty": "b"},
        {"question": "Q", "answer": "A" * 50, "subject": "algebra", "difficulty": "b"},
    ],
)
def test_normalize_record_rejects_invalid_input(raw: dict) -> None:
    assert normalize_record(raw, max_chars=20) is None


def test_split_balanced_is_stratified(tiny_records: list[dict]) -> None:
    train, dev, test = split_balanced(tiny_records, (0.7, 0.2, 0.1))
    assert len(train) + len(dev) + len(test) == len(tiny_records)
    for bucket in (train, dev, test):
        assert {record["subject"] for record in bucket} == {"algebra", "chemistry"}


def test_split_balanced_is_deterministic(tiny_records: list[dict]) -> None:
    train_first, _, _ = split_balanced(tiny_records, (0.7, 0.2, 0.1))
    train_second, _, _ = split_balanced(tiny_records, (0.7, 0.2, 0.1))
    assert [record["user"] for record in train_first] == [
        record["user"] for record in train_second
    ]


def test_build_texts_uses_chat_template() -> None:
    record = {
        "system": SYSTEM_PROMPT,
        "user": "What is 2 + 2?",
        "assistant": "It is 4.",
    }
    texts = build_texts(_FakeTokenizer(), [record])
    assert len(texts) == 1
    assert "What is 2 + 2?" in texts[0]
    assert "It is 4." in texts[0]


def test_parse_extra_headers() -> None:
    headers = parse_extra_headers("x-custom-auth=abc123; other = 42")
    assert headers == {"x-custom-auth": "abc123", "other": "42"}
    assert parse_extra_headers("") == {}
    assert parse_extra_headers("no-equals-pair") == {}
