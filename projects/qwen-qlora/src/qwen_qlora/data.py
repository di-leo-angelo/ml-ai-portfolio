"""Chat-format dataset logic shared by generation, training, and evaluation."""

from __future__ import annotations

import random
from typing import Any

SYSTEM_PROMPT = (
    "You are a patient, rigorous tutor who answers school-level questions "
    "clearly and step by step, checks your own work, and keeps every reply "
    "under about 300 words."
)


def build_user_prompt(question: str, subject: str, difficulty: str) -> str:
    """Assemble the generator prompt for one synthetic sample."""
    return (
        f"Subject: {subject}\n"
        f"Difficulty: {difficulty}\n"
        f"Reference question: {question}\n\n"
        "Create ONE new school-level question in this subject at this "
        "difficulty, topically similar but different from the reference "
        "question. Then answer it as a patient, rigorous tutor: step by "
        "step, concise, no more than 300 words. Return STRICT JSON: "
        '{"question": "...", "answer": "..."}'
    )


def clean_answer(answer: str, max_chars: int) -> str | None:
    """Strip scaffolding a generator sometimes adds; reject over-long text."""
    text = answer.strip()
    for prefix in ("ASSISTANT:", "Assistant:", "Response:", "Answer:"):
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()
    if not text or len(text) > max_chars:
        return None
    return text


def normalize_record(raw: dict, max_chars: int) -> dict | None:
    """Validate one raw generation record; return None if it is unusable."""
    answer = clean_answer(str(raw.get("answer", "")), max_chars)
    if answer is None:
        return None
    question = str(raw.get("question", "")).strip()
    subject = str(raw.get("subject", "")).strip()
    difficulty = str(raw.get("difficulty", "")).strip()
    if not question or not subject or not difficulty:
        return None
    return {
        "system": SYSTEM_PROMPT,
        "user": question,
        "assistant": answer,
        "subject": subject,
        "difficulty": difficulty,
    }


def split_balanced(
    records: list[dict],
    ratio: tuple[float, float, float],
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Stratified train/dev/test split; every subject appears in each split.

    The shuffle is seeded per subject, so the split is deterministic for a
    given record set and seed.
    """
    train_ratio, dev_ratio, _ = ratio
    by_subject: dict[str, list[dict]] = {}
    for record in records:
        by_subject.setdefault(record["subject"], []).append(record)

    buckets: dict[str, list[dict]] = {"train": [], "dev": [], "test": []}
    for subject in sorted(by_subject):
        items = by_subject[subject]
        random.Random(f"{seed}-{subject}").shuffle(items)
        n_train = int(round(len(items) * train_ratio))
        n_dev = int(round(len(items) * dev_ratio))
        buckets["train"].extend(items[:n_train])
        buckets["dev"].extend(items[n_train : n_train + n_dev])
        buckets["test"].extend(items[n_train + n_dev :])

    rng = random.Random(seed)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    return buckets["train"], buckets["dev"], buckets["test"]


def build_texts(
    tokenizer: Any, records: list[dict], enable_thinking: bool = False
) -> list[str]:
    """Render each record's messages with the tokenizer's chat template.

    Qwen3.x chat templates default to a reasoning block; the tutor answers
    in this project are direct step-by-step replies, so thinking is turned
    off for both training targets and inference prompts.
    """
    texts: list[str] = []
    for record in records:
        messages = [
            {"role": "system", "content": record["system"]},
            {"role": "user", "content": record["user"]},
            {"role": "assistant", "content": record["assistant"]},
        ]
        texts.append(
            tokenizer.apply_chat_template(
                messages, tokenize=False, enable_thinking=enable_thinking
            )
        )
    return texts
