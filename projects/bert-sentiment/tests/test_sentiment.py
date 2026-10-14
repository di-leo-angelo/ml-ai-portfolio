"""Unit tests for the BERT sentiment components (no dataset download needed)."""

from __future__ import annotations

import json

import numpy as np
import pytest
from datasets import Dataset, DatasetDict
from sentiment.data import QUICK_TRAIN_SIZE, prepare_splits, quick_subset, tokenize
from sentiment.env import parse_env_local
from sentiment.evaluate import compute_metrics
from sentiment.model_card import build_model_card
from transformers import EvalPrediction

FAKE_WIDTH = 6


def _fake_tokenizer() -> object:
    """Tokenizer duck-type with fixed-width output; no network access."""

    class FakeTokenizer:
        def __call__(
            self,
            texts: list[str],
            truncation: bool = True,
            max_length: int = FAKE_WIDTH,
        ) -> dict:
            return {
                "input_ids": [[7] * max_length for _ in texts],
                "attention_mask": [[1] * max_length for _ in texts],
            }

    return FakeTokenizer()


def _toy_dataset(n: int) -> Dataset:
    """Small in-memory SST-2-shaped dataset."""
    return Dataset.from_dict(
        {
            "sentence": [f"movie review sentence number {i}" for i in range(n)],
            "label": [i % 2 for i in range(n)],
        }
    )


def test_compute_metrics_known_values() -> None:
    labels = np.array([1, 1, 0, 0, 1, 0, 1, 1])
    pred_classes = np.array([1, 0, 0, 1, 1, 0, 1, 0])
    logits = np.full((8, 2), 1.0)
    logits[np.arange(8), pred_classes] = 2.0
    metrics = compute_metrics(EvalPrediction(label_ids=labels, predictions=logits))
    assert metrics["accuracy"] == 0.625
    expected_f1 = 2 * 0.75 * 0.6 / (0.75 + 0.6)  # precision/recall from counts
    assert metrics["f1"] == pytest.approx(expected_f1, rel=1e-12)


def test_compute_metrics_empty() -> None:
    metrics = compute_metrics(
        EvalPrediction(
            label_ids=np.array([], dtype=np.int64),
            predictions=np.array([]),
        )
    )
    assert metrics == {"accuracy": 0.0, "f1": 0.0}


def test_tokenize_drops_text_and_sets_width() -> None:
    tokenized = tokenize(_toy_dataset(4), _fake_tokenizer(), max_length=FAKE_WIDTH)
    assert "sentence" not in tokenized.column_names
    assert "label" in tokenized.column_names
    row = tokenized[0]
    assert len(row["input_ids"]) == FAKE_WIDTH
    assert len(row["attention_mask"]) == FAKE_WIDTH


def test_quick_subset_caps_and_preserves_order() -> None:
    dataset = _toy_dataset(100)
    subset = quick_subset(dataset, 64)
    assert len(subset) == 64
    assert subset[0]["sentence"] == dataset[0]["sentence"]
    assert len(quick_subset(dataset, 1_000)) == 100


def test_prepare_splits_quick_size(tmp_path) -> None:
    dataset = DatasetDict(
        {
            "train": _toy_dataset(40),
            "validation": _toy_dataset(20),
            "test": _toy_dataset(5),
        }
    )
    data_dir = tmp_path / "sst2"
    dataset.save_to_disk(str(data_dir))

    train_ds, val_ds = prepare_splits(
        data_dir, tokenizer=_fake_tokenizer(), quick=True, quick_train_size=12
    )
    assert len(train_ds) == 12
    assert len(val_ds) == 20  # validation always kept in full


def test_prepare_splits_full_size(tmp_path) -> None:
    dataset = DatasetDict(
        {
            "train": _toy_dataset(40),
            "validation": _toy_dataset(20),
            "test": _toy_dataset(5),
        }
    )
    data_dir = tmp_path / "sst2"
    dataset.save_to_disk(str(data_dir))
    train_ds, val_ds = prepare_splits(data_dir, tokenizer=_fake_tokenizer())
    assert len(train_ds) == 40
    assert len(val_ds) == 20


def test_parse_env_local(tmp_path) -> None:
    env_file = tmp_path / ".env.local"
    env_file.write_text(
        "# comment\n\nHF_TOKEN=hf_abc123\nOTHER='quoted value'\n",
        encoding="utf-8",
    )
    values = parse_env_local(env_file)
    assert values == {"HF_TOKEN": "hf_abc123", "OTHER": "quoted value"}
    assert parse_env_local(tmp_path / "missing") == {}


def test_quick_train_size_constant() -> None:
    assert QUICK_TRAIN_SIZE > 0


def test_build_model_card_contains_metrics(tmp_path) -> None:
    state = {
        "model": "bert-base-uncased",
        "train_samples": 67_349,
        "val_samples": 872,
        "epochs_completed": 3,
        "global_steps": 12_600,
        "quick": False,
        "best_eval": {"accuracy": 0.9412, "f1": 0.9408, "epoch": 2},
        "hyperparameters": {
            "learning_rate": 2e-5,
            "batch_size": 16,
            "weight_decay": 0.01,
            "warmup_ratio": 0.1,
            "warmup_steps": 1260,
            "max_grad_norm": 1.0,
            "max_length": 128,
        },
    }
    (tmp_path / "training_state.json").write_text(json.dumps(state), encoding="utf-8")
    card = build_model_card(
        tmp_path, "di-leo-angelo/sst2-sentiment-bert-base", "bert-base-uncased"
    )
    assert "di-leo-angelo/sst2-sentiment-bert-base" in card
    assert "0.9412" in card
    assert "base_model:bert-base-uncased" in card
    assert "text-classification" in card
    assert "67,349" in card
