"""SST-2 dataset loading, tokenization, and quick-subset selection."""

from __future__ import annotations

from pathlib import Path

from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizerBase

PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_DIR / "data" / "sst2"

SEED = 42
# 99.96% of SST-2 training sentences fit in 64 WordPiece tokens (only 25 of
# 67,349 sentences exceed 48 words; the longest is 52 words), so the usual
# 128-token ceiling wastes ~half of every CPU forward pass.
MAX_LENGTH = 64
QUICK_TRAIN_SIZE = 4_096

#: Short CLI names mapped to their Hugging Face Hub identifiers.
MODEL_IDS = {
    "bert-base": "bert-base-uncased",
    "distilbert": "distilbert-base-uncased",
}


def load_raw(data_dir: Path = DEFAULT_DATA_DIR) -> DatasetDict:
    """Load the on-disk SST-2 dataset (train/validation/test) from disk."""
    return DatasetDict.load_from_disk(str(data_dir))


def quick_subset(dataset: Dataset, size: int) -> Dataset:
    """Return the first ``size`` rows of a dataset (deterministic)."""
    return dataset.select(range(min(size, len(dataset))))


def tokenize(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = MAX_LENGTH,
) -> Dataset:
    """Tokenize the ``text`` column and drop the original strings."""

    def preprocess(examples: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        return tokenizer(examples["sentence"], truncation=True, max_length=max_length)

    return dataset.map(preprocess, batched=True, remove_columns="sentence")


def prepare_splits(
    data_dir: Path = DEFAULT_DATA_DIR,
    *,
    tokenizer: PreTrainedTokenizerBase,
    quick: bool = False,
    quick_train_size: int = QUICK_TRAIN_SIZE,
    max_length: int = MAX_LENGTH,
) -> tuple[Dataset, Dataset]:
    """Return the tokenized (train, validation) splits.

    In quick mode the train split is truncated to the first
    ``quick_train_size`` rows; the validation split is always kept in full so
    the reported dev metric stays comparable across runs. The 1,821-row test
    split is intentionally not loaded: its labels are not public (GLUE
    evaluation server).
    """
    raw = load_raw(data_dir)
    train = raw["train"]
    if quick:
        train = quick_subset(train, quick_train_size)
    return (
        tokenize(train, tokenizer, max_length=max_length),
        tokenize(raw["validation"], tokenizer, max_length=max_length),
    )
