"""Download and validate the SST-2 dataset (GLUE) from the Hugging Face Hub."""

from __future__ import annotations

from pathlib import Path

from datasets import load_dataset

PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_DIR / "data" / "sst2"
#: Fully-qualified Hub id (huggingface_hub 2.x rejects bare single-name ids).
DATASET_ID = "nyu-mll/glue"
DATASET_CONFIG = "sst2"
EXPECTED_SIZES = {"train": 67_349, "validation": 872, "test": 1_821}
EXPECTED_LABELS = ["negative", "positive"]


def main() -> None:
    """Fetch glue/sst2 from the Hub, validate its shape, and cache it on disk."""
    if (OUTPUT_DIR / "dataset_dict.index").is_file():
        print(f"Dataset already present at {OUTPUT_DIR}")
        return

    dataset = load_dataset(DATASET_ID, DATASET_CONFIG)

    for name, expected in EXPECTED_SIZES.items():
        found = len(dataset[name])
        if found != expected:
            raise ValueError(f"Expected {expected:,} {name} rows, found {found:,}")

    label_names = dataset["train"].features["label"].names
    if label_names != EXPECTED_LABELS:
        raise ValueError(f"Expected labels {EXPECTED_LABELS}, found {label_names}")

    dataset.save_to_disk(str(OUTPUT_DIR))
    print(f"Dataset ready at {OUTPUT_DIR}")
    print(
        "67,349 train | 872 validation | 1,821 test "
        "(test labels are not public: GLUE evaluation server)"
    )


if __name__ == "__main__":
    main()
