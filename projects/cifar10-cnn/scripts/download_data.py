"""Download and validate the CIFAR-10 dataset via torchvision."""

from __future__ import annotations

from pathlib import Path

import torchvision

PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_DIR / "data"
EXPECTED_SIZES = {"train": 50_000, "test": 10_000}
EXPECTED_NUM_CLASSES = 10


def main() -> None:
    """Fetch the CIFAR-10 archive, extract it, and validate its shape."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    datasets = {
        name: torchvision.datasets.CIFAR10(
            root=str(OUTPUT_DIR),
            train=name == "train",
            download=True,
        )
        for name in ("train", "test")
    }

    for name, dataset in datasets.items():
        expected = EXPECTED_SIZES[name]
        if len(dataset) != expected:
            raise ValueError(
                f"Expected {expected:,} {name} images, found {len(dataset):,}"
            )

    if len(datasets["train"].classes) != EXPECTED_NUM_CLASSES:
        raise ValueError(
            f"Expected {EXPECTED_NUM_CLASSES} classes, "
            f"found {len(datasets['train'].classes)}"
        )

    print(f"Dataset ready at {OUTPUT_DIR} (50,000 train | 10,000 test)")


if __name__ == "__main__":
    main()
