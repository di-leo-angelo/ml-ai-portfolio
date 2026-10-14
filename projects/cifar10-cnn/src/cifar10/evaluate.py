"""Evaluate the trained checkpoint on the official test set."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from .data import NUM_CLASSES, load_splits
from .model import CifarCNN

PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT_PATH = PROJECT_DIR / "models" / "best.pt"
DEFAULT_DATA_DIR = PROJECT_DIR / "data"
QUICK_TEST_SIZE = 2_000


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the evaluation command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate the CIFAR-10 checkpoint on the test set."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--quick",
        action="store_true",
        help=f"Evaluate only the first {QUICK_TEST_SIZE:,} test images.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Load the best checkpoint and report overall + per-class test accuracy."""
    args = parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _train_ds, _val_ds, test_ds = load_splits(args.data_dir)
    class_names = test_ds.classes  # type: ignore[attr-defined]
    if args.quick:
        test_ds = Subset(test_ds, range(QUICK_TEST_SIZE))
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    checkpoint_path = args.checkpoint
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_DIR / checkpoint_path
    checkpoint = torch.load(checkpoint_path)
    model = CifarCNN(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    total = 0
    correct = 0
    class_correct = [0] * NUM_CLASSES
    class_total = [0] * NUM_CLASSES
    with torch.no_grad():
        for images, labels in loader:
            preds = model(images.to(device)).argmax(dim=1).cpu()
            total += labels.numel()
            correct += int((preds == labels).sum())
            for class_idx in range(NUM_CLASSES):
                mask = labels == class_idx
                class_total[class_idx] += int(mask.sum())
                class_correct[class_idx] += int(((preds == labels) & mask).sum())

    names = class_names
    print(f"Checkpoint: {checkpoint_path} (best epoch {checkpoint['epoch']})")
    print(f"Test accuracy: {correct / total:.4f} ({correct:,}/{total:,})")
    print("Per-class accuracy:")
    for name, correct_c, total_c in zip(names, class_correct, class_total, strict=True):
        print(f"  {name:<12} {correct_c / total_c:.4f}  ({correct_c}/{total_c})")


if __name__ == "__main__":
    main()
