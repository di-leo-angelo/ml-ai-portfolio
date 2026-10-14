"""Training loop with AdamW, ReduceLROnPlateau, and early stopping."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import NUM_CLASSES, SEED, load_splits
from .model import CifarCNN

PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_DIR / "data"
DEFAULT_MODELS_DIR = PROJECT_DIR / "models"
DEFAULT_FIGURES_DIR = PROJECT_DIR / "reports" / "figures"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the training command-line arguments."""
    parser = argparse.ArgumentParser(description="Train the CIFAR-10 CNN.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Training epochs (default: 100, or 15 with --quick).",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Stop after this many epochs without a val-loss improvement.",
    )
    parser.add_argument(
        "--lr-patience",
        type=int,
        default=5,
        help="Shrink the learning rate after this many stalled epochs.",
    )
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Train a small subset for a fast pipeline check.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Override the number of CPU threads for PyTorch.",
    )
    return parser.parse_args(argv)


def make_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
    """Build the train/validation loaders for the selected dataset size."""
    train_ds, val_ds, _test_ds = load_splits(args.data_dir, quick=args.quick)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=0
    )
    return train_loader, val_loader


def train_epoch(
    model: CifarCNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    """Run one full pass over the training set; return the mean batch loss."""
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        loss = loss_fn(model(images), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(images)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: CifarCNN,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Return (loss, accuracy) over the loader's dataset."""
    model.eval()
    total_loss = 0.0
    correct = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        total_loss += loss_fn(logits, labels).item() * len(images)
        correct += int((logits.argmax(dim=1) == labels).sum())
    size = len(loader.dataset)
    return total_loss / size, correct / size


def plot_history(history: list[dict], path: Path) -> None:
    """Plot train/val loss and val accuracy over epochs."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="val")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[1].plot(epochs, [row["val_accuracy"] for row in history])
    axes[1].set_title("Validation accuracy")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    """Run the full training loop with LR scheduling and early stopping."""
    args = parse_args(argv)
    if args.epochs is None:
        args.epochs = 15 if args.quick else 100
    torch.manual_seed(SEED)
    if args.threads is not None:
        torch.set_num_threads(args.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = make_loaders(args)
    model = CifarCNN(num_classes=NUM_CLASSES).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_factor,
        patience=args.lr_patience,
        min_lr=args.min_lr,
    )

    args.models_dir.mkdir(parents=True, exist_ok=True)
    suffix = "quick" if args.quick else "best"
    checkpoint_path = args.models_dir / f"{suffix}.pt"
    history_path = args.models_dir / f"{suffix}_history.json"
    figure_path = DEFAULT_FIGURES_DIR / f"learning_curve_{suffix}.png"

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    history: list[dict] = []
    print(f"Device: {device} | parameters: {model.count_parameters():,}")
    print(
        f"Train samples: {len(train_loader.dataset):,} | "
        f"epochs: {args.epochs} | patience: {args.patience}"
    )

    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.perf_counter() - started

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                },
                checkpoint_path,
            )
            marker = " *saved*"
        else:
            epochs_without_improvement += 1
            marker = ""

        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 5),
                "val_loss": round(val_loss, 5),
                "val_accuracy": round(val_acc, 5),
                "lr": current_lr,
                "epoch_seconds": round(elapsed, 1),
            }
        )
        print(
            f"epoch {epoch:3d}/{args.epochs} | train_loss {train_loss:.4f} | "
            f"val_loss {val_loss:.4f} | val_acc {val_acc:.4f} | "
            f"lr {current_lr:.6f} | {elapsed:5.1f}s{marker}"
        )

        if epochs_without_improvement >= args.patience:
            print(
                f"Early stopping at epoch {epoch} "
                f"(no val-loss improvement in {args.patience} epochs)."
            )
            break
    else:
        print(f"Finished all {args.epochs} epochs.")

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model"])
    plot_history(history, figure_path)
    best_info = {
        key: checkpoint[key]
        for key in ("epoch", "train_loss", "val_loss", "val_accuracy")
    }
    history.append({"best": best_info})
    history_path.write_text(json.dumps(history, indent=2))
    print(f"Best val loss {best_info['val_loss']:.4f} at epoch {best_info['epoch']}")
    print(f"Checkpoint -> {checkpoint_path}")
    print(f"History    -> {history_path}")
    print(f"Plot       -> {figure_path}")


if __name__ == "__main__":
    main()
