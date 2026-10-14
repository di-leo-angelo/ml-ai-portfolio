"""Fine-tune a Hugging Face transformer on SST-2 with the Trainer API."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe: no display on CI or over SSH

import matplotlib.pyplot as plt  # noqa: E402
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from .data import DEFAULT_DATA_DIR, MAX_LENGTH, MODEL_IDS, SEED, prepare_splits
from .evaluate import compute_metrics

PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_MODELS_DIR = PROJECT_DIR / "models"
DEFAULT_RUNS_DIR = PROJECT_DIR / "runs"
DEFAULT_FIGURES_DIR = PROJECT_DIR / "reports" / "figures"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the training command-line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune a transformer on SST-2.")
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_IDS),
        default="bert-base",
        help="Base model to fine-tune (default: %(default)s; "
        "--quick forces distilbert).",
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Training epochs (default: 3, or 1 with --quick).",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="Early stop after this many epochs without a dev-accuracy gain.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Pipeline smoke test: distilbert on a 4,096-sample subset, 1 epoch.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Override the number of CPU threads for PyTorch.",
    )
    return parser.parse_args(argv)


def plot_history(log_history: list[dict], figure_path: Path) -> None:
    """Plot the train/eval loss curves from the Trainer log history."""
    train_losses = [(e["step"], e["loss"]) for e in log_history if "loss" in e]
    eval_losses = [(e["step"], e["eval_loss"]) for e in log_history if "eval_loss" in e]

    figure, ax = plt.subplots(figsize=(8, 5))
    if train_losses:
        ax.plot(
            [step for step, _ in train_losses],
            [loss for _, loss in train_losses],
            alpha=0.5,
            label="train loss",
        )
    if eval_losses:
        ax.plot(
            [step for step, _ in eval_losses],
            [loss for _, loss in eval_losses],
            marker="o",
            label="eval loss",
        )
    ax.set_xlabel("steps")
    ax.set_ylabel("loss")
    ax.set_title("Learning curve")
    ax.legend()
    figure.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_path, dpi=150)
    plt.close(figure)


def main(argv: list[str] | None = None) -> None:
    """Run the full training pipeline and save the best model."""
    args = parse_args(argv)
    if args.quick:
        args.model = "distilbert"
    if args.epochs is None:
        args.epochs = 1 if args.quick else 3

    torch.set_num_threads(args.threads or torch.get_num_threads())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hub_id = MODEL_IDS[args.model]

    tokenizer = AutoTokenizer.from_pretrained(hub_id)
    model = AutoModelForSequenceClassification.from_pretrained(hub_id)
    train_ds, val_ds = prepare_splits(
        args.data_dir,
        tokenizer=tokenizer,
        quick=args.quick,
        max_length=args.max_length,
    )

    # transformers v5 has no warmup_ratio; convert it to absolute steps.
    steps_per_epoch = math.ceil(len(train_ds) / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(args.warmup_ratio * total_steps))

    training_args = TrainingArguments(
        output_dir=str(DEFAULT_RUNS_DIR / args.model),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=warmup_steps,
        lr_scheduler_type="linear",
        max_grad_norm=1.0,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        logging_steps=50,
        dataloader_num_workers=0,
        seed=SEED,
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )

    print(f"Base model: {hub_id} | device: {device}")
    print(
        f"Train samples: {len(train_ds):,} | val samples: {len(val_ds):,} | "
        f"epochs: {args.epochs} | early-stopping patience: {args.patience}"
    )
    started = time.perf_counter()
    trainer.train()
    elapsed = time.perf_counter() - started

    # load_best_model_at_end guarantees trainer.model is the best-dev model.
    model_dir = args.models_dir / args.model
    model_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    state = trainer.state
    eval_entries = [e for e in state.log_history if "eval_accuracy" in e]
    best_entry = max(eval_entries, key=lambda e: e["eval_accuracy"])
    summary = {
        "model": hub_id,
        "seed": SEED,
        "quick": args.quick,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "epochs_completed": state.epoch,
        "global_steps": state.global_step,
        "best_metric": state.best_metric,
        "best_eval": {
            "accuracy": best_entry["eval_accuracy"],
            "f1": best_entry.get("eval_f1"),
            "epoch": best_entry.get("epoch"),
        },
        "wall_clock_seconds": round(elapsed, 1),
        "hyperparameters": {
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "warmup_steps": warmup_steps,
            "max_grad_norm": 1.0,
            "max_length": args.max_length,
        },
    }
    (model_dir / "training_state.json").write_text(json.dumps(summary, indent=2))
    (model_dir / "training_history.json").write_text(
        json.dumps(state.log_history, indent=2)
    )
    figure_path = DEFAULT_FIGURES_DIR / f"learning_curve_{args.model}.png"
    plot_history(state.log_history, figure_path)

    print(
        f"Best dev accuracy {summary['best_eval']['accuracy']:.4f} "
        f"(f1 {summary['best_eval']['f1']:.4f}) at epoch "
        f"{summary['best_eval']['epoch']} after {state.global_step:,} steps "
        f"({elapsed / 60:.1f} min)"
    )
    print(f"Model    -> {model_dir}")
    print(f"History  -> {model_dir / 'training_history.json'}")
    print(f"Plot     -> {figure_path}")


if __name__ == "__main__":
    main()
