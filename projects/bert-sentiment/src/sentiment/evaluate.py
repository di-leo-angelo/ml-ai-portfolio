"""Trainer metrics and standalone evaluation on the SST-2 dev set."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, TensorDataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
)

from .data import DEFAULT_DATA_DIR, MAX_LENGTH, MODEL_IDS, load_raw

PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_MODELS_DIR = PROJECT_DIR / "models"
QUICK_EVAL_SIZE = 500
LABEL_NAMES = ["negative", "positive"]


def compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
    """Accuracy and binary F1 from (labels, logits); used by the Trainer."""
    labels = eval_pred.label_ids
    if len(labels) == 0:
        return {"accuracy": 0.0, "f1": 0.0}
    preds = eval_pred.predictions.argmax(axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, average="binary", zero_division=0)),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the evaluation command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate a fine-tuned SST-2 checkpoint on the dev set."
    )
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_IDS),
        default="bert-base",
        help="Which fine-tuned checkpoint to evaluate (default: %(default)s).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint directory (default: models/<model>).",
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--quick",
        action="store_true",
        help=f"Evaluate only the first {QUICK_EVAL_SIZE} dev examples.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Report dev accuracy/F1, a confusion matrix, and error examples."""
    args = parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = args.checkpoint or DEFAULT_MODELS_DIR / args.model
    if not checkpoint.is_absolute():
        checkpoint = PROJECT_DIR / checkpoint

    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint))
    model = AutoModelForSequenceClassification.from_pretrained(str(checkpoint))
    model.to(device).eval()

    raw = load_raw(args.data_dir)
    val = raw["validation"]
    if args.quick:
        val = val.select(range(QUICK_EVAL_SIZE))
    texts = list(val["sentence"])
    labels = list(val["label"])
    encoded = tokenizer(
        texts, truncation=True, max_length=MAX_LENGTH, padding="max_length"
    )
    input_ids = torch.tensor(encoded["input_ids"], dtype=torch.long)
    attention_mask = torch.tensor(encoded["attention_mask"], dtype=torch.long)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    loader = DataLoader(
        TensorDataset(input_ids, attention_mask, labels_tensor),
        batch_size=args.batch_size,
        shuffle=False,
    )

    all_preds: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    with torch.no_grad():
        for input_ids, attention_mask, labels_batch in loader:
            logits = model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
            ).logits
            all_preds.append(logits.argmax(dim=-1).cpu())
            all_labels.append(labels_batch)

    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    hub_id = MODEL_IDS[args.model]
    f1 = float(f1_score(labels, preds, average="binary", zero_division=0))

    print(f"Checkpoint: {checkpoint} ({hub_id}, dev split of {len(labels):,} rows)")
    print(f"Dev accuracy: {float(accuracy_score(labels, preds)):.4f}")
    print(f"Dev F1 (binary): {f1:.4f}")
    print("Confusion matrix (rows=true, cols=predicted; [negative, positive]):")
    for row in confusion_matrix(labels, preds, labels=[0, 1]):
        print(f"  {row}")

    wrong = labels != preds
    if wrong.any():
        print(f"Misclassified examples (first {min(5, int(wrong.sum()))}):")
        for idx in [i for i in range(len(labels)) if wrong[i]][:5]:
            print(
                f"  true={LABEL_NAMES[labels[idx]]} "
                f"pred={LABEL_NAMES[preds[idx]]} | {texts[idx][:80]}"
            )
    else:
        print("No misclassified examples.")


if __name__ == "__main__":
    main()
