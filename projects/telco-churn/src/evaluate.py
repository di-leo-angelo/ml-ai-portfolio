"""Final holdout evaluation and reporting."""

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_FIGURES_DIR = PROJECT_DIR / "reports" / "figures"
DEFAULT_METRICS_PATH = PROJECT_DIR / "reports" / "metrics" / "metrics.json"


def calculate_metrics(
    y_true: pd.Series,
    y_probability: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, float | int]:
    """Calculate probability and threshold-based classification metrics."""
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1.")

    y_predicted = (y_probability >= threshold).astype(int)

    true_negative, false_positive, false_negative, true_positive = confusion_matrix(
        y_true, y_predicted, labels=[0, 1]
    ).ravel()

    return {
        "threshold": threshold,
        "sample_count": len(y_true),
        "positive_rate": float(y_true.mean()),
        "pr_auc": float(average_precision_score(y_true, y_probability)),
        "roc_auc": float(roc_auc_score(y_true, y_probability)),
        "brier_score": float(brier_score_loss(y_true, y_probability)),
        "accuracy": float(accuracy_score(y_true, y_predicted)),
        "precision": float(
            precision_score(
                y_true,
                y_predicted,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                y_predicted,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                y_predicted,
                zero_division=0,
            )
        ),
        "true_negatives": int(true_negative),
        "false_positives": int(false_positive),
        "false_negatives": int(false_negative),
        "true_positives": int(true_positive),
    }


def save_evaluation_figures(
    y_true: pd.Series,
    y_probability: np.ndarray,
    *,
    threshold: float = 0.5,
    output_dir: str | Path = DEFAULT_FIGURES_DIR,
) -> None:
    """Save ROC, PR, calibration, and confusion-matrix figures."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    y_predicted = (y_probability >= threshold).astype(int)

    figure, axis = plt.subplots(figsize=(7, 5))
    RocCurveDisplay.from_predictions(
        y_true,
        y_probability,
        ax=axis,
    )
    axis.set_title("ROC curve")
    figure.tight_layout()
    figure.savefig(output_dir / "roc_curve.png", dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 5))
    PrecisionRecallDisplay.from_predictions(
        y_true,
        y_probability,
        ax=axis,
    )
    axis.set_title("Precision-recall curve")
    figure.tight_layout()
    figure.savefig(
        output_dir / "precision_recall_curve.png",
        dpi=150,
    )
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 5))
    CalibrationDisplay.from_predictions(
        y_true,
        y_probability,
        n_bins=10,
        strategy="quantile",
        ax=axis,
    )
    axis.set_title("Probability calibration")
    figure.tight_layout()
    figure.savefig(output_dir / "calibration_curve.png", dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_predicted,
        ax=axis,
    )
    axis.set_title(f"Confusion matrix (threshold = {threshold:.2f})")
    figure.tight_layout()
    figure.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close(figure)


def save_metrics(
    metrics: dict[str, float | int],
    path: str | Path = DEFAULT_METRICS_PATH,
) -> None:
    """Write evaluation metrics to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    threshold: float = 0.5,
) -> dict[str, float | int]:
    """Evaluate a fitted probabilistic classifier on holdout data."""
    if not hasattr(model, "predict_proba"):
        raise TypeError("Model must implement predict_proba().")

    y_probability = model.predict_proba(X_test)[:, 1]

    metrics = calculate_metrics(
        y_test,
        y_probability,
        threshold=threshold,
    )

    save_evaluation_figures(
        y_test,
        y_probability,
        threshold=threshold,
    )
    save_metrics(metrics)

    return metrics
