"""Final holdout evaluation and reporting."""

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from business import (
    BusinessAssumptions,
    calculate_business_impact,
    calculate_sensitivity,
)
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


def _display_feature_names(model: Any) -> np.ndarray:
    """Return readable post-preprocessing feature names."""
    names = model.named_steps["preprocessor"].get_feature_names_out()
    return np.array(
        [name.removeprefix("numeric__").removeprefix("categorical__") for name in names]
    )


def save_coefficient_plot(
    model: Any,
    *,
    output_dir: str | Path = DEFAULT_FIGURES_DIR,
    max_features: int = 20,
) -> None:
    """Plot the largest positive and negative logistic coefficients."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    estimator = model.named_steps["model"]
    if not hasattr(estimator, "coef_"):
        raise TypeError("The fitted estimator must expose coef_.")

    names = _display_feature_names(model)
    coefficients = estimator.coef_[0]
    selected = np.argsort(np.abs(coefficients))[-max_features:]
    selected = selected[np.argsort(coefficients[selected])]

    colors = np.where(coefficients[selected] >= 0, "#d95f02", "#1b9e77")
    figure, axis = plt.subplots(figsize=(9, 7))
    axis.barh(names[selected], coefficients[selected], color=colors)
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_xlabel("Logistic coefficient (model scale)")
    axis.set_title("Largest churn-risk drivers")
    figure.tight_layout()
    figure.savefig(output_dir / "feature_coefficients.png", dpi=150)
    plt.close(figure)


def save_shap_summary(
    model: Any,
    X: pd.DataFrame,
    *,
    output_dir: str | Path = DEFAULT_FIGURES_DIR,
    max_rows: int = 500,
) -> None:
    """Save a SHAP summary for the fitted logistic pipeline."""
    import shap

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample = X.sample(
        n=min(max_rows, len(X)),
        random_state=123,
    )
    preprocessor = model.named_steps["preprocessor"]
    estimator = model.named_steps["model"]
    transformed = preprocessor.transform(sample)
    names = _display_feature_names(model)

    background = transformed[: min(100, len(transformed))]
    explainer = shap.LinearExplainer(estimator, background)
    shap_values = explainer(transformed)
    shap.summary_plot(
        shap_values.values,
        transformed,
        feature_names=names,
        max_display=20,
        show=False,
    )
    figure = plt.gcf()
    figure.suptitle("SHAP summary: contribution to predicted churn", y=1.01)
    figure.tight_layout()
    figure.savefig(
        output_dir / "shap_summary.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(figure)


def calculate_segment_metrics(
    X: pd.DataFrame,
    y_true: pd.Series,
    y_probability: np.ndarray,
    *,
    threshold: float,
    columns: tuple[str, ...] = ("Contract", "InternetService"),
) -> dict[str, dict[str, dict[str, float | int]]]:
    """Report threshold performance across selected customer segments."""
    predictions = (y_probability >= threshold).astype(int)
    report: dict[str, dict[str, dict[str, float | int]]] = {}

    for column in columns:
        if column not in X:
            continue

        column_report: dict[str, dict[str, float | int]] = {}
        for value in X[column].dropna().unique():
            mask = (X[column] == value).to_numpy()
            segment_target = y_true.to_numpy()[mask]
            segment_prediction = predictions[mask]
            column_report[str(value)] = {
                "sample_count": int(mask.sum()),
                "churn_rate": float(segment_target.mean()),
                "targeting_rate": float(segment_prediction.mean()),
                "precision": float(
                    precision_score(
                        segment_target,
                        segment_prediction,
                        zero_division=0,
                    )
                ),
                "recall": float(
                    recall_score(
                        segment_target,
                        segment_prediction,
                        zero_division=0,
                    )
                ),
            }
        report[column] = column_report

    return report


def save_business_impact_figure(
    impact: dict[str, Any],
    *,
    output_dir: str | Path = DEFAULT_FIGURES_DIR,
) -> None:
    """Visualize scenario costs and estimated savings."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = ["Do nothing", "Use model"]
    values = [
        impact["cost_of_doing_nothing"],
        impact["cost_with_model"],
    ]
    figure, axis = plt.subplots(figsize=(7, 5))
    bars = axis.bar(labels, values, color=["#7570b3", "#1b9e77"])
    axis.bar_label(bars, fmt="$%.0f", padding=3)
    axis.set_ylabel("Illustrative scenario cost")
    axis.set_title(
        f"Estimated savings: ${impact['estimated_savings']:,.0f} (assumption-based)"
    )
    figure.tight_layout()
    figure.savefig(output_dir / "business_impact.png", dpi=150)
    plt.close(figure)


def save_metrics(
    metrics: dict[str, Any],
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
    assumptions: BusinessAssumptions | None = None,
) -> dict[str, Any]:
    """Evaluate a fitted probabilistic classifier on holdout data."""
    if not hasattr(model, "predict_proba"):
        raise TypeError("Model must implement predict_proba().")

    assumptions = assumptions or BusinessAssumptions()
    y_probability = model.predict_proba(X_test)[:, 1]

    metrics = calculate_metrics(
        y_test,
        y_probability,
        threshold=threshold,
    )
    business_impact = calculate_business_impact(
        y_test,
        y_probability,
        threshold=threshold,
        assumptions=assumptions,
    )
    metrics["business_impact"] = business_impact
    metrics["business_sensitivity"] = calculate_sensitivity(
        y_test,
        y_probability,
        threshold=threshold,
        assumptions=assumptions,
    )
    metrics["segments"] = calculate_segment_metrics(
        X_test,
        y_test,
        y_probability,
        threshold=threshold,
    )

    save_evaluation_figures(
        y_test,
        y_probability,
        threshold=threshold,
    )
    save_coefficient_plot(model)
    save_shap_summary(model, X_test)
    save_business_impact_figure(business_impact)
    save_metrics(metrics)

    return metrics
