"""Cross-validated model comparison, tuning, and final training."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from evaluate import evaluate_model
from pipeline import RANDOM_STATE, build_candidate_models
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate

from data import (
    load_data,
    make_features_and_target,
    split_data,
)

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_DIR / "models" / "best_model.joblib"
DEFAULT_CV_RESULTS_PATH = PROJECT_DIR / "reports" / "metrics" / "cv_results.json"

SCORING = {
    "pr_auc": "average_precision",
    "roc_auc": "roc_auc",
}

LOGISTIC_PARAM_GRID = {
    "model__C": [0.01, 0.1, 1, 10],
    "model__class_weight": [None, "balanced"],
}


def make_cv(
    *,
    n_splits: int = 5,
    random_state: int = RANDOM_STATE,
) -> StratifiedKFold:
    """Create a reproducible stratified CV splitter."""
    return StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )


def compare_models(
    models: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    cv: StratifiedKFold | None = None,
) -> pd.DataFrame:
    """Compare candidate models with stratified cross-validation."""
    cv = cv or make_cv()
    rows: list[dict[str, float | str]] = []

    for name, model in models.items():
        scores = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring=SCORING,
            n_jobs=-1,
        )
        rows.append(
            {
                "model": name,
                "pr_auc_mean": float(scores["test_pr_auc"].mean()),
                "pr_auc_std": float(scores["test_pr_auc"].std()),
                "roc_auc_mean": float(scores["test_roc_auc"].mean()),
                "roc_auc_std": float(scores["test_roc_auc"].std()),
            }
        )

    results = pd.DataFrame(rows).sort_values(
        "pr_auc_mean",
        ascending=False,
    )
    return results.reset_index(drop=True)


def tune_logistic_regression(
    logistic_pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    cv: StratifiedKFold | None = None,
) -> GridSearchCV:
    """Tune logistic regression using PR-AUC as the selection metric."""
    cv = cv or make_cv()

    grid = GridSearchCV(
        logistic_pipeline,
        param_grid=LOGISTIC_PARAM_GRID,
        scoring="average_precision",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    grid.fit(X_train, y_train)
    return grid


def save_cv_results(
    results: pd.DataFrame,
    path: str | Path = DEFAULT_CV_RESULTS_PATH,
) -> None:
    """Write cross-validation comparison results to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        results.to_json(orient="records", indent=2) + "\n",
        encoding="utf-8",
    )


def save_model(
    model,
    path: str | Path = DEFAULT_MODEL_PATH,
) -> None:
    """Persist the fitted best model."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def main() -> None:
    """Train, select, evaluate, and save the Telco churn model."""
    data = load_data()
    features, target = make_features_and_target(data)
    X_train, X_test, y_train, y_test = split_data(features, target)

    models = build_candidate_models(X_train.columns)
    cv = make_cv()

    comparison = compare_models(models, X_train, y_train, cv=cv)
    save_cv_results(comparison)

    print("Cross-validated model comparison")
    print(comparison.to_string(index=False))
    print()

    # Notebook result: logistic regression was the strongest candidate.
    grid = tune_logistic_regression(
        models["logistic_regression"],
        X_train,
        y_train,
        cv=cv,
    )

    print("Best logistic params:")
    print(json.dumps(grid.best_params_, indent=2))
    print(f"Best CV PR-AUC: {grid.best_score_:.3f}")
    print()

    best_model = grid.best_estimator_
    save_model(best_model)

    metrics = evaluate_model(
        best_model,
        X_test,
        y_test,
        threshold=0.5,
    )

    print("Holdout evaluation")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
