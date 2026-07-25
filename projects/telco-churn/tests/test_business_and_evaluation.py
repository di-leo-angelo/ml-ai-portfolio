"""Tests for business decisions and evaluation reporting."""

import numpy as np
import pandas as pd
import pytest
import train
from business import (
    BusinessAssumptions,
    calculate_business_impact,
    calculate_sensitivity,
    select_business_threshold,
)
from evaluate import calculate_metrics, calculate_segment_metrics


@pytest.fixture
def outcomes() -> tuple[pd.Series, np.ndarray]:
    """Return simple labels and ordered churn probabilities."""
    return pd.Series([0, 0, 1, 1]), np.array([0.1, 0.4, 0.6, 0.9])


def test_business_impact_uses_explicit_assumptions(
    outcomes: tuple[pd.Series, np.ndarray],
) -> None:
    target, probability = outcomes
    assumptions = BusinessAssumptions(
        contact_cost=10,
        customer_value=500,
        retention_success_rate=0.20,
    )

    impact = calculate_business_impact(
        target,
        probability,
        threshold=0.5,
        assumptions=assumptions,
    )

    assert impact["targeted_customers"] == 2
    assert impact["cost_of_doing_nothing"] == 1000
    assert impact["campaign_cost"] == 20
    assert impact["cost_with_model"] == 820
    assert impact["estimated_savings"] == 180


def test_business_threshold_maximizes_oof_savings(
    outcomes: tuple[pd.Series, np.ndarray],
) -> None:
    target, probability = outcomes
    assumptions = BusinessAssumptions(
        contact_cost=60,
        customer_value=500,
        retention_success_rate=0.20,
    )

    threshold, impact = select_business_threshold(
        target,
        probability,
        assumptions=assumptions,
    )

    assert threshold == pytest.approx(0.6)
    assert impact["estimated_savings"] == 80


def test_sensitivity_changes_only_intervention_assumption(
    outcomes: tuple[pd.Series, np.ndarray],
) -> None:
    target, probability = outcomes
    assumptions = BusinessAssumptions()

    results = calculate_sensitivity(
        target,
        probability,
        threshold=0.5,
        assumptions=assumptions,
    )

    assert [result["estimated_savings"] for result in results] == [80, 180, 280]
    assert [result["assumptions"]["retention_success_rate"] for result in results] == [
        0.1,
        0.2,
        0.3,
    ]


def test_metrics_and_segments_use_frozen_threshold(
    outcomes: tuple[pd.Series, np.ndarray],
) -> None:
    target, probability = outcomes
    features = pd.DataFrame(
        {
            "Contract": ["Monthly", "Annual", "Monthly", "Annual"],
            "InternetService": ["DSL", "DSL", "Fiber", "Fiber"],
        }
    )

    metrics = calculate_metrics(target, probability, threshold=0.5)
    segments = calculate_segment_metrics(
        features,
        target,
        probability,
        threshold=0.5,
    )

    assert metrics["precision"] == 1
    assert metrics["recall"] == 1
    assert segments["Contract"]["Monthly"]["sample_count"] == 2
    assert segments["InternetService"]["Fiber"]["targeting_rate"] == 1


def test_nested_oof_predictions_never_fit_on_validation_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    features = pd.DataFrame({"value": np.arange(20)})
    target = pd.Series([0, 1] * 10)

    class FoldPredictor:
        def __init__(self, fit_indices: set[int]) -> None:
            self.fit_indices = fit_indices

        def predict_proba(self, validation: pd.DataFrame) -> np.ndarray:
            assert self.fit_indices.isdisjoint(validation.index)
            probability = np.full(len(validation), 0.5)
            return np.column_stack((1 - probability, probability))

    def fake_tune(
        _pipeline,
        fold_features: pd.DataFrame,
        _fold_target: pd.Series,
        *,
        cv,
    ) -> FoldPredictor:
        assert cv is not None
        return FoldPredictor(set(fold_features.index))

    monkeypatch.setattr(train, "tune_logistic_regression", fake_tune)

    probability = train.generate_nested_oof_probabilities(
        object(),
        features,
        target,
        outer_cv=train.make_cv(n_splits=5),
        inner_splits=2,
    )

    np.testing.assert_allclose(probability, 0.5)
