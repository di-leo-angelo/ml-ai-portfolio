"""Decision-threshold selection and transparent business scenario analysis."""

from dataclasses import asdict, dataclass, replace

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


@dataclass(frozen=True)
class BusinessAssumptions:
    """Illustrative retention-campaign assumptions, not observed company values."""

    contact_cost: float = 10.0
    customer_value: float = 500.0
    retention_success_rate: float = 0.20

    def __post_init__(self) -> None:
        if self.contact_cost < 0:
            raise ValueError("contact_cost must be non-negative.")
        if self.customer_value <= 0:
            raise ValueError("customer_value must be positive.")
        if not 0 <= self.retention_success_rate <= 1:
            raise ValueError("retention_success_rate must be between 0 and 1.")


def calculate_business_impact(
    y_true: pd.Series | np.ndarray,
    y_probability: np.ndarray,
    *,
    threshold: float,
    assumptions: BusinessAssumptions,
) -> dict[str, float | int | dict[str, float]]:
    """Estimate campaign economics under explicitly stated assumptions."""
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1.")

    y_true_array = np.asarray(y_true)
    y_probability = np.asarray(y_probability)
    if y_true_array.shape[0] != y_probability.shape[0]:
        raise ValueError("y_true and y_probability must have equal lengths.")

    y_predicted = (y_probability >= threshold).astype(int)
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(
        y_true_array,
        y_predicted,
        labels=[0, 1],
    ).ravel()

    targeted_customers = int(y_predicted.sum())
    actual_churners = int(y_true_array.sum())
    expected_prevented_churns = true_positive * assumptions.retention_success_rate
    campaign_cost = targeted_customers * assumptions.contact_cost
    cost_of_doing_nothing = actual_churners * assumptions.customer_value
    avoided_churn_loss = expected_prevented_churns * assumptions.customer_value
    cost_with_model = cost_of_doing_nothing - avoided_churn_loss + campaign_cost
    estimated_savings = cost_of_doing_nothing - cost_with_model

    return {
        "threshold": float(threshold),
        "assumptions": asdict(assumptions),
        "targeted_customers": targeted_customers,
        "actual_churners": actual_churners,
        "true_positives": int(true_positive),
        "false_positives": int(false_positive),
        "false_negatives": int(false_negative),
        "true_negatives": int(true_negative),
        "expected_prevented_churns": float(expected_prevented_churns),
        "campaign_cost": float(campaign_cost),
        "false_positive_contact_cost": float(false_positive * assumptions.contact_cost),
        "false_negative_churn_loss": float(false_negative * assumptions.customer_value),
        "cost_of_doing_nothing": float(cost_of_doing_nothing),
        "cost_with_model": float(cost_with_model),
        "estimated_savings": float(estimated_savings),
    }


def select_business_threshold(
    y_true: pd.Series | np.ndarray,
    y_probability: np.ndarray,
    *,
    assumptions: BusinessAssumptions,
) -> tuple[float, dict[str, float | int | dict[str, float]]]:
    """Choose the OOF threshold with the highest estimated campaign savings."""
    probabilities = np.asarray(y_probability)
    candidates = np.unique(np.concatenate(([0.0], probabilities, [1.0])))

    impacts = [
        calculate_business_impact(
            y_true,
            probabilities,
            threshold=float(threshold),
            assumptions=assumptions,
        )
        for threshold in candidates
    ]
    best = max(
        impacts,
        key=lambda impact: (
            float(impact["estimated_savings"]),
            float(impact["threshold"]),
        ),
    )
    return float(best["threshold"]), best


def calculate_sensitivity(
    y_true: pd.Series | np.ndarray,
    y_probability: np.ndarray,
    *,
    threshold: float,
    assumptions: BusinessAssumptions,
    success_rates: tuple[float, ...] = (0.10, 0.20, 0.30),
) -> list[dict[str, float | int | dict[str, float]]]:
    """Recalculate holdout economics across intervention-effect assumptions."""
    return [
        calculate_business_impact(
            y_true,
            y_probability,
            threshold=threshold,
            assumptions=replace(
                assumptions,
                retention_success_rate=success_rate,
            ),
        )
        for success_rate in success_rates
    ]
