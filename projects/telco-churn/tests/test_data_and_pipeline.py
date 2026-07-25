"""Tests for the Telco churn data and modeling pipeline."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal
from pipeline import build_candidate_models

from data import load_data, make_features_and_target, split_data


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Return a small, balanced Telco-like dataset."""
    row_count = 20
    return pd.DataFrame(
        {
            "customerID": [f"CUSTOMER-{index:02d}" for index in range(row_count)],
            "Churn": ["No", "Yes"] * (row_count // 2),
            "tenure": list(range(1, row_count + 1)),
            "MonthlyCharges": np.linspace(20, 100, row_count),
            "TotalCharges": [" "] + [str(value * 50) for value in range(1, row_count)],
            "SeniorCitizen": [0, 1] * (row_count // 2),
            "Contract": ["Month-to-month", "One year"] * (row_count // 2),
        }
    )


def write_csv(data: pd.DataFrame, path: Path) -> Path:
    """Write fixture data to a temporary CSV."""
    data.to_csv(path, index=False)
    return path


def test_load_data_converts_total_charges(
    sample_data: pd.DataFrame,
    tmp_path: Path,
) -> None:
    path = write_csv(sample_data, tmp_path / "telco.csv")

    loaded = load_data(path)

    assert pd.api.types.is_numeric_dtype(loaded["TotalCharges"])
    assert loaded["TotalCharges"].isna().sum() == 1


def test_load_data_rejects_duplicate_customer_ids(
    sample_data: pd.DataFrame,
    tmp_path: Path,
) -> None:
    sample_data.loc[1, "customerID"] = sample_data.loc[0, "customerID"]
    path = write_csv(sample_data, tmp_path / "duplicates.csv")

    with pytest.raises(ValueError, match="duplicate customer IDs"):
        load_data(path)


def test_features_and_split_are_reproducible(
    sample_data: pd.DataFrame,
    tmp_path: Path,
) -> None:
    data = load_data(write_csv(sample_data, tmp_path / "telco.csv"))
    features, target = make_features_and_target(data)

    assert "customerID" not in features
    assert "Churn" not in features
    assert set(target.unique()) == {0, 1}

    first = split_data(features, target, test_size=0.2)
    second = split_data(features, target, test_size=0.2)

    assert_frame_equal(first[0], second[0])
    assert_frame_equal(first[1], second[1])
    assert_series_equal(first[2], second[2])
    assert_series_equal(first[3], second[3])
    assert set(first[0].index).isdisjoint(first[1].index)
    assert first[2].mean() == target.mean()
    assert first[3].mean() == target.mean()


def test_logistic_pipeline_fits_and_predicts(
    sample_data: pd.DataFrame,
    tmp_path: Path,
) -> None:
    data = load_data(write_csv(sample_data, tmp_path / "telco.csv"))
    features, target = make_features_and_target(data)
    models = build_candidate_models(features.columns)

    probabilities = (
        models["logistic_regression"].fit(features, target).predict_proba(features)
    )

    assert probabilities.shape == (len(features), 2)
    assert np.all((probabilities >= 0) & (probabilities <= 1))
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
