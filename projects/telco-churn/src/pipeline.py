"""Leakage-safe preprocessing and model pipeline definitions."""

from collections.abc import Iterable

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 123

NUMERIC_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]


def get_categorical_features(
    feature_columns: Iterable[str],
) -> list[str]:
    """Return all feature columns not explicitly treated as numeric."""
    feature_columns = list(feature_columns)

    missing_numeric = set(NUMERIC_FEATURES) - set(feature_columns)
    if missing_numeric:
        raise ValueError(
            f"Missing expected numeric features: {sorted(missing_numeric)}"
        )

    # SeniorCitizen remains categorical despite being stored as 0/1.
    return [column for column in feature_columns if column not in NUMERIC_FEATURES]


def build_preprocessor(
    feature_columns: Iterable[str],
) -> ColumnTransformer:
    """Build numeric and categorical preprocessing pipelines."""
    categorical_features = get_categorical_features(feature_columns)

    numeric_pipeline = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(
                    drop="if_binary",
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        [
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ]
    )


def build_model_pipeline(
    estimator: BaseEstimator,
    feature_columns: Iterable[str],
) -> Pipeline:
    """Combine preprocessing and an estimator in one pipeline."""
    return Pipeline(
        [
            (
                "preprocessor",
                build_preprocessor(feature_columns),
            ),
            (
                "model",
                estimator,
            ),
        ]
    )


def build_candidate_models(
    feature_columns: Iterable[str],
    *,
    random_state: int = RANDOM_STATE,
) -> dict[str, Pipeline]:
    """Build the baseline and candidate classification pipelines."""
    feature_columns = list(feature_columns)

    return {
        "dummy": build_model_pipeline(
            DummyClassifier(strategy="prior"),
            feature_columns,
        ),
        "logistic_regression": build_model_pipeline(
            LogisticRegression(
                max_iter=1000,
                random_state=random_state,
            ),
            feature_columns,
        ),
        "random_forest": build_model_pipeline(
            RandomForestClassifier(
                n_estimators=300,
                random_state=random_state,
                n_jobs=-1,
            ),
            feature_columns,
        ),
        "hist_gradient_boosting": build_model_pipeline(
            HistGradientBoostingClassifier(
                random_state=random_state,
            ),
            feature_columns,
        ),
    }
