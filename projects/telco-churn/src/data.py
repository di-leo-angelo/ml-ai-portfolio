"""Data loading, validation, and train/test splitting."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 123
TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"

DEFAULT_DATA_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "raw" / "Telco-Customer-Churn.csv"
)

REQUIRED_COLUMNS = {
    ID_COLUMN,
    TARGET_COLUMN,
    "TotalCharges",
}


def load_data(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load and validate the raw Telco churn dataset."""
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            "Run scripts/download_data.py before training."
        )

    data = pd.read_csv(path)

    missing_columns = REQUIRED_COLUMNS - set(data.columns)
    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing_columns)}"
        )

    if data[ID_COLUMN].duplicated().any():
        raise ValueError("Dataset contains duplicate customer IDs.")

    target_values = set(data[TARGET_COLUMN].dropna().unique())
    if target_values != {"No", "Yes"}:
        raise ValueError(f"Unexpected target values: {sorted(target_values)}")

    # Blank strings become NaN; imputation happens later inside the pipeline.
    data["TotalCharges"] = pd.to_numeric(
        data["TotalCharges"],
        errors="coerce",
    )

    return data


def make_features_and_target(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate model features from the binary churn target."""
    features = data.drop(columns=[TARGET_COLUMN, ID_COLUMN]).copy()
    target = data[TARGET_COLUMN].map({"No": 0, "Yes": 1}).astype("int8")

    return features, target


def split_data(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    test_size: float = 0.20,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create a reproducible, stratified train/test split."""
    return train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )
