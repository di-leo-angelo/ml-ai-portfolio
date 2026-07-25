"""Download and validate IBM's Telco Customer Churn sample dataset."""

import csv
from pathlib import Path
from urllib.request import urlopen

DATA_URL = (
    "https://raw.githubusercontent.com/IBM/"
    "telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
)
OUTPUT_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "raw" / "Telco-Customer-Churn.csv"
)
EXPECTED_ROWS = 7_043
EXPECTED_COLUMNS = {
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
}


def validate_csv(path: Path) -> None:
    """Raise an error when the downloaded file has an unexpected shape."""
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        columns = set(reader.fieldnames or [])
        row_count = sum(1 for _ in reader)

    if columns != EXPECTED_COLUMNS:
        missing = EXPECTED_COLUMNS - columns
        extra = columns - EXPECTED_COLUMNS
        raise ValueError(f"Unexpected columns. Missing: {missing}; extra: {extra}")
    if row_count != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS:,} rows, found {row_count:,}")


def main() -> None:
    """Download the CSV, validate it, and report its local path."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with urlopen(DATA_URL, timeout=30) as response:  # noqa: S310
        OUTPUT_PATH.write_bytes(response.read())

    try:
        validate_csv(OUTPUT_PATH)
    except Exception:
        OUTPUT_PATH.unlink(missing_ok=True)
        raise

    print(f"Downloaded {EXPECTED_ROWS:,} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
