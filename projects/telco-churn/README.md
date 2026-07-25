# Telco Customer Churn

An end-to-end tabular classification project that predicts whether a fictional
telecommunications customer will churn. The implementation covers validated
data loading, leakage-safe preprocessing, model comparison, hyperparameter
tuning, holdout evaluation, generated reports, and automated tests.

> **Status:** Core training and evaluation workflow implemented.

## Problem

Customer retention teams cannot contact every subscriber. A churn model can
rank customers by risk so that limited retention resources are directed toward
the people most likely to leave.

- **Target:** `Churn` (`Yes` or `No`)
- **Prediction unit:** one customer
- **False negative:** a likely churner is missed
- **False positive:** a retention offer is spent on a customer who would stay

The data describes a fictional company. Financial impact must therefore be
presented as a scenario based on assumptions, not as observed business savings.

## Data

The project uses IBM's **Telco Customer Churn** sample:

- 7,043 customers
- 20 candidate predictors plus the `Churn` target
- Demographics, subscribed services, contract details, tenure, and charges

The download script retrieves the CSV from IBM, checks its columns and row
count, and saves it under `data/raw/`. The raw file is intentionally excluded
from Git. See [data/README.md](data/README.md) for source details and data notes.

## Method

1. Validate required columns, target values, and unique customer IDs.
2. Convert `TotalCharges` to numeric, preserving blanks as missing values.
3. Remove `customerID` and map the target to `0` and `1`.
4. Create a reproducible 80/20 stratified train/test split.
5. Keep imputation, scaling, and one-hot encoding inside each model pipeline to
   prevent preprocessing leakage.
6. Compare a dummy baseline, logistic regression, random forest, and histogram
   gradient boosting with five-fold stratified cross-validation.
7. Tune logistic regression on training data using PR-AUC.
8. Evaluate the selected model once on the untouched holdout set.

PR-AUC is the primary selection metric because churn is the minority class.
ROC-AUC, Brier score, threshold metrics, calibration, and the confusion matrix
provide complementary views of model quality.

## Results

Logistic regression was the strongest candidate:

- Logistic regression: CV PR-AUC `0.657 ± 0.034`, ROC-AUC `0.846 ± 0.015`
- Histogram gradient boosting: CV PR-AUC `0.641 ± 0.032`
- Random forest: CV PR-AUC `0.620 ± 0.033`
- Dummy baseline: CV PR-AUC `0.265`

Grid search selected `C=10` with no class weighting. On the 1,409-customer
holdout set, using the default `0.50` decision threshold, it achieved:

- PR-AUC: `0.650`
- ROC-AUC: `0.844`
- Brier score: `0.135`
- Accuracy: `0.797`
- Precision: `0.643`
- Recall: `0.529`
- F1: `0.581`
- Confusion matrix: 925 true negatives, 110 false positives, 176 false
  negatives, and 198 true positives

![Precision-recall curve](reports/figures/precision_recall_curve.png)

![Confusion matrix](reports/figures/confusion_matrix.png)

Additional ROC and calibration plots are generated under `reports/figures/`.

## Reproduce the project

From the repository root, install the project and development dependencies:

```bash
uv sync --extra dev
```

Download and validate the dataset:

```bash
uv run python projects/telco-churn/scripts/download_data.py
```

Run model comparison, tuning, final training, and holdout evaluation:

```bash
uv run python projects/telco-churn/src/train.py
```

The training command writes:

- `models/best_model.joblib`
- `reports/metrics/cv_results.json`
- `reports/metrics/metrics.json`
- ROC, precision-recall, calibration, and confusion-matrix figures

The dataset, fitted model, and generated metric JSON files are ignored by Git.
The report figures are retained for presentation.

## Run the tests

Run the complete repository test suite:

```bash
uv run pytest
```

Run only this project's tests:

```bash
uv run pytest projects/telco-churn/tests
```

The tests use small synthetic fixtures so they remain fast, deterministic,
offline, and independent of the ignored raw CSV. They cover data conversion,
duplicate-ID validation, feature/target separation, reproducible stratified
splitting, and a full pipeline fit/predict smoke test.

Run the quality checks with:

```bash
uv run ruff check .
uv run ruff format --check .
```

## Structure

```text
telco-churn/
├── data/                  # Dataset documentation and ignored raw CSV
├── notebooks/             # Notebook guidance
├── reports/
│   ├── figures/           # Generated evaluation plots
│   └── metrics/           # Ignored generated metric files
├── scripts/
│   └── download_data.py   # Download and source validation
├── src/
│   ├── data.py            # Loading, validation, and splitting
│   ├── evaluate.py        # Metrics and report figures
│   ├── pipeline.py        # Preprocessing and model pipelines
│   └── train.py           # Comparison, tuning, training, and evaluation
└── tests/                 # Project-local tests using synthetic fixtures
```

## Limitations and next steps

- The dataset is a public fictional sample, not current production data.
- The `0.50` threshold is a default operating point, not a
  business-cost-optimized threshold.
- Performance may change under population or service-offering drift.
- The current project does not estimate causal effects of retention actions.
- A future iteration could select a threshold from out-of-fold predictions
  using explicit retention costs and validate performance on newer data.
