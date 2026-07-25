# Telco Customer Churn

An end-to-end tabular classification project that predicts whether a fictional
telecommunications customer will churn. The focus is leakage-safe
preprocessing, model comparison, threshold selection, and honest evaluation.

> **Status:** In progress — project scaffold created, analysis not yet completed.

## Problem

Customer retention teams cannot contact every subscriber. A churn model can
rank customers by risk so that limited retention resources are directed toward
the people most likely to leave.

- **Target:** `Churn` (`Yes` or `No`)
- **Prediction unit:** one customer
- **False negative:** a likely churner is missed
- **False positive:** a retention offer is spent on a customer who would stay

The data describes a fictional company, so any financial impact must be clearly
labelled as a scenario based on assumptions—not as real-world savings.

## Data

The dataset is IBM's **Telco Customer Churn** sample:

- 7,043 customers
- 20 candidate predictors plus the `Churn` target
- Demographics, subscribed services, contract details, tenure, and charges

Download it from the IBM GitHub repository:

```bash
python projects/telco-churn/scripts/download_data.py
```

The CSV is saved to `data/raw/` and intentionally excluded from Git. See
[data/README.md](data/README.md) for the direct source and data notes.

## Planned workflow

1. Audit data types, missing values, target balance, and duplicate customers.
2. Create a stratified holdout test set before exploratory modeling.
3. Build preprocessing with `ColumnTransformer` inside each model pipeline.
4. Establish a `DummyClassifier` baseline.
5. Compare logistic regression and tree-based models using cross-validation.
6. Tune only promising models using training data.
7. Choose a decision threshold based on stated retention costs.
8. Evaluate once on the untouched test set.
9. Discuss calibration, important features, limitations, and failure modes.

## Evaluation plan

Model selection will use stratified cross-validation on the training set.

- **Primary metric:** average precision (PR-AUC), because churn is imbalanced
- **Secondary metrics:** ROC-AUC and log loss
- **At the chosen threshold:** precision, recall, F1, and confusion matrix
- **Probability quality:** calibration curve and Brier score

The final report will show uncertainty across CV folds and one set of metrics
on the untouched holdout set. Test results will not be used to tune the model.

## Structure

```text
telco-churn/
├── data/                  # Source notes and ignored local CSV
├── notebooks/             # Numbered exploration notebooks
├── reports/figures/       # Figures used in this README
├── scripts/               # Reproducible utility commands
├── src/                   # Data, pipeline, training, and evaluation modules
└── tests/                 # Tests for project logic
```

## Run locally

From the repository root:

```bash
uv sync --extra dev
python projects/telco-churn/scripts/download_data.py
jupyter lab
```

Training and evaluation commands will be added after their implementations.

## Results

Not available yet. This section will eventually contain:

- A cross-validated model comparison
- Holdout metrics and confusion matrix
- The selected threshold and its cost assumptions
- Calibration and feature interpretation
- Limitations and next steps
