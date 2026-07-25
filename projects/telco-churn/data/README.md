# Dataset

This project uses IBM's classic **Telco Customer Churn** sample. It represents
7,043 customers of a fictional telecommunications company.

- IBM project: <https://github.com/IBM/customer-churn-prediction>
- Direct CSV: <https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv>
- IBM documentation: <https://www.ibm.com/docs/en/cognos-analytics/12.0.x?topic=samples-telco-customer-churn>

Download the data reproducibly from the repository root:

```bash
python projects/telco-churn/scripts/download_data.py
```

This creates:

```text
data/raw/Telco-Customer-Churn.csv
```

The CSV is not committed. The download script verifies its expected columns and
row count, preventing an HTML error page or changed file from being accepted
silently.

## Important data notes

- `customerID` is an identifier, not a model feature.
- `SeniorCitizen` is coded as `0`/`1` but is categorical in meaning.
- `TotalCharges` contains blank strings for some new customers and needs numeric
  conversion plus missing-value handling.
- `Churn` is the binary target.
- All preprocessing must be fitted only on training folds.

The dataset is synthetic/sample data. Do not present estimated business impact
as an observed result from a real company.
