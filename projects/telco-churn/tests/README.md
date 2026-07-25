# Tests

Project-local tests live here so each portfolio project stays self-contained.

Current coverage:

- `TotalCharges` conversion and blank-to-NaN handling
- Duplicate `customerID` rejection
- Feature/target separation and reproducible stratified splits
- Logistic pipeline smoke fit and probability outputs

Use small fixtures rather than the downloaded CSV so CI does not need the raw dataset.
