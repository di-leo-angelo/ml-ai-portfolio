# Tests

Add focused tests as implementation progresses. Useful first tests include:

- The loader returns the expected columns and binary target.
- `customerID` never appears in model features.
- The split is reproducible, stratified, and has no overlapping customer IDs.
- `TotalCharges` blanks are handled inside preprocessing.
- The complete pipeline can fit and predict on a small fixture.
- Evaluation rejects a model or dataset with the wrong target classes.

Tests should use small in-memory fixtures rather than the full downloaded CSV.
