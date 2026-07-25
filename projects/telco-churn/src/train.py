"""Cross-validated model comparison and hyperparameter selection.

TODO:
    - Compare every candidate against the dummy baseline.
    - Use StratifiedKFold with a fixed random seed.
    - Report mean and standard deviation for each selection metric.
    - Tune only the strongest justified candidate.
    - Never inspect the holdout set during model selection.
"""
