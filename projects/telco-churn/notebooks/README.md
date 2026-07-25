# Notebooks

Use notebooks for exploration and communication, not as the only implementation.

Suggested sequence:

1. `01_eda.ipynb` — data audit, target balance, distributions, and leakage risks
2. `02_modeling.ipynb` — baseline experiments and error analysis

Before committing a notebook, restart the kernel and run all cells in order.
Move stable loading, preprocessing, training, and evaluation logic into `src/`.
