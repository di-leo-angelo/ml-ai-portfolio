# ML & AI Portfolio

A collection of reproducible machine learning, natural language processing,
and applied AI projects. Each project is designed to show the full workflow:
problem framing, data preparation, modeling, evaluation, and conclusions.

## Projects

| Project | Area | Highlights | Status |
| --- | --- | --- | --- |
| [Telco Customer Churn](projects/telco-churn/) | Tabular classification | Nested OOF thresholding, SHAP explanations, business scenario analysis | Implemented |
| [CIFAR-10 CNN](projects/cifar10-cnn/) | Image classification | 89.0% test accuracy, LR scheduling + early stopping, CPU-only training | Implemented |

## Repository structure

```text
.
├── .github/workflows/      # Continuous integration
├── projects/               # Self-contained portfolio projects
│   ├── cifar10-cnn/        # Data, source, reports, and project-local tests
│   └── telco-churn/        # Data, source, reports, and project-local tests
├── src/ml_ai_portfolio/    # Shared package code
├── tests/                  # Shared package tests
└── pyproject.toml          # Dependencies and tool configuration
```

Each finished project should include its own README with the question being
answered, dataset source, methodology, results, limitations, and instructions
for reproducing the work.

## Getting started

This repository requires Python 3.12 or newer. The recommended package manager
is [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/di-leo-angelo/ml-ai-portfolio.git
cd ml-ai-portfolio
uv sync --extra dev
```

Using `pip` instead:

```bash
python -m venv .venv

# Windows
.venv\Scripts\python -m pip install -e ".[dev]"

# macOS or Linux
.venv/bin/python -m pip install -e ".[dev]"
```

## Run the Telco Churn project

Download the dataset, then train and evaluate the models:

```bash
uv run python projects/telco-churn/scripts/download_data.py
uv run python projects/telco-churn/src/train.py
```

See the [project README](projects/telco-churn/README.md) for the methodology,
results, generated artifacts, limitations, and project-specific test command.

## Run the CIFAR-10 CNN project

Download the dataset, then train and evaluate (CPU-only; a full training run
takes several hours):

```bash
uv run python projects/cifar10-cnn/scripts/download_data.py
cd projects/cifar10-cnn/src
uv run python -m cifar10.train
uv run python -m cifar10.evaluate
```

See the [project README](projects/cifar10-cnn/README.md) for the methodology,
results, and project-specific details.

## Tests and quality checks

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

Tests are organized with the code they cover: shared package tests live in the
root `tests/` directory, while each portfolio project owns its tests under its
project directory. GitHub Actions runs all three checks for pushes and pull
requests.

## Contributing

Suggestions and fixes are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for
the expected project structure and development workflow.

## License

Released under the [MIT License](LICENSE).
