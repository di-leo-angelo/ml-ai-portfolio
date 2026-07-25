# ML & AI Portfolio

A collection of reproducible machine learning, natural language processing,
and applied AI projects. Each project is designed to show the full workflow:
problem framing, data preparation, modeling, evaluation, and conclusions.

## Projects

Projects will be listed here as they are completed.

| Project | Area | Highlights | Status |
| --- | --- | --- | --- |
| [Telco Customer Churn](projects/telco-churn/) | Tabular classification | Leakage-safe pipelines, model comparison, threshold analysis | In progress |

## Repository structure

```text
.
├── projects/             # Self-contained portfolio projects
├── notebooks/            # Exploratory notebooks
├── src/ml_ai_portfolio/  # Reusable Python code
├── tests/                # Automated tests
├── data/                 # Local datasets (contents are not committed)
├── models/               # Generated models (contents are not committed)
└── pyproject.toml        # Dependencies and tool configuration
```

Each finished project should include its own README with the question being
answered, dataset source, methodology, results, limitations, and instructions
for reproducing the work.

## Getting started

This repository requires Python 3.11 or newer. The recommended package manager
is [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/di-leo-angelo/ml-ai-portfolio.git
cd ml-ai-portfolio
uv sync --extra dev
```

Activate the environment:

```bash
# Windows
.venv\Scripts\activate

# macOS or Linux
source .venv/bin/activate
```

To open the notebooks:

```bash
jupyter lab
```

Using `pip` instead:

```bash
python -m venv .venv
python -m pip install -e ".[dev]"
```

## Quality checks

```bash
ruff check .
ruff format --check .
pytest
```

## Contributing

Suggestions and fixes are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for
the expected project structure and development workflow.

## License

Released under the [MIT License](LICENSE).
