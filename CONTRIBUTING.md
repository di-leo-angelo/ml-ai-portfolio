# Contributing

## Development setup

```bash
uv sync --extra dev
```

Run the checks before opening a pull request:

```bash
ruff check .
ruff format --check .
pytest
```

Ruff can apply safe lint fixes and formatting:

```bash
ruff check --fix .
ruff format .
```

## Adding a project

1. Create `projects/<project-name>/`.
2. Add a project README following `projects/README.md`.
3. Keep reusable logic in importable Python modules rather than notebook cells.
4. Add tests for data transformations and modeling logic.
5. Add the project to the table in the root README.

Use public or redistributable datasets and cite their source and license.
Never commit secrets, personal data, large datasets, or generated model files.
