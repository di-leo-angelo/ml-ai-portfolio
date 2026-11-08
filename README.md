# ML & AI Portfolio

A collection of reproducible machine learning, natural language processing,
and applied AI projects. Each project is designed to show the full workflow:
problem framing, data preparation, modeling, evaluation, and conclusions.

## Projects

| Project | Area | Highlights | Status |
| --- | --- | --- | --- |
| [Telco Customer Churn](projects/telco-churn/) | Tabular classification | Nested OOF thresholding, SHAP explanations, business scenario analysis | Implemented |
| [CIFAR-10 CNN](projects/cifar10-cnn/) | Image classification | 89.0% test accuracy, LR scheduling + early stopping, CPU-only training | Implemented |
| [BERT Sentiment (SST-2)](projects/bert-sentiment/) | NLP / fine-tuning | BERT-base 92.4% vs DistilBERT 90.1% dev accuracy, published to the Hugging Face Hub | Implemented |
| [Qwen3 QLoRA Tutor](projects/qwen-qlora/) | LLM fine-tuning (QLoRA) | Qwen3.8-27B 4-bit NF4 + LoRA on a rented 4090/5090, tone-focused synthetic Q&A dataset; LoRA adapter published to the HF Hub | Complete (trained, evaluated, results published) |
| [LLM Eval Harness](projects/llm-eval-harness/) | LLM evaluation framework | Open-source, dep-light eval harness: multi-backend generation, deterministic graders + LLM-as-judge, resume-safe stores, statistics, CI gate | In progress (M2/6: graders + judge port-validated against the QLoRA project) |

## Repository structure

```text
.
├── .github/workflows/      # Continuous integration
├── projects/               # Self-contained portfolio projects
│   ├── bert-sentiment/     # Data, source, reports, and project-local tests
│   ├── cifar10-cnn/        # Data, source, reports, and project-local tests
│   ├── llm-eval-harness/   # Eval framework: config, backends, graders, judge
│   ├── qwen-qlora/         # Data, source, reports, and project-local tests
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

## Run the BERT Sentiment project

Download the dataset, then fine-tune a Transformer on SST-2 (CPU-only; the
`--quick` smoke test takes ~15 min, the full BERT-base run takes ~11 h on an
i5 with no GPU):

```bash
uv run python projects/bert-sentiment/scripts/download_data.py
cd projects/bert-sentiment/src
uv run python -m sentiment.train --quick
uv run python -m sentiment.train
uv run python -m sentiment.evaluate
```

To publish a model to the Hugging Face Hub (dry run by default, uses the
`HF_TOKEN` from the git-ignored `.env.local`):

```bash
uv run python projects/bert-sentiment/scripts/publish.py --model bert-base
```

See the [project README](projects/bert-sentiment/README.md) for the
methodology, results, and project-specific details.

## Run the Qwen3 QLoRA project

Fine-tune Qwen3.8-27B with QLoRA on a fully synthetic school Q&A dataset
(proof of concept: the fine-tune targets tutor tone/style, and the dataset is
seed-anchored per subject - see the project README). Data generation runs
anywhere (needs an OpenAI-compatible API key in `projects/qwen-qlora/.env`);
training needs a 24+ GB GPU (rented):

```bash
uv run python projects/qwen-qlora/scripts/generate_data.py
uv run python projects/qwen-qlora/scripts/prepare_data.py
# On the GPU host (inside tmux):
pip install -r projects/qwen-qlora/requirements-gpu.txt
python projects/qwen-qlora/scripts/download_model.py
cd projects/qwen-qlora
python src/qwen_qlora/train.py --model-name models
```

See the [project README](projects/qwen-qlora/README.md) for the rental
workflow, hardware sizing, and the data-generation recipe.

## Run the llm-eval-harness project

Open-source, dependency-light eval framework - the backbone of the
portfolio chain, and the framework the QLoRA project's results above were
validated with: task specs (JSONL), multi-backend inference, deterministic
graders, and an LLM-as-judge with structured rubric scoring. Development
is milestone-based (see the project's PLAN.md). The dogfood config reuses
the QLoRA project's recorded eval outputs, so the core validation runs
CPU-only, with no GPU and no network:

```bash
# CPU-only: import the recorded QLoRA outputs and grade them
uv run python -m llm_eval_harness import-generations \
    --input projects/qwen-qlora/reports/auto_eval/raw_outputs.jsonl \
    --output projects/llm-eval-harness/reports/qwen-qlora-dogfood/generations.jsonl
uv run python -m llm_eval_harness grade \
    --config projects/llm-eval-harness/configs/dogfood_p1.toml
```

Live judging additionally needs `RUNPOD_API_KEY` in
`projects/qwen-qlora/.env`. See the [project README](projects/llm-eval-harness/README.md)
for the design notes, the QLoRA correspondence table, and the full
command reference.

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
