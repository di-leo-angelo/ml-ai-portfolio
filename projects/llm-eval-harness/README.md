# llm-eval-harness (P0)

Open-source, dependency-light eval framework - the backbone of the
portfolio chain: task specs (JSONL), multi-backend inference (local HF /
OpenAI-compatible API / RunPod), LLM-as-judge with structured rubric
scoring, calibration against human labels, bootstrap/McNemar statistics,
and a CI regression gate.

**Status: M1 - backends + generation** - run config, task/rubric data
model, P1 legacy import, and the generation layer are implemented and
tested: `api`/`runpod` backends (stdlib urllib, P1 retry constants,
response-shape normalization, optional concurrency), a `local` HF
backend (4-bit + LoRA, GPU pod), and resume-safe `generations.jsonl`
output (kill mid-run, re-run generates only the remainder). A 10-task
live smoke against the free RunPod endpoint - including a kill +
resume - is recorded under `reports/` (gitignored). The local-backend
pod smoke (adapter on/off, peak VRAM) awaits a GPU session. Judging/
grading (M2), statistics (M3), calibration (M4), and the CI gate (M5)
land in the order set out in [PLAN.md](PLAN.md).

## Design notes

- **TOML for config, JSON/JSONL for data.** Run configs are read with
  stdlib `tomllib` (read-only); the harness only ever writes JSON/JSONL.
- **Backends are discriminated by `kind`**: `local` (HF transformers,
  optionally + LoRA adapter), `api` (any OpenAI-compatible endpoint,
  including RunPod public endpoints), `runpod` (serverless job queue).
- **The judge is a backend config.** The P1 judge - Qwen3-32B-AWQ via
  the RunPod public OpenAI-compatible endpoint - is an `api` backend,
  so no new auth or deployment is needed.
- **The rubric is a contract.** Five dimensions (correctness, tone,
  format, safety, helpfulness) with 1..5 anchors each; anchors are
  quoted verbatim in the judge prompt, so a rubric change is a
  `RUBRIC_VERSION` bump and gate comparisons require the same version.
- **No new dependencies.** stdlib + numpy (statistics, M3); model code
  reuses `transformers`/`peft` already used by P1.

## Layout

| Path | Contents |
| --- | --- |
| `src/llm_eval_harness/config.py` | TOML run config: conditions, judge, gate + validation |
| `src/llm_eval_harness/tasks.py` | Task spec: JSONL load + validation |
| `src/llm_eval_harness/rubric.py` | five-dimension rubric with 1..5 anchors |
| `src/llm_eval_harness/legacy.py` | P1 (qwen-qlora) split/seed to task import |
| `src/llm_eval_harness/backends/` | `api`/`runpod` (OpenAI-compatible, stdlib urllib) + `local` HF (4-bit + LoRA) generation backends |
| `src/llm_eval_harness/runner.py` | run orchestration: manifest + resume-safe generation |
| `src/llm_eval_harness/cli.py` | `python -m llm_eval_harness generate / import-tasks` |
| `configs/` | run configs (dogfood example + free-endpoint smoke) + example tasks file |
| `tests/` | CPU-only unit tests (no models, no real network; loopback fake HTTP only) |
| `data/legacy/` | imported P1 task files (generated, gitignored) |
| `reports/` | run outputs (gitignored; final reports force-added) |

## Quickstart (M1)

```powershell
# from the repo root: CPU-only test suite, no models or API keys
uv run pytest projects/llm-eval-harness/tests -q

# convert a P1 split/seed file to the harness task schema
uv run python -m llm_eval_harness import-tasks `
    --input projects/qwen-qlora/data/test.jsonl `
    --output projects/llm-eval-harness/data/legacy/p1_test.jsonl

# generate outputs for every condition of a run config (resume-safe:
# re-running generates only the missing task/condition/seed keys)
uv run python -m llm_eval_harness generate `
    --config projects/llm-eval-harness/configs/smoke_runpod.toml `
    --env-file projects/qwen-qlora/.env --limit 10
```

The smoke config (`configs/smoke_runpod.toml`) targets the free public
RunPod endpoint - no deployment needed; `RUNPOD_API_KEY` is loaded from
`projects/qwen-qlora/.env` via `--env-file` and sent when present.
Run outputs (`manifest.json` + `generations.jsonl`) land in the config's
`output_dir` (the project's `reports/`, gitignored - final reports are
force-added, P1's convention).

The `local` backend is the only path that needs the GPU pod (setup per
[PLAN.md](PLAN.md) section 8); the committed dogfood config is
`configs/example_run.toml` (P1's test split, adapter on/off, P1 eval
decoding settings).

## Importing the P1 test split

P1's 230-row `test.jsonl` is gitignored - it was generated locally in
`projects/qwen-qlora/data/` (manifest + hashes in `data/split.json`) and
pushed to the GPU pod via `scp` for the training/eval runs. Once it is
present locally, import it as harness tasks:

```python
from llm_eval_harness.legacy import import_jsonl

tasks = import_jsonl("projects/qwen-qlora/data/test.jsonl")
# -> 230 Task objects with stable p1_XXXX ids and a source marker
```

The committed 10-row `seed_prompts.jsonl` is already importable, and
`tests/test_legacy.py` exercises the importer against it.
