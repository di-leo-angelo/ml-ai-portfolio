# llm-eval-harness (P0)

Open-source, dependency-light eval framework - the backbone of the
portfolio chain: task specs (JSONL), multi-backend inference (local HF /
OpenAI-compatible API / RunPod), LLM-as-judge with structured rubric
scoring, calibration against human labels, bootstrap/McNemar statistics,
and a CI regression gate.

**Status: M0 scaffold** - run config + task/rubric data model + P1 legacy
import are implemented and tested (CPU-only, no models, no API keys).
Backends (M1), judging/grading (M2), statistics (M3), calibration (M4),
and the CI gate (M5) land in the order set out in [PLAN.md](PLAN.md).

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
| `configs/` | example run config + example tasks file |
| `tests/` | CPU-only unit tests (no models, no network) |
| `data/legacy/` | imported P1 task files (generated, gitignored) |
| `reports/` | run outputs (gitignored; final reports force-added) |

## Quickstart (M0)

```powershell
# from the repo root: CPU-only test suite, no models or API keys
uv run pytest projects/llm-eval-harness/tests -q
```

## Importing the P1 test split

P1's 230-row `test.jsonl` is gitignored (generated on a GPU pod). Once
it is present locally (or on a pod), import it as harness tasks:

```python
from llm_eval_harness.legacy import import_jsonl

tasks = import_jsonl("projects/qwen-qlora/data/test.jsonl")
# -> 230 Task objects with stable p1_XXXX ids and a source marker
```

The committed 10-row `seed_prompts.jsonl` is already importable, and
`tests/test_legacy.py` exercises the importer against it.
