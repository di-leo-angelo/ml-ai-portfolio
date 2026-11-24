# llm-eval-harness (P0)

Open-source, dependency-light eval framework - the backbone of the
portfolio chain: task specs (JSONL), multi-backend inference (local HF /
OpenAI-compatible API / RunPod), LLM-as-judge with structured rubric
scoring, calibration against human labels, bootstrap/McNemar statistics,
and a CI regression gate.

**Status: M2 accepted (2026-11-24) - live acceptance complete, one
documented exception.**
M0-M2 shipped: run config + task specs; `local` HF / `api` / `runpod`
backends with resume-safe generation; the deterministic graders and the
LLM-as-judge ported from P1's frozen eval scripts (config-extensible
rubric, explicit parse coverage, position-swapped pairwise); P1 output
import; and Pearson/Spearman agreement stats. 104 tests green, and the
graders reproduce P1's `graded.jsonl` field-for-field on all 230 LoRA
rows (0/7-field mismatches). Live acceptance on the RTX 3090: local
smoke PASS (adapter on/off, peaks 18.8/19.3 GB, resume-safety
demonstrated), live judging at **100 % parse coverage on all 255**
(recovering the 10 rows P1 silently dropped), pairwise smoke 25/25 with
disagreements reported (21 position-bias / 4 tie). The judge
correlation bar (pearson >= 0.8 vs P1's legacy) is formally unmet and
documented with evidence as endpoint drift + ceiling saturation - P1's
own byte-exact prompt re-scored today tops out at ~0.67 against P1's
own legacy file. Milestone history, the acceptance evidence, and the
M4 labeling tooling (`tools/`) are in [PLAN.md](PLAN.md) section 10.

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
| `src/llm_eval_harness/backends/` | `api`/`runpod` (OpenAI-compatible, stdlib urllib) + `local` HF (4-bit + LoRA) generation backends; `imported` is a provenance-only kind (no generation) |
| `src/llm_eval_harness/graders/` | deterministic graders (P1 `auto_grade.py` port: exact-match, format) + LLM judge (prompt, parse ladder, resume-safe store, pairwise with swap) |
| `src/llm_eval_harness/stats/` | agreement statistics (Pearson/Spearman with midranks; M3 adds bootstrap/McNemar) |
| `src/llm_eval_harness/runner.py` | run orchestration: manifest + resume-safe generation + grade/judge/pairwise |
| `src/llm_eval_harness/cli.py` | `python -m llm_eval_harness generate / import-tasks / import-generations / grade / judge / pairwise` |
| `configs/` | run configs (dogfood example + free-endpoint smoke + `dogfood_p1.toml`, the M2 acceptance config) + example tasks file |
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

M2 (grading + judging, CPU-first):

```powershell
# import P1's recorded outputs (no GPU, no network) into harness generations
uv run python -m llm_eval_harness import-generations `
    --input projects/qwen-qlora/reports/auto_eval/raw_outputs.jsonl `
    --output projects/llm-eval-harness/reports/qwen-qlora-dogfood/generations.jsonl

# deterministic graders over the generations (CPU-only, no backends)
uv run python -m llm_eval_harness grade `
    --config projects/llm-eval-harness/configs/dogfood_p1.toml

# LLM-as-judge (resume-safe; --against prints Pearson/Spearman vs a
# P1-style legacy judge_scores.jsonl for the port validation)
uv run python -m llm_eval_harness judge `
    --config projects/llm-eval-harness/configs/dogfood_p1.toml `
    --env-file projects/qwen-qlora/.env `
    --against projects/qwen-qlora/reports/auto_eval/judge_scores.jsonl

# position-swapped pairwise smoke (25 paired base-vs-lora samples)
uv run python -m llm_eval_harness pairwise `
    --config projects/llm-eval-harness/configs/dogfood_p1.toml `
    --condition-a lora --condition-b base `
    --env-file projects/qwen-qlora/.env
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

P1's 230-row `test.jsonl` (and the `data/split.json` manifest) is
committed as reference data, so this works on a fresh clone:

```python
from llm_eval_harness.legacy import import_jsonl

tasks = import_jsonl("projects/qwen-qlora/data/test.jsonl")
# -> 230 Task objects with stable p1_XXXX ids and a source marker
```

The train/dev splits (1,270 rows) remain gitignored - regenerate them with
the P1 data pipeline (see its README). The committed 10-row
`seed_prompts.jsonl` is already importable, and `tests/test_legacy.py`
exercises the importer against it.

## Correspondence with P1

P1's frozen eval scripts (see its README) map onto this harness as
follows. Task ids are aligned by line order of `test.jsonl`: P1
`sample_id N` (0-based) is harness task `p1_{N+1:04d}`, and P1's
`"N:condition"` keys map through `legacy.p1_key_to_task_id`.

| P1 (`qwen-qlora`) | P0 (this harness) | Notes |
| --- | --- | --- |
| `auto_grade.py` `exact_match` | `graders/exact_match.py` (`exact_match` grader) | verbatim port: `MATH_SUBJECTS`, fraction-preferred tail extraction (last 500 chars), exact fraction arithmetic; non-math rows carry `exact_match: null` |
| `auto_grade.py` `format_metrics` | `graders/format.py` (`format` grader) | verbatim port: whitespace word count, `WORD_CAP = 350`, step-scaffolding regex `(?im)^\s*(step\s*\d+|\d+[.):])` |
| `graded.jsonl` | `grades.jsonl` (from `grade`) | recomputed whole each run, like P1 |
| `judge_run.py` batched judge (N=5, max_tokens 1024) | `graders/judge.py` `judge_pending` (from `judge`) | same endpoint/shape; `batch_size`/`max_tokens` from `[judge]` |
| judge dimensions `correctness` / `format` / `completeness` | `[judge.dimensions]` table in `configs/dogfood_p1.toml` | P1's dimensions were plain questions without 1-5 anchors; the config holds the P0-contract rendering (one anchor line per dimension, quoted verbatim in the prompt); the rubric version is `custom` so it never mixes with the built-in five |
| `judge_scores.jsonl` rows `{key, correctness, format, completeness}` | `judge_scores.jsonl` rows `{key: "p1_XXXX:cond", task_id, condition, model, judge_model, rubric_version, <dims>, parse, ts}` | P0 rows carry the parse method (`json`/`fence`/`span`/`regex`) so coverage is explicit; `judge --against <P1 file>` aligns the two key schemes and prints Pearson/Spearman per dimension |
| pairwise (planned in P1, never run) | `pairwise` command + `graders/judge.py` `pairwise_compare` | position-swapped; a preference is recorded only when both directions agree |
| judge retry policy (3 attempts, 5s x attempt, 408/429/5xx retry, other 4xx fail fast) | `backends/openai_compat.py` | shared by the judge via the normal backend interface; a 200 + error body fails the job instead of becoming a record |
| judge self-bias risk (Qwen3-32B judging Qwen3.8-27B) | config guard + data-driven guard | identical model = load error; same family = loud warning (the dogfood config prints it) |

M2 acceptance (live): judge the imported 230 lora + 25 base generations
with parse coverage >= 97% (P1 baseline 96.1%); per-dimension
correlation with P1's legacy `judge_scores.jsonl` >= 0.8; 25-pair
pairwise smoke with swap. Note P1's legacy file holds 245 rows (220
lora + 25 base): 10 lora rows were silently dropped by P1's parser -
the 10/255 incident that motivates the explicit parse coverage here -
so the correlation overlap is those 245 rows.
