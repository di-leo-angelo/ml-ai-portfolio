# P0 — llm-eval-harness: Implementation Plan

**Status:** draft v1 (2026-11-02), pending review
**Budget:** 1–2 weeks @ 10–15 h/wk (20–30 h total)
**Depends on:** P1 (qwen-qlora) — its 230-sample test set, RunPod judge-endpoint
pattern, and exact-match grading logic are the seed material
**Feeds:** P2 (pairwise judge + abstention probe), P3 (per-stage eval + red-team
suite), P4 (RAG / tool-RAG / cache / guardrail metrics)

## 1. What we're building

A config-driven, reproducible LLM evaluation harness:

1. **Task specs** (JSONL) — prompts + optional references + grader selection.
2. **Multi-backend inference** — local HF (optional LoRA adapter / 4-bit), any
   OpenAI-compatible API, RunPod serverless endpoints. One config selects N
   named *conditions* (e.g. `base` vs `lora`) evaluated on the same tasks.
3. **Structured LLM judge** — pointwise rubric scoring
   (correctness / tone / format / safety / helpfulness, 1–5, anchored anchors)
   and a **pairwise** mode (win/lose/tie with A/B position swap), plus
   deterministic graders (exact match, format heuristics, abstention probe).
4. **Calibration mode** — judge vs. a human-labeled subset: Spearman/Pearson,
   Cohen's κ, quadratic weighted κ, judge bias with bootstrap CI, 5×5
   confusion matrix per dimension.
5. **Statistics layer** — bootstrap 95% CIs (plain, stratified, paired),
   exact McNemar for paired binary outcomes, seed-variance runs.
6. **CI gate** — committed metric baseline + threshold config → exit 0/1/2,
   GitHub Actions job.

**Non-goals (P0):** public benchmark suites (a benchmark can be encoded as a
task file later, none built in), leaderboards/UIs, multi-tenant serving,
training. And: do **not** rewrite P1's frozen pipeline — P0 dogfoods the same
data and generalizes the logic. Acceptance bar: P1's reported numbers must be
reproducible from P0 outputs alone.

## 2. Design decisions

| Decision | Choice | Rationale |
| --- | --- | --- |
| Location | `projects/llm-eval-harness/` in this monorepo, package `llm_eval` (src layout, `pythonpath`/`testpaths` entries in root `pyproject.toml`) | Same convention as P1–P3; shared venv; P2–P5 import it directly. Extraction to a standalone open-source repo is a later rename, not a rewrite. |
| Runtime deps | **Zero new ones.** stdlib (`json`, `math`, `tomllib`, `urllib`, `dataclasses`) + `numpy` (bootstrap) + `transformers`/`peft` (local backend only) | P1's pod workflow (bare `pip install` of torch + requirements file) stays untouched; CPU-only usage needs nothing but numpy. |
| No scipy | McNemar = exact binomial via `math.comb`; Spearman = Pearson on midranks; κ by hand; bootstrap = numpy | The three hand-rolled stats functions force the unit tests that are the whole point (statistics layer as an artifact, not a claim). sklearn exists in the venv but is deliberately not used here. |
| HTTP | stdlib `urllib` for the OpenAI-compatible backend (port of P1's `judge_run.py`) | P1 chose it deliberately; fewer moving parts on rented pods. The `openai` SDK (already a root dep) is available if needed later. |
| Config format | TOML (stdlib `tomllib`) for run configs; JSONL for all data/outputs | No YAML dep; TOML is readable and validated in tests. |
| Resume safety | Append-only JSONL outputs, keyed skip on re-run (P1 pattern) | Every expensive stage (generation, judging) survives pod death. |
| Judge ≠ testee | Config guard: error if a condition's model equals the judge model, loud warning on same-family | Self-judge bias is real; the guard makes it a documented choice. |
| Observability | Per-record tokens, wall time, and cost estimate; totals in the report | "This eval cost $1.83" is a first-class report line (and a blog data point). |
| Outputs | Every metric in machine-readable `summary.json` **and** human-readable `report.md`; `summary.json` is the stable contract the CI gate reads | Report format may evolve; `summary.json` keys may not. |

## 3. Data model

### 3.1 Task spec (JSONL)

```json
{
  "task_id": "tutor-test/algebra/0042",
  "messages": [{"role": "system", "content": "…"}, {"role": "user", "content": "…"}],
  "reference": "…",
  "graders": ["exact_match", "format", "abstention", "rubric"],
  "out_of_scope": false,
  "metadata": {"subject": "algebra", "difficulty": "intermediate"}
}
```

- `reference` optional (required for `exact_match`; optional for the judge —
  passed as a rubric input when present).
- `out_of_scope: true` → the abstention grader scores *honest* refusal
  ("I don't know" counts as pass; confident wrong answer counts as fail).
  This is the probe P2 needs.
- **Legacy import:** `python -m llm_eval import-tasks` converts P1's
  `test.jsonl` (`system`/`user`/`assistant`/`subject`/`difficulty`) to this
  schema. Task IDs are deterministic (`tutor-test/{subject}/{index}`), so
  re-importing is idempotent.

### 3.2 Run config (TOML)

```toml
[run]
id = "tutor_20261102"
tasks = "data/tasks/tutor_test.jsonl"

[generation]
max_new_tokens = 512
temperature = 0.0            # greedy by default; --seeds overrides w/ sampling
concurrency = 1              # 1 = sequential (P1 default); 2-8 for APIs

[conditions.base]
backend = "local"
model = "Qwen/Qwen3.8-27B"
quantization = "nf4"

[conditions.lora]
backend = "local"
model = "Qwen/Qwen3.8-27B"
quantization = "nf4"
adapter = "../qwen-qlora/artifacts"

# [conditions.api]
# backend = "openai"
# base_url = "https://api.runpod.ai/v2/qwen3-32b-awq/openai/v1"
# model = "qwen3-32b-awq"
# api_key_env = "RUNPOD_API_KEY"
# price_per_mtok = {in = 0.0, out = 0.0}

[judge]
backend = "openai"
base_url = "https://api.runpod.ai/v2/qwen3-32b-awq/openai/v1"
model = "qwen3-32b-awq"
api_key_env = "RUNPOD_API_KEY"
temperature = 0.0
max_tokens = 1024
batch = 5                    # samples per judge call (P1 convention)

[judge.dimensions]
correctness = "1 = wrong final answer or broken reasoning … 5 = fully correct, valid method"
tone        = "1 = off-voice (lecture-y, dismissive) … 5 = step-by-step tutor voice, concise"
format      = "1 = rambling/unclear … 5 = clear steps, final answer stated"
safety      = "1 = unsafe/harmful for a student audience … 5 = safe"
helpfulness = "1 = does not answer … 5 = fully addresses the question"
```

Anchors are one-line low/high descriptions (not full 1–5 tables) — cheaper
prompt, less judge drift. The judge is itself a `backend` config, so it can be
local HF, any API, or the RunPod endpoint.

### 3.3 Run outputs

```
reports/<run_id>/
├── manifest.json       # config echo + task-file SHA-256 + git commit + ts
├── generations.jsonl   # resume-safe, one line per (task_id, condition[, seed])
├── grades.jsonl        # deterministic grader results
├── judge_scores.jsonl  # resume-safe
├── summary.json        # machine-readable metrics (STABLE — gate reads this)
├── report.md           # human-readable, CIs on every number
└── comparisons.md      # paired condition side-by-side (ported from P1)
```

Generation record:

```json
{"task_id": "tutor-test/algebra/0042", "condition": "lora", "output": "…",
 "seed": 0, "tokens": {"in": 412, "out": 318}, "wall_s": 12.4,
 "cost_usd": 0.0012, "backend": "local", "model": "Qwen/Qwen3.8-27B"}
```

`summary.json` key paths (stable contract, v1):

- `judge.{dim}.mean = {estimate, ci_low, ci_high, n, coverage}`
- `exact_match.pass_rate = {…}` (math subjects only, as in P1)
- `format.within_cap_rate`, `format.word_count = {median, p95}`
- `abstention.honest_rate` (out-of-scope tasks only)
- `pairwise.win_rate = {…}` (condition A vs B, position-swap-agreed only)
- `meta = {run_id, task_sha256, commit, ts, conditions, judge}`

### 3.4 Human labels (calibration)

```json
{"task_id": "tutor-test/algebra/0042", "condition": "lora",
 "scores": {"correctness": 4, "tone": 5, "format": 5, "safety": 5, "helpfulness": 4},
 "pass": true, "notes": "", "annotator": "self"}
```

**Labeling protocol (committed to the report):** n = 100–150 answers drawn
stratified by subject × difficulty from the P1 test-set outputs, **plus** a
deliberate hard slice (~20%): long answers, abstentions, visibly wrong
answers, base-condition outputs. Budget ≈ 1–2 min/answer → 2–3 h. This file
is the format P2's "human spot-check" step reuses.

## 4. Architecture

```
projects/llm-eval-harness/
├── PLAN.md / README.md
├── configs/          # run configs, gate config, baselines/ (committed)
├── data/             # tasks/ (imported, committed), human_labels/ (ignored)
├── reports/          # (ignored, like P1; final .md/.json force-added)
├── src/llm_eval/
│   ├── __init__.py
│   ├── config.py         # TOML load + dataclasses; .env parsing (P1 env.py pattern)
│   ├── tasks.py          # task-spec load/validate + legacy import
│   ├── backends/
│   │   ├── __init__.py   # Backend protocol, registry, "kind:model" parsing
│   │   ├── base.py       # GenerationJob/Record, resume-safe append, cost accounting
│   │   ├── local.py      # HF transformers + 4-bit + optional PeftModel adapter
│   │   └── openai_compat.py  # any /chat/completions; retries+backoff; response-shape
│   │                     #   normalization (OpenAI + RunPod serverless, ported)
│   ├── graders/
│   │   ├── __init__.py   # Grader protocol + registry (name → callable)
│   │   ├── exact_match.py    # port of auto_grade.extract_final_answer/answers_match
│   │   ├── format.py         # word count, step scaffolding, word cap
│   │   ├── abstention.py     # honest-refusal probe (P2)
│   │   └── judge.py          # pointwise + pairwise LLM judge; parse ladder; guards
│   ├── stats/
│   │   ├── bootstrap.py    # plain / stratified / paired percentile CIs (numpy)
│   │   ├── mcnemar.py      # exact two-sided p (math.comb) + exact CI on (b−c)/(b+c)
│   │   ├── agreement.py    # Pearson, Spearman (midrank), Cohen κ, quadratic-wtd κ, bias
│   │   └── seed_runs.py    # per-seed metrics → mean ± SD, min/max
│   ├── gate.py             # baseline + thresholds → exit 0/1/2 + gate_report.md
│   ├── report.py           # summary.json + report.md + comparisons.md rendering
│   └── cli.py              # argparse: import-tasks generate grade judge calibrate report gate
└── tests/                  # see §9
```

**Backend protocol:**

```python
class Backend(Protocol):
    name: str
    model: str

    def generate(self, jobs: Sequence[GenerationJob]) -> Iterator[GenerationRecord]: ...
```

- The runner loads existing keys from `generations.jsonl` and skips them (P1
  resume pattern), flushing after each write.
- `local`: lazy-load once; supports `adapter` (PeftModel on the same base) and
  a no-adapter condition from the *same* process (P1's base-vs-lora trick),
  peak-VRAM logging (P1 convention).
- `openai_compat`: sequential by default; `--concurrency N` → ThreadPool.
  Retries: 3 attempts, 5×attempt backoff, 300 s timeout (P1 constants).
  Response normalizer handles `choices[].message.content`, RunPod
  `output.content`, and bare `content` (port of P1 `extract_content`).

**Judge flow (pointwise):** batch of `batch` samples → one prompt with anchored
rubric → strict JSON (`response_format={"type":"json_object"}` when the
endpoint supports it; probe once per run and record the result in the
manifest). **Parse ladder** (fixes P1's 10/255 silent parse failures):
(1) `json.loads` on the content → (2) strip code fences / pre-JSON prose →
(3) per-dimension regex fallback (P1 pattern). Every record gets
`"parse": "json"|"fence"|"regex"`; unparseable samples are counted in
`coverage` — never silently dropped.

**Judge flow (pairwise):** (prompt, A, B) → `{"winner": "a"|"b"|"tie",
"confidence": 1-5, "rationale": "…"}`. Run twice, A/B swapped. A preference
counts only when both orders agree (position-bias control); ties and
disagreements are reported separately.

## 5. Calibration mode

`python -m llm_eval calibrate --run <run_id> --labels data/human_labels/tutor_labels.jsonl`

Per dimension, on the joined judge×human rows (matched by task_id +
condition):

| Metric | Purpose |
| --- | --- |
| Spearman ρ + Pearson r | monotonic vs linear agreement on 1–5 scores |
| Cohen's κ (strict) + κ (±1 tolerance) | agreement beyond chance, ordinal-appropriate |
| Quadratic weighted κ | penalizes large disagreements, not just off-by-ones |
| Judge bias = mean(judge − human), with bootstrap CI | systematic leniency (P1's known 32B saturation shows up here) |
| 5×5 confusion matrix per dimension | where exactly the judge drifts |

**Decision rule (documented in the report):** a dimension is
*gate-worthy* (i.e. allowed to drive the CI gate / P2 go-no-go) iff
Spearman ≥ 0.6 **and** |bias| ≤ 0.5; otherwise it is *report-only*. This
turns calibration from a footnote into an actual quality gate on the judge —
the core thesis of the project.

Outputs: `reports/calibration/<name>/{agreement.json, report.md}`. Honest
outcome is fine: if `tone` calibrates and `correctness` doesn't (judge
saturates), the report says exactly that and the gate uses the metrics that
did calibrate.

## 6. Statistics layer

All functions pure (no I/O), numpy-in/stdout-out, each unit-tested against
hand-computed values:

- **Bootstrap** (`stats/bootstrap.py`): percentile 95% CI, B = 10,000,
  fixed default seed (1337, overridable). Three variants:
  - *plain* — resample task-level metric values (for overall means/rates);
  - *stratified* — resample within strata (subject), preserving stratum
    sizes (for per-subject tables);
  - *paired* — resample task indices, compute per-task (A − B), CI on the
    mean difference (for condition comparisons on continuous metrics).
- **McNemar** (`stats/mcnemar.py`): paired binary outcomes (e.g.
  exact-match pass/fail on the same 230 tasks, base vs lora). Discordant
  cells b (A pass / B fail) and c (A fail / B pass); exact two-sided
  p = 2·Σᵢ≤min(b,c) C(b+c, i)·0.5^(b+c) via `math.comb`; exact (Clopper–Pearson)
  CI on (b−c)/(b+c). b+c = 0 → reported as "no discordant pairs", p = 1.
- **Seed-variance runs**: `generate --seeds 0,1,2` (sampling on, fixed
  per-run seeds) → per-seed metrics; report mean ± SD per metric across
  seeds plus min/max. With greedy decoding (default) seed variance is ~0 —
  the harness warns explicitly instead of pretending.
- **Agreement** (`stats/agreement.py`): Pearson; Spearman via midrank +
  Pearson (handles ties); Cohen's κ for 2×2 (agree/disagree); quadratic
  weighted κ for the 5×5; all by hand, no scipy/sklearn.

Every headline number in `report.md` carries its CI and n; coverage
(parse-success rate) is printed next to every judge metric.

## 7. CI gate

`python -m llm_eval gate --run <run_id> --baseline baselines/tutor.json --config configs/gate.toml`

- **Baseline:** committed `baselines/<name>.json` = a `summary.json` plus
  provenance (run_id, task-file SHA-256, commit). A mismatched task hash
  prints a loud warning (different question set = comparison suspect).
- **Thresholds** (per metric, any subset):

```toml
[metrics."judge.correctness.mean"]
min_delta = 0.0              # fail if new − baseline < 0

[metrics."judge.tone.mean"]
min_delta = -0.1             # tolerate a 0.1-point drop

[metrics."exact_match.pass_rate"]
min_delta = -0.02
ci_low_min = 0.30            # also fail if the CI lower bound drops below

[metrics."format.word_count.p95"]
max_delta = 50               # length-gaming guardrail

[guards]
judge_coverage_min = 0.95    # fail if parse coverage falls below 95%
```

- **Exit codes:** 0 = pass, 1 = regression, 2 = error (missing baseline,
  missing metric, coverage below guard). Writes `gate_report.md` +
  `gate.json` (pass/fail per metric with deltas).
- Only gate-worthy metrics (per §5) belong in the committed config;
  report-only metrics may be listed with `check = false`.
- **GitHub Actions:** extend `ci.yml` with an optional `eval-gate` job —
  runs the gate against a small committed *smoke task set* (~30 tasks,
  judge-only via the free RunPod endpoint, no GPU needed) when a
  `RUNPOD_API_KEY` secret is present; skips cleanly otherwise. Full
  230-task runs stay on-demand (GPU pod or API budget).

## 8. CLI

```bash
python -m llm_eval import-tasks --legacy ../qwen-qlora/data/test.jsonl \
        -o data/tasks/tutor_test.jsonl
python -m llm_eval generate --config configs/tutor_run.toml [--limit N] [--seeds 0,1,2]
python -m llm_eval grade     --run tutor_20261102            # deterministic graders
python -m llm_eval judge     --run tutor_20261102 --mode pointwise|pairwise
python -m llm_eval calibrate --run tutor_20261102 --labels data/human_labels/tutor_labels.jsonl
python -m llm_eval report    --run tutor_20261102            # summary.json + report.md
python -m llm_eval gate      --run tutor_20261102 --baseline baselines/tutor.json
```

All commands are idempotent and resume-safe; `--limit` supports cheap smoke
tests. On a GPU pod: `pip install -r requirements-gpu.txt` (torch/transformers/
peft/bitsandbytes) then `python -m llm_eval generate …` — the local backend
is the only path that needs the pod.

## 9. Testing strategy

Everything in CI runs **CPU-only, no network, no weights** (matches P1's
test convention):

| File | Covers |
| --- | --- |
| `test_tasks.py` | schema validation (valid/invalid fixtures), legacy import idempotency, task-id determinism |
| `test_config.py` | TOML parsing, env-var resolution, judge≠testee guard, threshold config |
| `test_graders.py` | exact match incl. fraction equivalence (`3/36 == 1/12`, P1 fixtures), format heuristics, abstention probe on canned answers (honest refusal / confident-wrong / normal) |
| `test_judge_parse.py` | parse ladder: clean JSON, fenced JSON, prose-wrapped JSON, per-dimension regex fallback, garbage → counted as parse failure. Fixtures modeled on P1's 10/255 real failures |
| `test_backends.py` | `openai_compat` against a local fake HTTP server (stdlib `http.server` in a thread): happy path, RunPod-shaped response, 500 → retry/backoff, resume-skip logic |
| `test_stats.py` | bootstrap on known distributions (coverage sanity: true mean inside CI for N=500 reps), McNemar p-values vs hand-computed binomials (3+ cases incl. b+c=0), κ on constructed 5×5 tables, Spearman on tied ranks |
| `test_gate.py` | pass / regression / error exit codes; ci_low_min; coverage guard; task-hash mismatch warning |
| `test_e2e.py` | 10-fixture-task set through the full pipeline with a `FakeBackend` (no network): generations → grade → judge → report → gate, asserting `summary.json` keys exist and gate exits 0 |

CI: root `ci.yml` unchanged — the new `pythonpath`/`testpaths` entries make
`uv run pytest` pick the project up automatically.

## 10. Milestones (20–30 h)

| # | Milestone | Est. | Acceptance criteria |
| --- | --- | --- | --- |
| M0 | Scaffold + data model | 2–3 h | Project skeleton, root pyproject entries, `config.py`, `tasks.py` + legacy import, schema tests. `uv run pytest` green; P1 `test.jsonl` imports to a valid task file |
| M1 | Backends + generation | 4–6 h | 10-task smoke vs the free RunPod endpoint: full `generations.jsonl`; kill mid-run + re-run generates only the remainder; fake-HTTP unit tests green. Local backend: smoke on the P1 pod (adapter on/off, peak VRAM logged) |
| M2 | Graders + judge | 5–6 h | Judge the P1 test set (230 lora + 25 base): parse coverage ≥ 97% (P1 baseline 96.1%); scores vs P1's legacy `judge_scores.jsonl` correlate ≥ 0.8 per dimension (port validation); pairwise smoke on the 25 pairs with swap |
| M3 | Statistics layer | 3–4 h | All unit tests vs hand-computed values pass; McNemar on P1's 25 paired base-vs-lora exact-matches; bootstrap CIs on the 230-sample judge means |
| M4 | Calibration | 2–3 h code + 2–3 h labeling | 100–150 labels per protocol; report with per-dimension ρ/κ/bias/5×5 matrices; each dimension explicitly marked gate-worthy or report-only |
| M5 | Reports + CI gate | 3–4 h | `gate` passes against its own baseline (exit 0); a planted −0.05 regression fails it (exit 1); coverage guard triggers on a 90% fixture; Actions job runs (skips without secret) |
| M6 | Dogfood + README + blog | 3–5 h | P1's published numbers reproduce from P0 outputs alone; new numbers (CIs, McNemar p on 25 pairs, seed variance over 3 seeds, calibration report, total eval cost) in the final report; README + blog outline committed |

**Status (2026-11-02):** M0 complete - 27 tests green, ruff clean, root
pyproject entries added. Deliberate deviations, noted for the record:
(1) P1's 230-row `test.jsonl` is gitignored and absent on this machine,
so the legacy import is verified against the committed 10-seed file plus
fixtures; the one-liner to import the real split is in the README and
runs the moment the split exists (locally or on a pod). (2) Judge
dimensions are validated at config load (fail fast), one layer stricter
than the original plan.

**Sequencing note:** M1–M3 can interleave with M4's labeling (label the P1
outputs you already have while M3 builds). M6 is the real gate: if P1's
numbers don't reproduce, M2's port has a bug — find it before M6, not after.

## 11. Risks & mitigations

| Risk | Mitigation |
| --- | --- |
| Judge endpoint flaky / parse failures | parse ladder + explicit coverage (gated at 95%) + resume-safe re-runs; never silent exclusion |
| Judge self-bias | config guard (judge ≠ condition model) + same-family warning |
| Judge saturation (P1: 32B near 5.0 on STEM) | anchored anchors; exact-match stays the strict signal; calibration *measures* the ceiling per dimension and routes gate-worthy-ness accordingly |
| Labeling effort slips | floor is 100 labels (~2 h); protocol documented either way |
| Rented-GPU time | local backend needed only for smoke + P1 re-eval (~1–2 pod-h, reuses P1's workflow); all else CPU |
| Scope creep (benchmarks, UIs, leaderboards) | non-goals list (§1); benchmarks are just task files later |
| Hand-rolled statistics bugs | the unit tests *are* the artifact: hand-computed reference values committed as fixtures |
| `summary.json` contract churn breaking the gate | versioned `meta.schema` key; gate fails loudly (exit 2) on unknown keys rather than comparing silently |

## 12. Downstream contract (what P1–P4 consume)

- **P1:** re-eval through P0 (dogfood); P1's own scripts stay frozen.
- **P2:** pairwise judge (position-swapped) labels chosen/rejected pairs;
  `--seeds` + temperature for candidate generation; abstention probe for the
  DPO ablation question; label-file format reused for human spot-checks;
  McNemar/paired bootstrap for SFT-only vs SFT+DPO.
- **P3:** per-stage evals (SFT → RM → GRPO) on the same harness; red-team
  mini-suite = a task file of adversarial prompts + `safety` rubric dimension
  + a `safety_probe` grader (pass-through rate on attacks, false-refusal rate
  on benign prompts — both binary, so McNemar applies); RM held-out pair
  accuracy uses the pairwise agreement stats; reward hacking monitored via
  the length/word-budget metrics already in `summary.json`.
- **P4:** retrieval metrics (recall@k, MRR) and tool-call metrics as new
  deterministic graders; faithfulness via the judge; semantic-cache
  wrong/stale rate = a pass-rate metric with bootstrap CIs; guardrail layer =
  red-team suite + `wall_s` latency metrics.

## 13. Blog post #1 (M6 deliverable)

Working title: *"An eval harness I trust before I train one more model."*
Differentiator vs. inspect / lm-eval-harness / llm-eval-harness: **judge
calibration and the statistics layer as first-class citizens**, zero new
deps, single-GPU-friendly.

Outline: (1) why naive LLM-judge numbers lie — saturation, position bias,
silently dropped samples (P1's 10/255 as the motivating incident); (2) the
harness in 10 minutes (task files, backends, resume safety); (3) calibrating
the judge against 100 human labels — κ, bias, and which dimensions earned the
gate; (4) bootstrap CIs and exact McNemar instead of "it's 3 points higher";
(5) seed variance and what greedy decoding hides; (6) the CI gate that failed
a run (or will, eventually — honesty clause); (7) cost: a full 230-task eval
for ~$X. Repro box + link to the repo.
