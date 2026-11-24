# HANDOVER — llm-eval-harness (P0) takeover

**Who this is for:** a new agent with a fresh clone of this repo, a local
RTX 3090 (24 GB), and no prior context. You own the remaining M2 live
acceptance items, the M1 local-backend smoke (which the 3090 unblocks —
it no longer needs a rented GPU pod), and M3–M6.

## 60-second orientation

- The repo is `ml-ai-portfolio`: a portfolio of self-contained ML projects.
  Two of them form a chain:
  - `projects/qwen-qlora` (**P1**) — a Qwen3.8-27B QLoRA tutor
    fine-tune. **Complete and frozen.** Its recorded eval outputs
    (255 generations, grades, 245 judge scores) are committed as
    reference data.
  - `projects/llm-eval-harness` (**P0**) — an open-source, dep-light
    eval framework. **In progress.** P1's eval scripts are the port
    targets; the port is validated: P0's graders reproduce P1's
    `graded.jsonl` field-for-field (230 rows × 7 fields, 0 mismatches).
- The 3090 unblocks exactly one previously-rented-pod task: running the
  `local` backend (4-bit 27B generation) on this machine.

## Read in this order (~15 min)

1. Root `README.md` — project table + repo conventions.
2. `projects/llm-eval-harness/README.md` — Design notes, Layout,
   Quickstart, and the "Correspondence with P1" table (the canonical
   P1↔P0 mapping).
3. `projects/llm-eval-harness/PLAN.md` §10 (Milestones) — current status,
   design deviations, and the acceptance runbook.
4. `projects/qwen-qlora/README.md` — Results and the "Frozen reference"
   note (why P1's scripts are port targets, not maintenance targets).

## State

- **Done:** M0 (config/tasks/rubric), M1 (backends + resume-safe
  generation + fake-HTTP tests — local-backend smoke still pending),
  M2 (deterministic graders, LLM judge, pairwise, agreement stats,
  `import-generations`; 102 harness tests).
- **Verified on a fresh clone:** 138 tests (all projects), ruff clean,
  and the full CPU parity flow end-to-end (setup step 4 below).
- **Pending:** T1 local smoke (needs the 3090), T2 judge acceptance and
  T3 pairwise smoke (need `RUNPOD_API_KEY` + network), then M3–M6.

## Your hardware: what fits on a 3090 (24 GB)

| Task | VRAM | Fits? |
| --- | --- | --- |
| 4-bit NF4 Qwen3.8-27B greedy generation (P1's exact recipe) | peak ~19 GB (P1 measured 19.3 GB on a 24 GB 4090) | **Yes** — same VRAM, just slower |
| same + LoRA adapter | + ~0.5 GB | **Yes** |
| QLoRA training | ~30 GB peak (a 48 GB 4090 was used) | **No** — and it's already done; never retrain |
| Qwen3-32B judge locally | ~19 GB weights (AWQ), and acceptance requires the *same endpoint P1 used* | **No** — keep the judge on the RunPod endpoint (network); do not try to run it locally |

Disk: base-model snapshot ~55–60 GB (bf16 on disk; 4-bit happens at
load), adapter 499 MB. Plan on 70 GB+ free.

## New-machine setup

```bash
git clone <repo> && cd ml-ai-portfolio
uv sync --extra dev          # heavy: torch/transformers/trl; be patient
```

1. Sanity: `uv run pytest -q` → expect **138 passed**;
   `uv run ruff check .` → clean.
2. CPU parity check (proves the clone + your setup can run the
   validation — no GPU, no network, no keys):

   ```bash
   uv run python -m llm_eval_harness import-generations \
       --input projects/qwen-qlora/reports/auto_eval/raw_outputs.jsonl \
       --output projects/llm-eval-harness/reports/qwen-qlora-dogfood/generations.jsonl
   uv run python -m llm_eval_harness grade \
       --config projects/llm-eval-harness/configs/dogfood_p1.toml
   ```

   Expect `[lora] n=230 exact=45% (52/115) steps=30% under-cap=92%`.
3. Models to disk (only if running T1):

   ```bash
   uv run python projects/qwen-qlora/scripts/download_model.py
   # Qwen/Qwen3.8-27B -> projects/qwen-qlora/models/ (~60 GB)
   uv run python -c "from huggingface_hub import snapshot_download; \
       print(snapshot_download('di-leo-angelo/qwen3.8-27b-qlora-tutor', \
       local_dir='projects/qwen-qlora/artifacts'))"
   # 499 MB adapter -> projects/qwen-qlora/artifacts/ (where
   # example_run.toml's adapter_path points)
   ```

4. Keys: ask the user for a `RUNPOD_API_KEY` value and put it in
   `projects/qwen-qlora/.env` (gitignored; `.env.example` shows the
   format). Needed only for T2/T3. T1 needs none (both model repos are
   public).

## The queue

### T1 — M1 local-backend smoke (the 3090 job, no keys)

`configs/example_run.toml` is set up for exactly this: two conditions
(`base` = bare 4-bit; `lora` = + adapter), P1's decoding
(temp 0, max_tokens 512), P1's committed test split.

```bash
uv run python -m llm_eval_harness generate \
    --config projects/llm-eval-harness/configs/example_run.toml --limit 10
```

- Watch for: `[lora] adapter active: …`, one `peak X.XGB` per generation
  (must stay ≤ ~19 GB on this card), and resume safety (kill mid-run,
  re-run → only the missing keys generate).
- **Acceptance (PLAN §10, M1):** adapter on/off both generate; peak
  VRAM logged and ≤ 24 GB.
- `model_id` is the HF repo id (auto-downloads to the HF cache); you may
  point it at the local `models/` snapshot dir instead if you prefer.
- The full 230×2 run is optional (hours) and only needed for the M6
  parity gate; M2 already runs on P1's recorded outputs.

### T2 — M2 live judge acceptance (needs `RUNPOD_API_KEY`)

```bash
uv run python -m llm_eval_harness judge \
    --config projects/llm-eval-harness/configs/dogfood_p1.toml \
    --env-file projects/qwen-qlora/.env \
    --against projects/qwen-qlora/reports/auto_eval/judge_scores.jsonl
```

- 255 samples = 51 batches of 5. A cold queue takes ~30–60 min; use
  `--limit 5` for a first smoke. Resume-safe: kill and re-run continues.
- Expect the same-family warning (Qwen3-32B judge on Qwen3.8-27B
  outputs) — **known and accepted** for the P1 replication; do not
  suppress or "fix" it.
- **Acceptance:** parse coverage ≥ 97% over all 255 (P1's baseline was
  96.1%; P0 counts explicitly, so it must do at least as well) and
  Pearson ≥ 0.8 per dimension vs the legacy file (Spearman is printed
  too).
- **Overlap caveat:** the P1 legacy file holds 245 rows (220 lora + 25
  base) — P1's parser silently dropped 10 lora rows (the "10/255
  incident"). Correlation is computed on that overlap; P0 must still
  score all 255.

### T3 — M2 pairwise smoke (same key)

```bash
uv run python -m llm_eval_harness pairwise \
    --config projects/llm-eval-harness/configs/dogfood_p1.toml \
    --condition-a lora --condition-b base \
    --env-file projects/qwen-qlora/.env --limit 25
```

- 25 paired base-vs-lora samples, position-swapped. A preference is
  recorded only when both directions agree; disagreements are a
  position-bias tell — report them, don't hide them.
- **Acceptance:** 25 pairs complete, outputs written, no crashes.

### After T1–T3

Update the `PLAN.md` §10 status entry and the P0 README status line
(same house style as the existing entries), then commit
`"M2 acceptance: local smoke + live judge + pairwise"`. M3 (bootstrap
CIs, McNemar) starts immediately — it's pure CPU. M4 (calibration on
human labels), M5 (CI gate), M6 (P1 reproduction gate) are specified in
PLAN.md §10.

## Conventions (this repo)

- `uv` + Python 3.12; tests colocated per project; root `pytest` runs all
  six suites; ruff clean (88 cols, double quotes).
- Commit style: `M2: …` / `fix: …` / `docs: …` — imperative subject +
  a body explaining why.
- TOML = config (read-only input); JSON/JSONL = data (the harness only
  ever appends, resume-safe by key).
- Gitignored by design: `.env` (secrets), P1 `data/train|dev|qa_raw`
  (regeneratable), `artifacts/` (on the HF Hub), `reports/*`
  (regeneratable) — **except the committed reference data, which is
  intentional and commented in the .gitignore files.** Don't commit
  secrets; don't un-commit the reference data without asking.
- P1 is frozen: don't modify `projects/qwen-qlora/scripts/` (the port
  targets) — docs/notes only. P0 owns all new eval work.

## Gotchas (each one cost real time)

1. **Parse ladder commits on valid JSON.** A reply that parses as JSON
   but is invalid on the rubric counts as a parse failure — it is never
   re-guessed by later stages. Don't "improve" coverage by loosening the
   ladder.
2. **Pairwise remap direction.** The swap remap applies to the swapped
   call, not the original. Regression-tested; don't break it.
3. **Torn-tail self-heal.** Both JSONL stores drop a torn final line on
   the next append; the loud malformed-line check is intentional. A
   crash mid-write must self-heal on resume, not error.
4. **The judge must stay the endpoint/model P1 used** (Qwen3-32B-AWQ via
   the public RunPod endpoint; the gateway wants the catalog model name
   `Qwen/Qwen3-32B-AWQ`, not the endpoint id). That is what makes the
   correlation meaningful.
5. **200 + error body = failed job** (an old M1 bug captured gateway 404
   bodies as records). The backend normalizes response shapes — don't
   reintroduce raw capture.
6. **Regenerating P1 data** (train/dev, only needed for M4 labeling):
   `projects/qwen-qlora/.env` needs `OPENAI_BASE_URL`/`OPENAI_API_KEY`
   (see `.env.example`), then `generate_data.py` + `prepare_data.py`.
   The 230-row test split and all eval outputs are committed — you
   should never need to regenerate for T1–T3.
7. **Long jobs exceed any 30-second shell window.** Run generation and
   judge runs detached (tmux on Linux, `Start-Process … -PassThru` + poll
   on Windows) and watch the log files.
8. **Idempotence differs per command.** `grade` recomputes
   `grades.jsonl` from scratch each run (like P1); `judge`/`generate`
   resume by key. Never hand-edit the JSONL outputs.

## What is NOT in the repo (ask the user; never guess)

- Key values: `RUNPOD_API_KEY` (T2/T3), an HF token if downloads hit
  rate limits, an OpenAI key if regenerating data.
- Old-machine-only state: P1 `logs/` (ops history), P1 train/dev/qa_raw
  data (regeneratable), harness `reports/` (regeneratable), local model
  caches.
- Nothing in the pending queue depends on old-machine state except the
  `RUNPOD_API_KEY`.

## Definition of done for this handover

T1 + T2 + T3 all green → PLAN §10 status entry + P0 README status line →
commit `"M2 acceptance: local smoke + live judge + pairwise"`. Then start
M3 (pure CPU).