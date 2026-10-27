# Dataset

This project fine-tunes Qwen3.8-27B with QLoRA on **synthetic** school Q&A
turns. The dataset is generated locally by a teacher model from the
committed seed prompts (`samples/seed_prompts.jsonl`) via any
OpenAI-compatible API the user has access to (public, self-hosted, or a
private gateway), so it contains no external content and no personal data. The full generation recipe - subjects,
difficulty mix, sample counts, and prompts - is declared in
`src/qwen_qlora/config.py` so the dataset can be reproduced exactly.

## Layout

| File | Rows | Description |
| --- | --- | --- |
| `samples/seed_prompts.jsonl` | 10 | Committed seeds: one reference question + subject + difficulty per row |
| `qa_raw.jsonl` | 1,500 | Generated Q&A turns (system, user, assistant) + metadata |
| `split.json` | 1 | Split manifest: sizes, per-subject counts, generator model, and SHA-256 hashes |
| `train.jsonl` / `dev.jsonl` / `test.jsonl` | 1,120 / 150 / 230 | Stratified splits (75/10/15 by subject, rounded per subject, fixed seed) |

## Regenerate

```powershell
# 1. Copy .env.example to .env and set OPENAI_API_KEY, then generate
uv run python projects/qwen-qlora/scripts/generate_data.py

# 2. Validate, split, and fingerprint the dataset
uv run python projects/qwen-qlora/scripts/prepare_data.py
```

Generation is resumable (per subject) and idempotent (de-duplicated by
question). Each sample uses a unique per-sample prompt variation, and any
generated question that duplicates an existing one is rejected and
regenerated, so no generated file contains duplicate questions.
`--concurrency N` runs N parallel API calls (default 1). All generated files
are git-ignored; only the seeds and this README are committed.
