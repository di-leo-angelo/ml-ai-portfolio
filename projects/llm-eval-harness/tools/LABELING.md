# M4 labeling - how to run it

Everything is in `data/human_labels/` (gitignored). The tooling is two
files in `tools/`.

## Prepare the set (already done, 2026-11-24)

```
python tools/prep_labels.py
```

- Draws **120 answers** (protocol: 100-150) from the dogfood generations
  (230 lora + 25 base), deterministic (seed 1337) - re-running produces
  the identical set.
- **~80% stratified**: proportional by subject x difficulty across all
  30 cells (the `stratum` field records which cell each item came from).
- **~20% deliberate hard slice** (per protocol):
  - `hard:long` - 8 longest answers
  - `hard:wrong-math` - 8 answers with a visibly wrong final answer
    (deterministic grader `exact_match = false` on math rows)
  - `hard:base` - 6 base-condition outputs (including the one base row
    with a wrong math answer)
  - `hard:mediocre` - 2 lora answers the judge scored <= 3 on some
    dimension
- Writes `label_set.jsonl` (the items) and `labeling.html` (the UI with
  the set embedded - a single self-contained file, no server, no network).

## Label

1. Open `data/human_labels/labeling.html` in any browser (double-click).
2. One screen = one model answer to the question shown. The reference
   answer is collapsed by default - score the model answer on its own
   merits first; expand the reference if you want to check a claim.
3. Score each item:

   | Key / control | Action |
   | --- | --- |
   | `1`-`5` | score the next unrated dimension (correctness -> format -> completeness) |
   | `Y` / `N` | pass yes / no ("would you accept this as a good tutor reply?") |
   | `Enter` or `->` / `<-` | next / previous |
   | auto-advance | on by default - item flips to the next once complete (toggle in the footer) |
   | dot grid | jump to any item (green = complete, yellow = partial) |

   An optional notes box is on every item. Each item takes ~1-2 min, so
   the full set is ~2-3 h; progress auto-saves in the browser
   (localStorage) mid-session. If the page shows a yellow banner
   ("blocks persistent storage"), work in shorter sessions and hit
   **Backup progress** every ~15 items; **Restore** reloads a backup at
   the start of a session.
4. When done (or at any checkpoint), **Download labels** produces
   `tutor_labels.jsonl` with only the *complete* items. Save it to
   `data/human_labels/tutor_labels.jsonl`.

## Label format (PLAN section 3.4)

```json
{"task_id": "p1_0036", "condition": "lora",
 "scores": {"correctness": 4, "format": 5, "completeness": 4},
 "pass": true, "notes": "", "annotator": "self"}
```

Deviation from the PLAN example, intentional: the dogfood judge scores
exactly three dimensions (correctness / format / completeness - the P1
rubric), so labels are restricted to those three. M4 calibration
compares human vs. judge on the same dimensions; labeling
tone/safety/helpfulness would calibrate nothing.

## What M4 does with it

`calibrate --run qwen-qlora-dogfood --labels
data/human_labels/tutor_labels.jsonl` joins the labels to the judge
store on `(task_id, condition)` and reports, per dimension: Spearman /
Pearson / kappa / bias / the 5x5 confusion matrix, and marks each
dimension gate-worthy or report-only. (M4 builds that command.)
