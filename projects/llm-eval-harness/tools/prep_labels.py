"""Build the M4 human-labeling set and the self-contained labeling page.

Draws n=120 answers (protocol: 100-150) from the P0 dogfood generations
(230 lora + 25 base) - stratified by subject x difficulty (proportional,
seeded) plus a ~20% deliberate hard slice (long answers, visibly wrong
math, base-condition outputs, judge-scored-mediocre answers). Fully
deterministic (seed 1337): re-running on the same interpreter reproduces
the identical set.

Outputs (both gitignored, under ``data/human_labels/``):
  label_set.jsonl  - the items, in labeling order
  labeling.html    - the labeling UI with the set embedded

The labeler's downloaded file is ``tutor_labels.jsonl`` (PLAN §3.4 format,
restricted to the three dimensions the dogfood judge actually scored:
correctness / format / completeness - tone/safety/helpfulness were not in
the P1 rubric port, so we label the dimensions the judge sees; that is
exactly what M4 calibrates).

Usage (from the project root):
    python tools/prep_labels.py
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
TEST = REPO / "projects/qwen-qlora/data/test.jsonl"
GENS = ROOT / "reports/qwen-qlora-dogfood/generations.jsonl"
GRADES = ROOT / "reports/qwen-qlora-dogfood/grades.jsonl"
JUDGE = ROOT / "reports/qwen-qlora-dogfood/judge_scores.jsonl"
TEMPLATE = ROOT / "tools/labeling_template.html"
OUT_DIR = ROOT / "data/human_labels"

SEED = 1337
TOTAL = 120
HARD_LONG = 8
HARD_WRONG = 8
HARD_BASE = 6
HARD_MEDIOCRE = 2  # lora, min judge dimension <= 3


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def task_id_for(index: int) -> str:
    # P1 convention (see llm_eval_harness.legacy): 0-based test row -> p1_NNNN
    return f"p1_{index + 1:04d}"


def main() -> None:
    tasks = load_jsonl(TEST)
    assert len(tasks) == 230, f"expected 230 test rows, got {len(tasks)}"
    gens = load_jsonl(GENS)
    grades = {(g["task_id"], g["condition"]): g for g in load_jsonl(GRADES)}
    judge = {(j["task_id"], j["condition"]): j for j in load_jsonl(JUDGE)}

    lora = [g for g in gens if g["condition"] == "lora"]
    base = [g for g in gens if g["condition"] == "base"]
    lora_by_id = {g["task_id"]: g for g in lora}
    assert len(lora) == 230 and len(base) == 25, (len(lora), len(base))
    assert len(grades) == 255 and len(judge) == 255, "grades/judge must cover 255"

    rng = random.Random(SEED)

    # --- hard slice (~20%): deduplicated by (task_id, condition) ---------
    hard: list[tuple[dict, str]] = []  # (generation row, stratum)
    picked: set[tuple[str, str]] = set()

    def take(g: dict, stratum: str) -> None:
        key = (g["task_id"], g["condition"])
        if key not in picked:
            picked.add(key)
            hard.append((g, stratum))

    longest = sorted(lora, key=lambda g: (-len(g["output"]), g["task_id"]))
    for g in longest:
        take(g, "hard:long")
        if sum(1 for h in hard if h[1] == "hard:long") >= HARD_LONG:
            break

    wrong = [
        g
        for g in lora
        if grades[(g["task_id"], "lora")].get("exact_match") is False
        and grades[(g["task_id"], "lora")].get("math")
    ]
    assert len(wrong) >= HARD_WRONG, f"only {len(wrong)} wrong-math lora rows"
    rng.shuffle(wrong)
    for g in wrong:
        take(g, "hard:wrong-math")
        if sum(1 for h in hard if h[1] == "hard:wrong-math") >= HARD_WRONG:
            break

    base_sorted = sorted(base, key=lambda g: g["task_id"])
    base_wrong = [
        g
        for g in base_sorted
        if grades[(g["task_id"], "base")].get("exact_match") is False
    ]
    rest = [g for g in base_sorted if g not in base_wrong]
    rng.shuffle(rest)
    for g in base_wrong + rest:
        take(g, "hard:base")
        if sum(1 for h in hard if h[1] == "hard:base") >= HARD_BASE:
            break

    mediocre = [
        g
        for g in lora
        if min(
            judge[(g["task_id"], "lora")]["correctness"],
            judge[(g["task_id"], "lora")]["format"],
            judge[(g["task_id"], "lora")]["completeness"],
        )
        <= 3
    ]
    rng.shuffle(mediocre)
    for g in mediocre:
        take(g, "hard:mediocre")
        if sum(1 for h in hard if h[1] == "hard:mediocre") >= HARD_MEDIOCRE:
            break

    hard_keys = {k[0] for k in picked}
    assert len(hard) >= 20, f"hard slice only {len(hard)} rows"

    # --- proportional strata over subject x difficulty (lora) -----------
    cells: dict[tuple[str, str], list[int]] = {}
    for i, t in enumerate(tasks):
        cells.setdefault((t["subject"], t["difficulty"]), []).append(i)

    strata_total = TOTAL - len(hard)
    quota: dict[tuple[str, str], int] = {
        c: round(strata_total * len(idxs) / 230) for c, idxs in cells.items()
    }
    diff = strata_total - sum(quota.values())
    for c in sorted(quota, key=lambda c: (-len(cells[c]), c)):
        if diff == 0:
            break
        quota[c] += 1 if diff > 0 else -1
        diff -= 1 if diff > 0 else -1

    rng2 = random.Random(SEED + 1)
    chosen: dict[str, dict] = {}  # task_id -> task row
    for subj, diffc in sorted(quota):
        q = quota[(subj, diffc)]
        pool = [i for i in cells[(subj, diffc)] if task_id_for(i) not in hard_keys]
        rng2.shuffle(pool)
        for i in pool[: min(q, len(pool))]:
            chosen[task_id_for(i)] = tasks[i]

    leftover = [
        i
        for i in range(len(tasks))
        if task_id_for(i) not in hard_keys and task_id_for(i) not in chosen
    ]
    rng2.shuffle(leftover)
    for i in leftover[: TOTAL - len(chosen) - len(hard)]:
        chosen[task_id_for(i)] = tasks[i]

    # --- assemble items ---------------------------------------------------
    items: list[dict] = []

    def add(gen: dict, stratum: str) -> None:
        idx = int(gen["task_id"].split("_")[1]) - 1
        t = tasks[idx]
        items.append(
            {
                "task_id": gen["task_id"],
                "condition": gen["condition"],
                "subject": t["subject"],
                "difficulty": t["difficulty"],
                "stratum": stratum,
                "system": t["system"],
                "question": t["user"],
                "reference": t["assistant"],
                "answer": gen["output"],
                "truncated_answer": len(gen["output"]) > 2000,
                "truncated_ref": len(t["assistant"]) > 1200,
            }
        )

    for gen, stratum in hard:
        add(gen, stratum)
    for tid, t in chosen.items():
        add(
            lora_by_id[tid],
            f"stratified:{t['subject']}/{t['difficulty']}",
        )

    rng3 = random.Random(SEED + 2)
    rng3.shuffle(items)
    assert len(items) == TOTAL, f"assembled {len(items)} items, want {TOTAL}"
    assert len({i["task_id"] + ":" + i["condition"] for i in items}) == TOTAL, (
        "duplicated item"
    )

    # --- write outputs ----------------------------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    set_file = OUT_DIR / "label_set.jsonl"
    set_file.write_text(
        "".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items),
        encoding="utf-8",
    )
    set_hash = hashlib.sha256(set_file.read_bytes()).hexdigest()[:16]
    html = TEMPLATE.read_text(encoding="utf-8")
    html = html.replace(
        "__LABEL_SET_JSON__", json.dumps(items, ensure_ascii=False)
    ).replace("__SET_HASH__", set_hash)
    page = OUT_DIR / "labeling.html"
    page.write_text(html, encoding="utf-8")

    # --- summary ----------------------------------------------------------
    print(f"set size: {len(items)} (seed {SEED})")
    print("strata:", dict(Counter(i["stratum"] for i in items)))
    print("conditions:", dict(Counter(i["condition"] for i in items)))
    print(f"  -> {set_file}")
    print(f"  -> {page}")
    print(
        "Labeling: open labeling.html in a browser, score with keys 1-5 / "
        "Y / N / arrows, then 'Download labels' -> save as "
        "data/human_labels/tutor_labels.jsonl. Progress auto-saves in the "
        "browser and can be backed up / restored from the header buttons."
    )


if __name__ == "__main__":
    main()
