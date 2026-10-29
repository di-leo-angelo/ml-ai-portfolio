#!/usr/bin/env python3
"""Grade raw_outputs.jsonl against reference answers. CPU-only, stdlib only.

- Math subjects: extract the final answer from response and reference,
  normalize (fractions via exact arithmetic), exact-match.
- All subjects: format heuristics (step scaffolding, ~350-word cap).
- Merges LLM-judge scores from judge_scores.jsonl when present.

Re-run after the judge step to regenerate the final summary.md.
Writes reports/auto_eval/graded.jsonl and reports/auto_eval/summary.md.
"""

import json
import re
import statistics
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUTO = ROOT / "reports" / "auto_eval"
RAW = AUTO / "raw_outputs.jsonl"
TEST_FILE = ROOT / "data" / "test.jsonl"
JUDGE = AUTO / "judge_scores.jsonl"
GRADED = AUTO / "graded.jsonl"
SUMMARY = AUTO / "summary.md"

MATH_SUBJECTS = {"probability", "algebra", "calculus", "physics", "computer_science"}
FRACTION = re.compile(r"(-?\d+)\s*[/÷]\s*(\d+)")
NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def extract_final_answer(text):
    """Last fraction (preferred) or last number in the tail of the text."""
    tail = text[-500:]
    fracs = FRACTION.findall(tail)
    if fracs:
        num, den = fracs[-1]
        if int(den) != 0:
            return Fraction(int(num), int(den))
        return None
    nums = NUMBER.findall(tail)
    if nums:
        try:
            return Fraction(nums[-1])
        except (ValueError, ZeroDivisionError):
            return None
    return None


def answers_match(pred, ref):
    if pred is None or ref is None:
        return None  # not gradable by exact match
    return abs(float(pred) - float(ref)) < 1e-9


def format_metrics(text):
    words = len(text.split())
    has_steps = bool(re.search(r"(?im)^\s*(step\s*\d+|\d+[.):])", text))
    return {
        "word_count": words,
        "step_scaffolding": has_steps,
        "within_word_cap": words <= 350,
    }


def summarize(rows):
    by_subject = {}
    for row in rows:
        by_subject.setdefault(row["subject"], []).append(row)

    def cell(rows_, fn):
        vals = [fn(r) for r in rows_ if fn(r) is not None]
        return f"{100 * sum(1 for v in vals if v) / len(vals):.0f}%" if vals else "-"

    lines = [
        f"# Auto-eval summary — Qwen3.8-27B + LoRA adapter (test set, n={len(rows)})",
        "",
        "Method: greedy generation (max_new_tokens=512). Exact match = final-answer",
        "equivalence for math subjects (fractions compared exactly; 3/36 == 1/12).",
        "Judge = Qwen3-32B-AWQ rubric scores merged from "
        "judge_scores.jsonl (if present).",
        "",
        "| subject | n | exact match | steps % | under ~350 words % "
        "| median words | median gen (s) |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for subject in sorted(by_subject):
        rows_ = by_subject[subject]
        matches = [r["exact_match"] for r in rows_ if r["math"]]
        n_match = sum(1 for m in matches if m)
        exact = (
            f"{100 * n_match / len(matches):.0f}% ({n_match}/{len(matches)})"
            if matches
            else "-"
        )
        med_words = statistics.median(r["word_count"] for r in rows_)
        med_wall = statistics.median(r["wall_s"] for r in rows_)
        lines.append(
            f"| {subject} | {len(rows_)} | {exact} "
            f"| {cell(rows_, lambda r: r['step_scaffolding'])} "
            f"| {cell(rows_, lambda r: r['within_word_cap'])} "
            f"| {med_words:.0f} | {med_wall:.0f} |"
        )
    all_matches = [r["exact_match"] for r in rows if r["math"]]
    n_all = sum(1 for m in all_matches if m)
    lines.append(
        f"| **overall** | **{len(rows)}** "
        f"| **{100 * n_all / len(all_matches):.0f}% ({n_all}/{len(all_matches)})** "
        f"| {cell(rows, lambda r: r['step_scaffolding'])} "
        f"| {cell(rows, lambda r: r['within_word_cap'])} | | |"
    )

    judge_rows = [r for r in rows if r.get("judge")]
    if judge_rows:
        lines += [
            "",
            "## LLM-judge scores (1-5)",
            "",
            "| subject | n | correctness | format | completeness |",
            "| --- | --- | --- | --- | --- |",
        ]
        for subject in sorted(by_subject):
            jr = [r for r in by_subject[subject] if r.get("judge")]
            if not jr:
                continue

            def avg(field, _jr=jr):
                return sum(r["judge"].get(field, 0) for r in _jr) / len(_jr)

            lines.append(
                f"| {subject} | {len(jr)} | {avg('correctness'):.2f} "
                f"| {avg('format'):.2f} | {avg('completeness'):.2f} |"
            )
    return "\n".join(lines) + "\n"


def main():
    samples = [
        json.loads(line) for line in TEST_FILE.open(encoding="utf-8") if line.strip()
    ]
    judge = {}
    if JUDGE.exists():
        for line in JUDGE.open(encoding="utf-8"):
            if line.strip():
                rec = json.loads(line)
                judge[rec["key"]] = rec

    rows = []
    for line in RAW.open(encoding="utf-8"):
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec["condition"] != "lora":
            continue
        sample = samples[rec["sample_id"]]
        is_math = rec["subject"] in MATH_SUBJECTS
        ref_ans = extract_final_answer(sample["assistant"])
        pred_ans = extract_final_answer(rec["output"])
        row = {
            "key": rec["key"],
            "subject": rec["subject"],
            "difficulty": rec["difficulty"],
            "math": is_math,
            "exact_match": answers_match(pred_ans, ref_ans) if is_math else None,
            "pred_answer": None if pred_ans is None else str(pred_ans),
            "ref_answer": None if ref_ans is None else str(ref_ans),
            "wall_s": rec["wall_s"],
            **format_metrics(rec["output"]),
        }
        if rec["key"] in judge:
            row["judge"] = judge[rec["key"]]
        rows.append(row)

    with GRADED.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    SUMMARY.write_text(summarize(rows), encoding="utf-8")
    print(summarize(rows))


if __name__ == "__main__":
    main()
