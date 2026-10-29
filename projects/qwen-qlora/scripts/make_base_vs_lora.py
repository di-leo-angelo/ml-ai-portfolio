#!/usr/bin/env python3
"""Build the paired base-vs-LoRA side-by-side report.

Reads reports/auto_eval/raw_outputs.jsonl (both conditions) and
data/test.jsonl, pairs the 25 comparison samples (keys N:base + N:lora),
and writes reports/base_vs_lora/comparisons.md: for each pair the question,
the base-model answer, the LoRA answer, format metrics for both, and the
judge scores (when present) for both.

CPU-only; run it after the full raw_outputs.jsonl is pulled back.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUTO = ROOT / "reports" / "auto_eval"
RAW = AUTO / "raw_outputs.jsonl"
TEST_FILE = ROOT / "data" / "test.jsonl"
JUDGE = AUTO / "judge_scores.jsonl"
OUT_DIR = ROOT / "reports" / "base_vs_lora"
OUT = OUT_DIR / "comparisons.md"


def metrics(text):
    words = len(text.split())
    has_steps = bool(re.search(r"(?im)^\s*(step\s*\d+|\d+[.):])", text))
    return words, has_steps


def judge_str(j):
    if not j:
        return ""
    return (f"judge {j['correctness']}/{j['format']}/{j['completeness']} "
            "(corr/format/complete)")


def main():
    samples = [json.loads(l) for l in TEST_FILE.open(encoding="utf-8") if l.strip()]
    recs = {}
    for line in RAW.open(encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            recs[r["key"]] = r
    judge = {}
    if JUDGE.exists():
        for line in JUDGE.open(encoding="utf-8"):
            if line.strip():
                j = json.loads(line)
                judge[j["key"]] = j

    pairs = []
    for i, s in enumerate(samples):
        base = recs.get(f"{i}:base")
        lora = recs.get(f"{i}:lora")
        if base and lora:
            pairs.append((i, s, base, lora))

    lines = [
        "# Base vs LoRA — paired comparison",
        "",
        "Same 4-bit Qwen3.8-27B weights with the LoRA adapter off (base) vs",
        "on (LoRA). Greedy generation, max_new_tokens=512. Judge rubric:",
        "Qwen3-32B-AWQ, 1-5 per dimension.",
        "",
        f"Pairs found: {len(pairs)} / 25",
        "",
    ]
    for i, s, base, lora in pairs:
        bw, bs = metrics(base["output"])
        lw, ls = metrics(lora["output"])
        jb = judge.get(f"{i}:base")
        jl = judge.get(f"{i}:lora")
        lines += [
            f"## [{s['subject']} | {s['difficulty']}] sample {i}",
            "",
            f"**Question:** {s['user']}",
            "",
            f"**Base model** ({base['wall_s']}s, {bw} words, steps={'yes' if bs else 'no'}"
            + (f", {judge_str(jb)}" if jb else "") + "):",
            "",
            base["output"],
            "",
            f"**LoRA (fine-tuned)** ({lora['wall_s']}s, {lw} words, steps={'yes' if ls else 'no'}"
            + (f", {judge_str(jl)}" if jl else "") + "):",
            "",
            lora["output"],
            "",
            "---",
            "",
        ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT} ({len(pairs)} pairs)")


if __name__ == "__main__":
    main()
