#!/usr/bin/env python3
"""Batch LLM-judge scoring for reports/auto_eval/raw_outputs.jsonl.

Calls the Qwen3-32B-AWQ public RunPod endpoint (no deployment needed) in
batches of 5 samples, scoring each (question, model answer, reference)
triple on correctness / format / completeness (1-5 each). Output:
reports/auto_eval/judge_scores.jsonl, one line per sample:
    {"key": "42:lora", "correctness": 4, "format": 5, "completeness": 4}

Resume-safe: keys already present in judge_scores.jsonl are skipped, so
re-running only judges what is missing. Run scripts/auto_grade.py after
this (or after each interruption) to regenerate summary.md.

CPU-side orchestration only; the sole network call is the judge request.
"""
import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUTO = ROOT / "reports" / "auto_eval"
RAW = AUTO / "raw_outputs.jsonl"
TEST_FILE = ROOT / "data" / "test.jsonl"
JUDGE_OUT = AUTO / "judge_scores.jsonl"

ENDPOINT = "https://api.runpod.ai/v2/qwen3-32b-awq/openai/v1/chat/completions"
BATCH = 5
MAX_RETRIES = 3

RUBRIC = (
    "You are grading tutor-style answers to school-level questions.\n"
    "For each numbered sample below (question, model answer, reference answer) "
    "give integer scores 1-5:\n"
    "correctness: is the model answer factually and mathematically correct "
    "(a different but valid method is fine)?\n"
    "format: does it follow a step-by-step tutor format (clear steps, concise)?\n"
    "completeness: does it fully address the question and state the final "
    "answer (with a self-check where appropriate)?\n"
    "Respond with ONLY a JSON array of objects, one per sample, in the same "
    'order as the samples: [{"key": "<key>", "correctness": <int>, '
    '"format": <int>, "completeness": <int>}]'
)


def api_key():
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.open(encoding="utf-8"):
            if line.startswith("RUNPOD_API_KEY="):
                return line.split("=", 2)[1].strip()
    return os.environ.get("RUNPOD_API_KEY", "")


def call_judge(payload, key):
    req = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            **({"Authorization": f"Bearer {key}"} if key else {}),
        },
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read().decode())


def extract_content(resp):
    """Pull the judge text out of OpenAI-style or RunPod-serverless responses."""
    if isinstance(resp, dict):
        if "choices" in resp:
            return resp["choices"][0]["message"]["content"]
        if "output" in resp:
            o = resp["output"]
            return o.get("content", json.dumps(o)) if isinstance(o, dict) else str(o)
        if "content" in resp:
            return resp["content"]
    return str(resp)


def build_prompt(batch, samples):
    parts = [RUBRIC, ""]
    for rec in batch:
        sample = samples[rec["sample_id"]]
        parts.append(f"### Sample {rec['key']} (subject: {rec['subject']})")
        parts.append(f"Question: {sample['user']}")
        parts.append(f"Model answer: {rec['output'][:2000]}")
        parts.append(f"Reference answer: {sample['assistant'][:1200]}")
        parts.append("")
    parts.append("Score every sample. Remember: respond with ONLY the JSON array.")
    return "\n".join(parts)


def parse_scores(text, batch):
    """Extract per-sample scores; fall back to per-key regex if JSON breaks."""
    rows = []
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if m:
        try:
            arr = json.loads(m.group(0))
        except json.JSONDecodeError:
            arr = None
        if isinstance(arr, list) and len(arr) == len(batch):
            for i, rec in enumerate(batch):
                obj = arr[i]
                if isinstance(obj, dict):
                    rows.append({
                        "key": rec["key"],
                        "correctness": int(obj.get("correctness", 0)),
                        "format": int(obj.get("format", 0)),
                        "completeness": int(obj.get("completeness", 0)),
                    })
    if len(rows) == len(batch):
        return rows
    rows = []
    for rec in batch:
        pat = re.compile(
            re.escape(rec["key"]) + r".{0,400}?"
            r"correctness[\"']?\s*[:=]\s*(\d+).{0,200}?"
            r"format[\"']?\s*[:=]\s*(\d+).{0,200}?"
            r"completeness[\"']?\s*[:=]\s*(\d+)",
            re.DOTALL,
        )
        m = pat.search(text)
        if m:
            rows.append({
                "key": rec["key"],
                "correctness": int(m.group(1)),
                "format": int(m.group(2)),
                "completeness": int(m.group(3)),
            })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0,
                    help="Judge at most N samples (0 = all missing). Smoke tests.")
    ap.add_argument("--only", choices=["lora", "base", "all"], default="all",
                    help="Condition filter (default: all).")
    args = ap.parse_args()

    samples = [json.loads(l) for l in TEST_FILE.open(encoding="utf-8") if l.strip()]
    records = [json.loads(l) for l in RAW.open(encoding="utf-8") if l.strip()]
    done = set()
    if JUDGE_OUT.exists():
        for line in JUDGE_OUT.open(encoding="utf-8"):
            if line.strip():
                done.add(json.loads(line)["key"])
    todo = [r for r in records
            if r["key"] not in done and (args.only == "all" or r["condition"] == args.only)]
    if args.limit:
        todo = todo[: args.limit]
    if not todo:
        print(f"nothing to do ({len(done)} already judged)")
        return

    key = api_key()
    batches = [todo[i:i + BATCH] for i in range(0, len(todo), BATCH)]
    print(f"judging {len(todo)} samples in {len(batches)} batches of {BATCH} "
          f"(endpoint {ENDPOINT})")
    ok = fail = 0
    with JUDGE_OUT.open("a", encoding="utf-8") as out:
        for bi, batch in enumerate(batches, 1):
            payload = {
                "messages": [{"role": "user", "content": build_prompt(batch, samples)}],
                "temperature": 0,
                "max_tokens": 1024,
            }
            rows = None
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    rows = parse_scores(extract_content(call_judge(payload, key)), batch)
                    if rows:
                        break
                    print(f"batch {bi}/{len(batches)}: unparseable (attempt {attempt})")
                except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
                    print(f"batch {bi}/{len(batches)}: error (attempt {attempt}): {e}")
                time.sleep(5 * attempt)
            if rows:
                for row in rows:
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
                ok += len(batch)
            else:
                fail += len(batch)
                print(f"batch {bi} FAILED, keys: {[r['key'] for r in batch]}")
            print(f"[{bi}/{len(batches)}] ok={ok} failed={fail}")
    print(f"done: {ok} judged, {fail} failed. Output: {JUDGE_OUT}")
    if fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
