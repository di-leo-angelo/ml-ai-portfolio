#!/usr/bin/env python3
"""Generate tutor responses over the test set with the LoRA adapter on/off.

Runs on the GPU pod (python3). Loads the 4-bit base model once, then
greedy-generates (deterministic) responses for:
  - condition "lora": all 230 test samples (adapter active)
  - condition "base": a 25-sample comparison set (same weights, no adapter)
The LoRA rows for the comparison set come from the same full 230-sample pass,
so base/lora pairs share identical prompts.

Output: reports/auto_eval/raw_outputs.jsonl (append-only, resume-safe:
re-running skips keys already present).
"""

import argparse
import json
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parents[1]
TEST_FILE = ROOT / "data" / "test.jsonl"
OUT_FILE = ROOT / "reports" / "auto_eval" / "raw_outputs.jsonl"
COMPARE_PER_SUBJECT = 2  # 2 x 10 subjects = 20
COMPARE_EXTRA = {"probability": 3, "algebra": 2}  # +5 -> 25 total


def load_samples():
    return [json.loads(line) for line in TEST_FILE.open() if line.strip()]


def pick_comparison_set(samples):
    by_subject = {}
    for i, sample in enumerate(samples):
        by_subject.setdefault(sample["subject"], []).append(i)
    picked = []
    for _, idxs in by_subject.items():
        picked.extend(idxs[:COMPARE_PER_SUBJECT])
    for subject, extra in COMPARE_EXTRA.items():
        picked.extend(
            by_subject.get(subject, [])[
                COMPARE_PER_SUBJECT : COMPARE_PER_SUBJECT + extra
            ]
        )
    return picked[:25]


def make_prompt(tokenizer, sample):
    messages = [
        {"role": "system", "content": sample["system"]},
        {"role": "user", "content": sample["user"]},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


@torch.inference_mode()
def generate(model, tokenizer, text, max_new_tokens):
    ids = tokenizer(text, return_tensors="pt").to(model.device)
    prompt_len = ids["input_ids"].shape[1]
    out = model.generate(**ids, do_sample=False, max_new_tokens=max_new_tokens)
    return tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models"))
    ap.add_argument("--adapter", default=str(ROOT / "artifacts"))
    ap.add_argument("--max-new-tokens", type=int, default=512)
    args = ap.parse_args()

    samples = load_samples()
    compare_ids = pick_comparison_set(samples)
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if OUT_FILE.exists():
        for line in OUT_FILE.open():
            if line.strip():
                done.add(json.loads(line)["key"])

    nf4 = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=nf4, device_map="auto", trust_remote_code=True
    )
    n_params = sum(p.numel() for p in base_model.parameters())
    print(f"base model loaded: {n_params / 1e9:.1f}B params (4-bit)", flush=True)

    jobs = [(f"{i}:lora", i, "lora") for i in range(len(samples))]
    jobs += [(f"{i}:base", i, "base") for i in compare_ids]

    lora_model = None
    peak_gb = 0.0
    count = 0
    for key, i, condition in jobs:
        if key in done:
            continue
        if condition == "lora" and lora_model is None:
            lora_model = PeftModel.from_pretrained(base_model, args.adapter)
            print("adapter active", flush=True)
        model = lora_model if condition == "lora" else base_model
        start = time.time()
        output = generate(
            model, tokenizer, make_prompt(tokenizer, samples[i]), args.max_new_tokens
        )
        record = {
            "key": key,
            "sample_id": i,
            "subject": samples[i]["subject"],
            "difficulty": samples[i]["difficulty"],
            "condition": condition,
            "wall_s": round(time.time() - start, 1),
            "output": output,
        }
        with OUT_FILE.open("a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        torch.cuda.reset_peak_memory_stats()
        peak_gb = max(peak_gb, torch.cuda.max_memory_allocated() / 1e9)
        count += 1
        print(
            f"[{count}] {key} {record['wall_s']}s peak {peak_gb:.1f}GB "
            f":: {output[:70]}",
            flush=True,
        )

    print(
        f"DONE: {count} generations, peak VRAM {peak_gb:.1f} GB. Output: {OUT_FILE}",
        flush=True,
    )


if __name__ == "__main__":
    main()
