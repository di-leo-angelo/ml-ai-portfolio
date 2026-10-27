"""Generate sample answers from the trained adapter on the held-out test set.

Run on a GPU host after training:

    cd projects/qwen-qlora/src
    uv run python -m qwen_qlora.evaluate

The base model loads in 4-bit NF4 (same footprint as training, so it runs on
a 24 GB card). Fine-tuned answers are compared side by side with the
reference answers, and the report is written to reports/eval_report.txt.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from peft import PeftModel

PROJECT_DIR = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(PROJECT_DIR / "src"))

from qwen_qlora.config import DEFAULT_MODEL, load_jsonl  # noqa: E402
from qwen_qlora.model import load_base_model, load_tokenizer  # noqa: E402

ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
REPORTS_DIR = PROJECT_DIR / "reports"
TEST_FILE = PROJECT_DIR / "data" / "test.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the fine-tuned adapter on the test split."
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL,
        help="Base model (HF repo id or local path, default: %(default)s).",
    )
    parser.add_argument(
        "--adapter",
        type=Path,
        default=ARTIFACTS_DIR,
        help="Directory with the saved adapter (default: %(default)s).",
    )
    parser.add_argument("--num-prompts", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not TEST_FILE.exists():
        raise SystemExit(f"{TEST_FILE} not found - run scripts/prepare_data.py first.")

    tokenizer = load_tokenizer(args.model_name)

    print(f"Loading {args.model_name} in 4-bit NF4, adapter: {args.adapter} ...")
    model = load_base_model(args.model_name)
    model = PeftModel.from_pretrained(model, str(args.adapter))
    model.eval()

    records = load_jsonl(TEST_FILE)[: args.num_prompts]
    blocks: list[str] = []
    for record in records:
        messages = [
            {"role": "system", "content": record["system"]},
            {"role": "user", "content": record["user"]},
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        prompt_len = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens, do_sample=False
            )
        generated = tokenizer.decode(
            output_ids[0, prompt_len:], skip_special_tokens=True
        ).strip()
        block = (
            f"### [{record['subject']} | {record['difficulty']}]\n"
            f"Question: {record['user']}\n\n"
            f"Tutor (fine-tuned):\n{generated}\n\n"
            f"Reference answer:\n{record['assistant']}\n"
        )
        print(block)
        blocks.append(block)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report = REPORTS_DIR / "eval_report.txt"
    report.write_text("\n".join(blocks), encoding="utf-8")
    print(f"Report written to {report}")


if __name__ == "__main__":
    main()
