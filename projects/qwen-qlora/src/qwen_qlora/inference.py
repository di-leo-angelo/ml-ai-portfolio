"""Interactive chat with the fine-tuned tutor (4-bit base + LoRA adapter).

Run on a GPU host, after training:

    cd projects/qwen-qlora/src
    uv run python -m qwen_qlora.inference

Type a question and press Enter; "exit" or Ctrl-C ends the session.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from peft import PeftModel

PROJECT_DIR = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(PROJECT_DIR / "src"))

from qwen_qlora.config import DEFAULT_MODEL  # noqa: E402
from qwen_qlora.data import SYSTEM_PROMPT  # noqa: E402
from qwen_qlora.model import load_base_model, load_tokenizer  # noqa: E402

ARTIFACTS_DIR = PROJECT_DIR / "artifacts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chat interactively with the fine-tuned tutor."
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
    parser.add_argument("--max-new-tokens", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = load_tokenizer(args.model_name)

    model = load_base_model(args.model_name)
    model = PeftModel.from_pretrained(model, str(args.adapter))
    model.eval()

    print("Fine-tuned tutor ready. Type a question (exit to quit).")
    history: list[dict] = []
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break
        history.append({"role": "user", "content": question})
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, *history]
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
        answer = tokenizer.decode(
            output_ids[0, prompt_len:], skip_special_tokens=True
        ).strip()
        print(f"Tutor: {answer}")
        history.append({"role": "assistant", "content": answer})
    print("Goodbye.")


if __name__ == "__main__":
    main()
