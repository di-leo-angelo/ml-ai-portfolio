"""QLoRA fine-tuning of Qwen3.8-27B on the synthetic school Q&A dataset.

Run this on a rented 24+ GB GPU (e.g. RunPod), inside tmux -- not on the
local machine:

    # On the GPU host, after cloning the repo and copying the prepared data/:
    pip install -r projects/qwen-qlora/requirements-gpu.txt
    python projects/qwen-qlora/scripts/download_model.py
    cd projects/qwen-qlora
    python src/qwen_qlora/train.py --model-name models

The trained artifact is a LoRA adapter of a few hundred MB, written to
artifacts/ -- that is what you copy back to your machine.

Implementation notes
--------------------
* Qwen3.8-27B is a hybrid model: 48 linear-attention (Mamba-style SSM)
  layers interleaved with 16 full-attention layers (64 total), plus a
  vision tower that stays frozen and unused for this text-only SFT.
* The training loop is hand-rolled (no TRL): single-sequence micro-batches
  (no padding), gradient accumulation, bf16 autocast, gradient
  checkpointing, cosine schedule with warmup, best-dev-loss saving. Every
  step is visible in this file.
* LoRA adapters attach to all linear layers except the vision tower, MTP
  head, and LM head.
* Sequences are rendered through the chat template with
  ``enable_thinking=False`` so the model learns the direct answer format,
  not a reasoning block.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, get_peft_model
from transformers import get_cosine_schedule_with_warmup

PROJECT_DIR = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(PROJECT_DIR / "src"))

from qwen_qlora.config import DEFAULT_MODEL, load_jsonl  # noqa: E402
from qwen_qlora.data import build_texts  # noqa: E402
from qwen_qlora.model import load_base_model, load_tokenizer  # noqa: E402

DATA_DIR = PROJECT_DIR / "data"
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"

#: Substring-matched module names that stay frozen (no LoRA adapters):
#: the vision tower, the multi-token-prediction head, and the LM head.
LORA_MODULES_TO_EXCLUDE = ("visual", "mtp", "lm_head")

WARMUP_RATIO = 0.03
LOG_EVERY_STEPS = 10


def seed_everything(seed: int) -> None:
    """Seed all RNGs so a training run is reproducible."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="QLoRA fine-tuning of the Qwen3.8-27B tutor model."
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL,
        help="Base model (HF repo id or local path, default: %(default)s).",
    )
    parser.add_argument("--r", type=int, default=16, help="LoRA rank.")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha.")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=8,
        help="Gradient accumulation steps (effective batch = 1 * 8).",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=1024,
        help="Max tokens per sequence; 1024 covers every sample in this "
        "dataset (longest is ~450 tokens).",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def prepare_sequences(
    tokenizer: Any, path: Path, max_length: int
) -> list[torch.Tensor]:
    """Render records through the chat template and tokenize them."""
    records = load_jsonl(path)
    texts = build_texts(tokenizer, records)
    sequences: list[torch.Tensor] = []
    for text in texts:
        encoded = tokenizer(
            text, truncation=True, max_length=max_length, return_tensors="pt"
        )
        sequences.append(encoded["input_ids"].squeeze(0))
    return sequences


def evaluate_dev(
    model: Any, dev_ids: list[torch.Tensor], device: torch.device
) -> float:
    """Mean dev loss (forward-only, single-sequence micro-batches)."""
    model.eval()
    total_loss, count = 0.0, 0
    with torch.inference_mode():
        for input_ids in dev_ids:
            ids = input_ids.unsqueeze(0).to(device)
            outputs = model(input_ids=ids, labels=ids)
            total_loss += outputs.loss.item()
            count += 1
    return total_loss / count


def main() -> None:
    args = parse_args()
    for name in ("train", "dev"):
        if not (DATA_DIR / f"{name}.jsonl").exists():
            raise SystemExit(
                f"data/{name}.jsonl not found - run scripts/generate_data.py "
                "and scripts/prepare_data.py first."
            )

    seed_everything(args.seed)
    tokenizer = load_tokenizer(args.model_name)

    print(f"Loading {args.model_name} in 4-bit NF4 (the big step) ...")
    model = load_base_model(args.model_name)
    for cfg in (model.config, getattr(model.config, "text_config", None)):
        if cfg is not None:
            cfg.use_cache = False
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    peft_config = LoraConfig(
        r=args.r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
        modules_to_exclude=list(LORA_MODULES_TO_EXCLUDE),
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    print("Rendering chat templates and tokenizing (thinking off) ...")
    train_ids = prepare_sequences(
        tokenizer, DATA_DIR / "train.jsonl", args.max_seq_length
    )
    dev_ids = prepare_sequences(
        tokenizer, DATA_DIR / "dev.jsonl", args.max_seq_length
    )
    print(f"{len(train_ids)} train / {len(dev_ids)} dev sequences prepared")

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, weight_decay=0.01
    )
    steps_per_epoch = math.ceil(len(train_ids) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(WARMUP_RATIO * total_steps)),
        num_training_steps=total_steps,
    )
    device = next(model.parameters()).device

    best_dev_loss = float("inf")
    global_step = 0
    start = time.time()
    for epoch in range(args.epochs):
        order = list(range(len(train_ids)))
        random.Random(args.seed + epoch).shuffle(order)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss, micro_step = 0.0, 0
        for index in order:
            input_ids = train_ids[index].unsqueeze(0).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(input_ids=input_ids, labels=input_ids)
                loss = outputs.loss / args.grad_accum
            loss.backward()
            running_loss += loss.item()
            micro_step += 1
            if micro_step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % LOG_EVERY_STEPS == 0:
                    elapsed = time.time() - start
                    avg = running_loss / LOG_EVERY_STEPS
                    print(
                        f"step {global_step}/{total_steps} "
                        f"train_loss {avg:.4f}  ({elapsed:.0f}s elapsed)"
                    )
                    running_loss = 0.0

        dev_loss = evaluate_dev(model, dev_ids, device)
        print(
            f"epoch {epoch + 1}/{args.epochs} done: dev_loss {dev_loss:.4f} "
            f"(best {best_dev_loss:.4f})  ({time.time() - start:.0f}s elapsed)"
        )
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(ARTIFACTS_DIR))
            tokenizer.save_pretrained(str(ARTIFACTS_DIR))
            print(f"  new best - adapter saved to {ARTIFACTS_DIR}")
        model.train()

    print(
        f"Training complete: {total_steps} steps, best dev loss "
        f"{best_dev_loss:.4f}, {time.time() - start:.0f}s total. "
        f"Adapter: {ARTIFACTS_DIR}"
    )


if __name__ == "__main__":
    main()
