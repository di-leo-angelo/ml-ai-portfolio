"""Shared model-loading helpers for training, evaluation, and inference.

Qwen3.8-27B is a multimodal conditional-generation model
(``Qwen3_5ForConditionalGeneration``) with a vision tower and a hybrid of
linear-attention (Mamba-style SSM) and full-attention layers. All
text-only workflows load the full model through transformers' Auto class
and simply never feed images.

The correct Auto class differs between transformers 5.x (the version the
model was built with) and 4.x, so ``load_base_model`` falls back gracefully
-- training on the pod pins transformers >= 5.8 (see
requirements-gpu.txt), the fallback only keeps older environments
importable.
"""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoTokenizer, BitsAndBytesConfig

try:  # transformers >= 5 (required for the qwen3_5 architecture)
    from transformers import AutoModelForMultimodalLM as _AutoLM
except ImportError:  # pragma: no cover - transformers 4.x fallback
    from transformers import AutoModelForCausalLM as _AutoLM


def load_tokenizer(model_name: str | Path) -> AutoTokenizer:
    """Load the tokenizer, guaranteeing a usable pad token."""
    tokenizer = AutoTokenizer.from_pretrained(str(model_name))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base_model(
    model_name: str | Path, load_in_4bit: bool = True
) -> PreTrainedModel:  # noqa: F821 - name exists at runtime
    """Load the base model, quantized to 4-bit NF4 by default.

    Double quantization + bf16 compute keeps the 27.8B base at roughly
    15-16 GB, leaving room for LoRA, optimizer state, and activations on a
    24 GB card.
    """
    quantization_config = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb4bit_use_double_quant=True,
        )
        if load_in_4bit
        else None
    )
    return _AutoLM.from_pretrained(
        str(model_name),
        quantization_config=quantization_config,
        device_map="auto",
        attn_implementation="sdpa",
    )
