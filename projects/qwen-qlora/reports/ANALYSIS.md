# What QLoRA actually did — mechanism, memory arithmetic, and cost

This document explains the *mechanism* of the fine-tune: what a 499 MB
adapter changes about a 27.5B model, why it fits on a consumer GPU, and
what the whole thing cost. Every number below is measured from the actual
run (2026-10-28), not an estimate. Sources: `logs/train.log`,
`data/split.json`, `reports/eval_report.txt`, and `nvidia-smi` reads from
the eval run.

## TL;DR

| Quantity | Value |
| --- | --- |
| Base model | Qwen3.8-27B (Apache-2.0), 27,481,459,440 params |
| Base weights | 52 GB on disk → 16.9 GB VRAM when loaded as NF4 |
| Trained parameters | 124,730,880 — **0.4539 %** of the model |
| Adapter file | 499,105,384 B ≈ 499 MB (fp32 storage of 124.7M bf16 params) |
| Training | 72 min 10 s, 280 steps, single 48 GB RTX 4090, peak ~30 GB |
| Best dev loss | 0.3671 (epoch 1 of 2; epoch 2: 0.3691, no improvement) |
| Inference eval | 255 generations (230 test + 25 base) on a standard 24 GB 4090, **19.3 GB peak** |
| Total compute cost | **$14.11 measured** on RunPod — training + probe + eval pods + LLM judge (§3) |

## 1. The memory arithmetic: why 24 GB is enough

A 27.5B model cannot be full-fine-tuned on any consumer card. The budget:

| Strategy | Per-parameter cost | Total for 27.48B params |
| --- | --- | --- |
| bf16 inference | 2 B (weights only) | ≈ 55 GB |
| Full SFT + AdamW | ≈ 16 B (bf16 weights + grads, fp32 master weights, fp32 m/v moments) | ≈ 440 GB |
| **QLoRA (frozen NF4 base + bf16 LoRA)** | 4.25 B frozen everywhere + ~16 B on 0.45 % of params | **≈ 15 GB + 2 GB + activations** |

The QLoRA decomposition, measured on this run:

1. **Base weights — frozen, 4-bit.** bitsandbytes NF4 (normalfloat4) stores
   27.48B weights at ~4.25 bits each (4-bit mantissa plus double-quantized
   scales) ≈ 14.6 GB theoretical. Measured: **16.9 GB** of VRAM occupied at
   load completion. These weights are *read-only* during training: no
   gradients, no optimizer state, no master copy. The model's knowledge
   lives here and is untouched by the fine-tune.
2. **LoRA parameters — trainable, bf16.** 124.7M params ≈ 0.25 GB bf16
   weights + ~0.25 GB gradients + ~1 GB fp32 AdamW moments ≈ **1.8 GB**.
3. **Activations + KV cache.** bf16 autocast activations with gradient
   checkpointing (training) and the KV cache of the 16 full-attention
   layers (inference). Measured peaks: **~30 GB** during training (on the
   48 GB card), **19.3 GB** during inference (on the 24 GB card).

Two consequences fall out of this arithmetic:

- The training run needed a 48 GB card because of *optimizer state and
  activations*, not because the model is 440 GB. The same adapter could be
  trained on a 24 GB card (shorter sequences, larger accumulation) at the
  cost of wall-clock time — and the inference eval proves the 24 GB fit:
  the fine-tuned model generated 230+ answers with **4.7 GB of headroom**
  (19.3 / 24 GB).
- Inference is strictly cheaper than training: drop gradients and
  optimizer state and a standard consumer card runs the whole model.

## 2. What the adapter changes (and what it doesn't)

**Mechanism.** Every linear layer W in the 64-layer text stack (48
linear-attention layers interleaved with 16 full-attention layers; the
vision tower, MTP head, and LM head are excluded) receives a low-rank
update

```
W' = W + (alpha / r) · B·A ,   r = 16, alpha = 32
```

where A is 16 × d_in and B is d_out × 16. Each layer gains
16·(d_in + d_out) new parameters against its d_in × d_out frozen weights.
Summed across all targeted layers the total is **124,730,880 trainable
parameters — 0.4539 % of the 27,481,459,440 total** (PEFT's own count,
`logs/train.log`). The frozen base contributes the other 99.55 % — at 4
bits, read-only.

**What the 0.45 % learned: presentation, not knowledge.** The training set
is 1,120 samples of *school Q&A in a fixed tutor voice* (system persona →
question → step-by-step answer ending in a self-check), clustered
topically around one committed seed per subject. The base model already
answers these questions correctly; what it lacks is the voice. The
fine-tune transfers that voice into a rank-16 subspace:

- **Same math, different clothing.** On the held-out test questions the
  two models agree on the answers and differ in presentation. Probability
  example (from `reports/eval_report.txt`): the base model writes a
  flowing paragraph; the fine-tuned model writes "Step 1 … Step 4 …
  Therefore the probability is 1/12" and checks its arithmetic. Both
  arrive at 1/12.
- **Tone is a low-dimensional property.** The formatting constraints —
  step numbering, "show your work", self-check, a hard word budget — are a
  small number of correlated features, which is exactly what a rank-16
  update encodes well. Knowledge, by contrast, lives in high-dimensional,
  distributed representations; reshaping *how* the model answers without
  changing *what* it knows is why a 0.45 % parameter budget suffices.
- **It saturated fast.** Best dev loss 0.3671 at epoch 1; epoch 2 ended at
  0.3691 (no improvement, so epoch 1's adapter was kept). Style transfer
  needs ~140 optimizer steps (1,120 samples, effective batch 8) — not the
  thousands that content learning needs.

**What it deliberately did not learn.** Topical coverage is intentionally
narrow: each subject's 150 samples are variations of one seed topic
(different numbers, contexts, framings — not 150 topics). The 230
held-out test samples are topically close to the training cluster, so the
eval measures *style transfer on in-distribution questions*, not
new-knowledge acquisition or generalization to unseen topics. Answer
content is bounded by the base model's knowledge.

## 3. Cost story (measured from RunPod billing, 2026-10-27/29)

| Item | Time | Measured cost |
| --- | --- | --- |
| Training pod (48 GB 4090) | 1 h 12 m | ≈ $2.57 (GPU $2.49 + disk) |
| Probe/setup pod (eval prep) | ≈ 1 h 15 m | ≈ $1.58 |
| Eval generation pod (24 GB 4090, EU-RO-1) | 2 h 51 m | $2.61 (GPU $2.53 + disk $0.08) |
| LLM judge (Qwen3-32B-AWQ, RunPod public endpoint) | ~270 calls | **$6.68** (~$0.025/call — per-token pricing, ~33× higher than the initial $0.001/call estimate) |
| **Total (RunPod, measured)** | | **$14.11** |

Data generation (1,500 samples) ran on the self-hosted NVFP4 gateway: $0
cloud spend. Itemized pod rows are approximate from the hourly billing
records; the total is the billing API's figure. Lesson baked in: an
LLM-as-judge loop is a real line item — budget per-token pricing for the
judge, not per-call.

For comparison: full-parameter SFT of a 27.5B model needs ≈ 440 GB of VRAM
— a multi-GPU H100 node ($2–4/hr per GPU) for days of iteration. The QLoRA
route produced a working adapter for the price of a few hours on a $0.70
card. The entire deliverable that came back over the wire: **499 MB**.

## 4. Reproducibility

- Everything is seeded (seed 42) and fingerprinted: `data/split.json`
  records sizes, split ratios, the generator model, and SHA-256 hashes of
  the raw/train/dev/test files.
- The base model is a public HF download (`scripts/download_model.py`,
  52 GB, ~2 min at pod bandwidth) — intentionally not committed to the
  repo.
- The adapter loads on top of the base with
  `PeftModel.from_pretrained('artifacts')`; the frozen NF4 base is shared
  between training and inference.
- Eval is resume-safe (`scripts/eval_run.py` skips already-written keys)
  and grading is CPU-only (`scripts/auto_grade.py`): exact match with
  fraction normalization for math, format heuristics, and merged
  LLM-judge scores for open-ended samples.
