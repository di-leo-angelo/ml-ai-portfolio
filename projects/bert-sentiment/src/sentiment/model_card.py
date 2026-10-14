"""Model-card (README.md) generation for Hugging Face Hub repositories."""

from __future__ import annotations

import json
from pathlib import Path

MODEL_CARD_TAGS = [
    "transformers",
    "pytorch",
    "text-classification",
    "sentiment-analysis",
    "sst2",
    "en",
    "dataset:glue",
]


def build_model_card(checkpoint: Path, repo_id: str, hub_id: str) -> str:
    """Compose the model card from the saved ``training_state.json``.

    The file is written into the checkpoint directory by the publisher, so
    it ships with the weights and renders as the Hub README.
    """
    state_path = checkpoint / "training_state.json"
    state = (
        json.loads(state_path.read_text(encoding="utf-8"))
        if state_path.is_file()
        else {}
    )
    best = state.get("best_eval", {})
    hyper = state.get("hyperparameters", {})
    train_samples = state.get("train_samples", "unknown")
    epochs = state.get("epochs_completed", "unknown")
    steps = state.get("global_steps", "unknown")
    quick = state.get("quick", False)

    scope = "a 4,096-sample subset" if quick else "the full 67,349-sample train split"
    lines = [
        "---",
        "library_name: transformers",
        "tags:",
        *[f"  - {tag}" for tag in MODEL_CARD_TAGS],
        f"  - base_model:{hub_id}",
        "---",
        "",
        f"# {repo_id.split('/')[-1].replace('-', ' ').title()} (SST-2 sentiment)",
        "",
        f"{hub_id} fine-tuned on the Stanford Sentiment Treebank (SST-2) for",
        "binary sentiment classification (negative / positive).",
        "",
        f"- **Dev accuracy:** {best.get('accuracy', float('nan')):.4f} on the",
        "  872-sample official dev split (the same split Hugging Face reports",
        "  for GLUE; the 1,821-sample test split has no public labels)",
    ]
    if best.get("f1") is not None:
        lines.append(f"- **Dev F1 (binary):** {best['f1']:.4f}")
    lines += [
        f"- **Train data:** {train_samples:,} sentences over {epochs} epochs"
        if isinstance(train_samples, int)
        else f"- **Train data:** {train_samples} sentences over {epochs} epochs",
        f"- **Steps:** {steps:,}",
        "- **Hardware:** CPU only (Intel i5-8250U, no GPU)",
        "- **Seed:** 42",
        "",
        "## Usage",
        "",
        "```python",
        "from transformers import pipeline",
        "",
        f'classifier = pipeline("text-classification", model="{repo_id}")',
        'print(classifier("The film was a fantastic, hilarious ride."))',
        "```",
        "",
        "## Training",
        "",
        f"Fine-tuned {hub_id} on {scope} with the Hugging Face `Trainer`:",
        "",
        f"- learning rate {hyper.get('learning_rate', 'n/a')}, "
        f"batch size {hyper.get('batch_size', 'n/a')}, "
        f"weight decay {hyper.get('weight_decay', 'n/a')}",
        f"- linear warmup for {hyper.get('warmup_steps', 'n/a')} steps "
        f"(ratio {hyper.get('warmup_ratio', 'n/a')}), then linear decay",
        f"- max sequence length {hyper.get('max_length', 'n/a')}, "
        "gradient clipping at 1.0",
        "- best checkpoint selected by dev accuracy, early stopping "
        "(patience 3 epochs)",
        "",
        "## Notes",
        "",
        "- Headline metric is the dev-split accuracy, matching Hugging Face's",
        "  standard GLUE reporting convention.",
        "- Training data: Stanford Sentiment Treebank (Socher et al., 2013),",
        "  non-commercial research and educational use.",
        "- Weights released for portfolio and educational purposes.",
    ]
    return "\n".join(lines) + "\n"
