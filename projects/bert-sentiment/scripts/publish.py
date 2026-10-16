"""Publish a fine-tuned SST-2 model to the Hugging Face Hub.

Dry run by default: it verifies the token and prints what would be uploaded.
Pass --push to actually create or update the Hub repository. The HF token is
read from the HF_TOKEN environment variable, falling back to the git-ignored
.env.local file in the repository root.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, whoami

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parents[1]
ENV_LOCAL_PATH = REPO_ROOT / ".env.local"
DEFAULT_MODELS_DIR = PROJECT_DIR / "models"

USER = "di-leo-angelo"
#: Short model name -> (checkpoint dir name, Hub repository id).
REPOS = {
    "bert-base": ("bert-base", f"{USER}/sst2-sentiment-bert-base"),
    "distilbert": ("distilbert", f"{USER}/sst2-sentiment-distilbert"),
}

sys.path.insert(0, str(PROJECT_DIR / "src"))

from sentiment.data import MODEL_IDS  # noqa: E402
from sentiment.env import parse_env_local  # noqa: E402
from sentiment.model_card import build_model_card  # noqa: E402


def resolve_token() -> str:
    """Return the HF token from the environment or .env.local."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        token = parse_env_local(ENV_LOCAL_PATH).get("HF_TOKEN", "")
    if not token:
        raise SystemExit(
            "No HF token found. Set the HF_TOKEN environment variable or add "
            f"HF_TOKEN=<token> to {ENV_LOCAL_PATH}."
        )
    return token


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the publishing command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Publish a fine-tuned SST-2 model to the Hugging Face Hub."
    )
    parser.add_argument(
        "--model",
        choices=sorted(REPOS),
        default="bert-base",
        help="Which fine-tuned checkpoint to publish (default: %(default)s).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint directory (default: models/<model>).",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Actually upload (default: dry run).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Verify the token, write the model card, and (optionally) upload."""
    args = parse_args(argv)
    dir_name, repo_id = REPOS[args.model]
    hub_id = MODEL_IDS[args.model]
    checkpoint = args.checkpoint or DEFAULT_MODELS_DIR / dir_name
    if not checkpoint.is_absolute():
        checkpoint = PROJECT_DIR / checkpoint
    if not checkpoint.is_dir():
        raise SystemExit(
            f"Checkpoint directory not found: {checkpoint}\n"
            "Train the model first: python -m sentiment.train --model "
            f"{args.model}"
        )

    token = resolve_token()
    profile = whoami(token=token)
    print(f"Authenticated as {profile.get('name')} (token: {token[:6]}...)")

    model_card = build_model_card(checkpoint, repo_id, hub_id)
    card_path = checkpoint / "README.md"
    card_path.write_text(model_card, encoding="utf-8")
    print(f"Model card written -> {card_path}")

    files = sorted(p.name for p in checkpoint.iterdir() if p.is_file())
    print(f"Files in {checkpoint}: {', '.join(files)}")

    if not args.push:
        print(
            f"Dry run complete. To upload, rerun with --push:\n"
            f"  python projects/bert-sentiment/scripts/publish.py "
            f"--model {args.model} --push"
        )
        return

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        folder_path=str(checkpoint),
        path_in_repo="",
        commit_message=f"Publish {dir_name} fine-tuned on SST-2 "
        "(weights, tokenizer, model card, training history)",
    )
    print(f"Published to https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
