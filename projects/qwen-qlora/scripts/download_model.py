"""Pre-download the base model on a remote GPU host.

Run this on the rented GPU machine *before* starting the training job, so
the large base model is cached locally and the training run never stalls on
a download mid-session:

    uv run python projects/qwen-qlora/scripts/download_model.py

The snapshot lands in the git-ignored ``models/`` directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "Qwen/Qwen3.8-27B"
DEFAULT_CACHE_DIR = PROJECT_DIR / "models"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the base model to the local cache."
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL,
        help="Hugging Face repository id (default: %(default)s).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Directory for the snapshot (default: %(default)s).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Downloading {args.model_name} into {args.cache_dir} ...")
    path = snapshot_download(args.model_name, local_dir=str(args.cache_dir))
    print(f"Base model ready at {path}")


if __name__ == "__main__":
    main()
