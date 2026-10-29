#!/usr/bin/env python3
"""Publish the trained LoRA adapter (artifacts/) to the Hugging Face Hub.

Reads HF_HUB_API_KEY from the project .env, creates the repo if needed,
and uploads each file individually (small files first, big safetensors
last) with per-file logging so a stall is attributable to a file.
Safe to re-run: existing repo is reused and files overwrite same names.
"""
import re
import sys
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
REPO_ID = "di-leo-angelo/qwen3.8-27b-qlora-tutor"
ART = ROOT / "artifacts"
BIG = "adapter_model.safetensors"


def get_token():
    text = (ROOT / ".env").read_text(encoding="utf-8")
    m = re.search(r"^HF_HUB_API_KEY=(.+)$", text, re.M)
    return m.group(1).strip() if m else None


def main():
    token = get_token()
    if not token:
        sys.exit("HF_HUB_API_KEY not found in .env")
    api = HfApi(token=token)
    print("whoami:", api.whoami().get("name"), flush=True)
    api.create_repo(repo_id=REPO_ID, repo_type="model", private=False, exist_ok=True)
    print("repo ready:", REPO_ID, flush=True)

    files = [p.name for p in sorted(ART.iterdir()) if p.is_file()]
    # small files first, big safetensors last
    files.sort(key=lambda n: (n == BIG, n))
    for name in files:
        size_mb = (ART / name).stat().st_size / 1e6
        print(f"uploading {name} ({size_mb:.1f} MB)...", flush=True)
        api.upload_file(
            path_or_fileobj=str(ART / name),
            path_in_repo=name,
            repo_id=REPO_ID,
            repo_type="model",
            revision="main",
        )
        print(f"done: {name}", flush=True)
    print("ALL FILES UPLOADED ->", f"https://huggingface.co/{REPO_ID}", flush=True)


if __name__ == "__main__":
    main()
