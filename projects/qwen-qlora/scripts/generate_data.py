"""Generate the synthetic school Q&A dataset.

Expands the committed seed questions (data/samples/seed_prompts.jsonl) into
fresh student-tutor exchanges via an OpenAI-compatible chat API (OpenAI,
OpenRouter, or a local runtime such as Ollama or vLLM). The full generation
recipe - subjects, difficulty mix, sample counts, and prompts - is declared
in src/qwen_qlora/config.py so the dataset can be reproduced exactly.

Requires a .env file in the project root (see .env.example):

    OPENAI_BASE_URL=https://api.openai.com/v1
    OPENAI_API_KEY=...
    OPENAI_MODEL=gpt-4o-mini
    OPENAI_EXTRA_HEADERS=x-custom-auth=...   # optional, for private gateways

Usage (from the repository root):

    uv run python projects/qwen-qlora/scripts/generate_data.py
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes
import json
import os
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
from openai import OpenAI

PROJECT_DIR = Path(__file__).resolve().parents[1]
SEEDS_FILE = PROJECT_DIR / "data" / "samples" / "seed_prompts.jsonl"
RAW_FILE = PROJECT_DIR / "data" / "qa_raw.jsonl"
ENV_FILE = PROJECT_DIR / ".env"
LOCK_FILE = RAW_FILE.parent / "qa_raw.lock"


def _pid_alive(pid: int) -> bool:
    """Return True if a process with this pid currently exists.

    Conservative by design: if liveness cannot be established (e.g. a
    protected system process yields ACCESS_DENIED), report the pid as alive
    so a second generator instance never starts. Only a definitive
    not-found result counts as dead.
    """
    if pid <= 0:
        return False
    if os.name == "nt":
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.restype = ctypes.wintypes.HANDLE
        handle = kernel32.OpenProcess(
            0x1000,  # PROCESS_QUERY_LIMITED_INFORMATION
            False,
            pid,
        )
        if handle:
            kernel32.CloseHandle(handle)
            return True
        err = ctypes.get_last_error()
        # ERROR_FILE_NOT_FOUND (2) / ERROR_INVALID_PARAMETER (87): pid is
        # gone. Anything else (e.g. ACCESS_DENIED 5) means the process
        # exists but is protected: fail safe and report it as alive.
        return err not in (2, 87)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def acquire_lock() -> None:
    """Create a lock file so two generators never rewrite the file at once.

    A stale lock (left by a killed process) is reclaimed automatically.
    """
    for _ in range(3):
        try:
            with LOCK_FILE.open("x", encoding="utf-8") as handle:
                handle.write(str(os.getpid()))
            return
        except FileExistsError:
            raw = LOCK_FILE.read_text(encoding="utf-8").strip()
            try:
                pid = int(raw)
            except ValueError:
                pid = 0
            if _pid_alive(pid):
                raise SystemExit(
                    f"Another generator is already running (pid {pid}); "
                    "refusing to start a second instance."
                ) from None
            try:
                LOCK_FILE.unlink()
            except FileNotFoundError:
                pass
    raise SystemExit(
        f"Could not acquire lock {LOCK_FILE}; is another generator running?"
    )


def release_lock() -> None:
    """Best-effort removal of the lock file (stale locks are auto-reclaimed)."""
    try:
        LOCK_FILE.unlink()
    except FileNotFoundError:
        pass

sys.path.insert(0, str(PROJECT_DIR / "src"))

from qwen_qlora.config import (  # noqa: E402
    DATASET_CONFIG,
    load_jsonl,
    save_jsonl,
)
from qwen_qlora.data import build_user_prompt, normalize_record  # noqa: E402
from qwen_qlora.env import parse_env_local, parse_extra_headers  # noqa: E402


def save_with_retry(records: list[dict], path: Path, attempts: int = 3) -> None:
    """Save with retries so transient file locks never kill a long run.

    The underlying save is atomic (temp file + replace), so a persistent
    failure leaves the previous good file untouched; the caller's process
    may then die and be relaunched - it resumes from the last save.
    """
    for attempt in range(attempts):
        try:
            save_jsonl(records, path)
            return
        except OSError:
            if attempt == attempts - 1:
                raise
            time.sleep(2.0 * (attempt + 1))

GENERATOR_SYSTEM = (
    "You generate high-quality synthetic tutoring data for school-level "
    "subjects. You always return a single valid JSON object and nothing "
    "else."
)

# Concrete angles cycled per sample so every prompt is unique. An
# identical prompt to a low-diversity endpoint tends to yield an
# identical question, which the final dedup would silently discard.
VARIATION_ANGLES = (
    "use different numerical values and a different concrete setup",
    "use a different real-world context or story setting",
    "focus on a different subtopic within the subject",
    "use different names, places, or entities",
    "ask for a different unknown or a different final quantity",
    "combine two related concepts in one problem",
    "frame it as a word problem with a scenario",
    "use a different unit, scale, or order of magnitude",
    "start from a different given value",
    "center it on a common mistake or misconception",
    "use a table, list, or data context",
    "use a different spatial or geometric arrangement",
)


def parse_generation(payload: str) -> dict | None:
    """Extract the JSON object from a generator response, if present."""
    text = payload.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[len("json") :].lstrip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def load_seeds() -> dict[str, dict]:
    """Map each configured subject to its committed seed prompt."""
    seeds: dict[str, dict] = {}
    for line in SEEDS_FILE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            seed = json.loads(line)
            seeds[seed["subject"]] = seed
    missing = set(DATASET_CONFIG.subjects) - set(seeds)
    if missing:
        raise SystemExit(f"Seed prompts missing for: {sorted(missing)}")
    return seeds


def generate_one(
    client: OpenAI,
    model_name: str,
    seed: dict,
    difficulty: str,
    variation: int,
    used: set[str],
    used_lock: threading.Lock,
    max_retries: int = 2,
) -> dict | None:
    """Generate and validate one sample; return None if it stays unusable.

    ``used`` holds the question texts already produced for this subject.
    A generated question that collides with an existing one is rejected
    and regenerated with an explicit avoid-hint, so the endpoint can
    never silently hand back a duplicate that the final dedup would
    discard.
    """
    angle = VARIATION_ANGLES[variation % len(VARIATION_ANGLES)]
    user_prompt = (
        build_user_prompt(seed["question"], seed["subject"], difficulty)
        + f"\n\nVariation {variation}: {angle}. Keep the concrete details "
        "(numbers, names, contexts) specific and distinct."
    )
    avoid_hint = ""
    for _ in range(max_retries + 1):
        response = client.chat.completions.create(
            model=model_name,
            temperature=0.8,
            max_tokens=2000,
            messages=[
                {"role": "system", "content": GENERATOR_SYSTEM},
                {"role": "user", "content": user_prompt + avoid_hint},
            ],
        )
        raw = parse_generation(response.choices[0].message.content or "")
        if raw is None:
            continue
        record = normalize_record(
            {
                "question": raw.get("question", ""),
                "answer": raw.get("answer", ""),
                "subject": seed["subject"],
                "difficulty": difficulty,
            },
            max_chars=DATASET_CONFIG.max_chars,
        )
        if record is None:
            time.sleep(1.0)
            continue
        key = record["user"].casefold().strip()
        with used_lock:
            if key in used:
                avoid_hint = (
                    "\n\nDo NOT use this question (it was already "
                    f"generated): {record['user']!r} Generate a different "
                    "question instead."
                )
                time.sleep(1.0)
                continue
            used.add(key)
        return record
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the synthetic Q&A training dataset."
    )
    parser.add_argument(
        "--samples-per-subject",
        type=int,
        default=DATASET_CONFIG.samples_per_subject,
        help="Samples per subject (default: %(default)s).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of parallel API calls per subject (default: %(default)s).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = parse_env_local(ENV_FILE)
    if not env.get("OPENAI_API_KEY", ""):
        raise SystemExit(
            "No API key found. Copy .env.example to .env and set "
            f"OPENAI_API_KEY in {ENV_FILE}."
        )
    base_url = env.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model_name = env.get("OPENAI_MODEL") or DATASET_CONFIG.generator_model
    extra_headers = parse_extra_headers(env.get("OPENAI_EXTRA_HEADERS", ""))
    client = OpenAI(
        base_url=base_url,
        api_key=env["OPENAI_API_KEY"],
        http_client=httpx.Client(headers=extra_headers),
    )

    seeds = load_seeds()
    difficulties = DATASET_CONFIG.difficulties
    existing = load_jsonl(RAW_FILE) if RAW_FILE.exists() else []
    done = Counter(record["subject"] for record in existing)
    used: dict[str, set[str]] = {
        subject: {
            record["user"].casefold().strip()
            for record in existing
            if record["subject"] == subject
        }
        for subject in DATASET_CONFIG.subjects
    }
    used_lock = threading.Lock()
    acquire_lock()

    print(f"Generator: {model_name} via {base_url}")
    for subject in DATASET_CONFIG.subjects:
        remaining = args.samples_per_subject - done.get(subject, 0)
        if remaining <= 0:
            print(f"{subject}: {done[subject]:,} already generated, skipping")
            continue
        print(f"{subject}: generating {remaining:,} samples ...")
        new_records: list[dict] = []
        completed = 0
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {
                pool.submit(
                    generate_one,
                    client,
                    model_name,
                    seeds[subject],
                    difficulties[(done[subject] + index) % len(difficulties)],
                    done[subject] + index,
                    used[subject],
                    used_lock,
                ): index
                for index in range(remaining)
            }
            for future in as_completed(futures):
                index = futures[future]
                record = future.result()
                completed += 1
                if record is None:
                    print(f"  sample {index}: unusable after retries, skipping")
                else:
                    new_records.append(record)
                if completed % 10 == 0:
                    print(f"  {completed}/{remaining} generated")
                    save_with_retry(existing + new_records, RAW_FILE)
        existing.extend(new_records)
        # Append per subject so an interrupted run resumes where it stopped.
        save_with_retry(existing, RAW_FILE)

    # De-duplicate (subject, question) pairs, then rewrite the canonical file.
    seen: set[tuple[str, str]] = set()
    unique: list[dict] = []
    for record in existing:
        key = (record["subject"], record["user"].casefold().strip())
        if key not in seen:
            seen.add(key)
            unique.append(record)
    save_with_retry(unique, RAW_FILE)

    totals = Counter(record["subject"] for record in unique)
    print(f"Wrote {len(unique):,} unique samples to {RAW_FILE}")
    for subject in DATASET_CONFIG.subjects:
        print(f"  {subject}: {totals.get(subject, 0):,}")
    print("Next: uv run python projects/qwen-qlora/scripts/prepare_data.py")
    release_lock()


if __name__ == "__main__":
    main()
