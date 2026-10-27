"""Parsing helper for the git-ignored .env credentials file."""

from __future__ import annotations

from pathlib import Path


def parse_env_local(path: Path) -> dict[str, str]:
    """Parse ``KEY=VALUE`` lines from a dotenv-style file.

    Blank lines and ``#`` comments are ignored, and single- or double-quoted
    values are unquoted. Returns an empty dict if the file does not exist.
    """
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip("\"'")
    return values


def parse_extra_headers(raw: str) -> dict[str, str]:
    """Parse ``KEY=VALUE`` pairs separated by semicolons into a headers dict."""
    headers: dict[str, str] = {}
    for pair in raw.split(";"):
        pair = pair.strip()
        if pair and "=" in pair:
            key, _, value = pair.partition("=")
            headers[key.strip()] = value.strip()
    return headers
