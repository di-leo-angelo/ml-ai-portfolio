"""Run configuration: who we evaluate, with what rubric, and how.

A run config is a TOML file (stdlib ``tomllib``, read-only) naming:

* one or more *conditions* - model/backend pairs under test
  (``[conditions.<name>]``), each with its own seed;
* a *judge* - itself a backend, plus the rubric dimensions it scores;
* decoding/generation settings shared by all conditions;
* an optional *gate* for CI (see PLAN.md, CI gate mode).

Backends are discriminated by their ``kind`` field:

* ``local``   - HF transformers on a rented GPU (optionally + LoRA adapter);
* ``api``     - any OpenAI-compatible chat endpoint;
* ``runpod``  - a RunPod serverless endpoint (used for the judge in P1).

TOML is for configs, JSON/JSONL for data: the harness only ever *reads*
TOML and *writes* JSON/JSONL.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .rubric import RUBRIC_VERSION, Rubric, rubric_from_names

VALID_KINDS: tuple[str, ...] = ("local", "api", "runpod")
VALID_DTYPES: tuple[str, ...] = ("bfloat16", "float16", "float32")


class ConfigError(ValueError):
    """Raised for any invalid or incomplete run configuration."""


# --- backends -----------------------------------------------------------


@dataclass(frozen=True)
class LocalBackend:
    """HF transformers on a GPU (optionally with a LoRA adapter)."""

    kind: str = "local"
    model_id: str = ""
    adapter_path: str | None = None
    dtype: str = "bfloat16"
    load_in_4bit: bool = False
    device_map: str = "auto"


@dataclass(frozen=True)
class ApiBackend:
    """Any OpenAI-compatible chat completions endpoint."""

    kind: str = "api"
    base_url: str = "https://api.openai.com/v1"
    api_key_env: str = "OPENAI_API_KEY"
    model: str = ""
    extra_headers: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class RunPodBackend:
    """A RunPod serverless endpoint (see P1's judge_run.py)."""

    kind: str = "runpod"
    endpoint_id: str = ""
    api_key_env: str = "RUNPOD_API_KEY"
    model: str | None = None


Backend = LocalBackend | ApiBackend | RunPodBackend

_BACKEND_FIELDS: dict[str, frozenset[str]] = {
    "local": frozenset(LocalBackend.__dataclass_fields__),
    "api": frozenset(ApiBackend.__dataclass_fields__),
    "runpod": frozenset(RunPodBackend.__dataclass_fields__),
}


def backend_from_dict(raw: dict[str, Any], where: str) -> Backend:
    """Build a typed backend from a TOML table (``kind``-discriminated)."""
    kind = raw.get("kind")
    if kind not in VALID_KINDS:
        raise ConfigError(f"{where}: kind must be one of {list(VALID_KINDS)}")
    unknown = set(raw) - _BACKEND_FIELDS[kind]
    if unknown:
        raise ConfigError(
            f"{where}: unknown field(s) {sorted(unknown)} for {kind} backend"
        )
    if kind == "local":
        dtype = str(raw.get("dtype", "bfloat16"))
        if dtype not in VALID_DTYPES:
            raise ConfigError(f"{where}: dtype must be one of {list(VALID_DTYPES)}")
        backend = LocalBackend(
            model_id=str(raw.get("model_id", "")).strip(),
            adapter_path=raw.get("adapter_path"),
            dtype=dtype,
            load_in_4bit=bool(raw.get("load_in_4bit", False)),
            device_map=str(raw.get("device_map", "auto")),
        )
        if not backend.model_id:
            raise ConfigError(f"{where}: local backend needs a model_id")
        return backend
    if kind == "api":
        backend = ApiBackend(
            base_url=str(raw.get("base_url", "https://api.openai.com/v1")),
            api_key_env=str(raw.get("api_key_env", "OPENAI_API_KEY")),
            model=str(raw.get("model", "")).strip(),
            extra_headers=dict(raw.get("extra_headers", {})),
        )
        if not backend.model:
            raise ConfigError(f"{where}: api backend needs a model")
        return backend
    backend = RunPodBackend(
        endpoint_id=str(raw.get("endpoint_id", "")).strip(),
        api_key_env=str(raw.get("api_key_env", "RUNPOD_API_KEY")),
        model=raw.get("model"),
    )
    if not backend.endpoint_id:
        raise ConfigError(f"{where}: runpod backend needs an endpoint_id")
    return backend


# --- conditions / judge / gate ------------------------------------------


@dataclass(frozen=True)
class Condition:
    """One model/backend under test, with its own seed."""

    name: str
    backend: Backend
    seed: int = 0


@dataclass(frozen=True)
class JudgeConfig:
    """The judge: a backend plus the rubric dimensions it scores."""

    backend: Backend
    dimension_names: tuple[str, ...]
    temperature: float = 0.0
    max_tokens: int = 512

    def rubric(self) -> Rubric:
        return rubric_from_names(self.dimension_names)


@dataclass(frozen=True)
class GateConfig:
    """CI gate: fail if any metric regresses past the tolerance."""

    metrics: tuple[str, ...]
    min_delta: float = 0.0
    ci_level: float = 0.95
    baseline: str | None = None


@dataclass(frozen=True)
class RunConfig:
    """Everything one run needs, loaded from TOML."""

    name: str
    data: str
    conditions: tuple[Condition, ...]
    judge: JudgeConfig
    rubric_version: str
    temperature: float
    max_tokens: int
    output_dir: str = "reports"
    gate: GateConfig | None = None

    def __post_init__(self) -> None:
        names = [condition.name for condition in self.conditions]
        if len(names) != len(set(names)):
            raise ConfigError(f"duplicate condition names: {names}")


def _condition_from_dict(name: str, raw: dict[str, Any]) -> Condition:
    backend_raw = raw.get("backend")
    if not isinstance(backend_raw, dict):
        raise ConfigError(f"[conditions.{name}]: missing backend table")
    backend = backend_from_dict(backend_raw, f"[conditions.{name}]")
    seed = int(raw.get("seed", 0))
    if seed < 0:
        raise ConfigError(f"[conditions.{name}]: seed must be >= 0")
    unknown = set(raw) - {"backend", "seed"}
    if unknown:
        raise ConfigError(f"[conditions.{name}]: unknown field(s) {sorted(unknown)}")
    return Condition(name=name, backend=backend, seed=seed)


def _judge_from_dict(raw: dict[str, Any]) -> JudgeConfig:
    backend_raw = raw.get("backend")
    if not isinstance(backend_raw, dict):
        raise ConfigError("[judge]: missing backend table")
    backend = backend_from_dict(backend_raw, "[judge]")
    names_raw = raw.get("dimensions")
    if not isinstance(names_raw, (list, tuple)) or not names_raw:
        raise ConfigError("[judge]: dimensions must be a non-empty list")
    names = tuple(str(name) for name in names_raw)
    # Fail fast: an unknown dimension is a config error, not a mid-run
    # surprise (rubric validation happens here, not at judge time).
    try:
        rubric_from_names(names)
    except ValueError as exc:
        raise ConfigError(f"[judge]: {exc}") from exc
    temperature = float(raw.get("temperature", 0.0))
    max_tokens = int(raw.get("max_tokens", 512))
    if not 0.0 <= temperature <= 2.0:
        raise ConfigError("[judge]: temperature must be in [0, 2]")
    if max_tokens < 1:
        raise ConfigError("[judge]: max_tokens must be >= 1")
    unknown = set(raw) - {"backend", "dimensions", "temperature", "max_tokens"}
    if unknown:
        raise ConfigError(f"[judge]: unknown field(s) {sorted(unknown)}")
    return JudgeConfig(
        backend=backend,
        dimension_names=names,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def _gate_from_dict(raw: dict[str, Any]) -> GateConfig:
    metrics = raw.get("metrics")
    if not isinstance(metrics, (list, tuple)) or not metrics:
        raise ConfigError("[gate]: metrics must be a non-empty list")
    ci_level = float(raw.get("ci_level", 0.95))
    if not 0.5 < ci_level < 1.0:
        raise ConfigError("[gate]: ci_level must be in (0.5, 1.0)")
    unknown = set(raw) - {"metrics", "min_delta", "ci_level", "baseline"}
    if unknown:
        raise ConfigError(f"[gate]: unknown field(s) {sorted(unknown)}")
    return GateConfig(
        metrics=tuple(str(metric) for metric in metrics),
        min_delta=float(raw.get("min_delta", 0.0)),
        ci_level=ci_level,
        baseline=raw.get("baseline"),
    )


_TOP_FIELDS: frozenset[str] = frozenset(
    {
        "name",
        "data",
        "conditions",
        "judge",
        "temperature",
        "max_tokens",
        "output_dir",
        "gate",
    }
)


def load_run_config(path: str | Path) -> RunConfig:
    """Load, validate, and return a run config from a TOML file."""
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError(f"config not found: {config_path}")
    try:
        with config_path.open("rb") as handle:
            raw = tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"invalid TOML in {config_path}: {exc}") from exc

    unknown = set(raw) - _TOP_FIELDS
    if unknown:
        raise ConfigError(f"unknown top-level field(s) {sorted(unknown)}")
    name = str(raw.get("name", "")).strip()
    if not name:
        raise ConfigError("missing top-level 'name'")
    data = str(raw.get("data", "")).strip()
    if not data:
        raise ConfigError("missing top-level 'data' (path to a tasks JSONL)")

    conditions_raw = raw.get("conditions")
    if not isinstance(conditions_raw, dict) or not conditions_raw:
        raise ConfigError("missing [conditions]: at least one is required")
    conditions = tuple(
        _condition_from_dict(condition_name, spec)
        for condition_name, spec in conditions_raw.items()
    )

    judge_raw = raw.get("judge")
    if not isinstance(judge_raw, dict):
        raise ConfigError("missing [judge] table")
    judge = _judge_from_dict(judge_raw)

    temperature = float(raw.get("temperature", 0.0))
    max_tokens = int(raw.get("max_tokens", 1024))
    if not 0.0 <= temperature <= 2.0:
        raise ConfigError("temperature must be in [0, 2]")
    if max_tokens < 1:
        raise ConfigError("max_tokens must be >= 1")

    gate = _gate_from_dict(raw["gate"]) if isinstance(raw.get("gate"), dict) else None
    return RunConfig(
        name=name,
        data=data,
        conditions=conditions,
        judge=judge,
        rubric_version=RUBRIC_VERSION,
        temperature=temperature,
        max_tokens=max_tokens,
        output_dir=str(raw.get("output_dir", "reports")),
        gate=gate,
    )
