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

import os
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .rubric import (
    CUSTOM_RUBRIC_VERSION,
    RUBRIC_VERSION,
    Rubric,
    rubric_from_names,
    rubric_from_specs,
)

VALID_KINDS: tuple[str, ...] = ("local", "api", "runpod", "imported")
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


@dataclass(frozen=True)
class ImportedBackend:
    """Provenance-only condition: outputs already recorded elsewhere (P1).

    Never generates - ``import-generations`` fills ``generations.jsonl``
    from P1's recorded outputs. The condition exists so the judge guard,
    the manifest, and the reports know which model produced the
    imported outputs (e.g. the M2 dogfood of P1's 230 lora + 25 base
    generations, judged without re-running any GPU code).
    """

    kind: str = "imported"
    model: str = ""
    source: str = ""


Backend = LocalBackend | ApiBackend | RunPodBackend | ImportedBackend

_BACKEND_FIELDS: dict[str, frozenset[str]] = {
    "local": frozenset(LocalBackend.__dataclass_fields__),
    "api": frozenset(ApiBackend.__dataclass_fields__),
    "runpod": frozenset(RunPodBackend.__dataclass_fields__),
    "imported": frozenset(ImportedBackend.__dataclass_fields__),
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
    if kind == "imported":
        backend = ImportedBackend(
            model=str(raw.get("model", "")).strip(),
            source=str(raw.get("source", "")).strip(),
        )
        if not backend.model:
            raise ConfigError(f"{where}: imported backend needs a model")
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
    """The judge: a backend plus the rubric dimensions it scores.

    ``dimensions`` accepts either a list of built-in dimension names or
    a table of custom ``name -> 1..5 anchor line`` entries (``M2``); the
    table form is how a dogfood run reproduces a legacy rubric exactly
    (e.g. P1's frozen three: correctness/format/completeness).
    """

    backend: Backend
    dimension_names: tuple[str, ...]
    temperature: float = 0.0
    max_tokens: int = 512
    dimension_specs: dict[str, str] | None = None
    batch_size: int = 5

    def rubric(self) -> Rubric:
        if self.dimension_specs is not None:
            return rubric_from_specs(self.dimension_specs)
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
    judge_warnings: tuple[str, ...] = ()

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
    dimensions_raw = raw.get("dimensions")
    # Two shapes (fail fast either way: an invalid rubric is a config
    # error, not a mid-run surprise). A list selects built-in dimension
    # names; a table supplies custom dimensions, each value a 1..5
    # anchor line quoted verbatim in the judge prompt.
    if isinstance(dimensions_raw, dict):
        if not dimensions_raw:
            raise ConfigError("[judge]: dimensions table must not be empty")
        dimension_specs: dict[str, str] = {}
        for key, value in dimensions_raw.items():
            name = str(key).strip()
            line = str(value).strip()
            if not name or not line:
                raise ConfigError(
                    "[judge]: dimensions need non-empty names and anchor lines"
                )
            dimension_specs[name] = line
        names = tuple(dimension_specs)
        try:
            rubric_from_specs(dimension_specs)
        except ValueError as exc:
            raise ConfigError(f"[judge]: {exc}") from exc
    elif isinstance(dimensions_raw, (list, tuple)) and dimensions_raw:
        names = tuple(str(name) for name in dimensions_raw)
        try:
            rubric_from_names(names)
        except ValueError as exc:
            raise ConfigError(f"[judge]: {exc}") from exc
        dimension_specs = None
    else:
        raise ConfigError(
            "[judge]: dimensions must be a non-empty list of names or a "
            "table of name -> 1..5 anchor line"
        )
    temperature = float(raw.get("temperature", 0.0))
    max_tokens = int(raw.get("max_tokens", 512))
    if not 0.0 <= temperature <= 2.0:
        raise ConfigError("[judge]: temperature must be in [0, 2]")
    if max_tokens < 1:
        raise ConfigError("[judge]: max_tokens must be >= 1")
    batch_size = int(raw.get("batch_size", 5))
    if batch_size < 1:
        raise ConfigError("[judge]: batch_size must be >= 1")
    unknown = set(raw) - {
        "backend",
        "dimensions",
        "temperature",
        "max_tokens",
        "batch_size",
    }
    if unknown:
        raise ConfigError(f"[judge]: unknown field(s) {sorted(unknown)}")
    return JudgeConfig(
        backend=backend,
        dimension_names=names,
        temperature=temperature,
        max_tokens=max_tokens,
        dimension_specs=dimension_specs,
        batch_size=batch_size,
    )


# --- judge != testee guard ------------------------------------------------


def _normalize_model(model: str) -> str:
    return re.sub(r"\s+", "", model).lower()


def model_family(model: str) -> str:
    """Coarse model-family key for the judge!=testee guard.

    The org prefix before ``/`` when present (``Qwen/Qwen3-32B-AWQ`` ->
    ``qwen``), else the leading alphabetic run (``Qwen3.8-27B`` ->
    ``qwen3``), lowercased. Deliberately coarse: it only decides whether
    to WARN about self-family bias, never whether a run passes.
    """
    if "/" in model:
        org = model.split("/", 1)[0]
        if org:
            return org.lower()
    match = re.match(r"[a-z]+", model.lower())
    return match.group(0) if match else model.lower()


def same_model_family(model_a: str, model_b: str) -> bool:
    """True when two model names look like the same model family."""
    family_a, family_b = model_family(model_a), model_family(model_b)
    if len(family_a) < 3 or len(family_b) < 3:
        return False
    return family_a == family_b or family_a.startswith(family_b) or family_b.startswith(
        family_a
    )


def judge_model_string(backend: Backend) -> str | None:
    """The model a backend produces outputs with (None when unknown).

    Mirrors ``build_backend``: a ``runpod`` backend without an explicit
    model sends the endpoint id as the model name.
    """
    if isinstance(backend, LocalBackend):
        if not backend.model_id:
            return None
        return (
            f"{backend.model_id}+LoRA" if backend.adapter_path else backend.model_id
        )
    if isinstance(backend, RunPodBackend):
        return backend.model or backend.endpoint_id or None
    if isinstance(backend, (ApiBackend, ImportedBackend)):
        return backend.model or None
    return None


def judge_model_guard(
    conditions: tuple[Condition, ...], judge: JudgeConfig
) -> tuple[str | None, list[str]]:
    """Judge != testee guard (PLAN.md section 2): a judge must not grade
    its own outputs.

    Returns ``(error, warnings)``: ``error`` (when a condition's model is
    identical to the judge's) is raised as ConfigError by the loader;
    ``warnings`` flag same-family pairs (a Qwen judge on Qwen outputs)
    for the CLI to print loudly.
    """
    judge_model = judge_model_string(judge.backend)
    warnings: list[str] = []
    if not judge_model:
        return None, warnings
    for condition in conditions:
        testee = judge_model_string(condition.backend)
        if not testee:
            continue
        if _normalize_model(testee) == _normalize_model(judge_model):
            return (
                f"[judge]: judge model {judge_model!r} is identical to "
                f"condition {condition.name!r} - a judge must not grade "
                "its own outputs"
            ), warnings
        if same_model_family(testee, judge_model):
            warnings.append(
                f"[judge] WARNING: judge model {judge_model!r} is in the "
                f"same family as condition {condition.name!r} "
                f"({testee!r}); self-family bias is a known judge risk - "
                "interpret those scores accordingly"
            )
    return None, warnings


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
    error, warnings = judge_model_guard(conditions, judge)
    if error:
        raise ConfigError(error)
    return RunConfig(
        name=name,
        data=data,
        conditions=conditions,
        judge=judge,
        rubric_version=(
            CUSTOM_RUBRIC_VERSION
            if judge.dimension_specs is not None
            else RUBRIC_VERSION
        ),
        temperature=temperature,
        max_tokens=max_tokens,
        output_dir=str(raw.get("output_dir", "reports")),
        gate=gate,
        judge_warnings=tuple(warnings),
    )


# --- env files -----------------------------------------------------------


def parse_env_file(path: str | Path) -> dict[str, str]:
    """Parse ``KEY=VALUE`` lines from a dotenv-style file (P1's env.py pattern).

    Blank lines and ``#`` comments are ignored; single- or double-quoted
    values are unquoted. Returns an empty dict when the file is absent.
    """
    file_path = Path(path)
    if not file_path.is_file():
        return {}
    values: dict[str, str] = {}
    for raw_line in file_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip("\"'")
    return values


def apply_env_file(path: str | Path, override: bool = False) -> int:
    """Load a dotenv file into ``os.environ`` (never clobbering existing
    variables unless ``override=True``). Returns the number applied."""
    values = parse_env_file(path)
    applied = 0
    for key, value in values.items():
        if override or key not in os.environ:
            os.environ[key] = value
            applied += 1
    return applied
