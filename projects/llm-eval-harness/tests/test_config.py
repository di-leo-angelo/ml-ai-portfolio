"""Unit tests for run-config loading and validation (no models, no network)."""

from __future__ import annotations

from pathlib import Path

import pytest
from llm_eval_harness.config import (
    ApiBackend,
    ConfigError,
    LocalBackend,
    RunPodBackend,
    load_run_config,
)

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"

_VALID_TOML = """
name = "unit-run"
data = "configs/example_tasks.jsonl"
temperature = 0.0
max_tokens = 512

[conditions.base]
seed = 1

[conditions.base.backend]
kind = "api"
model = "gpt-4o-mini"
api_key_env = "OPENAI_API_KEY"

[judge]
dimensions = ["correctness", "tone"]
temperature = 0.0
max_tokens = 256

[judge.backend]
kind = "runpod"
endpoint_id = "ep-test"
"""


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "run.toml"
    path.write_text(body, encoding="utf-8")
    return path


def test_load_valid_config(tmp_path: Path) -> None:
    config = load_run_config(_write(tmp_path, _VALID_TOML))
    assert config.name == "unit-run"
    assert config.data == "configs/example_tasks.jsonl"
    assert [c.name for c in config.conditions] == ["base"]
    assert isinstance(config.conditions[0].backend, ApiBackend)
    assert config.conditions[0].seed == 1
    assert config.judge.dimension_names == ("correctness", "tone")
    assert isinstance(config.judge.backend, RunPodBackend)
    assert config.gate is None
    assert config.rubric_version == "1"


def test_backend_kinds_dispatch_local(tmp_path: Path) -> None:
    body = _VALID_TOML.replace(
        'kind = "api"\nmodel = "gpt-4o-mini"\napi_key_env = "OPENAI_API_KEY"',
        'kind = "local"\nmodel_id = "Qwen/Qwen3-0.6B"\nload_in_4bit = true',
    )
    config = load_run_config(_write(tmp_path, body))
    backend = config.conditions[0].backend
    assert isinstance(backend, LocalBackend)
    assert backend.model_id == "Qwen/Qwen3-0.6B"
    assert backend.load_in_4bit is True
    assert backend.dtype == "bfloat16"


def test_gate_parses(tmp_path: Path) -> None:
    body = (
        _VALID_TOML
        + """
[gate]
metrics = ["judge.correctness.mean"]
min_delta = -0.1
ci_level = 0.95
baseline = "baselines/unit.json"
"""
    )
    config = load_run_config(_write(tmp_path, body))
    assert config.gate is not None
    assert config.gate.metrics == ("judge.correctness.mean",)
    assert config.gate.min_delta == -0.1
    assert config.gate.baseline == "baselines/unit.json"


def test_missing_name_raises(tmp_path: Path) -> None:
    body = _VALID_TOML.replace('name = "unit-run"', 'name = ""')
    with pytest.raises(ConfigError, match="name"):
        load_run_config(_write(tmp_path, body))


def test_unknown_backend_kind_raises(tmp_path: Path) -> None:
    body = _VALID_TOML.replace('kind = "api"', 'kind = "cuda"')
    with pytest.raises(ConfigError, match="kind must be one of"):
        load_run_config(_write(tmp_path, body))


def test_unknown_backend_field_raises(tmp_path: Path) -> None:
    body = _VALID_TOML.replace(
        'api_key_env = "OPENAI_API_KEY"',
        'api_key_env = "OPENAI_API_KEY"\nnot_a_field = 1',
    )
    with pytest.raises(ConfigError, match="unknown field"):
        load_run_config(_write(tmp_path, body))


def test_local_without_model_id_raises(tmp_path: Path) -> None:
    body = _VALID_TOML.replace(
        'kind = "api"\nmodel = "gpt-4o-mini"\napi_key_env = "OPENAI_API_KEY"',
        'kind = "local"\ndtype = "bfloat16"',
    )
    with pytest.raises(ConfigError, match="model_id"):
        load_run_config(_write(tmp_path, body))


def test_runpod_without_endpoint_id_raises(tmp_path: Path) -> None:
    body = _VALID_TOML.replace('endpoint_id = "ep-test"', "")
    with pytest.raises(ConfigError, match="endpoint_id"):
        load_run_config(_write(tmp_path, body))


def test_bad_temperature_raises(tmp_path: Path) -> None:
    body = _VALID_TOML.replace(
        "temperature = 0.0\nmax_tokens = 512", "temperature = 5.0\nmax_tokens = 512"
    )
    with pytest.raises(ConfigError, match="temperature"):
        load_run_config(_write(tmp_path, body))


def test_unknown_dimension_raises(tmp_path: Path) -> None:
    body = _VALID_TOML.replace(
        'dimensions = ["correctness", "tone"]', 'dimensions = ["vibes"]'
    )
    with pytest.raises(ConfigError, match="unknown rubric dimensions"):
        load_run_config(_write(tmp_path, body))


def test_invalid_toml_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="invalid TOML"):
        load_run_config(_write(tmp_path, "name = [unclosed"))


def test_example_config_loads() -> None:
    config = load_run_config(CONFIG_DIR / "example_run.toml")
    assert config.name == "qwen-qlora-dogfood"
    assert [c.name for c in config.conditions] == ["base", "lora"]
    assert len(config.judge.dimension_names) == 5
    assert config.gate is None
