"""Generation backends (M1) and the config-to-backend factory.

``build_backend`` maps a validated ``config.Backend`` onto an
implementation:

* ``local``  -> :class:`LocalHFBackend` (HF transformers, GPU pod);
* ``api``    -> :class:`OpenAICompatBackend` (any /chat/completions);
* ``runpod`` -> :class:`OpenAICompatBackend` pointed at a RunPod
  serverless endpoint (``https://api.runpod.ai/v2/<endpoint_id>/openai/v1``).
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Sequence
from typing import Protocol

from ..config import (
    ApiBackend,
    ImportedBackend,
    LocalBackend,
    RunPodBackend,
)
from ..config import Backend as BackendConfig
from .base import (
    GenerationError,
    GenerationJob,
    GenerationRecord,
    GenerationStore,
)
from .local import LocalHFBackend
from .openai_compat import OpenAICompatBackend

__all__ = [
    "Backend",
    "GenerationError",
    "GenerationJob",
    "GenerationRecord",
    "GenerationStore",
    "LocalHFBackend",
    "OpenAICompatBackend",
    "build_backend",
]


class Backend(Protocol):
    """A named generation backend (PLAN.md section 4)."""

    kind: str
    name: str
    model: str

    def generate(self, jobs: Sequence[GenerationJob]) -> Iterator[GenerationRecord]: ...

    def release(self) -> None:
        """Free any in-process model (the runner calls it between conditions)."""
        ...


def build_backend(
    condition_name: str, backend: BackendConfig, concurrency: int = 1
) -> Backend:
    """Instantiate the implementation for a validated backend config."""
    if isinstance(backend, LocalBackend):
        return LocalHFBackend(
            name=condition_name,
            model_id=backend.model_id,
            adapter_path=backend.adapter_path,
            dtype=backend.dtype,
            load_in_4bit=backend.load_in_4bit,
            device_map=backend.device_map,
        )
    if isinstance(backend, ApiBackend):
        return OpenAICompatBackend(
            name=condition_name,
            base_url=backend.base_url,
            model=backend.model,
            api_key=os.environ.get(backend.api_key_env) or None,
            extra_headers=dict(backend.extra_headers),
            concurrency=concurrency,
        )
    if isinstance(backend, RunPodBackend):
        return OpenAICompatBackend(
            name=condition_name,
            base_url=f"https://api.runpod.ai/v2/{backend.endpoint_id}/openai/v1",
            model=backend.model or backend.endpoint_id,
            api_key=os.environ.get(backend.api_key_env) or None,
            concurrency=concurrency,
        )
    if isinstance(backend, ImportedBackend):
        raise GenerationError(
            f"condition {condition_name!r}: 'imported' backends carry "
            "provenance only - import the recorded outputs with "
            "`import-generations` instead of generating them"
        )
    raise TypeError(f"unsupported backend config: {backend!r}")
