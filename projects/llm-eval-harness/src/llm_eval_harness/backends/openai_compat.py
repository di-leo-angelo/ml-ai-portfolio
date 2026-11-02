"""OpenAI-compatible chat completions backend (port of P1's judge_run.py).

Talks to any ``/chat/completions`` endpoint using only stdlib ``urllib``
(P1's choice: fewer moving parts on rented pods).

Retry policy (P1 constants): up to 3 attempts, sleeping 5s x attempt on
network errors and 408/429/5xx; any other 4xx fails fast. Response shapes
are normalized across OpenAI (``choices[].message.content``), RunPod
serverless (``output.content`` / ``output`` string), and bare ``content``
(port of P1's ``extract_content``).
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from .base import GenerationError, GenerationJob, GenerationRecord

#: HTTP statuses worth retrying (P1 convention).
RETRYABLE_STATUS: frozenset[int] = frozenset({408, 429, 500, 502, 503, 504})


def extract_content(resp: Any) -> str:
    """Pull the text out of OpenAI-style or RunPod-serverless responses."""
    if isinstance(resp, dict):
        if "choices" in resp:
            return resp["choices"][0]["message"]["content"]
        if "output" in resp:
            output = resp["output"]
            if isinstance(output, dict):
                return output.get("content", json.dumps(output))
            return str(output)
        if "content" in resp:
            return resp["content"]
    return str(resp)


def extract_usage(resp: Any) -> tuple[int | None, int | None]:
    """(prompt_tokens, completion_tokens) when the endpoint reports usage."""
    if isinstance(resp, dict):
        usage = resp.get("usage")
        if isinstance(usage, dict):
            return usage.get("prompt_tokens"), usage.get("completion_tokens")
    return None, None


class OpenAICompatBackend:
    """Generation backend for any OpenAI-compatible chat endpoint."""

    kind = "api"

    def __init__(
        self,
        name: str,
        base_url: str,
        model: str,
        api_key: str | None = None,
        extra_headers: dict[str, str] | None = None,
        timeout: float = 300.0,
        max_retries: int = 3,
        retry_delay: float = 5.0,
        concurrency: int = 1,
    ) -> None:
        if max_retries < 1:
            raise ValueError("max_retries must be >= 1")
        if concurrency < 1:
            raise ValueError("concurrency must be >= 1")
        self.name = name
        self.model = model
        self.concurrency = concurrency
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.endpoint = base_url.rstrip("/") + "/chat/completions"
        self._headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"
        if extra_headers:
            self._headers.update(extra_headers)

    def _post(self, payload: dict[str, Any]) -> Any:
        """POST with retries/backoff; raise GenerationError after the last."""
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            request = urllib.request.Request(
                self.endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers=self._headers,
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                if exc.code not in RETRYABLE_STATUS:
                    raise GenerationError(
                        f"HTTP {exc.code} from {self.endpoint} "
                        f"(not retried): {exc.reason}"
                    ) from exc
                last_error = exc
            except (urllib.error.URLError, OSError) as exc:
                last_error = exc
            if attempt < self.max_retries:
                time.sleep(self.retry_delay * attempt)
        raise GenerationError(
            f"{self.max_retries} attempts failed for {self.endpoint}: {last_error}"
        ) from last_error

    def _build_payload(self, job: GenerationJob) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": job.messages(),
            "temperature": job.temperature,
            "max_tokens": job.max_tokens,
        }
        if job.seed:
            payload["seed"] = job.seed
        return payload

    def _generate_one(self, job: GenerationJob) -> GenerationRecord:
        started = time.monotonic()
        response = self._post(self._build_payload(job))
        output = extract_content(response)
        tokens_in, tokens_out = extract_usage(response)
        return GenerationRecord(
            task_id=job.task_id,
            condition=self.name,
            seed=job.seed,
            output=output,
            backend="api",
            model=self.model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            wall_s=round(time.monotonic() - started, 3),
        )

    def generate(self, jobs: Sequence[GenerationJob]) -> Iterator[GenerationRecord]:
        if not jobs:
            return
        if self.concurrency > 1 and len(jobs) > 1:
            failures: list[str] = []
            with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
                futures = {pool.submit(self._generate_one, job): job for job in jobs}
                for future in as_completed(futures):
                    job = futures[future]
                    try:
                        yield future.result()
                    except GenerationError as exc:
                        failures.append(f"{job.task_id}: {exc}")
            if failures:
                raise GenerationError(
                    f"{len(failures)} generation(s) failed: {failures[:5]}"
                ) from None
            return
        for job in jobs:
            yield self._generate_one(job)
