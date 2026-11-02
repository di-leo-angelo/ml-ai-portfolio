"""M1 backend + runner tests (CPU-only; loopback HTTP, no real network)."""

from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from llm_eval_harness.backends.base import (
    GenerationError,
    GenerationJob,
    GenerationRecord,
    GenerationStore,
)
from llm_eval_harness.backends.openai_compat import OpenAICompatBackend, extract_content
from llm_eval_harness.config import apply_env_file, parse_env_file
from llm_eval_harness.runner import load_run_tasks, run_generation


class _Controller:
    """Per-test response script + request capture for the fake server."""

    def __init__(self) -> None:
        self.script: list[tuple] = []
        self.requests: list[dict] = []
        self.default = ("ok", {"choices": [{"message": {"content": "ok"}}]})

    def next_action(self) -> tuple:
        return self.script.pop(0) if self.script else self.default


class _Handler(BaseHTTPRequestHandler):
    controller: _Controller = _Controller()  # replaced by the fixture

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        self.controller.requests.append(body)
        kind, payload = self.controller.next_action()
        if kind == "error":
            self.send_response(payload)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode("utf-8"))

    def log_message(self, *args: object) -> None:
        pass


@pytest.fixture()
def fake_api():
    controller = _Controller()
    _Handler.controller = controller
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield controller, f"http://127.0.0.1:{server.server_address[1]}/v1"
    server.shutdown()
    server.server_close()


def _job(task_id: str = "t1") -> GenerationJob:
    return GenerationJob(
        task_id=task_id, prompt="Say hi", seed=0, temperature=0.0, max_tokens=16
    )


def test_happy_path_openai_shape(fake_api) -> None:
    controller, base_url = fake_api
    controller.script = [
        (
            "ok",
            {
                "choices": [{"message": {"content": "hi there"}}],
                "usage": {"prompt_tokens": 7, "completion_tokens": 2},
            },
        )
    ]
    backend = OpenAICompatBackend(
        name="cond", base_url=base_url, model="fake-model", retry_delay=0.01
    )
    (record,) = list(backend.generate([_job()]))
    assert record.output == "hi there"
    assert (record.condition, record.backend, record.model) == (
        "cond",
        "api",
        "fake-model",
    )
    assert (record.tokens_in, record.tokens_out) == (7, 2)
    request = controller.requests[0]
    assert request["model"] == "fake-model"
    assert request["messages"][-1] == {"role": "user", "content": "Say hi"}


def test_retry_on_5xx_then_success(fake_api) -> None:
    controller, base_url = fake_api
    controller.script = [
        ("error", 500),
        ("error", 429),
        ("ok", {"output": {"content": "back"}}),
    ]
    backend = OpenAICompatBackend(
        name="c", base_url=base_url, model="m", retry_delay=0.01
    )
    (record,) = list(backend.generate([_job()]))
    assert record.output == "back"
    assert len(controller.requests) == 3


def test_gives_up_after_max_retries(fake_api) -> None:
    controller, base_url = fake_api
    controller.script = [("error", 502), ("error", 502), ("error", 502)]
    backend = OpenAICompatBackend(
        name="c", base_url=base_url, model="m", retry_delay=0.01
    )
    with pytest.raises(GenerationError, match="3 attempts"):
        list(backend.generate([_job()]))
    assert len(controller.requests) == 3


def test_4xx_fails_fast(fake_api) -> None:
    controller, base_url = fake_api
    controller.script = [("error", 401)]
    backend = OpenAICompatBackend(
        name="c", base_url=base_url, model="m", retry_delay=0.01
    )
    with pytest.raises(GenerationError, match="401"):
        list(backend.generate([_job()]))
    assert len(controller.requests) == 1


def test_concurrent_generation(fake_api) -> None:
    controller, base_url = fake_api
    backend = OpenAICompatBackend(
        name="c", base_url=base_url, model="m", concurrency=2, retry_delay=0.01
    )
    jobs = [
        GenerationJob(
            task_id=f"t{i}", prompt="hi", seed=0, temperature=0.0, max_tokens=8
        )
        for i in range(4)
    ]
    records = list(backend.generate(jobs))
    assert {record.task_id for record in records} == {"t0", "t1", "t2", "t3"}
    assert len(controller.requests) == 4


def test_response_shape_normalization() -> None:
    assert extract_content({"choices": [{"message": {"content": "a"}}]}) == "a"
    assert extract_content({"output": {"content": "b"}}) == "b"
    assert extract_content({"output": "c"}) == "c"
    assert extract_content({"content": "d"}) == "d"
    assert extract_content("raw") == "raw"


def _write_run_toml(path: Path, base_url: str, data: Path, out_dir: Path) -> None:
    lines = [
        'name = "resumetest"',
        f'data = "{data.as_posix()}"',
        "temperature = 0.0",
        "max_tokens = 16",
        f'output_dir = "{out_dir.as_posix()}"',
        "",
        "[conditions.api_smoke]",
        "seed = 0",
        "",
        "[conditions.api_smoke.backend]",
        'kind = "api"',
        f'base_url = "{base_url}"',
        'model = "fake-model"',
        'api_key_env = "FAKE_HARNESS_KEY"',
        "",
        "[judge]",
        'dimensions = ["correctness"]',
        "",
        "[judge.backend]",
        'kind = "api"',
        f'base_url = "{base_url}"',
        'model = "fake-model"',
        'api_key_env = "FAKE_HARNESS_KEY"',
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_resume_skips_existing_and_writes_manifest(tmp_path, fake_api) -> None:
    controller, base_url = fake_api
    data = tmp_path / "tasks.jsonl"
    data.write_text(
        "\n".join(json.dumps({"id": f"t{i:04d}", "prompt": f"Q{i}"}) for i in range(3))
        + "\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"
    config = tmp_path / "run.toml"
    _write_run_toml(config, base_url, data, out_dir)
    # Simulate a previous killed run: 2 of 3 records already written.
    out_dir.mkdir(parents=True)
    seeded = [
        GenerationRecord(
            task_id=f"t{i:04d}",
            condition="api_smoke",
            seed=0,
            output="seeded",
            backend="api",
            model="fake-model",
        )
        for i in range(2)
    ]
    with (out_dir / "generations.jsonl").open("w", encoding="utf-8") as handle:
        for record in seeded:
            handle.write(json.dumps(record.to_dict()) + "\n")

    assert run_generation(config) == 0

    lines = (out_dir / "generations.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    assert [json.loads(line)["task_id"] for line in lines[2:]] == ["t0002"]
    assert len(controller.requests) == 1  # only the missing task was generated

    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["data"]["tasks"] == 3
    assert len(manifest["data"]["sha256"]) == 64
    assert manifest["conditions"]["api_smoke"]["backend"]["model"] == "fake-model"
    assert manifest["conditions"]["api_smoke"]["seed"] == 0
    assert manifest["rubric_version"]


def test_second_run_is_noop(tmp_path, fake_api) -> None:
    controller, base_url = fake_api
    data = tmp_path / "tasks.jsonl"
    data.write_text(json.dumps({"id": "a", "prompt": "P"}) + "\n", encoding="utf-8")
    out_dir = tmp_path / "out"
    config = tmp_path / "run.toml"
    _write_run_toml(config, base_url, data, out_dir)
    assert run_generation(config) == 0
    controller.requests.clear()
    assert run_generation(config) == 0
    assert len(controller.requests) == 0  # everything already present


def test_store_drops_torn_tail(tmp_path) -> None:
    path = tmp_path / "generations.jsonl"
    good = GenerationRecord(
        task_id="t1", condition="c", seed=0, output="ok", backend="api", model="m"
    )
    path.write_text(
        json.dumps(good.to_dict()) + "\n" + '{"task_id": "t2", "condi',
        encoding="utf-8",
    )
    assert GenerationStore(path).existing_keys() == {("t1", "c", 0)}


def test_store_rejects_corrupt_middle_line(tmp_path) -> None:
    path = tmp_path / "generations.jsonl"
    good = GenerationRecord(
        task_id="t1", condition="c", seed=0, output="ok", backend="api", model="m"
    )
    line = json.dumps(good.to_dict())
    path.write_text(line + "\n" + "{broken" + "\n" + line + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed"):
        GenerationStore(path).existing_keys()


def test_env_file_parse_and_apply(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("HARNESS_TEST_KEY", raising=False)
    monkeypatch.delenv("HARNESS_TEST_QUOTED", raising=False)
    env = tmp_path / ".env"
    env.write_text(
        '# comment\nHARNESS_TEST_KEY=secret\nHARNESS_TEST_QUOTED="v2"\n',
        encoding="utf-8",
    )
    assert parse_env_file(env) == {
        "HARNESS_TEST_KEY": "secret",
        "HARNESS_TEST_QUOTED": "v2",
    }
    assert apply_env_file(env) == 2
    assert os.environ["HARNESS_TEST_KEY"] == "secret"
    assert os.environ["HARNESS_TEST_QUOTED"] == "v2"
    monkeypatch.setenv("HARNESS_TEST_KEY", "original")
    assert apply_env_file(env) == 0  # both keys already set; nothing clobbered
    assert os.environ["HARNESS_TEST_KEY"] == "original"
    assert os.environ["HARNESS_TEST_QUOTED"] == "v2"


def test_load_run_tasks_new_schema(tmp_path) -> None:
    path = tmp_path / "tasks.jsonl"
    path.write_text(
        json.dumps({"id": "a", "prompt": "P1"})
        + "\n"
        + json.dumps({"prompt": "P2", "system": "s"})
        + "\n",
        encoding="utf-8",
    )
    tasks = load_run_tasks(path)
    assert [task.id for task in tasks] == ["a", "task_0002"]


def test_load_run_tasks_p1_shape(tmp_path) -> None:
    path = tmp_path / "p1.jsonl"
    path.write_text(
        json.dumps(
            {
                "system": "tutor",
                "user": "Solve x",
                "assistant": "x = 1",
                "subject": "algebra",
                "difficulty": "beginner",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (task,) = load_run_tasks(path)
    assert task.id == "p1_0001"
    assert task.source == "qwen-qlora:chat"
    assert task.reference == "x = 1"


def test_load_run_tasks_garbage(tmp_path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"foo": "bar"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unrecognised"):
        load_run_tasks(path)
