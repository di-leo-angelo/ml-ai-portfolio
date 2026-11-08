"""Unit tests for the LLM-as-judge (M2).

Covers the judge prompt (anchors quoted verbatim, P1 truncation), the
parse ladder (strict JSON / fenced / span / regex, invalid rows
dropped), the resume-safe store (incl. torn tail), the pointwise
runner against a fake backend (no network), the position-swapped
pairwise logic, and the judge!=testee guard.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from llm_eval_harness.backends.base import (
    GenerationError,
    GenerationJob,
    GenerationRecord,
)
from llm_eval_harness.config import ApiBackend, JudgeConfig
from llm_eval_harness.graders.judge import (
    ANSWER_TRUNCATE,
    JudgeScoreStore,
    build_judge_prompt,
    build_pairwise_prompt,
    judge_key,
    judge_model_warnings,
    judge_pending,
    pairwise_compare,
    parse_pairwise,
    parse_scores,
)
from llm_eval_harness.rubric import rubric_from_names, rubric_from_specs
from llm_eval_harness.tasks import Task

BATCH = [
    {
        "key": "t1:base",
        "subject": "algebra",
        "question": "Q1",
        "answer": "A1",
        "reference": "R1",
    },
    {
        "key": "t2:lora",
        "subject": "physics",
        "question": "Q2",
        "answer": "A2",
        "reference": None,
    },
]


def _custom_rubric():
    return rubric_from_specs(
        {
            "correctness": "1 = wrong ... 5 = right",
            "format": "1 = rambling ... 5 = crisp",
        }
    )


class FakeJudge:
    """Backend stub: canned replies (or exceptions) per job, no network."""

    kind = "fake"
    name = "judge"
    model = "judge-model"

    def __init__(self, replies: list) -> None:
        self.replies = list(replies)
        self.jobs: list[GenerationJob] = []

    def generate(self, jobs):
        self.jobs.extend(jobs)
        for job in jobs:
            reply = self.replies.pop(0)
            if isinstance(reply, Exception):
                raise reply
            yield GenerationRecord(
                task_id=job.task_id,
                condition="judge",
                seed=0,
                output=reply,
                backend="fake",
                model=self.model,
            )


# --- prompt -------------------------------------------------------------------


def test_prompt_quotes_anchors_verbatim() -> None:
    prompt = build_judge_prompt(_custom_rubric(), BATCH)
    assert "correctness (1-5): 1 = wrong ... 5 = right" in prompt
    assert "format (1-5): 1 = rambling ... 5 = crisp" in prompt
    assert "(subject: algebra)" in prompt
    assert "Reference answer: R1" in prompt
    # a None reference is omitted, not printed as "None"
    assert "Reference answer: None" not in prompt
    assert '"key": "<key>", "correctness": <int>, "format": <int>' in prompt


def test_prompt_built_in_anchors() -> None:
    rubric = rubric_from_names(["correctness", "tone"])
    prompt = build_judge_prompt(rubric, BATCH)
    assert (
        "correctness (1-5): 1 = Wrong or missing core answer; steps "
        "contradict the reference. | 5 = Fully correct, including steps, "
        "units, and edge cases." in prompt
    )
    assert rubric.version == "1"


def test_prompt_truncates_long_answer() -> None:
    long = "x" * ANSWER_TRUNCATE + "MARKER" * 50
    batch = [
        {"key": "t1:base", "question": "Q", "answer": long, "reference": None}
    ]
    prompt = build_judge_prompt(_custom_rubric(), batch)
    assert long[:ANSWER_TRUNCATE] in prompt
    assert "MARKER" not in prompt


def test_pairwise_prompt_lists_dimensions_and_answers() -> None:
    prompt = build_pairwise_prompt(_custom_rubric(), "Q", "R", "AAA", "BBB")
    assert "correctness, format" in prompt
    assert "Reference answer: R" in prompt
    assert "Answer A: AAA" in prompt
    assert "Answer B: BBB" in prompt
    no_ref = build_pairwise_prompt(_custom_rubric(), "Q", None, "A", "B")
    assert "Reference answer" not in no_ref


# --- parse ladder ---------------------------------------------------------------


def _json_rows() -> str:
    return json.dumps(
        [
            {"key": "t1:base", "correctness": 4, "format": 5},
            {"key": "t2:lora", "correctness": 2, "format": 3},
        ]
    )


def test_parse_strict_json() -> None:
    rows = parse_scores(_json_rows(), BATCH, _custom_rubric())
    assert [r["key"] for r in rows] == ["t1:base", "t2:lora"]
    assert rows[0]["correctness"] == 4
    assert rows[1]["format"] == 3
    assert all(r["parse"] == "json" for r in rows)


def test_parse_fenced_json() -> None:
    text = "Sure!\n```json\n" + _json_rows() + "\n```"
    rows = parse_scores(text, BATCH, _custom_rubric())
    assert len(rows) == 2
    assert all(r["parse"] == "fence" for r in rows)


def test_parse_json_embedded_in_prose() -> None:
    text = "Here you go: " + _json_rows() + " hope that helps"
    rows = parse_scores(text, BATCH, _custom_rubric())
    assert len(rows) == 2
    assert all(r["parse"] == "span" for r in rows)


def test_parse_regex_fallback() -> None:
    text = (
        "t1:base: correctness: 4, format: 5\n"
        "t2:lora: correctness: 2, format: 3"
    )
    rows = parse_scores(text, BATCH, _custom_rubric())
    assert {
        r["key"]: (r["correctness"], r["format"], r["parse"]) for r in rows
    } == {"t1:base": (4, 5, "regex"), "t2:lora": (2, 3, "regex")}


def test_parse_positional_when_keys_missing() -> None:
    text = json.dumps(
        [{"correctness": 4, "format": 5}, {"correctness": 2, "format": 3}]
    )
    rows = parse_scores(text, BATCH, _custom_rubric())
    assert [r["key"] for r in rows] == ["t1:base", "t2:lora"]


def test_parse_invalid_rows_dropped() -> None:
    text = json.dumps(
        [
            {"key": "t1:base", "correctness": 4, "format": 9},  # out of range
            {"key": "t2:lora", "correctness": "4", "format": 3},  # string
            {"key": "t9:base", "correctness": 1, "format": 1},  # unknown key
        ]
    )
    assert parse_scores(text, BATCH, _custom_rubric()) == []


def test_parse_unparseable_returns_empty() -> None:
    assert parse_scores("no scores at all", BATCH, _custom_rubric()) == []


# --- resume-safe store -----------------------------------------------------------


def test_store_resume_and_torn_tail(tmp_path: Path) -> None:
    path = tmp_path / "judge_scores.jsonl"
    path.write_text(
        json.dumps({"key": "t1:base", "correctness": 4}) + "\n"
        + '{"key": "t2:lo',
        encoding="utf-8",
    )
    store = JudgeScoreStore(path)
    assert store.existing_keys() == {"t1:base"}  # torn tail dropped
    store.append_row({"key": "t2:lora", "correctness": 3})
    assert JudgeScoreStore(path).existing_keys() == {"t1:base", "t2:lora"}


def test_store_malformed_middle_line_raises(tmp_path: Path) -> None:
    path = tmp_path / "judge_scores.jsonl"
    path.write_text(
        json.dumps({"key": "t1:base"}) + "\n"
        + "not json\n"
        + json.dumps({"key": "t2:base"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="malformed"):
        JudgeScoreStore(path).existing_keys()


# --- pointwise runner (fake backend, no network) ---------------------------------


def _judge_config() -> JudgeConfig:
    backend = ApiBackend(
        base_url="http://localhost:1/v1",
        model="judge-model",
        api_key_env="UNSET_TEST_KEY",
    )
    return JudgeConfig(
        backend=backend,
        dimension_names=("correctness", "format"),
        temperature=0.0,
        max_tokens=1024,
        dimension_specs={
            "correctness": "1 = wrong ... 5 = right",
            "format": "1 = rambling ... 5 = crisp",
        },
        batch_size=1,
    )


def _generations() -> list[dict]:
    return [
        {
            "task_id": "t1",
            "condition": "base",
            "seed": 0,
            "output": "A1",
            "model": "testee-1",
        },
        {
            "task_id": "t2",
            "condition": "base",
            "seed": 0,
            "output": "A2",
            "model": "testee-1",
        },
        {
            "task_id": "t3",
            "condition": "lora",
            "seed": 0,
            "output": "A3",
            "model": "testee-2",
        },
    ]


def _tasks() -> dict[str, Task]:
    return {
        "t1": Task(prompt="Q1", reference="R1", id="t1"),
        "t2": Task(prompt="Q2", reference=None, id="t2"),
        "t3": Task(prompt="Q3", reference="R3", id="t3"),
    }


def _reply(scores: dict[str, tuple[int, int]]) -> str:
    return json.dumps(
        [
            {"key": key, "correctness": c, "format": f}
            for key, (c, f) in scores.items()
        ]
    )


def test_judge_pending_writes_rows(tmp_path: Path) -> None:
    backend = FakeJudge(
        [
            _reply({"t1:base": (4, 5)}),
            _reply({"t2:base": (2, 3)}),
            _reply({"t3:lora": (5, 4)}),
        ]
    )
    code = judge_pending(
        backend, _judge_config(), _generations(), _tasks(), tmp_path
    )
    assert code == 0
    lines = (tmp_path / "judge_scores.jsonl").read_text().splitlines()
    rows = [json.loads(line) for line in lines]
    assert [r["key"] for r in rows] == ["t1:base", "t2:base", "t3:lora"]
    assert rows[0]["judge_model"] == "judge-model"
    assert rows[0]["model"] == "testee-1"
    assert rows[0]["rubric_version"] == "custom"
    assert rows[0]["parse"] == "json"
    assert len(backend.jobs) == 3
    assert backend.jobs[0].task_id == "judge_batch_001"
    assert backend.jobs[0].max_tokens == 1024


def test_judge_pending_resumes_without_duplicates(tmp_path: Path) -> None:
    judge_pending(
        FakeJudge([_reply({"t1:base": (4, 5)})]),
        _judge_config(),
        _generations()[:1],
        _tasks(),
        tmp_path,
    )
    second = FakeJudge([])
    code = judge_pending(
        second, _judge_config(), _generations()[:1], _tasks(), tmp_path
    )
    assert code == 0
    assert second.jobs == []  # already judged: no backend calls
    lines = (tmp_path / "judge_scores.jsonl").read_text().splitlines()
    assert len(lines) == 1  # no duplicate keys


def test_judge_pending_failed_batch_counts(tmp_path: Path) -> None:
    backend = FakeJudge(
        [
            GenerationError("boom"),
            _reply({"t2:base": (1, 1)}),
            _reply({"t3:lora": (5, 4)}),
        ]
    )
    code = judge_pending(
        backend, _judge_config(), _generations(), _tasks(), tmp_path
    )
    assert code == 1  # one batch failed after retries
    rows = [
        json.loads(line)
        for line in (tmp_path / "judge_scores.jsonl").read_text().splitlines()
    ]
    assert [r["key"] for r in rows] == ["t2:base", "t3:lora"]


def test_judge_key() -> None:
    assert judge_key("p1_0001", "lora") == "p1_0001:lora"


# --- pairwise (position-swapped) ---------------------------------------------------


def test_pairwise_agrees_across_swap() -> None:
    # forward says a; backward (B,A) says b -> maps back to a -> preference a
    backend = FakeJudge(
        [
            '{"winner": "a", "confidence": 4, "rationale": "A clearer"}',
            '{"winner": "b", "confidence": 3, "rationale": "B clearer"}',
        ]
    )
    result = pairwise_compare(backend, _custom_rubric(), "Q", "R", "A", "B")
    assert result["preference"] == "a"


def test_pairwise_prompts_swap_positions() -> None:
    backend = FakeJudge(['{"winner": "a"}', '{"winner": "b"}'])
    result = pairwise_compare(backend, _custom_rubric(), "Q", None, "AAA", "BBB")
    assert "Answer A: AAA" in backend.jobs[0].prompt
    assert "Answer A: BBB" in backend.jobs[1].prompt
    assert result["preference"] == "a"


def test_pairwise_disagreement() -> None:
    # forward a; backward a -> maps to b -> position-bias tell
    backend = FakeJudge(['{"winner": "a"}', '{"winner": "a"}'])
    result = pairwise_compare(backend, _custom_rubric(), "Q", None, "A", "B")
    assert result["preference"] == "disagreement"


def test_pairwise_double_tie() -> None:
    backend = FakeJudge(['{"winner": "tie"}', '{"winner": "tie"}'])
    result = pairwise_compare(backend, _custom_rubric(), "Q", None, "A", "B")
    assert result["preference"] == "tie"


def test_parse_pairwise_shapes() -> None:
    assert parse_pairwise('{"winner": "b", "confidence": 5}')["winner"] == "b"
    assert parse_pairwise('```json\n{"winner": "tie"}\n```')["winner"] == "tie"
    assert parse_pairwise("I pick winner: a because ...")["winner"] == "a"
    assert parse_pairwise("no signal") is None


# --- judge != testee guard (data-driven) ----------------------------------------------


def test_data_guard_identical_raises() -> None:
    with pytest.raises(ValueError, match="identical"):
        judge_model_warnings("Qwen3.8-27B", {"qwen3.8-27b "})


def test_data_guard_same_family_warns() -> None:
    warnings = judge_model_warnings("Qwen/Qwen3-32B-AWQ", {"Qwen3.8-27B"})
    assert any("same family" in warning for warning in warnings)


def test_data_guard_different_family_silent() -> None:
    assert judge_model_warnings("Qwen/Qwen3-32B-AWQ", {"gpt-4o-mini"}) == []