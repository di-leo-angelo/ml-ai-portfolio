"""LLM-as-judge (M2): pointwise rubric scoring + position-swapped pairwise.

Port of P1's ``scripts/judge_run.py`` - batches of N samples per judge
call, strict-JSON responses, resume-safe ``judge_scores.jsonl``, and a
per-key regex fallback - generalized to the harness rubric: dimensions
and anchors come from the config (the built-in five, or a custom spec
such as P1's frozen three), and every scored record carries how it was
parsed (``json`` / ``fence`` / ``span`` / ``regex``) so parse coverage
is explicit and counted, never silent (P1's 10/255 silently-dropped
samples are the motivating incident, PLAN.md section 13).

The judge is driven through the normal backend interface (one
``GenerationJob`` per batch), so retries, the 200+error-body guard, and
response-shape normalization from M1 all apply.
"""

from __future__ import annotations

import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..backends import Backend
from ..backends.base import GenerationError, GenerationJob
from ..config import JudgeConfig, _normalize_model, same_model_family
from ..rubric import Rubric, rubric_prompt_lines

#: P1's prompt truncation constants.
ANSWER_TRUNCATE = 2000
REFERENCE_TRUNCATE = 1200

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def judge_key(task_id: str, condition: str) -> str:
    """Resume key for one judged (task, condition) pair."""
    return f"{task_id}:{condition}"


# --- prompt ---------------------------------------------------------------


def build_judge_prompt(rubric: Rubric, batch: list[dict[str, Any]]) -> str:
    """Judge prompt for one batch of samples.

    ``batch`` items: ``{"key", "question", "answer", "reference",
    "subject"}`` - ``reference`` may be None (the line is then omitted),
    ``subject`` is optional. Anchors are quoted verbatim from the rubric.
    """
    fields = ", ".join(f'"{name}": <int>' for name in rubric.names())
    parts: list[str] = [
        "You are grading tutor-style answers to school-level questions.",
        "For each numbered sample below (question, model answer, reference "
        "answer) give integer scores 1-5:",
        *rubric_prompt_lines(rubric),
        "Respond with ONLY a JSON array of objects, one per sample, in the "
        f'same order as the samples: [{{"key": "<key>", {fields}}}]',
    ]
    for sample in batch:
        subject = f" (subject: {sample['subject']})" if sample.get("subject") else ""
        parts.append(f"### Sample {sample['key']}{subject}")
        parts.append(f"Question: {sample['question']}")
        parts.append(f"Model answer: {str(sample['answer'])[:ANSWER_TRUNCATE]}")
        if sample.get("reference"):
            parts.append(
                f"Reference answer: {str(sample['reference'])[:REFERENCE_TRUNCATE]}"
            )
        parts.append("")
    return "\n".join(parts)


# --- parse ladder ----------------------------------------------------------


def _as_int(value: Any) -> int | None:
    """Strict int coercion: bools rejected, integral floats accepted."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _rows_from_list(
    raw: Any, batch: list[dict[str, Any]], rubric: Rubric, parse: str
) -> list[dict[str, Any]]:
    """Rows from a JSON list: key-matched, else positional.

    Entries with a missing/wrong-type/out-of-range score are dropped
    (and counted as parse failures by the caller), never guessed.
    """
    if not isinstance(raw, list):
        return []
    has_keys = all(isinstance(item, dict) and "key" in item for item in raw)
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        if has_keys:
            key = item.get("key")
            sample = next((s for s in batch if s["key"] == key), None)
            if sample is None:
                continue
            key = sample["key"]
        elif index < len(batch):
            key = batch[index]["key"]
        else:
            continue
        row: dict[str, Any] = {"key": key}
        valid = True
        for name in rubric.names():
            score = _as_int(item.get(name))
            if score is None or not 1 <= score <= 5:
                valid = False
                break
            row[name] = score
        if valid:
            row["parse"] = parse
            rows.append(row)
    return rows


def _regex_rows(
    text: str, batch: list[dict[str, Any]], rubric: Rubric
) -> list[dict[str, Any]]:
    """P1's per-key regex fallback, generalized over the dimensions."""
    rows: list[dict[str, Any]] = []
    for sample in batch:
        parts = [re.escape(str(sample["key"])), r".{0,400}?"]
        for index, name in enumerate(rubric.names()):
            separator = "" if index == 0 else r".{0,200}?"
            parts.append(f"{separator}{re.escape(name)}[\"']?\\s*[:=]\\s*(\\d+)")
        match = re.search("".join(parts), text, re.DOTALL)
        if match:
            row = {"key": sample["key"]}
            row.update(
                dict(zip(rubric.names(), (int(g) for g in match.groups()), strict=True))
            )
            row["parse"] = "regex"
            rows.append(row)
    return rows


def parse_scores(
    text: str, batch: list[dict[str, Any]], rubric: Rubric
) -> list[dict[str, Any]]:
    """Parse a judge reply into per-sample score rows (the parse ladder).

    Ladder: strict JSON -> fenced JSON -> first ``[...]`` span ->
    per-sample regex fallback. Every row records how it was parsed;
    samples still missing afterwards are parse failures (coverage).
    A stage that *parses* JSON commits: if the reply is valid JSON its
    rows (even a partial/invalid set) are the judge's answer - invalid
    rows are counted as parse failures, never re-guessed by regex.
    """
    try:
        raw = json.loads(text)
        if isinstance(raw, dict):
            raw = next(
                (value for value in raw.values() if isinstance(value, list)),
                None,
            )
        return _rows_from_list(raw, batch, rubric, "json")
    except (json.JSONDecodeError, ValueError):
        pass
    fence = _FENCE_RE.search(text)
    if fence:
        try:
            return _rows_from_list(json.loads(fence.group(1)), batch, rubric, "fence")
        except (json.JSONDecodeError, ValueError):
            pass
    span_start, span_end = text.find("["), text.rfind("]")
    if span_start != -1 and span_end > span_start:
        try:
            return _rows_from_list(
                json.loads(text[span_start : span_end + 1]), batch, rubric, "span"
            )
        except (json.JSONDecodeError, ValueError):
            pass
    return _regex_rows(text, batch, rubric)


# --- resume-safe store ------------------------------------------------------


class JudgeScoreStore:
    """Append-only ``judge_scores.jsonl``, resume-safe by row key.

    Same torn-tail semantics as GenerationStore: a malformed final line
    is a crash mid-write and is dropped; any other malformed line fails
    loudly.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def existing_keys(self) -> set[str]:
        if not self.path.is_file():
            return set()
        lines = [
            line
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        keys: set[str] = set()
        for index, line in enumerate(lines):
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                if index == len(lines) - 1:
                    continue  # torn tail: drop, keep resuming
                raise ValueError(f"malformed line {index + 1} in {self.path}") from None
            if "key" not in raw:
                raise ValueError(f"missing 'key' on line {index + 1} in {self.path}")
            keys.add(str(raw["key"]))
        return keys

    def append_row(self, row: dict[str, Any]) -> None:
        self._drop_torn_tail()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _drop_torn_tail(self) -> None:
        """Self-heal a torn final line before appending.

        A crash mid-write leaves an invalid final line; dropping it here
        (not just at read time) keeps the file well-formed for every
        later reader - otherwise the torn line becomes a *middle* line
        on the next resume and the loud malformed-line check trips.
        """
        if not self.path.is_file():
            return
        lines = self.path.read_text(encoding="utf-8").splitlines()
        if not lines:
            return
        try:
            json.loads(lines[-1])
        except json.JSONDecodeError:
            keep = lines[:-1]
            self.path.write_text(
                "".join(line + "\n" for line in keep), encoding="utf-8"
            )


# --- pointwise judging ------------------------------------------------------


def judge_pending(
    backend: Backend,
    judge: JudgeConfig,
    generations: list[dict[str, Any]],
    tasks_by_id: dict[str, Any],
    out_dir: str | Path,
    limit: int = 0,
) -> int:
    """Judge all not-yet-judged generations; write ``judge_scores.jsonl``.

    Returns 0 when every pending sample was scored (or nothing was
    pending), 1 when any batch failed after the backend's own retries.
    Re-running resumes: keys already in the store are skipped.
    """
    rubric = judge.rubric()
    store = JudgeScoreStore(Path(out_dir) / "judge_scores.jsonl")
    done = store.existing_keys()
    todo = [
        record
        for record in generations
        if judge_key(str(record["task_id"]), str(record["condition"])) not in done
    ]
    if limit > 0:
        todo = todo[:limit]
    if not todo:
        print(f"nothing to judge ({len(done)} already in judge_scores.jsonl)")
        return 0
    batches = [
        todo[i : i + judge.batch_size] for i in range(0, len(todo), judge.batch_size)
    ]
    print(
        f"judging {len(todo)} samples in {len(batches)} batches of "
        f"{judge.batch_size} (skipping {len(generations) - len(todo)} "
        "already judged)"
    )
    judged = parse_failed = batch_failed = 0
    for batch_index, batch in enumerate(batches, 1):
        samples: list[dict[str, Any]] = []
        for record in batch:
            task = tasks_by_id[str(record["task_id"])]
            samples.append(
                {
                    "key": judge_key(str(record["task_id"]), str(record["condition"])),
                    "subject": task.subject,
                    "question": task.prompt,
                    "answer": str(record.get("output", "")),
                    "reference": task.reference,
                }
            )
        job = GenerationJob(
            task_id=f"judge_batch_{batch_index:03d}",
            prompt=build_judge_prompt(rubric, samples),
            seed=0,
            temperature=judge.temperature,
            max_tokens=judge.max_tokens,
        )
        try:
            record = next(backend.generate([job]))
            rows = parse_scores(record.output, samples, rubric)
        except (GenerationError, StopIteration, ValueError) as exc:
            batch_failed += 1
            print(
                f"batch {batch_index}/{len(batches)} FAILED: {exc} "
                f"(keys: {[s['key'] for s in samples]})",
                flush=True,
            )
            continue
        by_key = {judge_key(str(r["task_id"]), str(r["condition"])): r for r in batch}
        for row in rows:
            source = by_key[str(row["key"])]
            row.update(
                {
                    "task_id": source["task_id"],
                    "condition": source["condition"],
                    "model": source.get("model"),
                    "judge_model": backend.model,
                    "rubric_version": rubric.version,
                    "ts": datetime.now(UTC).isoformat(),
                }
            )
            store.append_row(row)
        judged += len(rows)
        parse_failed += len(samples) - len(rows)
        print(
            f"[{batch_index}/{len(batches)}] judged={judged} "
            f"parse_failed={parse_failed}",
            flush=True,
        )
    coverage = judged / len(todo) if todo else 0.0
    print(
        f"done: {judged}/{len(todo)} judged (parse coverage "
        f"{100 * coverage:.1f}%), {parse_failed} parse failures, "
        f"{batch_failed} failed batches. "
        f"Output: {Path(out_dir) / 'judge_scores.jsonl'}"
    )
    return 1 if batch_failed else 0


# --- pairwise (position-swapped) --------------------------------------------


def build_pairwise_prompt(
    rubric: Rubric,
    question: str,
    reference: str | None,
    answer_a: str,
    answer_b: str,
) -> str:
    """Pairwise judge prompt; answers truncated like the pointwise one."""
    parts = [
        "You are comparing two tutor-style answers to the same question.",
        f"Judge which answer is better on: {', '.join(rubric.names())}.",
        f"Question: {question}",
    ]
    if reference:
        parts.append(f"Reference answer: {reference[:REFERENCE_TRUNCATE]}")
    parts += [
        f"Answer A: {answer_a[:ANSWER_TRUNCATE]}",
        f"Answer B: {answer_b[:ANSWER_TRUNCATE]}",
        'Respond with ONLY a JSON object: {"winner": "a" | "b" | "tie", '
        '"confidence": <int 1-5>, "rationale": "<one sentence>"}',
    ]
    return "\n".join(parts)


def parse_pairwise(text: str) -> dict[str, Any] | None:
    """Winner from a pairwise reply: strict JSON, fenced JSON, then regex."""
    candidates = [text]
    fence = _FENCE_RE.search(text)
    if fence:
        candidates.append(fence.group(1))
    for candidate in candidates:
        try:
            raw = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(raw, dict):
            continue
        winner = str(raw.get("winner", "")).lower().strip()
        if winner in {"a", "b", "tie"}:
            return {
                "winner": winner,
                "confidence": _as_int(raw.get("confidence")),
                "rationale": str(raw.get("rationale", "")),
                "parse": "json",
            }
    match = re.search(r"winner[\"']?\s*[:=]\s*[\"']?([abAB]|tie)", text)
    if match:
        return {
            "winner": match.group(1).lower(),
            "confidence": None,
            "rationale": "",
            "parse": "regex",
        }
    return None


def _pairwise_call(
    backend: Backend,
    rubric: Rubric,
    question: str,
    reference: str | None,
    answer_a: str,
    answer_b: str,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    job = GenerationJob(
        task_id="pairwise",
        prompt=build_pairwise_prompt(rubric, question, reference, answer_a, answer_b),
        seed=0,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    try:
        record = next(backend.generate([job]))
    except (GenerationError, StopIteration):
        return {"winner": None, "parse": "failed"}
    return parse_pairwise(str(record.output)) or {
        "winner": None,
        "parse": "unparseable",
    }


def pairwise_compare(
    backend: Backend,
    rubric: Rubric,
    question: str,
    reference: str | None,
    answer_a: str,
    answer_b: str,
    temperature: float = 0.0,
    max_tokens: int = 256,
) -> dict[str, Any]:
    """One position-swapped pairwise comparison (P2-ready, M2 smoke).

    The pair is judged twice - (A,B) then (B,A); a preference is
    recorded only when both directions agree, a double tie is a tie,
    and disagreement (the position-bias tell) is reported as such.
    """
    forward = _pairwise_call(
        backend,
        rubric,
        question,
        reference,
        answer_a,
        answer_b,
        temperature,
        max_tokens,
    )
    backward = _pairwise_call(
        backend,
        rubric,
        question,
        reference,
        answer_b,
        answer_a,
        temperature,
        max_tokens,
    )
    # Map the backward call's "b" back onto the original A/B positions.
    remap = {"a": "b", "b": "a", "tie": "tie"}
    f = forward.get("winner")  # forward is (A,B): the winner as-is
    b_raw = backward.get("winner")
    b = remap[b_raw] if b_raw else None  # backward is (B,A): remap to A/B
    if f == b and f in {"a", "b"}:
        preference = f
    elif f == b == "tie":
        preference = "tie"
    else:
        preference = "disagreement"
    return {
        "forward": forward,
        "backward": backward,
        "preference": preference,
    }


# --- data-driven guard -------------------------------------------------------


def judge_model_warnings(judge_model: str, testee_models: set[str]) -> list[str]:
    """Same-family warnings for the judge against the models that
    actually produced the generations being judged.

    The config-level guard (``judge_model_guard``) checks the config's
    conditions; this one checks the ``model`` field of the generation
    records themselves - which is what matters for imported runs, where
    the provenance lives in the data. Raises ValueError on an identical
    model.
    """
    warnings: list[str] = []
    for model in sorted(testee_models):
        if not model:
            continue
        if _normalize_model(model) == _normalize_model(judge_model):
            raise ValueError(
                f"judge model {judge_model!r} is identical to a condition "
                "model in the generations being judged - a judge must not "
                "grade its own outputs"
            )
        if same_model_family(model, judge_model):
            warnings.append(
                f"[judge] WARNING: judge model {judge_model!r} is in the "
                f"same family as testee model {model!r}; self-family bias "
                "is a known judge risk - interpret scores accordingly"
            )
    return warnings
