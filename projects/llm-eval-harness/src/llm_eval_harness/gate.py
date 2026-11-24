"""CI regression gate (M5, PLAN section 7).

``gate`` compares the metrics table of a run's ``summary.json`` (the
M5 report) against a *committed baseline* (a ``summary.json`` plus its
provenance in ``meta``: run id, task-file SHA-256, git commit), under
the per-metric thresholds and guards from a gate TOML:

```toml
[metrics."judge.correctness.mean"]
min_delta = 0.0        # fail if new - baseline < min_delta
max_delta = 50         # fail if new - baseline > max_delta
ci_low_min = 0.30      # fail if the new CI lower bound < ci_low_min
check = false          # report-only: shown in gate_report.md, not enforced

[guards]
judge_coverage_min = 0.95   # parse coverage floor
```

Exit codes: 0 = pass, 1 = regression, 2 = error (missing baseline or
summary, unknown schema, missing metric or CI, coverage below the
guard). A task-file hash mismatch is a loud *warning* (a different
question set makes the comparison suspect) - it does not fail the
gate by itself.
"""

from __future__ import annotations

import json
import sys
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

GATE_SCHEMA = 1
SUMMARY_SCHEMA = 1

_METRIC_KEYS = frozenset({"min_delta", "max_delta", "ci_low_min", "check"})
_GUARD_KEYS = frozenset({"judge_coverage_min"})


def load_gate_config(path: str | Path) -> dict[str, Any]:
    """Parse and validate a gate TOML. Raises ValueError on bad shape."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"gate config not found: {config_path}")
    try:
        with config_path.open("rb") as handle:
            raw = tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"invalid TOML in {config_path}: {exc}") from exc
    unknown = set(raw) - {"metrics", "guards"}
    if unknown:
        raise ValueError(f"unknown top-level gate config key(s) {sorted(unknown)}")
    metrics: dict[str, Any] = {}
    for name, spec in (raw.get("metrics") or {}).items():
        if not isinstance(spec, dict):
            raise ValueError(f'[metrics."{name}"]: must be a table')
        bad = set(spec) - _METRIC_KEYS
        if bad:
            raise ValueError(f'[metrics."{name}"]: unknown key(s) {sorted(bad)}')
        for key in ("min_delta", "max_delta", "ci_low_min"):
            if key in spec and not isinstance(spec[key], (int, float)):
                raise ValueError(f'[metrics."{name}"].{key}: must be a number')
        if "check" in spec and not isinstance(spec["check"], bool):
            raise ValueError(f'[metrics."{name}"].check: must be a bool')
        metrics[str(name)] = spec
    guards: dict[str, Any] = {}
    for key, value in (raw.get("guards") or {}).items():
        if key not in _GUARD_KEYS:
            raise ValueError(f"[guards]: unknown key(s) {[key]}")
        if not isinstance(value, (int, float)):
            raise ValueError(f"[guards].{key}: must be a number")
        guards[str(key)] = value
    if not metrics and not guards:
        raise ValueError("gate config checks nothing (no metrics, no guards)")
    return {"metrics": metrics, "guards": guards}


def evaluate(
    new: dict[str, Any], base: dict[str, Any], cfg: dict[str, Any]
) -> dict[str, Any]:
    """Pure comparison: new summary vs baseline under the gate config.

    Returns the gate result dict (serializable to ``gate.json``) with
    ``exit_code`` per the section 7 contract.
    """
    errors: list[str] = []
    warnings: list[str] = []

    for label, doc in (("new summary", new), ("baseline", base)):
        schema = doc.get("meta", {}).get("schema")
        if schema != SUMMARY_SCHEMA:
            errors.append(
                f"{label} has unknown summary schema {schema!r} "
                f"(gate expects {SUMMARY_SCHEMA}) - refusing to compare"
            )
    if not errors:
        new_sha = new.get("meta", {}).get("tasks_sha256")
        base_sha = base.get("meta", {}).get("tasks_sha256")
        if new_sha and base_sha and new_sha != base_sha:
            warnings.append(
                "task file SHA-256 MISMATCH: the new run and the baseline "
                f"were graded on different question sets "
                f"(new {new_sha[:12]}... vs baseline {base_sha[:12]}...) - "
                "the comparison is suspect"
            )

    metrics_out: list[dict[str, Any]] = []
    for name in sorted(cfg["metrics"]):
        spec = cfg["metrics"][name]
        checked = bool(spec.get("check", True))
        entry: dict[str, Any] = {
            "name": name,
            "checked": checked,
            "thresholds": {k: v for k, v in spec.items()},
        }
        new_m = new.get("metrics", {}).get(name)
        base_m = base.get("metrics", {}).get(name)

        if not checked:
            if new_m is not None and base_m is not None:
                entry["value"] = new_m.get("value")
                entry["baseline"] = base_m.get("value")
                if isinstance(entry["value"], (int, float)) and isinstance(
                    entry["baseline"], (int, float)
                ):
                    entry["delta"] = round(
                        float(entry["value"]) - float(entry["baseline"]), 6
                    )
            entry["passed"] = None  # report-only: not evaluated
            metrics_out.append(entry)
            continue

        if new_m is None:
            errors.append(
                f"metric {name!r} required by the gate config is "
                "missing from the new summary"
            )
            entry["error"] = "missing from new summary"
            metrics_out.append(entry)
            continue
        if base_m is None:
            errors.append(
                f"metric {name!r} required by the gate config is missing "
                "from the baseline"
            )
            entry["error"] = "missing from baseline"
            metrics_out.append(entry)
            continue

        value = float(new_m["value"])
        baseline_value = float(base_m["value"])
        delta = value - baseline_value
        entry["value"] = value
        entry["baseline"] = baseline_value
        entry["delta"] = round(delta, 6)

        reasons: list[str] = []
        if "min_delta" in spec and delta < float(spec["min_delta"]):
            reasons.append(f"delta {delta:+.6f} < min_delta {spec['min_delta']}")
        if "max_delta" in spec and delta > float(spec["max_delta"]):
            reasons.append(f"delta {delta:+.6f} > max_delta {spec['max_delta']}")
        if "ci_low_min" in spec:
            ci = new_m.get("ci")
            if not (isinstance(ci, (list, tuple)) and len(ci) == 2):
                errors.append(
                    f"metric {name!r} sets ci_low_min but the new summary has no CI"
                )
                entry["error"] = "no CI in new summary"
                metrics_out.append(entry)
                continue
            if float(ci[0]) < float(spec["ci_low_min"]):
                reasons.append(
                    f"CI lower bound {float(ci[0]):.6f} < "
                    f"ci_low_min {spec['ci_low_min']}"
                )
        entry["passed"] = not reasons
        entry["reasons"] = reasons
        metrics_out.append(entry)

    guards_out: dict[str, Any] = {}
    if "judge_coverage_min" in cfg["guards"]:
        threshold = float(cfg["guards"]["judge_coverage_min"])
        cov = new.get("metrics", {}).get("judge.coverage", {}).get("value")
        if not isinstance(cov, (int, float)):
            errors.append(
                f"guard judge_coverage_min = {threshold} set, but the new "
                "summary has no judge.coverage metric (run the judge first)"
            )
            guards_out["judge_coverage_min"] = {
                "threshold": threshold,
                "value": None,
                "passed": False,
            }
        else:
            ok = float(cov) >= threshold
            guards_out["judge_coverage_min"] = {
                "threshold": threshold,
                "value": float(cov),
                "passed": ok,
            }
            if not ok:
                errors.append(
                    f"judge coverage {float(cov):.4f} is below the guard "
                    f"judge_coverage_min = {threshold} - parse failures are "
                    "too high to trust any judge metric this run"
                )

    failures = [m for m in metrics_out if m.get("passed") is False]
    passed = not errors and not failures
    exit_code = 2 if errors else (1 if failures else 0)
    return {
        "schema": GATE_SCHEMA,
        "generated_at": datetime.now(UTC).isoformat(),
        "run_id": new.get("meta", {}).get("run_id"),
        "baseline_run_id": base.get("meta", {}).get("run_id"),
        "passed": passed,
        "exit_code": exit_code,
        "warnings": warnings,
        "errors": errors,
        "metrics": metrics_out,
        "guards": guards_out,
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:+.4f}"
    return str(value)


def render_gate_md(
    result: dict[str, Any], baseline_path: Path, config_path: Path
) -> str:
    """Markdown gate report (``gate_report.md``)."""
    verdict = (
        "PASS"
        if result["passed"]
        else ("ERROR" if result["exit_code"] == 2 else "REGRESSION")
    )
    lines: list[str] = [
        f"# Gate report: {result['run_id']}",
        "",
        f"Verdict: **{verdict}** (exit code {result['exit_code']}). "
        f"Baseline: `{baseline_path}` - run {result['baseline_run_id']}. "
        f"Config: `{config_path}`.",
        "",
        "| metric | baseline | new | delta | thresholds | checked | verdict |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for entry in result["metrics"]:
        thresholds = (
            ", ".join(
                f"{key}={value}"
                for key, value in sorted(entry.get("thresholds", {}).items())
            )
            or "-"
        )
        if entry.get("error"):
            verdict_s = f"error: {entry['error']}"
        elif entry["passed"] is None:
            verdict_s = "report-only"
        elif entry["passed"]:
            verdict_s = "pass"
        else:
            verdict_s = "; ".join(entry.get("reasons", []))
        lines.append(
            f"| {entry['name']} | {_fmt(entry.get('baseline'))} "
            f"| {_fmt(entry.get('value'))} | {_fmt(entry.get('delta'))} "
            f"| {thresholds} | {'yes' if entry['checked'] else 'no'} | {verdict_s} |"
        )
    if result["guards"]:
        lines += ["", "Guards:"]
        for name, guard in result["guards"].items():
            lines.append(
                f"- {name} = {guard['threshold']}: value "
                f"{guard['value'] if guard['value'] is not None else 'missing'} "
                f"-> {'pass' if guard['passed'] else 'FAIL'}"
            )
    if result["warnings"]:
        lines += ["", "Warnings:"]
        lines += [f"- {warning}" for warning in result["warnings"]]
    if result["errors"]:
        lines += ["", "Errors:"]
        lines += [f"- {error}" for error in result["errors"]]
    lines += [
        "",
        "Note: report-only metrics (checked = no) may appear in this "
        "report but must not drive the gate or a go/no-go decision "
        "(PLAN sections 5 and 7).",
        "",
    ]
    return "\n".join(lines)


def resolve_summary_path(run: str) -> Path:
    """``--run`` -> the run's ``summary.json``.

    A path to the run directory, a path to a summary file, or a run
    name under ``reports/``.
    """
    path = Path(run)
    if path.is_dir():
        return path / "summary.json"
    if path.suffix == ".json" and path.is_file():
        return path
    return Path("reports") / run / "summary.json"


def run_gate(
    run: str,
    baseline_path: str | Path,
    config_path: str | Path,
    output_dir: str | Path | None = None,
) -> int:
    """CLI orchestration for ``gate``: load, compare, write, return code."""
    summary_path = resolve_summary_path(run)
    if not summary_path.is_file():
        print(
            f"error: run summary not found: {summary_path} (run `report` first)",
            file=sys.stderr,
        )
        return 2
    try:
        new = json.loads(summary_path.read_text(encoding="utf-8"))
        base = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
        cfg = load_gate_config(config_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    result = evaluate(new, base, cfg)
    out_dir = Path(output_dir) if output_dir else summary_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "gate.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (out_dir / "gate_report.md").write_text(
        render_gate_md(result, Path(baseline_path), Path(config_path)), encoding="utf-8"
    )

    for warning in result["warnings"]:
        print(f"WARNING: {warning}", file=sys.stderr)
    for metric in result["metrics"]:
        if metric["passed"] is None and metric.get("error"):
            continue
        if metric["passed"] is None:
            print(
                f"  {metric['name']}: report-only (value {_fmt(metric.get('value'))})"
            )
        else:
            status = "pass" if metric["passed"] else "FAIL"
            print(f"  {metric['name']}: delta {_fmt(metric.get('delta'))} -> {status}")
    for name, guard in result["guards"].items():
        print(
            f"  guard {name}: {guard['value']} (threshold {guard['threshold']}) "
            f"-> {'pass' if guard['passed'] else 'FAIL'}"
        )
    for error in result["errors"]:
        print(f"ERROR: {error}", file=sys.stderr)
    verdict = (
        "PASS"
        if result["passed"]
        else ("ERROR" if result["exit_code"] == 2 else "REGRESSION")
    )
    print(f"gate: {verdict} (exit {result['exit_code']})")
    print(f"Output: {out_dir / 'gate.json'}, {out_dir / 'gate_report.md'}")
    return result["exit_code"]
