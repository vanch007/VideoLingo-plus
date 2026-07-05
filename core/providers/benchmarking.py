from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


BENCHMARK_TARGETS = {"llm", "asr", "tts", "e2e"}

TARGET_METRICS = {
    "llm": [
        "json_validity",
        "line_count_preservation",
        "terminology_accuracy",
        "retry_count",
        "latency_seconds",
        "estimated_cost",
    ],
    "asr": [
        "cer",
        "wer",
        "timestamp_drift_seconds",
        "word_timestamp_coverage",
        "rtf",
    ],
    "tts": [
        "duration_ratio",
        "asr_content_score",
        "reference_leak_score",
        "natural_speed_factor",
        "rtf",
        "missing_or_tiny_audio_rate",
    ],
    "e2e": [
        "total_runtime_seconds",
        "repair_count",
        "quality_pass_rate",
        "manual_review_count",
        "cost_per_finished_minute",
    ],
}


@dataclass(frozen=True)
class BenchmarkPlan:
    target: str
    dataset: str
    providers: list[str]
    status: str
    metrics: list[str]
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    notes: list[str] = field(default_factory=list)
    dataset_exists: bool = False
    dataset_status: str = "unknown"
    validation_errors: list[str] = field(default_factory=list)
    validation_warnings: list[str] = field(default_factory=list)
    planned_only: bool = True


def parse_provider_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _load_yaml(path: Path) -> Any:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _validate_llm_dataset(path: Path) -> tuple[str, list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return "invalid_dataset", ["LLM dataset must contain at least one JSONL row."], warnings
    for index, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"line {index}: invalid JSON: {exc.msg}")
            continue
        if not isinstance(item, dict):
            errors.append(f"line {index}: row must be a JSON object.")
            continue
        for key in ("id", "source_language", "target_language", "source_lines", "expected_line_count"):
            if key not in item:
                errors.append(f"line {index}: missing required field '{key}'.")
        source_lines = item.get("source_lines")
        if not isinstance(source_lines, list) or not all(isinstance(value, str) and value for value in source_lines):
            errors.append(f"line {index}: source_lines must be a non-empty list of strings.")
        expected_line_count = item.get("expected_line_count")
        if not isinstance(expected_line_count, int) or expected_line_count < 1:
            errors.append(f"line {index}: expected_line_count must be a positive integer.")
        elif isinstance(source_lines, list) and expected_line_count != len(source_lines):
            warnings.append(f"line {index}: expected_line_count does not match source_lines length.")
        terms = item.get("terms", [])
        if terms is not None:
            if not isinstance(terms, list):
                errors.append(f"line {index}: terms must be a list when present.")
            else:
                for term_index, term in enumerate(terms, start=1):
                    if not isinstance(term, dict) or not term.get("src") or not term.get("tgt"):
                        errors.append(f"line {index}: term {term_index} must include src and tgt.")
    return ("invalid_dataset" if errors else "valid"), errors, warnings


def _validate_yaml_items(path: Path, target: str) -> tuple[str, list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    data = _load_yaml(path)
    if not isinstance(data, dict):
        return "invalid_dataset", [f"{target.upper()} dataset must be a YAML object."], warnings
    items = data.get("items")
    if not isinstance(items, list) or not items:
        return "invalid_dataset", [f"{target.upper()} dataset must include a non-empty items list."], warnings
    status = "seed_schema_only" if data.get("status") == "seed_schema_only" else "valid"
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            errors.append(f"item {index}: must be an object.")
            continue
        if not item.get("id"):
            errors.append(f"item {index}: missing required field 'id'.")
        if not item.get("language") and target != "e2e":
            errors.append(f"item {index}: missing required field 'language'.")
        if target == "tts":
            if not item.get("text"):
                errors.append(f"item {index}: missing required field 'text'.")
            duration = item.get("target_duration")
            if not isinstance(duration, (int, float)) or duration <= 0:
                errors.append(f"item {index}: target_duration must be a positive number.")
        elif target == "asr":
            if not item.get("audio"):
                status = "seed_schema_only"
                warnings.append(f"item {index}: audio is empty; ASR benchmark cannot run yet.")
            if not item.get("expected_transcript"):
                errors.append(f"item {index}: missing required field 'expected_transcript'.")
            window = item.get("expected_time_window")
            if not isinstance(window, dict):
                errors.append(f"item {index}: expected_time_window must be an object.")
            else:
                start = window.get("start")
                end = window.get("end")
                if not isinstance(start, (int, float)) or not isinstance(end, (int, float)) or end <= start:
                    errors.append(f"item {index}: expected_time_window must include numeric start/end with end > start.")
        elif target == "e2e":
            if not item.get("input"):
                status = "seed_schema_only"
                warnings.append(f"item {index}: input is empty; E2E smoke benchmark cannot run yet.")
            for key in ("source_language", "target_language", "profile"):
                if not item.get(key):
                    errors.append(f"item {index}: missing required field '{key}'.")
            if not isinstance(item.get("expected_gates"), dict):
                errors.append(f"item {index}: expected_gates must be an object.")
    if errors:
        return "invalid_dataset", errors, warnings
    return status, errors, warnings


def validate_benchmark_dataset(target: str, dataset: str) -> tuple[str, list[str], list[str]]:
    path = Path(dataset)
    if not path.exists():
        return "missing_dataset", ["Dataset path does not exist."], []
    try:
        if target == "llm":
            return _validate_llm_dataset(path)
        if target in {"asr", "tts", "e2e"}:
            return _validate_yaml_items(path, target)
    except Exception as exc:
        return "invalid_dataset", [f"Could not parse dataset: {type(exc).__name__}: {exc}"], []
    return "invalid_dataset", [f"Unsupported benchmark target: {target}"], []


def build_benchmark_plan(target: str, dataset: str, providers: str | list[str] | None = None) -> BenchmarkPlan:
    if target not in BENCHMARK_TARGETS:
        raise ValueError(f"Unknown benchmark target: {target}")
    provider_list = parse_provider_list(providers) if isinstance(providers, str) else list(providers or [])
    dataset_exists = Path(dataset).exists()
    dataset_status, validation_errors, validation_warnings = validate_benchmark_dataset(target, dataset)
    notes: list[str] = []
    status = dataset_status if dataset_status != "valid" else "ready"
    if dataset_status == "missing_dataset":
        notes.append("Dataset path does not exist; create/freeze fixtures before running a real benchmark.")
    elif dataset_status == "seed_schema_only":
        notes.append("Dataset is a schema seed with placeholder media paths; replace placeholders before running a real benchmark.")
    elif dataset_status == "invalid_dataset":
        notes.append("Dataset exists but does not match the required benchmark fixture schema.")
    if not provider_list:
        status = "missing_providers" if dataset_status == "valid" else status
        notes.append("No providers were specified; pass a comma-separated provider list.")
    return BenchmarkPlan(
        target=target,
        dataset=dataset,
        providers=provider_list,
        status=status,
        metrics=TARGET_METRICS[target],
        notes=notes,
        dataset_exists=dataset_exists,
        dataset_status=dataset_status,
        validation_errors=validation_errors,
        validation_warnings=validation_warnings,
    )


def benchmark_plan_payload(target: str, dataset: str, providers: str | list[str] | None = None) -> dict[str, Any]:
    return asdict(build_benchmark_plan(target, dataset, providers))


def write_benchmark_plan(path: str, payload: dict[str, Any]) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(output)


def benchmark_plan_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Benchmark Plan",
        "",
        f"- Target: `{payload['target']}`",
        f"- Dataset: `{payload['dataset']}`",
        f"- Status: `{payload['status']}`",
        f"- Dataset status: `{payload.get('dataset_status', 'unknown')}`",
        f"- Planned only: `{payload.get('planned_only', True)}`",
        f"- Providers: {', '.join(payload.get('providers') or []) or 'missing'}",
        f"- Metrics: {', '.join(payload.get('metrics') or [])}",
    ]
    if payload.get("notes"):
        lines.extend(["", "## Notes"])
        lines.extend(f"- {item}" for item in payload["notes"])
    if payload.get("validation_errors"):
        lines.extend(["", "## Validation Errors"])
        lines.extend(f"- {item}" for item in payload["validation_errors"])
    if payload.get("validation_warnings"):
        lines.extend(["", "## Validation Warnings"])
        lines.extend(f"- {item}" for item in payload["validation_warnings"])
    lines.extend(["", "This is a non-mutating benchmark plan. It does not call external model providers."])
    return "\n".join(lines) + "\n"


def write_benchmark_summary(path: str, payload: dict[str, Any]) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(benchmark_plan_markdown(payload), encoding="utf-8")
    return str(output)
