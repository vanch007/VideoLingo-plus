from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.providers.dubbing_repair_rules import summarize_over_duration
from core.providers.dubbing_repair_types import (
    DUBBING_OVERDURATION_REPORT_JSON,
    DUBBING_REPAIR_HISTORY_JSONL,
    DUBBING_REPAIR_PLAN_JSON,
    safe_json_value,
)


def write_over_duration_report(
    report: dict[str, Any] | None = None,
    path: str = DUBBING_OVERDURATION_REPORT_JSON,
) -> str:
    report = report or summarize_over_duration()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(safe_json_value(report), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return path


def write_repair_plan(plan: dict[str, Any], path: str = DUBBING_REPAIR_PLAN_JSON) -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(safe_json_value(plan), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return path


def load_repair_plan(path: str = DUBBING_REPAIR_PLAN_JSON) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def append_repair_history(record: dict[str, Any], path: str = DUBBING_REPAIR_HISTORY_JSONL) -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(safe_json_value(record), ensure_ascii=False, allow_nan=False) + "\n")
    return path


def compact_summary(summary: dict[str, Any] | None) -> dict[str, Any] | None:
    if not summary:
        return None
    keys = [
        "total",
        "ok",
        "warn",
        "fail",
        "missing_audio_count",
        "silent_fallback_count",
        "max_duration_ratio",
        "avg_duration_ratio",
    ]
    out = {key: summary.get(key) for key in keys if key in summary}
    gate = summary.get("quality_gate")
    if isinstance(gate, dict):
        out["quality_gate"] = {
            "passed": gate.get("passed"),
            "reasons": gate.get("reasons", {}),
        }
    return out
