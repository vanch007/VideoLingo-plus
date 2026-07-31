from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any

from core.constants import AUDIO_DIR


DUBBING_REPAIR_PLAN_JSON = os.path.join(AUDIO_DIR, "dubbing_repair_plan.json")
DUBBING_REPAIR_HISTORY_JSONL = os.path.join(AUDIO_DIR, "dubbing_repair_history.jsonl")
DUBBING_OVERDURATION_REPORT_JSON = os.path.join(AUDIO_DIR, "dubbing_over_duration_report.json")
KNOWN_BACKENDS = {"indextts2", "omnivoice", "qwen3_tts", "voxcpm2", "higgs", "dots", "zonos2", "moss"}
ASR_RESULT_COLUMNS = (
    "asr_transcript",
    "asr_content_score",
    "asr_leakage_score",
    "asr_status",
    "asr_line_results",
    "asr_fingerprint",
    "asr_language",
    "asr_backend",
)
REGENERATE_REASONS = {
    "missing_audio",
    "silent_or_tiny_audio",
    "low_content_score",
    "reference_leak",
    "over_duration",
    "under_duration",
    "speech_rate_fast",
}
TIMELINE_REASONS = {"start_drift", "end_drift"}
OVER_DURATION_SPEED_FIT = "speed_fit_possible"
OVER_DURATION_REWRITE = "rewrite_needed"
OVER_DURATION_MANUAL = "timeline_or_manual_needed"
OVER_DURATION_UNKNOWN = "unknown"
UNDER_DURATION_SLOW_FIT = "slow_fit_possible"
UNDER_DURATION_REWRITE = "expand_rewrite_needed"


@dataclass(frozen=True)
class RepairAction:
    number: int
    status: str
    reasons: list[str]
    action: str
    backend: str | None = None
    rewrite: bool = False
    delete_audio: bool = False
    full_remap_recommended: bool = False
    text: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "number": self.number,
            "status": self.status,
            "reasons": self.reasons,
            "action": self.action,
            "backend": self.backend,
            "rewrite": self.rewrite,
            "delete_audio": self.delete_audio,
            "full_remap_recommended": self.full_remap_recommended,
            "text": self.text,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class RepairApplySummary:
    planned: int
    applied: int
    skipped: int
    dry_run: bool
    full_remap: bool
    plan_path: str
    eval_summary: dict[str, Any] | None = None


@dataclass(frozen=True)
class RepairBatchSummary:
    batches: int
    batch_limit: int | None
    applied: int
    skipped: int
    stopped_reason: str
    history_path: str
    final_eval_summary: dict[str, Any] | None
    runs: list[dict[str, Any]]


def blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str):
        return not value.strip()
    return False


def safe_json_value(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {key: safe_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [safe_json_value(item) for item in value]
    return value


def split_reasons(value: Any) -> list[str]:
    if blank(value):
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value).split(",") if item.strip()]


def parse_filter(value: str | list[str] | set[str] | tuple[str, ...] | None) -> set[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        parsed = {item.strip() for item in value.split(",") if item.strip()}
    else:
        parsed = {str(item).strip() for item in value if str(item).strip()}
    return parsed or None


def float_or_none(value: Any) -> float | None:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
