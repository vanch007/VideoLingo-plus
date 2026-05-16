from __future__ import annotations

import time
from typing import Any

import pandas as pd

from core.config_utils import load_key
from core.constants import TTS_TASKS_FILE
from core.dubbing_quality import evaluate_dubbing, row_available_duration
from core.providers.dubbing_repair_types import (
    OVER_DURATION_MANUAL,
    OVER_DURATION_REWRITE,
    OVER_DURATION_SPEED_FIT,
    OVER_DURATION_UNKNOWN,
    UNDER_DURATION_REWRITE,
    UNDER_DURATION_SLOW_FIT,
    float_or_none,
)


def classify_over_duration(eval_row: pd.Series | dict[str, Any]) -> str:
    ratio = float_or_none(eval_row.get("duration_ratio"))
    if ratio is None or ratio <= 0:
        return OVER_DURATION_UNKNOWN

    from core.dubbing_quality import get_quality_config

    quality = get_quality_config()
    if ratio <= quality.max_speed_factor:
        return OVER_DURATION_SPEED_FIT

    rewrite_ratio_max = float(load_key("dubbing_quality.over_duration_rewrite_ratio_max", 2.0))
    if ratio <= rewrite_ratio_max:
        return OVER_DURATION_REWRITE
    return OVER_DURATION_MANUAL


def under_duration_target(eval_row: pd.Series | dict[str, Any]) -> float:
    from core.dubbing_quality import get_quality_config

    quality = get_quality_config()
    available = float_or_none(eval_row.get("available_duration"))
    if available is None or available <= 0:
        available = row_available_duration(eval_row)
    ratio_target = available * quality.min_duration_ratio
    fit_margin = max(0.0, float(load_key("dubbing_quality.under_duration_fit_margin_seconds", 0.06)))
    early_end_target = available - quality.max_early_end_drift + fit_margin
    target_ratio = min(1.0, float(load_key("dubbing_quality.slow_fit_target_ratio", quality.min_duration_ratio)))
    preferred = available * target_ratio
    return max(0.05, min(available, max(ratio_target, early_end_target, preferred)))


def classify_under_duration(
    eval_row: pd.Series | dict[str, Any],
    task_row: pd.Series | dict[str, Any] | None = None,
) -> str:
    current_duration = float_or_none(eval_row.get("final_audio_dur"))
    if current_duration is None or current_duration <= 0:
        current_duration = float_or_none(eval_row.get("real_dur"))
    if current_duration is None or current_duration <= 0:
        return UNDER_DURATION_REWRITE

    target_duration = under_duration_target(eval_row)
    if current_duration >= target_duration:
        return UNDER_DURATION_SLOW_FIT

    current_speed = float_or_none(eval_row.get("speed_factor"))
    if current_speed is None and task_row is not None:
        current_speed = float_or_none(task_row.get("speed_factor"))
    current_speed = current_speed if current_speed and current_speed > 0 else 1.0

    slow_factor = current_duration / target_duration
    composite_speed = current_speed * slow_factor
    min_speed = float(load_key("dubbing_quality.slow_fit_min_speed", load_key("speed_factor.min", 0.8)))
    return UNDER_DURATION_SLOW_FIT if slow_factor < 1.0 and composite_speed >= min_speed else UNDER_DURATION_REWRITE


def summarize_over_duration(
    tasks_df: pd.DataFrame | None = None,
    eval_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    if tasks_df is None:
        tasks_df = pd.read_excel(TTS_TASKS_FILE)
    if eval_df is None:
        eval_df, summary = evaluate_dubbing(tasks_df)
    else:
        _, summary = evaluate_dubbing(tasks_df)

    rows: list[dict[str, Any]] = []
    counts = {
        OVER_DURATION_SPEED_FIT: 0,
        OVER_DURATION_REWRITE: 0,
        OVER_DURATION_MANUAL: 0,
        OVER_DURATION_UNKNOWN: 0,
    }
    failing = eval_df[eval_df["reason"].fillna("").str.contains("over_duration", na=False)].copy()
    for _, row in failing.sort_values("duration_ratio", ascending=False, kind="stable").iterrows():
        bucket = classify_over_duration(row)
        counts[bucket] = counts.get(bucket, 0) + 1
        rows.append(
            {
                "number": int(row["number"]),
                "bucket": bucket,
                "available_duration": float_or_none(row.get("available_duration")),
                "final_audio_dur": float_or_none(row.get("final_audio_dur")),
                "duration_ratio": float_or_none(row.get("duration_ratio")),
                "text": str(row.get("text", "")),
                "suggested_action": {
                    OVER_DURATION_SPEED_FIT: "speed_fit_existing_audio",
                    OVER_DURATION_REWRITE: "rewrite_and_regenerate",
                    OVER_DURATION_MANUAL: "timeline_or_manual_review",
                    OVER_DURATION_UNKNOWN: "manual_review",
                }.get(bucket, "manual_review"),
            }
        )

    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source": TTS_TASKS_FILE,
        "summary": summary,
        "thresholds": {
            "max_duration_ratio": float(load_key("dubbing_quality.max_duration_ratio", 1.08)),
            "max_speed_factor": float(load_key("dubbing_quality.max_speed_factor", 1.35)),
            "rewrite_ratio_max": float(load_key("dubbing_quality.over_duration_rewrite_ratio_max", 2.0)),
        },
        "counts": counts,
        "rows": rows,
    }
