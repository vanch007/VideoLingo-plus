from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd

from core.config_utils import load_key
from core.constants import AUDIO_DIR, TTS_TASKS_FILE
from core.dubbing_quality import evaluate_dubbing, parse_list
from core.timing_utils import srt_time_to_seconds


TIMELINE_RESCUE_JSON = os.path.join(AUDIO_DIR, "timeline_rescue_report.json")
TIMELINE_RESCUE_XLSX = os.path.join(AUDIO_DIR, "timeline_rescue_report.xlsx")
TIMELINE_RESCUE_TASKS_XLSX = os.path.join(AUDIO_DIR, "tts_tasks_timeline_rescue.xlsx")


def _safe(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {key: _safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_safe(item) for item in value]
    return value


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _window(row: pd.Series | dict[str, Any]) -> tuple[float, float]:
    sub_times = parse_list(row.get("sub_times"))
    if len(sub_times) == 2 and all(isinstance(item, (int, float)) for item in sub_times):
        return float(sub_times[0]), float(sub_times[1])
    return srt_time_to_seconds(str(row.get("start_time", "0"))), srt_time_to_seconds(str(row.get("end_time", "0")))


def _row_metrics(row: pd.Series | dict[str, Any]) -> dict[str, Any]:
    start, end = _window(row)
    timestamp_duration = max(end - start, 0.0)
    declared_duration = _float(row.get("duration"), timestamp_duration)
    gap = _float(row.get("gap"), 0.0)
    raw_gap = _float(row.get("raw_gap"), gap)
    tolerance = _float(row.get("tolerance"), 0.0)
    duration_ratio = _float(row.get("duration_ratio"), 0.0)
    final_audio_dur = _float(row.get("final_audio_dur"), _float(row.get("real_dur"), 0.0))

    issues: list[str] = []
    if raw_gap < -0.02 or gap < -0.02:
        issues.append("negative_gap")
    if tolerance < -0.02:
        issues.append("negative_tolerance")
    if timestamp_duration > 0 and abs(declared_duration - timestamp_duration) > 0.25:
        issues.append("duration_mismatch")
    if duration_ratio > float(load_key("dubbing_quality.max_duration_ratio", 1.08)):
        issues.append("over_duration")

    return {
        "number": int(row["number"]),
        "start": start,
        "end": end,
        "timestamp_duration": timestamp_duration,
        "declared_duration": declared_duration,
        "gap": gap,
        "raw_gap": raw_gap,
        "tolerance": tolerance,
        "final_audio_dur": final_audio_dur,
        "duration_ratio": duration_ratio,
        "issues": issues,
        "text": str(row.get("text", "")),
    }


def _cluster_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    start = min(row["start"] for row in rows)
    end = max(row["end"] for row in rows)
    current_window = max(end - start, 0.05)
    declared_window = max(sum(max(row["declared_duration"], row["timestamp_duration"]) for row in rows), 0.05)
    audio_sum = sum(row["final_audio_dur"] for row in rows)
    required_speed_current = audio_sum / current_window
    required_speed_declared = audio_sum / declared_window
    max_speed = float(load_key("dubbing_quality.max_speed_factor", 1.35))
    rewrite_ratio_max = float(load_key("dubbing_quality.over_duration_rewrite_ratio_max", 2.0))
    issue_counts: dict[str, int] = {}
    for row in rows:
        for issue in row["issues"]:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1

    if required_speed_declared <= max_speed:
        action = "rebuild_monotonic_from_declared_duration"
    elif required_speed_declared <= rewrite_ratio_max:
        action = "rebuild_monotonic_plus_rewrite"
    elif required_speed_current <= max_speed:
        action = "cluster_speed_fit_current_window"
    else:
        action = "source_realign_or_script_condense"

    return {
        "start": start,
        "end": end,
        "row_count": len(rows),
        "numbers": [row["number"] for row in rows],
        "current_window": current_window,
        "declared_window": declared_window,
        "final_audio_dur": audio_sum,
        "required_speed_current_window": required_speed_current,
        "required_speed_declared_window": required_speed_declared,
        "issue_counts": issue_counts,
        "suggested_action": action,
        "sample_text": " ".join(row["text"] for row in rows[:3])[:240],
    }


def _build_clusters(metrics: list[dict[str, Any]], max_gap: float) -> list[dict[str, Any]]:
    clusters: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []

    for row in metrics:
        if not row["issues"]:
            if current:
                clusters.append(current)
                current = []
            continue

        if not current:
            current = [row]
            continue

        previous = current[-1]
        overlaps_or_touches = row["start"] <= previous["end"] + max_gap
        contiguous_numbers = row["number"] <= previous["number"] + 1
        if overlaps_or_touches or contiguous_numbers:
            current.append(row)
        else:
            clusters.append(current)
            current = [row]

    if current:
        clusters.append(current)
    return [_cluster_metrics(cluster) for cluster in clusters]


def build_timeline_rescue_report(
    tasks_df: pd.DataFrame | None = None,
    eval_df: pd.DataFrame | None = None,
    *,
    max_gap: float = 0.05,
    limit_clusters: int | None = None,
) -> dict[str, Any]:
    if tasks_df is None:
        tasks_df = pd.read_excel(TTS_TASKS_FILE)
    if eval_df is None:
        eval_df, dubbing_summary = evaluate_dubbing(tasks_df)
    else:
        _, dubbing_summary = evaluate_dubbing(tasks_df)

    merged = tasks_df.merge(
        eval_df[["number", "status", "reason", "available_duration", "final_audio_dur", "duration_ratio"]],
        on="number",
        how="left",
        suffixes=("", "_eval"),
    )
    metrics = [_row_metrics(row) for _, row in merged.sort_values("number", kind="stable").iterrows()]

    duplicate_windows: dict[tuple[float, float], int] = {}
    for row in metrics:
        key = (round(row["start"], 3), round(row["end"], 3))
        duplicate_windows[key] = duplicate_windows.get(key, 0) + 1
    for row in metrics:
        key = (round(row["start"], 3), round(row["end"], 3))
        if duplicate_windows.get(key, 0) > 1 and "duplicate_window" not in row["issues"]:
            row["issues"].append("duplicate_window")

    issue_counts: dict[str, int] = {}
    for row in metrics:
        for issue in row["issues"]:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1

    issue_rows = [row for row in metrics if row["issues"]]
    clusters = _build_clusters(metrics, max_gap=max_gap)
    clusters = sorted(clusters, key=lambda item: item["required_speed_declared_window"], reverse=True)
    if limit_clusters is not None:
        clusters = clusters[:limit_clusters]

    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source": TTS_TASKS_FILE,
        "dubbing_summary": dubbing_summary,
        "thresholds": {
            "max_gap": max_gap,
            "max_speed_factor": float(load_key("dubbing_quality.max_speed_factor", 1.35)),
            "rewrite_ratio_max": float(load_key("dubbing_quality.over_duration_rewrite_ratio_max", 2.0)),
        },
        "summary": {
            "total_rows": len(metrics),
            "issue_rows": len(issue_rows),
            "clusters": len(clusters),
            "issue_counts": issue_counts,
        },
        "clusters": clusters,
        "rows": issue_rows,
    }


def write_timeline_rescue_report(
    report: dict[str, Any] | None = None,
    *,
    json_path: str = TIMELINE_RESCUE_JSON,
    xlsx_path: str = TIMELINE_RESCUE_XLSX,
) -> dict[str, str]:
    report = report or build_timeline_rescue_report()
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    Path(json_path).write_text(json.dumps(_safe(report), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

    try:
        with pd.ExcelWriter(xlsx_path) as writer:
            pd.DataFrame(report["clusters"]).to_excel(writer, index=False, sheet_name="clusters")
            pd.DataFrame(report["rows"]).to_excel(writer, index=False, sheet_name="rows")
    except Exception:
        xlsx_path = ""

    return {"json_path": json_path, "xlsx_path": xlsx_path}


def _task_window_health(df: pd.DataFrame) -> dict[str, Any]:
    windows: dict[tuple[float, float], int] = {}
    negative_gaps = 0
    for _, row in df.iterrows():
        start, end = _window(row)
        key = (round(start, 3), round(end, 3))
        windows[key] = windows.get(key, 0) + 1
        if _float(row.get("raw_gap"), _float(row.get("gap"), 0.0)) < -0.02:
            negative_gaps += 1
    return {
        "rows": int(len(df)),
        "duplicate_window_rows": int(sum(count for count in windows.values() if count > 1)),
        "negative_gap_rows": int(negative_gaps),
        "min_duration": _float(pd.to_numeric(df.get("duration", pd.Series(dtype=float)), errors="coerce").min(), 0.0),
        "median_duration": _float(pd.to_numeric(df.get("duration", pd.Series(dtype=float)), errors="coerce").median(), 0.0),
    }


def write_timeline_rescue_candidate_tasks(
    output_path: str = TIMELINE_RESCUE_TASKS_XLSX,
    *,
    pre_merge: bool = True,
) -> dict[str, Any]:
    from core.step8_1_gen_audio_task import process_srt
    from core.step8_2_gen_dub_chunks import pre_merge_short_chunks

    df = process_srt()
    df = df[df["text"].notna() & (df["text"].astype(str).str.strip() != "")]
    df = df[df["duration"] > 0].reset_index(drop=True)
    raw_count = int(len(df))
    if pre_merge and not df.empty:
        df = pre_merge_short_chunks(df)

    tolerance_cap = float(load_key("tolerance", 2.5))
    starts = []
    ends = []
    for _, row in df.iterrows():
        start, end = _window(row)
        starts.append(start)
        ends.append(end)

    gaps: list[float] = []
    raw_gaps: list[float] = []
    for idx in range(len(df)):
        if idx + 1 >= len(df):
            raw_gap = 0.0
        else:
            raw_gap = starts[idx + 1] - ends[idx]
        raw_gaps.append(raw_gap)
        gaps.append(max(0.0, raw_gap))

    df["raw_gap"] = raw_gaps
    df["gap"] = gaps
    df["tolerance"] = [max(0.0, min(gap, tolerance_cap)) for gap in gaps]
    df["tol_dur"] = df["duration"] + df["tolerance"]
    df["cut_off"] = 1
    df["number"] = range(1, len(df) + 1)
    df.to_excel(output_path, index=False)

    return {
        "path": output_path,
        "raw_rows": raw_count,
        "pre_merge": pre_merge,
        "health": _task_window_health(df),
    }
