import ast
import json
import math
import os
from dataclasses import dataclass
from typing import Any

import pandas as pd

from core.all_whisper_methods.audio_preprocess import get_audio_duration
from core.config_utils import load_key
from core.constants import AUDIO_DIR, SEGS_DIR, TTS_TASKS_FILE
from core.providers.quality_gate import summarize_quality_gate
from core.timing_utils import srt_time_to_seconds


DUBBING_EVAL_XLSX = os.path.join(AUDIO_DIR, "dubbing_eval.xlsx")
DUBBING_EVAL_JSON = os.path.join(AUDIO_DIR, "dubbing_eval.json")
DUBBING_MERGE_ISSUES_JSON = os.path.join(AUDIO_DIR, "dubbing_merge_issues.json")


@dataclass(frozen=True)
class DubbingQualityConfig:
    mode: str
    enabled: bool
    max_duration_ratio: float
    min_duration_ratio: float
    rewrite_estimate_ratio: float
    rewrite_actual_ratio: float
    max_speed_factor: float
    max_natural_speed_factor: float
    max_rewrite_rounds: int
    allow_silence_fallback: bool
    max_start_drift: float
    max_end_drift: float
    max_early_end_drift: float
    asr_readback: bool
    content_score_min: float
    leak_score_max: float
    loudness_target_lufs: float


def get_quality_config() -> DubbingQualityConfig:
    mode = load_key("dubbing_quality.mode", load_key("quality_mode", "high_sync"))
    return DubbingQualityConfig(
        mode=mode,
        enabled=mode in {"dubbing", "high_sync"},
        max_duration_ratio=float(load_key("dubbing_quality.max_duration_ratio", 1.08)),
        min_duration_ratio=float(load_key("dubbing_quality.min_duration_ratio", 0.90)),
        rewrite_estimate_ratio=float(load_key("dubbing_quality.rewrite_estimate_ratio", 1.08)),
        rewrite_actual_ratio=float(load_key("dubbing_quality.rewrite_actual_ratio", 1.08)),
        max_speed_factor=float(load_key("dubbing_quality.max_speed_factor", load_key("speed_factor.accept", 2.0))),
        max_natural_speed_factor=float(load_key("dubbing_quality.max_natural_speed_factor", 1.12)),
        max_rewrite_rounds=int(load_key("dubbing_quality.max_rewrite_rounds", 2)),
        allow_silence_fallback=bool(load_key("dubbing_quality.allow_silence_fallback", False)),
        max_start_drift=float(load_key("dubbing_quality.max_start_drift", 0.12)),
        max_end_drift=float(load_key("dubbing_quality.max_end_drift", 0.18)),
        max_early_end_drift=float(load_key("dubbing_quality.max_early_end_drift", 0.45)),
        asr_readback=bool(load_key("dubbing_quality.asr_readback", False)),
        content_score_min=float(load_key("dubbing_quality.content_score_min", 0.88)),
        leak_score_max=float(load_key("dubbing_quality.leak_score_max", 0.12)),
        loudness_target_lufs=float(load_key("dubbing_quality.loudness_target_lufs", -20.0)),
    )


def parse_list(value: Any, default: list | None = None) -> list:
    if isinstance(value, list):
        return value
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return default or []
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            return parsed if isinstance(parsed, list) else (default or [])
        except (SyntaxError, ValueError):
            return default or []
    return default or []


def row_start_seconds(row: pd.Series | dict) -> float:
    return srt_time_to_seconds(row["start_time"])


def row_end_seconds(row: pd.Series | dict) -> float:
    return srt_time_to_seconds(row["end_time"])


def row_available_duration(row: pd.Series | dict) -> float:
    candidates: list[float] = []
    tolerance = float(row.get("tolerance", 0) or 0)

    try:
        window = row_end_seconds(row) - row_start_seconds(row)
        if window > 0:
            candidates.append(window + tolerance)
    except (KeyError, TypeError, ValueError):
        pass

    if "available_duration" in row and pd.notna(row["available_duration"]):
        candidates.append(float(row["available_duration"]))
    duration = float(row.get("duration", 0) or 0)
    if duration > 0:
        candidates.append(duration + tolerance)
    return max(candidates or [0.05])


def row_target_duration(row: pd.Series | dict) -> float:
    available = row_available_duration(row)
    if "target_duration" in row and pd.notna(row["target_duration"]):
        return max(float(row["target_duration"]), available, 0.05)
    return available


def normalize_lines(value: Any) -> list[str]:
    lines = parse_list(value)
    if lines:
        return [str(line).strip() for line in lines]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def apply_dubbing_budget_columns(df: pd.DataFrame) -> pd.DataFrame:
    quality = get_quality_config()
    out = df.copy()
    if "tolerance" not in out.columns:
        out["tolerance"] = 0.0
    out["available_duration"] = out.apply(row_available_duration, axis=1)
    out["target_duration"] = out["available_duration"]
    out["max_duration_ratio"] = quality.max_duration_ratio
    out["max_speed_factor"] = quality.max_speed_factor
    out["max_natural_speed_factor"] = quality.max_natural_speed_factor
    out["quality_mode"] = quality.mode
    return out


def estimated_rewrite_needed(row: pd.Series | dict) -> bool:
    quality = get_quality_config()
    if not quality.enabled:
        return False
    est_dur = float(row.get("est_dur", 0) or 0)
    if_too_fast = int(row.get("if_too_fast", 0) or 0)
    return if_too_fast >= 2 or est_dur > row_available_duration(row) * quality.rewrite_estimate_ratio


def actual_rewrite_needed(row: pd.Series | dict) -> bool:
    quality = get_quality_config()
    if not quality.enabled:
        return False
    real_dur = float(row.get("real_dur", 0) or 0)
    if real_dur <= 0:
        return False
    return real_dur > row_available_duration(row) * quality.rewrite_actual_ratio


def natural_speed_rewrite_needed(row: pd.Series | dict) -> bool:
    quality = get_quality_config()
    if not quality.enabled:
        return False
    real_dur = float(row.get("real_dur", 0) or 0)
    if real_dur <= 0:
        return False
    return real_dur > row_available_duration(row) * quality.max_natural_speed_factor


def actual_expand_needed(row: pd.Series | dict) -> bool:
    quality = get_quality_config()
    if not quality.enabled:
        return False
    real_dur = float(row.get("real_dur", 0) or 0)
    if real_dur <= 0:
        return False
    min_speed = max(float(load_key("speed_factor.min", 0.8)), 0.1)
    projected_duration = real_dur / min_speed
    return projected_duration < row_available_duration(row) * quality.min_duration_ratio


def audio_file_for(number: int, line_index: int) -> str:
    return os.path.join(SEGS_DIR, f"{number}_{line_index}.wav")


def temp_audio_file_for(number: int, line_index: int) -> str:
    return os.path.join(AUDIO_DIR, "temp", f"{number}_{line_index}_temp.wav")


def _flatten_times(row: pd.Series | dict, column: str) -> list[list[float]]:
    times = parse_list(row.get(column))
    if not times:
        return []
    if len(times) == 2 and all(isinstance(item, (int, float)) for item in times):
        return [[float(times[0]), float(times[1])]]
    flattened = []
    for item in times:
        if isinstance(item, list) and len(item) == 2:
            flattened.append([float(item[0]), float(item[1])])
    return flattened


def evaluate_dubbing(tasks_df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    quality = get_quality_config()
    if tasks_df is None:
        tasks_df = pd.read_excel(TTS_TASKS_FILE)

    rows = []
    for _, row in tasks_df.iterrows():
        number = int(row["number"])
        lines = normalize_lines(row.get("lines", row.get("text", ""))) or [str(row.get("text", ""))]
        new_times = _flatten_times(row, "new_sub_times")
        target_start = row_start_seconds(row)
        target_end = row_end_seconds(row) + float(row.get("tolerance", 0) or 0)
        available = row_available_duration(row)
        real_dur = float(row.get("real_dur", 0) or 0)
        actual_start = new_times[0][0] if new_times else None
        actual_end = new_times[-1][1] if new_times else None
        content_score = row.get("asr_content_score", row.get("content_score", None))
        leakage_score = row.get("asr_leakage_score", row.get("leakage_score", None))
        speed_factor = float(row.get("speed_factor", 1.0) or 1.0)

        missing_files = []
        silent_fallbacks = []
        line_durations = []
        for line_index, _ in enumerate(lines):
            audio_file = audio_file_for(number, line_index)
            if not os.path.exists(audio_file):
                missing_files.append(audio_file)
                line_durations.append(0.0)
                continue
            size = os.path.getsize(audio_file)
            dur = get_audio_duration(audio_file)
            line_durations.append(dur)
            if size < int(load_key("dubbing_quality.min_audio_size", 1000)) or dur <= 0.12:
                silent_fallbacks.append(audio_file)

        final_audio_dur = sum(line_durations) if line_durations and not missing_files else real_dur
        duration_ratio = final_audio_dur / available if available > 0 else 0
        start_drift = None if actual_start is None else actual_start - target_start
        end_drift = None if actual_end is None else actual_end - target_end
        overlap_or_gap = None
        if new_times:
            diffs = [new_times[i][0] - new_times[i - 1][1] for i in range(1, len(new_times))]
            overlap_or_gap = min(diffs) if diffs else 0.0

        status = "ok"
        reasons = []
        if missing_files:
            status = "fail"
            reasons.append("missing_audio")
        if silent_fallbacks:
            status = "fail"
            reasons.append("silent_or_tiny_audio")
        ends_too_early = end_drift is not None and end_drift < -quality.max_early_end_drift
        if duration_ratio > quality.max_duration_ratio:
            status = "warn" if status == "ok" else status
            reasons.append("over_duration")
        if (0 < duration_ratio < quality.min_duration_ratio) or ends_too_early:
            status = "warn" if status == "ok" else status
            reasons.append("under_duration")
        if start_drift is not None and abs(start_drift) > quality.max_start_drift:
            status = "warn" if status == "ok" else status
            reasons.append("start_drift")
        if end_drift is not None and end_drift > quality.max_end_drift and "over_duration" not in reasons:
            status = "warn" if status == "ok" else status
            reasons.append("over_duration")
        if speed_factor > quality.max_natural_speed_factor:
            status = "warn" if status == "ok" else status
            reasons.append("speech_rate_fast")
        if content_score is not None and pd.notna(content_score) and float(content_score) < quality.content_score_min:
            status = "warn" if status == "ok" else status
            reasons.append("low_content_score")
        if leakage_score is not None and pd.notna(leakage_score) and float(leakage_score) > quality.leak_score_max:
            status = "warn" if status == "ok" else status
            reasons.append("reference_leak")

        rows.append({
            "number": number,
            "status": status,
            "reason": ",".join(reasons),
            "text": row.get("text", ""),
            "lines": lines,
            "target_start": target_start,
            "target_end": target_end,
            "actual_start": actual_start,
            "actual_end": actual_end,
            "start_drift": start_drift,
            "end_drift": end_drift,
            "duration": float(row.get("duration", 0) or 0),
            "available_duration": available,
            "target_duration": row_target_duration(row),
            "real_dur": real_dur,
            "final_audio_dur": final_audio_dur,
            "duration_ratio": duration_ratio,
            "speed_factor": speed_factor,
            "overlap_or_gap": overlap_or_gap,
            "missing_audio_count": len(missing_files),
            "silent_fallback_count": len(silent_fallbacks),
            "line_durations": line_durations,
            "content_score": None if content_score is None or pd.isna(content_score) else float(content_score),
            "leakage_score": None if leakage_score is None or pd.isna(leakage_score) else float(leakage_score),
        })

    eval_df = pd.DataFrame(rows)
    def safe_float(value: Any) -> float:
        try:
            if value is None or pd.isna(value):
                return 0.0
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    content_scores = pd.to_numeric(eval_df["content_score"], errors="coerce").dropna() if "content_score" in eval_df else pd.Series(dtype="float64")
    leakage_scores = pd.to_numeric(eval_df["leakage_score"], errors="coerce").dropna() if "leakage_score" in eval_df else pd.Series(dtype="float64")
    asr_scored_rows = int(max(len(content_scores), len(leakage_scores)))
    asr_readback_effective = asr_scored_rows > 0

    summary = {
        "mode": quality.mode,
        "total": int(len(eval_df)),
        "ok": int((eval_df["status"] == "ok").sum()) if not eval_df.empty else 0,
        "warn": int((eval_df["status"] == "warn").sum()) if not eval_df.empty else 0,
        "fail": int((eval_df["status"] == "fail").sum()) if not eval_df.empty else 0,
        "max_duration_ratio": safe_float(eval_df["duration_ratio"].max()) if not eval_df.empty else 0.0,
        "min_duration_ratio": safe_float(eval_df["duration_ratio"].min()) if not eval_df.empty else 0.0,
        "avg_duration_ratio": safe_float(eval_df["duration_ratio"].mean()) if not eval_df.empty else 0.0,
        "max_abs_start_drift": safe_float(pd.to_numeric(eval_df["start_drift"], errors="coerce").abs().max()) if "start_drift" in eval_df else 0.0,
        "max_abs_end_drift": safe_float(pd.to_numeric(eval_df["end_drift"], errors="coerce").abs().max()) if "end_drift" in eval_df else 0.0,
        "max_speed_factor": safe_float(eval_df["speed_factor"].max()) if "speed_factor" in eval_df else 0.0,
        "max_natural_speed_factor": quality.max_natural_speed_factor,
        "missing_audio_count": int(eval_df["missing_audio_count"].sum()) if not eval_df.empty else 0,
        "silent_fallback_count": int(eval_df["silent_fallback_count"].sum()) if not eval_df.empty else 0,
        "avg_content_score": safe_float(content_scores.mean()) if not content_scores.empty else None,
        "max_leakage_score": safe_float(leakage_scores.max()) if not leakage_scores.empty else None,
        "asr_readback": asr_readback_effective,
        "asr_readback_configured": quality.asr_readback,
        "asr_scored_rows": asr_scored_rows,
        "loudness_target_lufs": quality.loudness_target_lufs,
    }
    summary["quality_gate"] = summarize_quality_gate(eval_df).to_dict()
    return eval_df, summary


def write_dubbing_eval(tasks_df: pd.DataFrame | None = None) -> dict[str, Any]:
    os.makedirs(AUDIO_DIR, exist_ok=True)
    eval_df, summary = evaluate_dubbing(tasks_df)
    eval_df.to_excel(DUBBING_EVAL_XLSX, index=False)

    def sanitize(value: Any) -> Any:
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        if isinstance(value, dict):
            return {key: sanitize(item) for key, item in value.items()}
        if isinstance(value, list):
            return [sanitize(item) for item in value]
        return value

    with open(DUBBING_EVAL_JSON, "w", encoding="utf-8") as f:
        json.dump(
            sanitize({"summary": summary, "rows": eval_df.to_dict(orient="records")}),
            f,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    return summary


if __name__ == "__main__":
    print(json.dumps(write_dubbing_eval(), ensure_ascii=False, indent=2))
