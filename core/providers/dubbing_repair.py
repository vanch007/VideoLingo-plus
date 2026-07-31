from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Any

import pandas as pd

from core.config_utils import load_key
from core.constants import SEGS_DIR, TEMP_DIR, TTS_TASKS_FILE
from core.dubbing_quality import (
    actual_expand_needed,
    actual_rewrite_needed,
    audio_file_for,
    evaluate_dubbing,
    normalize_lines,
    parse_list,
    row_available_duration,
    row_start_seconds,
    temp_audio_file_for,
    write_dubbing_eval,
)
from core.providers.dubbing_repair_io import (
    append_repair_history,
    compact_summary as _compact_summary,
    load_repair_plan,
    write_over_duration_report,
    write_repair_plan,
)
from core.providers.dubbing_repair_rules import (
    classify_over_duration,
    classify_under_duration,
    summarize_over_duration,
    under_duration_target as _under_duration_target,
)
from core.providers.dubbing_repair_types import (
    ASR_RESULT_COLUMNS,
    DUBBING_REPAIR_HISTORY_JSONL,
    DUBBING_REPAIR_PLAN_JSON,
    KNOWN_BACKENDS,
    OVER_DURATION_MANUAL,
    OVER_DURATION_REWRITE,
    OVER_DURATION_SPEED_FIT,
    REGENERATE_REASONS,
    TIMELINE_REASONS,
    UNDER_DURATION_SLOW_FIT,
    RepairAction,
    RepairApplySummary,
    RepairBatchSummary,
    blank as _blank,
    parse_filter as _action_filter,
    parse_filter as _reason_filter,
    split_reasons as _split_reasons,
)

__all__ = [
    "OVER_DURATION_MANUAL",
    "OVER_DURATION_REWRITE",
    "OVER_DURATION_SPEED_FIT",
    "UNDER_DURATION_SLOW_FIT",
    "RepairAction",
    "RepairApplySummary",
    "RepairBatchSummary",
    "_under_duration_target",
    "append_repair_history",
    "apply_repair_plan",
    "build_repair_plan",
    "choose_repair_action",
    "classify_over_duration",
    "classify_under_duration",
    "load_repair_plan",
    "run_repair_batches",
    "summarize_over_duration",
    "write_over_duration_report",
    "write_repair_plan",
]


def _task_row(tasks_df: pd.DataFrame, number: int) -> pd.Series | None:
    matched = tasks_df[tasks_df["number"] == number]
    if matched.empty:
        return None
    return matched.iloc[0]


def _content_failure_backend() -> str:
    """Return the configured non-Index clone backend for content failures."""
    fallback = str(
        load_key("dubbing_repair.low_content_fallback_backend", "dots") or "dots"
    ).strip()
    return fallback if fallback in KNOWN_BACKENDS else "dots"


def _resolve_backend(row: pd.Series | dict[str, Any], reasons: set[str], backend_fallback: str) -> str | None:
    if backend_fallback != "auto":
        if backend_fallback not in KNOWN_BACKENDS:
            raise ValueError(f"Unknown backend fallback: {backend_fallback}")
        return backend_fallback

    current = row.get("tts_method") or row.get("tts_backend") or row.get("backend")
    current = str(current).strip() if not _blank(current) else ""
    if current in KNOWN_BACKENDS and reasons <= {"missing_audio", "silent_or_tiny_audio"}:
        return current

    if reasons & {"reference_leak", "low_content_score"}:
        return _content_failure_backend()
    if "over_duration" in reasons or "under_duration" in reasons or "speech_rate_fast" in reasons:
        return "indextts2"
    return None


def choose_repair_action(
    eval_row: pd.Series | dict[str, Any],
    task_row: pd.Series | dict[str, Any] | None = None,
    *,
    backend_fallback: str = "auto",
) -> RepairAction:
    task_row = {} if task_row is None else task_row
    reasons = _split_reasons(eval_row.get("reason"))
    reason_set = set(reasons)
    number = int(eval_row["number"])
    rewrite_enabled = bool(load_key("rewrite_text_for_dubbing", True))
    over_duration_bucket = classify_over_duration(eval_row) if "over_duration" in reason_set else ""
    under_duration_bucket = classify_under_duration(eval_row, task_row) if "under_duration" in reason_set else ""
    speed_fit_only = reason_set == {"over_duration"} and over_duration_bucket == OVER_DURATION_SPEED_FIT
    manual_only = reason_set == {"over_duration"} and over_duration_bucket == OVER_DURATION_MANUAL
    slow_fit_only = reason_set == {"under_duration"} and under_duration_bucket == UNDER_DURATION_SLOW_FIT
    regen_reasons = set(REGENERATE_REASONS)
    if speed_fit_only or manual_only:
        regen_reasons.discard("over_duration")
    if slow_fit_only:
        regen_reasons.discard("under_duration")
    needs_regen = bool(reason_set & regen_reasons)
    needs_timeline = bool(reason_set & TIMELINE_REASONS)
    rewrite = (
        rewrite_enabled
        and bool(reason_set & {"over_duration", "under_duration", "speech_rate_fast"})
        and not speed_fit_only
        and not slow_fit_only
        and not manual_only
    )
    backend = _resolve_backend(task_row, reason_set, backend_fallback) if needs_regen else None
    notes: list[str] = []
    if rewrite:
        max_repair_rounds = max(
            0,
            int(load_key("dubbing_repair.max_rewrite_rounds", load_key("dubbing_quality.max_rewrite_rounds", 2))),
        )
        if _repair_rounds(dict(task_row)) >= max_repair_rounds:
            return RepairAction(
                number=number,
                status=str(eval_row.get("status", "")),
                reasons=reasons,
                action="manual_review",
                backend=None,
                rewrite=False,
                delete_audio=False,
                text=str(task_row.get("text") or eval_row.get("text") or ""),
                notes=["Repair rewrite rounds are exhausted; manual script edit is required."],
            )
    max_auto_attempts = max(1, int(load_key("dubbing_repair.max_auto_regeneration_attempts", 2)))
    repair_attempts = int(task_row.get("repair_attempts", 0) or 0)
    terminal_status = str(task_row.get("repair_status", "") or "")
    if (
        reason_set & {"low_content_score", "reference_leak"}
        and repair_attempts >= max_auto_attempts
        and terminal_status in {"rewrite_failed", "rewrite_limit_reached", "asr_quality_unstable", "manual_review"}
    ):
        return RepairAction(
            number=number,
            status=str(eval_row.get("status", "")),
            reasons=reasons,
            action="manual_review",
            backend=None,
            rewrite=False,
            delete_audio=False,
            text=str(task_row.get("text") or eval_row.get("text") or ""),
            notes=[
                "Automatic ASR-quality repair already hit the retry limit; manual listening or script/reference change is required."
            ],
        )

    if "reference_leak" in reason_set and backend == "voxcpm2":
        notes.append("voxcpm2 is not recommended for automatic leak repair; verify with ASR before accepting.")
    if "low_content_score" in reason_set:
        notes.append("ASR content score is below threshold; regenerate with a different MLX clone backend or cleaner reference.")
    if rewrite:
        if "over_duration" in reason_set or "speech_rate_fast" in reason_set:
            notes.append("Text will be shortened before regeneration.")
        else:
            notes.append("Text will be naturally expanded before regeneration.")
    if "speech_rate_fast" in reason_set:
        notes.append("Observed speed factor exceeds the natural speech threshold; prefer shorter text over accelerated audio.")
    if speed_fit_only:
        notes.append("Existing audio can be sped up within the configured max_speed_factor.")
    if slow_fit_only:
        notes.append("Existing audio can be slowed within the configured slow_fit_min_speed.")
    if manual_only:
        notes.append("Duration ratio is too high for automatic cinematic repair; timeline or manual script rewrite is recommended.")
    if needs_timeline:
        notes.append("Run with --full-remap to recompute chunk timing after selected regeneration.")

    if speed_fit_only:
        action = "speed_fit_existing_audio"
    elif slow_fit_only:
        action = "slow_fit_existing_audio"
    elif manual_only:
        action = "timeline_or_manual_review"
    elif needs_regen and rewrite:
        action = "expand_and_regenerate" if reason_set == {"under_duration"} else "rewrite_and_regenerate"
    elif needs_regen and backend:
        action = "regenerate_with_backend"
    elif needs_regen:
        action = "regenerate"
    elif needs_timeline:
        action = "timeline_remap"
    else:
        action = "manual_review"

    text = str(task_row.get("text") or eval_row.get("text") or "")
    return RepairAction(
        number=number,
        status=str(eval_row.get("status", "")),
        reasons=reasons,
        action=action,
        backend=backend,
        rewrite=rewrite,
        delete_audio=needs_regen,
        full_remap_recommended=needs_timeline or bool(reason_set & {"over_duration", "speech_rate_fast"}),
        text=text,
        notes=notes,
    )


def build_repair_plan(
    tasks_df: pd.DataFrame | None = None,
    eval_df: pd.DataFrame | None = None,
    *,
    limit: int | None = None,
    backend_fallback: str = "auto",
    reasons: str | list[str] | set[str] | tuple[str, ...] | None = None,
    actions: str | list[str] | set[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    if tasks_df is None:
        tasks_df = pd.read_excel(TTS_TASKS_FILE)
    if eval_df is None:
        eval_df, summary = evaluate_dubbing(tasks_df)
    else:
        _, summary = evaluate_dubbing(tasks_df)

    items: list[dict[str, Any]] = []
    include_reasons = _reason_filter(reasons)
    include_actions = _action_filter(actions)
    failing = eval_df[eval_df["status"] != "ok"].copy() if not eval_df.empty else eval_df
    if include_reasons and not failing.empty:
        failing = failing[
            failing["reason"].apply(lambda value: bool(set(_split_reasons(value)) & include_reasons))
        ].copy()
    if not failing.empty:
        failing["_repair_priority"] = failing.apply(_repair_priority, axis=1)
        failing = failing.sort_values(["_repair_priority", "number"], kind="stable")
    for _, eval_row in failing.iterrows():
        if limit is not None and len(items) >= limit:
            break
        task = _task_row(tasks_df, int(eval_row["number"]))
        action = choose_repair_action(eval_row, task, backend_fallback=backend_fallback)
        if include_actions and action.action not in include_actions:
            continue
        items.append(action.to_dict())

    reason_counts: dict[str, int] = {}
    action_counts: dict[str, int] = {}
    for item in items:
        action_counts[item["action"]] = action_counts.get(item["action"], 0) + 1
        for reason in item["reasons"]:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source": TTS_TASKS_FILE,
        "plan_path": DUBBING_REPAIR_PLAN_JSON,
        "backend_fallback": backend_fallback,
        "reason_filter": sorted(include_reasons) if include_reasons else [],
        "action_filter": sorted(include_actions) if include_actions else [],
        "summary": summary,
        "planned": len(items),
        "reason_counts": reason_counts,
        "action_counts": action_counts,
        "items": items,
    }


def _repair_priority(eval_row: pd.Series | dict[str, Any]) -> int:
    reasons = set(_split_reasons(eval_row.get("reason")))
    status = str(eval_row.get("status", ""))
    if status == "fail" and "missing_audio" in reasons:
        return 0
    if status == "fail" and "silent_or_tiny_audio" in reasons:
        return 1
    if status == "fail":
        return 2
    if "low_content_score" in reasons or "reference_leak" in reasons:
        return 3
    if "over_duration" in reasons:
        bucket = classify_over_duration(eval_row)
        if bucket == OVER_DURATION_SPEED_FIT:
            return 4
        if bucket == OVER_DURATION_REWRITE:
            return 5
        return 6
    if "speech_rate_fast" in reasons:
        return 5
    if "under_duration" in reasons:
        bucket = classify_under_duration(eval_row)
        if bucket == UNDER_DURATION_SLOW_FIT:
            return 4
        return 6
    if "start_drift" in reasons or "end_drift" in reasons:
        return 7
    return 9


def _audio_paths_for_row(row: pd.Series | dict[str, Any]) -> list[str]:
    number = int(row["number"])
    lines = normalize_lines(row.get("lines", row.get("text", ""))) or [str(row.get("text", ""))]
    paths: list[str] = []
    for line_index in range(len(lines)):
        paths.extend(
            [
                audio_file_for(number, line_index),
                temp_audio_file_for(number, line_index),
                os.path.join(TEMP_DIR, f"{number}_{line_index}_temp.wav"),
            ]
        )
    return paths


def _delete_audio_for_row(row: pd.Series | dict[str, Any]) -> None:
    for path in _audio_paths_for_row(row):
        if os.path.exists(path):
            os.remove(path)


def _ensure_columns(tasks_df: pd.DataFrame) -> pd.DataFrame:
    defaults = {
        "repair_action": "",
        "repair_reason": "",
        "repair_backend": "",
        "repair_attempts": 0,
        "repair_rewrite_rounds": 0,
        "repair_status": "",
        "rewritten_for_dubbing": False,
        "rewrite_reason": "",
        "dubbing_rewrite_rounds": 0,
        "speed_factor": 1.0,
        "keep_gaps": True,
    }
    for column, default in defaults.items():
        if column not in tasks_df.columns:
            tasks_df[column] = default
    return tasks_df


def _write_row(tasks_df: pd.DataFrame, idx: int, row_dict: dict[str, Any]) -> None:
    for key, value in row_dict.items():
        if key in tasks_df.columns:
            tasks_df.at[idx, key] = value


def _clear_asr_results(row_dict: dict[str, Any]) -> None:
    for column in ASR_RESULT_COLUMNS:
        if column in row_dict:
            row_dict[column] = None


def _same_lines(left: list[str], right: list[str]) -> bool:
    return [line.strip() for line in left] == [line.strip() for line in right]


def _repair_max_rewrite_rounds(quality) -> int:
    return max(0, int(load_key("dubbing_repair.max_rewrite_rounds", quality.max_rewrite_rounds)))


def _repair_rounds(row_dict: dict[str, Any]) -> int:
    return int(row_dict.get("repair_rewrite_rounds", 0) or 0)


def _finalize_selected_row(row: dict[str, Any]) -> list[list[float]]:
    from core.all_whisper_methods.audio_preprocess import get_audio_duration
    from core.dubbing_quality import get_quality_config
    from core.step10_gen_audio import adjust_audio_speed

    number = int(row["number"])
    lines = normalize_lines(row.get("lines", row.get("text", ""))) or [str(row.get("text", ""))]
    existing_times = parse_list(row.get("new_sub_times"))
    start = float(existing_times[0][0]) if existing_times else row_start_seconds(row)
    speed_factor = float(row.get("speed_factor", 1.0) or 1.0)
    real_dur = float(row.get("real_dur", 0) or 0)
    available = row_available_duration(row)
    projected_duration = real_dur / max(speed_factor, 0.1)
    if projected_duration > available > 0:
        required_speed = real_dur / available
        speed_factor = max(speed_factor, min(required_speed, get_quality_config().max_speed_factor))
        row["speed_factor"] = round(speed_factor, 3)
    new_times: list[list[float]] = []
    cur = start
    for line_index in range(len(lines)):
        temp_file = os.path.join(TEMP_DIR, f"{number}_{line_index}_temp.wav")
        output_file = os.path.join(SEGS_DIR, f"{number}_{line_index}.wav")
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        adjust_audio_speed(temp_file, output_file, speed_factor)
        duration = get_audio_duration(output_file)
        new_times.append([cur, cur + duration])
        cur += duration
    return new_times


def _speed_fit_existing_row(row: dict[str, Any]) -> tuple[list[list[float]], float]:
    from core.all_whisper_methods.audio_preprocess import get_audio_duration
    from core.dubbing_quality import get_quality_config
    from core.step10_gen_audio import adjust_audio_speed

    number = int(row["number"])
    lines = normalize_lines(row.get("lines", row.get("text", ""))) or [str(row.get("text", ""))]
    output_files = [audio_file_for(number, line_index) for line_index in range(len(lines))]
    if not all(os.path.exists(path) for path in output_files):
        raise FileNotFoundError(f"Cannot speed-fit row {number}; existing segment wav is missing")

    available = row_available_duration(row)
    current_duration = sum(get_audio_duration(path) for path in output_files)
    quality = get_quality_config()
    target_ratio = min(1.0, float(load_key("dubbing_quality.speed_fit_target_ratio", 0.995)))
    target_duration = max(available * target_ratio, 0.05)
    speed_factor = current_duration / target_duration if target_duration > 0 else 1.0
    speed_factor = max(1.0, round(speed_factor, 3))
    if speed_factor > quality.max_speed_factor:
        raise ValueError(
            f"Cannot speed-fit row {number}; required speed factor {speed_factor:.3f} "
            f"exceeds max_speed_factor {quality.max_speed_factor:.3f}"
        )

    Path(TEMP_DIR).mkdir(parents=True, exist_ok=True)
    row["speed_factor"] = speed_factor
    new_times: list[list[float]] = []
    cur = row_start_seconds(row)
    real_dur = 0.0
    for line_index, output_file in enumerate(output_files):
        temp_file = temp_audio_file_for(number, line_index)
        Path(temp_file).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(output_file, temp_file)
        adjust_audio_speed(temp_file, output_file, speed_factor)
        duration = get_audio_duration(output_file)
        real_dur += duration
        new_times.append([cur, cur + duration])
        cur += duration
    return new_times, real_dur


def _slow_fit_existing_row(row: dict[str, Any]) -> tuple[list[list[float]], float]:
    from core.all_whisper_methods.audio_preprocess import get_audio_duration
    from core.step10_gen_audio import adjust_audio_speed

    number = int(row["number"])
    lines = normalize_lines(row.get("lines", row.get("text", ""))) or [str(row.get("text", ""))]
    output_files = [audio_file_for(number, line_index) for line_index in range(len(lines))]
    if not all(os.path.exists(path) for path in output_files):
        raise FileNotFoundError(f"Cannot slow-fit row {number}; existing segment wav is missing")

    current_duration = sum(get_audio_duration(path) for path in output_files)
    target_duration = _under_duration_target(
        {
            "available_duration": row_available_duration(row),
            "final_audio_dur": current_duration,
            "real_dur": row.get("real_dur", current_duration),
            "speed_factor": row.get("speed_factor", 1.0),
        }
    )
    slow_factor = current_duration / target_duration if target_duration > 0 else 1.0
    slow_factor = min(1.0, round(slow_factor, 3))
    current_speed = float(row.get("speed_factor", 1.0) or 1.0)
    min_speed = float(load_key("dubbing_quality.slow_fit_min_speed", load_key("speed_factor.min", 0.8)))
    if current_speed * slow_factor < min_speed:
        raise ValueError(
            f"Cannot slow-fit row {number}; composite speed {current_speed * slow_factor:.3f} "
            f"is below slow_fit_min_speed {min_speed:.3f}"
        )

    existing_times = parse_list(row.get("new_sub_times"))
    start = float(existing_times[0][0]) if existing_times else row_start_seconds(row)
    Path(TEMP_DIR).mkdir(parents=True, exist_ok=True)
    row["speed_factor"] = round(current_speed * slow_factor, 3)
    new_times: list[list[float]] = []
    cur = start
    real_dur = 0.0
    for line_index, output_file in enumerate(output_files):
        temp_file = temp_audio_file_for(number, line_index)
        Path(temp_file).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(output_file, temp_file)
        adjust_audio_speed(temp_file, output_file, slow_factor)
        duration = get_audio_duration(output_file)
        real_dur += duration
        new_times.append([cur, cur + duration])
        cur += duration
    return new_times, real_dur


def _duration_rewrite_direction(row: dict[str, Any]) -> str:
    reasons = set(_split_reasons(row.get("repair_reason")))
    action = str(row.get("repair_action", ""))
    if "over_duration" in reasons or "speech_rate_fast" in reasons or action == "rewrite_and_regenerate":
        return "shorten"
    if "under_duration" in reasons or action == "expand_and_regenerate":
        return "expand"
    return "shorten"


def _duration_rewrite_reason(row: dict[str, Any], base_reason: str, direction: str = "shorten") -> str:
    real_dur = float(row.get("real_dur", 0) or 0)
    available = row_available_duration(row)
    ratio = real_dur / available if available > 0 and real_dur > 0 else 0.0
    detail = (
        f"{base_reason}. Measured TTS duration is {real_dur:.2f}s for a hard "
        f"{available:.2f}s window"
    )
    if ratio > 0:
        detail += f" ({ratio:.2f}x)."
    else:
        detail += "."
    if direction == "expand":
        return (
            f"{detail} The speech finishes too early. Expand naturally with context already implied by "
            "the source; keep numbers and entities unchanged, and do not invent new claims."
        )
    return (
        f"{detail} Compress more aggressively than a normal translation: keep only the essential meaning, "
        "numbers, and entities; remove filler and optional adjectives."
    )


def _rewrite_row(row_dict: dict[str, Any], rewrite_task_lines) -> bool:
    original_lines = normalize_lines(row_dict.get("lines", row_dict.get("text", "")))
    direction = _duration_rewrite_direction(row_dict)
    reason = _duration_rewrite_reason(
        row_dict,
        str(row_dict.get("repair_reason", "over_duration")),
        direction=direction,
    )
    rewritten = rewrite_task_lines(
        row_dict,
        reason=reason,
        direction=direction,
        max_retries=int(load_key("dubbing_repair.llm_retry_attempts", 1)),
        retry_interval=int(load_key("dubbing_repair.llm_retry_interval", 2)),
        timeout_seconds=float(load_key("dubbing_repair.llm_timeout_seconds", load_key("llm.timeout_seconds", 180))),
    )
    if not rewritten or _same_lines(original_lines, rewritten):
        row_dict["repair_status"] = "rewrite_failed"
        return False
    row_dict["lines"] = rewritten
    row_dict["text"] = " ".join(rewritten)
    row_dict["rewritten_for_dubbing"] = True
    row_dict["rewrite_reason"] = reason
    row_dict["dubbing_rewrite_rounds"] = int(row_dict.get("dubbing_rewrite_rounds", 0) or 0) + 1
    row_dict["repair_rewrite_rounds"] = _repair_rounds(row_dict) + 1
    row_dict["repair_status"] = "rewritten"
    return True


def apply_repair_plan(
    plan: dict[str, Any],
    *,
    dry_run: bool = True,
    full_remap: bool = False,
    tasks_path: str = TTS_TASKS_FILE,
) -> RepairApplySummary:
    tasks_df = _ensure_columns(pd.read_excel(tasks_path))
    items = list(plan.get("items", []))
    applied = 0
    skipped = 0

    if dry_run:
        return RepairApplySummary(
            planned=len(items),
            applied=0,
            skipped=0,
            dry_run=True,
            full_remap=full_remap,
            plan_path=str(plan.get("plan_path", DUBBING_REPAIR_PLAN_JSON)),
        )

    from core.dubbing_rewrite import rewrite_task_lines
    from core.dubbing_quality import get_quality_config
    from core.step10_gen_audio import merge_chunks, process_row

    Path(TEMP_DIR).mkdir(parents=True, exist_ok=True)
    Path(SEGS_DIR).mkdir(parents=True, exist_ok=True)

    for item in items:
        number = int(item["number"])
        matched = tasks_df[tasks_df["number"] == number]
        if matched.empty:
            skipped += 1
            continue

        idx = matched.index[0]
        row_dict = tasks_df.loc[idx].to_dict()
        row_dict["repair_action"] = item["action"]
        row_dict["repair_reason"] = ",".join(item.get("reasons", []))
        row_dict["repair_backend"] = item.get("backend") or ""
        row_dict["repair_attempts"] = int(row_dict.get("repair_attempts", 0) or 0) + 1

        if item.get("backend"):
            if "tts_backend" not in tasks_df.columns:
                tasks_df["tts_backend"] = ""
            row_dict["tts_backend"] = item["backend"]
            if "tts_method" in tasks_df.columns:
                row_dict["tts_method"] = ""

        quality = get_quality_config()
        max_repair_rounds = _repair_max_rewrite_rounds(quality)
        if item.get("rewrite"):
            if _repair_rounds(row_dict) >= max_repair_rounds:
                row_dict["repair_action"] = "manual_review"
                row_dict["repair_status"] = "rewrite_limit_reached"
                _write_row(tasks_df, idx, row_dict)
                skipped += 1
                continue
            if not _rewrite_row(row_dict, rewrite_task_lines):
                row_dict["repair_action"] = "manual_review"
                _write_row(tasks_df, idx, row_dict)
                skipped += 1
                continue

        if item.get("delete_audio"):
            _delete_audio_for_row(row_dict)

        if item.get("action") in {"regenerate", "regenerate_with_backend", "rewrite_and_regenerate", "expand_and_regenerate"}:
            _clear_asr_results(row_dict)
            row_dict["speed_factor"] = 1.0
            row_dict["new_sub_times"] = None
            _write_row(tasks_df, idx, row_dict)
            number, real_dur = process_row(row_dict, tasks_df)
            row_dict["real_dur"] = real_dur
            while (
                item.get("rewrite")
                and _repair_rounds(row_dict) < max_repair_rounds
            ):
                if actual_rewrite_needed(row_dict):
                    row_dict["repair_action"] = "rewrite_and_regenerate"
                    row_dict["repair_reason"] = "over_duration"
                elif actual_expand_needed(row_dict):
                    row_dict["repair_action"] = "expand_and_regenerate"
                    row_dict["repair_reason"] = "under_duration"
                else:
                    break
                if not _rewrite_row(row_dict, rewrite_task_lines):
                    row_dict["repair_action"] = "manual_review"
                    break
                _delete_audio_for_row(row_dict)
                row_dict["speed_factor"] = 1.0
                row_dict["new_sub_times"] = None
                _write_row(tasks_df, idx, row_dict)
                number, real_dur = process_row(row_dict, tasks_df)
                row_dict["real_dur"] = real_dur
            if full_remap:
                _write_row(tasks_df, idx, row_dict)
            else:
                row_dict["new_sub_times"] = _finalize_selected_row(row_dict)
                _write_row(tasks_df, idx, row_dict)
            tasks_df.loc[tasks_df["number"] == number, "real_dur"] = real_dur
            applied += 1
        elif item.get("action") == "speed_fit_existing_audio":
            _clear_asr_results(row_dict)
            row_dict["new_sub_times"], row_dict["real_dur"] = _speed_fit_existing_row(row_dict)
            _write_row(tasks_df, idx, row_dict)
            tasks_df.loc[tasks_df["number"] == number, "real_dur"] = row_dict["real_dur"]
            applied += 1
        elif item.get("action") == "slow_fit_existing_audio":
            _clear_asr_results(row_dict)
            row_dict["new_sub_times"], row_dict["real_dur"] = _slow_fit_existing_row(row_dict)
            _write_row(tasks_df, idx, row_dict)
            tasks_df.loc[tasks_df["number"] == number, "real_dur"] = row_dict["real_dur"]
            applied += 1
        elif item.get("action") == "timeline_remap" and full_remap:
            _write_row(tasks_df, idx, row_dict)
            applied += 1
        else:
            _write_row(tasks_df, idx, row_dict)
            skipped += 1

    if full_remap and applied:
        tasks_df = merge_chunks(tasks_df)

    tasks_df.to_excel(tasks_path, index=False)
    eval_summary = write_dubbing_eval(tasks_df)
    return RepairApplySummary(
        planned=len(items),
        applied=applied,
        skipped=skipped,
        dry_run=False,
        full_remap=full_remap,
        plan_path=str(plan.get("plan_path", DUBBING_REPAIR_PLAN_JSON)),
        eval_summary=eval_summary,
    )


def run_repair_batches(
    *,
    batch_limit: int | None,
    max_batches: int,
    backend_fallback: str = "auto",
    reasons: str | list[str] | set[str] | tuple[str, ...] | None = None,
    actions: str | list[str] | set[str] | tuple[str, ...] | None = None,
    full_remap: bool = False,
    history_path: str = DUBBING_REPAIR_HISTORY_JSONL,
) -> RepairBatchSummary:
    if max_batches < 1:
        raise ValueError("max_batches must be >= 1")

    runs: list[dict[str, Any]] = []
    total_applied = 0
    total_skipped = 0
    final_summary: dict[str, Any] | None = None
    stopped_reason = "max_batches_reached"

    for batch_index in range(1, max_batches + 1):
        plan = build_repair_plan(
            limit=batch_limit,
            backend_fallback=backend_fallback,
            reasons=reasons,
            actions=actions,
        )
        plan_path = write_repair_plan(plan)
        if plan["planned"] == 0:
            stopped_reason = "no_repair_items"
            final_summary = plan.get("summary")
            record = {
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "batch": batch_index,
                "plan_path": plan_path,
                "planned": 0,
                "reason_counts": {},
                "action_counts": {},
                "reason_filter": plan.get("reason_filter", []),
                "action_filter": plan.get("action_filter", []),
                "before": _compact_summary(plan.get("summary")),
                "after": _compact_summary(final_summary),
                "stopped_reason": stopped_reason,
            }
            append_repair_history(record, history_path)
            runs.append(record)
            break

        apply_summary = apply_repair_plan(plan, dry_run=False, full_remap=full_remap)
        total_applied += apply_summary.applied
        total_skipped += apply_summary.skipped
        final_summary = apply_summary.eval_summary
        record = {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "batch": batch_index,
            "batch_limit": batch_limit,
            "plan_path": plan_path,
            "planned": plan["planned"],
            "reason_counts": plan["reason_counts"],
            "action_counts": plan.get("action_counts", {}),
            "reason_filter": plan.get("reason_filter", []),
            "action_filter": plan.get("action_filter", []),
            "applied": apply_summary.applied,
            "skipped": apply_summary.skipped,
            "full_remap": full_remap,
            "before": _compact_summary(plan.get("summary")),
            "after": _compact_summary(final_summary),
        }
        append_repair_history(record, history_path)
        runs.append(record)

        if apply_summary.applied == 0:
            stopped_reason = "no_rows_applied"
            break

    return RepairBatchSummary(
        batches=len(runs),
        batch_limit=batch_limit,
        applied=total_applied,
        skipped=total_skipped,
        stopped_reason=stopped_reason,
        history_path=history_path,
        final_eval_summary=final_summary,
        runs=runs,
    )
