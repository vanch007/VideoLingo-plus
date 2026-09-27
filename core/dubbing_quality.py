import ast
import json
import math
import os
import re
from pathlib import Path
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
            # Pandas/numpy scalars stored inside object columns may be written
            # to Excel as ``np.float64(1.23)``. Strip only that known wrapper
            # before literal evaluation; never use eval on workbook content.
            normalized = re.sub(
                r"(?:np\.)?float(?:16|32|64)?\(\s*([-+0-9.eE]+)\s*\)",
                r"\1",
                value,
            )
            parsed = ast.literal_eval(normalized)
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
    avail = row_available_duration(row)
    if real_dur <= 0 or avail <= 0:
        return False
    if avail <= 0.6:
        short_limit = float(load_key("dubbing_quality.short_utterance_max_speed_factor", 1.35))
        return real_dur > avail * short_limit
    return real_dur > avail * quality.rewrite_actual_ratio


def natural_speed_rewrite_needed(row: pd.Series | dict) -> bool:
    quality = get_quality_config()
    if not quality.enabled:
        return False
    real_dur = float(row.get("real_dur", 0) or 0)
    avail = row_available_duration(row)
    if real_dur <= 0 or avail <= 0:
        return False
    limit = float(quality.max_natural_speed_factor)
    if avail <= 0.6:
        short_limit = float(load_key("dubbing_quality.short_utterance_max_speed_factor", 1.35))
        limit = max(limit, short_limit)
    return real_dur > avail * limit


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



def measure_acoustic_activity(audio_file: str, base_offset_seconds: float = 0.0) -> dict[str, float | None]:
    """Measure acoustic onset and offset in seconds using local RMS energy."""
    try:
        from pydub import AudioSegment
        from pydub.silence import detect_nonsilent
        if not os.path.exists(audio_file) or os.path.getsize(audio_file) < 100:
            return {"onset": None, "offset": None, "tail_padding": None}
        audio = AudioSegment.from_file(audio_file)
        if len(audio) == 0:
            return {"onset": None, "offset": None, "tail_padding": None}
        thresh = min(-35, math.floor(audio.dBFS - 12)) if math.isfinite(audio.dBFS) else -35
        spans = detect_nonsilent(audio, min_silence_len=80, silence_thresh=thresh, seek_step=10)
        if spans:
            onset = round(base_offset_seconds + spans[0][0] / 1000.0, 3)
            offset = round(base_offset_seconds + spans[-1][1] / 1000.0, 3)
            tail_padding = round((len(audio) - spans[-1][1]) / 1000.0, 3)
            return {"onset": onset, "offset": offset, "tail_padding": tail_padding}
    except Exception:
        pass
    return {"onset": None, "offset": None, "tail_padding": None}


def measure_voice_similarity(ref_audio: str, gen_audio: str) -> float | None:
    try:
        import librosa
        import numpy as np
        from scipy.spatial.distance import cosine
        if not os.path.exists(ref_audio) or not os.path.exists(gen_audio):
            return None
        y_ref, sr_ref = librosa.load(ref_audio, sr=16000)
        y_gen, sr_gen = librosa.load(gen_audio, sr=16000)
        if len(y_ref) < 1000 or len(y_gen) < 1000:
            return None
        mfcc_ref = np.mean(librosa.feature.mfcc(y=y_ref, sr=sr_ref, n_mfcc=20), axis=1)
        mfcc_gen = np.mean(librosa.feature.mfcc(y=y_gen, sr=sr_gen, n_mfcc=20), axis=1)
        sim = float(1.0 - cosine(mfcc_ref, mfcc_gen))
        return round(sim, 4) if math.isfinite(sim) else None
    except Exception:
        return None


def measure_emotion_fidelity(emotion_ref_audio: str, gen_audio: str) -> float | None:
    try:
        import librosa
        import numpy as np
        if not os.path.exists(emotion_ref_audio) or not os.path.exists(gen_audio):
            return None
        y_ref, sr = librosa.load(emotion_ref_audio, sr=16000)
        y_gen, _ = librosa.load(gen_audio, sr=16000)
        rms_ref = float(np.std(librosa.feature.rms(y=y_ref)))
        rms_gen = float(np.std(librosa.feature.rms(y=y_gen)))
        ratio = min(rms_ref, rms_gen) / max(rms_ref, rms_gen, 1e-6)
        return round(float(ratio), 4) if math.isfinite(ratio) else None
    except Exception:
        return None

def evaluate_dubbing(tasks_df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    quality = get_quality_config()
    if tasks_df is None:
        tasks_df = pd.read_excel(TTS_TASKS_FILE)

    rows = []
    for _, row in tasks_df.iterrows():
        number = int(row["number"])
        lines = normalize_lines(row.get("lines", row.get("text", ""))) or [str(row.get("text", ""))]
        new_times = _flatten_times(row, "new_sub_times")
        source_start = row_start_seconds(row)
        source_end = row_end_seconds(row)
        tolerance = float(row.get("tolerance", 0) or 0)
        target_start = source_start
        target_end = source_end + tolerance
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

        acoustic_onset = None
        acoustic_offset = None
        acoustic_tail_padding = None
        if line_durations and not missing_files and actual_start is not None:
            first_audio = audio_file_for(number, 0)
            last_audio = audio_file_for(number, len(lines) - 1)
            act0 = measure_acoustic_activity(first_audio, actual_start)
            last_win_start = new_times[-1][0] if new_times else actual_start
            actN = measure_acoustic_activity(last_audio, last_win_start)
            acoustic_onset = act0.get("onset")
            acoustic_offset = actN.get("offset")
            acoustic_tail_padding = actN.get("tail_padding")

        spk_ref = row.get("resolved_speaker_ref")
        emo_ref = row.get("resolved_emotion_ref")
        first_audio = audio_file_for(number, 0)
        voice_sim = measure_voice_similarity(str(spk_ref), first_audio) if spk_ref and pd.notna(spk_ref) else None
        emo_fid = measure_emotion_fidelity(str(emo_ref), first_audio) if emo_ref and pd.notna(emo_ref) else None

        acoustic_start_drift = None if acoustic_onset is None else round(acoustic_onset - target_start, 3)
        # End drift relative to source speech end:
        acoustic_end_drift = None if acoustic_offset is None else round(acoustic_offset - source_end, 3)
        # End drift relative to task window end (with tolerance):
        acoustic_window_drift = None if acoustic_offset is None else round(acoustic_offset - target_end, 3)

        native_onset = None
        native_durations = []
        has_native_files = True
        first_native_audio = None
        for line_index, _ in enumerate(lines):
            audio_file = audio_file_for(number, line_index)
            native_candidate = Path(audio_file).with_name(f"{Path(audio_file).stem}.native{Path(audio_file).suffix}")
            if not native_candidate.is_file():
                rec_file = Path(audio_file).with_suffix(".conditioning.json")
                if rec_file.is_file():
                    try:
                        rec_data = json.loads(rec_file.read_text(encoding="utf-8"))
                        n_path = rec_data.get("native_audio", {}).get("path")
                        if n_path and os.path.isfile(n_path):
                            native_candidate = Path(n_path)
                    except Exception:
                        pass
            if native_candidate.is_file():
                native_durations.append(get_audio_duration(str(native_candidate)))
                if line_index == 0:
                    first_native_audio = str(native_candidate)
            else:
                has_native_files = False
                break

        if has_native_files and first_native_audio and actual_start is not None:
            act_native = measure_acoustic_activity(first_native_audio, actual_start)
            native_onset = act_native.get("onset")
            native_duration = round(sum(native_durations), 3) if native_durations else None
        else:
            native_onset = None
            native_duration = None

        four_layer_timing = {
            "layer1_source": {"onset": source_start, "offset": source_end, "duration": round(source_end - source_start, 3)},
            "layer2_native_take": {"onset": native_onset, "duration": native_duration},
            "layer3_processed_segment": {"onset": acoustic_onset, "offset": acoustic_offset, "tail_padding": acoustic_tail_padding, "duration": final_audio_dur, "speed_factor": speed_factor},
            "layer4_timeline_window": {"target_start": target_start, "target_end": target_end, "actual_start": actual_start, "actual_end": actual_end, "start_drift": start_drift, "end_drift": end_drift},
        }

        status = "ok"
        reasons = []
        if missing_files:
            status = "fail"
            reasons.append("missing_audio")
        if silent_fallbacks:
            status = "fail"
            reasons.append("silent_or_tiny_audio")
        ends_too_early = end_drift is not None and end_drift < -quality.max_early_end_drift
        acoustic_ends_too_early = (
            acoustic_end_drift is not None and acoustic_end_drift < -quality.max_early_end_drift
        )
        acoustic_ends_too_late = (
            (acoustic_end_drift is not None and acoustic_end_drift > quality.max_end_drift)
            or (acoustic_window_drift is not None and acoustic_window_drift > quality.max_end_drift)
        )
        if duration_ratio > quality.max_duration_ratio:
            status = "warn" if status == "ok" else status
            reasons.append("over_duration")
        if (0 < duration_ratio < quality.min_duration_ratio) or ends_too_early or acoustic_ends_too_early:
            status = "warn" if status == "ok" else status
            reasons.append("under_duration")
        if (start_drift is not None and abs(start_drift) > quality.max_start_drift) or (
            acoustic_start_drift is not None and abs(acoustic_start_drift) > quality.max_start_drift
        ):
            status = "warn" if status == "ok" else status
            reasons.append("start_drift")
        if ((end_drift is not None and end_drift > quality.max_end_drift) or acoustic_ends_too_late) and "over_duration" not in reasons:
            status = "warn" if status == "ok" else status
            reasons.append("over_duration")
        row_tolerance = float(row.get("tolerance", 0.0) or 0.0)
        unexplained_tail_padding = max(0.0, (acoustic_tail_padding or 0.0) - row_tolerance)
        if unexplained_tail_padding > 0.5 and (
            final_audio_dur > 0 and (unexplained_tail_padding / final_audio_dur) > 0.25
        ):
            status = "warn" if status == "ok" else status
            if "acoustic_tail_padding" not in reasons:
                reasons.append("acoustic_tail_padding")
        if speed_factor > quality.max_natural_speed_factor:
            status = "warn" if status == "ok" else status
            reasons.append("speech_rate_fast")
        if content_score is not None and pd.notna(content_score) and float(content_score) < quality.content_score_min:
            status = "warn" if status == "ok" else status
            reasons.append("low_content_score")
        if leakage_score is not None and pd.notna(leakage_score) and float(leakage_score) > quality.leak_score_max:
            status = "warn" if status == "ok" else status
            reasons.append("reference_leak")

        repair_action = str(row.get("repair_action", "") or "")
        repair_status = str(row.get("repair_status", "") or "")
        manual_review_reason = ",".join(reasons)
        if status == "warn" and reasons and repair_action == "manual_review":
            reasons = ["manual_review"]

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
            "source_end": source_end,
            "acoustic_onset": acoustic_onset,
            "acoustic_offset": acoustic_offset,
            "acoustic_start_drift": acoustic_start_drift,
            "acoustic_end_drift": acoustic_end_drift,
            "acoustic_window_drift": acoustic_window_drift,
            "acoustic_tail_padding": acoustic_tail_padding,
            "voice_similarity": voice_sim,
            "emotion_fidelity": emo_fid,
            "four_layer_timing": four_layer_timing,
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
            "repair_action": repair_action,
            "repair_status": repair_status,
            "manual_review_reason": manual_review_reason if reasons == ["manual_review"] else "",
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
        "max_abs_acoustic_start_drift": safe_float(pd.to_numeric(eval_df["acoustic_start_drift"], errors="coerce").abs().max()) if "acoustic_start_drift" in eval_df else 0.0,
        "max_abs_acoustic_end_drift": safe_float(pd.to_numeric(eval_df["acoustic_end_drift"], errors="coerce").abs().max()) if "acoustic_end_drift" in eval_df else 0.0,
        "max_acoustic_tail_padding": safe_float(pd.to_numeric(eval_df["acoustic_tail_padding"], errors="coerce").max()) if "acoustic_tail_padding" in eval_df else 0.0,
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
    # Readback/timing success does not establish identity or emotional fidelity.
    summary["quality_gate"]["scope"] = "content_timing_audio_integrity"
    voice_sims = [s for s in eval_df.get("voice_similarity", []) if pd.notna(s) and isinstance(s, (int, float))]
    emo_fids = [s for s in eval_df.get("emotion_fidelity", []) if pd.notna(s) and isinstance(s, (int, float))]
    avg_voice_sim = safe_float(pd.Series(voice_sims).mean()) if voice_sims else None
    avg_emo_fid = safe_float(pd.Series(emo_fids).mean()) if emo_fids else None
    summary["perceptual_quality"] = {
        "voice_similarity": f"acoustic_voiceprint_similarity: {avg_voice_sim:.4f}" if avg_voice_sim is not None else "missing evidence",
        "emotion_fidelity": f"prosodic_energy_fidelity: {avg_emo_fid:.4f}" if avg_emo_fid is not None else "missing evidence",
        "avg_voice_similarity": avg_voice_sim,
        "avg_emotion_fidelity": avg_emo_fid,
        "independent_speaker_embedding": "missing evidence",
        "independent_emotion_classification": "missing evidence",
        "status": "perceptual_pending",
        "acceptance_note": "Acoustic MFCC similarity and RMS energy dynamics measured as proxy; calibrated independent speaker embedding and emotion classification remain pending.",
    }
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
