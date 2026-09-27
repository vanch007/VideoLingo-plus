"""Separate speaker identity from source-row delivery for local dubbing.

No character names or external scripts are used. Plans describe conditioning
inputs; they deliberately do not claim perceptual voice/emotion quality.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from core.config_utils import load_key
from core.timing_utils import srt_time_to_seconds
from core.all_tts_functions.tts_utils import get_reference_audio_path

SCHEMA_VERSION = 1
_HASH_CACHE = {}


def clean_value(value):
    if value is None:
        return None
    text = str(value).strip()
    return None if text.lower() in {"", "nan", "none", "<na>"} else text


def audio_identity(path):
    """Content identity cached by stat; used for plans and resume invalidation."""
    if not clean_value(path):
        return None
    file = Path(str(path)).expanduser().resolve()
    if not file.is_file():
        return {"path": str(file), "missing": True}
    stat = file.stat()
    key = (str(file), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    if key not in _HASH_CACHE:
        _HASH_CACHE[key] = hashlib.sha256(file.read_bytes()).hexdigest()
    return {"path": str(file), "sha256": _HASH_CACHE[key]}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def validate_emotion_alpha(value):
    alpha = float(value)
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("emo_alpha must be finite and between 0 and 1")
    return alpha


def _source_emotion_clip(row, tasks, root, *, materialize=True):
    """Cut the original utterance without the speaker-reference +/-1s padding."""
    start = srt_time_to_seconds(str(row["start_time"]))
    end = srt_time_to_seconds(str(row["end_time"]))
    if not math.isfinite(start + end) or start < 0 or end <= start:
        raise ValueError(f"Invalid emotion source window for row {row['number']}")
    # A dubbed window is not permission to borrow neighbouring delivery.
    if tasks is not None:
        for _, other in tasks.iterrows():
            if int(other["number"]) == int(row["number"]):
                continue
            a = srt_time_to_seconds(str(other["start_time"]))
            b = srt_time_to_seconds(str(other["end_time"]))
            if min(end, b) - max(start, a) > 0.01:
                raise ValueError(f"Overlapping source rows: cannot isolate emotion for row {row['number']}")
    source = root / "vocal.wav"
    if not source.is_file():
        source = root / "vocal.mp3"
    if not source.is_file():
        raise FileNotFoundError("Separated vocal stem required for source-row emotion cloning")
    spec = {"source": audio_identity(source), "start": start, "end": end, "schema": SCHEMA_VERSION}
    destination = root / "emotion_refers" / f"{int(row['number'])}_{_digest(spec)[:16]}.wav"
    if materialize and not destination.is_file():
        # Seek only the requested slice rather than decode the whole film for every row.
        with sf.SoundFile(source) as handle:
            sr = handle.samplerate
            if end > len(handle) / sr + 1 / sr:
                raise ValueError(f"Emotion window exceeds source audio for row {row['number']}")
            handle.seek(round(start * sr))
            samples = handle.read(round(end * sr) - round(start * sr), dtype="float32", always_2d=True)
        if not len(samples) or float(np.max(np.abs(samples))) < 1e-5:
            raise ValueError(f"Empty/silent emotion reference for row {row['number']}")
        min_samples = int(0.2 * sr)
        if len(samples) < min_samples:
            pad_len = min_samples - len(samples)
            samples = np.pad(samples, ((0, pad_len), (0, 0)), mode="edge")
        destination.parent.mkdir(parents=True, exist_ok=True)
        sf.write(destination, samples, sr, subtype="PCM_16")
    return str(destination.resolve()), spec


def resolve_reference_plan(row, tasks, *, audio_root="output/audio", materialize=True):
    root = Path(audio_root)
    number = int(row["number"])
    speaker = clean_value(row.get("speaker"))
    warnings = []
    ref = clean_value(row.get("ref_audio"))
    ref_number = number
    strategy = str(load_key("mlx_tts.reference_strategy", "row"))
    cfg = dict(load_key("mlx_tts.backends.indextts2", {}) or {})
    workflow_method = load_key("tts_method", "mlx_router")
    method = clean_value(row.get("tts_method")) or workflow_method
    explicit_emotion = clean_value(row.get("emotion_ref")) or clean_value(row.get("emotion_ref_audio"))
    policy = {"strategy": strategy}
    short_or_fragment = False
    if strategy == "hybrid":
        threshold = float(load_key("mlx_tts.hybrid_short_threshold_seconds", 2.0))
        if not math.isfinite(threshold) or threshold <= 0:
            raise ValueError("hybrid_short_threshold_seconds must be finite and positive")
        duration = srt_time_to_seconds(str(row["end_time"])) - srt_time_to_seconds(str(row["start_time"]))
        fragment = str(clean_value(row.get("reference_is_fragment")) or "").lower() in {"true", "1", "yes"}
        min_units = int(load_key("mlx_tts.hybrid_min_source_units", 4))
        if min_units < 1:
            raise ValueError("hybrid_min_source_units must be positive")
        source_text = clean_value(row.get("origin")) or ""
        tokens = re.findall(r"[\u4e00-\u9fff]|[^\W_]+", source_text.lower())
        # Count Han characters / other-language words, collapsing consecutive
        # repeats. A long held 'ah' still provides very little identity evidence.
        units = sum(index == 0 or token != tokens[index - 1] for index, token in enumerate(tokens))
        short_text = bool(tokens) and units < min_units
        short_or_fragment = duration < threshold or fragment or short_text
        policy.update(source_duration=duration, short_threshold_seconds=threshold, is_fragment=fragment,
                      source_units=units if tokens else None, min_source_units=min_units,
                      decision_reason="fragment" if fragment else "short_duration" if duration < threshold
                      else "short_source_text" if short_text else "long_source_utterance")
    # Full source utterances supply both identity and delivery. Resolve the
    # source slice first; do not require an unrelated speaker anchor for them.
    shared_source = (
        strategy == "hybrid" and not short_or_fragment and not ref and not explicit_emotion
        and cfg.get("emotion_reference_strategy") == "source_row"
        and (method in {"mlx_indextts2", "mlx_router"} or workflow_method in {"mlx_indextts2", "mlx_router"})
    )
    selected_strategy = ("speaker_anchor" if short_or_fragment else "row") if strategy == "hybrid" else strategy
    if not ref and not shared_source:
        if speaker is None or speaker.lower() in {"unknown", "unassigned"}:
            # Unknown is not the first known actor. Only its own source can be used.
            own = root / "refers" / f"{number}.wav"
            if not own.is_file():
                raise ValueError(f"Row {number} has unknown speaker and no own reference")
            ref = str(own)
            selected_strategy = "own_row_unknown"
            warnings.append("speaker_identity_unknown")
        else:
            ref, fallback = get_reference_audio_path(
                number, refers_dir=str(root / "refers"), task_df=tasks, speaker=speaker,
                prefer_speaker_anchor=selected_strategy == "speaker_anchor",
                speaker_anchor_target_ms=int(float(load_key("mlx_tts.speaker_anchor_target_seconds", 6.0)) * 1000),
                strict_speaker_match=True,
            )
            ref_number = fallback if fallback is not None else number
    emotion_strategy = cfg.get("emotion_reference_strategy", "explicit_only")
    emotion = explicit_emotion
    source_window = None
    emotion_source = "explicit" if emotion else "speaker_reference"
    requires_emotion = (
        (method in {"mlx_indextts2", "mlx_router"} or workflow_method in {"mlx_indextts2", "mlx_router"})
        and emotion_strategy == "source_row"
    )
    if not emotion and requires_emotion:
        emotion, source_window = _source_emotion_clip(row, tasks, root, materialize=materialize)
        emotion_source = "source_row"
        duration = source_window["end"] - source_window["start"]
        if duration < 0.75:
            warnings.append("short_emotion_reference_review_required")
        if duration > 8:
            warnings.append("long_row_may_average_emotion_changes")
    if emotion and materialize and not Path(emotion).expanduser().is_file():
        raise FileNotFoundError(f"Emotion reference missing: {emotion}")
    if shared_source:
        if not emotion:
            raise ValueError("Hybrid long utterances require their own source reference")
        ref = emotion
        selected_strategy = "row_shared"
        emotion_source = "row_shared_reference"
        if speaker is None or speaker.lower() in {"unknown", "unassigned"}:
            warnings.append("speaker_identity_unknown")
    if materialize and not Path(ref).expanduser().is_file():
        raise FileNotFoundError(f"Speaker reference missing: {ref}")
    alpha = validate_emotion_alpha(clean_value(row.get("emo_alpha")) or cfg.get("emo_alpha", 1.0))
    plan = {
        "schema_version": SCHEMA_VERSION, "number": number, "speaker": speaker,
        "speaker_reference": audio_identity(ref), "speaker_reference_number": ref_number,
        "speaker_strategy": "explicit" if clean_value(row.get("ref_audio")) else selected_strategy,
        "reference_policy": policy,
        "emotion_reference": audio_identity(emotion), "emotion_source": emotion_source,
        "emotion_source_window": source_window, "emo_alpha": alpha,
        "requires_separate_emotion": bool(emotion or requires_emotion),
        "backend_config_fingerprint": _digest(cfg),
        "warnings": warnings,
        "voice_similarity": "missing evidence", "emotion_fidelity": "missing evidence",
    }
    plan["fingerprint"] = _digest(plan)
    return plan


def prepare_reference_plan(tasks, *, audio_root="output/audio"):
    """Resolve before checking generation caches. Never rewrite manual overrides."""
    tasks = tasks.copy()
    plans = []
    for idx, row in tasks.iterrows():
        plan = resolve_reference_plan(row.to_dict(), tasks, audio_root=audio_root)
        plans.append(plan)
        tasks.at[idx, "reference_plan_fingerprint"] = plan["fingerprint"]
        tasks.at[idx, "resolved_speaker_ref"] = plan["speaker_reference"]["path"]
        tasks.at[idx, "resolved_emotion_ref"] = (plan["emotion_reference"] or {}).get("path", "")
    manifest = Path(audio_root) / "reference_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({
        "schema_version": SCHEMA_VERSION, "stage": "planned", "rows": plans,
        "note": "Conditioning provenance only; perceptual voice/emotion acceptance is separate.",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return tasks


def reference_plan_is_current(tasks):
    """Read-only check used by CLI/UI resume; never create clips during status."""
    if "reference_plan_fingerprint" not in tasks.columns or tasks.empty:
        return False
    try:
        return all(
            row["reference_plan_fingerprint"] == resolve_reference_plan(
                row.to_dict(), tasks, materialize=False
            )["fingerprint"] for _, row in tasks.iterrows()
        )
    except (ValueError, KeyError, OSError):
        return False


def write_conditioning_receipt(request, backend: str | None = None):
    """A receipt exists only after successful synthesis, adjacent to its WAV."""
    import shutil
    path = Path(request.output_path).resolve()
    plan = request.metadata.get("reference_plan")
    effective_backend = backend if backend is not None else request.metadata.get("backend")
    if plan and path.is_file():
        # Preserve an immutable copy of the raw native output audio
        native_path = path.with_name(f"{path.stem}.native{path.suffix}")
        try:
            shutil.copyfile(path, native_path)
            native_audio_id = audio_identity(native_path)
        except Exception:
            native_audio_id = audio_identity(path)
        current_audio_id = audio_identity(path)
        receipt = {
            "reference_plan": plan,
            "text": request.text,
            "language": request.language,
            "target_duration": request.target_duration,
            "backend": effective_backend,
            "native_audio": native_audio_id,
            "transform_history": [],
            "audio": current_audio_id,
        }
        path.with_suffix(".conditioning.json").write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8"
        )


def refresh_conditioning_receipt_audio(output_path: str | Path, transform: str = "post_processing") -> None:
    """Update receipt audio hash while preserving native audio hash and transformation lineage."""
    path = Path(output_path).resolve()
    receipt_path = path.with_suffix(".conditioning.json")
    if path.is_file() and receipt_path.is_file():
        try:
            data = json.loads(receipt_path.read_text(encoding="utf-8"))
            current_id = audio_identity(path)
            prev_audio = data.get("audio")
            if "native_audio" not in data and prev_audio:
                data["native_audio"] = prev_audio
            if prev_audio and prev_audio.get("sha256") != current_id.get("sha256"):
                history = data.setdefault("transform_history", [])
                history.append({
                    "stage": transform,
                    "input_audio": prev_audio,
                    "output_audio": current_id,
                    "timestamp": time.time(),
                })
            data["audio"] = current_id
            receipt_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass


def conditioning_receipt_matches(request):
    path = Path(request.output_path).resolve()
    try:
        receipt = json.loads(path.with_suffix(".conditioning.json").read_text(encoding="utf-8"))
        return (
            path.is_file()
            and receipt["audio"] == audio_identity(path)
            and receipt["reference_plan"]["fingerprint"] == request.metadata["reference_plan"]["fingerprint"]
            and receipt["text"] == request.text
            and receipt["language"] == request.language
            and receipt["target_duration"] == request.target_duration
            and receipt.get("backend") == request.metadata.get("backend")
        )
    except (OSError, ValueError, KeyError):
        return False
