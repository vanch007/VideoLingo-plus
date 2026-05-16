from __future__ import annotations

import ast
import json
import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from core.config_utils import load_key
from core.constants import SEGS_DIR, TTS_TASKS_FILE
from core.dubbing_quality import normalize_lines
from core.providers.contracts import ASRVerificationResult
from core.providers.quality import content_similarity, reference_leak_score


@dataclass(frozen=True)
class ReadbackSummary:
    status: str
    checked: int
    skipped: int
    failed: int
    min_content_score: float | None
    max_leakage_score: float | None
    output_file: str


def extract_transcript(payload: str) -> str:
    """Extract transcript text from common ASR JSON or raw text outputs."""
    payload = (payload or "").strip()
    if not payload:
        return ""
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return payload

    if isinstance(data, str):
        return data.strip()
    if isinstance(data, dict):
        for key in ("text", "transcript", "result", "sentence"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        segments = data.get("segments") or data.get("chunks") or data.get("results")
        if isinstance(segments, list):
            parts = []
            for item in segments:
                if isinstance(item, dict):
                    text = item.get("text") or item.get("transcript") or item.get("sentence")
                    if text:
                        parts.append(str(text))
                elif isinstance(item, str):
                    parts.append(item)
            return " ".join(part.strip() for part in parts if part.strip())
    if isinstance(data, list):
        parts = []
        for item in data:
            if isinstance(item, dict):
                text = item.get("text") or item.get("transcript") or item.get("sentence")
                if text:
                    parts.append(str(text))
            elif isinstance(item, str):
                parts.append(item)
        return " ".join(part.strip() for part in parts if part.strip())
    return payload


def _command_template() -> list[str]:
    raw = load_key("dubbing_quality.asr_readback_command", [])
    if isinstance(raw, list):
        return [str(item) for item in raw]
    if isinstance(raw, str) and raw.strip():
        return shlex.split(raw)
    return []


def _format_command(template: list[str], *, audio_path: str, language: str) -> list[str]:
    return [
        part.format(audio=audio_path, language=language, output_dir=str(Path(audio_path).parent))
        for part in template
    ]


_BUILTIN_MODEL_CACHE: dict[tuple[str, bool], Any] = {}


def _readback_backend() -> str:
    backend = str(load_key("dubbing_quality.asr_readback_backend", "command") or "command").strip().lower()
    return backend or "command"


def _result_text(result: Any) -> str:
    text = getattr(result, "text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()
    segments = getattr(result, "segments", None)
    if isinstance(segments, list):
        parts = [str(getattr(segment, "text", "")).strip() for segment in segments]
        return " ".join(part for part in parts if part)
    if isinstance(result, (dict, list, str)):
        try:
            return extract_transcript(json.dumps(result, ensure_ascii=False))
        except TypeError:
            return extract_transcript(str(result))
    return extract_transcript(str(result))


def _get_builtin_model() -> tuple[Any, bool]:
    import stable_whisper

    model_name = str(load_key("dubbing_quality.asr_readback_model", load_key("whisper.model", "large-v3-turbo")))
    use_mlx = bool(load_key("dubbing_quality.asr_readback_use_mlx", True))
    cache_key = (model_name, use_mlx)
    if cache_key in _BUILTIN_MODEL_CACHE:
        return _BUILTIN_MODEL_CACHE[cache_key], use_mlx

    if use_mlx:
        model = stable_whisper.load_mlx_whisper(model_name)
    else:
        model = stable_whisper.load_model(model_name)
    _BUILTIN_MODEL_CACHE[cache_key] = model
    return model, use_mlx


def _run_builtin_asr(audio_path: str, language: str) -> ASRVerificationResult:
    """Run local stable-whisper readback in-process with cached model loading."""
    try:
        import librosa

        model, use_mlx = _get_builtin_model()
        audio, _ = librosa.load(audio_path, sr=16000, mono=True)
        options: dict[str, Any] = {"verbose": False}
        if language and language != "auto":
            options["language"] = language
        if not use_mlx:
            options.update({"word_timestamps": False, "vad": True, "regroup": True})
        result = model.transcribe(audio, **options)
        transcript = _result_text(result)
        return ASRVerificationResult(status="ok" if transcript else "fail", transcript=transcript)
    except Exception as exc:
        return ASRVerificationResult(status="fail", warnings=[f"builtin_asr_failed:{type(exc).__name__}:{exc}"])


def _run_asr(audio_path: str, language: str) -> ASRVerificationResult:
    backend = _readback_backend()
    template = _command_template()
    if backend in {"builtin", "stable_ts", "stable_ts_mlx"} or template == ["builtin"]:
        return _run_builtin_asr(audio_path, language)
    if not template:
        return ASRVerificationResult(status="skipped", warnings=["asr_readback_command_not_configured"])

    timeout = int(load_key("dubbing_quality.asr_readback_timeout_seconds", 120))
    cmd = _format_command(template, audio_path=audio_path, language=language)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()[-800:]
        return ASRVerificationResult(status="fail", warnings=[f"asr_command_failed:{detail}"])
    transcript = extract_transcript(result.stdout)
    return ASRVerificationResult(status="ok" if transcript else "fail", transcript=transcript)


def _parse_line_results(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(value)
                return parsed if isinstance(parsed, list) else []
            except (SyntaxError, ValueError):
                return []
    return []


def _audio_file_for(number: int, line_index: int) -> str:
    return os.path.join(SEGS_DIR, f"{number}_{line_index}.wav")


def verify_tasks_df(tasks_df: pd.DataFrame, *, limit: int | None = None, force: bool = False) -> tuple[pd.DataFrame, ReadbackSummary]:
    language = load_key("target_language", "auto")
    checked = 0
    skipped = 0
    failed = 0
    verified = 0
    min_content_score: float | None = None
    max_leakage_score: float | None = None
    out = tasks_df.copy()

    for column in ("asr_transcript", "asr_content_score", "asr_leakage_score", "asr_status", "asr_line_results"):
        if column not in out.columns:
            out[column] = None

    for idx, row in out.iterrows():
        if limit is not None and checked >= limit:
            break
        if not force and pd.notna(row.get("asr_content_score")):
            skipped += 1
            continue

        number = int(row["number"])
        lines = normalize_lines(row.get("lines", row.get("text", ""))) or [str(row.get("text", ""))]
        src_lines = normalize_lines(row.get("src_lines", row.get("origin", "")))
        line_results = []
        transcripts = []
        content_scores = []
        leak_scores = []

        for line_index, expected in enumerate(lines):
            audio_path = _audio_file_for(number, line_index)
            if not os.path.exists(audio_path):
                failed += 1
                line_results.append({
                    "line_index": line_index,
                    "status": "fail",
                    "audio_path": audio_path,
                    "warning": "missing_audio",
                })
                continue

            result = _run_asr(audio_path, language)
            checked += 1
            if result.status == "skipped":
                skipped += 1
                line_results.append({
                    "line_index": line_index,
                    "status": "skipped",
                    "audio_path": audio_path,
                    "warning": ",".join(result.warnings),
                })
                continue
            if result.status != "ok" or not result.transcript:
                failed += 1
                line_results.append({
                    "line_index": line_index,
                    "status": "fail",
                    "audio_path": audio_path,
                    "warning": ",".join(result.warnings),
                })
                continue

            transcript = result.transcript
            verified += 1
            ref_text = src_lines[line_index] if line_index < len(src_lines) else row.get("origin", "")
            content_score = content_similarity(expected, transcript)
            leak_score = reference_leak_score(ref_text, transcript)
            transcripts.append(transcript)
            content_scores.append(content_score)
            leak_scores.append(leak_score)
            min_content_score = content_score if min_content_score is None else min(min_content_score, content_score)
            max_leakage_score = leak_score if max_leakage_score is None else max(max_leakage_score, leak_score)
            line_results.append({
                "line_index": line_index,
                "status": "ok",
                "audio_path": audio_path,
                "expected": expected,
                "transcript": transcript,
                "content_score": content_score,
                "leakage_score": leak_score,
            })

        if content_scores:
            out.at[idx, "asr_content_score"] = min(content_scores)
            out.at[idx, "asr_leakage_score"] = max(leak_scores) if leak_scores else 0.0
            out.at[idx, "asr_transcript"] = " ".join(transcripts)
            out.at[idx, "asr_status"] = "ok"
        elif line_results:
            statuses = {item.get("status") for item in line_results}
            out.at[idx, "asr_status"] = "skipped" if statuses == {"skipped"} else "fail"
        out.at[idx, "asr_line_results"] = json.dumps(line_results, ensure_ascii=False)

    status = "ok"
    if failed:
        status = "fail"
    elif verified == 0:
        status = "skipped"
    return out, ReadbackSummary(
        status=status,
        checked=checked,
        skipped=skipped,
        failed=failed,
        min_content_score=min_content_score,
        max_leakage_score=max_leakage_score,
        output_file=TTS_TASKS_FILE,
    )


def verify_tasks_file(path: str = TTS_TASKS_FILE, *, limit: int | None = None, force: bool = False) -> ReadbackSummary:
    tasks_df = pd.read_excel(path)
    out, summary = verify_tasks_df(tasks_df, limit=limit, force=force)
    out.to_excel(path, index=False)
    return summary
