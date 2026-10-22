from __future__ import annotations

import ast
import hashlib
import json
import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from core.config_utils import load_key
from core.constants import AUDIO_DIR, SEGS_DIR, TTS_TASKS_FILE
from core.dubbing_quality import normalize_lines
from core.runtime_context import effective_target_language
from core.providers.contracts import ASRVerificationResult
from core.providers.quality import content_similarity, reference_leak_score
from core.pipeline.artifact_manifest import hash_path


@dataclass(frozen=True)
class ReadbackSummary:
    status: str
    checked: int
    skipped: int
    failed: int
    cached: int
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


def _safe_load_key(key: str, default: Any = None) -> Any:
    try:
        return load_key(key, default)
    except (FileNotFoundError, KeyError):
        return default


def _format_command(template: list[str], *, audio_path: str, language: str) -> list[str]:
    return [
        part.format(audio=audio_path, language=language, output_dir=str(Path(audio_path).parent))
        for part in template
    ]


_BUILTIN_MODEL_CACHE: dict[tuple[str, bool], Any] = {}


def _readback_backend() -> str:
    backend = str(_safe_load_key("dubbing_quality.asr_readback_backend", "command") or "command").strip().lower()
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


def _get_builtin_model(model_name: str | None = None) -> tuple[Any, bool]:
    import stable_whisper

    model_name = model_name or str(load_key("dubbing_quality.asr_readback_model", load_key("whisper.model", "large-v3-turbo")))
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


def _run_builtin_asr(audio_path: str, language: str, *, model_name: str | None = None) -> ASRVerificationResult:
    """Run local stable-whisper readback in-process with cached model loading."""
    try:
        model_name = model_name or str(load_key("dubbing_quality.asr_readback_model", load_key("whisper.model", "large-v3-turbo")))
        use_mlx = bool(load_key("dubbing_quality.asr_readback_use_mlx", True))
        if use_mlx:
            import mlx_whisper
            options: dict[str, Any] = {}
            if language and language != "auto":
                options["language"] = language
            result = mlx_whisper.transcribe(str(audio_path), path_or_hf_repo=model_name, **options)
            transcript = _result_text(result)
            return ASRVerificationResult(status="ok" if transcript else "fail", transcript=transcript)

        import librosa
        model, _ = _get_builtin_model(model_name)
        audio, _ = librosa.load(audio_path, sr=16000, mono=True)
        options = {"verbose": False}
        if language and language != "auto":
            options["language"] = language
            options.update({"word_timestamps": False, "vad": True, "regroup": True})
        result = model.transcribe(audio, **options)
        transcript = _result_text(result)
        return ASRVerificationResult(status="ok" if transcript else "fail", transcript=transcript)
    except Exception as exc:
        return ASRVerificationResult(status="fail", warnings=[f"builtin_asr_failed:{type(exc).__name__}:{exc}"])


def _run_moss_asr(audio_path: str, language: str) -> ASRVerificationResult:
    """Run the same local MOSS route used by source transcription for TTS readback."""
    try:
        from core.providers.moss_asr import run_moss_asr, transcript_from_segments

        run = run_moss_asr(audio_path, purpose="readback")
        transcript = transcript_from_segments(run.segments)
        warnings = [] if transcript else [f"moss_asr_empty:{run.output_dir}"]
        return ASRVerificationResult(
            status="ok" if transcript else "fail",
            transcript=transcript,
            warnings=warnings,
        )
    except Exception as exc:
        return ASRVerificationResult(status="fail", warnings=[f"moss_asr_failed:{type(exc).__name__}:{exc}"])


def _run_asr(audio_path: str, language: str) -> ASRVerificationResult:
    backend = _readback_backend()
    template = _command_template()
    if backend in {"moss", "moss-mlx", "mlx-moss"}:
        return _run_moss_asr(audio_path, language)
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
    temp_path = os.path.join(AUDIO_DIR, "temp", f"{number}_{line_index}_temp.wav")
    seg_path = os.path.join(SEGS_DIR, f"{number}_{line_index}.wav")
    if os.path.exists(temp_path):
        if not os.path.exists(seg_path) or os.path.getmtime(temp_path) >= os.path.getmtime(seg_path):
            return temp_path
    return os.path.join(SEGS_DIR, f"{number}_{line_index}.wav")


def _audio_signature(path: str) -> dict[str, Any]:
    try:
        stat = os.stat(path)
    except OSError:
        return {"path": path, "exists": False}
    return {
        "path": path,
        "exists": True,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _asr_config_signature(language: str) -> dict[str, Any]:
    backend = _readback_backend()
    signature: dict[str, Any] = {
        "language": language,
        "backend": backend,
        "secondary_model": _safe_load_key("dubbing_quality.asr_readback_secondary_model", ""),
    }
    if backend in {"moss", "moss-mlx", "mlx-moss"}:
        from core.providers.moss_asr import moss_asr_health

        health = moss_asr_health()
        signature.update({"model": health.model, "python": health.python, "root": health.root})
    elif backend in {"builtin", "stable_ts", "stable_ts_mlx"} or _command_template() == ["builtin"]:
        signature["model"] = str(_safe_load_key("dubbing_quality.asr_readback_model", _safe_load_key("whisper.model", "large-v3-turbo")))
        signature["use_mlx"] = bool(_safe_load_key("dubbing_quality.asr_readback_use_mlx", True))
    else:
        signature["command"] = _command_template()
        template = signature["command"]
        if len(template) >= 2 and Path(template[1]).suffix == ".py" and Path(template[1]).is_file():
            signature["command_script_sha256"] = hashlib.sha256(Path(template[1]).read_bytes()).hexdigest()
    return signature


def _readback_fingerprint(
    *,
    number: int,
    lines: list[str],
    src_lines: list[str],
    language: str,
) -> str:
    payload = {
        "version": 3,
        "readback_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "completion_sha256":hashlib.sha256(Path(__file__).resolve().parents[1].joinpath("dubbing_content_gate.py").read_bytes()).hexdigest(),
        "content_score_min":float(_safe_load_key("dubbing_quality.content_score_min",.88)),
        "scoring_sha256": hashlib.sha256(Path(__file__).with_name("quality.py").read_bytes()).hexdigest(),
        "number_scoring_sha256": hashlib.sha256(Path(__file__).resolve().parents[1].joinpath("translation_invariants.py").read_bytes()).hexdigest(),
        "number": number,
        "lines": lines,
        "src_lines": src_lines,
        "audio": [_audio_signature(_audio_file_for(number, line_index)) for line_index in range(len(lines))],
        "asr": _asr_config_signature(language),
    }
    from core.tts_pronunciation import pronunciation_policy_signature, pronunciation_terms, spoken_plan
    terms = pronunciation_terms()
    payload['pronunciation_policy'] = pronunciation_policy_signature(terms)
    payload['spoken_lines'] = [spoken_plan(line, language, terms)['spoken_text'] for line in lines]
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()


def _is_cached_row(row: pd.Series, fingerprint: str) -> bool:
    transcript = row.get("asr_transcript")
    status = row.get("asr_status")
    if (not isinstance(status, str) or status != "ok" or not isinstance(transcript, str)
            or not transcript.strip()):
        return False
    if pd.isna(row.get("asr_content_score")):
        return False
    stored = row.get("asr_fingerprint")
    return isinstance(stored, str) and stored == fingerprint


def _update_score_summary(
    *,
    content_score: Any,
    leakage_score: Any,
    min_content_score: float | None,
    max_leakage_score: float | None,
) -> tuple[float | None, float | None]:
    if pd.notna(content_score):
        content = float(content_score)
        min_content_score = content if min_content_score is None else min(min_content_score, content)
    if pd.notna(leakage_score):
        leakage = float(leakage_score)
        max_leakage_score = leakage if max_leakage_score is None else max(max_leakage_score, leakage)
    return min_content_score, max_leakage_score


def verify_tasks_df(tasks_df: pd.DataFrame, *, limit: int | None = None, force: bool = False) -> tuple[pd.DataFrame, ReadbackSummary]:
    language = effective_target_language(_safe_load_key("target_language", "auto"))
    batch_transcripts = {}
    template = _command_template() if _readback_backend() == "command" else []
    if (bool(_safe_load_key("dubbing_quality.asr_readback_batch", False)) and len(template) >= 2
            and Path(template[1]).resolve() == Path(__file__).resolve().parents[2] / "scripts/qwen3_asr_readback.py"):
        items = []
        selected = tasks_df if limit is None else tasks_df.head(limit)
        for _, row in selected.iterrows():
            number = int(row["number"])
            lines = normalize_lines(row.get("lines", row.get("text", "")))
            src = normalize_lines(row.get("src_lines", row.get("origin", "")))
            fp = _readback_fingerprint(number=number, lines=lines, src_lines=src, language=language)
            if not force and _is_cached_row(row, fp):
                continue
            items.extend({"audio": str(Path(_audio_file_for(number, i)).resolve()), "language": language}
                         for i in range(len(lines)) if Path(_audio_file_for(number, i)).is_file())
        if items:
            import tempfile
            with tempfile.TemporaryDirectory(prefix="videolingo_readback_") as temp:
                manifest = Path(temp) / "inputs.json"
                manifest.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
                result = subprocess.run(template[:2] + ["--manifest", str(manifest)], capture_output=True, text=True,
                                        timeout=max(120, int(_safe_load_key("dubbing_quality.asr_readback_timeout_seconds", 120)) * len(items)))
                if result.returncode:
                    raise RuntimeError(f"Local batch speech readback failed: {result.stderr[-1200:]}")
                returned = json.loads(result.stdout)
                if (not isinstance(returned, list) or any(not isinstance(v, dict) or not isinstance(v.get("text"), str) for v in returned)
                        or [v.get("audio") for v in returned] != [v["audio"] for v in items]):
                    raise ValueError("Batch readback changed audio identity/cardinality")
                batch_transcripts = {v["audio"]: v["text"] for v in returned}
    checked = 0
    skipped = 0
    failed = 0
    verified = 0
    cached = 0
    min_content_score: float | None = None
    max_leakage_score: float | None = None
    out = tasks_df.copy()

    for column in (
        "asr_transcript",
        "asr_content_score",
        "asr_leakage_score",
        "asr_status",
        "asr_line_results",
        "asr_fingerprint",
        "asr_language",
        "asr_backend",
    ):
        if column not in out.columns:
            out[column] = None
    for column in (
        "asr_transcript",
        "asr_status",
        "asr_line_results",
        "asr_fingerprint",
        "asr_language",
        "asr_backend",
    ):
        out[column] = out[column].astype(object)

    for idx, row in out.iterrows():
        if limit is not None and checked >= limit:
            break

        number = int(row["number"])
        lines = normalize_lines(row.get("lines", row.get("text", ""))) or [str(row.get("text", ""))]
        src_lines = normalize_lines(row.get("src_lines", row.get("origin", "")))
        fingerprint = _readback_fingerprint(number=number, lines=lines, src_lines=src_lines, language=language)
        if not force and _is_cached_row(row, fingerprint):
            cached += 1
            skipped += 1
            min_content_score, max_leakage_score = _update_score_summary(
                content_score=row.get("asr_content_score"),
                leakage_score=row.get("asr_leakage_score"),
                min_content_score=min_content_score,
                max_leakage_score=max_leakage_score,
            )
            continue

        # A failed fresh witness must not retain a prior successful transcript,
        # score or fingerprint when audio/text changed or force=True was used.
        for column in ("asr_transcript", "asr_content_score", "asr_leakage_score",
                       "asr_fingerprint", "asr_language", "asr_backend"):
            out.at[idx, column] = None
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

            if str(Path(audio_path).resolve()) in batch_transcripts:
                transcript = batch_transcripts[str(Path(audio_path).resolve())]
                result = ASRVerificationResult(status="ok" if transcript else "fail", transcript=transcript)
            else:
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
            if (result.status != "ok" or not isinstance(result.transcript, str)
                    or not result.transcript.strip()):
                failed += 1
                line_results.append({
                    "line_index": line_index,
                    "status": "fail",
                    "audio_path": audio_path,
                    "warning": ",".join(result.warnings) or "empty_or_non_text_readback",
                })
                continue

            transcript = result.transcript
            verified += 1
            ref_text = src_lines[line_index] if line_index < len(src_lines) else row.get("origin", "")
            from core.tts_pronunciation import mandarin_transliterations, apply_mandarin_transliterations, pronunciation_terms
            trans_map = mandarin_transliterations(tasks_df.attrs.get('source_terminology', {}).get('terms', pronunciation_terms())) if language.startswith('zh') else {}
            expected_spoken = apply_mandarin_transliterations(expected, trans_map)[0] if trans_map else expected
            content_score = max(content_similarity(expected, transcript), content_similarity(expected_spoken, transcript))
            primary_transcript, secondary_evidence = transcript, None
            secondary_model = _safe_load_key("dubbing_quality.asr_readback_secondary_model", "")
            from core.dubbing_content_gate import probable_tail_truncation, probable_prefix_truncation, unexpected_boundary_speech
            boundary_issues=[]
            from core.tts_pronunciation import spoken_name_issues, spoken_name_witnesses
            boundary_issues += spoken_name_issues(expected, transcript, language,
                tasks_df.attrs.get('source_terminology', {}).get('terms', pronunciation_terms()))
            if probable_tail_truncation(expected, transcript) and probable_tail_truncation(expected_spoken, transcript):
                boundary_issues.append("possible_tail_loss")
            if probable_prefix_truncation(expected, transcript) and probable_prefix_truncation(expected_spoken, transcript):
                boundary_issues.append("possible_prefix_loss")
            if unexpected_boundary_speech(expected, transcript) and unexpected_boundary_speech(expected_spoken, transcript):
                boundary_issues.append("unexpected_boundary_speech")
            from core.providers.quality import spoken_numeric_values
            expected_values = spoken_numeric_values(expected)
            if expected_values and expected_values != spoken_numeric_values(transcript):
                boundary_issues.append('spoken_numeric_value_mismatch')
            threshold=float(_safe_load_key("dubbing_quality.content_score_min", .88))
            if secondary_model and (content_score < threshold or boundary_issues):
                secondary = _run_builtin_asr(audio_path, language, model_name=str(secondary_model))
                secondary_score = max(content_similarity(expected, secondary.transcript or ""), content_similarity(expected_spoken, secondary.transcript or ""))
                secondary_evidence = {"model": secondary_model, "status": secondary.status,
                                      "transcript": secondary.transcript, "content_score": secondary_score,
                                      "trigger":boundary_issues or ["low_content_score"],
                                      "warnings": secondary.warnings}
                numeric_match = not expected_values or expected_values == spoken_numeric_values(secondary.transcript)
                sec_t = secondary.transcript or ""
                secondary_complete=(not (probable_tail_truncation(expected, sec_t) and probable_tail_truncation(expected_spoken, sec_t))
                                    and not (probable_prefix_truncation(expected, sec_t) and probable_prefix_truncation(expected_spoken, sec_t))
                                    and not (unexpected_boundary_speech(expected, sec_t) and unexpected_boundary_speech(expected_spoken, sec_t)))
                secondary_complete = secondary_complete and not spoken_name_issues(expected, sec_t, language,
                    tasks_df.attrs.get('source_terminology', {}).get('terms', pronunciation_terms()))
                if (secondary.status == "ok" and secondary_score >= threshold and numeric_match
                        and secondary_complete and (secondary_score > content_score or boundary_issues)):
                    transcript, content_score = secondary.transcript, secondary_score
            leak_score = reference_leak_score(ref_text, transcript, expected=expected)
            transcripts.append(transcript)
            content_scores.append(content_score)
            leak_scores.append(leak_score)
            min_content_score = content_score if min_content_score is None else min(min_content_score, content_score)
            max_leakage_score = leak_score if max_leakage_score is None else max(max_leakage_score, leak_score)
            line_results.append({
                "line_index": line_index,
                "status": "ok",
                "audio_path": audio_path,
                "audio_sha256": hash_path(audio_path),
                "expected": expected,
                "transcript": transcript,
                "primary_transcript": primary_transcript,
                "secondary_readback": secondary_evidence,
                "spoken_name_witnesses": spoken_name_witnesses(
                    expected, [{'backend':'primary','status':'ok','transcript':primary_transcript},
                        {'backend':'secondary',**(secondary_evidence or {'status':'unavailable'})}], language,
                    tasks_df.attrs.get('source_terminology',{}).get('terms',pronunciation_terms())),
                "content_score": content_score,
                "leakage_score": leak_score,
            })

        if content_scores and all(item.get("status") == "ok" for item in line_results):
            full_expected = " ".join(lines)
            full_transcript = " ".join(transcripts)
            overall_score = content_similarity(full_expected, full_transcript)
            out.at[idx, "asr_content_score"] = max(overall_score, min(content_scores))
            out.at[idx, "asr_leakage_score"] = max(leak_scores) if leak_scores else 0.0
            out.at[idx, "asr_transcript"] = full_transcript
            out.at[idx, "asr_status"] = "ok"
            out.at[idx, "asr_fingerprint"] = fingerprint
            out.at[idx, "asr_language"] = language
            out.at[idx, "asr_backend"] = _readback_backend()
        elif line_results:
            statuses = {item.get("status") for item in line_results}
            out.at[idx, "asr_status"] = "skipped" if statuses == {"skipped"} else "fail"
            out.at[idx, "asr_fingerprint"] = None
        out.at[idx, "asr_line_results"] = json.dumps(line_results, ensure_ascii=False)

    status = "ok"
    if failed:
        status = "fail"
    elif verified == 0 and cached == 0:
        status = "skipped"
    return out, ReadbackSummary(
        status=status,
        checked=checked,
        skipped=skipped,
        failed=failed,
        cached=cached,
        min_content_score=min_content_score,
        max_leakage_score=max_leakage_score,
        output_file=TTS_TASKS_FILE,
    )


def verify_tasks_file(path: str = TTS_TASKS_FILE, *, limit: int | None = None, force: bool = False) -> ReadbackSummary:
    tasks_df = pd.read_excel(path)
    out, summary = verify_tasks_df(tasks_df, limit=limit, force=force)
    out.to_excel(path, index=False)
    return summary
