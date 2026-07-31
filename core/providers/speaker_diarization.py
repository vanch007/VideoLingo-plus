from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from core.config_utils import load_key


class SpeakerDiarizationError(RuntimeError):
    """Raised when speaker boundaries are not reliable enough for production."""


def normalize_speaker(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    speaker = str(value).strip()
    if not speaker or speaker.lower() in {"nan", "none", "unknown", "<na>"}:
        return None
    return speaker


def same_known_speaker(left: Any, right: Any) -> bool:
    left_speaker = normalize_speaker(left)
    right_speaker = normalize_speaker(right)
    return bool(left_speaker and right_speaker and left_speaker == right_speaker)


def apply_speaker_overrides_to_rows(
    rows: Iterable[dict[str, Any]], overrides: Iterable[dict[str, Any]] | None
) -> int:
    """Apply verified, timestamp-anchored speaker fixes without changing timing."""
    configured = list(overrides or [])
    applied = 0
    for row in rows:
        token = str(row.get("word", row.get("text", ""))).strip().strip('"')
        try:
            start = float(row["start"])
        except (KeyError, TypeError, ValueError):
            continue
        for override in configured:
            if not isinstance(override, dict):
                raise ValueError("speaker_diarization.word_speaker_overrides must contain mappings")
            expected_token = str(override.get("text", "")).strip()
            speaker = normalize_speaker(override.get("speaker"))
            if not expected_token or not speaker:
                raise ValueError("speaker override requires non-empty text and speaker")
            try:
                expected_start = float(override["start"])
                tolerance = float(override.get("tolerance_seconds", 0.03))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("speaker override requires numeric start/tolerance_seconds") from exc
            if token == expected_token and abs(start - expected_start) <= tolerance:
                if normalize_speaker(row.get("speaker")) != speaker:
                    row["speaker"] = speaker
                    row["speaker_source"] = "verified-config-override"
                    applied += 1
                break
    return applied


def _validated_turns(segments: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    turns: list[dict[str, Any]] = []
    for index, segment in enumerate(segments):
        try:
            start = float(segment["start"])
            end = float(segment["end"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SpeakerDiarizationError(f"MOSS turn {index} has invalid timestamps") from exc
        speaker = normalize_speaker(segment.get("speaker"))
        if not speaker or not math.isfinite(start) or not math.isfinite(end) or start < 0 or end < start:
            raise SpeakerDiarizationError(f"MOSS turn {index} is not a valid speaker interval")
        turns.append({"start": start, "end": end, "speaker": speaker})
    if not turns:
        raise SpeakerDiarizationError("MOSS returned no speaker turns")
    return sorted(turns, key=lambda item: (item["start"], item["end"]))


def align_speakers_to_words(
    result: dict[str, Any],
    diarization_segments: list[dict[str, Any]],
    *,
    max_gap_seconds: float = 0.35,
    drop_unassigned_fillers: Iterable[str] = ("嗯", "啊", "哎", "呃"),
    word_speaker_overrides: Iterable[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Attach MOSS speaker IDs to native ASR words without changing timestamps."""
    turns = _validated_turns(diarization_segments)
    assigned = 0
    labels: list[str] = []
    total = 0
    filler_set = {str(item).strip() for item in drop_unassigned_fillers if str(item).strip()}
    dropped_fillers = 0
    speaker_overrides_applied = 0

    for segment in result.get("segments", []):
        segment_labels: list[str] = []
        for word in segment.get("words", []) or []:
            start = float(word["start"])
            end = float(word["end"])
            midpoint = (start + end) / 2.0
            ranked: list[tuple[float, float, dict[str, Any]]] = []
            for turn in turns:
                overlap = max(0.0, min(end, turn["end"]) - max(start, turn["start"]))
                if turn["start"] <= midpoint <= turn["end"]:
                    distance = 0.0
                else:
                    distance = min(abs(midpoint - turn["start"]), abs(midpoint - turn["end"]))
                if overlap > 0 or distance <= max_gap_seconds:
                    ranked.append((overlap, -distance, turn))
            if not ranked:
                token = str(word.get("word", word.get("text", ""))).strip()
                if token in filler_set:
                    word["_drop_unassigned_filler"] = True
                    dropped_fillers += 1
                    continue
                total += 1
                word["speaker"] = None
                continue
            total += 1
            selected = max(ranked, key=lambda item: (item[0], item[1]))[2]
            word["speaker"] = selected["speaker"]
            word["speaker_source"] = "moss-transcribe-diarize"
            speaker_overrides_applied += apply_speaker_overrides_to_rows(
                [word], word_speaker_overrides
            )
            assigned_speaker = word["speaker"]
            segment_labels.append(assigned_speaker)
            labels.append(assigned_speaker)
            assigned += 1

        if dropped_fillers:
            segment["words"] = [
                word
                for word in segment.get("words", []) or []
                if not word.pop("_drop_unassigned_filler", False)
            ]
            if segment["words"]:
                segment["start"] = float(segment["words"][0]["start"])
                segment["end"] = float(segment["words"][-1]["end"])
                segment["text"] = "".join(
                    str(word.get("word", word.get("text", "")))
                    for word in segment["words"]
                )

        if segment_labels:
            segment["speaker"] = Counter(segment_labels).most_common(1)[0][0]
            segment["speaker_mixed"] = len(set(segment_labels)) > 1

    result["segments"] = [segment for segment in result.get("segments", []) if segment.get("words")]
    transitions = sum(left != right for left, right in zip(labels, labels[1:]))
    return {
        "word_count": total,
        "assigned_words": assigned,
        "coverage": assigned / total if total else 0.0,
        "speakers": sorted(set(labels)),
        "speaker_count": len(set(labels)),
        "speaker_transitions": transitions,
        "turn_count": len(turns),
        "dropped_unassigned_fillers": dropped_fillers,
        "speaker_overrides_applied": speaker_overrides_applied,
    }


def group_word_rows_by_speaker(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group only contiguous words with the same known speaker."""
    groups: list[dict[str, Any]] = []
    for row in rows:
        speaker = normalize_speaker(row.get("speaker"))
        text = str(row.get("text", "")).strip().strip('"')
        if not text:
            continue
        if speaker and groups and groups[-1]["speaker"] == speaker:
            groups[-1]["words"].append(text)
        else:
            groups.append({"speaker": speaker, "words": [text]})
    return groups


def speaker_coverage(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    labels = [normalize_speaker(row.get("speaker")) for row in rows]
    assigned = sum(label is not None for label in labels)
    return {
        "word_count": len(labels),
        "assigned_words": assigned,
        "coverage": assigned / len(labels) if labels else 0.0,
        "speakers": sorted({label for label in labels if label}),
    }


def validate_cached_speaker_rows(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Fail closed when an old ASR cache predates required diarization."""
    cfg = dict(load_key("speaker_diarization", {}) or {})
    if not bool(cfg.get("enabled", True)):
        return {"status": "disabled"}
    report = speaker_coverage(rows)
    minimum = float(cfg.get("min_word_coverage", 0.98))
    if report["coverage"] < minimum and bool(cfg.get("required", True)):
        raise SpeakerDiarizationError(
            "cached cleaned_chunks.xlsx has speaker coverage "
            f"{report['coverage']:.1%}, below required {minimum:.1%}; archive/clean the "
            "current output and rerun ASR so speaker-safe downstream artifacts are rebuilt"
        )
    report["status"] = "pass" if report["coverage"] >= minimum else "fail"
    return report


def apply_speaker_diarization(
    result: dict[str, Any],
    audio_path: str,
    *,
    transcribe_range=None,
    source_backend: str = "source-asr",
) -> dict[str, Any]:
    cfg = dict(load_key("speaker_diarization", {}) or {})
    if not bool(cfg.get("enabled", True)):
        return {"status": "disabled"}

    backend = str(cfg.get("backend", "moss-mlx")).strip().lower()
    if backend != "moss-mlx":
        raise SpeakerDiarizationError(
            f"speaker diarization backend {backend!r} is not installed; supported: moss-mlx"
        )

    from core.providers.moss_asr import run_moss_asr

    failure: Exception | None = None
    report: dict[str, Any] = {"status": "pending", "backend": backend}
    try:
        run = run_moss_asr(audio_path, purpose="speaker_diarization")
        coverage_cfg = dict(cfg.get("source_coverage", {}) or {})
        if bool(coverage_cfg.get("enabled", True)):
            if transcribe_range is None:
                raise SpeakerDiarizationError(
                    "source coverage reconciliation requires a native word-ASR range callback"
                )
            from core.providers.source_coverage import reconcile_source_coverage

            source_report = reconcile_source_coverage(
                result,
                run.segments,
                transcribe_range,
                backend=source_backend,
                report_path=str(
                    coverage_cfg.get("report_path", "output/log/source_asr_coverage.json")
                ),
                required=bool(coverage_cfg.get("required", True)),
                min_turn_overlap_ratio=float(
                    coverage_cfg.get("min_turn_overlap_ratio", 0.20)
                ),
                min_text_similarity=float(coverage_cfg.get("min_text_similarity", 0.20)),
                recovery_merge_gap_seconds=float(
                    coverage_cfg.get("recovery_merge_gap_seconds", 1.50)
                ),
                recovery_padding_seconds=float(
                    coverage_cfg.get("recovery_padding_seconds", 0.50)
                ),
            )
            report["source_coverage"] = {
                "status": source_report["status"],
                "before": source_report["before"]["coverage"],
                "after": source_report["after"]["coverage"],
                "recovered_ranges": len(source_report["recovered_ranges"]),
            }
        report = align_speakers_to_words(
            result,
            run.segments,
            max_gap_seconds=float(cfg.get("max_gap_seconds", 0.35)),
            drop_unassigned_fillers=cfg.get(
                "drop_unassigned_fillers", ["嗯", "啊", "哎", "呃"]
            ),
            word_speaker_overrides=cfg.get("word_speaker_overrides", []),
        )
        report.update({"status": "pass", "backend": backend, "evidence_dir": run.output_dir})
        minimum = float(cfg.get("min_word_coverage", 0.98))
        if report["coverage"] < minimum:
            raise SpeakerDiarizationError(
                f"speaker coverage {report['coverage']:.1%} is below required {minimum:.1%}"
            )
    except Exception as exc:
        failure = exc
        report.update({"status": "fail", "error": f"{type(exc).__name__}: {exc}"})

    report_path = Path(str(cfg.get("report_path", "output/log/speaker_diarization.json")))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    result["speaker_diarization"] = report
    if failure is not None and bool(cfg.get("required", True)):
        raise SpeakerDiarizationError(f"required MOSS speaker diarization failed: {failure}") from failure
    return report


# Compatibility name for callers added during the initial MOSS integration.
apply_moss_speaker_diarization = apply_speaker_diarization
