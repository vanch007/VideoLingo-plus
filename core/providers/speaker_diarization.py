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
    backend_name: str = "moss-transcribe-diarize",
    probabilities_path: str | None = None,
    threshold: float = 0.5,
    max_gap_seconds: float = 0.35,
    drop_unassigned_fillers: Iterable[str] = ("嗯", "啊", "哎", "呃"),
    word_speaker_overrides: Iterable[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Attach MOSS speaker IDs to native ASR words without changing timestamps."""
    turns = _validated_turns(diarization_segments)
    probs_data = None
    if probabilities_path and Path(probabilities_path).is_file():
        try:
            import numpy as np
            probs_data = np.load(probabilities_path)
        except Exception:
            probs_data = None

    assigned = 0
    labels: list[str] = []
    total = 0
    filler_set = {str(item).strip() for item in drop_unassigned_fillers if str(item).strip()}
    dropped_fillers = 0
    speaker_overrides_applied = 0

    for segment in result.get("segments", []):
        segment_labels: list[str] = []
        seg_words = segment.get("words", []) or []
        for w_idx, word in enumerate(seg_words):
            start = float(word["start"])
            end = float(word["end"])
            assigned_by_prob = False
            if probs_data is not None:
                try:
                    stride = float(probs_data.get("frame_stride", 0.01))
                    probs_arr = probs_data["probs"]
                    import re
                    token_str = str(word.get("word", word.get("text", ""))).strip('"').strip("'")
                    is_cjk_char = bool(len(token_str) == 1 and re.match(r"^[\u4e00-\u9fff]$", token_str))
                    eval_end = end
                    if is_cjk_char and w_idx + 1 < len(seg_words):
                        next_w = seg_words[w_idx + 1]
                        next_token = str(next_w.get("word", next_w.get("text", ""))).strip('"').strip("'")
                        next_start = float(next_w.get("start", 0))
                        if abs(next_start - end) <= 0.02 and bool(len(next_token) == 1 and re.match(r"^[\u4e00-\u9fff]$", next_token)):
                            eval_end = float(next_w.get("end", end))

                    s_frame = max(0, int(start / stride))
                    e_frame = max(s_frame + 1, int(math.ceil(eval_end / stride)))
                    e_frame = min(e_frame, probs_arr.shape[0])
                    if s_frame < probs_arr.shape[0]:
                        sl_probs = probs_arr[s_frame:e_frame, :]
                        mean_scores = sl_probs.mean(axis=0)
                        import numpy as np
                        sorted_indices = np.argsort(mean_scores)[::-1]
                        top_idx = int(sorted_indices[0])
                        top_prob = float(mean_scores[top_idx])
                        runner_up_idx = int(sorted_indices[1]) if len(sorted_indices) > 1 else -1
                        runner_up_prob = float(mean_scores[runner_up_idx]) if runner_up_idx >= 0 else 0.0
                        margin = top_prob - runner_up_prob

                        if top_prob >= threshold:
                            word["speaker_score"] = round(top_prob, 4)
                            if runner_up_prob >= threshold and margin < 0.05:
                                word["speaker"] = None
                                word["speaker_overlap"] = True
                                word["speaker_ambiguous"] = True
                                word["speaker_candidates"] = [f"S{top_idx + 1:02d}", f"S{runner_up_idx + 1:02d}"]
                                word["speaker_source"] = f"{backend_name}_tied_overlap"
                            else:
                                spk_str = f"S{top_idx + 1:02d}"
                                word["speaker"] = spk_str
                                word["speaker_source"] = backend_name
                                speaker_overrides_applied += apply_speaker_overrides_to_rows(
                                    [word], word_speaker_overrides
                                )
                                assigned_spk = word["speaker"]
                                segment_labels.append(assigned_spk)
                                labels.append(assigned_spk)
                                assigned += 1
                            total += 1
                            assigned_by_prob = True
                except Exception:
                    assigned_by_prob = False
            if assigned_by_prob:
                continue

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
                assigned_sub = False
                if probs_data is not None:
                    try:
                        stride = float(probs_data.get("frame_stride", 0.01))
                        probs_arr = probs_data["probs"]
                        s_frame = max(0, int(start / stride))
                        e_frame = max(s_frame + 1, int(math.ceil(end / stride)))
                        e_frame = min(e_frame, probs_arr.shape[0])
                        if s_frame < probs_arr.shape[0]:
                            sl_probs = probs_arr[s_frame:e_frame, :]
                            mean_scores = sl_probs.mean(axis=0)
                            sorted_scores = sorted(mean_scores, reverse=True)
                            top_prob = float(sorted_scores[0])
                            runner_up = float(sorted_scores[1]) if len(sorted_scores) > 1 else 0.0
                            margin = top_prob - runner_up
                            # Only assign subthreshold if confident (>=0.30) and distinct margin (>=0.10)
                            if top_prob >= 0.30 and margin >= 0.10:
                                top_idx = int(mean_scores.argmax())
                                spk_str = f"S{top_idx + 1:02d}"
                                word["speaker"] = spk_str
                                word["speaker_score"] = round(top_prob, 4)
                                word["speaker_source"] = f"{backend_name}_subthreshold"
                                speaker_overrides_applied += apply_speaker_overrides_to_rows(
                                    [word], word_speaker_overrides
                                )
                                assigned_spk = word["speaker"]
                                segment_labels.append(assigned_spk)
                                labels.append(assigned_spk)
                                assigned += 1
                                total += 1
                                assigned_sub = True
                    except Exception:
                        pass
                if assigned_sub:
                    continue
                # Retain reactions / unassigned tokens as UNKNOWN rather than dropping
                total += 1
                word["speaker"] = None
                continue
            total += 1
            selected = max(ranked, key=lambda item: (item[0], item[1]))[2]
            word["speaker"] = selected["speaker"]
            word["speaker_source"] = backend_name
            speaker_overrides_applied += apply_speaker_overrides_to_rows(
                [word], word_speaker_overrides
            )
            assigned_speaker = word["speaker"]
            segment_labels.append(assigned_speaker)
            labels.append(assigned_speaker)
            assigned += 1

        # Keep all spoken words and reactions without dropping

        # Pure acoustic diarization without artificial smoothing or hardcoded lexical overwrites

    # Recompute segment labels and overall labels after smoothing
    labels = []
    assigned = 0
    for segment in result.get("segments", []):
        seg_spks = [w["speaker"] for w in segment.get("words", []) if w.get("speaker")]
        if seg_spks:
            segment["speaker"] = Counter(seg_spks).most_common(1)[0][0]
            segment["speaker_mixed"] = len(set(seg_spks)) > 1
        for w in segment.get("words", []) or []:
            if w.get("speaker"):
                labels.append(w["speaker"])
                assigned += 1

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


def group_word_rows_by_speaker(rows: Iterable[dict[str, Any]], max_gap_seconds: float = 0.8) -> list[dict[str, Any]]:
    """Group contiguous words with the same known speaker, or contiguous unassigned words within speech flow."""
    groups: list[dict[str, Any]] = []
    prev_end: float | None = None
    for row in rows:
        speaker = normalize_speaker(row.get("speaker"))
        text = str(row.get("text", "")).strip().strip('"')
        if not text:
            continue
        start_val = row.get("start")
        end_val = row.get("end")
        has_timing = (start_val is not None and end_val is not None)
        try:
            start = float(start_val) if has_timing else 0.0
            end = float(end_val) if has_timing else 0.0
        except (ValueError, TypeError):
            has_timing = False
            start, end = 0.0, 0.0

        can_group = False
        if groups:
            prev_group = groups[-1]
            if speaker is not None:
                if prev_group["speaker"] == speaker:
                    can_group = True
            elif prev_group["speaker"] is None:
                if has_timing and prev_end is not None and (start - prev_end) <= max_gap_seconds:
                    can_group = True

        if can_group:
            groups[-1]["words"].append(text)
        else:
            groups.append({"speaker": speaker, "words": [text]})
        prev_end = end if has_timing else None
    return groups


def speaker_coverage(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    assigned = 0
    speakers = set()
    total = 0
    for row in rows:
        total += 1
        label = normalize_speaker(row.get("speaker"))
        if label:
            assigned += 1
            speakers.add(label)
        elif row.get("speaker_overlap") and row.get("speaker_candidates"):
            assigned += 1
            for cand in row["speaker_candidates"]:
                if cand:
                    speakers.add(cand)
    return {
        "word_count": total,
        "assigned_words": assigned,
        "coverage": assigned / total if total else 0.0,
        "speakers": sorted(speakers),
    }


def validate_cached_speaker_rows(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Fail closed when an old ASR cache predates required diarization."""
    cfg = dict(load_key("speaker_diarization", {}) or {})
    if not bool(cfg.get("enabled", True)):
        return {"status": "disabled"}
    report = speaker_coverage(rows)
    minimum = float(cfg.get("min_word_coverage", 0.95))
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
    shadow_backend = str(cfg.get("shadow_backend", "") or "").strip().lower()
    supported_backends = ("moss-mlx", "nemotron-mlx")
    if backend not in supported_backends:
        raise SpeakerDiarizationError(
            f"speaker diarization backend {backend!r} is not installed; supported: {', '.join(supported_backends)}"
        )

    failure: Exception | None = None
    report: dict[str, Any] = {"status": "pending", "backend": backend}
    try:
        probabilities_path: str | None = None
        if backend == "nemotron-mlx":
            from core.providers.nemotron_diarization import run_nemotron_diarization

            run = run_nemotron_diarization(audio_path)
            backend_name = "nemotron-mlx"
            probabilities_path = run.probabilities_path
        else:
            from core.providers.moss_asr import run_moss_asr

            run = run_moss_asr(audio_path, purpose="speaker_diarization")
            backend_name = "moss-transcribe-diarize"
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
                min_turn_duration=float(coverage_cfg.get("min_turn_duration", 0.50)),
                recovery_merge_gap_seconds=float(
                    coverage_cfg.get("recovery_merge_gap_seconds", 1.50)
                ),
                recovery_padding_seconds=float(
                    coverage_cfg.get("recovery_padding_seconds", 1.00)
                ),
                min_coverage=float(coverage_cfg.get("min_coverage", 0.95)),
            )
            report["source_coverage"] = {
                "status": source_report["status"],
                "before": source_report["before"]["coverage"],
                "after": source_report["after"]["coverage"],
                "recovered_ranges": len(source_report["recovered_ranges"]),
            }
        align_report = align_speakers_to_words(
            result,
            run.segments,
            backend_name=backend_name,
            probabilities_path=probabilities_path,
            threshold=float(cfg.get("threshold", 0.5)),
            max_gap_seconds=float(cfg.get("max_gap_seconds", 0.35)),
            drop_unassigned_fillers=cfg.get(
                "drop_unassigned_fillers", ["嗯", "啊", "哎", "呃"]
            ),
            word_speaker_overrides=cfg.get("word_speaker_overrides", []),
        )
        report.update(align_report)
        report.update({"status": "pass", "backend": backend, "evidence_dir": run.output_dir})

        report_target_path = Path(str(cfg.get("report_path", "output/log/speaker_diarization.json")))
        default_shadow_path = report_target_path.parent / "speaker_diarization_shadow.json"
        if shadow_backend and shadow_backend != backend and shadow_backend in supported_backends:
            import copy
            shadow_result = copy.deepcopy(result)
            try:
                if shadow_backend == "nemotron-mlx":
                    from core.providers.nemotron_diarization import run_nemotron_diarization

                    shadow_run = run_nemotron_diarization(audio_path)
                    shadow_report = align_speakers_to_words(
                        shadow_result,
                        shadow_run.segments,
                        backend_name="nemotron-mlx",
                        probabilities_path=shadow_run.probabilities_path,
                        threshold=float(cfg.get("threshold", 0.5)),
                        max_gap_seconds=float(cfg.get("max_gap_seconds", 0.35)),
                    )
                    shadow_report["evidence_dir"] = shadow_run.output_dir
                else:
                    from core.providers.moss_asr import run_moss_asr

                    shadow_run = run_moss_asr(audio_path, purpose="speaker_diarization")
                    shadow_report = align_speakers_to_words(
                        shadow_result,
                        shadow_run.segments,
                        backend_name="moss-transcribe-diarize",
                        max_gap_seconds=float(cfg.get("max_gap_seconds", 0.35)),
                    )
                    shadow_report["evidence_dir"] = shadow_run.output_dir
                shadow_report["status"] = "pass"
                shadow_report["backend"] = shadow_backend
                shadow_report["primary_backend"] = backend
                shadow_path = Path(str(cfg.get("shadow_report_path", default_shadow_path)))
                shadow_path.parent.mkdir(parents=True, exist_ok=True)
                shadow_path.write_text(json.dumps(shadow_report, ensure_ascii=False, indent=2), encoding="utf-8")
                report["shadow"] = {
                    "status": "pass",
                    "backend": shadow_backend,
                    "coverage": shadow_report.get("coverage", 0.0),
                    "speaker_count": shadow_report.get("speaker_count", 0),
                    "report_path": str(shadow_path),
                }
            except Exception as shadow_err:
                report["shadow"] = {"status": "fail", "backend": shadow_backend, "error": str(shadow_err)}

        minimum = float(cfg.get("min_word_coverage", 0.95))
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
        raise SpeakerDiarizationError(f"required {backend} speaker diarization failed: {failure}") from failure
    return report


# Compatibility name for callers added during the initial MOSS integration.
apply_moss_speaker_diarization = apply_speaker_diarization
