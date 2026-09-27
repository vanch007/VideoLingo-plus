from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, Iterable

from core.asr_schema import validate_word_timestamps


class SourceCoverageError(RuntimeError):
    """Raised when diarized source speech is absent from native word ASR."""


_TEXT_RE = re.compile(r"[^\w\u3400-\u9fff]+", re.UNICODE)


def _normalized_text(value: Any) -> str:
    return _TEXT_RE.sub("", str(value or "").lower())


def _words(result: dict[str, Any]) -> list[dict[str, Any]]:
    return sorted(
        [word for segment in result.get("segments", []) for word in (segment.get("words") or [])],
        key=lambda word: (float(word["start"]), float(word["end"])),
    )


def analyze_source_coverage(
    result: dict[str, Any],
    reference_segments: Iterable[dict[str, Any]],
    *,
    min_turn_overlap_ratio: float = 0.20,
    min_text_similarity: float = 0.20,
    min_text_chars: int = 2,
    min_turn_duration: float = 0.30,
) -> dict[str, Any]:
    """Use MOSS speech turns as a coverage witness, never as word timing.

    A turn is suspect when native ASR words barely overlap it, or when a
    multi-character MOSS turn overlaps unrelated native text. The latter
    catches long timestamp hallucinations that would otherwise hide a gap.
    """
    native_words = _words(result)
    turns: list[dict[str, Any]] = []
    for index, raw in enumerate(reference_segments):
        start = float(raw["start"])
        end = float(raw["end"])
        reference_text = _normalized_text(raw.get("text"))
        if end <= start or len(reference_text) < min_text_chars or end - start < min_turn_duration:
            continue

        overlapping = [
            word
            for word in native_words
            if min(float(word["end"]), end) > max(float(word["start"]), start)
        ]
        overlap_seconds = sum(
            max(0.0, min(float(word["end"]), end) - max(float(word["start"]), start))
            for word in overlapping
        )
        overlap_ratio = min(1.0, overlap_seconds / (end - start))
        native_text = _normalized_text("".join(str(word.get("word", "")) for word in overlapping))
        similarity = SequenceMatcher(None, reference_text, native_text).ratio() if native_text else 0.0
        longest_native_word = max(
            (float(word["end"]) - float(word["start"]) for word in overlapping),
            default=0.0,
        )
        sparse_mismatch = (
            similarity < min_text_similarity
            and len(native_text) < max(2, int(len(reference_text) * 0.60))
        )
        timestamp_hallucination = similarity < min_text_similarity and longest_native_word > 1.50
        # Equal-density disagreements (for example MOSS "鄙人" vs stable-ts
        # "必然") are correction candidates, not missing speech. Coverage
        # fails only on a temporal hole or an obviously sparse/overlong token.
        missing = (
            overlap_ratio < min_turn_overlap_ratio
            or sparse_mismatch
            or timestamp_hallucination
        )
        turns.append(
            {
                "index": index,
                "start": start,
                "end": end,
                "speaker": raw.get("speaker"),
                "reference_text": str(raw.get("text", "")).strip(),
                "native_text": native_text,
                "overlap_ratio": round(overlap_ratio, 4),
                "text_similarity": round(similarity, 4),
                "longest_native_word": round(longest_native_word, 4),
                "status": "missing" if missing else "covered",
            }
        )

    missing_turns = [turn for turn in turns if turn["status"] == "missing"]
    has_any = any(True for _ in reference_segments)
    status = "pass" if not missing_turns else "fail"
    if has_any and not turns:
        status = "not_applicable"
    return {
        "status": status,
        "eligible_turns": len(turns),
        "covered_turns": len(turns) - len(missing_turns),
        "missing_turns": len(missing_turns),
        "coverage": (len(turns) - len(missing_turns)) / len(turns) if turns else 1.0,
        "turns": turns,
    }


def analyze_acoustic_coverage(
    result: dict[str, Any],
    reference_segments: Iterable[dict[str, Any]],
    *,
    min_turn_overlap_ratio: float = 0.20,
    min_turn_duration: float = 0.50,
) -> dict[str, Any]:
    """Analyze temporal coverage of speech turns without requiring reference text.

    Used for acoustic diarization backends (e.g. Nemotron) where turns have
    precise speech intervals but no transcription text.
    """
    native_words = _words(result)
    turns: list[dict[str, Any]] = []
    for index, raw in enumerate(reference_segments):
        start = float(raw["start"])
        end = float(raw["end"])
        if end <= start or end - start < min_turn_duration:
            continue

        overlapping = [
            word
            for word in native_words
            if min(float(word["end"]), end) > max(float(word["start"]), start)
        ]
        overlap_seconds = sum(
            max(0.0, min(float(word["end"]), end) - max(float(word["start"]), start))
            for word in overlapping
        )
        overlap_ratio = min(1.0, overlap_seconds / (end - start))
        missing = overlap_ratio < min_turn_overlap_ratio
        turns.append(
            {
                "index": index,
                "start": start,
                "end": end,
                "speaker": raw.get("speaker"),
                "reference_text": "",
                "native_text": "".join(str(word.get("word", "")) for word in overlapping),
                "overlap_ratio": round(overlap_ratio, 4),
                "text_similarity": None,
                "longest_native_word": 0.0,
                "status": "missing" if missing else "covered",
                "coverage_mode": "acoustic",
            }
        )

    missing_turns = [turn for turn in turns if turn["status"] == "missing"]
    has_any = any(True for _ in reference_segments)
    status = "pass" if not missing_turns else "fail"
    if has_any and not turns:
        status = "not_applicable"
    return {
        "status": status,
        "eligible_turns": len(turns),
        "covered_turns": len(turns) - len(missing_turns),
        "missing_turns": len(missing_turns),
        "coverage": (len(turns) - len(missing_turns)) / len(turns) if turns else 1.0,
        "turns": turns,
        "coverage_mode": "acoustic",
    }


def _merge_missing_ranges(turns: list[dict[str, Any]], max_gap_seconds: float) -> list[dict[str, Any]]:
    ranges: list[dict[str, Any]] = []
    for turn in sorted(turns, key=lambda item: (item["start"], item["end"])):
        if ranges and turn["start"] - ranges[-1]["end"] <= max_gap_seconds:
            ranges[-1]["end"] = max(ranges[-1]["end"], turn["end"])
            ranges[-1]["turn_indexes"].append(turn["index"])
        else:
            ranges.append(
                {"start": turn["start"], "end": turn["end"], "turn_indexes": [turn["index"]]}
            )
    return ranges


def _replace_range_with_recovery(
    result: dict[str, Any],
    recovery: dict[str, Any],
    *,
    start: float,
    end: float,
) -> None:
    merged_words: list[tuple[dict[str, Any], bool]] = []
    for segment in result.get("segments", []):
        for word in segment.get("words", []) or []:
            midpoint = (float(word["start"]) + float(word["end"])) / 2.0
            if not start <= midpoint <= end:
                merged_words.append((dict(word), False))

    for segment in recovery.get("segments", []):
        for word in segment.get("words", []) or []:
            w_s = float(word["start"])
            w_e = float(word["end"])
            midpoint = (w_s + w_e) / 2.0
            overlaps = max(w_s, start) < min(w_e, end)
            if overlaps or (start - 0.1 <= midpoint <= end + 0.1):
                merged_words.append((dict(word), True))

    # Rebuild at word granularity. Keeping a pre-recovery segment containing
    # words on both sides of the repaired range would make segment ordering
    # disagree with word ordering and violate the timestamp contract.
    result["segments"] = []
    for word, recovered in sorted(
        merged_words, key=lambda item: (float(item[0]["start"]), float(item[0]["end"]))
    ):
        segment = {
            "start": float(word["start"]),
            "end": float(word["end"]),
            "text": str(word.get("word", "")),
            "words": [word],
        }
        if recovered:
            segment["source_coverage_recovery"] = True
        result["segments"].append(segment)


def reconcile_source_coverage(
    result: dict[str, Any],
    reference_segments: list[dict[str, Any]],
    transcribe_range: Callable[[float, float], dict[str, Any]],
    *,
    backend: str,
    report_path: str | Path,
    required: bool = True,
    min_turn_overlap_ratio: float = 0.20,
    min_text_similarity: float = 0.20,
    min_turn_duration: float = 0.50,
    recovery_merge_gap_seconds: float = 1.50,
    recovery_padding_seconds: float = 0.50,
    min_coverage: float = 0.95,
) -> dict[str, Any]:
    has_text = any(bool(str(item.get("text", "")).strip()) for item in reference_segments)
    if has_text:
        before = analyze_source_coverage(
            result,
            reference_segments,
            min_turn_overlap_ratio=min_turn_overlap_ratio,
            min_text_similarity=min_text_similarity,
            min_turn_duration=min_turn_duration,
        )
    else:
        before = analyze_acoustic_coverage(
            result,
            reference_segments,
            min_turn_overlap_ratio=min_turn_overlap_ratio,
            min_turn_duration=min_turn_duration,
        )
    ranges = _merge_missing_ranges(
        [turn for turn in before["turns"] if turn["status"] == "missing"],
        recovery_merge_gap_seconds,
    )
    recovered_ranges: list[dict[str, Any]] = []
    for item in ranges:
        padded_start = max(0.0, float(item["start"]) - recovery_padding_seconds)
        padded_end = float(item["end"]) + recovery_padding_seconds
        recovery = transcribe_range(padded_start, padded_end)
        validate_word_timestamps(recovery, backend=f"{backend}-coverage-recovery", allow_empty=True)
        _replace_range_with_recovery(result, recovery, start=float(item["start"]), end=float(item["end"]))
        recovered_ranges.append({**item, "padded_start": padded_start, "padded_end": padded_end})

    validate_word_timestamps(result, backend=backend)
    if has_text:
        after = analyze_source_coverage(
            result,
            reference_segments,
            min_turn_overlap_ratio=min_turn_overlap_ratio,
            min_text_similarity=min_text_similarity,
            min_turn_duration=min_turn_duration,
        )
    else:
        after = analyze_acoustic_coverage(
            result,
            reference_segments,
            min_turn_overlap_ratio=min_turn_overlap_ratio,
            min_turn_duration=min_turn_duration,
        )
        # Strict acoustic coverage: missing turns must not be statistically waived
        if after.get("missing_turns", 0) > 0:
            after["status"] = "fail"

    report = {
        "status": after["status"],
        "backend": backend,
        "before": before,
        "after": after,
        "recovered_ranges": recovered_ranges,
        "timestamp_contract": "native-word-timestamps-only",
    }
    destination = Path(report_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    result["source_coverage"] = report
    if required and after["status"] != "pass":
        raise SourceCoverageError(
            f"source ASR still misses {after['missing_turns']} diarized speech turns after recovery; "
            f"see {destination}"
        )
    return report
