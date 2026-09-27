from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from core.config_utils import load_key
from core.providers.speaker_diarization import normalize_speaker
from core.step6_generate_final_timeline import get_sentence_timestamps


DEFAULT_SENTENCES_PATH = Path("output/log/sentence_splitbymeaning.txt")
DEFAULT_WORDS_PATH = Path("output/log/cleaned_chunks.xlsx")
DEFAULT_REPORT_PATH = Path("output/log/translation_speaker_context.json")


@dataclass(frozen=True)
class SpeakerContextMetrics:
    total_lines: int
    known_speaker_lines: int
    coverage: float
    speaker_count: int
    speakers: list[str]
    speaker_transitions: int
    context_lines_before: int
    context_lines_after: int
    report_path: str


def load_speaker_aware_rows(
    sentences_path: str | Path = DEFAULT_SENTENCES_PATH,
    words_path: str | Path = DEFAULT_WORDS_PATH,
    report_path: str | Path = DEFAULT_REPORT_PATH,
) -> tuple[pd.DataFrame, SpeakerContextMetrics]:
    """Align final translation lines back to diarized words without changing text."""
    sentences_path = Path(sentences_path)
    words_path = Path(words_path)
    lines = [line.strip() for line in sentences_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    words = pd.read_excel(words_path)
    sentence_df = pd.DataFrame({"Source": lines})
    timestamps = get_sentence_timestamps(words, sentence_df)
    if len(timestamps) != len(lines):
        raise RuntimeError(
            f"Speaker-aware translation alignment returned {len(timestamps)} rows for {len(lines)} lines."
        )

    rows = pd.DataFrame({
        "line_id": [f"L{index + 1:05d}" for index in range(len(lines))],
        "text": lines,
        "speaker": [normalize_speaker(item[2]) for item in timestamps],
        "start": [float(item[0]) for item in timestamps],
        "end": [float(item[1]) for item in timestamps],
    })
    labels = rows["speaker"].tolist()
    known = [label for label in labels if label]
    required = bool(load_key("translation_context.require_speakers", load_key("speaker_diarization.required", True)))
    minimum = float(load_key("translation_context.min_line_coverage", 0.60))
    coverage = len(known) / max(len(rows), 1)
    if required and coverage < minimum:
        raise RuntimeError(
            f"Translation speaker coverage {coverage:.1%} is below required {minimum:.1%}; "
            "refusing to flatten dialogue into speakerless LLM input."
        )
    transitions = sum(
        left != right
        for left, right in zip(labels, labels[1:])
        if left is not None and right is not None
    )
    report_path = Path(report_path)
    metrics = SpeakerContextMetrics(
        total_lines=len(rows),
        known_speaker_lines=len(known),
        coverage=coverage,
        speaker_count=len(set(known)),
        speakers=sorted(set(known)),
        speaker_transitions=transitions,
        context_lines_before=int(load_key("translation_context.previous_lines", 4)),
        context_lines_after=int(load_key("translation_context.next_lines", 3)),
        report_path=str(report_path),
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(asdict(metrics), ensure_ascii=False, indent=2), encoding="utf-8")
    return rows, metrics


def format_dialogue(rows: pd.DataFrame, *, include_line_ids: bool = True) -> str:
    formatted = []
    for _, row in rows.iterrows():
        line_id = f"[{row['line_id']}]" if include_line_ids else ""
        speaker = normalize_speaker(row.get("speaker")) or "UNKNOWN"
        formatted.append(f"{line_id}[{speaker}] {row['text']}")
    return "\n".join(formatted)


def split_speaker_aware_batches(
    rows: pd.DataFrame,
    *,
    chunk_size: int = 2000,
    max_lines: int = 10,
) -> list[dict[str, Any]]:
    """Create deterministic batches while retaining global row indices and speakers."""
    batches: list[dict[str, Any]] = []
    current_indices: list[int] = []
    current_chars = 0
    for idx, row in rows.iterrows():
        line_chars = len(str(row["text"])) + 1
        if current_indices and (current_chars + line_chars > chunk_size or len(current_indices) >= max_lines):
            batches.append(_batch(rows, current_indices, len(batches)))
            current_indices = []
            current_chars = 0
        current_indices.append(idx)
        current_chars += line_chars
    if current_indices:
        batches.append(_batch(rows, current_indices, len(batches)))
    return batches


def _batch(rows: pd.DataFrame, indices: list[int], batch_index: int) -> dict[str, Any]:
    selected = rows.loc[indices].copy()
    return {
        "batch_id": f"B{batch_index + 1:04d}",
        "batch_index": batch_index,
        "start_index": indices[0],
        "end_index": indices[-1],
        "source_text": "\n".join(selected["text"].astype(str)),
        "speaker_context": format_dialogue(selected),
        "rows": selected,
    }


def surrounding_context(rows: pd.DataFrame, batch: dict[str, Any]) -> tuple[str | None, str | None]:
    before_count = int(load_key("translation_context.previous_lines", 4))
    after_count = int(load_key("translation_context.next_lines", 3))
    start = int(batch["start_index"])
    end = int(batch["end_index"])
    before = rows.iloc[max(0, start - before_count):start]
    after = rows.iloc[end + 1:end + 1 + after_count]
    return (
        format_dialogue(before) if not before.empty else None,
        format_dialogue(after) if not after.empty else None,
    )


def speaker_profiles_context(terminology: dict[str, Any]) -> str:
    payload = {
        "topic": terminology.get("topic", ""),
        "speaker_profiles": terminology.get("speaker_profiles", []),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)
