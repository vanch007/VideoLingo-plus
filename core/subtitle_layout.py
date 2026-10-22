"""Display-only segmentation. Accepted translation and dubbing units stay intact."""
from __future__ import annotations

import math
import re

import pandas as pd


def split_display_text(text: str, max_chars: int) -> list[str]:
    text = str(text).strip()
    if len(text) <= max_chars:
        return [text]
    # Retain complete Latin words; CJK characters can break independently.
    tokens = re.findall(r"[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]|[^\s\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]+|\s+", text)
    pieces, buf = [], ""
    for token in tokens:
        if buf.strip() and len((buf + token).strip()) > max_chars:
            pieces.append(buf.strip())
            buf = ""
        buf += token
    if buf.strip():
        pieces.append(buf.strip())
    return pieces


def fit_display_windows(frame: pd.DataFrame, words: pd.DataFrame, *, max_cps: float,
                        min_duration: float, max_lead_in: float, max_linger: float):
    """Borrow only observed gaps; keep speech and semantic/TTS timings intact."""
    windows = [tuple(map(float, stamp)) for stamp in frame["timestamp"]]
    if not {"start", "end"} <= set(words.columns):
        return windows, []
    speech = [(float(row.start), float(row.end)) for row in words.itertuples()
              if pd.notna(row.start) and pd.notna(row.end)
              and math.isfinite(float(row.start)) and math.isfinite(float(row.end))
              and float(row.end) > float(row.start)]
    adjustments = []
    fitted = []
    for index, (_, row) in enumerate(frame.iterrows()):
        start, end = windows[index]
        if not math.isfinite(start) or not math.isfinite(end) or end <= start:
            fitted.append((start, end))
            continue
        need = max(min_duration, len(str(row["Translation"])) / max(max_cps, 0.001)) - (end - start)
        previous = [b for a, b in speech if b <= start]
        following = [a for a, b in speech if a >= end]
        # Crossing/overlapping native words block padding on that side.
        lower = max(previous) if previous else start
        upper = min(following) if following else end
        if any(a < start < b for a, b in speech):
            lower = start
        if any(a < end < b for a, b in speech):
            upper = end
        # Adjacent captions share a gap at its midpoint, so separate extensions
        # cannot collide or display over another dialogue row.
        if index:
            lower = max(lower, (windows[index - 1][1] + start) / 2)
        if index + 1 < len(windows):
            upper = min(upper, (end + windows[index + 1][0]) / 2)
        linger = min(max(0.0, need), max(0.0, max_linger), max(0.0, upper - end))
        lead = min(max(0.0, need - linger), max(0.0, max_lead_in), max(0.0, start - lower))
        fitted.append((start - lead, end + linger))
        if lead or linger:
            adjustments.append({"line_id": str(row.get("LineID", f"L{index+1:05d}")),
                                "spoken": [start, end], "display": [start - lead, end + linger],
                                "basis": "available_native_speech_gap", "accepted": False})
    return fitted, adjustments


def layout_subtitles(frame: pd.DataFrame, words: pd.DataFrame, *,
                     chars_per_line: int = 38, max_lines: int = 2,
                     max_cps: float = 22, min_duration: float = 0.8,
                     max_lead_in: float = 0.20, max_linger: float = 0.85):
    cap = max(8, int(chars_per_line) * int(max_lines))
    records = []
    display_windows, timing_adjustments = fit_display_windows(
        frame, words, max_cps=max_cps, min_duration=min_duration,
        max_lead_in=max_lead_in, max_linger=max_linger)
    for position, (index, row) in enumerate(frame.iterrows()):
        text = str(row["Translation"])
        visual_lines = split_display_text(text, chars_per_line)
        pieces = ["\n".join(visual_lines[i:i + max_lines])
                  for i in range(0, len(visual_lines), max_lines)]
        start, end = display_windows[position]
        # Never manufacture micro-cues just to satisfy a visual length limit.
        if end - start < min_duration * len(pieces):
            pieces = [text]
        boundaries = [float(start)]
        total = sum(len(piece) for piece in pieces)
        cursor = 0
        native = [float(t) for t in words.get("start", []) if start < float(t) < end]
        for part in pieces[:-1]:
            cursor += len(part)
            ideal = start + (end - start) * cursor / max(total, 1)
            remaining = len(pieces) - len(boundaries)
            low = boundaries[-1] + min_duration
            high = end - remaining * min_duration
            candidates = [t for t in native if low <= t <= high]
            boundary = min(candidates, key=lambda t: abs(t - ideal)) if candidates else min(high, max(low, ideal))
            boundaries.append(boundary)
        boundaries.append(float(end))
        # Bilingual source partitions are display estimates, never new ASR words.
        source = str(row["Source"])
        source_cuts = [0]
        cursor = 0
        for part in pieces[:-1]:
            cursor += len(part)
            pos = round(len(source) * cursor / max(total, 1))
            if " " in source:
                spaces = [m.end() for m in re.finditer(r"\s+", source) if m.end() > source_cuts[-1]]
                pos = min(spaces, key=lambda p: abs(p - pos)) if spaces else len(source)
            source_cuts.append(pos)
        source_cuts.append(len(source))
        for part_index, part in enumerate(pieces):
            record = row.to_dict()
            record.update(
                Translation=part,
                Source=source[source_cuts[part_index]:source_cuts[part_index + 1]].strip(),
                timestamp=(boundaries[part_index], boundaries[part_index + 1]),
                parent_line_id=str(row.get("LineID", f"L{index+1:05d}")),
                display_part=part_index + 1,
                display_alignment="native_boundary_estimate" if len(pieces) > 1 else "original",
                spoken_timestamp=tuple(row["timestamp"]),
                display_padded=tuple(row["timestamp"]) != (start, end),
            )
            record["duration"] = record["timestamp"][1] - record["timestamp"][0]
            records.append(record)
    result = pd.DataFrame(records)
    issues = []
    for index, row in result.iterrows():
        duration = float(row["duration"])
        cps = len(row["Translation"].replace("\n", "")) / max(duration, 0.001)
        reasons = []
        if not math.isfinite(duration) or duration <= 0:
            reasons.append("invalid_timing")
        if len(row["Translation"].replace("\n", "")) > cap:
            reasons.append("too_many_characters")
        if cps > max_cps + 1e-6:
            reasons.append("reading_speed")
        if duration + 1e-6 < min_duration:
            reasons.append("short_cue")
        if reasons:
            issues.append({"cue": index + 1, "line_id": row["parent_line_id"],
                           "reasons": reasons, "cps": round(cps, 2), "duration": round(duration, 3)})
    report = {
        "status": "failed" if any("invalid_timing" in i["reasons"] for i in issues) else ("needs_review" if issues else "pass"),
        "semantic_rows": len(frame), "display_cues": len(result),
        "max_chars_per_cue": cap, "max_cps": max_cps,
        "alignment_basis": "source_dialogue_with_observed_silence_padding; not_target_word_alignment",
        "timing_adjustments": timing_adjustments,
        "max_lead_in": max_lead_in, "max_linger": max_linger,
        "issues": issues,
    }
    return result, report
