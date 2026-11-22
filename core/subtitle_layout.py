"""Display-only segmentation. Accepted translation and dubbing units stay intact."""
from __future__ import annotations

import math
import re
import platform
import subprocess
from pathlib import Path

import pandas as pd


def probe_video_dimensions(video_path: str | Path | None = None) -> tuple[int, int] | None:
    """Return (width, height) for the active source video, or None if unavailable."""
    candidates = [Path(video_path)] if video_path is not None else [
        Path("output/source_canonical.mp4"),
        Path("output/source.mp4"),
    ]
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            out = subprocess.check_output(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=width,height",
                    "-of",
                    "csv=s=x:p=0",
                    str(candidate),
                ],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            if "x" in out:
                w_str, h_str = out.splitlines()[0].split("x", 1)
                w, h = int(w_str), int(h_str)
                if w > 0 and h > 0:
                    return w, h
        except Exception:
            pass
        try:
            import cv2
            cap = cv2.VideoCapture(str(candidate))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            if w > 0 and h > 0:
                return w, h
        except Exception:
            pass
    return None


def compute_subtitle_line_limits(width: int | None, height: int | None) -> tuple[int, int]:
    """Return (cjk_max_chars, latin_max_chars) fitted to video orientation and aspect ratio."""
    if not width or not height or width <= 0 or height <= 0 or height <= width:
        return 22, 42
    aspect = width / height
    cjk_cap = max(10, min(18, round(24 * aspect)))
    latin_cap = max(18, min(36, round(42 * aspect)))
    return int(cjk_cap), int(latin_cap)


def compute_subtitle_burn_style(
    width: int | None,
    height: int | None,
    *,
    platform_system: str | None = None,
) -> dict[str, int | str | bool]:
    """Compute libass force_style parameters (PlayResY=288 baseline) for landscape or portrait."""
    sys_name = platform_system or platform.system()
    if sys_name == "Darwin":
        font_name = "PingFang SC"
    elif sys_name == "Linux":
        font_name = "NotoSansCJK-Regular"
    else:
        font_name = "Arial"

    is_portrait = bool(width and height and width > 0 and height > 0 and height > width)
    cjk_cap, latin_cap = compute_subtitle_line_limits(width, height)
    if not is_portrait:
        return {
            "is_portrait": False,
            "font_size": 20,
            "src_font_size": 14,
            "font_name": font_name,
            "margin_v": 18,
            "margin_lr": 20,
            "outline": 1,
            "shadow": 0,
            "cjk_max_chars": cjk_cap,
            "latin_max_chars": latin_cap,
        }

    aspect = width / height
    # Fit cjk_cap CJK characters inside ~72-76% of frame width with comfortable side margins.
    font_size = max(10, min(16, round(288 * aspect / max(cjk_cap, 1))))
    src_font_size = max(9, min(14, round(font_size * 0.85)))
    return {
        "is_portrait": True,
        "font_size": int(font_size),
        "src_font_size": int(src_font_size),
        "font_name": font_name,
        "margin_v": 26,
        "margin_lr": 24,
        "outline": 1,
        "shadow": 1,
        "cjk_max_chars": cjk_cap,
        "latin_max_chars": latin_cap,
    }


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


_SENTENCE_END = set(".?!。？！…")
_CLAUSE_END = set(",;:，；：、—")
_WORD_CHAR = re.compile(r"[\w\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]")


def balanced_pages(text: str, chars_per_line: int, max_lines: int) -> list[str]:
    """Split a long cue into the fewest pages that fit, with balanced lengths, cutting preferably after a
    sentence, then after a clause, then between words. Fixed two-line paging left orphans such as a final
    page holding only "name."."""
    text = str(text).strip()
    lines = split_display_text(text, chars_per_line)
    if len(lines) <= max_lines:
        return [text]
    pages_needed = -(-len(lines) // max_lines)
    cjk = bool(re.search(r"[\u3400-\u9fff]", text)) and " " not in text
    cuts = [i for i in range(1, len(text)) if (cjk and _WORD_CHAR.match(text[i])) or (not cjk and text[i] == " ")]
    def penalty(i):
        before = text[:i].rstrip()[-1:]
        return 0.0 if before in _SENTENCE_END else (0.35 if before in _CLAUSE_END else 1.0)
    def fits(a, b):
        part = text[a:b].strip()
        return bool(_WORD_CHAR.search(part)) and len(split_display_text(part, chars_per_line)) <= max_lines
    points = [0] + cuts + [len(text)]
    ideal = len(text) / pages_needed
    best = {(0, 0): (0.0, None)}
    for k in range(1, pages_needed + 1):
        for j_index, j in enumerate(points[1:], 1):
            if (k < pages_needed) == (j == len(text)):
                continue
            for i in points[:j_index]:
                prev = best.get((k - 1, i))
                if prev is None or not fits(i, j):
                    continue
                cost = prev[0] + ((j - i - ideal) / max(ideal, 1)) ** 2 + (penalty(j) if j < len(text) else 0)
                if (k, j) not in best or cost < best[(k, j)][0]:
                    best[(k, j)] = (cost, i)
    if (pages_needed, len(text)) not in best:
        return ["\n".join(lines[i:i + max_lines]) for i in range(0, len(lines), max_lines)]
    bounds, j = [len(text)], len(text)
    for k in range(pages_needed, 0, -1):
        j = best[(k, j)][1]
        bounds.append(j)
    bounds.reverse()
    return ["\n".join(split_display_text(text[a:b].strip(), chars_per_line)) for a, b in zip(bounds, bounds[1:])]


def snap_source_cut(source: str, pos: int, low: int) -> int:
    """Move a proportional source cut to the nearest punctuation or word boundary so that both sides keep
    real words (never a cue holding only "。")."""
    candidates = []
    for p in range(low + 1, len(source)):
        if not (_WORD_CHAR.search(source[low:p]) and _WORD_CHAR.search(source[p:])):
            continue
        if " " in source and source[p - 1] != " " and source[p] != " ":
            continue  # Latin source: only between words
        bonus = 0 if source[p - 1] in _SENTENCE_END | _CLAUSE_END else 1.5
        candidates.append((abs(p - pos) + bonus, p))
    return min(candidates)[1] if candidates else pos


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
        pieces = balanced_pages(text, chars_per_line, max_lines)
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
            source_cuts.append(snap_source_cut(source, pos, source_cuts[-1]))
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
