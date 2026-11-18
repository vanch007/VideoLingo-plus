"""Fast Mode: Subtitle segmentation by punctuation & display length mapping to dubbing timeline."""
from __future__ import annotations

import math
import os
from pathlib import Path
import re
import pandas as pd
from rich import print as rprint

from core.subtitle_layout import compute_subtitle_line_limits, probe_video_dimensions
from core.subtitle_measure import atoms, break_lines, has_cjk
from core.timing_utils import seconds_to_srt_time


def _split_text_by_punctuation(text: str, is_cjk: bool = True) -> list[str]:
    """Split text into major clauses using sentence-level punctuation (period, semicolon, question/exclamation)."""
    text = text.strip()
    if not text:
        return []
    if is_cjk:
        parts = re.split(r"([。？！；\n]+)", text)
    else:
        parts = re.split(r"([\.\?!;\n]+(?:\s+|$))", text)
    clauses = []
    for idx in range(0, len(parts), 2):
        chunk = parts[idx].strip()
        punct = parts[idx + 1].strip() if idx + 1 < len(parts) else ""
        if chunk:
            clause = (chunk + punct).strip()
            if clause:
                clauses.append(clause)
    if not clauses and text:
        clauses = [text]
    return clauses


def _tokens(text: str) -> list[str]:
    # Unbreakable display atoms (see core/subtitle_measure): whole Latin words, names that
    # travel together inside CJK text, phrase-level CJK units, punctuation attached.
    return atoms(text)


def _lines(text: str, cap: int) -> list[str]:
    # Width-aware, balanced rows within `cap` characters; an unbreakable token raises
    # ValueError instead of being cut or torn.
    return break_lines(text, cap)


def _wrap_lines(text: str, max_chars: int = 22, max_lines: int = 2, is_cjk: bool = True) -> str:
    lines = _lines(text, max_chars)
    if len(lines) > max_lines:
        raise ValueError("Subtitle requires additional display cues; truncation is forbidden")
    return "\n".join(lines)


def _display_parts(text: str, cap: int) -> list[str]:
    # Split by punctuation first; further overflow becomes additional cues.
    parts = []
    for clause in _split_text_by_punctuation(text, is_cjk=bool(re.search(r"[\u3400-\u9fff]", text))):
        lines = _lines(clause, cap)
        parts.extend("\n".join(lines[i:i+2]) for i in range(0, len(lines), 2))
    return parts


def _refine(parts: list[str], count: int, cap: int) -> list[str]:
    parts = list(parts)
    while len(parts) < count:
        candidates = [(len(_tokens(p)), i) for i, p in enumerate(parts) if len(_tokens(p)) > 1]
        if not candidates:
            # Only CJK text may be cut between glyphs; a Latin word or name is never torn.
            char_candidates = [(len(p.strip()), i) for i, p in enumerate(parts)
                               if len(p.strip()) > 1 and has_cjk(p)]
            if not char_candidates:
                while len(parts) < count:
                    parts.append('')
                break
            _, i = max(char_candidates)
            text_val = parts[i].strip()
            mid = len(text_val) // 2
            parts[i:i+1] = [_wrap_lines(text_val[:mid], cap), _wrap_lines(text_val[mid:], cap)]
            continue
        _, i = max(candidates)
        tokens = _tokens(parts[i].replace('\n', ' '))
        midpoint = len(tokens) // 2
        parts[i:i+1] = [_wrap_lines(''.join(tokens[:midpoint]), cap),
                         _wrap_lines(''.join(tokens[midpoint:]), cap)]
    return parts


def _resolve_line_caps(
    *, width: int | None = None, height: int | None = None, is_portrait: bool | None = None
) -> tuple[int, int]:
    if width is not None and height is not None:
        return compute_subtitle_line_limits(width, height)
    if is_portrait is True:
        return 14, 24
    dims = probe_video_dimensions()
    return compute_subtitle_line_limits(*(dims or (None, None)))


def build_cues(tts_results: list[dict], *, width: int | None = None, height: int | None = None, is_portrait: bool | None = None) -> list[dict]:
    cjk_cap, latin_cap = _resolve_line_caps(width=width, height=height, is_portrait=is_portrait)
    cues = []
    for row in tts_results:
        source, target = str(row.get("origin", "")).strip(), str(row.get("text", "")).strip()
        if not source or not target:
            raise ValueError("Empty subtitle source or translation")
        start, end = float(row["final_start"]), float(row["final_end"])
        if not math.isfinite(start + end) or start < 0 or end <= start:
            raise ValueError("Invalid subtitle clock")
        src_cjk = bool(re.search(r"[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]", source))
        tgt_cjk = bool(re.search(r"[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]", target))
        src_cap, tgt_cap = (cjk_cap if src_cjk else latin_cap), (cjk_cap if tgt_cjk else latin_cap)
        aa, bb = _display_parts(source, src_cap), _display_parts(target, tgt_cap)
        count = max(len(aa), len(bb))
        aa, bb = _refine(aa, count, src_cap), _refine(bb, count, tgt_cap)
        normalize = lambda value: re.sub(r"\s+", "", value)
        if normalize("".join(aa)) != normalize(source) or normalize("".join(bb)) != normalize(target):
            raise ValueError("Subtitle text coverage changed")
        weights = [max(1, len(normalize(part))) for part in bb]
        elapsed = 0
        for a, b, weight in zip(aa, bb, weights):
            cue_start = start + (end-start)*elapsed/sum(weights)
            elapsed += weight
            cue_end = start + (end-start)*elapsed/sum(weights)
            if round(cue_end*1000) <= round(cue_start*1000):
                raise ValueError("Subtitle interval cannot be represented at millisecond resolution")
            cues.append(dict(index=len(cues)+1, start=cue_start, end=cue_end, src=a, tgt=b,
                             unit_id=row.get("unit_id"), number=row.get("number")))
    return cues


def build_fast_subtitles(tts_results: list[dict], cum_extension: float = 0.0, *, width: int | None = None, height: int | None = None, is_portrait: bool | None = None) -> None:
    out_dir = Path("output")
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    cjk_cap, latin_cap = _resolve_line_caps(width=width, height=height, is_portrait=is_portrait)
    cues = build_cues(tts_results, width=width, height=height, is_portrait=is_portrait)
    import json
    (out_dir / "log").mkdir(exist_ok=True)
    (out_dir / "log/fast_subtitle_coverage.json").write_text(json.dumps({
        "policy": "fast-v2", "status": "pass", "units": len(tts_results), "cues": cues,
        "layout_limits": [cjk_cap, latin_cap],
        "basis": "exact_ordered_text_coverage_and_display_bounds",
    }, ensure_ascii=False, indent=2))

    def _write_srt_file(path: Path, lines_builder):
        with path.open("w", encoding="utf-8") as f:
            for c in cues:
                st_str = seconds_to_srt_time(c["start"])
                et_str = seconds_to_srt_time(c["end"])
                content = lines_builder(c)
                f.write(f"{c['index']}\n{st_str} --> {et_str}\n{content}\n\n")

    _write_srt_file(out_dir / "trans.srt", lambda c: c["tgt"])
    _write_srt_file(out_dir / "dub.srt", lambda c: c["tgt"])
    _write_srt_file(audio_dir / "trans_subs_for_audio.srt", lambda c: c["tgt"])

    _write_srt_file(out_dir / "src.srt", lambda c: c["src"])
    _write_srt_file(out_dir / "dub_orig.srt", lambda c: c["src"])
    _write_srt_file(audio_dir / "src_subs_for_audio.srt", lambda c: c["src"])

    _write_srt_file(out_dir / "dub_bilingual.srt", lambda c: f"{c['src']}\n{c['tgt']}")
    _write_srt_file(out_dir / "src_trans.srt", lambda c: f"{c['src']}\n{c['tgt']}")
    _write_srt_file(out_dir / "trans_src.srt", lambda c: f"{c['tgt']}\n{c['src']}")

    timeline_rows = [
        {
            "LineID": c["index"],
            "Source": c["src"].replace('\n', ' '),
            "Translation": c["tgt"].replace('\n', ' '),
            "start": c["start"],
            "end": c["end"],
            "duration": c["end"] - c["start"],
            "timestamp": f"({c['start']:.3f}, {c['end']:.3f})",
        }
        for c in cues
    ]
    tdf = pd.DataFrame(timeline_rows)
    tdf.to_excel(out_dir / "final_timeline.xlsx", index=False)
    tdf.to_excel(audio_dir / "final_timeline.xlsx", index=False)
    tdf.to_excel(audio_dir / "dub_display_timeline.xlsx", index=False)

    rprint(f"[bold green]✅ Fast bilingual subtitles generated successfully: {len(cues)} cues.[/bold green]")
