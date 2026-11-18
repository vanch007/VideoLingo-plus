"""Bilingual burned-in subtitles as one ASS script with explicit stacking.

Two independent ``subtitles=`` filters place the source above the translation with a margin
*guessed* from an assumed number of translation rows, so a translation that wraps to two rows
can run into the source line. Here every row is broken by us (core.subtitle_measure) and the
source is positioned above the translation rows actually shown at that moment, so the two
tracks cannot overlap. Wrapping is off in the player (WrapStyle 2); nothing is ever cut.

Geometry matches the libass SRT path the project used before: PlayResY = 288 and PlayResX = 288 *
aspect, so FontSize values keep their meaning.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from core.subtitle_measure import break_lines, has_cjk

# One text row occupies about this many font sizes (font ascent + descent + line gap).
LINE_HEIGHT = 1.08
ROW_GAP = 6  # PlayRes units between the two tracks


@dataclass(frozen=True)
class TrackStyle:
    font: str
    size: int
    primary: str = "&H00FFFFFF"
    outline_colour: str = "&H00000000"
    outline: float = 1.0
    shadow: float = 0.0
    back_colour: str = "&H00000000"
    border_style: int = 1
    margin_v: int = 16          # distance of the lowest row from the bottom edge
    margin_lr: int = 20


def ass_colour(value: str, default: str = "&H00FFFFFF") -> str:
    """&HBBGGRR, &HAABBGGRR or #RRGGBB (as used in config.yaml) -> &HAABBGGRR."""
    raw = str(value or "").strip()
    if raw.startswith("#") and len(raw) == 7:
        return f"&H00{raw[5:7]}{raw[3:5]}{raw[1:3]}".upper()
    digits = raw.strip("&").upper().lstrip("H")
    if not re.fullmatch(r"[0-9A-F]{6}|[0-9A-F]{8}", digits):
        return default
    return "&H" + (digits if len(digits) == 8 else "00" + digits)


_TIME = re.compile(r"(\d+):(\d\d):(\d\d)[,.](\d{1,3})")


def _seconds(text: str) -> float:
    h, m, s, ms = _TIME.search(text).groups()
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")) / 1000


def parse_srt(content: str) -> list[tuple[float, float, str]]:
    cues = []
    for block in re.split(r"\n\s*\n", content.replace("\r\n", "\n").strip()):
        lines = [l for l in block.split("\n")]
        stamp = next((i for i, l in enumerate(lines) if "-->" in l), None)
        if stamp is None:
            continue
        a, b = lines[stamp].split("-->")
        text = "\n".join(l.strip() for l in lines[stamp + 1:] if l.strip())
        if text:
            cues.append((_seconds(a), _seconds(b), text))
    return cues


def read_srt(path: str) -> list[tuple[float, float, str]]:
    with open(path, encoding="utf-8-sig") as handle:
        return parse_srt(handle.read())


def _time(seconds: float) -> str:
    cs = max(0, round(seconds * 100))
    return f"{cs // 360000}:{cs // 6000 % 60:02d}:{cs // 100 % 60:02d}.{cs % 100:02d}"


def _rows(text: str, cap: int) -> list[str]:
    """Rows of one cue: authored line breaks are kept, a line over the row cap is broken."""
    out: list[str] = []
    for line in text.split("\n"):
        out.extend(break_lines(line, cap, lenient=True) or [""])
    return out


def _escape(row: str) -> str:
    return row.replace("\\", "").replace("{", "(").replace("}", ")")


def build_ass(width: int, height: int, src_cues, trans_cues, src: TrackStyle | None, trans: TrackStyle,
              caps: tuple[int, int]) -> str:
    """ASS script. ``src`` None = translation only. caps = (cjk_cap, latin_cap) characters per row."""
    play_y = 288
    play_x = max(1, round(play_y * width / max(height, 1)))

    def cap_for(text: str) -> int:
        return caps[0] if has_cjk(text) else caps[1]

    def style_line(name: str, s: TrackStyle) -> str:
        return (f"Style: {name},{s.font},{s.size},{ass_colour(s.primary)},&H000000FF,{ass_colour(s.outline_colour)},"
                f"{ass_colour(s.back_colour, '&H00000000')},0,0,0,0,100,100,0,0,{s.border_style},{s.outline},{s.shadow},"
                f"2,{s.margin_lr},{s.margin_lr},{s.margin_v},1")

    lines = ["[Script Info]", "ScriptType: v4.00+", f"PlayResX: {play_x}", f"PlayResY: {play_y}",
             "WrapStyle: 2", "ScaledBorderAndShadow: yes", "", "[V4+ Styles]",
             "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, "
             "Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
             "Alignment, MarginL, MarginR, MarginV, Encoding", style_line("Trans", trans)]
    if src is not None:
        lines.append(style_line("Src", src))
    lines += ["", "[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"]

    trans_rows = [(a, b, _rows(t, cap_for(t))) for a, b, t in trans_cues]
    for a, b, rows in trans_rows:
        lines.append(f"Dialogue: 1,{_time(a)},{_time(b)},Trans,,0,0,{trans.margin_v},,"
                     + "\\N".join(_escape(r) for r in rows))

    if src is not None:
        gap = ROW_GAP
        for a, b, text in src_cues:
            rows = _rows(text, cap_for(text))
            # Cut the source cue where the translation changes: the stack height is constant per piece.
            marks = sorted({a, b, *[t for s, e, _ in trans_rows for t in (s, e) if a < t < b]})
            for left, right in zip(marks, marks[1:]):
                heights = [len(r) for s, e, r in trans_rows if s < right and e > left]
                stack = max(heights + [1])
                margin = trans.margin_v + round(stack * trans.size * LINE_HEIGHT) + gap
                lines.append(f"Dialogue: 0,{_time(left)},{_time(right)},Src,,0,0,{margin},,"
                             + "\\N".join(_escape(r) for r in rows))
    return "\n".join(lines) + "\n"


def ass_filter(path: str, fonts_dir: str | None = None) -> str:
    """ffmpeg filter expression for an ASS file (absolute path, quotes escaped)."""
    import os
    escaped = os.path.abspath(path).replace("\\", "\\\\").replace("'", "'\\''").replace(":", "\\:")
    expr = f"ass=filename='{escaped}'"
    if fonts_dir:
        expr += ":fontsdir='" + os.path.abspath(fonts_dir).replace("'", "'\\''") + "'"
    return expr


def burn_filter(width: int, height: int, src_srt: str | None, trans_srt: str, src: TrackStyle | None,
                trans: TrackStyle, out_ass: str) -> str:
    """Write the bilingual ASS next to the renders and return its ffmpeg filter expression."""
    from core.subtitle_layout import compute_subtitle_line_limits
    caps = compute_subtitle_line_limits(width, height)
    src_cues = read_srt(src_srt) if (src is not None and src_srt) else []
    with open(out_ass, "w", encoding="utf-8") as handle:
        handle.write(build_ass(width, height, src_cues, read_srt(trans_srt), src if src_cues else None, trans, caps))
    return ass_filter(out_ass)
