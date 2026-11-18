"""Display-width measurement and line breaking for burned-in subtitles.

Unit: one CJK glyph = 1.0 em. Latin letters, digits and punctuation are narrower, so a
character count over- or under-states how much of a row a line fills. Lines are broken
between *atoms* (whole Latin words, phrase-level CJK units, never inside a name) and are
balanced across rows. Nothing is ever dropped: the atoms of a text always concatenate back
to that text (whitespace aside).

BudouX is optional. With it, Chinese/Japanese lines break between phrases; without it a CJK
atom is one glyph plus its attached punctuation, which is the previous behaviour.
"""
from __future__ import annotations

import functools
import re
import unicodedata

# Average Latin advance relative to one CJK em; the length caps in subtitle_layout
# (22 CJK / 42 Latin on a landscape row) are in this ratio.
LATIN_EM = 22 / 42

CJK_RE = re.compile(r"[㐀-鿿぀-ヿ가-힯豈-﫿]")
_OPENERS = set("（([{「『《〈【“‘")
_CLOSERS = set("，。？！；：、）)]}」』》〉】”’,.?!;:%…—")
_NARROW = set("iljtfIr.,;:'’!|()[]\" -")
_WIDE_LATIN = set("mwMW@")
# Letters/digits of any alphabet (é, ä, ñ, ß ...) but never CJK, which is broken per phrase.
_L = r"(?:(?![㐀-鿿぀-ヿ가-힯豈-﫿])[^\W_])"
_WORD = re.compile(rf"{_L}+(?:[’'._\-/]{_L}+)*")


def has_cjk(text: str) -> bool:
    return bool(CJK_RE.search(text))


# Measured against libass/PingFang: the plain model ran 2-5% narrower than real renders.
LATIN_CALIBRATION = 1.04


def char_width(ch: str) -> float:
    if ch in "\n\r":
        return 0.0
    if unicodedata.east_asian_width(ch) in ("W", "F"):
        return 1.0
    return _latin_width(ch) * LATIN_CALIBRATION


def _latin_width(ch: str) -> float:
    if ch == " ":
        return 0.3
    if ch in _NARROW:
        return 0.3
    if ch in _WIDE_LATIN:
        return 0.85
    if ch.isupper():
        return 0.64
    if ch.isdigit() or ch.islower():
        return 0.54
    return 0.6


def text_width(text: str) -> float:
    return sum(char_width(c) for c in text)


@functools.lru_cache(maxsize=1)
def _budoux_parser():
    try:  # optional dependency
        import budoux
        return budoux.load_default_simplified_chinese_parser()
    except Exception:
        return None


def _cjk_units(run: str) -> list[str]:
    """Phrase-level units of a CJK run (BudouX when installed, else single glyphs)."""
    parser = _budoux_parser()
    if parser is not None and len(run) > 1:
        try:
            units = parser.parse(run)
            if "".join(units) == run:
                return units
        except Exception:
            pass
    return list(run)


def atoms(text: str) -> list[str]:
    """Unbreakable display units; leading whitespace is kept on the unit that follows it.

    Latin words stay whole. In text containing CJK, consecutive Latin words ("Epidemic
    Sounds", "Claude Code") travel together. Closing punctuation never starts a unit and
    opening punctuation never ends one.
    """
    text = text.strip()
    if not text:
        return []
    mixed = has_cjk(text)
    raw: list[str] = []
    pos, n = 0, len(text)
    while pos < n:
        space = re.match(r"\s*", text[pos:]).group()
        start = pos + len(space)
        if start >= n:
            break
        word = _WORD.match(text, start)
        if word:
            raw.append(space + word.group())
            pos = word.end()
        elif CJK_RE.match(text[start]):
            end = start
            while end < n and CJK_RE.match(text[end]):
                end += 1
            units = _cjk_units(text[start:end])
            units[0] = space + units[0]
            raw.extend(units)
            pos = end
        else:
            raw.append(space + text[start])
            pos = start + 1

    merged: list[str] = []
    for unit in raw:
        core = unit.strip()
        if merged and core and core[0] in _CLOSERS:
            merged[-1] += unit
        elif merged and merged[-1].strip() and merged[-1].strip()[-1] in _OPENERS:
            merged[-1] += unit
        elif (mixed and merged and unit[:1].isspace() and _WORD.fullmatch(core)
              and re.fullmatch(rf"\s*{_L}(?:{_L}|[’'._\-/ ])*", merged[-1])):
            merged[-1] += unit  # "Claude" + " Code": a name travels together
        else:
            merged.append(unit)
    return merged


def _break_atom(atom: str, capacity: float, cap_chars: int) -> list[str]:
    raise ValueError(f"Unbreakable subtitle token wider than one row: {atom.strip()!r}")


def _hard_split(atom: str, capacity: float, cap_chars: int) -> list[str]:
    """Last resort for a URL / compound word wider than a row: cut after separators, then by glyph."""
    pieces: list[str] = []
    for part in re.findall(r"[^/._\-?&=]*[/._\-?&=]?", atom.strip()):
        if not part:
            continue
        cur = ""
        for ch in part:
            if cur and (text_width(cur + ch) > capacity or len(cur + ch) > cap_chars):
                pieces.append(cur)
                cur = ""
            cur += ch
        if cur:
            pieces.append(cur)
    packed: list[str] = []
    for piece in pieces:  # re-join neighbours that still fit one row
        if packed and text_width(packed[-1] + piece) <= capacity and len(packed[-1] + piece) <= cap_chars:
            packed[-1] += piece
        else:
            packed.append(piece)
    return [(" " if i == 0 and atom[:1].isspace() else "") + p for i, p in enumerate(packed)]


def line_capacity(text: str, cap_chars: int) -> float:
    """Row capacity in em for a text whose length cap is cap_chars (CJK glyphs or Latin chars)."""
    return float(cap_chars) if has_cjk(text) else cap_chars * LATIN_EM


def break_lines(text: str, cap_chars: int, lenient: bool = False) -> list[str]:
    """Break text into the fewest rows within cap_chars and the row width, balanced.

    A row never exceeds cap_chars characters nor the em capacity. A single atom that cannot
    fit a row raises ValueError('Unbreakable ...') instead of being cut, unless ``lenient`` (render
    time only): then it is split after separators / by glyph so the burn never fails on a URL.
    """
    items = atoms(text)
    if not items:
        return []
    capacity = line_capacity(text, cap_chars) + 1e-6
    # A grouped name is a preference: when it cannot fit a row it falls back to its words.
    expanded: list[str] = []
    for a in items:
        if (text_width(a.strip()) > capacity or len(a.strip()) > cap_chars) and re.search(r"\S\s+\S", a.strip()):
            expanded.extend(re.findall(r"\s*\S+", a))
        else:
            expanded.append(a)
    if lenient:
        fitted: list[str] = []
        for a in expanded:
            if text_width(a.strip()) > capacity or len(a.strip()) > cap_chars:
                fitted.extend(_hard_split(a, capacity, cap_chars))
            else:
                fitted.append(a)
        expanded = fitted
    items = expanded
    widths = [text_width(a.strip()) for a in items]
    lens = [len(a.strip()) for a in items]
    for a, w, n in zip(items, widths, lens):
        if w > capacity or n > cap_chars:
            _break_atom(a, capacity, cap_chars)

    def span(i: int, j: int) -> tuple[float, int]:
        """Width and length of atoms[i:j] as one row (interior spaces kept)."""
        first = items[i].lstrip()
        body = first + "".join(items[i + 1:j])
        return text_width(body), len(body)

    count = len(items)
    # Fewest rows by greedy, then balance with DP over exactly that many rows.
    rows, i = 0, 0
    while i < count:
        j = i + 1
        while j < count and (lambda w, n: w <= capacity and n <= cap_chars)(*span(i, j + 1)):
            j += 1
        rows += 1
        i = j
    INF = float("inf")
    best = [[INF] * (count + 1) for _ in range(rows + 1)]
    back = [[0] * (count + 1) for _ in range(rows + 1)]
    best[0][0] = 0.0
    for r in range(1, rows + 1):
        for j in range(r, count + 1):
            for i in range(r - 1, j):
                if best[r - 1][i] == INF:
                    continue
                w, n = span(i, j)
                if w > capacity or n > cap_chars:
                    continue
                slack = capacity - w
                # Squared slack balances every row, the last one included, so no row is left with a
                # single orphan glyph ("…马匪" / "的"). A lone short row costs nothing: rows == 1.
                cost = best[r - 1][i] + slack * slack
                if cost < best[r][j]:
                    best[r][j], back[r][j] = cost, i
    lines, j = [], count
    for r in range(rows, 0, -1):
        i = back[r][j]
        lines.append(("".join([items[i].lstrip()] + items[i + 1:j])).strip())
        j = i
    return lines[::-1]
