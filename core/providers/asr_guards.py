"""Model-independent sanity checks for one recognized chunk.

LLM-style recognizers (MOSS, Qwen3-ASR) can loop on one phrase, skip speech, or answer in the
wrong script. These pure functions only *detect*; the caller re-recognizes a smaller window and
records chunks that stay suspicious as ``needs_review`` instead of accepting them silently.
"""
from __future__ import annotations

import re
import unicodedata

REPEAT_MIN_CHARS = 40       # shorter texts are never judged repetitive
REPEAT_COVERAGE = 0.5       # share of the text covered by a looping unit
REPEAT_MAX_UNIT = 120       # longest looping unit (characters)
SPARSE_CHARS_PER_SECOND = 2.0
SPARSE_MIN_SECONDS = 20.0   # shorter windows are too noisy to judge
SPARSE_MIN_SPEECH_SHARE = 0.3
SCRIPT_MIN_LETTERS = 100
SCRIPT_MIN_SHARE = 0.2
JA_MIN_KANA_SHARE = 0.05    # Japanese without kana is Chinese
ZH_MAX_KANA_SHARE = 0.2     # "Chinese" with this much kana is Japanese

_REPEATED_UNIT = re.compile(rf"(.{{1,{REPEAT_MAX_UNIT}}}?)\1{{3,}}", re.S)
_NOISE = re.compile(r"[\W_]+", re.U)


def _normalized(text: str) -> str:
    return _NOISE.sub("", text or "").lower()


def repeated_share(text: str) -> float:
    """Share of the normalized text consumed by a unit repeated 4+ times in a row."""
    norm = _normalized(text)
    if not norm:
        return 0.0
    covered = sum(len(m.group()) for m in _REPEATED_UNIT.finditer(norm))
    return covered / len(norm)


def is_repetitive(text: str) -> bool:
    norm = _normalized(text)
    return len(norm) >= REPEAT_MIN_CHARS and repeated_share(text) >= REPEAT_COVERAGE


def is_sparse(text: str, seconds: float, speech_seconds: float | None) -> bool:
    """Little text although the audio is mostly active (speech dropped)."""
    if speech_seconds is None or seconds < SPARSE_MIN_SECONDS:
        return False
    if speech_seconds < SPARSE_MIN_SPEECH_SHARE * seconds:
        return False
    return len(_normalized(text)) / seconds < SPARSE_CHARS_PER_SECOND


def _script_of(ch: str) -> str | None:
    if not ch.isalpha():
        return None
    code = ord(ch)
    if 0x3040 <= code <= 0x30FF:
        return "Kana"
    if 0x3400 <= code <= 0x9FFF or 0xF900 <= code <= 0xFAFF:
        return "Han"
    if 0xAC00 <= code <= 0xD7AF or 0x1100 <= code <= 0x11FF:
        return "Hangul"
    name = unicodedata.name(ch, "")
    if name.startswith("LATIN"):
        return "Latin"
    return "Other"


def script_counts(text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for ch in text or "":
        script = _script_of(ch)
        if script:
            counts[script] = counts.get(script, 0) + 1
    return counts


_EXPECTED = {"zh": {"Han"}, "yue": {"Han"}, "ja": {"Han", "Kana"}, "ko": {"Hangul"}}


def script_mismatch(text: str, language: str) -> str | None:
    """A reason when the letters of ``text`` do not belong to ``language``; None when fine."""
    counts = script_counts(text)
    total = sum(counts.values())
    if total < SCRIPT_MIN_LETTERS:
        return None
    expected = _EXPECTED.get(language, {"Latin"})
    if sum(counts.get(s, 0) for s in expected) / total < SCRIPT_MIN_SHARE:
        return "wrong_script"
    cjk = counts.get("Han", 0) + counts.get("Kana", 0)
    if language == "ja" and cjk and counts.get("Kana", 0) / cjk < JA_MIN_KANA_SHARE:
        return "japanese_without_kana"
    if language in {"zh", "yue"} and cjk and counts.get("Kana", 0) / cjk > ZH_MAX_KANA_SHARE:
        return "chinese_with_kana"
    return None


def assess(text: str, seconds: float, language: str, speech_seconds: float | None = None) -> list[str]:
    """Reasons a recognized chunk is suspicious (empty list = accept)."""
    reasons = []
    if is_repetitive(text):
        reasons.append("repetition_loop")
    if is_sparse(text, seconds, speech_seconds):
        reasons.append("sparse_text")
    mismatch = script_mismatch(text, language)
    if mismatch:
        reasons.append(mismatch)
    return reasons
