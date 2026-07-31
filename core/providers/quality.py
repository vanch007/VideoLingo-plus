from __future__ import annotations

import re
from difflib import SequenceMatcher


VI_NUMBER_WORDS = {
    "hai mươi": "20",
    "mười chín": "19",
    "mười tám": "18",
    "mười bảy": "17",
    "mười sáu": "16",
    "mười lăm": "15",
    "mười bốn": "14",
    "mười ba": "13",
    "mười hai": "12",
    "mười một": "11",
    "mười": "10",
    "chín": "9",
    "tám": "8",
    "bảy": "7",
    "sáu": "6",
    "năm": "5",
    "bốn": "4",
    "ba": "3",
    "hai": "2",
    "một": "1",
    "không": "0",
}


def normalize_vietnamese_number_words(text: str) -> str:
    for phrase, digit in sorted(VI_NUMBER_WORDS.items(), key=lambda item: len(item[0]), reverse=True):
        text = re.sub(rf"\b{re.escape(phrase)}\b", digit, text)
    return text


def normalize_for_content_score(text: str) -> str:
    text = (text or "").lower()
    text = normalize_vietnamese_number_words(text)
    text = re.sub(r"\s+", "", text)
    return re.sub(r"[^\w\u00c0-\u024f\u1e00-\u1eff\u4e00-\u9fff]", "", text)


def _cross_script_phonetic_normalize(text: str) -> str:
    """Romanize CJK ASR output so spoken names can match Latin subtitles."""
    if not re.search(r"[\u4e00-\u9fff]", text or ""):
        return normalize_for_content_score(text)
    try:
        from pypinyin import lazy_pinyin

        return normalize_for_content_score(" ".join(lazy_pinyin(text or "")))
    except ImportError:
        return normalize_for_content_score(text)


def content_similarity(expected: str, actual: str) -> float:
    expected_norm = normalize_for_content_score(expected)
    actual_norm = normalize_for_content_score(actual)
    if not expected_norm and not actual_norm:
        return 1.0
    if not expected_norm or not actual_norm:
        return 0.0
    direct = SequenceMatcher(None, expected_norm, actual_norm).ratio()
    cross_script = bool(re.search(r"[\u4e00-\u9fff]", expected or "")) != bool(
        re.search(r"[\u4e00-\u9fff]", actual or "")
    )
    if not cross_script:
        return direct
    phonetic_expected = _cross_script_phonetic_normalize(expected)
    phonetic_actual = _cross_script_phonetic_normalize(actual)
    return max(direct, SequenceMatcher(None, phonetic_expected, phonetic_actual).ratio())


def reference_leak_score(reference_text: str | None, actual: str) -> float:
    reference_text = reference_text or ""
    actual = actual or ""
    ref_cjk = "".join(re.findall(r"[\u4e00-\u9fff]", reference_text))
    actual_cjk = "".join(re.findall(r"[\u4e00-\u9fff]", actual))
    if ref_cjk:
        if not actual_cjk:
            return 0.0
        matcher = SequenceMatcher(None, ref_cjk, actual_cjk)
        return max((block.size for block in matcher.get_matching_blocks()), default=0) / max(len(ref_cjk), 1)

    ref_norm = re.sub(r"\d+", "", normalize_for_content_score(reference_text))
    actual_norm = re.sub(r"\d+", "", normalize_for_content_score(actual))
    if not ref_norm or not actual_norm:
        return 0.0
    longest = 0
    matcher = SequenceMatcher(None, ref_norm, actual_norm)
    for block in matcher.get_matching_blocks():
        longest = max(longest, block.size)
    return longest / max(len(ref_norm), 1)
