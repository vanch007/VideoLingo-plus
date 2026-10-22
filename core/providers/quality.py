from __future__ import annotations

import re
from difflib import SequenceMatcher
from functools import lru_cache


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


_ZH_DIGITS_MAP = {'零': '0', '一': '1', '二': '2', '两': '2', '三': '3', '四': '4', '五': '5', '六': '6', '七': '7', '八': '8', '九': '9'}


def normalize_chinese_number_words(text: str) -> str:
    def replace_decimal(match):
        int_part = match.group(1)
        frac_part = match.group(2)
        int_digits = ''.join(_ZH_DIGITS_MAP.get(c, c) for c in int_part)
        frac_digits = ''.join(_ZH_DIGITS_MAP.get(c, c) for c in frac_part)
        return f"{int_digits}.{frac_digits}"
    text = re.sub(r"([零一二两三四五六七八九]+)点([零一二两三四五六七八九]+)", replace_decimal, text or "")
    def replace_integer(match):
        phrase = match.group().replace('〇','零').replace('兩','两').replace('萬','万').replace('億','亿')
        scales = {'十':10,'百':100,'千':1000,'万':10000,'亿':100000000}
        if not any(char in scales for char in phrase):
            return ''.join(_ZH_DIGITS_MAP[char] for char in phrase)
        total, section, digit = 0, 0, 0
        for char in phrase:
            if char in _ZH_DIGITS_MAP:
                digit = int(_ZH_DIGITS_MAP[char])
            else:
                scale = scales[char]
                if scale < 10000:
                    section += (digit or 1) * scale
                elif scale == 10000:
                    total += (section + digit or 1) * scale
                    section = 0
                else:
                    total = (total + section + digit or 1) * scale
                    section = 0
                digit = 0
        return str(total + section + digit)
    return re.sub(r'[零〇一二两兩三四五六七八九十百千万萬亿億]+', replace_integer, text)


def spoken_numeric_values(text: str):
    """Exact quantities across CJK/digits/English, including CJK boundaries."""
    from decimal import Decimal
    from core.translation_invariants import number_annotations
    text = normalize_chinese_number_words(text or '')
    text = re.sub(r'(?<=[\u3400-\u9fff])(?=\d)|(?<=\d)(?=[\u3400-\u9fff])', ' ', text)
    return [Decimal(item['value']) for item in number_annotations(text)]


def normalize_for_content_score(text: str) -> str:
    text = (text or "").lower()
    text = normalize_vietnamese_number_words(text)
    text = normalize_chinese_number_words(text)
    from core.translation_invariants import number_annotations
    from decimal import Decimal
    for item in number_annotations(text):
        canonical = format(Decimal(item["value"]).normalize(), "f")
        text = re.sub(r"(?<!\w)" + re.escape(item["text"]) + r"(?!\w)", canonical, text)
    text = re.sub(r"\s+", "", text)
    cleaned = re.sub(r"[^\w\u00c0-\u024f\u1e00-\u1eff\u4e00-\u9fff]", "", text)
    for variant, canonical in [
        ("wanna", "wantto"),
        ("gonna", "goingto"),
    ]:
        cleaned = cleaned.replace(variant, canonical)
    return cleaned


@lru_cache(maxsize=1)
def _english_pronunciations():
    try:
        import cmudict
        return cmudict.dict()
    except ImportError:
        return {}


def _homophone_transcript(expected: str, actual: str) -> str:
    """Exact dictionary homophones are spelling uncertainty, not lost audio.

    Do not fuzzily excuse near names or a different final consonant. The raw
    transcript remains in the report; only this comparison uses equal sounds.
    """
    dictionary = _english_pronunciations()
    forms = {}
    for word in re.findall(r"[a-z]+(?:'[a-z]+)?", expected.lower()):
        for phones in dictionary.get(word, []):
            forms.setdefault(tuple(phones), word)
    def canonical(match):
        word = match.group().lower()
        return next((forms[tuple(p)] for p in dictionary.get(word, []) if tuple(p) in forms), word)
    return re.sub(r"[a-z]+(?:'[a-z]+)?", canonical, actual.lower())


def _cross_script_phonetic_normalize(text: str) -> str:
    """Romanize CJK ASR output so spoken names can match Latin subtitles."""
    if not re.search(r"[\u4e00-\u9fff]", text or ""):
        return normalize_for_content_score(text)
    try:
        from pypinyin import lazy_pinyin

        return normalize_for_content_score(" ".join(lazy_pinyin(text or "")))
    except ImportError:
        return normalize_for_content_score(text)


@lru_cache(maxsize=8192)
def mandarin_reading(text: str) -> tuple[str, ...] | None:
    """Exact tone-bearing Mandarin readings, never a fuzzy sound similarity.

    ASR cannot distinguish 它/他 or 初版/出版 in an audio-only check. Keep the
    written translation for semantic review; compare equal sounds here. Tone,
    occurrence count and ordering stay significant. Latin spelling stays exact.
    """
    if not re.search(r"[\u3400-\u4dbf\u4e00-\u9fff]", text or ''):
        return None
    try:
        from pypinyin import lazy_pinyin, Style
    except ImportError:
        return None
    tokens = re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff]+|[A-Za-z0-9]+(?:['’][A-Za-z]+)?", text.casefold())
    readings = []
    for token in tokens:
        if re.fullmatch(r"[\u3400-\u4dbf\u4e00-\u9fff]+", token):
            readings.extend('zh:' + value for value in lazy_pinyin(token, style=Style.TONE3,
                neutral_tone_with_five=True))
        else:
            readings.append('literal:' + token)
    return tuple(readings)


def content_similarity(expected: str, actual: str) -> float:
    expected_norm = normalize_for_content_score(expected)
    actual_norm = normalize_for_content_score(actual)
    if not expected_norm and not actual_norm:
        return 1.0
    if not expected_norm or not actual_norm:
        return 0.0
    direct = SequenceMatcher(None, expected_norm, actual_norm).ratio()
    score = direct
    expected_reading, actual_reading = mandarin_reading(expected), mandarin_reading(actual)
    if expected_reading and expected_reading == actual_reading:
        score = 1.0
    homophones = normalize_for_content_score(_homophone_transcript(expected, actual))
    score = max(score, SequenceMatcher(None, expected_norm, homophones).ratio())
    cross_script = bool(re.search(r"[\u4e00-\u9fff]", expected or "")) != bool(
        re.search(r"[\u4e00-\u9fff]", actual or "")
    )
    if cross_script:
        phonetic_expected = _cross_script_phonetic_normalize(expected)
        phonetic_actual = _cross_script_phonetic_normalize(actual)
        score = max(score, SequenceMatcher(None, phonetic_expected, phonetic_actual).ratio())
    return score


def reference_leak_score(reference_text: str | None, actual: str, expected: str | None = None) -> float:
    reference_text = reference_text or ""
    actual = actual or ""
    if expected is not None:
        # Approved words (e.g. product names kept in a Chinese translation)
        # cannot prove reference leakage. Score only extra observed speech.
        observed = normalize_for_content_score(actual)
        wanted = normalize_for_content_score(expected)
        actual = ' '.join(observed[j1:j2] for tag, _, _, j1, j2 in
                          SequenceMatcher(None, wanted, observed, autojunk=False).get_opcodes()
                          if tag in {'insert', 'replace'})
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
