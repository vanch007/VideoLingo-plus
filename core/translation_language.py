"""Human language names for translation prompts; retain locale distinctions."""

LANGUAGE_NAMES = {
    "zh": "简体中文", "zh-cn": "简体中文", "zh-sg": "简体中文",
    "zh-tw": "繁體中文", "zh-hk": "繁體中文",
    "en": "英语", "vi": "越南语", "id": "印度尼西亚语", "ms": "马来语",
    "ja": "日语", "ko": "韩语", "th": "泰语", "fr": "法语", "de": "德语",
    "es": "西班牙语", "pt": "葡萄牙语", "ru": "俄语", "ar": "阿拉伯语",
    "it": "意大利语", "hi": "印地语", "tr": "土耳其语",
    "hy": "Armenian", "az": "Azerbaijani", "be": "Belarusian", "bs": "Bosnian",
    "bg": "Bulgarian", "my": "Burmese", "ca": "Catalan", "hr": "Croatian",
    "cs": "Czech", "da": "Danish", "nl": "Dutch", "et": "Estonian", "fi": "Finnish",
    "gl": "Galician", "el": "Greek", "gu": "Gujarati", "he": "Hebrew", "hu": "Hungarian",
    "is": "Icelandic", "kn": "Kannada", "kk": "Kazakh", "km": "Khmer", "lo": "Lao",
    "lv": "Latvian", "lt": "Lithuanian", "mk": "Macedonian", "ml": "Malayalam",
    "mr": "Marathi", "mn": "Mongolian", "ne": "Nepali", "no": "Norwegian",
    "fa": "Persian", "pl": "Polish", "ro": "Romanian", "sr": "Serbian",
    "sk": "Slovak", "sl": "Slovenian", "sv": "Swedish", "ta": "Tamil",
    "te": "Telugu", "uk": "Ukrainian", "ur": "Urdu", "cy": "Welsh",
    "auto": "the detected source language",
}


def translation_language_name(value: str) -> str:
    code = str(value).strip().lower().replace("_", "-")
    if code in LANGUAGE_NAMES:
        return LANGUAGE_NAMES[code]
    base = code.split("-", 1)[0]
    if base in LANGUAGE_NAMES:
        return f"{LANGUAGE_NAMES[base]} ({value})"
    return str(value)  # Freeform language/style descriptions remain supported.
