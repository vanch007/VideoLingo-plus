"""Pronunciation estimates guide candidate search, never certify audio duration."""
from __future__ import annotations

import re


SMALL = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
TENS = "zero ten twenty thirty forty fifty sixty seventy eighty ninety".split()


def cardinal(number: int) -> list[str]:
    if number < 20:
        return [SMALL[number]]
    if number < 100:
        return [TENS[number // 10]] + (cardinal(number % 10) if number % 10 else [])
    if number < 1000:
        return cardinal(number // 100) + ["hundred"] + (cardinal(number % 100) if number % 100 else [])
    for scale, name in ((10**12,"trillion"),(10**9,"billion"),(10**6,"million"),(1000,"thousand")):
        if number >= scale:
            return cardinal(number // scale) + [name] + (cardinal(number % scale) if number % scale else [])


def english_spoken_words(text: str) -> list[str]:
    def number(match):
        parts = match.group().replace(",", "").split(".")
        words = cardinal(int(parts[0]))
        if len(parts) > 1:
            words += ["point", *[SMALL[int(d)] for d in parts[1]]]
        return " ".join(words)
    expanded = re.sub(r"\d+(?:,\d{3})*(?:\.\d+)?",number,text.replace("’","'"))
    return re.findall(r"[a-z]+(?:'[a-z]+)*",expanded.casefold())


def english_syllables(text: str) -> int:
    from core.providers.quality import _english_pronunciations
    dictionary = _english_pronunciations()
    total = 0
    for word in english_spoken_words(text):
        pronunciations = dictionary.get(word)
        if pronunciations:
            total += min(sum(phone[-1:].isdigit() for phone in phones) for phones in pronunciations)
        else:
            # Names absent from the dictionary keep an explicit approximate
            # count. Natural synthesis remains the only release criterion.
            count = len(re.findall(r"[aeiouy]+",word))
            if word.endswith("e") and not word.endswith(("le","ye")) and count > 1:
                count -= 1
            total += max(1,count)
    return total


def chinese_spoken_units(text: str) -> int:
    """Advisory Mandarin syllable count, including embedded English speech.

    Han characters normally occupy one syllable; an uninterrupted clause is
    not one word. Latin words use their pronunciation, not their letter count.
    Numeric estimates remain approximate; measured complete audio decides fit.
    """
    han = len(re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff]", text))
    return han + english_syllables(text)


def _nominal_question(phrase: str) -> bool:
    return bool(re.fullmatch(r"[^呢]{1,10}呢", phrase) and not re.search(
        r"为什么|怎么|如何|哪|谁|有|不|没|要|会|想|说|懂|听|叫|给|走|用|生气", phrase))


def source_form_conflicts(source: str, candidate: str, seconds: float) -> list[str]:
    """Keep fast nominal questions compact, including mixed repeated calls."""
    phrases = [p.strip() for p in re.split(r"[。！？!?；;，,]", source) if p.strip()]
    nominal = [p for p in phrases if _nominal_question(p)]
    rapid = (phrases and 0 < seconds <= 1.2 and len(nominal) == len(phrases)) or (
        len(nominal) >= 2 and 0 < seconds <= 1.5 * len(nominal))
    if (rapid
            and re.search(r"\bwhere\s+(?:is|are|was|were)\b|\bwhere['’]s\b", candidate, re.I)):
        return ["short_nominal_question_expanded: retain the named object as a compact noun question; do not expand it into Where is/are"]
    return []


def source_form_hint(source: str) -> str:
    phrases = [p.strip() for p in re.split(r"[。！？!?；;，,]",source) if p.strip()]
    if any(_nominal_question(p) for p in phrases):
        return ("源文含短促的‘名词/称谓＋呢’责问。对应部分必须保留名词式短问句，"
                "不用扩成 Where is/Where are the 的完整句；重复责问仍逐次保留。")
    if len(re.sub(r"\W","",source)) <= 5 and re.search(r"嘛[。！？!?]?$",source):
        return "这是极短的回应。保留核心词和语气，不增加 It's..., that's all, right? 等说明结构。"
    if any(re.fullmatch(r"(?:就|便)?是[\u4e00-\u9fff]{1,6}", phrase) for phrase in phrases):
        return ("源文含短促的名词定义或回应。该部分可直接说对应的核心词；"
                "不要增加 It's..., just, that's what it is 等说明框架。仍须保留其他分句的独立意思。")
    definition = re.match(r"([\u4e00-\u9fff]{1,6})就是", source)
    if definition and not re.search(r"我|你|他|她|它|这|那|怎么|什么|为什么|今天|明天|昨天", definition[1]):
        return ("源文开头是在解释一个名词的定义。可用该名词的简短标签加冒号引出定义，"
                "不用扩成 The ... is that / Here's what ... means。标签后的时间、行动者、动作、数量和结尾问句仍完整保留。")
    return ""


def contextual_need_replies(source: str, neighbors) -> list[dict]:
    """Bind a compact negated-need reply to an action stated nearby.

    This is source evidence for one reply fragment, not permission to omit
    other assertions or to accept the complete translation.
    """
    if not isinstance(neighbors, list):
        return []
    replies = []
    for phrase in re.split(r"[。！？!?；;，,]", source):
        phrase = phrase.strip()
        match = re.fullmatch(r"(?:不用|不必|无需|无须)([\u4e00-\u9fff]{1,8})", phrase)
        if match:
            witnesses = [text for text in neighbors if isinstance(text, str) and match[1] in text]
            if witnesses:
                replies.append({"source": phrase, "action": match[1], "context_witnesses": witnesses})
    return replies


def contextual_compaction_hint(source: str, context: str) -> str:
    import json
    try:
        data = json.loads(context)
    except (TypeError, ValueError):
        return ""
    if not isinstance(data, dict):
        return ""
    replies = contextual_need_replies(source, data.get("surrounding_source", []))
    if not replies:
        return ""
    phrases = "、".join(reply["source"] for reply in replies)
    return (f"邻句已明确说出当前短答‘{phrases}’所否定的动作，该短答可用上下文明确的 No need 表达，"
            "不用再次展开动作。仅省略已明确的动作名称，后面的独立断言和重复仍须完整保留。")
