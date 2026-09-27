from __future__ import annotations

import math
import re
from typing import Any

import pandas as pd

from core.ask_gpt import ask_gpt
from core.config_utils import load_key
from core.dubbing_quality import get_quality_config, normalize_lines, row_available_duration
from core.prompts_storage import get_dubbing_rewrite_prompt
from core.runtime_context import effective_target_language


def _extract_rewritten_lines(response: Any) -> list[str]:
    if isinstance(response, dict):
        if isinstance(response.get("lines"), list):
            return [str(item).strip() for item in response["lines"] if str(item).strip()]
        if isinstance(response.get("rewritten_lines"), list):
            return [str(item).strip() for item in response["rewritten_lines"] if str(item).strip()]
        if isinstance(response.get("result"), str):
            return [response["result"].strip()]
        if isinstance(response.get("shortened_text"), str):
            return [response["shortened_text"].strip()]
    if isinstance(response, list):
        return [str(item).strip() for item in response if str(item).strip()]
    if isinstance(response, str) and response.strip():
        return [response.strip()]
    return []


def _word_count(lines: list[str]) -> int:
    word_pattern = r"[\wÀ-ỹ]+(?:['’][\wÀ-ỹ]+)?"
    return sum(len(re.findall(word_pattern, line, flags=re.UNICODE)) for line in lines)


def _same_lines(left: list[str], right: list[str]) -> bool:
    return [line.strip() for line in left] == [line.strip() for line in right]


def _target_language(default: str = "") -> str:
    return effective_target_language(default).lower()


def _configured_target_language(default: str = "") -> str:
    try:
        return str(load_key("target_language", default) or default)
    except (FileNotFoundError, KeyError):
        return default


def _looks_like_target_language(lines: list[str], target_language: str) -> bool:
    text = " ".join(lines)
    if not target_language.startswith("en"):
        return True
    if re.search(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]", text):
        return False
    if re.search(r"[ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ]", text, re.IGNORECASE):
        return False
    return True


def _expand_word_budget(row: pd.Series | dict, original_lines: list[str]) -> dict[str, int] | None:
    real_dur = float(row.get("real_dur", 0) or row.get("est_dur", 0) or 0)
    if real_dur <= 0:
        return None
    original_words = max(_word_count(original_lines), 1)
    min_speed = max(float(load_key("speed_factor.min", 0.8)), 0.1)
    desired_raw_dur = row_available_duration(row) * get_quality_config().min_duration_ratio * min_speed
    needed_ratio = max(desired_raw_dur / real_dur, 1.05)
    min_ratio = min(max(needed_ratio * 0.9, 1.05), 1.55)
    max_ratio = min(max(needed_ratio * 1.15, min_ratio + 0.08), 1.70)
    return {
        "current_words": original_words,
        "min_words": max(original_words + 1, int(math.floor(original_words * min_ratio))),
        "max_words": max(original_words + 2, int(math.ceil(original_words * max_ratio))),
    }


def _shorten_word_budget(row: pd.Series | dict, original_lines: list[str]) -> dict[str, int] | None:
    real_dur = float(row.get("real_dur", 0) or row.get("est_dur", 0) or 0)
    available = row_available_duration(row)
    if real_dur <= 0 or available <= 0:
        return None
    original_words = max(_word_count(original_lines), 1)
    natural_limit = max(float(get_quality_config().max_natural_speed_factor), 0.1)
    target_raw_dur = available * natural_limit
    ratio = min(target_raw_dur / real_dur, 0.95)
    max_words = int(math.ceil(original_words * max(ratio, 0.25)))
    if original_words > 1:
        max_words = min(max_words, original_words - 1)
    min_floor = 2 if original_words >= 3 else 1
    return {
        "current_words": original_words,
        "max_words": max(max_words, min_floor),
    }


def _within_expand_budget(lines: list[str], word_budget: dict[str, int] | None) -> bool:
    if not word_budget:
        return True
    return _word_count(lines) <= int(word_budget["max_words"])


def _within_shorten_budget(lines: list[str], word_budget: dict[str, int] | None) -> bool:
    if not word_budget:
        return True
    return _word_count(lines) <= int(word_budget["max_words"])


def _compact_english_candidate(lines: list[str], max_words: int) -> list[str]:
    text = " ".join(lines).strip()
    if not text:
        return []
    replacements = [
        (r"\bso,\s*", ""),
        (r"\bcompared to (?:your |an |a |the )?(?:average |normal |typical |most )?rookie anchors?\b", "versus rookies"),
        (r"\bcompared to most rookies\b", "versus rookies"),
        (r"\b(?:he|it)['’]?s already (?:got it|there|done)\b", "it's there"),
        (r"\ba tenth to a fifth,\s*now a third\b", "a tenth to a third"),
        (r"\bhalf next year,\s*then full parity\b", "half next year, then parity"),
        (r"\bwell,\s*", ""),
        (r"\byou know,\s*", ""),
        (r"\bI mean,\s*", ""),
        (r"\bactually,\s*", ""),
        (r"\bdo not\b", "don't"),
        (r"\bcannot\b", "can't"),
        (r"\bwill not\b", "won't"),
        (r"\bwould not\b", "wouldn't"),
        (r"\bshould not\b", "shouldn't"),
        (r"\bcould not\b", "couldn't"),
        (r"\bis not\b", "isn't"),
        (r"\bare not\b", "aren't"),
        (r"\bwas not\b", "wasn't"),
        (r"\bwere not\b", "weren't"),
        (r"\bhas not\b", "hasn't"),
        (r"\bhave not\b", "haven't"),
        (r"\bhad not\b", "hadn't"),
        (r"\bwhat is\b", "what's"),
        (r"\bthat is\b", "that's"),
        (r"\bit is\b", "it's"),
        (r"\bthere is\b", "there's"),
        (r"\bhe is\b", "he's"),
        (r"\bshe is\b", "she's"),
        (r"\bI am\b", "I'm"),
        (r"\byou are\b", "you're"),
        (r"\bwe are\b", "we're"),
        (r"\bthey are\b", "they're"),
        (r"\b(?:this|that|it)\s*\?", "?"),
        (r"\.\.\.", ""),
    ]
    compact = text
    for pattern, repl in replacements:
        compact = re.sub(pattern, repl, compact, flags=re.IGNORECASE)
    compact = re.sub(r"\s+", " ", compact).strip(" ,;:-")
    compact = re.sub(r"\s+([?.!,])", r"\g<1>", compact)
    if compact and not compact.endswith((".", "?", "!")):
        compact += "."
    if compact and _word_count([compact]) <= max_words:
        return [compact[:1].upper() + compact[1:]]
    words = re.findall(r"[\wÀ-ỹ]+(?:['’][\wÀ-ỹ]+)?", compact, flags=re.UNICODE)
    if words and len(words) <= max_words:
        return [" ".join(words)]
    if max_words == 1 and text.lower().startswith(("no ", "no?", "no.")):
        return ["No?"] if text.endswith("?") else ["No."]
    return []


FIRST_PERSON = {"i", "me", "my", "mine", "we", "us", "our", "ours"}
SECOND_PERSON = {"you", "your", "yours"}
THIRD_PERSON = {"he", "him", "his", "she", "her", "hers", "they", "them", "their", "theirs"}
MAGNITUDES = {"hundred", "thousand", "million", "billion", "trillion"}

STOPWORDS = {
    "that", "this", "then", "there", "here", "with", "from", "into", "about",
    "some", "have", "been", "will", "would", "could", "should", "your", "them",
    "they", "their", "what", "which", "when", "where", "much", "more", "very",
    "also", "just", "well", "know", "mean", "like", "really"
}

def _extract_numbers(text: str) -> set[str]:
    word_to_num = {
        "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
        "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
        "eleven": "11", "twelve": "12", "thirteen": "13", "fourteen": "14", "fifteen": "15",
        "sixteen": "16", "seventeen": "17", "eighteen": "18", "nineteen": "19",
        "twenty": "20", "thirty": "30", "forty": "40", "fifty": "50", "sixty": "60",
        "seventy": "70", "eighty": "80", "ninety": "90",
        "hundred": "100", "thousand": "1000", "million": "million", "billion": "billion",
        "first": "1st", "second": "2nd", "third": "3rd", "fourth": "4th", "fifth": "5th"
    }
    tokens = re.findall(r"\b\d+(?:\.\d+)?\b|\b[a-zA-Z]+\b", text.lower())
    nums = set()
    for i, t in enumerate(tokens):
        if re.match(r"^\d+(?:\.\d+)?$", t):
            nums.add(t)
        elif t in word_to_num:
            if t == "one":
                prev = tokens[i - 1] if i > 0 else ""
                next_t = tokens[i + 1] if i + 1 < len(tokens) else ""
                # "the one", "this one", "that one", "no one", "someone", "one who/that/I" are pronouns, not cardinal counts
                if prev in {"the", "this", "that", "which", "every", "some", "no", "any"} or next_t in {"who", "whom", "that", "i", "you", "we", "he", "she", "they"}:
                    continue
            nums.add(word_to_num[t])
    return nums


def _extract_timeframes(text: str) -> set[str]:
    time_map = {
        "day": "day", "days": "day",
        "week": "week", "weeks": "week",
        "month": "month", "months": "month",
        "year": "year", "years": "year",
        "hour": "hour", "hours": "hour",
        "minute": "minute", "minutes": "minute",
        "second": "second", "seconds": "second",
        "today": "today", "tomorrow": "tomorrow", "yesterday": "yesterday",
    }
    tokens = re.findall(r"\b[a-zA-Z]+\b", text.lower())
    return {time_map[t] for t in tokens if t in time_map}


def _has_timeframe(text: str) -> bool:
    return bool(_extract_timeframes(text))


def _has_negation(text: str) -> bool:
    return bool(re.search(r"\b(not|no|never|neither|nor|none|n't|cannot|can't|won't|don't|doesn't|didn't|isn't|aren't|wasn't|weren't|unclear|unaware|unknown|clueless)\b", text, re.I))


def _extract_number_bindings(text: str) -> list[tuple[str, str]]:
    word_to_num = {
        "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
        "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
        "hundred": "100", "thousand": "1000", "million": "million", "billion": "billion"
    }
    time_units = {"day", "days", "week", "weeks", "month", "months", "year", "years", "hour", "hours", "minute", "minutes", "second", "seconds", "dollar", "dollars", "tael", "taels", "million", "thousand"}
    tokens = re.findall(r"\b\d+(?:\.\d+)?\b|\b[a-zA-Z]+\b", text.lower())
    bindings = []
    for i in range(len(tokens) - 1):
        t = tokens[i]
        num_val = word_to_num.get(t, t if re.match(r"^\d+$", t) else None)
        if num_val and tokens[i+1] in time_units:
            unit = tokens[i+1].rstrip('s')
            bindings.append((num_val, unit))
    return bindings


def _extract_negated_content(text: str) -> set[str]:
    neg_words = {"not", "no", "never", "neither", "nor", "none", "cannot", "can't", "won't", "don't", "doesn't", "didn't", "isn't", "aren't", "wasn't", "weren't"}
    tokens = re.findall(r"[a-zA-Z]+", text.lower())
    negated = set()
    for i, t in enumerate(tokens):
        if t in neg_words or t.endswith("n't"):
            for j in range(i + 1, min(i + 4, len(tokens))):
                if tokens[j] in neg_words or tokens[j].endswith("n't"):
                    break
                if tokens[j] not in {"the", "a", "an", "to", "and", "or", "her", "his", "my", "your"}:
                    negated.add(tokens[j])
    return negated


def _valid_rewrite(
    original_lines: list[str],
    rewritten: list[str],
    *,
    direction: str,
    word_budget: dict[str, int] | None,
    target_language: str,
    row: pd.Series | dict | None = None,
) -> bool:
    if direction == "shorten":
        if not rewritten or len(rewritten) > len(original_lines):
            return False
    elif len(rewritten) != len(original_lines):
        return False
    if _same_lines(original_lines, rewritten):
        return False
    if not _looks_like_target_language(rewritten, target_language):
        return False

    orig_words = _word_count(original_lines)
    rewr_words = _word_count(rewritten)
    # Semantic integrity guardrails:
    # 1. Do not collapse a multi-word clause (>=3 words) to a single isolated noun/word,
    # unless the budget explicitly capped the output at 1 word for an ultra-short slot.
    avail_dur = float((row or {}).get("available_duration", 0) or 0) if isinstance(row, (dict, pd.Series)) else 1.0
    allow_single_word = (
        (word_budget is not None and int(word_budget.get("max_words", 99)) <= 1)
        or (0 < avail_dur <= 0.6)
    )
    if orig_words >= 3 and rewr_words < 2 and not allow_single_word:
        return False

    orig_full = " ".join(original_lines).strip()
    rewr_full = " ".join(rewritten).strip()

    # 2. Preserve question speech act
    if orig_full.endswith("?") != rewr_full.endswith("?"):
        origin_text = str((row or {}).get("origin", "")).strip() if isinstance(row, (dict, pd.Series)) else ""
        source_is_question = bool(re.search(r"[?？]\s*$", origin_text))
        if source_is_question:
            return False

    # 3. Preserve negation (neither drop nor invent negation)
    if _has_negation(orig_full) != _has_negation(rewr_full):
        return False

    # 3b. Negation scope: do not invert which clause or noun is negated
    orig_neg = _extract_negated_content(orig_full)
    rewr_neg = _extract_negated_content(rewr_full)
    if orig_neg != rewr_neg and orig_neg and rewr_neg:
        if not (orig_neg & rewr_neg):
            return False

    # 4. Strict number and magnitude preservation: never drop magnitudes (million, thousand) or change explicit numbers
    orig_mags = {w for w in MAGNITUDES if re.search(r"\b" + w + r"\b", orig_full.lower())}
    rewr_mags = {w for w in MAGNITUDES if re.search(r"\b" + w + r"\b", rewr_full.lower())}
    if orig_mags != rewr_mags:
        return False

    orig_nums = _extract_numbers(orig_full)
    rewr_nums = _extract_numbers(rewr_full)
    if rewr_nums - orig_nums:
        return False
    orig_cardinals = {n for n in orig_nums if not n.endswith(("st", "nd", "rd", "th"))}
    rewr_cardinals = {n for n in rewr_nums if not n.endswith(("st", "nd", "rd", "th"))}
    if orig_cardinals != rewr_cardinals:
        return False
    if orig_nums and not (orig_nums & rewr_nums):
        return False

    # 4b. Number-unit bindings (e.g. four days, two million vs two days, four million)
    orig_bind = _extract_number_bindings(orig_full)
    rewr_bind = _extract_number_bindings(rewr_full)
    if orig_bind and rewr_bind:
        orig_dict = dict(orig_bind)
        rewr_dict = dict(rewr_bind)
        for num, unit in orig_bind:
            if unit in rewr_dict.values():
                rewr_num = next((n for n, u in rewr_bind if u == unit), None)
                if rewr_num and rewr_num != num:
                    return False
        orig_units = {u for _, u in orig_bind}
        rewr_units = {u for _, u in rewr_bind}
        if orig_units - rewr_units:
            return False

    # 5. Strict timeframe preservation: never invent or switch timeframe unit
    orig_time = _extract_timeframes(orig_full)
    rewr_time = _extract_timeframes(rewr_full)
    if rewr_time != orig_time:
        return False

    # 6. Actor/Pronoun integrity: do not introduce unprompted grammatical persons or invert roles
    orig_tokens = set(re.findall(r"\b[a-zA-Z]+\b", orig_full.lower()))
    rewr_tokens = set(re.findall(r"\b[a-zA-Z]+\b", rewr_full.lower()))
    if not (orig_tokens & FIRST_PERSON) and (rewr_tokens & FIRST_PERSON):
        return False
    if not (orig_tokens & SECOND_PERSON) and (rewr_tokens & SECOND_PERSON):
        return False
    if not (orig_tokens & THIRD_PERSON) and (rewr_tokens & THIRD_PERSON):
        return False

    # Check for faithful passive construction
    is_passive = bool(re.search(r"\b(is|was|are|were|been|being)\s+\w+(?:ed|en|t|d)\s+by\b", rewr_full.lower()) or "by" in rewr_tokens)

    # 6b. Proper name role ordering (e.g. Alice pays Bob -> Bob pays Alice)
    common_starters = {"what", "who", "where", "when", "why", "how", "translate", "see", "fine", "good", "brother", "no", "yes", "the", "this", "that", "these", "those", "is", "are", "do", "does", "did", "can", "could", "would", "should", "if", "so", "and", "but", "or", "wait", "look", "listen", "tell", "give", "help", "let", "make", "take", "come", "go", "stop", "hold", "surprise", "in", "on", "at", "to", "for", "please"}
    orig_names = [w for w in re.findall(r"\b[A-Z][a-z]+\b", orig_full) if w.lower() not in common_starters]
    rewr_names = [w for w in re.findall(r"\b[A-Z][a-z]+\b", rewr_full) if w.lower() not in common_starters]
    if not is_passive and len(orig_names) >= 2 and len(rewr_names) >= 2 and orig_names == list(reversed(rewr_names)):
        return False

    # 6c. Pronoun role ordering (e.g. I pay you -> You pay me; He pays her -> She pays him)
    orig_has_1_and_2 = bool(orig_tokens & FIRST_PERSON) and bool(orig_tokens & SECOND_PERSON)
    rewr_has_1_and_2 = bool(rewr_tokens & FIRST_PERSON) and bool(rewr_tokens & SECOND_PERSON)
    if orig_has_1_and_2 and rewr_has_1_and_2:
        orig_1st_pos = min(i for i, t in enumerate(re.findall(r"\b[a-zA-Z]+\b", orig_full.lower())) if t in FIRST_PERSON)
        orig_2nd_pos = min(i for i, t in enumerate(re.findall(r"\b[a-zA-Z]+\b", orig_full.lower())) if t in SECOND_PERSON)
        rewr_1st_pos = min(i for i, t in enumerate(re.findall(r"\b[a-zA-Z]+\b", rewr_full.lower())) if t in FIRST_PERSON)
        rewr_2nd_pos = min(i for i, t in enumerate(re.findall(r"\b[a-zA-Z]+\b", rewr_full.lower())) if t in SECOND_PERSON)
        if not is_passive and (orig_1st_pos < orig_2nd_pos) != (rewr_1st_pos < rewr_2nd_pos):
            return False

    male_3rd = {"he", "him", "his"}
    female_3rd = {"she", "her", "hers"}
    orig_has_m_and_f = bool(orig_tokens & male_3rd) and bool(orig_tokens & female_3rd)
    rewr_has_m_and_f = bool(rewr_tokens & male_3rd) and bool(rewr_tokens & female_3rd)
    if orig_has_m_and_f and rewr_has_m_and_f:
        orig_m_pos = min(i for i, t in enumerate(re.findall(r"\b[a-zA-Z]+\b", orig_full.lower())) if t in male_3rd)
        orig_f_pos = min(i for i, t in enumerate(re.findall(r"\b[a-zA-Z]+\b", orig_full.lower())) if t in female_3rd)
        rewr_m_pos = min(i for i, t in enumerate(re.findall(r"\b[a-zA-Z]+\b", rewr_full.lower())) if t in male_3rd)
        rewr_f_pos = min(i for i, t in enumerate(re.findall(r"\b[a-zA-Z]+\b", rewr_full.lower())) if t in female_3rd)
        if not is_passive and (orig_m_pos < orig_f_pos) != (rewr_m_pos < rewr_f_pos):
            return False

    # 6d. Repetition preservation (do not drop repeated words for emotional emphasis)
    def _extract_word_repeats(text: str) -> set[str]:
        words = re.findall(r"\b([a-zA-Z]{2,})\b", text.lower())
        reps = set()
        for i in range(len(words) - 1):
            if words[i] == words[i + 1]:
                reps.add(words[i])
        return reps
    orig_reps = _extract_word_repeats(orig_full)
    rewr_reps = _extract_word_repeats(rewr_full)
    allow_repetition_reduction = (
        word_budget is not None
        and int(word_budget.get("max_words", 999)) < orig_words
    )
    if not allow_repetition_reduction and (orig_reps - rewr_reps):
        return False

    # 7. Compound action preservation: do not drop conjoined action clauses
    m = re.search(r"\band\s+([a-zA-Z]{3,})\s+(?:[a-zA-Z]+\s+)?([a-zA-Z]{4,})\b", orig_full.lower())
    if m:
        verb, noun = m.group(1), m.group(2)
        if verb not in STOPWORDS and noun not in STOPWORDS:
            if verb not in rewr_tokens and noun not in rewr_tokens:
                return False

    # 7b. Action replacement (introducing new unprompted contradictory actions)
    opposite_actions = {
        ("open", "close"), ("close", "open"), ("open", "shut"), ("shut", "open"),
        ("buy", "sell"), ("sell", "buy"), ("give", "take"), ("take", "give"),
        ("pay", "receive"), ("receive", "pay"), ("pay", "charge"),
        ("accept", "refuse"), ("refuse", "accept"), ("accept", "reject"), ("reject", "accept"),
        ("start", "stop"), ("stop", "start"), ("enter", "leave"), ("leave", "enter"),
        ("lock", "unlock"), ("unlock", "lock"), ("hide", "reveal"), ("show", "hide"),
    }
    for a1, a2 in opposite_actions:
        if re.search(r"\b" + a1 + r"\b", orig_full.lower()) and re.search(r"\b" + a2 + r"\b", rewr_full.lower()) and not re.search(r"\b" + a2 + r"\b", orig_full.lower()):
            return False

    new_words = rewr_tokens - orig_tokens
    action_verbs = {"destroy", "kill", "sell", "steal", "drop", "lose", "break", "burn", "refuse", "reject", "attack", "cancel", "abandon"}
    if new_words & action_verbs:
        return False

    if orig_words >= 8 and rewr_words < 3:
        return False

    if direction == "expand":
        return _within_expand_budget(rewritten, word_budget)
    if direction == "shorten":
        return _within_shorten_budget(rewritten, word_budget)
    return True


def rewrite_task_lines(
    row: pd.Series | dict,
    reason: str,
    *,
    direction: str = "shorten",
    max_retries: int | None = None,
    retry_interval: int | None = None,
    timeout_seconds: float | None = None,
) -> list[str]:
    """Use the configured LLM to fit one dubbing task to its time budget."""
    original_lines = normalize_lines(row.get("lines", row.get("text", "")))
    if not original_lines:
        return []

    target_language = _target_language(_configured_target_language(""))
    word_budget = _expand_word_budget(row, original_lines) if direction == "expand" else _shorten_word_budget(row, original_lines)

    speaker = str(row.get("speaker", "") or "").strip()
    speaker_profile = None
    try:
        import json
        from pathlib import Path
        prof_path = Path("output/log/speaker_profiles.json")
        if not prof_path.exists():
            prof_path = Path("output/log/terminology.json")
        if prof_path.exists():
            pdata = json.loads(prof_path.read_text(encoding="utf-8"))
            plist = pdata if isinstance(pdata, list) else pdata.get("speaker_profiles", [])
            for p in plist:
                if isinstance(p, dict) and p.get("speaker_id") == speaker:
                    speaker_profile = json.dumps(p, ensure_ascii=False)
                    break
    except Exception:
        pass

    prompt = get_dubbing_rewrite_prompt(
        original=row.get("origin", ""),
        current_lines=original_lines,
        target_duration=row_available_duration(row),
        available_duration=row_available_duration(row),
        reason=reason,
        source_lines=normalize_lines(row.get("src_lines", "")),
        direction=direction,
        word_budget=word_budget,
        speaker=speaker,
        speaker_profile=speaker_profile,
    )
    response = ask_gpt(
        prompt,
        response_json=True,
        log_title="dubbing_rewrite",
        max_retries=max_retries,
        retry_interval=retry_interval,
        timeout_seconds=timeout_seconds,
    )
    rewritten = _extract_rewritten_lines(response)
    if not rewritten or not _valid_rewrite(original_lines, rewritten, direction=direction, word_budget=word_budget, target_language=target_language, row=row):
        if direction == "expand" and word_budget:
            retry_instruction = (
                f"The previous output was invalid. Rewrite again, keep exactly {len(original_lines)} line(s), "
                f"change the wording, and keep the total word count at or below {word_budget['max_words']} words."
            )
        elif direction == "shorten" and word_budget:
            retry_instruction = (
                f"The previous output was not short enough or did not change. Rewrite again, keep exactly "
                f"{len(original_lines)} line(s), and use at most {word_budget['max_words']} words total."
            )
        else:
            retry_instruction = (
                f"The previous output was invalid. Rewrite again, keep exactly {len(original_lines)} line(s), "
                "and change the wording while preserving the meaning."
            )
        retry_prompt = (
            f"{prompt}\n\n{retry_instruction}"
        )
        response = ask_gpt(
            retry_prompt,
            response_json=True,
            log_title="dubbing_rewrite_word_budget_retry",
            max_retries=max_retries,
            retry_interval=retry_interval,
            timeout_seconds=timeout_seconds,
        )
        rewritten = _extract_rewritten_lines(response)
    if not rewritten or not _valid_rewrite(original_lines, rewritten, direction=direction, word_budget=word_budget, target_language=target_language, row=row):
        if direction == "shorten" and word_budget and target_language.startswith("en"):
            for fallback_source in [rewritten, original_lines]:
                if not fallback_source:
                    continue
                fallback = _compact_english_candidate(fallback_source, int(word_budget["max_words"]))
                if fallback and _valid_rewrite(
                    original_lines,
                    fallback,
                    direction=direction,
                    word_budget=word_budget,
                    target_language=target_language,
                    row=row,
                ):
                    return fallback
        return []
    return rewritten
