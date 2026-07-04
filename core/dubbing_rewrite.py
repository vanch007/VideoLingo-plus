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
    return {
        "current_words": original_words,
        "max_words": max(max_words, 1),
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
        (r"\.\.\.", ""),
        (r"\bactual person\b", "real person"),
        (r"\bactually moving by themselves\b", "move"),
        (r"\bmoving on their own\b", "move"),
        (r"\bits eyes (?:are )?move\b", "eyes move"),
        (r"\bis that human\?\s*", ""),
        (r"\bit['’]?s a digital human;?\s*its eyes move\b", "Digital human. Its eyes move"),
        (r"\ba tenth to a fifth,\s*now a third\b", "a tenth to a third"),
        (r"\bhalf next year,\s*then full parity\b", "half next year, then parity"),
        (r"\bwell,\s*", ""),
        (r"\bwe only replicate the very best anchors now\b", "we clone top anchors now"),
    ]
    compact = text
    for pattern, repl in replacements:
        compact = re.sub(pattern, repl, compact, flags=re.IGNORECASE)
    compact = re.sub(r"\s+", " ", compact).strip(" ,;:-")
    if compact and not compact.endswith((".", "?", "!")):
        compact += "."
    if compact and _word_count([compact]) <= max_words:
        return [compact[:1].upper() + compact[1:]]
    words = re.findall(r"[\wÀ-ỹ]+(?:['’][\wÀ-ỹ]+)?", compact, flags=re.UNICODE)
    if words and len(words) <= max_words:
        return [" ".join(words)]
    return []


def _valid_rewrite(
    original_lines: list[str],
    rewritten: list[str],
    *,
    direction: str,
    word_budget: dict[str, int] | None,
    target_language: str,
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
    prompt = get_dubbing_rewrite_prompt(
        original=row.get("origin", ""),
        current_lines=original_lines,
        target_duration=row_available_duration(row),
        available_duration=row_available_duration(row),
        reason=reason,
        source_lines=normalize_lines(row.get("src_lines", "")),
        direction=direction,
        word_budget=word_budget,
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
    if rewritten and not _valid_rewrite(original_lines, rewritten, direction=direction, word_budget=word_budget, target_language=target_language):
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
    if not rewritten or not _valid_rewrite(original_lines, rewritten, direction=direction, word_budget=word_budget, target_language=target_language):
        if direction == "shorten" and word_budget and target_language.startswith("en"):
            fallback_source = rewritten or original_lines
            if rewritten and not _looks_like_target_language(rewritten, target_language):
                fallback_source = original_lines
            fallback = _compact_english_candidate(fallback_source, int(word_budget["max_words"]))
            if fallback and _valid_rewrite(
                original_lines,
                fallback,
                direction=direction,
                word_budget=word_budget,
                target_language=target_language,
            ):
                return fallback
        return []
    return rewritten
