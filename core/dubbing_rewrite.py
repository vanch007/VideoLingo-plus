from __future__ import annotations

import math
import re
from typing import Any

import pandas as pd

from core.ask_gpt import ask_gpt
from core.config_utils import load_key
from core.dubbing_quality import get_quality_config, normalize_lines, row_available_duration
from core.prompts_storage import get_dubbing_rewrite_prompt


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
    return sum(len(re.findall(r"[\wÀ-ỹ]+", line, flags=re.UNICODE)) for line in lines)


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


def _within_expand_budget(lines: list[str], word_budget: dict[str, int] | None) -> bool:
    if not word_budget:
        return True
    return _word_count(lines) <= int(word_budget["max_words"])


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

    word_budget = _expand_word_budget(row, original_lines) if direction == "expand" else None
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
    if rewritten and direction == "expand" and not _within_expand_budget(rewritten, word_budget):
        retry_prompt = (
            f"{prompt}\n\nThe previous output was too long. Rewrite again and keep the total word count "
            f"at or below {word_budget['max_words']} words. Keep the same line count."
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
    if rewritten and direction == "expand" and not _within_expand_budget(rewritten, word_budget):
        return original_lines
    return rewritten or original_lines
