from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable

import pandas as pd

from core.config_utils import load_key
from core.constants import AUDIO_DIR
from core.providers.asr_readback import ReadbackSummary, verify_tasks_df


CONTENT_GATE_REPORT = Path(AUDIO_DIR) / "dubbing_content_gate.json"


@dataclass(frozen=True)
class ContentGateSummary:
    status: str
    rounds: int
    regenerated_rows: list[int]
    rewritten_rows: list[int]
    remaining_low_score_rows: list[int]
    remaining_severely_unreadable_rows: list[int]
    remaining_tail_truncation_rows: list[int]
    readback: dict
    report_path: str


def _tokens(text: str) -> list[str]:
    return re.findall(r"[\w\u00c0-\u024f\u1e00-\u1eff\u4e00-\u9fff]+", (text or "").lower())


def _token_matches(left: str, right: str) -> bool:
    if left == right:
        return True
    if min(len(left), len(right)) <= 2:
        return False
    return SequenceMatcher(None, left, right).ratio() >= 0.72


def probable_tail_truncation(expected: str, transcript: str) -> bool:
    """Detect prefix-like ASR output that is missing the expected sentence tail."""
    expected_tokens = _tokens(expected)
    actual_tokens = _tokens(transcript)
    if not expected_tokens or not actual_tokens or len(actual_tokens) >= len(expected_tokens):
        return False

    # Greedily align heard tokens in order. This tolerates an accented name in
    # the middle while proving that the take ends before the expected final
    # token. Merely searching for the tail word fails when it occurs twice.
    expected_cursor = 0
    matched_positions: list[int] = []
    for actual_token in actual_tokens:
        match_position = next(
            (
                position
                for position in range(expected_cursor, len(expected_tokens))
                if _token_matches(expected_tokens[position], actual_token)
            ),
            None,
        )
        if match_position is None:
            continue
        matched_positions.append(match_position)
        expected_cursor = match_position + 1

    minimum_matches = max(2, math.ceil(len(actual_tokens) * 0.65))
    prefix_like = len(matched_positions) >= minimum_matches and matched_positions[0] <= 1
    reaches_expected_end = bool(matched_positions) and matched_positions[-1] == len(expected_tokens) - 1
    return prefix_like and not reaches_expected_end


def probable_prefix_truncation(expected: str, transcript: str) -> bool:
    """Detect suffix-like ASR output that is missing the expected opening words."""
    expected_tokens = _tokens(expected)
    actual_tokens = _tokens(transcript)
    if not expected_tokens or not actual_tokens or len(actual_tokens) >= len(expected_tokens):
        return False

    expected_cursor = 0
    matched_positions: list[int] = []
    for actual_token in actual_tokens:
        match_position = next(
            (
                position
                for position in range(expected_cursor, len(expected_tokens))
                if _token_matches(expected_tokens[position], actual_token)
            ),
            None,
        )
        if match_position is None:
            continue
        matched_positions.append(match_position)
        expected_cursor = match_position + 1

    minimum_matches = max(2, math.ceil(len(actual_tokens) * 0.65))
    suffix_like = (
        len(matched_positions) >= minimum_matches
        and matched_positions[-1] >= len(expected_tokens) - 2
    )
    misses_expected_start = bool(matched_positions) and matched_positions[0] > 0
    return suffix_like and misses_expected_start


def _row_has_tail_truncation(row: pd.Series) -> bool:
    expected = str(row.get("text", ""))
    transcript = str(row.get("asr_transcript", ""))
    # Keep the historical field/report name for compatibility; both missing
    # openings and missing endings are now hard incomplete-speech failures.
    return probable_tail_truncation(expected, transcript) or probable_prefix_truncation(
        expected, transcript
    )


def _has_substantive_spoken_readback(row: pd.Series) -> bool:
    """Treat full-length but accented/homophonic readback as a warning.

    The hard content gate exists to prevent missing or cut-off speech. A local
    ASR may spell cross-language cloned speech differently while still hearing
    the complete utterance. Empty or substantially shorter transcripts remain
    severe, and tail detection is evaluated separately.
    """
    transcript = row.get("asr_transcript", "")
    if transcript is None or (not isinstance(transcript, (list, dict)) and pd.isna(transcript)):
        return False
    expected = _tokens(str(row.get("text", "")))
    actual = _tokens(str(transcript))
    if not expected or not actual:
        return False
    if len(expected) == 1:
        return True
    if len(actual) < math.ceil(len(expected) * 0.60):
        return False
    unmatched = list(actual)
    matched = 0
    for expected_token in expected:
        match_index = next(
            (
                index
                for index, actual_token in enumerate(unmatched)
                if _token_matches(expected_token, actual_token)
            ),
            None,
        )
        if match_index is not None:
            matched += 1
            unmatched.pop(match_index)
    return matched >= max(1, math.ceil(len(expected) * 0.35))


def _failing_indices(tasks_df: pd.DataFrame, threshold: float) -> tuple[list[int], list[int]]:
    scores = pd.to_numeric(tasks_df.get("asr_content_score"), errors="coerce")
    low_indices = tasks_df.index[scores.isna() | scores.lt(threshold)].tolist()
    # A missing final word can still score above the aggregate threshold on a
    # long sentence, so tail checks must cover every row.
    tail_indices = [idx for idx in tasks_df.index if _row_has_tail_truncation(tasks_df.loc[idx])]
    return low_indices, tail_indices


def _repair_indices(
    tasks_df: pd.DataFrame,
    low_indices: list[int],
    tail_indices: list[int],
) -> list[int]:
    severe_threshold = float(load_key("dubbing_quality.content_completion_regenerate_score_min", 0.65))
    scores = pd.to_numeric(tasks_df.get("asr_content_score"), errors="coerce")
    severe = [
        idx
        for idx in low_indices
        if (pd.isna(scores.loc[idx]) or scores.loc[idx] < severe_threshold)
        and not _has_substantive_spoken_readback(tasks_df.loc[idx])
    ]
    return sorted(set(severe) | set(tail_indices))


def _write_report(summary: ContentGateSummary) -> None:
    CONTENT_GATE_REPORT.parent.mkdir(parents=True, exist_ok=True)
    CONTENT_GATE_REPORT.write_text(
        json.dumps(asdict(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _readback_is_globally_unavailable(readback: ReadbackSummary) -> bool:
    """Distinguish provider failure from a few unrecognized short segments."""
    if readback.status == "skipped":
        return True
    return (
        readback.status == "fail"
        and readback.min_content_score is None
        and readback.cached == 0
    )


def enforce_content_completion(
    tasks_df: pd.DataFrame,
    *,
    repair_rows: Callable[[pd.DataFrame, list[int], bool], tuple[pd.DataFrame, list[int]]],
    verify: Callable[..., tuple[pd.DataFrame, ReadbackSummary]] = verify_tasks_df,
    checkpoint: Callable[[pd.DataFrame], None] | None = None,
) -> tuple[pd.DataFrame, ContentGateSummary]:
    """Run ASR readback and repair probable incomplete TTS before final merge."""
    threshold = float(
        load_key(
            "dubbing_quality.content_completion_score_min",
            load_key("dubbing_quality.content_score_min", 0.88),
        )
    )
    max_rounds = max(1, int(load_key("dubbing_quality.content_completion_max_rounds", 2)))
    fail_on_unverified = bool(load_key("dubbing_quality.content_completion_fail_on_unverified", True))
    fail_on_incomplete = bool(load_key("dubbing_quality.content_completion_fail_on_incomplete", True))
    fail_on_unreadable = bool(load_key("dubbing_quality.content_completion_fail_on_unreadable", True))

    # Fingerprints cover text, reference, backend/model and audio signatures.
    # Reuse exact cached rows on resume; newly generated or edited rows have no
    # matching fingerprint and are still read back automatically.
    verified, readback = verify(tasks_df, force=False)
    # Persist the initial readback before attempting any repair. If synthesis,
    # rewriting, or a provider fails during repair, resume can reuse every
    # unchanged row instead of transcribing the entire video again.
    if checkpoint is not None:
        checkpoint(verified)
    if _readback_is_globally_unavailable(readback) and fail_on_unverified:
        raise RuntimeError(
            f"Dubbing content gate cannot verify generated speech: ASR status={readback.status}."
        )

    regenerated: list[int] = []
    rewritten: list[int] = []
    rounds = 0
    low_indices, tail_indices = _failing_indices(verified, threshold)

    while rounds < max_rounds:
        selected = _repair_indices(verified, low_indices, tail_indices)
        if not selected:
            break
        rewrite = rounds > 0
        verified, rewritten_now = repair_rows(verified, selected, rewrite)
        regenerated.extend(int(verified.at[idx, "number"]) for idx in selected)
        rewritten.extend(rewritten_now)
        rounds += 1
        # Reuse unchanged rows by fingerprint. repair_rows clears ASR fields and
        # changes audio signatures only for regenerated rows, so force=False
        # rechecks exactly those rows instead of cold-starting ASR for the full
        # video after every repair round.
        verified, readback = verify(verified, force=False)
        if checkpoint is not None:
            checkpoint(verified)
        if _readback_is_globally_unavailable(readback) and fail_on_unverified:
            raise RuntimeError(
                f"Dubbing content gate lost ASR verification after repair round {rounds}: "
                f"status={readback.status}."
            )
        low_indices, tail_indices = _failing_indices(verified, threshold)

    remaining_low = [int(verified.at[idx, "number"]) for idx in low_indices]
    remaining_severe_indices = _repair_indices(verified, low_indices, [])
    remaining_severe = [int(verified.at[idx, "number"]) for idx in remaining_severe_indices]
    remaining_tail = [int(verified.at[idx, "number"]) for idx in tail_indices]
    status = "fail" if (remaining_tail or remaining_severe) else ("warn" if remaining_low else "ok")
    summary = ContentGateSummary(
        status=status,
        rounds=rounds,
        regenerated_rows=sorted(set(regenerated)),
        rewritten_rows=sorted(set(rewritten)),
        remaining_low_score_rows=remaining_low,
        remaining_severely_unreadable_rows=remaining_severe,
        remaining_tail_truncation_rows=remaining_tail,
        readback=asdict(readback),
        report_path=str(CONTENT_GATE_REPORT),
    )
    _write_report(summary)
    if remaining_tail and fail_on_incomplete:
        raise RuntimeError(
            "Dubbing content gate still detects incomplete sentence tails in rows "
            f"{remaining_tail}; refusing to continue to full-audio/video merge."
        )
    if remaining_severe and fail_on_unreadable:
        raise RuntimeError(
            "Dubbing content gate still cannot reliably understand rows "
            f"{remaining_severe}; refusing to continue to full-audio/video merge."
        )
    return verified, summary
