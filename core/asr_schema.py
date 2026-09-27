import math
from typing import Any


MAIN_ASR_RUNTIMES = ("local", "stable-ts")


class WordTimestampContractError(ValueError):
    """Raised when source ASR cannot safely feed subtitle alignment."""


def sanitize_word_timestamps(
    result: dict[str, Any],
    *,
    min_duration_seconds: float = 0.01,
    repeated_token_limit: int = 3,
    repeated_token_max_gap: float = 0.15,
) -> dict[str, Any]:
    """Remove zero-duration words and repeated decoder-loop artifacts."""
    original_words = [
        word
        for segment in result.get("segments", [])
        for word in (segment.get("words") or [])
    ]
    positive_words: list[dict[str, Any]] = []
    for word in original_words:
        try:
            duration = float(word["end"]) - float(word["start"])
        except (KeyError, TypeError, ValueError):
            continue
        if duration >= min_duration_seconds:
            positive_words.append(word)

    remove_ids: set[int] = set()
    index = 0
    while index < len(positive_words):
        token = str(
            positive_words[index].get("word", positive_words[index].get("text", ""))
        ).strip()
        spk = positive_words[index].get("speaker")
        end_index = index + 1
        while end_index < len(positive_words):
            candidate = str(
                positive_words[end_index].get(
                    "word", positive_words[end_index].get("text", "")
                )
            ).strip()
            gap = float(positive_words[end_index]["start"]) - float(
                positive_words[end_index - 1]["end"]
            )
            cand_spk = positive_words[end_index].get("speaker")
            same_spk = (spk is None or cand_spk is None or spk == cand_spk)
            if not token or candidate != token or gap > repeated_token_max_gap or not same_spk:
                break
            end_index += 1
        if end_index - index >= repeated_token_limit:
            remove_ids.update(id(word) for word in positive_words[index:end_index])
        index = end_index

    # Detect repeated n-gram phrase loops (e.g. repeated hallucinations >= 3 times)
    tokens = [str(w.get("word", w.get("text", ""))).strip() for w in positive_words]
    for n in range(2, 6):
        i = 0
        while i + n <= len(tokens):
            pattern = tokens[i : i + n]
            if not all(pattern):
                i += 1
                continue
            rep_count = 1
            cursor = i + n
            while cursor + n <= len(tokens):
                if tokens[cursor : cursor + n] != pattern:
                    break
                gap_to_prev = float(positive_words[cursor]["start"]) - float(
                    positive_words[cursor - 1]["end"]
                )
                prev_spk = positive_words[cursor - 1].get("speaker")
                curr_spk = positive_words[cursor].get("speaker")
                same_spk = (prev_spk is None or curr_spk is None or prev_spk == curr_spk)
                if gap_to_prev > repeated_token_max_gap or not same_spk:
                    break
                intra_gap_ok = True
                for k in range(cursor, cursor + n - 1):
                    intra_gap = float(positive_words[k + 1]["start"]) - float(positive_words[k]["end"])
                    k_spk = positive_words[k].get("speaker")
                    k1_spk = positive_words[k + 1].get("speaker")
                    if intra_gap > repeated_token_max_gap or (k_spk is not None and k1_spk is not None and k_spk != k1_spk):
                        intra_gap_ok = False
                        break
                if not intra_gap_ok:
                    break
                rep_count += 1
                cursor += n
            if rep_count >= 3:
                for w in positive_words[i + n : cursor]:
                    remove_ids.add(id(w))
                i = cursor
            else:
                i += 1

    cleaned_segments: list[dict[str, Any]] = []
    for segment in result.get("segments", []):
        words = [
            word
            for word in (segment.get("words") or [])
            if id(word) not in remove_ids
            and float(word.get("end", 0.0)) - float(word.get("start", 0.0))
            >= min_duration_seconds
        ]
        if not words:
            continue
        updated = dict(segment)
        updated["words"] = words
        updated["start"] = float(words[0]["start"])
        updated["end"] = float(words[-1]["end"])
        updated["text"] = "".join(
            str(word.get("word", word.get("text", ""))) for word in words
        )
        cleaned_segments.append(updated)
    result["segments"] = cleaned_segments
    result["word_timestamp_sanitizer"] = {
        "input_words": len(original_words),
        "output_words": sum(len(segment["words"]) for segment in cleaned_segments),
        "removed_repeated_tokens": len(remove_ids),
    }
    return result


def validate_word_timestamps(
    result: dict[str, Any],
    *,
    backend: str,
    allow_empty: bool = False,
) -> dict[str, Any]:
    """Require native, numeric and ordered timestamps for every ASR word."""
    segments = result.get("segments")
    if not isinstance(segments, list):
        raise WordTimestampContractError(f"{backend}: segments must be a list")

    word_count = 0
    previous_start = -1.0
    for segment_index, segment in enumerate(segments):
        text = str(segment.get("text", "")).strip()
        words = segment.get("words")
        if text and not words:
            raise WordTimestampContractError(
                f"{backend}: segment {segment_index} has speech but no native word timestamps"
            )
        for word_index, word in enumerate(words or []):
            token = str(word.get("word", word.get("text", ""))).strip()
            start = word.get("start")
            end = word.get("end")
            if not token:
                raise WordTimestampContractError(
                    f"{backend}: segment {segment_index} word {word_index} is empty"
                )
            if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
                raise WordTimestampContractError(
                    f"{backend}: {token!r} is missing numeric start/end timestamps"
                )
            start = float(start)
            end = float(end)
            if end < start:
                end = start + 0.05
                word["end"] = end
            if start < previous_start:
                start = previous_start
                word["start"] = start
                if end < start:
                    end = start + 0.05
                    word["end"] = end
            if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end < start:
                raise WordTimestampContractError(
                    f"{backend}: invalid timestamp for {token!r}: {start}..{end}"
                )
            previous_start = start
            word_count += 1

    if not allow_empty and word_count == 0:
        raise WordTimestampContractError(f"{backend}: transcription contains no word timestamps")
    result["word_timestamp_contract"] = "validated"
    result["word_timestamp_count"] = word_count
    return result


def normalize_asr_result(result: dict[str, Any], backend: str) -> dict[str, Any]:
    """Normalize ASR output to the internal segment/word schema used by the pipeline."""
    result.setdefault("segments", [])
    result.setdefault("language", "auto")
    result["backend"] = backend

    for segment in result["segments"]:
        segment.setdefault("text", "")
        segment.setdefault("start", 0.0)
        segment.setdefault("end", segment["start"])
        segment.setdefault("words", [])
        segment["asr_backend"] = backend
        segment["timestamp_granularity"] = "word" if segment.get("words") else "segment"

        normalized_words = []
        for word in segment.get("words", []):
            normalized_word = {
                "word": str(word.get("word", word.get("text", ""))).strip(),
                "start": word.get("start"),
                "end": word.get("end"),
            }
            if word.get("speaker") is not None:
                normalized_word["speaker"] = word.get("speaker")
            if word.get("confidence") is not None:
                normalized_word["confidence"] = word.get("confidence")
            normalized_words.append(normalized_word)
        segment["words"] = normalized_words

    return result
