from typing import Any


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
