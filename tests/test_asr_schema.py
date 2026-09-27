import pytest

from core.asr_schema import (
    WordTimestampContractError,
    sanitize_word_timestamps,
    validate_word_timestamps,
)


def test_word_timestamp_contract_accepts_native_words():
    result = {
        "segments": [
            {
                "text": "hello world",
                "words": [
                    {"word": "hello", "start": 0.0, "end": 0.4},
                    {"word": "world", "start": 0.5, "end": 0.9},
                ],
            }
        ]
    }

    validated = validate_word_timestamps(result, backend="test")
    assert validated["word_timestamp_contract"] == "validated"
    assert validated["word_timestamp_count"] == 2


def test_word_timestamp_contract_rejects_segment_only_asr():
    result = {"segments": [{"text": "hello world", "start": 0.0, "end": 1.0, "words": []}]}

    with pytest.raises(WordTimestampContractError, match="no native word timestamps"):
        validate_word_timestamps(result, backend="segment-only")


def test_word_timestamp_contract_rejects_missing_word_end():
    result = {"segments": [{"text": "hello", "words": [{"word": "hello", "start": 0.0}]}]}

    with pytest.raises(WordTimestampContractError, match="missing numeric"):
        validate_word_timestamps(result, backend="broken")


def test_word_timestamp_contract_rejects_non_finite_timestamps():
    result = {"segments": [{"text": "hello", "words": [{"word": "hello", "start": float("nan"), "end": 1.0}]}]}

    with pytest.raises(WordTimestampContractError, match="invalid timestamp"):
        validate_word_timestamps(result, backend="broken")


def test_sanitizer_removes_zero_duration_and_repeated_decoder_loop():
    result = {
        "segments": [
            {
                "start": 1.0,
                "end": 3.0,
                "text": "张麻子作作作作",
                "words": [
                    {"word": "张麻子", "start": 1.0, "end": 1.5},
                    {"word": "作", "start": 1.5, "end": 1.5},
                    {"word": "作", "start": 1.5, "end": 1.7},
                    {"word": "作", "start": 1.7, "end": 1.8},
                    {"word": "作", "start": 1.8, "end": 2.0},
                ],
            }
        ]
    }
    sanitize_word_timestamps(result)
    assert result["segments"][0]["text"] == "张麻子"
    assert [word["word"] for word in result["segments"][0]["words"]] == ["张麻子"]

def test_sanitizer_preserves_spaced_repetitions_and_speaker_turns():
    words = [{'word': w, 'start': i * 5.0, 'end': i * 5.0 + 0.5, 'speaker': f'S{i // 2 + 1:02d}'} for i, w in enumerate(['快', '走', '快', '走', '快', '走'])]
    raw = {'segments': [{'words': words, 'start': 0.0, 'end': 25.5, 'text': '快走快走快走'}]}
    clean = sanitize_word_timestamps(raw)
    assert clean['word_timestamp_sanitizer']['output_words'] == 6
    assert clean['word_timestamp_sanitizer']['removed_repeated_tokens'] == 0
