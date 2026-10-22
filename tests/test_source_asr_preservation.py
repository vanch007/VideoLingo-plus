from copy import deepcopy

import pandas as pd
import pytest

from core.asr_schema import WordTimestampContractError, sanitize_word_timestamps
from core.all_whisper_methods.audio_preprocess import process_transcription, save_results


def test_english_recognition_keeps_spaces_and_punctuation_after_alignment():
    from core.all_whisper_methods.local_aligned import bind_recognition_words
    source = "which is to turn that on and that's done right here."
    words = [{'word': w, 'start': i * .3, 'end': (i + 1) * .3}
             for i, w in enumerate(source.rstrip('.').split())]
    result = bind_recognition_words(words, [{'text': source, 'start': 0., 'end': 4.}], 0., 'en')
    assert all(w['recognition_text'] == source for w in result)


def test_orphan_collapsed_phrase_cannot_be_attached_to_a_neighbor():
    from core.all_whisper_methods.qwen3_aligned import merge_zero_duration_words
    words = [{'word': 'Go', 'start': 0., 'end': 1., 'recognition_segment_id': 'A'},
             {'word': 'No', 'start': 1., 'end': 1., 'recognition_segment_id': 'B'},
             {'word': 'Stop', 'start': 1.2, 'end': 2., 'recognition_segment_id': 'C'}]
    with pytest.raises(ValueError, match='recognized phrase'):
        merge_zero_duration_words(words, language='en')


def test_real_repeated_questions_are_not_removed_as_decoder_loops():
    words = [{"word": token, "start": i * 0.15, "end": (i + 1) * 0.15}
             for i, token in enumerate(["马匪", "呢"] * 3)]
    raw = {"language": "zh", "segments": [{"text": "马匪呢？马匪呢？马匪呢？", "words": words}]}
    result = sanitize_word_timestamps(deepcopy(raw))
    assert "".join(w["word"] for w in result["segments"][0]["words"]) == "马匪呢" * 3
    assert result["segments"][0]["text"] == raw["segments"][0]["text"]
    assert result["word_timestamp_sanitizer"]["suspected_repetition_words"] > 0


def test_eight_ms_demonstrative_and_zero_ms_final_character_survive_export(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    raw = {"language": "zh", "segments": [{"text": "这就是惊喜", "words": [
        {"word": "这", "start": 1.0, "end": 1.008},
        {"word": "就是", "start": 1.008, "end": 1.5},
        {"word": "惊", "start": 1.5, "end": 1.8},
        {"word": "喜", "start": 1.8, "end": 1.8},
    ]}]}
    result = sanitize_word_timestamps(raw)
    save_results(process_transcription(result))
    exported = pd.read_excel("output/log/cleaned_chunks.xlsx")
    assert "".join(exported.text.str.strip('"')) == "这就是惊喜"
    assert all(exported.end > exported.start)


def test_long_native_tokens_are_never_dropped_at_either_export_stage(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    token = "https://example.invalid/long-native-token"
    raw = {"segments": [{"text": token, "words": [{"word": token, "start": 1.0, "end": 2.0}]}]}
    save_results(process_transcription(raw))
    assert pd.read_excel("output/log/cleaned_chunks.xlsx").text.iloc[0].strip('"') == token


def test_collapsed_english_tokens_keep_the_word_boundary():
    raw = {"language": "en", "segments": [{"text": "I agree.", "words": [
        {"word": "I", "start": 1.0, "end": 1.0},
        {"word": "agree.", "start": 1.0, "end": 1.4},
    ]}]}
    assert sanitize_word_timestamps(raw)["segments"][0]["words"][0]["word"] == "I agree."


def test_no_acoustic_timing_for_recognized_text_requires_realigning():
    raw = {"segments": [{"text": "剿匪", "words": [{"word": "剿匪", "start": 2., "end": 2.}]}]}
    with pytest.raises(WordTimestampContractError, match="realignment required"):
        sanitize_word_timestamps(raw)
