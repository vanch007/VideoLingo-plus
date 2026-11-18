from core.providers.source_coverage import _replace_range_with_recovery
from core.text_join import join_tokens


def test_latin_tokens_keep_a_space_and_cjk_tokens_do_not():
    assert join_tokens(["US", "dollar"]) == "US dollar"
    assert join_tokens(["哎", "黄", "老", "爷", "easy", "easy"]) == "哎黄老爷easy easy"
    assert join_tokens(["你", "好", "iPhone", "15", "手机"]) == "你好iPhone 15手机"
    assert join_tokens([]) == ""


def _word(text, start, rec_id, rec_text):
    return {"word": text, "start": start, "end": start + 0.3, "recognition_segment_id": rec_id, "recognition_text": rec_text}


def test_recovery_merge_keeps_intact_phrase_text_with_spacing_and_punctuation():
    result = {"segments": [{"start": 1.0, "end": 1.3, "text": "US", "words": [_word("US", 1.0, "p1", "US dollar，好")]},
                           {"start": 1.3, "end": 1.6, "text": "dollar", "words": [_word("dollar", 1.3, "p1", "US dollar，好")]},
                           {"start": 1.6, "end": 1.9, "text": "好", "words": [_word("好", 1.6, "p1", "US dollar，好")]}]}
    recovery = {"segments": [{"start": 5.0, "end": 5.3, "text": "不", "words": [_word("不", 5.0, "r1", "不")]}]}
    _replace_range_with_recovery(result, recovery, start=4.8, end=5.5)
    texts = {w["word"]: w["recognition_text"] for s in result["segments"] for w in s["words"]}
    assert texts["US"] == texts["dollar"] == texts["好"] == "US dollar，好"


def test_recovery_merge_rebuilds_a_partial_phrase_with_spacing():
    # Only part of the phrase survives the cut: the text is rebuilt from its words.
    result = {"segments": [{"start": 1.0, "end": 1.3, "text": "US", "words": [_word("US", 1.0, "p1", "US dollar test")]},
                           {"start": 1.3, "end": 1.6, "text": "dollar", "words": [_word("dollar", 1.3, "p1", "US dollar test")]}]}
    recovery = {"segments": []}
    _replace_range_with_recovery(result, recovery, start=8.0, end=9.0)
    assert {w["recognition_text"] for s in result["segments"] for w in s["words"]} == {"US dollar"}
