from core.all_tts_functions.tts_main import clean_text_for_tts


def test_clean_text_for_tts_preserves_ranges_and_slashes():
    text = clean_text_for_tts("1-10 người, muốn nồi to/lòng kép, giá chỉ trên 100k.")

    assert "1-10 người" in text
    assert "to/lòng" in text
    assert "100k" in text


def test_clean_text_for_tts_collapses_whitespace_without_removing_punctuation():
    text = clean_text_for_tts("Xin   chào \n bạn !")

    assert text == "Xin chào bạn!"
