import pytest

from core.all_tts_functions.tts_main import clean_text_for_tts


def test_clean_text_for_tts_preserves_ranges_and_slashes():
    text = clean_text_for_tts("1-10 người, muốn nồi to/lòng kép, giá chỉ trên 100k.")

    assert "1-10 người" in text
    assert "to/lòng" in text
    assert "100k" in text


def test_clean_text_for_tts_collapses_whitespace_without_removing_punctuation():
    text = clean_text_for_tts("Xin   chào \n bạn !")

    assert text == "Xin chào bạn!"


def test_tts_main_rejects_removed_tts_method(monkeypatch, tmp_path):
    from core.all_tts_functions import tts_main as tts_module
    out = tmp_path / "row.wav"

    def fake_load_key(key, default=None):
        if key == "tts_method":
            return "mlx_indextts2"
        return default

    monkeypatch.setattr(tts_module, "load_key", fake_load_key)

    with pytest.raises(ValueError, match="Unknown TTS method"):
        tts_module.tts_main(
            "Hello.",
            str(out),
            1,
            task_df=None,
            task_row={"tts_method": "edge_tts", "tts_backend": ""},
        )


def test_single_character_emotional_reply_is_synthesized(monkeypatch, tmp_path):
    from core.all_tts_functions import tts_main as module
    from core.all_tts_functions import mlx_router

    calls = []
    monkeypatch.setattr(module, "load_key", lambda key, default=None: "mlx_indextts2")
    monkeypatch.setattr(module, "get_audio_duration", lambda path: 0.6)
    monkeypatch.setattr(mlx_router, "mlx_router_tts", lambda text, *args, **kwargs: calls.append(text))
    module.tts_main("I...", str(tmp_path / "reply.wav"), 1, None)
    assert calls == ["I..."]
