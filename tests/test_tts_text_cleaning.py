from core.all_tts_functions.tts_main import clean_text_for_tts


def test_clean_text_for_tts_preserves_ranges_and_slashes():
    text = clean_text_for_tts("1-10 người, muốn nồi to/lòng kép, giá chỉ trên 100k.")

    assert "1-10 người" in text
    assert "to/lòng" in text
    assert "100k" in text


def test_clean_text_for_tts_collapses_whitespace_without_removing_punctuation():
    text = clean_text_for_tts("Xin   chào \n bạn !")

    assert text == "Xin chào bạn!"


def test_tts_main_uses_row_level_edge_tts_fallback(monkeypatch, tmp_path):
    from core.all_tts_functions import edge_tts as edge_module
    from core.all_tts_functions import tts_main as tts_module

    calls = {}
    out = tmp_path / "row.wav"

    def fake_load_key(key, default=None):
        if key == "tts_method":
            return "mlx_indextts2"
        if key == "edge_tts":
            return {"voice": "en-US-AvaMultilingualNeural"}
        return default

    def fake_edge(text, save_path):
        calls["text"] = text
        calls["save_path"] = save_path
        out.write_bytes(b"fake-wav")

    monkeypatch.setattr(tts_module, "load_key", fake_load_key)
    monkeypatch.setattr(edge_module, "edge_tts", fake_edge)
    monkeypatch.setattr(tts_module, "get_audio_duration", lambda path: 1.0)

    tts_module.tts_main(
        "Hello.",
        str(out),
        1,
        task_df=None,
        task_row={"tts_method": "edge_tts", "tts_backend": ""},
    )

    assert calls["text"] == "Hello."
    assert calls["save_path"] == str(out)
