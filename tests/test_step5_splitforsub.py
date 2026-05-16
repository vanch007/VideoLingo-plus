from pathlib import Path

import pandas as pd

from core import step5_splitforsub


def test_align_subs_remerges_translation_with_target_language_joiner(monkeypatch):
    languages = []

    monkeypatch.setattr(
        step5_splitforsub,
        "ask_gpt",
        lambda *args, **kwargs: {
            "align": [
                {"target_part_1": "Xin chào"},
                {"target_part_2": "bạn"},
            ]
        },
    )
    monkeypatch.setattr(step5_splitforsub, "get_align_prompt", lambda *args, **kwargs: "prompt")

    def fake_load_key(key, default=None):
        if key == "target_language":
            return "vi"
        return default

    def fake_get_joiner(language):
        languages.append(language)
        return " "

    monkeypatch.setattr(step5_splitforsub, "load_key", fake_load_key)
    monkeypatch.setattr(step5_splitforsub, "get_joiner", fake_get_joiner)

    _, _, remerged = step5_splitforsub.align_subs("你好朋友", "Xin chào bạn", "你好\n朋友")

    assert languages == ["vi"]
    assert remerged == "Xin chào bạn"


def test_split_for_sub_keeps_audio_remerged_on_original_rows(tmp_path: Path, monkeypatch):
    input_path = tmp_path / "translation_results.xlsx"
    split_path = tmp_path / "translation_results_for_subtitles.xlsx"
    remerged_path = tmp_path / "translation_results_remerged.xlsx"

    pd.DataFrame({"Source": ["abcdef"], "Translation": ["hello verylong world"]}).to_excel(input_path, index=False)

    monkeypatch.setattr(step5_splitforsub, "INPUT_FILE", str(input_path))
    monkeypatch.setattr(step5_splitforsub, "OUTPUT_SPLIT_FILE", str(split_path))
    monkeypatch.setattr(step5_splitforsub, "OUTPUT_REMERGED_FILE", str(remerged_path))
    monkeypatch.setattr(step5_splitforsub, "record_llm_stage", lambda *args, **kwargs: None)

    def fake_load_key(key, default=None):
        if key == "subtitle":
            return {"max_length": 5, "target_multiplier": 1}
        if key == "max_workers":
            return 1
        return default

    calls = {"count": 0}

    def fake_split_align(src_lines, tr_lines):
        calls["count"] += 1
        if calls["count"] == 1:
            return ["abc", "def"], ["hello verylong", "world"], ["hello verylong world"]
        return ["abc", "def"], ["hello", "world"], ["hello", "world"]

    monkeypatch.setattr(step5_splitforsub, "load_key", fake_load_key)
    monkeypatch.setattr(step5_splitforsub, "split_align_subs", fake_split_align)

    step5_splitforsub.split_for_sub_main()

    split_df = pd.read_excel(split_path)
    remerged_df = pd.read_excel(remerged_path)

    assert split_df["Source"].tolist() == ["abc", "def"]
    assert remerged_df["Source"].tolist() == ["abcdef"]
    assert remerged_df["Translation"].tolist() == ["hello verylong world"]
