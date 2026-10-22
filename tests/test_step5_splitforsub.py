from pathlib import Path

import pandas as pd
import pytest

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


def test_local_layout_preserves_reviewed_text_speakers_and_cues(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    original = pd.DataFrame({"LineID": ["L1"], "Speaker": ["S01"], "Source": ["一定给县长一个惊喜。"], "Translation": ["We'll definitely give the County Magistrate a surprise."]})
    for name, filename in [("INPUT_FILE", "input.xlsx"), ("OUTPUT_SPLIT_FILE", "subtitle.xlsx"), ("OUTPUT_REMERGED_FILE", "audio.xlsx")]:
        monkeypatch.setattr(step5_splitforsub, name, str(tmp_path / filename))
    original.to_excel(tmp_path / "input.xlsx", index=False)
    monkeypatch.setattr(step5_splitforsub, "load_key", lambda k, default=None: True if k == "translation.preserve_cues" else default)
    monkeypatch.setattr(step5_splitforsub, "get_translation_strategy", lambda: "localized")
    monkeypatch.setattr(step5_splitforsub, "record_llm_stage", lambda *a, **kw: None)
    monkeypatch.setattr(step5_splitforsub, "split_align_subs", lambda *a: (_ for _ in ()).throw(AssertionError("Accepted translation must not be rewritten for layout")))
    step5_splitforsub.split_for_sub_main()
    pd.testing.assert_frame_equal(pd.read_excel(tmp_path / "subtitle.xlsx"), original)
    pd.testing.assert_frame_equal(pd.read_excel(tmp_path / "audio.xlsx"), original)


@pytest.mark.parametrize("source_language,target_language,limits", [
    ("en", "zh", (22, 42)), ("zh", "en", (42, 22)),
])
def test_locked_subtitle_stage_uses_renderer_language_limits(
        tmp_path, monkeypatch, source_language, target_language, limits):
    from core import atomic_dubbing, bilingual_alignment, plan_subtitles, runtime_context, utterance_plan
    monkeypatch.chdir(tmp_path)
    Path("output/log").mkdir(parents=True)
    Path("output/log/translation_speech_preflight.json").write_text("{}")
    frame = pd.DataFrame({"LineID": ["L1"], "Source": ["Save it."], "Translation": ["保存它。"]})
    frame.to_excel("input.xlsx", index=False)
    for key, filename in [("INPUT_FILE", "input.xlsx"), ("OUTPUT_SPLIT_FILE", "split.xlsx"),
                          ("OUTPUT_REMERGED_FILE", "remerged.xlsx")]:
        monkeypatch.setattr(step5_splitforsub, key, filename)
    plan = {"units": [{"unit_id": "L1", "source": "Save it.", "target": "保存它。"}]}
    monkeypatch.setattr(utterance_plan, "plan_active", lambda: True)
    monkeypatch.setattr(utterance_plan, "load_plan", lambda: plan)
    monkeypatch.setattr(atomic_dubbing, "verify_preflight_identity", lambda _: None)
    monkeypatch.setattr(plan_subtitles, "align_target_words", lambda _: {"units": [{"unit_id": "L1", "words": []}]})
    monkeypatch.setattr(plan_subtitles, "attach_source_words", lambda _: None)
    monkeypatch.setattr(runtime_context, "effective_source_language", lambda *_: source_language)
    monkeypatch.setattr(runtime_context, "effective_target_language", lambda *_: target_language)
    # A user-configured large limit must still obey the renderer's hard cap.
    monkeypatch.setattr(bilingual_alignment, "load_key", lambda key, default=None: 80 if key.endswith("max_length") else default)
    monkeypatch.setattr(step5_splitforsub, "load_key", lambda key, default=None: 80 if key.endswith("max_length") else default)
    seen = []
    def alignment(_, *, chars, source_chars):
        seen.append((chars, source_chars))
        return {"units": []}
    monkeypatch.setattr(bilingual_alignment, "ensure_bilingual_alignment", alignment)
    monkeypatch.setattr(plan_subtitles, "layout_plan", lambda _, **kwargs: (frame.copy(), {"limits": (kwargs["chars"], kwargs["source_chars"])}))
    monkeypatch.setattr(step5_splitforsub, "record_llm_stage", lambda *a, **kw: None)
    step5_splitforsub.split_for_sub_main()
    assert seen == [limits]
    assert bilingual_alignment.display_limits() == limits
    pd.testing.assert_frame_equal(pd.read_excel("remerged.xlsx"), frame)
