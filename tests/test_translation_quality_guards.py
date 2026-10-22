from pathlib import Path
import json

import pandas as pd
import pytest

from core import translation_review as review
from core.source_verification import compare_transcripts
from core.subtitle_layout import layout_subtitles
from core.translation_context import ground_dialogue_context, speaker_profiles_context
from core.translation_invariants import english_number_value, repair_fact_conflicts, source_fact_conflicts, normalize_explicit_source_facts


@pytest.fixture(autouse=True)
def english_review_fixtures(monkeypatch):
    original = review.load_key
    monkeypatch.setattr(review, 'load_key', lambda key, default=None:
                        'en' if key == 'target_language' else original(key, default))


@pytest.fixture
def batch_review(monkeypatch):
    """These findings/quote tests exercise the diagnostic batch backend."""
    current = review.load_key
    monkeypatch.setattr(review, 'load_key', lambda key, default=None:
                        'batch' if key == 'translation.review_backend' else current(key, default))


def test_explicit_payment_agent_guard_does_not_confuse_another_action_subject():
    assert source_fact_conflicts("你出钱。", "I'll pay.", "en")
    assert source_fact_conflicts("我出钱。", "You pay.", "en")
    assert not source_fact_conflicts("你出钱，我剿匪。", "You pay; I'll fight the bandits.", "en")
    assert not source_fact_conflicts("我出一百八十万，给你们出城剿匪。", "I'll pay 1.8 million for you to leave town and fight bandits.", "en")


def test_departing_a_location_cannot_become_conquering_it_during_compression():
    assert source_fact_conflicts("出城剿匪。","Take over the city and fight bandits.","en")
    assert not source_fact_conflicts("出城剿匪。","Leave town to fight bandits.","en")
    assert not source_fact_conflicts("攻占城镇。","Take over the town.","en")


def test_compression_cannot_erase_an_explicit_departure_location():
    assert 'explicit_departure_location_lost:settlement' in source_fact_conflicts('出城剿匪。', 'Go out and fight bandits.', 'en')
    assert not source_fact_conflicts('出城剿匪。', 'Leave town and fight bandits.', 'en')
    assert not source_fact_conflicts('出村办事。', 'Leave the village on business.', 'en')
    assert not source_fact_conflicts('出境办事。', 'Go abroad on business.', 'en')


def test_number_words_and_digits_are_equivalent():
    assert english_number_value("one million eight hundred thousand") == english_number_value("1.8 million")
    assert english_number_value("1,800,000") == english_number_value("1.8 million")
    assert english_number_value("one hundred and five") == 105


def test_person_pronoun_one_neither_adds_a_numeric_fact_nor_becomes_digit_in_dialogue():
    from core.translation_invariants import number_annotations
    from core.local_translation import clean_dialogue_rendering
    source='杀人者是你家的人。'
    text='The one who killed him is from your house. No one else was there.'
    assert number_annotations(text)==[]
    assert clean_dialogue_rendering(source,text,'en')[0]==text
    actual=number_annotations('One witness received one million and eight hundred thousand in three days.')
    assert [a['value'] for a in actual]==['1','1800000','3']


def test_repair_cannot_invent_currency():
    assert repair_fact_conflicts("给你一百八十万", "I'll pay you 1.8 million.", "I'll pay you 1.8 million dollars.", "en")
    assert not repair_fact_conflicts("给你一百八十万美元", "I'll pay you 1.8 million.", "I'll pay you 1.8 million dollars.", "en")


def test_initial_translation_cannot_change_explicit_possession_or_add_payer():
    assert source_fact_conflicts("修我的车", "Fix your car.", "en")
    assert not source_fact_conflicts("修我的车", "Fix my car.", "en")
    assert source_fact_conflicts("给你两千修他的车", "I'll give you 2,000 to fix his car.", "en")
    assert not source_fact_conflicts("给你两千修他的车", "You'll get 2,000 to fix his car.", "en")
    assert not source_fact_conflicts("我给你两千修他的车", "I'll give you 2,000 to fix his car.", "en")
    assert not source_fact_conflicts("把肉捣碎", "Pound the meat.", "en")
    assert not source_fact_conflicts("三磅肉", "Three pounds of meat.", "en")


def test_first_person_beloved_can_use_a_natural_relative_clause():
    source = "他是我的挚爱，你不能夺我所爱。"
    assert not source_fact_conflicts(source, "He is the one I love most. You can't take him away from me.", "en")
    assert source_fact_conflicts(source, "He is the one she loves most.", "en")
    assert source_fact_conflicts("修我的车", "Fix the car I love.", "en")


def test_neutral_payer_normalization_preserves_other_actions_and_modal():
    source = "三天后给你两千修好他的车。"
    text = "In three days, I'll give you 2,000 to fix his car."
    result, reasons = normalize_explicit_source_facts(source, text, "en")
    assert result == "In three days, you'll receive 2,000 to fix his car."
    assert reasons == ["omitted_payer_kept_unspecified"]
    assert normalize_explicit_source_facts("我给你两千修好他的车。", text, "en") == (text, [])


def test_multiple_owners_are_not_blindly_reassigned():
    source = "我的车和他的房子"
    text = "Your car and his house"
    assert normalize_explicit_source_facts(source, text, "en") == (text, [])


def test_omitted_numeric_tael_unit_is_removed_without_changing_other_facts():
    text = "I'll give you one million eight hundred thousand taels of silver in three days."
    result, reasons = normalize_explicit_source_facts("三天后我出一百八十万", text, "en")
    assert result == "I'll give you one million eight hundred thousand in three days."
    assert reasons == ["unsupported_numeric_tael_unit_removed"]
    assert normalize_explicit_source_facts("三天后我出一百八十万两银子", text, "en") == (text, [])
    assert normalize_explicit_source_facts("三天后我出一百八十万", "I'll pay 1.8 million dollars.", "en")[0].endswith("dollars.")
    assert source_fact_conflicts("三天后我出一百八十万", "I'll pay 1.8 million dollars.", "en")


def test_explicit_tael_without_silver_word_is_preserved_but_digit_two_is_not_a_unit():
    text = "I'll pay 1.8 million taels."
    assert normalize_explicit_source_facts("我出一百八十万两", text, "en") == (text, [])
    assert not source_fact_conflicts("我出一百八十万两", text, "en")
    assert source_fact_conflicts("我出两千", "I'll pay two thousand taels.", "en")


@pytest.mark.parametrize("text", ["Fix S01's leg.", "Tell SPEAKER_00.", "[S01] Hello.", "L00027 says hello."])
def test_internal_roles_cannot_become_spoken_dialogue(text):
    assert not review.is_valid_target_language(text, "en")


def test_equal_repair_does_not_resolve_a_critical_finding(tmp_path, monkeypatch):
    monkeypatch.setattr(review, "ask_gpt", lambda *a, **kw: {
        "issues": [{"line_id": "L1", "severity": "critical", "problem": "Name is accurate but action is wrong",
                    "suggested_fix": "Hello."}]
    })
    _, report = review.review_and_repair_translation(
        pd.DataFrame([{"LineID": "L1", "Speaker": "S01", "Source": "你好", "Translation": "Hello."}]),
        {}, report_path=tmp_path / "review.json")
    assert report["status"] == "failed"


def test_final_audit_does_not_publish_speaker_leak(tmp_path, monkeypatch):
    replies = [
        {"issues": [{"line_id": "L1", "severity": "critical", "problem": "correct actor",
                     "suggested_fix": "Fix his leg."}]},
        {"issues": [{"line_id": "L1", "severity": "critical", "problem": "clarify referent",
                     "suggested_fix": "Fix S01's leg."}]},
    ]
    monkeypatch.setattr(review, "ask_gpt", lambda *a, **kw: replies.pop(0))
    result, report = review.review_and_repair_translation(
        pd.DataFrame([{"LineID": "L1", "Speaker": "S02", "Source": "修好他的腿", "Translation": "Fix my leg."}]),
        {}, report_path=tmp_path / "review.json", max_repair_rounds=1)
    assert report["status"] == "failed"
    assert result.iloc[0]["Translation"] == "Fix his leg."


def test_translation_specialist_audit_never_rewrites_the_candidate(tmp_path, monkeypatch, batch_review):
    monkeypatch.setattr(review, "ask_gpt", lambda *a, **kw: {
        "issues": [{"line_id": "L1", "severity": "critical", "problem": "faulty self-audit",
                    "suggested_fix": "Open the door."}]
    })
    original = pd.DataFrame([{"LineID": "L1", "Speaker": "S01", "Source": "不要开门", "Translation": "Don't open the door."}])
    result, report = review.review_and_repair_translation(original, {}, report_path=tmp_path / "review.json", audit_only=True)
    pd.testing.assert_frame_equal(result, original)
    assert report["status"] == "needs_review"
    assert report["accepted"] is False
    assert report["review_mode"] == "audit_only"
    assert report["repairs_applied_count"] == 0


def test_audit_quote_errors_are_recorded_without_fabricated_evidence(tmp_path, monkeypatch, batch_review):
    reply = {"issues": [{"line_id": "L1", "severity": "critical", "source_quote": "你好",
                         "translation_quote": "Hi!", "problem": "unfounded change", "suggested_fix": "Hi!"},
                        {"line_id": "L2", "severity": "critical", "problem": "wrong batch", "suggested_fix": "Hello."}]}
    def audit(prompt, **kwargs):
        assert kwargs["valid_def"](reply)["status"] == "success"
        return reply
    monkeypatch.setattr(review, "ask_gpt", audit)
    original = pd.DataFrame([{"LineID": "L1", "Source": "你好", "Translation": "Hello."}])
    result, report = review.review_and_repair_translation(original, {}, tmp_path / "review.json", audit_only=True)
    pd.testing.assert_frame_equal(result, original)
    assert report["status"] == "needs_review"
    assert report["review_executed"] is True
    assert len(report["issues"]) == 1
    finding = report["issues"][0]
    assert finding["translation_quote"] == "Hi!"  # do not relabel canonical text as model evidence
    assert finding["reviewed_translation"] == "Hello."
    assert finding["evidence_status"] == "unverified"
    assert len(report["evidence_warnings"]) == 2


def test_audit_service_failure_preserves_pending_candidate(tmp_path, monkeypatch, batch_review):
    def unavailable(*args, **kwargs):
        raise TimeoutError("local review unavailable")
    monkeypatch.setattr(review, "ask_gpt", unavailable)
    original = pd.DataFrame([{"LineID": "L1", "Source": "你好", "Translation": "Hello."}])
    result, report = review.review_and_repair_translation(original, {}, tmp_path / "review.json", audit_only=True)
    pd.testing.assert_frame_equal(result, original)
    assert report["status"] == "needs_review"
    assert report["review_executed"] is False
    assert report["accepted"] is False


def test_pending_audit_cannot_waive_invalid_target_text(tmp_path, monkeypatch):
    monkeypatch.setattr(review, "ask_gpt", lambda *a, **kw: {"issues": []})
    _, report = review.review_and_repair_translation(
        pd.DataFrame([{"LineID": "L1", "Source": "你好", "Translation": "C01 says 你好"}]),
        {}, tmp_path / "review.json", audit_only=True)
    assert report["status"] == "failed"


def test_acoustic_disagreement_is_not_silently_accepted():
    assert compare_transcripts("你出钱，我剿匪。", "你出钱，我交费。")["status"] == "needs_review"
    severe = compare_transcripts("完全损坏的识别文字", "明天在门口见")
    assert severe["status"] == "needs_review"
    assert severe["disagreement_level"] == "high"
    assert compare_transcripts("Hello, there!", "hello there")["status"] == "pass"
    assert compare_transcripts("Hello", "")["status"] == "failed"


def test_unsupported_role_and_misaligned_intent_are_not_forwarded():
    rows = pd.DataFrame([{"line_id": "L1", "speaker": "S01", "text": "他说他明天来。"}])
    summary = ground_dialogue_context({
        "speaker_profiles": [{"speaker_id": "S01", "role": "king", "evidence": ["invented text"],
                              "translation_guidance": "Avoid pronouns forever."}],
        "dialogue_units": [{"line_id": "L1", "speaker_id": "S02", "source_quote": "他说",
                            "speech_act": "promise"}],
    }, rows)
    assert summary["speaker_profiles"][0]["role"] == "unknown"
    assert summary["dialogue_units"][0]["speech_act"] == "unknown"
    assert "Avoid pronouns" not in speaker_profiles_context(summary)


def test_grounded_relayer_annotation_reaches_translation():
    rows = pd.DataFrame([{"line_id": "L1", "speaker": "S02", "text": "他说他明天来。"},
                         {"line_id": "L2", "speaker": "S01", "text": "好。"}])
    summary = ground_dialogue_context({
        "speaker_profiles": [],
        "dialogue_units": [{"line_id": "L1", "speaker_id": "S02", "source_quote": "他说他明天来",
                            "speech_act": "relay", "relays_speaker": "S01", "addressed_to": "unknown"}],
    }, rows)
    payload = json.loads(speaker_profiles_context(summary))
    assert payload["dialogue_units"][0]["relays_speaker"] == "S01"
    assert payload["speaker_profiles"][0]["identity_verified"] is False


def test_layout_preserves_complete_text_and_speaker_window():
    text = "In three days, he will give you one million eight hundred thousand taels of silver to leave the city, deal with the bandits, and get his leg fixed."
    original = pd.DataFrame([{"LineID": "L27", "speaker": "S03", "Source": "三天之后他给你一百八十万两银子出城剿匪修好他的腿",
                              "Translation": text, "timestamp": (10.0, 18.0)}])
    words = pd.DataFrame({"start": [10 + i * .4 for i in range(20)]})
    display, report = layout_subtitles(original, words)
    assert len(display) > 1
    assert " ".join(" ".join(display.Translation).split()) == text
    assert "".join(display.Source) == original.iloc[0].Source
    assert original.iloc[0].Translation == text
    assert set(display.speaker) == {"S03"}
    assert display.iloc[0].timestamp[0] == 10
    assert display.iloc[-1].timestamp[1] == 18
    assert all(len(line) <= 38 for part in display.Translation for line in part.splitlines())
    assert all(len(part.splitlines()) <= 2 for part in display.Translation)
    assert all(left[1] == right[0] for left, right in zip(display.timestamp, display.timestamp[1:]))


def test_display_padding_uses_silence_without_changing_semantic_units():
    frame = pd.DataFrame([
        {"LineID": "L1", "Source": "好", "Translation": "Okay.", "timestamp": (1., 1.2), "speaker": "S01"},
        {"LineID": "L2", "Source": "对", "Translation": "Right.", "timestamp": (3., 3.3), "speaker": "S02"},
    ])
    original = frame.copy(deep=True)
    words = pd.DataFrame({"start": [1., 3., 5.], "end": [1.2, 3.3, 5.2]})
    display, report = layout_subtitles(frame, words)
    pd.testing.assert_frame_equal(frame, original)
    assert display.iloc[0].timestamp == (1., 1.8)
    assert display.iloc[1].timestamp == (3., 3.8)
    assert display.iloc[0].spoken_timestamp == (1., 1.2)
    assert display.iloc[0].timestamp[1] <= display.iloc[1].timestamp[0]
    assert report["status"] == "pass"
    assert len(report["timing_adjustments"]) == 2


def test_display_padding_cannot_cover_untranslated_or_overlapping_speech():
    frame = pd.DataFrame([{"LineID": "L1", "Source": "好", "Translation": "All right.",
                           "timestamp": (1., 1.2), "speaker": "S01"}])
    words = pd.DataFrame({"start": [.9, 1., 1.3], "end": [1.05, 1.2, 1.8]})
    display, report = layout_subtitles(frame, words)
    assert display.iloc[0].timestamp == (1., 1.3)
    assert report["status"] == "needs_review"
    words.loc[2, "start"] = 1.1
    display, report = layout_subtitles(frame, words)
    assert display.iloc[0].timestamp == (1., 1.2)
    assert report["timing_adjustments"] == []


def test_display_padding_shares_adjacent_caption_gap_without_overlap():
    frame = pd.DataFrame([
        {"LineID": "L1", "Source": "好", "Translation": "All right.", "timestamp": (1., 1.1)},
        {"LineID": "L2", "Source": "对", "Translation": "Yes, that's right.", "timestamp": (1.4, 1.5)},
    ])
    words = pd.DataFrame({"start": [1., 1.4, 3.], "end": [1.1, 1.5, 3.2]})
    display, _ = layout_subtitles(frame, words)
    assert display.iloc[0].timestamp[1] <= display.iloc[1].timestamp[0]
    assert display.iloc[0].timestamp[1] <= 1.25
    assert display.iloc[1].timestamp[0] >= 1.25


def test_canonical_video_wins_over_older_render(tmp_path, monkeypatch):
    from core import step1_ytdlp
    (tmp_path / "AI字幕.mp4").write_bytes(b"old render")
    (tmp_path / "source.mp4").write_bytes(b"original")
    monkeypatch.setattr(step1_ytdlp, "load_key", lambda key: ["mp4"])
    assert Path(step1_ytdlp.find_video_files(str(tmp_path))).name == "source.mp4"


def test_stale_pipeline_checkpoint_is_never_restamped(tmp_path, monkeypatch):
    from core.pipeline.runner import PipelineRun, PipelineStep, run_pipeline
    monkeypatch.chdir(tmp_path)
    path = Path("output/log/cleaned_chunks.xlsx")
    path.parent.mkdir(parents=True)
    path.write_bytes(b"old unproven artifact")
    with pytest.raises(RuntimeError, match="Stale transcribe checkpoint"):
        run_pipeline(PipelineRun("new", "subtitle", "zh", "en", []),
                     [PipelineStep("transcribe", "ASR", action=lambda: None)])
    assert path.read_bytes() == b"old unproven artifact"
    assert not Path(str(path) + ".manifest.json").exists()


def test_source_uncertainty_produces_pending_candidate_not_quality_pass(tmp_path, monkeypatch):
    monkeypatch.setattr(review, "ask_gpt", lambda *a, **kw: {
        "issues": [{"line_id": "L1", "severity": "warning", "issue_type": "source_uncertainty",
                    "problem": "Name is unclear in ASR", "suggested_fix": ""}]
    })
    original = pd.DataFrame([{"LineID": "L1", "Speaker": "S01", "Source": "胡文", "Translation": "Hu Wen"}])
    result, report = review.review_and_repair_translation(original, {}, report_path=tmp_path / "review.json")
    assert report["status"] == "needs_review"
    pd.testing.assert_frame_equal(result, original)


def test_forced_alignment_retains_collapsed_tokens():
    from core.all_whisper_methods.qwen3_aligned import merge_zero_duration_words
    words = [{"word": "胡", "start": 1., "end": 1.},
             {"word": "万", "start": 1., "end": 1.3},
             {"word": "呢", "start": 1.3, "end": 1.3}]
    merged = merge_zero_duration_words(words)
    assert merged == [{"word": "胡万呢", "start": 1., "end": 1.3}]
    with pytest.raises(ValueError):
        merge_zero_duration_words([words[0]])
