from types import SimpleNamespace
import json

import pytest

from core.all_whisper_methods.local_aligned import apply_source_review
from core.all_whisper_methods.qwen3_aligned import merge_zero_duration_words
from core.dialogue_segmentation import group_dialogue_words, is_lexically_joined_span
from core.pipeline.runner import PipelineRun, PipelineStep, run_pipeline


def test_alignment_context_windows_do_not_include_neighboring_speech_twice():
    from scripts.align_source_audio import alignment_windows
    groups = [{'start': 0., 'end': 2.48}, {'start': 2.51, 'end': 5.2}, {'start': 6., 'end': 8.}]
    windows = alignment_windows(groups, 9.)
    assert windows[0][1] == windows[1][0] == pytest.approx(2.495)
    assert windows[1][1] == 5.45 and windows[2][0] == 5.75
    assert all(a[1] <= b[0] for a, b in zip(windows, windows[1:]))
    assert groups[0]['end'] == 2.48


def test_source_review_is_audio_bound_and_retains_auditable_before_text():
    original = [{"start": 2., "end": 3., "text": "你出钱"}, {"start": 3., "end": 4., "text": "我交费"}]
    review = {"audio_sha256": "sha256:original", "edits": [{"id": "verb", "start": 3., "end": 4.,
              "before": "我交费", "after": "我剿匪", "evidence": ["published reference and acoustic review"]}]}
    updated, edits = apply_source_review(original, review, audio_sha256="sha256:original", start=0, end=10)
    assert updated[1]["text"] == "我剿匪"
    assert original[1]["text"] == "我交费"
    assert edits[0]["observed_before"] == "我交费"
    with pytest.raises(ValueError, match="different original audio"):
        apply_source_review(original, review, audio_sha256="sha256:other", start=0, end=10)
    original[1]["text"] = "我已经付钱"
    with pytest.raises(ValueError, match="no longer matches"):
        apply_source_review(original, review, audio_sha256="sha256:original", start=0, end=10)


def test_review_requires_evidence_and_never_applies_outside_its_audio_chunk():
    original = [{"start": 2., "end": 3., "text": "原文"}]
    review = {"audio_sha256": "sha256:original", "edits": [{"start": 12., "end": 13.,
              "before": "原文", "after": "校对稿", "evidence": []}]}
    assert apply_source_review(original, review, audio_sha256="sha256:original", start=0, end=10)[1] == []
    with pytest.raises(ValueError, match="evidence"):
        apply_source_review(original, review, audio_sha256="sha256:original", start=10, end=20)


def test_recognized_phrase_survives_unstable_acoustic_labels_with_unknown_identity(tmp_path):
    rows = [{"text": "你出", "start": 1., "end": 1.3, "speaker": "S01"},
            {"text": "钱我", "start": 1.3, "end": 1.7, "speaker": "S02"},
            {"text": "剿匪", "start": 1.7, "end": 2., "speaker": "S01"}]
    for row in rows:
        row.update(recognition_segment_id="source-1", recognition_text="你出钱我剿匪", recognition_start=1., recognition_end=2.)
    nlp = SimpleNamespace(make_doc=lambda text: [])
    groups, report = group_dialogue_words(rows, nlp, "")
    assert len(groups) == 1 and groups[0]["speaker"] is None
    assert [r["speaker"] for r in rows] == ["S01", "S02", "S01"]
    path = tmp_path / "boundaries.json"
    path.write_text(json.dumps(report))
    assert is_lexically_joined_span(rows, 0, 2, path)
    rows[1]["recognition_segment_id"] = "changed"
    with pytest.raises(ValueError, match="source changed"):
        is_lexically_joined_span(rows, 0, 2, path)


def test_stage_limited_run_does_not_claim_a_rendered_or_accepted_video(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run = PipelineRun(run_id="source-check", profile="subtitle", source="zh", target="en",
                      steps=["probe"], subtitle_only=True, stop_after="probe")
    result = run_pipeline(run, [PipelineStep("probe", "Source probe", action=lambda: None)])
    assert result.status == "stopped_after_step"
    assert result.quality_status == "not_evaluated"
    assert result.quality_gate_passed is False


def test_collapsed_character_joins_nearest_word_and_never_migrates_to_next_phrase():
    rows = [{"word": "怪", "start": 9.42, "end": 9.66, "recognition_segment_id": "a"},
            {"word": "了", "start": 10.3, "end": 10.3, "recognition_segment_id": "a"},
            {"word": "四", "start": 11.5, "end": 11.74, "recognition_segment_id": "a"},
            {"word": "筒", "start": 11.82, "end": 11.98, "recognition_segment_id": "a"},
            {"word": "啊", "start": 12., "end": 12., "recognition_segment_id": "a"},
            {"word": "全", "start": 12.1, "end": 12.3, "recognition_segment_id": "b"}]
    merged = merge_zero_duration_words(rows, language="zh")
    assert [r["word"] for r in merged] == ["怪了", "四", "筒啊", "全"]
    assert merged[0]["end"] == 10.3
    assert rows[0]["word"] == "怪"


def test_collapsed_run_preserves_order_and_english_spaces():
    merged = merge_zero_duration_words([
        {"word": "I", "start": 1., "end": 1.},
        {"word": "do", "start": 1., "end": 1.},
        {"word": "agree.", "start": 1., "end": 1.5}], language="en")
    assert merged[0]["word"] == "I do agree."


def test_recognition_phrases_and_punctuation_survive_identical_acoustic_labels():
    rows = [{"text": "这是惊喜", "start": 1., "end": 2., "speaker": "S01", "recognition_segment_id": "a",
             "recognition_text": "这是惊喜！"},
            {"text": "什么", "start": 2., "end": 2.2, "speaker": "S01", "recognition_segment_id": "b",
             "recognition_text": "什么？"}]
    groups, _ = group_dialogue_words(rows, SimpleNamespace(), "")
    assert [g["source_text"] for g in groups] == ["这是惊喜！", "什么？"]
    rows[0]["text"] += "什"
    with pytest.raises(ValueError, match="differ from"):
        group_dialogue_words(rows, SimpleNamespace(), "")


def test_recognized_source_units_bypass_comma_and_llm_splitting(tmp_path, monkeypatch):
    from core import step3_1_spacy_split as spacy_step, step3_2_splitbymeaning as meaning_step
    monkeypatch.chdir(tmp_path)
    log = tmp_path / 'output/log'
    log.mkdir(parents=True)
    source = '三天后我出钱，你们剿匪，恢复我的生意。\n他说，这是惊喜。'
    def marked(_):
        (log / 'sentence_by_mark.txt').write_text(source)
        (log / 'lexical_boundary_review.json').write_text(json.dumps({
            'policy': 'recognition_phrases_and_punctuation_preserved; acoustic_words_timing_and_labels_unchanged'}))
    monkeypatch.setattr(spacy_step, 'init_nlp', lambda: object())
    monkeypatch.setattr(spacy_step, 'split_by_mark', marked)
    monkeypatch.setattr(spacy_step, 'split_by_comma_main', lambda nlp: pytest.fail('Recognized promise split at a comma'))
    spacy_step.split_by_spacy()
    assert (log / 'sentence_splitbynlp.txt').read_text() == source
    monkeypatch.setattr(meaning_step, 'init_nlp', lambda: pytest.fail('Source units sent back through LLM splitting'))
    monkeypatch.setattr(meaning_step, 'record_llm_stage', lambda *args, **kwargs: None)
    meaning_step.split_sentences_by_meaning()
    assert (log / 'sentence_splitbymeaning.txt').read_text() == source
    assert json.loads((log / 'source_segmentation.json').read_text())['llm_rewording'] is False


def test_summary_wrapper_repair_cannot_invent_missing_character_analysis():
    from core.step4_1_summarize import normalize_summary_response
    summary = {'topic': 'A conversation.', 'terms': []}
    characters = [{'character_id': 'C01', 'label': 'unknown', 'evidence': []}]
    assert normalize_summary_response([summary, characters]) == {**summary, 'characters': characters}
    assert normalize_summary_response(summary) == summary
    for invalid in ([summary, []], [summary, [{'label': 'invented'}]], [summary, characters, {}]):
        assert normalize_summary_response(invalid) == invalid
