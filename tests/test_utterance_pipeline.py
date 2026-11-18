"""Regression cases for long takes, independent clocks and late text mutation."""
import copy
import json
from types import SimpleNamespace

import pandas as pd
import pytest
from pydub import AudioSegment
from pydub.generators import Sine

from core import utterance_plan as up
from core import atomic_dubbing as ad
from core import speech_budget as sb
from core.plan_subtitles import layout_plan
from core.source_verification import normalize_transcript


@pytest.fixture(autouse=True)
def mock_semantic_model_for_pipeline_unit_tests(monkeypatch):
    from core import duration_candidate_audit as audit
    from core import runtime_context
    from core import bilingual_alignment
    monkeypatch.setattr(bilingual_alignment, 'prepare_candidate_alignment', lambda *a, **k: [])
    # These fixtures explicitly test Chinese-to-English. The developer's
    # selected UI language must not change their contracts.
    monkeypatch.setattr(runtime_context, 'effective_target_language', lambda *_: 'en')
    monkeypatch.setattr(sb, 'effective_target_language', lambda *_: 'en')
    monkeypatch.setattr(sb,"audit_duration_candidate",lambda *a,**k:{"equivalent":True,"issues":[]})
    monkeypatch.setattr(audit,"source_category_terms",lambda *_:[])


def source_plan(source="你出钱。我剿匪！", *, step=.4, max_seconds=12.):
    tokens = list(normalize_transcript(source))
    words = pd.DataFrame([{"text": t, "start": i * step, "end": (i + 1) * step} for i, t in enumerate(tokens)])
    parents = pd.DataFrame([{"line_id": "L00001", "text": source, "speaker": "S01"}])
    return up.build_plan(parents, words, gap_extension=0, max_seconds=max_seconds)


def translated_plan(source="你出钱。我剿匪！"):
    plan = source_plan(source)
    for i, unit in enumerate(plan["units"]):
        unit.update(target=["You pay.", "I'll fight the bandits."][i], status="translated")
    plan["status"] = "translated"
    return plan


def test_atomic_boundaries_preserve_punctuation_and_repeated_dialogue():
    plan = source_plan("马匪呢？马匪呢？马匪呢？")
    assert [u["source"] for u in plan["units"]] == ["马匪呢？"] * 3
    assert len({u["unit_id"] for u in plan["units"]}) == 3
    assert "".join(u["source"] for u in plan["units"]) == plan["parents"][0]["source"]
    assert all(a["end"] <= b["start"] for a, b in zip(plan["units"], plan["units"][1:]))


def test_needs_review_speech_can_never_be_reused_as_accepted(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ad.PREFLIGHT_REPORT.parent.mkdir(parents=True)
    ad.PREFLIGHT_REPORT.write_text(json.dumps({'status': 'needs_review', 'issues': ['missing speech']}))
    frame = pd.DataFrame([{'LineID': 'L1', 'Source': 'Go', 'Translation': '走'}])
    with pytest.raises(ValueError, match='repeat preflight'):
        ad.verify_preflight_identity(frame)


def test_title_case_text_cannot_waive_a_bad_readback():
    tasks = pd.DataFrame([{'text': 'Stop Now.', 'asr_transcript': 'Start now.',
                          'asr_content_score': .4, 'asr_status': 'ok'}])
    assert 'readback_below_threshold' in ad._content_issues(tasks)[0]


@pytest.mark.parametrize('transcript', [None, float('nan'), pd.NA, '', '  ', 42, []])
def test_missing_readback_is_rejected_without_entering_text_comparison(transcript):
    tasks = pd.DataFrame([{'text': 'Keep the source.', 'asr_transcript': transcript,
                          'asr_content_score': 1., 'asr_status': 'ok'}])
    assert ad._content_issues(tasks) == {0: ['empty_or_unavailable_readback']}


@pytest.mark.parametrize('status', ['fail', None, pd.NA])
def test_failed_readback_cannot_reuse_matching_old_text(status):
    tasks = pd.DataFrame([{'text': 'Keep the source.', 'asr_transcript': 'Keep the source.',
                          'asr_content_score': 1., 'asr_status': status}])
    assert ad._content_issues(tasks) == {0: ['empty_or_unavailable_readback']}


def test_duration_context_resolves_a_child_unit_through_its_exact_parent_source():
    unit = {'unit_id': 'L2.01', 'parent_line_id': 'L2', 'source': '不用翻译，就是惊喜。',
            'parent_source': '不用翻译，就是惊喜。',
            'source_slices': [{'line_id': 'L2', 'source': '不用翻译，就是惊喜。'}]}
    data = {'characters': [{'character_id': 'C1', 'name': 'Tang'}], 'dialogue_units': [
        {'line_id': 'L1', 'source_text': '我让你翻译给我听，什么叫惊喜？'},
        {'line_id': 'L2', 'source_text': unit['source'], 'character_id': 'C1', 'confidence': .9},
        {'line_id': 'L3', 'source_text': '难道你听不懂什么叫惊喜？'}]}
    context = json.loads(sb.duration_context(unit, data))
    assert context['context_source_line_ids'] == ['L2']
    assert context['dialogue_role_hints']['speaker'] == 'Tang'
    assert context['surrounding_source'] == [data['dialogue_units'][0]['source_text'], data['dialogue_units'][2]['source_text']]
    data['dialogue_units'][1]['source_text'] = '修改后的另一句台词。'
    context = json.loads(sb.duration_context(unit, data))
    assert context['context_source_line_ids'] == [] and context['surrounding_source'] == []
    assert context['dialogue_role_hints']['speaker'] == 'unknown'


def test_duration_context_does_not_assign_one_role_to_conflicting_merged_turns():
    unit = {'unit_id': 'L1.01__L2.01', 'parent_line_id': 'L1', 'source': '走！ 一起走吧。',
            'parent_source': '走！ 一起走吧。',
            'source_slices': [{'line_id': 'L1', 'source': '走！'}, {'line_id': 'L2', 'source': '一起走吧。'}]}
    data = {'characters': [{'character_id': 'C1', 'name': 'A'}, {'character_id': 'C2', 'name': 'B'}],
            'dialogue_units': [{'line_id': 'L1', 'source_text': '走！', 'character_id': 'C1'},
                               {'line_id': 'L2', 'source_text': '一起走吧。', 'character_id': 'C2'}]}
    context = json.loads(sb.duration_context(unit, data))
    assert context['context_source_line_ids'] == ['L1', 'L2']
    assert context['dialogue_role_hints']['speaker'] == 'unknown'


def test_rapid_repetitions_are_planned_together_before_translation():
    plan = source_plan("马匪呢？马匪呢？马匪呢？", step=.16)
    assert len(plan["units"]) == 1
    assert plan["units"][0]["source"] == "马匪呢？马匪呢？马匪呢？"
    assert plan["units"][0]["split_basis"] == "rapid_repetition"
    assert plan["units"][0]["source_duration"] == pytest.approx(1.44)


def test_short_repetitions_cannot_merge_across_source_turns():
    parents = pd.DataFrame([{"line_id": f"L{i}", "text": "好！", "speaker": f"S{i}"} for i in range(2)])
    words = pd.DataFrame([{"text": "好", "start": i * .3, "end": (i + 1) * .3} for i in range(2)])
    plan = up.build_plan(parents, words, gap_extension=0)
    assert len(plan["units"]) == 2
    assert [u["speaker"] for u in plan["units"]] == ["S0", "S1"]


@pytest.mark.parametrize("speaker,gap,merged", [("S01",0.,True), ("S02",0.,False),
                                               (None,0.,False), ("S01",.3,False)])
def test_subtitle_rows_share_a_short_speech_budget_only_for_a_contiguous_known_voice(speaker, gap, merged):
    parents = pd.DataFrame([{"line_id":"L1", "text":"走！", "speaker":"S01"},
                            {"line_id":"L2", "text":"一起走吧。", "speaker":speaker}])
    words = pd.DataFrame([{"text":"走", "start":0., "end":.4, "speaker":"S01"},
                          {"text":"一起走吧", "start":.4+gap, "end":2.+gap, "speaker":speaker}])
    plan = up.build_plan(parents, words, gap_extension=0)
    assert len(plan["units"]) == (1 if merged else 2)
    if merged:
        unit = plan["units"][0]
        assert unit["source_duration"] == 2.
        assert unit["source_slices"] == [{"line_id":"L1","source":"走！"},
                                          {"line_id":"L2","source":"一起走吧。"}]
        assert unit["source"] == "走！ 一起走吧。"
        broken = copy.deepcopy(plan)
        broken["units"][0]["source_slices"][1]["source"] = "不要走。"
        with pytest.raises(ValueError, match="lineage changed"):
            up.validate_plan(broken)


def test_complete_word_evidence_can_resolve_each_slice_of_a_mixed_parent():
    parents = pd.DataFrame([{"line_id": "L1", "text": "好了。走吧！", "speaker": None}])
    words = pd.DataFrame([{"text": word, "start": i * .3, "end": (i + 1) * .3,
                           "speaker": "S01" if i < 2 else "S02"}
                          for i, word in enumerate("好了走吧")])
    plan = up.build_plan(parents, words, gap_extension=0)
    assert [u["speaker"] for u in plan["units"]] == ["S01", "S02"]
    words.at[1, "speaker"] = None
    plan = up.build_plan(parents, words, gap_extension=0)
    assert plan["units"][0]["speaker"] is None


def test_short_clause_coalescing_requires_same_measured_voice_and_parent():
    parents = pd.DataFrame([{"line_id": "L1", "text": "全是他们。人呢？人呢？", "speaker": "S01"}])
    words = pd.DataFrame([{"text": t, "start": i * .2, "end": (i + 1) * .2, "speaker": "S01"}
                          for i, t in enumerate("全是他们人呢人呢")])
    assert len(up.build_plan(parents, words, gap_extension=0)["units"]) == 1
    words.loc[4:, "speaker"] = "S02"
    plan = up.build_plan(parents, words, gap_extension=0)
    assert len(plan["units"]) == 2
    assert [u["speaker"] for u in plan["units"]] == ["S01", "S02"]


def test_long_dialogue_cuts_clauses_at_measured_words():
    plan = source_plan("你出钱，我剿匪，三天之后给你答复。", step=.6, max_seconds=5.5)
    assert len(plan["units"]) >= 2
    assert max(u["source_duration"] for u in plan["units"]) <= 5.5
    assert "".join(u["source"] for u in plan["units"]) == plan["parents"][0]["source"]
    assert all(u["split_basis"] in {"clause", "sentence", "parent"} for u in plan["units"])


def test_long_unpunctuated_text_is_cut_only_at_measured_word_boundaries():
    plan = source_plan("我愿意等你三天然后再出城剿匪", step=.5, max_seconds=5.5)
    assert len(plan["units"]) == 2
    assert max(u["source_duration"] for u in plan["units"]) <= 5.5
    assert "".join(u["source"] for u in plan["units"]) == plan["parents"][0]["source"]  # nothing sliced away
    assert {u["split_basis"] for u in plan["units"]} == {"parent"}


def test_complete_sentence_keeps_its_shared_budget_before_display_splitting():
    plan = source_plan("三天之后，我给你答复，然后一起去解决问题。", step=.4)
    assert len(plan["units"]) == 1
    assert plan["units"][0]["source_duration"] > 5.5
    assert plan["units"][0]["source"] == plan["parents"][0]["source"]


@pytest.mark.parametrize("quiet,expected_lead", [([], 0.), ([[.5, 1.]], .12)])
def test_short_command_reserves_only_measured_quiet_lead_without_shifting_next_actor(quiet, expected_lead):
    parents = pd.DataFrame([{"line_id":"L1", "text":"走！", "speaker":"S01"},
                            {"line_id":"L2", "text":"好。", "speaker":"S02"}])
    words = pd.DataFrame([{"text":"走", "start":1., "end":1.5, "speaker":"S01"},
                          {"text":"好", "start":1.5, "end":2., "speaker":"S02"}])
    plan = up.build_plan(parents, words, gap_extension=0, quiet_spans=quiet)
    first, second = plan["units"]
    assert first["source_start"] == 1.
    assert first["start"] == pytest.approx(1. - expected_lead)
    assert first["source_duration"] == .5
    assert first["duration"] == pytest.approx(.5 + expected_lead)
    assert second["source_start"] == second["start"] == first["end"] == 1.5
    if expected_lead:
        broken = copy.deepcopy(plan)
        broken["units"][0]["lead_pause_span"] = None
        with pytest.raises(ValueError, match="quiet source evidence"):
            up.validate_plan(broken)


def test_lead_and_trailing_reserves_cannot_double_book_a_short_pause():
    parents = pd.DataFrame([{"line_id":"L1", "text":"先走。", "speaker":"S01"},
                            {"line_id":"L2", "text":"好。", "speaker":"S02"}])
    words = pd.DataFrame([{"text":"先走", "start":0., "end":1., "speaker":"S01"},
                          {"text":"好", "start":1.2, "end":1.7, "speaker":"S02"}])
    plan = up.build_plan(parents, words, quiet_spans=[[1.,1.2]])
    left, right = plan["units"]
    assert left["end"] <= right["start"] - .019
    assert right["source_start"] == 1.2


def test_duration_feedback_cannot_cycle_back_to_a_measured_rejected_draft(monkeypatch):
    monkeypatch.setattr(sb, "effective_target_language", lambda *_: "en")
    prompts = []
    def reply(prompt, **kwargs):
        prompts.append(prompt)
        assert kwargs["valid_def"]({"1":"You pay the money!"})["status"] == "error"
        assert kwargs["valid_def"]({"1":"You pay."})["status"] == "success"
        return {"1":"You pay."}
    monkeypatch.setattr(sb, "ask_gpt", reply)
    result = sb.retranslate_unit("你出钱。", "You need to pay.",
        {"target_max_words":2,"target_speech_seconds":1.,"max_speed_factor":1.08}, "",
        measured_seconds=1.4, measured_candidates=[{"text":"You pay the money.","natural_seconds":1.3}])
    assert result == "You pay."
    assert "已经实测超时的候选" in prompts[0]


def test_measured_chinese_duration_feedback_uses_syllables_and_keeps_speed_cap(monkeypatch):
    monkeypatch.setattr(sb, 'effective_target_language', lambda *_: 'zh')
    from core import runtime_context
    monkeypatch.setattr(runtime_context, 'effective_target_language', lambda *_: 'zh')
    prompts = []
    def reply(prompt, **kwargs):
        prompts.append(prompt)
        return {'1': '它会完成全部工作。'}
    monkeypatch.setattr(sb, 'ask_gpt', reply)
    result = sb.retranslate_unit('It will finish all the work.', '它将会把全部工作都完成。',
        {'target_max_words': 1, 'target_speech_seconds': 1.5, 'max_speed_factor': 1.08}, '',
        measured_seconds=2.5)
    assert result == '它会完成全部工作。'
    assert '读音单元' in prompts[0]
    assert '不超过 1 个读出来的词' not in prompts[0]
    assert '1.080' in prompts[0]


@pytest.mark.parametrize('recovery_passes', [True, False])
def test_duration_rewrite_recovers_display_failure_outside_json_validator(monkeypatch, recovery_passes):
    from core import bilingual_alignment as ba
    monkeypatch.setattr(sb, 'effective_target_language', lambda *_: 'zh')
    checked = []
    candidate, recovered = '先保存，再打开。', '打开它。保存它。'
    def prepare(unit, context):
        checked.append(unit['target'])
        assert unit['unit_id'] == 'L1'
        if unit['target'] == candidate or not recovery_passes:
            raise ValueError('Bilingual semantic alignment failed')
        return [{'source': 'Open it. Save it.', 'target': recovered}]
    monkeypatch.setattr(ba, 'prepare_candidate_alignment', prepare)
    monkeypatch.setattr(ba, 'translate_source_partition', lambda *a: recovered)
    def reply(prompt, **kwargs):
        assert kwargs['valid_def']({'1': candidate})['status'] == 'success'
        assert not checked, 'JSON validation must not hide model-assisted alignment recovery'
        return {'1': candidate}
    monkeypatch.setattr(sb, 'ask_gpt', reply)
    args = ('Open it. Save it.', '打开并存储。',
            {'unit_id': 'L1', 'target_speech_seconds': 3., 'max_speed_factor': 1.08, 'target_max_words': 8}, '')
    if recovery_passes:
        assert sb.retranslate_unit(*args) == recovered
    else:
        with pytest.raises(ValueError, match='Bilingual'):
            sb.retranslate_unit(*args)
    assert checked == [candidate] + [recovered] * (1 if recovery_passes else 3)


def test_ordered_display_recovery_still_rejects_changed_source_meaning(monkeypatch):
    from core import bilingual_alignment as ba
    monkeypatch.setattr(sb, 'effective_target_language', lambda *_: 'zh')
    monkeypatch.setattr(ba, 'translate_source_partition', lambda *a: '打开它。删除它。')
    monkeypatch.setattr(sb, 'audit_duration_candidate', lambda *a:
                        {'equivalent': False, 'issues': ['Save became delete']})
    with pytest.raises(ValueError, match='changed source meaning'):
        sb.retranslate_unit('Open it. Save it.', '打开并存储。',
            {'unit_id': 'L1', 'target_speech_seconds': 3., 'max_speed_factor': 1.08,
             'target_max_words': 8, 'source_semantic_feedback': 'Bilingual display failure'}, '')


@pytest.mark.parametrize('recovery_passes', [True, False])
def test_exhausted_semantic_feedback_uses_source_only_recovery_and_reaudits(monkeypatch, recovery_passes):
    from core import bilingual_alignment as ba
    monkeypatch.setattr(sb, 'effective_target_language', lambda *_: 'zh')
    monkeypatch.setattr(sb, 'load_key', lambda key, default=None:
        True if key == 'translation.speech_budget.semantic_candidate_gate' else
        2 if key == 'translation.speech_budget.semantic_edit_retries' else
        '' if key == 'translation.speech_budget.candidate_provider' else default)
    primary, fresh = '加上审核要求的错误内容。', '打开它。保存它。'
    calls, ordered, aligned = [], [], []
    def reply(prompt, **kwargs):
        calls.append(prompt)
        result = {'1': primary} if kwargs['response_json'] else primary
        assert kwargs['valid_def'](result)['status'] == 'success'
        return result
    monkeypatch.setattr(sb, 'ask_gpt', reply)
    def translate(source, language, context, budget):
        ordered.append(dict(budget))
        assert source == 'Open it. Save it.'
        assert budget['max_speed_factor'] == 1.08
        assert budget['target_speech_seconds'] == 3.
        return fresh
    monkeypatch.setattr(ba, 'translate_source_partition', translate)
    checked = []
    def audit(source, anchor, candidate, context):
        checked.append(candidate)
        return {'equivalent': recovery_passes and candidate == fresh,
                'issues': [] if recovery_passes and candidate == fresh else ['Changed source facts']}
    monkeypatch.setattr(sb, 'audit_duration_candidate', audit)
    monkeypatch.setattr(ba, 'prepare_candidate_alignment', lambda unit, context: aligned.append(unit['target']))
    args = ('Open it. Save it.', '打开并保存。',
            {'unit_id': 'L1', 'target_speech_seconds': 3., 'max_speed_factor': 1.08,
             'target_max_words': 8, 'source_semantic_feedback': {'issues': ['Contradictory reviewer advice']},
             'meaning_anchor': 'An unaccepted draft'}, '')
    if recovery_passes:
        assert sb.retranslate_unit(*args) == fresh
        assert aligned == [fresh]
    else:
        with pytest.raises(ValueError, match='changed source meaning'):
            sb.retranslate_unit(*args)
        assert aligned == []
    assert len(calls) == 3
    assert 'Contradictory reviewer advice' not in calls[-1]
    assert 'source_semantic_feedback' not in ordered[0]
    assert 'meaning_anchor' not in ordered[0]
    assert checked[:3] == [primary, primary, primary]
    assert checked[3:] == [fresh] * (1 if recovery_passes else 3)


def test_whole_source_semantic_recovery_preserves_a_clause_before_partitioning(monkeypatch):
    from core import bilingual_alignment as ba
    monkeypatch.setattr(sb, 'effective_target_language', lambda *_: 'zh')
    monkeypatch.setattr(sb, 'load_key', lambda key, default=None:
        True if key == 'translation.speech_budget.semantic_candidate_gate' else
        2 if key == 'translation.speech_budget.semantic_edit_retries' else
        '' if key == 'translation.speech_budget.candidate_provider' else default)
    rejected, faithful = '我删掉它。', '保存它。'
    def reply(prompt, **kwargs):
        value = {'1': rejected} if kwargs['response_json'] else faithful
        if not kwargs['response_json']:
            assert 'REJECTED_REVIEW_ADVICE' not in prompt
            assert prompt.endswith('Save it.')
        assert kwargs['valid_def'](value)['status'] == 'success'
        return value
    monkeypatch.setattr(sb, 'ask_gpt', reply)
    monkeypatch.setattr(sb, 'audit_duration_candidate', lambda source, anchor, candidate, context:
        {'equivalent': candidate == faithful, 'issues': [] if candidate == faithful else ['Wrong action']})
    def partition(*_):
        raise AssertionError('A faithful, display-valid whole clause must not need source partition recovery')
    monkeypatch.setattr(ba, 'translate_source_partition', partition)
    aligned = []
    monkeypatch.setattr(ba, 'prepare_candidate_alignment', lambda unit, context: aligned.append(unit['target']))
    result = sb.retranslate_unit('Save it.', '删掉它。',
        {'unit_id': 'test', 'target_speech_seconds': 3., 'max_speed_factor': 1.08,
         'target_max_words': 8, 'source_semantic_feedback': {'issues': ['REJECTED_REVIEW_ADVICE']}}, '{}')
    assert result == faithful and aligned == [faithful]


def test_local_alternative_duration_candidate_keeps_source_validation_and_provenance(tmp_path, monkeypatch):
    from core import llm_provider
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sb,"effective_target_language",lambda *_:"en")
    monkeypatch.setattr(sb,"load_key",lambda key,default=None:
        "omlx_duration" if key=="translation.speech_budget.candidate_provider" else default)
    provider = llm_provider.LLMProviderConfig(name="omlx_duration", model="test-local",
                                             base_url="http://127.0.0.1:8000/v1", api_key="")
    monkeypatch.setattr(llm_provider,"get_llm_provider_config",lambda name:provider)
    calls = []
    def reply(prompt, **kwargs):
        calls.append(kwargs.get("provider_override"))
        if not kwargs.get("provider_override"):
            raise ValueError("Primary repeated measured-overlong phrasing")
        assert kwargs["valid_def"]({"1":"I pay."})["status"] == "error"
        assert kwargs["valid_def"]({"1":"You pay."})["status"] == "success"
        return {"1":"You pay."}
    monkeypatch.setattr(sb,"ask_gpt",reply)
    candidate = sb.retranslate_unit("你出钱。","You pay the money.",
        {"target_speech_seconds":1.,"target_max_words":2,"max_speed_factor":1.08},"",measured_seconds=1.4)
    assert candidate == "You pay."
    assert calls == [None,provider]
    record = json.loads((tmp_path/"output/log/translation_duration_candidate_provenance.json").read_text())["candidates"][0]
    assert record["model"] == "test-local"
    assert record["acceptance"] == "requires_measured_speech_and_semantic_audit"


def test_valid_primary_duration_candidate_does_not_use_another_model(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sb,"effective_target_language",lambda *_:"en")
    monkeypatch.setattr(sb,"load_key",lambda key,default=None:
        "omlx_duration" if key=="translation.speech_budget.candidate_provider" else default)
    calls = []
    def reply(prompt, **kwargs):
        calls.append(kwargs.get("provider_override"))
        assert kwargs["valid_def"]({"1":"You pay."})["status"] == "success"
        return {"1":"You pay."}
    monkeypatch.setattr(sb,"ask_gpt",reply)
    assert sb.retranslate_unit("你出钱。","You need to pay.",
        {"target_speech_seconds":1.,"target_max_words":2,"max_speed_factor":1.08},"",measured_seconds=1.4) == "You pay."
    assert calls == [None]
    assert not (tmp_path/"output/log/translation_duration_candidate_provenance.json").exists()


def test_duration_candidates_preserve_source_emphasis_without_false_goddamn_rejection(monkeypatch):
    monkeypatch.setattr(sb,"effective_target_language",lambda *_:"en")
    def reply(prompt, **kwargs):
        assert kwargs["valid_def"]({"1":"What surprise?"})["status"] == "error"
        assert kwargs["valid_def"]({"1":"What goddamn surprise?"})["status"] == "success"
        return {"1":"What goddamn surprise?"}
    monkeypatch.setattr(sb,"ask_gpt",reply)
    assert sb.retranslate_unit("什么他妈的叫惊喜？","What do you mean by a goddamn surprise?",
        {"target_speech_seconds":2.,"target_max_words":5,"max_speed_factor":1.08},"") == "What goddamn surprise?"


def test_candidate_search_rejects_longer_pronunciation_even_with_fewer_words(monkeypatch):
    monkeypatch.setattr(sb,"effective_target_language",lambda *_:"en")
    def reply(prompt,**kwargs):
        assert "音节" in prompt
        assert kwargs["valid_def"]({"1":"Understood."})["status"] == "error"
        assert kwargs["valid_def"]({"1":"Got it."})["status"] == "success"
        return {"1":"Got it."}
    monkeypatch.setattr(sb,"ask_gpt",reply)
    assert sb.retranslate_unit("明白了。","Understood.",
        {"target_speech_seconds":.8,"target_max_words":2,"max_speed_factor":1.08},"",measured_seconds=1.4) == "Got it."


def test_duration_edit_cannot_omit_or_change_a_numeric_value(monkeypatch):
    monkeypatch.setattr(sb,"effective_target_language",lambda *_:"en")
    def reply(prompt,**kwargs):
        assert kwargs["valid_def"]({"1":"I'll pay."})["status"] == "error"
        assert kwargs["valid_def"]({"1":"I'll pay two million."})["status"] == "error"
        assert kwargs["valid_def"]({"1":"I'll pay 1.8 million."})["status"] == "success"
        return {"1":"I'll pay 1.8 million."}
    monkeypatch.setattr(sb,"ask_gpt",reply)
    assert sb.retranslate_unit("我出一百八十万。","I'll pay 1.8 million.",
        {"target_speech_seconds":3.,"target_max_words":5,"max_speed_factor":1.08},"") == "I'll pay 1.8 million."


def test_duration_context_does_not_mix_output_schema_with_scene_role_metadata():
    unit={"unit_id":"L1","source":"你出钱。","parent_source":"你出钱。我剿匪。"}
    data={"topic":"A film dialogue about money.", "characters":[{"character_id":"C1","speaking_style":"long scene profile"}],
          "dialogue_units":[{"line_id":"L1","speech_act":"要求","character_id":"C1"}],
          "terms":[{"src":"钱","tgt":"money","note":"payment"},{"src":"剿匪","tgt":"fight bandits"}]}
    context=sb.duration_context(unit,data)
    assert '"line_id"' not in context and '"character_id"' not in context
    assert "long scene profile" not in context and "fight bandits" not in context
    assert "payment" in context and "我剿匪" in context
    assert json.loads(context)['topic'] == data['topic']


def test_small_duration_variance_keeps_meaning_and_retries_natural_audio_before_rewriting(tmp_path,monkeypatch):
    from core import tts_reference_plan
    from core.providers import asr_readback
    monkeypatch.chdir(tmp_path)
    plan=source_plan("明白了。")
    plan["units"][0].update(target="Got it.",status="translated")
    monkeypatch.setattr(ad,"load_plan",lambda:copy.deepcopy(plan))
    monkeypatch.setattr(ad,"materialize_references",lambda tasks:None)
    monkeypatch.setattr(tts_reference_plan,"prepare_reference_plan",lambda tasks:tasks)
    values={"translation.speech_budget.max_retries":1,"translation.speech_budget.max_speed_factor":1.08}
    monkeypatch.setattr(ad,"load_key",lambda key,default=None:values.get(key,default))
    seeds=[]
    def generate(tasks,indices,seed_offset=0,**kwargs):
        seeds.append(seed_offset)
        for idx in indices:
            output=ad._raw_path(tasks.loc[idx,"unit_id"])
            output.parent.mkdir(parents=True,exist_ok=True)
            Sine(300).to_audio_segment(duration=1350 if seed_offset==0 else 1200).export(output,format="wav")
    monkeypatch.setattr(ad,"_generate",generate)
    def verify(tasks):
        tasks=tasks.copy()
        tasks["asr_transcript"]=tasks.text
        tasks["asr_content_score"]=1.
        tasks["asr_status"]="ok"
        return tasks,SimpleNamespace(status="pass")
    monkeypatch.setattr(asr_readback,"verify_tasks_df",verify)
    def unexpected_edit(*a,**kw):raise AssertionError("A marginal overrun rewrote the source meaning before trying a complete take")
    monkeypatch.setattr(sb,"retranslate_unit",unexpected_edit)
    frame=pd.DataFrame([{"LineID":plan["units"][0]["unit_id"],"Source":"明白了。","Translation":"Got it."}])
    result=ad.translation_speech_preflight(frame,{})
    assert result.Translation.tolist()==["Got it."]
    assert seeds==[0,1]
    report=json.loads(ad.PREFLIGHT_REPORT.read_text())
    assert report["status"]=="pass" and not report["changes"]
    assert report["attempts"][0]["fits"] is False
    assert report["audio"][0]["speed_factor"]<=1.08


def test_semantically_changed_candidate_is_regenerated_before_audio_acceptance(monkeypatch):
    monkeypatch.setattr(sb,"effective_target_language",lambda *_:"en")
    outputs=iter([{"1":"You fund it."},{"1":"You pay."}])
    prompts=[]
    monkeypatch.setattr(sb,"ask_gpt",lambda prompt,**kw:(prompts.append(prompt) or next(outputs)))
    verdicts=iter([{"equivalent":False,"issues":["Meaning of payment was replaced by broader financing"]},
                   {"equivalent":True,"issues":[]}])
    monkeypatch.setattr(sb,"audit_duration_candidate",lambda *a,**kw:next(verdicts))
    assert sb.retranslate_unit("你出钱。","You pay the money.",
        {"target_speech_seconds":1.,"target_max_words":2,"max_speed_factor":1.08},"") == "You pay."
    assert "Semantic source comparison rejected" in prompts[1]


def test_natural_variants_survive_retries_and_resume_uses_the_correct_seed(tmp_path, monkeypatch):
    from pathlib import Path
    from core.all_tts_functions import mlx_router
    from core.providers import mlx_tts
    from core.providers.contracts import TTSRequest
    from core import tts_reference_plan
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ad, "load_key", lambda key, default=None: {
        "tts_method":"mlx_indextts2", "mlx_tts.backends.indextts2":{"seed":42}
    }.get(key, default))
    monkeypatch.setattr(mlx_router, "build_mlx_tts_request", lambda text, output, *_args, **_kw:
        TTSRequest(text=text, output_path=output, language="en", ref_audio="ref.wav",
                   metadata={"reference_plan":{"fingerprint":"reference"}, "backend":"indextts2"}))
    seeds = []
    class Backend:
        def __init__(self, cfg): self.cfg = cfg
        def synthesize_batch(self, requests):
            seeds.append(self.cfg["seed"])
            for request in requests:
                Sine(200 + self.cfg["seed"]).to_audio_segment(duration=800).export(request.output_path, format="wav")
                tts_reference_plan.write_conditioning_receipt(request)
    monkeypatch.setattr(mlx_tts, "IndexTTS2Backend", Backend)
    tasks = up.tasks_from_plan(translated_plan())
    ad._generate(tasks, [0])
    raw = ad._raw_path(tasks.iloc[0].unit_id)
    first = raw.read_bytes()
    receipt = json.loads(raw.with_suffix(".conditioning.json").read_text())
    original_variant = Path(receipt["selected_candidate"])
    ad._generate(tasks, [0], seed_offset=1)
    assert raw.read_bytes() != first
    assert original_variant.read_bytes() == first
    ad._generate(tasks, [0])
    assert seeds == [42,43]
    assert raw.read_bytes() == first


def test_model_profile_change_cannot_reuse_a_cached_natural_take(tmp_path, monkeypatch):
    from core.all_tts_functions import mlx_router
    from core.providers import mlx_tts
    from core.providers.contracts import TTSRequest
    from core import tts_reference_plan
    monkeypatch.chdir(tmp_path)
    cfg = {"seed": 42, "profile": "v25", "model": "/models/2.5"}
    monkeypatch.setattr(ad, "load_key", lambda key, default=None: {
        "tts_method": "mlx_indextts2", "mlx_tts.backends.indextts2": cfg
    }.get(key, default))
    monkeypatch.setattr(mlx_router, "build_mlx_tts_request", lambda text, output, *_args, **_kw:
        TTSRequest(text=text, output_path=output, language="zh", ref_audio="ref.wav",
                   metadata={"reference_plan": {"fingerprint": "reference"}, "backend": "indextts2"}))
    generated = []
    class Backend:
        def __init__(self, config): self.config = config
        def synthesize_batch(self, requests):
            for request in requests:
                generated.append((self.config["model"], request.output_path))
                Sine(200 + len(generated) * 30).to_audio_segment(duration=800).export(request.output_path, format="wav")
                tts_reference_plan.write_conditioning_receipt(request)
    monkeypatch.setattr(mlx_tts, "IndexTTS2Backend", Backend)
    tasks = up.tasks_from_plan(translated_plan())
    ad._generate(tasks, [0])
    uid = tasks.iloc[0].unit_id
    raw = ad._raw_path(uid)
    receipt = json.loads(raw.with_suffix('.conditioning.json').read_text())
    old = {"text": tasks.iloc[0].text, "generation_settings": receipt['generation_settings'],
           "natural_audio_path": receipt['selected_candidate'], "candidate_sha256": ad.hash_path(raw)}
    cfg.update(profile="standard", model="/models/2.0")
    ad._generate(tasks, [0], resume_takes={uid: old})
    assert [item[0] for item in generated] == ["/models/2.5", "/models/2.0"]
    assert generated[0][1] != generated[1][1]
    ad._generate(tasks, [0])
    assert len(generated) == 2


def test_plan_rejects_stale_source_and_bad_word_times():
    parents = pd.DataFrame([{"line_id": "L1", "text": "他出钱", "speaker": "S01"}])
    words = pd.DataFrame({"text": ["我", "出", "钱"], "start": [0., .4, .8], "end": [.4, .8, 1.2]})
    with pytest.raises(ValueError, match="Source/word identity"):
        up.build_plan(parents, words)
    words.at[0, "end"] = float("nan")
    with pytest.raises(ValueError, match="word timing"):
        up.build_plan(parents, words)


def test_resume_reuses_bound_natural_take_without_reusing_acceptance(tmp_path):
    from dataclasses import replace
    from core.providers.contracts import TTSRequest
    from core.tts_reference_plan import write_conditioning_receipt
    path = tmp_path / 'candidate.wav'
    Sine(240).to_audio_segment(duration=300).export(path, format='wav')
    settings = {'seed': 43, 'fit_duration': False, 'runtime_signature': 'same-runtime'}
    request = TTSRequest(text='You pay.', output_path=str(path), language='en',
        metadata={'backend': 'indextts2', 'reference_plan': {'fingerprint': 'bound-voice'},
                  'generation_settings': settings})
    write_conditioning_receipt(request)
    previous = {'text': request.text, 'natural_audio_path': str(path),
                'candidate_sha256': ad.hash_path(path), 'generation_settings': settings,
                'content_complete': False, 'fits': False}
    requested = replace(request, output_path=str(tmp_path/'new.wav'),
        metadata={**request.metadata, 'generation_settings': {**settings, 'seed': 42}})
    retained = ad._resume_natural_take(requested, previous)
    assert retained.output_path == str(path)
    assert retained.metadata['generation_settings']['seed'] == 43
    assert ad._resume_natural_take(replace(requested, text='I pay.'), previous) is None
    changed_voice = replace(requested, metadata={**requested.metadata,
                            'reference_plan': {'fingerprint': 'other-voice'}})
    assert ad._resume_natural_take(changed_voice, previous) is None
    changed_runtime = replace(requested, metadata={**requested.metadata,
        'generation_settings': {**settings, 'runtime_signature': 'different'}})
    assert ad._resume_natural_take(changed_runtime, previous) is None
    path.write_bytes(b'changed')
    assert ad._resume_natural_take(requested, previous) is None


@pytest.mark.parametrize("edit", ["text", "lines", "src_lines", "sub_times", "unit_id", "new_sub_times"])
def test_any_late_script_mapping_or_clock_mutation_is_blocked(edit):
    plan = translated_plan()
    tasks = up.tasks_from_plan(plan)
    tasks["new_sub_times"] = [[[u["start"], u["end"]]] for u in plan["units"]]
    up.validate_tasks(tasks, plan, generated=True)
    tasks.at[0, edit] = {"text": "Changed", "lines": ["Changed"], "src_lines": ["假的"],
                        "sub_times": [0., 10.], "unit_id": "OTHER", "new_sub_times": [[0.]]}[edit]
    with pytest.raises(ValueError):
        up.validate_tasks(tasks, plan, generated=True)


def test_word_count_estimate_does_not_replace_measured_speech_acceptance(monkeypatch):
    monkeypatch.setattr(sb, "effective_target_language", lambda *_: "en")
    prompts = []
    def reply(prompt, **kwargs):
        prompts.append(prompt)
        data = {"1": "You need to pay the money."}
        assert kwargs["valid_def"](data)["status"] == "success"
        return data
    monkeypatch.setattr(sb, "ask_gpt", reply)
    candidate = sb.retranslate_unit("你出钱。", "You pay the money.",
                                   {"target_max_words": 2, "target_speech_seconds": .8,
                                    "max_speed_factor": 1.08}, "")
    assert candidate == "You need to pay the money."
    assert "不是硬性词数限制" in prompts[0]
    assert "0.864 秒（包含上限）" in prompts[0]


def test_a_one_syllable_pronunciation_estimate_difference_can_reach_real_speech_measurement(monkeypatch):
    monkeypatch.setattr(sb,'effective_target_language',lambda *_:'en')
    monkeypatch.setattr(sb,'load_key',lambda key,default=None:default)
    def reply(prompt,**kwargs):
        candidate={'1':'You pay the fee.'}
        assert kwargs['valid_def'](candidate)['status']=='success'
        return candidate
    monkeypatch.setattr(sb,'ask_gpt',reply)
    candidate=sb.retranslate_unit('你出钱。','You need to pay.',
        {'target_speech_seconds':1.,'target_max_words':2,'max_speed_factor':1.08},'',measured_seconds=1.8)
    assert candidate=='You pay the fee.'


def test_syllable_estimates_cannot_veto_a_new_faithful_candidate_before_measurement(monkeypatch):
    monkeypatch.setattr(sb, 'effective_target_language', lambda *_: 'en')
    monkeypatch.setattr(sb, 'load_key', lambda key, default=None: default)
    def reply(prompt, **kwargs):
        candidate = {'1': 'You pay for it all.'}
        assert kwargs['valid_def'](candidate)['status'] == 'success'
        assert '不是验收阈值' in prompt
        return candidate
    monkeypatch.setattr(sb, 'ask_gpt', reply)
    candidate = sb.retranslate_unit('全由你出钱。', 'You need to pay the full bill.',
        {'target_speech_seconds': 1., 'target_max_words': 2, 'max_speed_factor': 1.08}, '', measured_seconds=3.)
    assert candidate == 'You pay for it all.'


def test_translation_budget_exposes_the_allowed_natural_duration(monkeypatch):
    values = {"translation.speech_budget.words_per_second": 2.7,
              "translation.speech_budget.max_speed_factor": 1.08}
    monkeypatch.setattr(sb, "load_key", lambda key, default=None: values.get(key, default))
    budget = sb.budget_for({"line_id": "L1", "duration": 1.})
    assert budget["target_speech_seconds"] == 1.
    assert budget["max_natural_speech_seconds"] == 1.08
    assert budget["max_speed_factor"] == 1.08


def test_spoken_translation_has_one_native_source_and_no_screen_layout_constraint():
    from core.local_translation import spoken_translation_prompt
    prompt = spoken_translation_prompt(["你出钱。"], "人物关系仅供消歧", "en",
        [{"target_speech_seconds":1.,"target_max_words":3,"max_speed_factor":1.08}],
        feedback={"rejected":"You need to pay the money."})
    assert prompt.count("【源文】") == prompt.count("【约束要求】") == 1
    assert prompt.index("【源文】") < prompt.index("【约束要求】")
    assert "CPS" not in prompt and "屏幕" not in prompt
    assert "1.080 秒（包含上限）" in prompt
    assert prompt.endswith("只输出译文JSON对象，不要有任何额外说明。")


@pytest.mark.parametrize("measured,rejected", [(1.08, False), (1.09, True)])
def test_an_unchanged_script_is_rejected_only_when_its_measured_duration_exceeds_the_cap(monkeypatch, measured, rejected):
    monkeypatch.setattr(sb, "effective_target_language", lambda *_: "en")
    monkeypatch.setattr(sb, "ask_gpt", lambda *a, **k: {"1": "You pay."})
    args = ("你出钱。", "You pay.", {"target_max_words": 1, "target_speech_seconds": 1.,
                                       "max_speed_factor": 1.08}, "")
    if rejected:
        with pytest.raises(ValueError, match="unchanged draft"):
            sb.retranslate_unit(*args, measured_seconds=measured)
    else:
        assert sb.retranslate_unit(*args, measured_seconds=measured) == "You pay."


def test_spoken_budget_counts_decimal_and_large_numbers_as_spoken():
    assert sb.spoken_word_count("1.8 million taels") == 5
    assert sb.spoken_word_count("1,800,000 taels") == 6


@pytest.mark.parametrize("source,expected", [("什么叫惊喜？", "What is a surprise?"),
                                           ("什么他妈的叫惊喜？", "What the hell is a surprise?")])
def test_unspoken_question_emphasis_guard_preserves_actual_source_profanity(source, expected):
    assert sb.normalize_question_emphasis(source, "What the hell is a surprise?")[0] == expected


def test_silent_take_fails_even_with_a_valid_wav_header(tmp_path):
    raw = tmp_path / "silent.wav"
    AudioSegment.silent(duration=1200).export(raw, format="wav")
    with pytest.raises(ValueError, match="silent"):
        ad.fit_take(raw, tmp_path / "fitted.wav", 1.2)


def test_overlong_take_is_never_truncated_to_pass(tmp_path):
    raw, out = tmp_path / "raw.wav", tmp_path / "out.wav"
    Sine(400).to_audio_segment(duration=1800).export(raw, format="wav")
    report = ad.fit_take(raw, out, .8)
    assert not report["fits"]
    assert len(AudioSegment.from_wav(out)) == 1800
    assert report["speed_factor"] == 1.


@pytest.mark.parametrize("milliseconds,fits", [(1080, True), (1090, False)])
def test_one_point_zero_eight_is_an_inclusive_speed_limit(tmp_path, milliseconds, fits):
    raw = tmp_path / "raw.wav"
    Sine(400).to_audio_segment(duration=milliseconds).export(raw, format="wav")
    report = ad.fit_take(raw, tmp_path / "out.wav", 1.)
    assert report["fits"] == fits
    assert report["speed_factor"] <= 1.08


def test_atempo_rounding_margin_cannot_reject_an_audio_that_fits_at_the_cap(tmp_path, monkeypatch):
    from core import audio_speed
    raw = tmp_path / "raw.wav"
    Sine(400).to_audio_segment(duration=1060).export(raw, format="wav")
    speeds = []
    def adjust(source, output, speed):
        speeds.append(speed)
        Sine(400).to_audio_segment(duration=1005 if speed < 1.08 else 998).export(output, format="wav")
    monkeypatch.setattr(audio_speed, "adjust_audio_speed", adjust)
    report = ad.fit_take(raw, tmp_path / "out.wav", 1.)
    assert speeds == [1.06, 1.08]
    assert report["fits"] and report["speed_factor"] == 1.08
    assert len(AudioSegment.from_wav(tmp_path / "out.wav")) == 1000


def test_near_unity_speed_still_fits_the_actual_window(tmp_path, monkeypatch):
    from core import audio_speed
    raw, out = tmp_path / "raw.wav", tmp_path / "out.wav"
    Sine(400).to_audio_segment(duration=1521).export(raw, format="wav")
    speeds = []
    original_adjust = audio_speed.adjust_audio_speed
    def adjust(source, output, speed):
        speeds.append(speed)
        original_adjust(source, output, speed)
    monkeypatch.setattr(audio_speed, "adjust_audio_speed", adjust)
    report = ad.fit_take(raw, out, 1.52)
    assert report["fits"] and report["speed_factor"] <= 1.08
    assert speeds and 1 < speeds[0] < 1.001
    assert len(AudioSegment.from_wav(out)) == 1520


def test_failed_fitting_keeps_the_complete_current_readback_witness(tmp_path, monkeypatch):
    from pathlib import Path
    from core import audio_speed
    raw, out = tmp_path / "raw.wav", tmp_path / "out.wav"
    Sine(400).to_audio_segment(duration=1521).export(raw, format="wav")
    # Simulate a backend whose processed take still exceeds the window.
    # A pre-existing output must not survive as the current readback witness.
    Sine(800).to_audio_segment(duration=300).export(out, format="wav")
    complete = AudioSegment.from_file(raw).set_channels(1).set_frame_rate(24000)
    monkeypatch.setattr(audio_speed, "adjust_audio_speed", lambda *args: None)
    original_load = AudioSegment.from_file
    def load(path, *args, **kwargs):
        return complete if Path(path) == out else original_load(path, *args, **kwargs)
    monkeypatch.setattr(AudioSegment, "from_file", load)
    report = ad.fit_take(raw, out, 1.52)
    assert not report["fits"]
    assert report["timing_issue"] == "processed_duration_over_budget"
    assert report["speed_factor"] <= 1.08
    actual = original_load(out)
    assert actual.raw_data == complete.raw_data
    assert len(actual) > round(1.52 * 1000) and actual[1520:].rms > 0


def test_encoder_overhead_guides_a_smaller_candidate_without_faking_natural_duration(tmp_path,monkeypatch):
    from core import audio_speed
    from core.homura_translation import syllable_targets
    raw=tmp_path/'raw.wav'
    Sine(400).to_audio_segment(duration=1070).export(raw,format='wav')
    monkeypatch.setattr(audio_speed,'adjust_audio_speed',lambda source,output,speed:
        Sine(400).to_audio_segment(duration=1020).export(output,format='wav'))
    report=ad.fit_take(raw,tmp_path/'out.wav',1.)
    assert not report['fits'] and report['timing_issue']=='processed_duration_over_budget'
    assert report['natural_seconds']==1.07 and report['required_speed']==1.07
    assert report['processed_seconds']==1.02 and report['speed_factor']<=1.08
    assert 0<report['calibrated_max_natural_seconds']<report['natural_seconds']
    assert len(AudioSegment.from_wav(tmp_path/'out.wav'))==1020
    current='You need to pay.'
    budget={'max_natural_speech_seconds':report['calibrated_max_natural_seconds'],
            'duration_witness':{'content_complete':True,'text':current,'natural_seconds':1.07,
                'natural_audio_sha256':'natural','conditioning_signature':'voice','readback_signature':'readback'}}
    assert syllable_targets(current,budget)


def test_natural_short_take_pads_clock_without_expanding_dialogue(tmp_path):
    raw, out = tmp_path / "raw.wav", tmp_path / "out.wav"
    Sine(400).to_audio_segment(duration=700).export(raw, format="wav")
    report = ad.fit_take(raw, out, 1.2)
    assert report["fits"]
    assert report["speed_factor"] == 1.
    assert report["trailing_silence_seconds"] > .3
    assert len(AudioSegment.from_wav(out)) == 1200


def test_readback_empty_low_score_and_lost_suffix_all_block_acceptance():
    rows = pd.DataFrame([
        {"text": "You pay.", "asr_transcript": "", "asr_status": "ok", "asr_content_score": 0.},
        {"text": "I'll fight the bandits.", "asr_transcript": "I'll", "asr_status": "ok", "asr_content_score": .3},
        {"text": "You pay.", "asr_transcript": "You pay.", "asr_status": "ok", "asr_content_score": 1.},
    ])
    assert set(ad._content_issues(rows)) == {0, 1}


def test_high_character_similarity_cannot_waive_a_wrong_spoken_amount():
    rows = pd.DataFrame([
        {"text": "You'll get 1.8 million.", "asr_transcript": "You'll get one million eight hundred thousand.", "asr_status": "ok", "asr_content_score": 1.},
        {"text": "You'll get 1.8 million.", "asr_transcript": "You'll get 1.9 million.", "asr_status": "ok", "asr_content_score": .99},
    ])
    assert ad._content_issues(rows) == {1: ["spoken_numeric_value_mismatch"]}


def test_high_similarity_cannot_waive_a_changed_spoken_owner():
    rows=pd.DataFrame([{'origin':'修我的车。','text':'Please fix my car.',
                        'asr_transcript':'Please fix your car.','asr_status':'ok','asr_content_score':.99}])
    assert ad._content_issues(rows)=={0:['spoken_source_fact_mismatch']}


def test_high_similarity_cannot_accept_a_longer_replacement_for_a_name_tail():
    rows=pd.DataFrame([{'text':'The witness identified your man, Hu Wan.',
                        'asr_transcript':'The witness identified your man. Who am I?',
                        'asr_status':'ok','asr_content_score':.97}])
    assert ad._content_issues(rows)=={0:['incomplete_speech']}


def test_a_new_duration_candidate_has_its_own_complete_speech_retakes(tmp_path,monkeypatch):
    from core import tts_reference_plan
    from core.providers import asr_readback
    monkeypatch.chdir(tmp_path)
    plan=source_plan('你出钱。');up.save_plan(plan)
    frame=pd.DataFrame([{'LineID':plan['units'][0]['unit_id'],'Source':'你出钱。','Translation':'You pay the money.'}])
    monkeypatch.setattr(ad,'materialize_references',lambda *_:None)
    monkeypatch.setattr(tts_reference_plan,'prepare_reference_plan',lambda df:df.assign(reference_plan_fingerprint='fixed'))
    monkeypatch.setattr(sb,'retranslate_unit',lambda *a,**k:'You pay.')
    generated=[]
    def generate(df,indices,**kwargs):
        for idx in indices:
            text=df.loc[idx,'text'];generated.append(text)
            p=ad._raw_path(df.loc[idx,'unit_id']);p.parent.mkdir(parents=True,exist_ok=True)
            Sine(400).to_audio_segment(duration=1900 if text=='You pay the money.' else 600).export(p,format='wav')
    monkeypatch.setattr(ad,'_generate',generate)
    def verify(df):
        df=df.assign(asr_status='ok',asr_content_score=1.,asr_transcript=df.text)
        if generated[-1]=='You pay.' and generated.count('You pay.')<3:
            df['asr_transcript']='You';df['asr_content_score']=.3
        return df,SimpleNamespace(status='ok',checked=len(df))
    monkeypatch.setattr(asr_readback,'verify_tasks_df',verify)
    accepted=ad.translation_speech_preflight(frame,{})
    assert list(accepted.Translation)==['You pay.']
    assert generated==['You pay the money.','You pay.','You pay.','You pay.']
    report=json.loads(ad.PREFLIGHT_REPORT.read_text())
    assert report['status']=='pass' and len(report['changes'])==1


def test_irreducible_short_phrase_can_use_a_later_complete_natural_take(tmp_path, monkeypatch):
    from core import tts_reference_plan
    from core.providers import asr_readback
    monkeypatch.chdir(tmp_path)
    plan = source_plan('好吧。')
    up.save_plan(plan)
    frame = pd.DataFrame([{'LineID': plan['units'][0]['unit_id'], 'Source': '好吧。', 'Translation': 'Okay.'}])
    monkeypatch.setattr(ad, 'materialize_references', lambda *_: None)
    monkeypatch.setattr(tts_reference_plan, 'prepare_reference_plan', lambda df: df.assign(reference_plan_fingerprint='fixed'))
    generated, edits = [], []
    def generate(df, indices, **kwargs):
        for idx in indices:
            generated.append(df.loc[idx, 'text'])
            path = ad._raw_path(df.loc[idx, 'unit_id'])
            path.parent.mkdir(parents=True, exist_ok=True)
            Sine(400).to_audio_segment(duration=780 if len(generated) == 4 else 900).export(path, format='wav')
    def no_shorter(*args, **kwargs):
        edits.append(args)
        raise ValueError('No shorter faithful wording exists')
    monkeypatch.setattr(ad, '_generate', generate)
    monkeypatch.setattr(sb, 'retranslate_unit', no_shorter)
    monkeypatch.setattr(asr_readback, 'verify_tasks_df', lambda df:
        (df.assign(asr_status='ok', asr_content_score=1., asr_transcript=df.text), SimpleNamespace(status='ok', checked=len(df))))
    accepted = ad.translation_speech_preflight(frame, {})
    report = json.loads(ad.PREFLIGHT_REPORT.read_text())
    assert list(accepted.Translation) == ['Okay.'] and generated == ['Okay.'] * 4
    assert len(edits) == 1 and not report['changes']
    assert report['status'] == 'pass' and report['audio'][0]['speed_factor'] <= 1.08


@pytest.mark.parametrize("policy_changed", [False, True])
@pytest.mark.parametrize("content_also_failed", [False, True])
def test_duration_feedback_is_checkpointed_before_acceptance_and_reused_on_resume(tmp_path, monkeypatch, policy_changed, content_also_failed):
    from core import tts_reference_plan
    from core.providers import asr_readback
    monkeypatch.chdir(tmp_path)
    plan = source_plan("你出钱。")
    up.save_plan(plan)
    initial = pd.DataFrame([{"LineID": plan['units'][0]['unit_id'], "Source": "你出钱。", "Translation": "You pay the money."}])
    monkeypatch.setattr(ad, 'materialize_references', lambda *_: None)
    monkeypatch.setattr(tts_reference_plan, 'prepare_reference_plan', lambda df: df.assign(reference_plan_fingerprint='fixed'))
    readbacks = []
    def verify(df):
        df = df.copy()
        df['asr_status'] = 'ok'; df['asr_content_score'] = 1.; df['asr_transcript'] = df.text
        readbacks.append(list(df.text))
        if content_also_failed and len(readbacks) == 1:
            for idx in df.index[df.text == 'You pay the money.']:
                df.at[idx, 'asr_transcript'] = 'You pay'
                df.at[idx, 'asr_content_score'] = .3
        return df, SimpleNamespace(status='ok', checked=len(df))
    monkeypatch.setattr(asr_readback, 'verify_tasks_df', verify)
    def compact(source, current, budget, context, **kwargs):
        assert budget['duration_witness']['content_complete'] is True
        assert budget['duration_witness']['readback_signature']
        assert len(readbacks) == (2 if content_also_failed else 1)
        return 'You pay.'
    monkeypatch.setattr(sb, 'retranslate_unit', compact)
    generated = []
    def generate(df, indices, **kwargs):
        for idx in indices:
            text = df.loc[idx,'text']; generated.append(text)
            path = ad._raw_path(df.loc[idx,'unit_id']); path.parent.mkdir(parents=True, exist_ok=True)
            Sine(400).to_audio_segment(duration=1900 if text == 'You pay the money.' else 600).export(path, format='wav')
    monkeypatch.setattr(ad, '_generate', generate)
    accepted = ad.translation_speech_preflight(initial, {})
    assert list(accepted.Translation) == ['You pay.']
    assert generated == ['You pay the money.'] * (2 if content_also_failed else 1) + ['You pay.']
    ad.verify_preflight_identity(accepted)
    if policy_changed:
        monkeypatch.setattr(ad, 'synthesis_contract_signature', lambda: 'changed fit policy')
        monkeypatch.setattr(ad, 'candidate_policy_signature', lambda: 'changed candidate search policy')
        with pytest.raises(ValueError, match='configuration changed'):
            ad.verify_preflight_identity(accepted)
    generated.clear()
    resumed = ad.translation_speech_preflight(initial, {})
    assert list(resumed.Translation) == ['You pay.']
    assert generated == ['You pay.']


def test_partial_overlong_speech_never_changes_translation_or_calibrates_a_model(tmp_path, monkeypatch):
    from core import tts_reference_plan
    from core.providers import asr_readback
    monkeypatch.chdir(tmp_path)
    plan=source_plan('你出钱。');up.save_plan(plan)
    frame=pd.DataFrame([{'LineID':plan['units'][0]['unit_id'],'Source':'你出钱。','Translation':'You pay the money.'}])
    monkeypatch.setattr(ad,'materialize_references',lambda *_:None)
    monkeypatch.setattr(tts_reference_plan,'prepare_reference_plan',lambda df:df.assign(reference_plan_fingerprint='fixed'))
    monkeypatch.setattr(asr_readback,'verify_tasks_df',lambda df:
        (df.assign(asr_status='ok',asr_content_score=.3,asr_transcript='You pay'),SimpleNamespace(status='ok',checked=len(df))))
    def unexpected_edit(*args,**kwargs):
        pytest.fail('Partial speech must regenerate unchanged text before any translation duration feedback')
    monkeypatch.setattr(sb,'retranslate_unit',unexpected_edit)
    texts=[]
    def generate(df,indices,**kwargs):
        for idx in indices:
            texts.append(df.loc[idx,'text'])
            path=ad._raw_path(df.loc[idx,'unit_id']);path.parent.mkdir(parents=True,exist_ok=True)
            Sine(400).to_audio_segment(duration=1900).export(path,format='wav')
    monkeypatch.setattr(ad,'_generate',generate)
    with pytest.raises(RuntimeError,match='needs review'):
        ad.translation_speech_preflight(frame,{})
    report=json.loads(ad.PREFLIGHT_REPORT.read_text())
    assert report['status']=='needs_review' and not report['changes']
    assert len(texts)==3 and set(texts)=={'You pay the money.'}
    assert all(take['content_complete'] is False for take in report['attempts'])
    assert report['issues'][0]['next_action']=='requires_speech_review'


def test_primary_source_semantics_are_checked_before_speech_generation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    plan=source_plan('你出钱。');up.save_plan(plan)
    frame=pd.DataFrame([{'LineID':plan['units'][0]['unit_id'],'Source':'你出钱。','Translation':"I'll pay."}])
    monkeypatch.setattr(sb,'audit_duration_candidate',lambda *args:{'equivalent':False,'issues':['Changed payer']})
    monkeypatch.setattr(sb,'retranslate_unit',lambda *args,**kwargs: (_ for _ in ()).throw(ValueError('No safe repair')))
    monkeypatch.setattr(ad,'_generate',lambda *a,**k:pytest.fail('Invalid source meaning must not reach synthesis'))
    with pytest.raises(RuntimeError,match='source semantic gate'):
        ad.translation_speech_preflight(frame,{})
    report=json.loads(ad.PREFLIGHT_REPORT.read_text())
    assert report['status']=='needs_review' and report['stage']=='source_semantics_before_synthesis'


def test_audit_outage_preserves_draft_checkpoint_without_retranslation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    plan = source_plan(); up.save_plan(plan)
    frame = pd.DataFrame([{'LineID': u['unit_id'], 'Source': u['source'], 'Translation': text}
        for u, text in zip(plan['units'], ["I'll pay.", "I'll fight the bandits."])])
    edits, reviews = [], []
    def audit(source, anchor, text, context):
        reviews.append(text)
        if source == '我剿匪！':
            raise TimeoutError('Review server unavailable')
        return {'equivalent': text == 'You pay.', 'issues': [] if text == 'You pay.' else ['Payer changed']}
    monkeypatch.setattr(sb, 'audit_duration_candidate', audit)
    def edit(source, current, budget, context):
        assert source == '你出钱。', 'Audit outage must not cause translation edits'
        edits.append(source)
        return 'You pay.'
    monkeypatch.setattr(sb, 'retranslate_unit', edit)
    monkeypatch.setattr(ad, '_generate', lambda *a, **k: pytest.fail('Unknown audit cannot reach speech'))
    with pytest.raises(RuntimeError, match='source semantic gate'):
        ad.translation_speech_preflight(frame, {})
    assert edits == ['你出钱。']
    state = json.loads(ad.CANDIDATE_STATE.read_text())
    assert state['status'] == 'unaccepted_drafts'
    assert [u['target'] for u in state['candidates']] == ['You pay.', "I'll fight the bandits."]
    report = json.loads(ad.PREFLIGHT_REPORT.read_text())
    assert report['issues'][0]['verdict']['failure_kind'] == 'audit_unavailable'
    reviews.clear()
    monkeypatch.setattr(sb, 'audit_duration_candidate', lambda src, anchor, text, ctx:
        reviews.append(text) or {'equivalent': True, 'issues': []})
    monkeypatch.setattr(ad, 'materialize_references', lambda *_:
        (_ for _ in ()).throw(RuntimeError('Reached speech only after fresh reviews')))
    with pytest.raises(RuntimeError, match='fresh reviews'):
        ad.translation_speech_preflight(frame, {})
    assert reviews == ['You pay.', "I'll fight the bandits."]
    assert edits == ['你出钱。']


def test_source_semantic_repair_is_bounded_and_reaudited_before_any_speech(tmp_path, monkeypatch):
    from core import tts_reference_plan
    from core.providers import asr_readback
    monkeypatch.chdir(tmp_path)
    plan = source_plan('你出钱。'); up.save_plan(plan)
    frame = pd.DataFrame([{'LineID': plan['units'][0]['unit_id'], 'Source': '你出钱。', 'Translation': "I'll pay."}])
    old_load = ad.load_key
    monkeypatch.setattr(ad, 'load_key', lambda key, default=None:
        True if key == 'translation.speech_budget.semantic_candidate_gate' else old_load(key, default))
    checks = []
    monkeypatch.setattr(sb, 'audit_duration_candidate', lambda src, anchor, text, ctx:
        checks.append(text) or {'equivalent': text == 'You pay.', 'issues': [] if text == 'You pay.' else ['Changed payer']})
    def revise(source, current, budget, context, **kwargs):
        assert 'duration_witness' not in budget
        assert budget['source_semantic_feedback']['equivalent'] is False
        return 'You pay.'
    monkeypatch.setattr(sb, 'retranslate_unit', revise)
    monkeypatch.setattr(ad, 'materialize_references', lambda *_: None)
    monkeypatch.setattr(tts_reference_plan, 'prepare_reference_plan', lambda df: df.assign(reference_plan_fingerprint='fixed'))
    def generate(df, indices, **kwargs):
        assert checks[-1] == 'You pay.'
        for idx in indices:
            assert df.loc[idx, 'text'] == 'You pay.'
            path = ad._raw_path(df.loc[idx, 'unit_id']); path.parent.mkdir(parents=True, exist_ok=True)
            Sine(400).to_audio_segment(duration=600).export(path, format='wav')
    monkeypatch.setattr(ad, '_generate', generate)
    monkeypatch.setattr(asr_readback, 'verify_tasks_df', lambda df:
        (df.assign(asr_status='ok', asr_content_score=1., asr_transcript=df.text), SimpleNamespace(status='ok', checked=len(df))))
    accepted = ad.translation_speech_preflight(frame, {})
    assert list(accepted.Translation) == ['You pay.']
    report = json.loads(ad.PREFLIGHT_REPORT.read_text())
    assert report['changes'][0]['reason'] == 'source_semantics_before_synthesis'
    assert report['status'] == 'pass'


@pytest.mark.parametrize('recovery_passes', [True, False])
def test_unchanged_words_with_recovered_display_are_reaudited(tmp_path, monkeypatch, recovery_passes):
    from core import bilingual_alignment as ba
    monkeypatch.chdir(tmp_path)
    plan = source_plan('你出钱。'); up.save_plan(plan)
    frame = pd.DataFrame([{'LineID': plan['units'][0]['unit_id'], 'Source': '你出钱。', 'Translation': 'You pay.'}])
    monkeypatch.setattr(sb, 'audit_duration_candidate', lambda *a: {'equivalent': True, 'issues': []})
    recovered, checks, edits = False, [], []
    def align(*args, **kwargs):
        checks.append(recovered)
        if not recovered or not recovery_passes:
            raise ValueError('Display needs correspondence recovery')
        return []
    def repair(source, current, budget, context):
        nonlocal recovered
        recovered = True
        edits.append(current)
        return current
    monkeypatch.setattr(ba, 'prepare_candidate_alignment', align)
    monkeypatch.setattr(sb, 'retranslate_unit', repair)
    monkeypatch.setattr(ad, 'materialize_references', lambda *_:
        (_ for _ in ()).throw(RuntimeError('Reached speech after recovered display')))
    error = 'Reached speech' if recovery_passes else 'source semantic gate'
    with pytest.raises(RuntimeError, match=error):
        ad.translation_speech_preflight(frame, {})
    assert checks[0] is False and checks[1] is True
    assert 1 <= len(edits) <= 2
    state = json.loads(ad.CANDIDATE_STATE.read_text())
    assert state['changes'] == []


@pytest.mark.parametrize("second_take_fits", [True, False])
def test_reviewed_script_is_preserved_but_cannot_bypass_measured_acceptance(tmp_path, monkeypatch, second_take_fits):
    from core import tts_reference_plan
    from core.providers import asr_readback
    monkeypatch.chdir(tmp_path)
    plan = source_plan("你出钱。")
    up.save_plan(plan)
    frame = pd.DataFrame([{"LineID": plan['units'][0]['unit_id'], "Source": "你出钱。", "Translation": "You pay the money."}])
    original_load_key = ad.load_key
    def config(key, default=None):
        if key == 'translation_exact_replacements':
            return {"你出钱。": "You pay the money."}
        if key == 'translation.speech_budget.max_retries':
            return 1
        return original_load_key(key, default)
    monkeypatch.setattr(ad, 'load_key', config)
    monkeypatch.setattr(ad, 'materialize_references', lambda *_: None)
    monkeypatch.setattr(tts_reference_plan, 'prepare_reference_plan', lambda df: df.assign(reference_plan_fingerprint='fixed'))
    def verify(df):
        return df.assign(asr_status='ok', asr_content_score=1., asr_transcript=df.text), SimpleNamespace(status='ok', checked=len(df))
    monkeypatch.setattr(asr_readback, 'verify_tasks_df', verify)
    def rewrite(*args, **kwargs):
        pytest.fail('A reviewed script must not be replaced by a new model draft')
    monkeypatch.setattr(sb, 'retranslate_unit', rewrite)
    generated = []
    def generate(df, indices, **kwargs):
        for idx in indices:
            generated.append(df.loc[idx, 'text'])
            path = ad._raw_path(df.loc[idx, 'unit_id']); path.parent.mkdir(parents=True, exist_ok=True)
            duration = 600 if len(generated) > 1 and second_take_fits else 1900
            Sine(400).to_audio_segment(duration=duration).export(path, format='wav')
    monkeypatch.setattr(ad, '_generate', generate)
    if second_take_fits:
        accepted = ad.translation_speech_preflight(frame, {})
        assert list(accepted.Translation) == ['You pay the money.']
        ad.verify_preflight_identity(accepted)
    else:
        with pytest.raises(RuntimeError, match='needs review'):
            ad.translation_speech_preflight(frame, {})
    assert generated == ['You pay the money.', 'You pay the money.']


@pytest.mark.parametrize("legacy_checkpoint", [False, True])
def test_new_editorial_input_preserves_other_drafts_without_reusing_its_previous_words(tmp_path, monkeypatch, legacy_checkpoint):
    from core import tts_reference_plan
    from core.providers import asr_readback
    monkeypatch.chdir(tmp_path)
    plan = source_plan("你出钱。我办事。")
    up.save_plan(plan)
    initial = pd.DataFrame([{"LineID": u['unit_id'], "Source": u['source'], "Translation": text}
                            for u, text in zip(plan['units'], ['You pay the money.', "I'll do the work."])])
    monkeypatch.setattr(ad, 'materialize_references', lambda *_: None)
    monkeypatch.setattr(tts_reference_plan, 'prepare_reference_plan', lambda df: df.assign(reference_plan_fingerprint='fixed'))
    monkeypatch.setattr(asr_readback, 'verify_tasks_df', lambda df: (
        df.assign(asr_status='ok', asr_content_score=1., asr_transcript=df.text), SimpleNamespace(status='ok', checked=len(df))))
    monkeypatch.setattr(sb, 'retranslate_unit', lambda source, *a, **k: 'You pay.' if source == '你出钱。' else "I'll do it.")
    generated = []
    def generate(df, indices, **kwargs):
        for idx in indices:
            text = df.loc[idx, 'text']; generated.append(text)
            path = ad._raw_path(df.loc[idx, 'unit_id']); path.parent.mkdir(parents=True, exist_ok=True)
            duration = 1900 if text in ['You pay the money.', "I'll do the work."] else 600
            Sine(400).to_audio_segment(duration=duration).export(path, format='wav')
    monkeypatch.setattr(ad, '_generate', generate)
    ad.translation_speech_preflight(initial, {})
    if legacy_checkpoint:
        saved = json.loads(ad.CANDIDATE_STATE.read_text())
        saved.pop('input_rows')
        ad.CANDIDATE_STATE.write_text(json.dumps(saved))
    changed = initial.copy(); changed.at[1, 'Translation'] = 'I handle it.'
    generated.clear()
    accepted = ad.translation_speech_preflight(changed, {})
    assert list(accepted.Translation) == ['You pay.', 'I handle it.']
    assert generated == ['You pay.', 'I handle it.']
    ad.verify_preflight_identity(accepted)


def test_bilingual_visual_limit_and_audio_word_clock_share_unit_identity():
    plan = translated_plan()
    unit = plan["units"][1]
    unit["end"] = 6.; unit["duration"] = 4.8
    unit['source'] = '我剿匪！他们逃跑。'
    unit["target"] = "I fight the bandits. They flee."
    words = unit["target"].split()
    unit["target_words"] = [{"word": word, "start": i * .32, "end": (i + 1) * .32} for i, word in enumerate(words)]
    plan["units"][0]["target_words"] = [{"word": "You", "start": 0., "end": .2}, {"word": "pay.", "start": .2, "end": .5}]
    paired = {'units': [{**unit, 'status': 'pass', 'pairs': [
        {'source': '我剿匪！', 'target': 'I fight the bandits.', 'semantic_review': {'equivalent': True, 'issues': []}},
        {'source': '他们逃跑。', 'target': 'They flee.', 'semantic_review': {'equivalent': True, 'issues': []}}]}]}
    frame, report = layout_plan(plan, chars=24, source_chars=12, bilingual_alignment=paired)
    assert len(frame) > len(plan["units"])
    for row in frame.itertuples():
        assert len(row.Translation) <= 24 and len(row.Source) <= 12
        assert len((row.Source + "\n" + row.Translation).splitlines()) <= 2
        assert row.UnitID in {u["unit_id"] for u in plan["units"]}
    cuts = [row.timestamp[0] for row in frame.itertuples() if row.UnitID == unit["unit_id"]][1:]
    assert all(any(abs(at - (unit["start"] + w["start"])) < .001 for w in unit["target_words"]) for at in cuts)


@pytest.mark.parametrize('changed', ['fitted', 'natural'])
def test_audio_hash_change_invalidates_translation_acceptance(tmp_path, monkeypatch, changed):
    monkeypatch.chdir(tmp_path)
    frame = pd.DataFrame([{"LineID": "L1.01", "Source": "好", "Translation": "Fine."}])
    wav = tmp_path / "speech.wav"; wav.write_bytes(b"first audio")
    raw = tmp_path / "natural.wav"; raw.write_bytes(b"natural audio")
    plan = source_plan("好")
    plan["units"][0]["unit_id"] = "L1.01"
    up.save_plan(plan)
    report = {"version": 2, "status": "pass", "translation_signature": up.signature(frame.to_dict("records")),
              "synthesis_contract_signature": ad.synthesis_contract_signature(),
              "unit_clock_signature": ad.unit_clock_signature(plan),
              "audio": [{"unit_id": "L1.01", "path": str(wav), "sha256": ad.hash_path(wav),
                         "raw_path": str(raw), "raw_sha256": ad.hash_path(raw)}]}
    ad.PREFLIGHT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    ad.PREFLIGHT_REPORT.write_text(json.dumps(report))
    ad.verify_preflight_identity(frame)
    (wav if changed == 'fitted' else raw).write_bytes(b"different audio")
    with pytest.raises(ValueError, match="speech changed"):
        ad.verify_preflight_identity(frame)


@pytest.mark.parametrize("changed", ["clock", "model"])
def test_preflight_acceptance_cannot_outlive_its_clock_or_synthesis_policy(tmp_path, monkeypatch, changed):
    monkeypatch.chdir(tmp_path)
    plan = translated_plan()
    up.save_plan(plan)
    frame = pd.DataFrame([{"LineID": u["unit_id"], "Source": u["source"], "Translation": u["target"]} for u in plan["units"]])
    report = {"version": 2, "status": "pass", "translation_signature": up.signature(frame.to_dict("records")),
              "synthesis_contract_signature": ad.synthesis_contract_signature(),
              "unit_clock_signature": ad.unit_clock_signature(plan), "audio": []}
    ad.PREFLIGHT_REPORT.write_text(json.dumps(report))
    ad.verify_preflight_identity(frame)
    if changed == "clock":
        plan["units"][0]["end"] -= .1
        up.save_plan(plan)
    else:
        monkeypatch.setattr(ad, "load_key", lambda *_: "new model")
    with pytest.raises(ValueError, match="clock changed|configuration changed"):
        ad.verify_preflight_identity(frame)


def test_legacy_mismatched_source_arrays_fail_instead_of_guessing(tmp_path, monkeypatch):
    from core import step11_merge_full_audio as merge
    monkeypatch.chdir(tmp_path)
    tasks = pd.DataFrame([{"number": 1, "lines": ["Hello.", "Goodbye."],
                           "src_lines": ["你好，再见。"], "origin": "你好，再见。",
                           "new_sub_times": [[0., 1.], [1., 2.]]}])
    path = tmp_path / "tasks.xlsx"; tasks.to_excel(path, index=False)
    with pytest.raises(ValueError, match="cardinality mismatch"):
        merge.load_and_flatten_data(path)
