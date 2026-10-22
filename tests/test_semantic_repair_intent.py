import json

import pytest

from core.local_translation import spoken_translation_prompt, source_only_translation_prompt
from core.duration_candidate_audit import _imperative_comparison_source
from core.speech_budget import budget_for


def repair_budget():
    return {**budget_for({'duration': 3 / 1.08, 'line_id': 'test'}),
            'max_natural_speech_seconds': 3}


def test_unmeasured_semantic_repair_does_not_claim_overrun():
    prompt = spoken_translation_prompt(['A named item is missing.'], '{}', 'zh',
        [repair_budget()], feedback={
            'current_draft': '上一稿。', 'natural_seconds': None,
            'source_semantic_feedback': {'equivalent': False, 'issues': ['missing item']}})
    assert '【语义修正】' in prompt
    assert '【实测时长反馈】' not in prompt
    assert '实测录音已严重超时' not in prompt


def test_complete_in_budget_take_is_not_instructed_to_shrink():
    prompt = spoken_translation_prompt(['Keep the time condition.'], '{}', 'zh',
        [repair_budget()], feedback={
            'current_draft': '上一稿。', 'natural_seconds': 2.9,
            'source_semantic_feedback': {'equivalent': False, 'issues': ['missing condition']}})
    assert '【实测时长反馈】' not in prompt
    assert '恢复遗漏的事实' in prompt


def test_measured_overrun_still_demands_complete_compaction():
    prompt = spoken_translation_prompt(['Complete meaning.'], '{}', 'zh',
        [repair_budget()], feedback={
            'current_draft': '上一稿。', 'natural_seconds': 3.5})
    assert '【实测时长反馈】' in prompt and '3.500' in prompt and '3.000' in prompt
    assert '不能删掉源文事实' in prompt


def test_source_only_recovery_cannot_copy_failed_targets_or_repeat_current_source():
    source = 'one important thing for us to browse the web,'
    context = json.dumps({'topic': 'software setup', 'failed_target': 'TARGET_SENTINEL',
        'source_continuation': {'current_source_quote': source,
                                'preceding_source_quote': 'You can turn on'}})
    prompt = source_only_translation_prompt(source, context, 'zh', repair_budget())
    assert prompt.endswith(source) and prompt.count(source) == 1
    assert 'TARGET_SENTINEL' not in prompt
    assert '名词短语' in prompt and '【配音长度目标】' not in prompt


def test_source_only_nominal_recovery_binds_modifier_to_its_source_head():
    source = 'one important thing for us to browse the web,'
    prompt = source_only_translation_prompt(source, '{}', 'zh', repair_budget())
    assert '中心词是thing' in prompt
    assert 'for us to browse the web是它的目的或能力修饰' in prompt
    assert '保留为一个完整名词短语，不改成陈述句' in prompt
    assert '只输出译文' in prompt and 'JSON' not in prompt


def test_source_only_finite_sentence_is_not_forced_into_a_noun_phrase():
    prompt = source_only_translation_prompt('I found the option.', '{}', 'zh', repair_budget())
    assert '中心词' not in prompt and '保留为一个完整名词短语' not in prompt


def test_encouragement_canonicalization_retains_the_entire_operation():
    source = 'Go ahead and turn this on.'
    action, proof = _imperative_comparison_source(source, 'Just turn it on.')
    assert action == 'turn this on.'
    assert proof['source_quote'] + proof['operation_quote'] == source
    assert _imperative_comparison_source(source, 'Turn this off.')[0] == 'turn this on.'


@pytest.mark.parametrize('back', [
    'I turned this on.', 'Should I turn this on?', "Do not turn this on.",
    'Turn this on if you want.', 'Never turn it on.', 'You can turn it on.',
])
def test_statement_question_negation_or_condition_is_never_canonicalized(back):
    source = 'Go ahead and turn this on.'
    assert _imperative_comparison_source(source, back) == (source, None)


@pytest.mark.parametrize('source', [
    'Go ahead and never turn this on.', 'Go ahead and turn it on before the job.',
    'Go ahead with the job.', 'Go ahead and tell me if it works.',
])
def test_source_factual_qualifiers_are_not_stripped(source):
    assert _imperative_comparison_source(source, 'Turn it on.') == (source, None)


def test_reverse_comparison_still_rejects_the_opposite_action(monkeypatch):
    from core import duration_candidate_audit as audit
    def reply(prompt, **kwargs):
        title = kwargs['log_title']
        if title == 'candidate_blind_backtranslation':
            return 'Turn this off.'
        if title == 'candidate_backtranslation_fidelity':
            return {'faithful': True, 'differences': []}
        assert title == 'candidate_reverse_semantics'
        data = json.loads(prompt.rsplit('\n', 1)[-1])
        assert data['A'] == 'turn this on.' and data['B'] == 'Turn this off.'
        return {'equivalent': False, 'differences': [{
            'source_quote': 'turn this on', 'backtranslation_quote': 'Turn this off',
            'difference': 'on/off reversal'}]}
    monkeypatch.setattr(audit, 'ask_gpt', reply)
    result = audit.reverse_semantic_check('Go ahead and turn this on.', '把它关掉。', '{}', {})
    assert result['equivalent'] is False
    assert result['source_speech_act_evidence']['operation_quote'] == 'turn this on.'
