import pytest

from core.duration_candidate_audit import validate_defect_evidence


@pytest.fixture(autouse=True)
def isolate_forward_audit(monkeypatch):
    # This module tests forward-review recovery. Reverse inference and its
    # integration have separate positive/negative contracts.
    from core import duration_candidate_audit as audit
    monkeypatch.setattr(audit, 'reverse_semantic_check', lambda *args:
                        {'equivalent': True, 'issues': [], 'defects': []})


def test_semantic_rejection_requires_actual_source_and_candidate_evidence():
    source, candidate = 'I save money.', '我花钱。'
    result = {'equivalent': False, 'issues': ['Action reversed'], 'source_constraints': [],
              'defects': [{'source_quote': 'save', 'candidate_quote': '花钱', 'difference': 'Save became spend'}]}
    assert validate_defect_evidence(result, source, candidate)['status'] == 'success'
    for field, value in [('source_quote', 'spend'), ('candidate_quote', '省钱'), ('difference', '')]:
        wrong = {**result, 'defects': [{**result['defects'][0], field: value}]}
        assert validate_defect_evidence(wrong, source, candidate)['status'] == 'error'


@pytest.mark.parametrize('decision,defects,expected', [
    ('same', [], 'success'),
    ('same', [{'source_quote':'Save', 'candidate_quote':'删除', 'difference':'Reversed action'}], 'error'),
    ('different', [], 'error'),
    ('different', [{'source_quote':'Save', 'candidate_quote':'删除', 'difference':'Reversed action'}], 'success'),
    ('unknown', [], 'error'),
])
def test_alternate_contract_never_discards_fact_differences(decision, defects, expected):
    from core import duration_candidate_audit as audit
    result = audit._decision_review_response({'decision':decision,'differences':defects})
    assert validate_defect_evidence(result,'Save it.','删除它。')['status'] == expected


def test_alternate_contract_cannot_hide_differences_behind_an_extra_boolean():
    from core import duration_candidate_audit as audit
    value={'decision':'different','differences':[{'source_quote':'Save','candidate_quote':'删除',
           'difference':'Reversed action'}],'equivalent':True,'issues':[],'defects':[]}
    result=audit._decision_review_response(value)
    assert validate_defect_evidence(result,'Save it.','删除它。')['status'] == 'error'


@pytest.mark.parametrize('passes', [True, False])
def test_contradictory_schema_gets_one_distinct_bound_review(tmp_path, monkeypatch, passes):
    from core import duration_candidate_audit as audit, runtime_context
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runtime_context,'effective_source_language',lambda *_:'en')
    monkeypatch.setattr(runtime_context,'effective_target_language',lambda *_:'zh')
    calls = []
    source, target = 'Save it.', '保存它。' if passes else '删除它。'
    def reply(prompt, **kwargs):
        calls.append(kwargs['log_title'])
        if kwargs['log_title'] != 'duration_candidate_decision_recovery':
            raise ValueError('An equivalent candidate cannot contain unresolved semantic issues')
        assert source in prompt and target in prompt
        assert kwargs['max_retries'] == 1
        response = {'decision':'same' if passes else 'different', 'differences':[] if passes else
            [{'source_quote':'Save','candidate_quote':'删除','difference':'Reversed action'}]}
        normalized = kwargs['response_normalizer'](response)
        assert kwargs['valid_def'](normalized)['status'] == 'success'
        return normalized
    monkeypatch.setattr(audit,'ask_gpt',reply)
    result = audit.audit_duration_candidate(source,'',target,'{}',_resolve_ambiguity=False)
    assert result['equivalent'] is passes
    assert calls == ['duration_candidate_semantics','duration_candidate_semantics_compact',
                     'duration_candidate_decision_recovery']


@pytest.mark.parametrize('verdict', [
    {'equivalent': False, 'issues': ['Changed meaning'], 'defects': []},
    {'equivalent': True, 'issues': [], 'defects': [{'source_quote': 'save', 'candidate_quote': '', 'difference': 'Omitted'}]},
])
def test_semantic_verdict_cannot_contradict_its_evidence(verdict):
    assert validate_defect_evidence(verdict, 'I save money.', '我省钱。')['status'] == 'error'


def test_chinese_audit_keeps_omissions_blocking(tmp_path, monkeypatch):
    from core import duration_candidate_audit as audit, runtime_context
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runtime_context, 'effective_target_language', lambda *_: 'zh')
    monkeypatch.setattr(runtime_context, 'effective_source_language', lambda *_: 'en')
    verdict = {'equivalent': False, 'issues': ['Omitted animations'], 'source_constraints': [],
               'defects': [{'source_quote': 'animations', 'candidate_quote': '', 'difference': 'Object omitted'}]}
    monkeypatch.setattr(audit, 'ask_gpt', lambda prompt, **kwargs: verdict)
    result = audit.audit_duration_candidate('She chooses graphics and animations.', '', '她选择图形。', '{}')
    assert result['equivalent'] is False
    assert result['defects'] == verdict['defects']


@pytest.mark.parametrize('second_passes', [True, False])
def test_source_resolution_requires_a_new_complete_gate_without_neighbor_facts(tmp_path, monkeypatch, second_passes):
    import json
    from core import duration_candidate_audit as audit, runtime_context
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runtime_context, 'effective_target_language', lambda *_: 'zh')
    monkeypatch.setattr(runtime_context, 'effective_source_language', lambda *_: 'en')
    rejected = {'equivalent': False, 'issues': ['Unresolved word sense'], 'source_constraints': [],
                'defects': [{'source_quote': 'Title', 'candidate_quote': '标题', 'difference': 'Unresolved word sense'}]}
    reviews = []
    def reply(prompt, **kwargs):
        if kwargs['log_title'] == 'source_only_disambiguation':
            assert 'CURRENT_CANDIDATE' not in prompt
            return '标题'
        if kwargs['log_title'] == 'duration_glossary_meaning':
            return {'preserved': True, 'translation_quote': '标题', 'issue': ''}
        reviews.append(prompt)
        if len(reviews) == 1:
            return rejected
        assert 'NEIGHBOR_FACTS' not in prompt
        return {'equivalent': True, 'issues': [], 'source_constraints': [], 'defects': []} if second_passes else rejected
    monkeypatch.setattr(audit, 'ask_gpt', reply)
    def reverse(*args):
        if len(reviews) < 2:
            raise RuntimeError('Isolate source-resolution recovery from objection reconciliation')
        return {'equivalent': True, 'issues': [], 'defects': []}
    monkeypatch.setattr(audit, 'reverse_semantic_check', reverse)
    monkeypatch.setattr(audit, 'review_scoped_objection', lambda *args:
        {'equivalent': False, 'issues': ['Still unresolved'], 'defects': rejected['defects']})
    result = audit.audit_duration_candidate('Title.', '', '标题。CURRENT_CANDIDATE',
        json.dumps({'parent_source': 'Title. NEIGHBOR_FACTS', 'surrounding_source': ['NEIGHBOR_FACTS']}))
    assert result['equivalent'] is second_passes
    assert result['initial_verdict']['equivalent'] is False
    assert result['source_meaning_resolution']['blind_to_candidate'] is True
    assert len(reviews) == 2


def test_source_resolution_failure_retains_rejection(tmp_path, monkeypatch):
    from core import duration_candidate_audit as audit, runtime_context
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runtime_context, 'effective_target_language', lambda *_: 'zh')
    rejected = {'equivalent': False, 'issues': ['Lost action'], 'source_constraints': [],
                'defects': [{'source_quote': 'Save', 'candidate_quote': '', 'difference': 'Lost action'}]}
    monkeypatch.setattr(audit, 'ask_gpt', lambda *a, **k: rejected)
    result = audit.audit_duration_candidate('Save.', '', '打开。', '{}')
    assert result['equivalent'] is False
    assert result['source_meaning_resolution']['status'] == 'unavailable'


@pytest.mark.parametrize('failure_type', ['IncompleteLLMResponseError', 'validation_error'])
@pytest.mark.parametrize('passes', [True, False])
def test_incomplete_audit_rechecks_unchanged_text_and_keeps_semantic_gate(tmp_path, monkeypatch, passes, failure_type):
    from core import duration_candidate_audit as audit, runtime_context
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runtime_context, 'effective_target_language', lambda *_: 'zh')
    monkeypatch.setattr(runtime_context, 'effective_source_language', lambda *_: 'en')
    source, candidate = 'Save it.', '保存它。' if passes else '删除它。'
    calls = []
    def reply(prompt, **kwargs):
        calls.append((prompt, kwargs))
        if len(calls) == 1:
            error = audit.IncompleteLLMResponseError if failure_type == 'IncompleteLLMResponseError' else Exception
            raise error('No complete audit result')
        assert source in prompt and candidate in prompt
        assert kwargs['max_retries'] == 1
        value = {'equivalent': passes, 'issues': [] if passes else ['Save became delete'],
                 'style_notes': [], 'source_constraints': [], 'defects': [] if passes else
                 [{'source_quote': 'Save', 'candidate_quote': '删除', 'difference': 'Action reversed'}]}
        assert kwargs['valid_def'](value)['status'] == 'success'
        return value
    monkeypatch.setattr(audit, 'ask_gpt', reply)
    verdict = audit.audit_duration_candidate(source, '', candidate, '{}', _resolve_ambiguity=False)
    assert verdict['equivalent'] is passes
    assert len(calls) == 2
    assert calls[0][0] in calls[1][0]
    assert calls[1][1]['log_title'] == 'duration_candidate_semantics_compact'


def test_incomplete_audit_recovery_is_bounded_and_never_accepts_unknown(tmp_path, monkeypatch):
    from core import duration_candidate_audit as audit, runtime_context
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runtime_context, 'effective_target_language', lambda *_: 'zh')
    calls = []
    def unavailable(*args, **kwargs):
        calls.append(kwargs)
        raise audit.IncompleteLLMResponseError('length')
    monkeypatch.setattr(audit, 'ask_gpt', unavailable)
    with pytest.raises(audit.IncompleteLLMResponseError):
        audit.audit_duration_candidate('Save it.', '', '保存它。', '{}')
    assert len(calls) == 3  # original, concise review, then independent source reference


@pytest.mark.parametrize('passes', [True, False])
def test_unusable_reviews_get_blind_source_evidence_without_accepting_changed_meaning(tmp_path, monkeypatch, passes):
    from core import duration_candidate_audit as audit, runtime_context
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runtime_context, 'effective_target_language', lambda *_: 'zh')
    monkeypatch.setattr(runtime_context, 'effective_source_language', lambda *_: 'en')
    original_calls, source_calls = [], []
    candidate = '保存它。' if passes else '删除它。'
    def reply(prompt, **kwargs):
        title = kwargs['log_title']
        if title == 'source_only_disambiguation':
            source_calls.append(prompt)
            assert candidate not in prompt
            return '保存它。'
        if title == 'duration_glossary_meaning':
            return {'preserved': False, 'translation_quote': candidate, 'issue': 'Save became delete'}
        if title == 'duration_candidate_semantics_compact':
            raise Exception('Validation still failed: fabricated candidate quotation')
        if 'Independent source-only contextual translation' not in prompt:
            original_calls.append(prompt)
            raise Exception('Validation still failed: fabricated candidate quotation')
        return {'equivalent': True, 'issues': [], 'style_notes': [], 'source_constraints': [], 'defects': []}
    monkeypatch.setattr(audit, 'ask_gpt', reply)
    monkeypatch.setattr(audit, 'reverse_semantic_check', lambda *args:
        {'equivalent': passes, 'issues': [] if passes else ['Save became delete'],
         'defects': [] if passes else [{'source_quote': 'Save', 'candidate_quote': '删除',
                                        'difference': 'Save became delete'}]})
    for _ in range(2):
        result = audit.audit_duration_candidate('Save it.', '', candidate, '{}')
        assert result['equivalent'] is passes
        assert result['source_meaning_resolution']['blind_to_candidate'] is True
    # A deterministic unusable request is remembered with source/context,
    # model and policy binding; it is never promoted to a passing review.
    assert len(original_calls) == 1
    assert len(source_calls) == 2
