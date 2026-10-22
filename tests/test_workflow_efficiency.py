"""Less inference must preserve the same evidence, not skip failed checks."""
import json
from dataclasses import replace
from types import SimpleNamespace

import pandas as pd
import pytest


def semantic_setup(tmp_path, monkeypatch):
    from core import duration_candidate_audit as audit, runtime_context
    from core.llm_provider import LLMProviderConfig
    monkeypatch.chdir(tmp_path)
    provider = LLMProviderConfig(name='omlx', model='test-v1', base_url='http://localhost/v1', api_key='')
    monkeypatch.setattr(audit, 'get_llm_provider_config', lambda *_: provider)
    monkeypatch.setattr(runtime_context, 'effective_source_language', lambda *_: 'en')
    monkeypatch.setattr(runtime_context, 'effective_target_language', lambda *_: 'zh')
    calls = []
    def complete(source, anchor, candidate, context, **kwargs):
        calls.append((source, candidate, context))
        return {'equivalent': True, 'issues': [], 'style_notes': [], 'source_constraints': [],
                'defects': [], 'meaning_checks': [], 'clause_checks': [],
                'reverse_check': {'equivalent': True, 'differences': [], 'issues': [], 'defects': [],
                                  'backtranslation': source, 'blind_to_source': True}}
    monkeypatch.setattr(audit, '_audit_duration_candidate', complete)
    return audit, provider, calls


def test_unchanged_gate_reuses_complete_receipt_despite_different_anchor(tmp_path, monkeypatch):
    audit, _, calls = semantic_setup(tmp_path, monkeypatch)
    first = audit.audit_duration_candidate('Save it.', 'First draft.', '保存它。', '{"topic":"files","parent_source":"Save it."}')
    second = audit.audit_duration_candidate('Save it.', '', '保存它。', '{"parent_source": "Save it.", "topic": "files"}')
    assert len(calls) == 1
    assert not first['audit_receipt']['reused'] and second['audit_receipt']['reused']
    assert first['reverse_check'] == second['reverse_check']


@pytest.mark.parametrize('change', ['source', 'target', 'context', 'provider', 'language', 'policy'])
def test_changed_semantic_inputs_force_full_audit(tmp_path, monkeypatch, change):
    audit, provider, calls = semantic_setup(tmp_path, monkeypatch)
    args = ['Save it.', '', '保存它。', '{}']
    audit.audit_duration_candidate(*args)
    if change == 'source': args[0] = 'Delete it.'
    if change == 'target': args[2] = '删除它。'
    if change == 'context': args[3] = '{"topic":"banking"}'
    if change == 'provider': monkeypatch.setattr(audit, 'get_llm_provider_config', lambda *_: replace(provider, model='test-v2'))
    if change == 'language':
        from core import runtime_context
        monkeypatch.setattr(runtime_context, 'effective_target_language', lambda *_: 'ja')
    if change == 'policy':
        from core.pipeline import artifact_manifest
        original = artifact_manifest.hash_path
        monkeypatch.setattr(artifact_manifest, 'hash_path', lambda p: original(p) + '-changed')
    result = audit.audit_duration_candidate(*args)
    assert len(calls) == 2 and not result['audit_receipt']['reused']


@pytest.mark.parametrize('kind', ['corrupt', 'missing_reverse', 'rejected', 'unavailable'])
def test_incomplete_or_corrupted_evidence_never_becomes_a_reused_pass(tmp_path, monkeypatch, kind):
    audit, _, calls = semantic_setup(tmp_path, monkeypatch)
    args = ('Save it.', '', '保存它。', '{}')
    if kind == 'corrupt':
        audit.audit_duration_candidate(*args)
        path = next((tmp_path/'output/log/source_semantic_receipts').glob('*.json'))
        saved = json.loads(path.read_text()); saved['verdict']['reverse_check']['backtranslation'] = 'Delete it.'
        path.write_text(json.dumps(saved))
        assert audit.audit_duration_candidate(*args)['audit_receipt']['reused'] is False
        assert len(calls) == 2
        return
    complete = audit._audit_duration_candidate
    def broken(*a, **kw):
        value = complete(*a, **kw)
        if kind == 'unavailable': raise RuntimeError('audit unavailable')
        if kind == 'missing_reverse': value.pop('reverse_check')
        if kind == 'rejected': value.update(equivalent=False, issues=['wrong action'])
        return value
    monkeypatch.setattr(audit, '_audit_duration_candidate', broken)
    for _ in range(2):
        if kind == 'unavailable':
            with pytest.raises(RuntimeError): audit.audit_duration_candidate(*args)
        else:
            value = audit.audit_duration_candidate(*args)
            assert not value['audit_receipt']['reused']
    assert len(calls) == 2
    assert not list((tmp_path/'output/log/source_semantic_receipts').glob('*.json'))


def test_publication_reuses_exact_preflight_semantics_but_checks_changed_wording(tmp_path, monkeypatch):
    from core import translation_review, speech_budget
    audit, _, calls = semantic_setup(tmp_path, monkeypatch)
    monkeypatch.setattr(translation_review, 'load_key', lambda key, default=None: 'zh' if key=='target_language' else default)
    monkeypatch.setattr(speech_budget, 'duration_context', lambda *_: '{"surrounding_source":[]}')
    audit.audit_duration_candidate('Save it.', '', '保存它。', '{"surrounding_source":[]}')
    frame = pd.DataFrame([{'LineID':'u1', 'Source':'Save it.', 'Translation':'保存它。'}])
    _, report = translation_review.review_source_units(frame, {}, tmp_path/'review.json')
    assert report['status'] == 'pass' and report['semantic_receipts_reused'] == 1 and len(calls) == 1
    frame.loc[0, 'Translation'] = '删除它。'
    monkeypatch.setattr(audit, '_audit_duration_candidate', lambda *a, **kw:
                        {'equivalent':False, 'issues':['Save became delete'], 'source_constraints':[]})
    _, report = translation_review.review_source_units(frame, {}, tmp_path/'review.json')
    assert report['status'] != 'pass' and report['semantic_receipts_reused'] == 0


@pytest.mark.parametrize('evidence', ['complete', 'missing', 'unbound_quote'])
def test_named_glossary_receipt_requires_same_complete_evidence(tmp_path, monkeypatch, evidence):
    audit, _, calls = semantic_setup(tmp_path, monkeypatch)
    complete = audit._audit_duration_candidate
    def named(*a, **kw):
        value = complete(*a, **kw)
        value['meaning_checks'] = ([{'src': 'Parakeet', 'meaning': 'Parakeet',
            'preserved': True, 'translation_quote': 'Parakeet负责转录。', 'issue': ''}]
            if evidence != 'missing' else [])
        if evidence == 'unbound_quote':
            value['meaning_checks'][0]['translation_quote'] = '另一个不在译文中的名字'
        return value
    monkeypatch.setattr(audit, '_audit_duration_candidate', named)
    context = json.dumps({'glossary_meanings': [{'src': 'Parakeet', 'tgt': 'Parakeet'}]})
    args = ('Parakeet transcribes.', '', 'Parakeet负责转录。', context)
    audit.audit_duration_candidate(*args)
    second = audit.audit_duration_candidate(*args)
    assert second['audit_receipt']['reused'] is (evidence == 'complete')
    assert len(calls) == (1 if evidence == 'complete' else 2)


def test_empty_duration_guidance_reuses_audit_but_real_guidance_invalidates(tmp_path, monkeypatch):
    from core.speech_budget import with_source_category_terms
    audit, _, calls = semantic_setup(tmp_path, monkeypatch)
    args = ('Save it.', '', '保存它。')
    context = '{"parent_source":"Save it.","topic":"files"}'
    audit.audit_duration_candidate(*args, context)
    unchanged = audit.audit_duration_candidate(*args, with_source_category_terms(context, []))
    assert unchanged['audit_receipt']['reused'] and len(calls) == 1
    guidance = [{'src': 'it', 'tgt': 'document'}]
    changed = audit.audit_duration_candidate(*args, with_source_category_terms(context, guidance))
    assert not changed['audit_receipt']['reused'] and len(calls) == 2


@pytest.mark.parametrize('kind', ['complete', 'missing_fidelity', 'wrong_quote', 'unfaithful_repair'])
def test_reverse_repair_receipt_requires_bound_fidelity_and_repeated_source_comparison(tmp_path, monkeypatch, kind):
    audit, _, _ = semantic_setup(tmp_path, monkeypatch)
    source, candidate = 'The paper proves it.', '论文证明了它。'
    defect = {'source_quote': 'proves', 'backtranslation_quote': 'might prove',
              'candidate_quote': candidate, 'difference': 'Incorrect reverse modality'}
    fidelity = {'faithful':False,'differences':[{'candidate_quote':'证明',
        'backtranslation_quote':'might prove','difference':'Unsupported uncertainty'}]}
    if kind == 'wrong_quote': fidelity['differences'][0]['candidate_quote']='可能证明'
    original = {'equivalent': False, 'differences': [defect], 'defects': [defect],
                'backtranslation': 'The paper might prove it.', 'backtranslation_fidelity':fidelity}
    if kind == 'missing_fidelity':original.pop('backtranslation_fidelity')
    value = {'equivalent': True, 'issues': [], 'style_notes': [], 'source_constraints': [],
             'defects': [], 'meaning_checks': [], 'clause_checks': [],
             'reverse_check': {'equivalent': True, 'differences': [], 'blind_to_source': True,
                'backtranslation': source,
                'backtranslation_recovery': {'blind_to_source':True,'original_verdict':original,
                    'original_backtranslation':original['backtranslation'],
                    'repaired_fidelity':{'faithful':kind!='unfaithful_repair','differences':[]}}}}
    assert audit._reusable_semantic_verdict(value, source, candidate, '{}') is (kind == 'complete')


@pytest.mark.parametrize('response_json', [True, False])
def test_same_invalid_reply_stops_after_correction_and_records_all_requests(tmp_path, monkeypatch, response_json):
    from core import ask_gpt as api
    from core.llm_provider import LLMProviderConfig
    monkeypatch.setattr(api, 'LOG_FOLDER', str(tmp_path))
    monkeypatch.setattr(api, 'get_llm_provider_config', lambda: LLMProviderConfig(name='omlx', model='test', base_url='http://localhost/v1', api_key=''))
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason='stop', message=SimpleNamespace(content='{"wrong":1}' if response_json else 'wrong'))])
    monkeypatch.setattr(api, 'create_chat_client', lambda _: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    with pytest.raises(api.RepeatedInvalidLLMResponseError):
        api.ask_gpt('source', response_json=response_json, max_retries=4, retry_interval=0,
                    valid_def=lambda _: {'status':'error','message':'Missing required evidence'})
    assert len(calls) == 2
    events = [json.loads(line) for line in (tmp_path/'request_events.jsonl').read_text().splitlines()]
    assert [e['attempt'] for e in events] == [1, 2]
    assert all(e['outcome']=='validation_rejected' and e['elapsed_seconds']>=0 for e in events)
    errors = json.loads((tmp_path/'error.json').read_text())
    assert errors[-1]['metadata']['repeated_invalid_response'] is True
    assert errors[-1]['metadata']['request_messages'][-1]['role'] == 'user'
    assert not (tmp_path/'default.json').exists()


def test_chinese_target_word_cut_is_rejected_before_model_review(monkeypatch):
    from core import bilingual_alignment as ba
    monkeypatch.setattr(ba, '_source_word_spans', lambda t: [(2,4)] if t=='所有其他工具' else [])
    unit = {'unit_id':'u', 'source':'all other tools', 'target':'所有其他工具'}
    pairs = [{'source':'all other', 'target':'所有其'}, {'source':'tools', 'target':'他工具'}]
    with pytest.raises(ValueError, match='target word'):
        ba.validate_pairs(unit, {'pairs':pairs}, chars=40, source_chars=40, check_timing=False)


def test_repeated_rejected_partition_is_never_reviewed_again(monkeypatch):
    from core import bilingual_alignment as ba
    unit={'unit_id':'u', 'source':'one two three four', 'target':'甲乙丙丁'}
    pairs=[{'source':'one two','target':'甲乙'}, {'source':'three four','target':'丙丁'}]
    monkeypatch.setattr(ba, 'validate_pairs', lambda u, data, **kw: data['pairs'])
    monkeypatch.setattr(ba, '_decode_pairs', lambda u, data: data)
    monkeypatch.setattr(ba, 'ask_gpt', lambda *a, **kw: {'pairs':pairs})
    monkeypatch.setattr(ba, '_native_anchor_pairs', lambda *a, **kw: {'pairs':pairs})
    calls=[]
    def reject(*args):
        calls.append(args)
        return [{'equivalent':False, 'issues':['wrong correspondence']}] * 2
    monkeypatch.setattr(ba, '_review_pairs', reject)
    with pytest.raises(ValueError, match='already rejected'):
        ba.align_unit(unit, chars=5, source_chars=12, check_timing=False)
    assert len(calls) == 1


def test_coalescing_repairs_words_with_exact_coverage_and_keeps_semantic_review_required(monkeypatch):
    from core import bilingual_alignment as ba
    monkeypatch.setattr(ba, '_source_word_spans', lambda t: [(2,4)] if t=='所有其他工具' else [])
    unit = {'unit_id':'u', 'source':'all other tools', 'target':'所有其他工具'}
    data = {'pairs':[{'source':'all other', 'target':'所有其'}, {'source':'tools', 'target':'他工具'}]}
    merged = ba.coalesce_lexical_pairs(unit, data, chars=22, source_chars=42)
    assert merged == {'pairs':[{'source':'all other tools', 'target':'所有其他工具'}]}
    assert data['pairs'][0]['target'] == '所有其'  # proposals are not edited in place
    with pytest.raises(ValueError, match='semantic correspondence review'):
        ba.validate_pairs(unit, merged, chars=22, source_chars=42, require_review=True, check_timing=False)
    # An overflowing join cannot silently replace the display contract.
    blocked = ba.coalesce_lexical_pairs(unit, data, chars=22, source_chars=12)
    assert blocked == data
    with pytest.raises(ValueError, match='target word'):
        ba.validate_pairs(unit, blocked, chars=22, source_chars=12, check_timing=False)
