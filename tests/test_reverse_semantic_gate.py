import json
import pytest

from core import duration_candidate_audit as audit


def positive():
    return {'equivalent': True, 'issues': [], 'style_notes': [], 'source_constraints': [], 'defects': []}


def test_blind_recovery_uses_a_distinct_target_only_grammar_contract():
    target = '第三个工作是动画师。这次，我喜欢它抓取了标志这一点。'
    first = audit._blind_backtranslation_repair_prompt(target)
    recovered = audit._blind_grammar_recovery_prompt(target)
    assert first != recovered
    assert target in first and target in recovered
    assert '去+动词+了' in recovered
    assert 'a/an' in recovered
    assert 'SOURCE_SENTINEL' not in recovered and 'REJECTED_ENGLISH_SENTINEL' not in recovered


def test_clausal_attitude_reference_preserves_its_complete_actual_scope():
    candidate = '这次，我喜欢它去抓取了特定的标志这一点'
    bound = audit._candidate_clause_reference(candidate)
    assert bound['clause_quote'] == '它去抓取了特定的标志'
    assert bound['attitude_quote'] == '喜欢' and bound['reference_quote'] == '这一点'
    assert audit._candidate_clause_reference('我喜欢图上的这一点。') is None
    assert audit._candidate_clause_reference('我喜欢这一点数据。') is None


def negative():
    return {'equivalent': False, 'issues': ['Added singing'], 'style_notes': [], 'source_constraints': [],
            'defects': [{'source_quote': 'stop', 'candidate_quote': '唱',
                         'backtranslation_quote': 'sing', 'difference': 'An unsupported activity'}]}


def test_reverse_translator_is_blind_and_false_acceptance_is_challenged(monkeypatch):
    calls = []
    def reply(prompt, **kwargs):
        calls.append(kwargs['log_title'])
        assert kwargs['max_retries'] == (2 if kwargs['log_title'] == 'candidate_backtranslation_fidelity' else 1)
        if kwargs['log_title'] == 'candidate_blind_backtranslation':
            assert '唱' in prompt
            assert 'SOURCE_ONLY_SENTINEL' not in prompt
            assert 'CONTEXT_ONLY_SENTINEL' not in prompt
            return 'I sing and stop.'
        if kwargs['log_title'] == 'candidate_backtranslation_fidelity':
            assert 'SOURCE_ONLY_SENTINEL' not in prompt and 'CONTEXT_ONLY_SENTINEL' not in prompt
            value = {'faithful': True, 'differences': []}
            assert kwargs['valid_def'](value)['status'] == 'success'
            return value
        assert 'SOURCE_ONLY_SENTINEL' in prompt
        assert 'CONTEXT_ONLY_SENTINEL' in prompt
        assert 'NEIGHBOR_ONLY_SENTINEL' not in prompt
        assert 'PARENT_ONLY_SENTINEL' not in prompt
        result = {'equivalent': False, 'differences': [
            {'source_quote': 'stop', 'backtranslation_quote': 'sing', 'difference': 'Added singing'}]}
        assert kwargs['valid_def'](result)['status'] == 'success'
        return result
    monkeypatch.setattr(audit, 'ask_gpt', reply)
    result = audit.reverse_semantic_check('SOURCE_ONLY_SENTINEL I stop.', '我唱到这儿停下。',
        json.dumps({'topic': 'CONTEXT_ONLY_SENTINEL',
                    'surrounding_source': ['NEIGHBOR_ONLY_SENTINEL'],
                    'parent_source': 'PARENT_ONLY_SENTINEL'}), {})
    assert result['equivalent'] is False
    assert result['blind_to_source'] is True
    assert len(calls) == 3


@pytest.mark.parametrize('quote', ['not in actual audio text', None])
def test_reverse_defects_cannot_fabricate_evidence(quote):
    result = {'equivalent': False, 'differences': [
        {'source_quote': 'stop', 'backtranslation_quote': quote, 'difference': 'Added singing'}]}
    assert audit.validate_reverse_evidence(result, 'I stop.', 'I sing.')['status'] == 'error'


def test_actual_target_objection_review_cannot_inherit_a_wrong_auxiliary_word_sense(monkeypatch):
    source = "When something's clicked on, you can hear the clicking sound."
    candidate = '点东西时，能听到点击声。'
    defect = {'source_quote': "When something's clicked on", 'candidate_quote': candidate,
              'difference': 'WRONG_AUXILIARY_INTERPRETATION'}
    def reply(prompt, **kwargs):
        assert source in prompt and candidate in prompt
        assert 'WRONG_AUXILIARY_INTERPRETATION' not in prompt
        assert 'ORDERING_SENTINEL' not in prompt
        value = {'equivalent': True, 'source_quote': source, 'candidate_quote': candidate,
                 'candidate_meaning': 'Clicking an object can produce a clicking sound.',
                 'explanation': 'The complete actual target describes the same action and sound.'}
        assert kwargs['valid_def'](value)['status'] == 'success'
        return value
    monkeypatch.setattr(audit, 'ask_gpt', reply)
    assert audit.review_scoped_objection(source,candidate,defect,'{}',{},
        'ORDERING_SENTINEL')['equivalent'] is True


@pytest.mark.parametrize('scoped_pass', [True, False])
def test_source_resolution_keeps_actual_target_review_but_cannot_resolve_twice(tmp_path, monkeypatch, scoped_pass):
    from core import runtime_context
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runtime_context, 'effective_target_language', lambda *_: 'zh')
    monkeypatch.setattr(runtime_context, 'effective_source_language', lambda *_: 'en')
    source, candidate = 'Hmm, chips.', '嗯，芯片。'
    resolved_context = json.dumps({'topic': 'RESOLVED_WORD_SENSE'})
    rejected = {'equivalent': False, 'issues': ['Ambiguous noun'], 'defects': [
        {'source_quote': 'chips', 'candidate_quote': '芯片', 'difference': 'Ambiguous noun'}]}
    calls = []
    def resolve(*args):
        calls.append('resolve')
        return {'context': resolved_context, 'references': [], 'blind_to_candidate': True}
    def reply(prompt, **kwargs):
        return positive() if 'RESOLVED_WORD_SENSE' in prompt else rejected
    def reverse(source, candidate, context, *args):
        if context != resolved_context:
            raise RuntimeError('Source word sense unresolved')
        return {'equivalent': False, 'issues': ['Auxiliary article'], 'differences': [
            {'source_quote': 'chips', 'backtranslation_quote': 'a chip', 'difference': 'Auxiliary article'}],
            'defects': [{'source_quote': 'chips', 'candidate_quote': candidate, 'difference': 'Auxiliary article'}],
            'backtranslation': 'Um, a chip.', 'blind_to_source': True}
    def scoped(source, candidate, defect, *args):
        calls.append('scoped')
        return {'equivalent': scoped_pass, 'explanation': 'Actual target evidence',
                'source_quote': 'chips', 'candidate_quote': '芯片', 'candidate_meaning': 'chips',
                'issues': [] if scoped_pass else ['Actual factual discrepancy'],
                'defects': [] if scoped_pass else [defect]}
    monkeypatch.setattr(audit, 'ask_gpt', reply)
    monkeypatch.setattr(audit, '_resolve_source_meanings', resolve)
    monkeypatch.setattr(audit, 'reverse_semantic_check', reverse)
    monkeypatch.setattr(audit, 'review_scoped_objection', scoped)
    verdict = audit.audit_duration_candidate(source, '', candidate, '{}')
    assert verdict['equivalent'] is scoped_pass
    assert calls == ['resolve', 'scoped']


@pytest.mark.parametrize('forward_pass,reverse_pass,expected', [(True, False, False), (True, True, True), (False, True, False)])
def test_both_gates_are_required(tmp_path, monkeypatch, forward_pass, reverse_pass, expected):
    from core import runtime_context
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runtime_context, 'effective_target_language', lambda *_: 'zh')
    monkeypatch.setattr(runtime_context, 'effective_source_language', lambda *_: 'en')
    calls = []
    monkeypatch.setattr(audit, 'ask_gpt', lambda *a, **k: positive() if forward_pass else negative())
    def reverse(*args):
        calls.append(args)
        return positive() if reverse_pass else negative()
    monkeypatch.setattr(audit, 'reverse_semantic_check', reverse)
    result = audit.audit_duration_candidate('I stop.', '', '我唱歌。', '{}', _resolve_ambiguity=False)
    assert result['equivalent'] is expected
    assert len(calls) == int(forward_pass)
    if forward_pass and not reverse_pass:
        assert result['forward_verdict']['equivalent'] is True
        assert result['reverse_check']['equivalent'] is False


def test_unavailable_reverse_review_never_passes(tmp_path, monkeypatch):
    from core import runtime_context
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runtime_context, 'effective_target_language', lambda *_: 'zh')
    monkeypatch.setattr(runtime_context, 'effective_source_language', lambda *_: 'en')
    monkeypatch.setattr(audit, 'ask_gpt', lambda *a, **k: positive())
    def unavailable(*args):
        raise audit.IncompleteLLMResponseError('No complete review')
    monkeypatch.setattr(audit, 'reverse_semantic_check', unavailable)
    with pytest.raises(audit.IncompleteLLMResponseError):
        audit.audit_duration_candidate('I stop.', '', '我停下。', '{}')


def test_source_resolution_cannot_import_neighbor_facts(monkeypatch):
    from core import runtime_context
    monkeypatch.setattr(runtime_context, 'effective_target_language', lambda *_: 'zh')
    monkeypatch.setattr(runtime_context, 'effective_source_language', lambda *_: 'en')
    def reply(prompt, **kwargs):
        assert 'TOPIC_SENTINEL' in prompt
        assert 'NEIGHBOR_ONLY_SENTINEL' not in prompt
        assert 'PARENT_ONLY_SENTINEL' not in prompt
        assert 'CANDIDATE_ONLY_SENTINEL' not in prompt
        return '制作每一个标题'
    monkeypatch.setattr(audit, 'ask_gpt', reply)
    result = audit._resolve_source_meanings('This person builds every single title,',
        json.dumps({'topic': 'TOPIC_SENTINEL', 'parent_source': 'PARENT_ONLY_SENTINEL',
                    'surrounding_source': ['NEIGHBOR_ONLY_SENTINEL']}),
        {'defects': [{'source_quote': 'builds every single title',
                     'candidate_quote': 'CANDIDATE_ONLY_SENTINEL'}]}, {})
    assert result['references'][0]['src'] == 'builds every single title'
    assert result['blind_to_candidate'] is True
    assert audit.current_source_meanings('This person builds every single title,', result['context']) == []
    assert json.loads(result['context'])['source_meaning_evidence'] == result['references']


@pytest.mark.parametrize('whole_pass,fragment_pass,expected', [
    (True, True, True), (True, False, False), (False, True, False), (True, None, False)])
def test_disputed_rejection_requires_whole_and_every_fragment_proof(tmp_path, monkeypatch,
                                                                  whole_pass, fragment_pass, expected):
    from core import runtime_context
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runtime_context, 'effective_source_language', lambda *_: 'en')
    monkeypatch.setattr(runtime_context, 'effective_target_language', lambda *_: 'zh')
    forward = {'equivalent': False, 'issues': ['Claimed lexical difference'], 'defects': [
        {'source_quote': 'builds', 'candidate_quote': '制作', 'difference': 'Claimed lexical difference'}]}
    monkeypatch.setattr(audit, 'ask_gpt', lambda *a, **kw: dict(forward))
    monkeypatch.setattr(audit, '_resolve_source_meanings', lambda *a: None)
    calls = []
    def reverse(source, candidate, *args):
        calls.append((source, candidate))
        if len(calls) > 1 and fragment_pass is None:
            raise RuntimeError('No fragment evidence')
        passed = whole_pass if len(calls) == 1 else fragment_pass
        return {'equivalent': passed, 'issues': [] if passed else ['Sustained mismatch'],
                'defects': [] if passed else [{'source_quote': source, 'candidate_quote': candidate,
                                                'difference': 'Sustained mismatch'}]}
    monkeypatch.setattr(audit, 'reverse_semantic_check', reverse)
    def scoped(source, candidate, defect, *args):
        assert source == 'He builds each title.' and candidate == '他制作每个标题。'
        return reverse(source, candidate)
    monkeypatch.setattr(audit, 'review_scoped_objection', scoped)
    result = audit.audit_duration_candidate('He builds each title.', '', '他制作每个标题。', '{}')
    assert result['equivalent'] is expected
    assert len(calls) == (2 if whole_pass else 1)
    if expected:
        assert result['initial_verdict']['equivalent'] is False
        assert result['objection_review']['all_objections_disproved'] is True


def test_missing_glossary_cannot_be_reconciled_by_general_reverse_agreement(tmp_path, monkeypatch):
    from core import runtime_context
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runtime_context, 'effective_source_language', lambda *_: 'en')
    monkeypatch.setattr(runtime_context, 'effective_target_language', lambda *_: 'zh')
    monkeypatch.setattr(audit, '_glossary_meaning_check', lambda *a, **k:
        {'src': 'Claude', 'meaning': '克劳德', 'preserved': False,
         'translation_quote': '', 'issue': 'Required name omitted'})
    def unexpected(*args):
        pytest.fail('A missing glossary fact must not enter objection reconciliation')
    monkeypatch.setattr(audit, 'reverse_semantic_check', unexpected)
    result = audit.audit_duration_candidate('Claude writes.', '', '它写。',
        json.dumps({'glossary_meanings': [{'src': 'Claude', 'tgt': '克劳德'}]}))
    assert result['equivalent'] is False


@pytest.mark.parametrize('fidelity,expected', [([False, True], True), ([True], False), ([False, False], 'blocked'), ([None], None)])
def test_only_an_unfaithful_auxiliary_backtranslation_can_be_repaired_then_recompared(monkeypatch, fidelity, expected):
    source, candidate = 'The paper proves my point.', '论文证明我的观点。'
    failures = [{'source_quote': 'proves', 'backtranslation_quote': 'might prove', 'difference': 'Added uncertainty'},
                {'source_quote': 'point', 'backtranslation_quote': 'points', 'difference': 'Changed number'}]
    calls = []
    def reply(prompt, **kwargs):
        calls.append(kwargs['log_title'])
        if kwargs['log_title'] == 'candidate_blind_backtranslation':
            return 'The paper might prove my points.'
        if kwargs['log_title'] == 'candidate_blind_backtranslation_repair':
            assert source not in prompt
            return 'The paper proves my point.'
        if kwargs['log_title'] == 'candidate_reverse_semantics':
            if calls.count('candidate_reverse_semantics')>1:
                assert 'The paper proves my point.' in prompt
                return {'equivalent': True, 'differences': []}
            return {'equivalent': False, 'differences': failures}
        assert candidate in prompt
        if calls.count('candidate_backtranslation_fidelity') == 1:
            assert source not in prompt
        at = calls.count('candidate_backtranslation_fidelity') - 1
        if fidelity[at] is None:
            raise RuntimeError('No exact Chinese evidence')
        value = {'faithful': fidelity[at], 'differences': [] if fidelity[at] else [{
            'candidate_quote':'证明','backtranslation_quote':'might prove' if at==0 else 'proves',
            'difference':'Changed the target facts'}]}
        assert kwargs['valid_def'](value)['status'] == 'success'
        return value
    monkeypatch.setattr(audit, 'ask_gpt', reply)
    if expected is None:
        with pytest.raises(RuntimeError, match='No exact Chinese evidence'):
            audit.reverse_semantic_check(source, candidate, '{}', {})
    elif expected == 'blocked':
        with pytest.raises(audit.IncompleteLLMResponseError, match='remained unfaithful'):
            audit.reverse_semantic_check(source, candidate, '{}', {})
    else:
        result = audit.reverse_semantic_check(source, candidate, '{}', {})
        assert result['equivalent'] is expected
        if expected:
            assert result['backtranslation_recovery']['original_verdict']['equivalent'] is False
            assert result['backtranslation'] == 'The paper proves my point.'
        else:
            assert result['differences'] == failures


@pytest.mark.parametrize('proofs,expected', [([True, True],True),([True,False],False),([None],False)])
def test_reverse_objections_require_every_actual_target_span_not_a_majority(tmp_path,monkeypatch,proofs,expected):
    from core import runtime_context
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runtime_context,'effective_source_language',lambda *_:'en')
    monkeypatch.setattr(runtime_context,'effective_target_language',lambda *_:'zh')
    source,candidate = 'She builds three titles.','她制作三个标题。'
    defects = [{'source_quote':'builds','candidate_quote':candidate,'backtranslation_quote':'creates',
                'difference':'Alleged different action'},
               {'source_quote':'titles','candidate_quote':candidate,'backtranslation_quote':'title',
                'difference':'Alleged different number'}]
    reverse = {'equivalent':False,'differences':defects,'defects':defects,
               'issues':[d['difference'] for d in defects], 'blind_to_source':True,
               'backtranslation':'She creates three title.'}
    monkeypatch.setattr(audit,'ask_gpt',lambda *a,**k:positive())
    monkeypatch.setattr(audit,'reverse_semantic_check',lambda *a:reverse)
    monkeypatch.setattr(audit,'_resolve_source_meanings',lambda *a:None)
    called=[]
    def scoped(s,c,d,*args):
        assert (s,c)==(source,candidate)
        passed = proofs[len(called)]
        called.append(d)
        if passed is None:
            raise audit.IncompleteLLMResponseError('No scoped evidence')
        return {'equivalent':passed,'source_quote':source,'candidate_quote':candidate,
                'candidate_meaning':'She builds three titles.', 'explanation':'Exact action and count comparison'}
    monkeypatch.setattr(audit,'review_scoped_objection',scoped)
    verdict=audit.audit_duration_candidate(source,'',candidate,'{}')
    assert verdict['equivalent'] is expected
    if expected:
        assert audit._reusable_semantic_verdict(verdict,source,candidate,'{}')
        tampered=json.loads(json.dumps(verdict))
        tampered['reverse_objection_review']['objections'].pop()
        assert not audit._reusable_semantic_verdict(tampered,source,candidate,'{}')
        tampered=json.loads(json.dumps(verdict))
        tampered['reverse_objection_review']['objections'][0]['proof']['candidate_quote']='NOT_IN_TARGET'
        assert not audit._reusable_semantic_verdict(tampered,source,candidate,'{}')


def test_auxiliary_schema_recovery_is_blind_and_keeps_exact_evidence(monkeypatch):
    calls=[]
    def reply(prompt,**kwargs):
        calls.append(kwargs['log_title'])
        assert 'SOURCE_SENTINEL' not in prompt
        if len(calls)==1:
            raise audit.IncompleteLLMResponseError('Invalid quotations')
        value={'faithful':False,'differences':[{'candidate_quote':'三个','backtranslation_quote':'two',
                                              'difference':'Changed explicit count'}]}
        assert kwargs['valid_def'](value)['status']=='success'
        fabricated={**value,'differences':[{**value['differences'][0],'candidate_quote':'四个'}]}
        assert kwargs['valid_def'](fabricated)['status']=='error'
        return value
    monkeypatch.setattr(audit,'ask_gpt',reply)
    assert audit.backtranslation_fidelity('保存三个文件。','Save two files.',{})['faithful'] is False
    assert calls==['candidate_backtranslation_fidelity','candidate_backtranslation_fidelity_recovery']


def test_auxiliary_repair_has_only_actual_target_not_the_failed_interpretation():
    target = '第三个工作，其实是动画师。'
    prompt = audit._blind_backtranslation_repair_prompt(target)
    assert target in prompt
    assert '一般现在时is/are' in prompt
    assert '过去、曾经、将来或具体日期' in prompt
    assert 'previous_english' not in prompt
    assert 'target_bound_errors' not in prompt
    assert 'Job three is actually the animator.' not in prompt
    assert 'The third job was actually an animator.' not in prompt


def test_source_skill_name_variant_is_protected_as_a_whole_not_a_platform_verb():
    from core.tts_pronunciation import original_name_conflicts, spoken_plan
    from core.local_translation import TranslationNameGuard
    source = 'Running the skill, YouTube edits, so it finds the skill.'
    assert original_name_conflicts(source,'跑这个技能，YouTube就编辑。','zh')
    names = TranslationNameGuard([source],'zh')
    assert 'YouTube' not in names.lines[0] and 'edits' not in names.lines[0]
    token = names.bindings['1'][0]['token']
    target = names.restore({'1':f'运行{token}技能，所以它找到了技能。'})['1']
    assert 'YouTube edits' in target
    assert original_name_conflicts(source,target,'zh') == []
    assert spoken_plan(target,'zh')['spoken_text'] == '运行优兔剪辑技能，所以它找到了技能。'


def test_explicit_missing_quote_normalization_remains_a_bound_rejection():
    value = {'faithful':False,'differences':[{'candidate_quote':'或许',
        'backtranslation_quote':'无对应词','difference':'Certainty changed'}]}
    normalized = audit._normalize_explicit_missing_quotes(value,'或许证明。','Proves it.')
    assert normalized['faithful'] is False and normalized['differences'][0]['backtranslation_quote'] == ''
    assert audit.validate_backtranslation_fidelity(normalized,'或许证明。','Proves it.')['status'] == 'success'
    assert value['differences'][0]['backtranslation_quote'] == '无对应词'
    fabricated = {**value,'differences':[{**value['differences'][0],'backtranslation_quote':'a fabricated quote'}]}
    assert audit.validate_backtranslation_fidelity(audit._normalize_explicit_missing_quotes(
        fabricated,'或许证明。','Proves it.'),'或许证明。','Proves it.')['status'] == 'error'
    literal = audit._normalize_explicit_missing_quotes(value,'或许证明。','标签为无对应词。')
    assert literal['differences'][0]['backtranslation_quote'] == '无对应词'
    positive = {**value,'faithful':True}
    assert audit._normalize_explicit_missing_quotes(positive,'或许证明。','Proves it.') is positive


@pytest.mark.parametrize('source,index,reference',[
    ('Job three is actually the animator.',3,'the third job'),
    ('Step two is to save it.',2,'the second step'),
    ('Model 3 is new.',None,None),
    ('Three jobs are available.',None,None),
    ('Job 30 is important.',None,None),
])
def test_numbered_label_binds_only_a_common_list_reference(source,index,reference):
    proof = audit._source_numbered_label(source)
    if index is None:
        assert proof is None
    else:
        assert proof['index']==index and proof['ordinal_reference']==reference
        assert proof['source_quote'] in source
        assert proof['scope'].startswith('same numbered list item only')


def test_invalid_reverse_evidence_gets_one_bound_correction_without_passing_error(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from core import ask_gpt as api
    from core.llm_provider import LLMProviderConfig
    provider = LLMProviderConfig(name='unit', model='unit', base_url='http://invalid.test/v1', api_key='')
    monkeypatch.setattr(api, 'get_llm_provider_config', lambda: provider)
    monkeypatch.setattr(api, 'LOG_FOLDER', str(tmp_path / 'logs'))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(api.time, 'sleep', lambda *_: None)
    monkeypatch.setattr(audit, 'ask_gpt', api.ask_gpt)
    outputs = iter([
        'I sing.',
        json.dumps({'equivalent':False,'differences':[{'source_quote':'NOT_THE_SOURCE',
            'backtranslation_quote':'sing','difference':'Unbound source evidence'}]}),
        json.dumps({'equivalent':False,'differences':[{'source_quote':'Save',
            'backtranslation_quote':'sing','difference':'Saving changed to singing'}]}),
        json.dumps({'faithful':True,'differences':[]}),
    ])
    requests = []
    def create(**kwargs):
        requests.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason='stop',
            message=SimpleNamespace(content=next(outputs)))])
    monkeypatch.setattr(api, 'create_chat_client', lambda _: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    result = audit.reverse_semantic_check('Save it.', '我唱歌。', '{}', {})
    assert result['equivalent'] is False
    assert len(requests) == 4
    correction = requests[2]['messages']
    assert len(correction) == 1 and correction[0]['role'] == 'user'
    assert '【A 原文：唯一待比较源文】\nSave it.' in correction[0]['content']
    assert 'NOT_THE_SOURCE' not in correction[0]['content']
    assert result['differences'][0]['source_quote'] == 'Save'
    errors = json.loads((tmp_path / 'logs/error.json').read_text())
    assert errors[0]['metadata']['outcome'] == 'validation_rejected'
    assert errors[0]['response']['equivalent'] is False
    assert audit.reverse_semantic_check('Save it.', '我唱歌。', '{}', {})['equivalent'] is False
    assert len(requests) == 4
    failures = json.loads((tmp_path / 'output/log/reverse_audit_request_failures.json').read_text())
    assert len(failures) == 1 and failures[0]['accepted'] is False


def test_numbered_role_comparison_retains_actual_category_and_fact_gates(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    prompts = []
    def reply(prompt, **kwargs):
        prompts.append((kwargs['log_title'], prompt))
        if kwargs['log_title'] == 'candidate_blind_backtranslation':
            return 'The third job is actually an animator.'
        return {'equivalent': True, 'differences': []}
    monkeypatch.setattr(audit, 'ask_gpt', reply)
    result = audit.reverse_semantic_check('Job three is actually the animator.',
        '第三个工作，其实是动画师。', '{}', {})
    assert result['equivalent'] is True
    comparison = prompts[1][1]
    assert 'noun是源文逐字明示的类别' in comparison
    assert 'the/an并不能单独证明汉语改变了指称' in comparison
    assert 'only/this/that限定遗漏' in comparison
    assert '编号数值、名词类别、完整谓语及任何时间/否定等限定若变化仍必须拒绝' in comparison
    # Numbered-list interpretation never enters the blind translator.
    assert 'source_numbered_label' not in prompts[0][1]
