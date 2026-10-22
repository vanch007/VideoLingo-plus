"""Native translation, complete-duration feedback and same-name cache isolation."""
import json
import os
import pytest

from core import homura_translation as ht
from core import speech_budget as sb
from core import llm_provider as lp


@pytest.fixture(autouse=True)
def isolated_source_category_model(monkeypatch, request):
    from core import duration_candidate_audit as audit
    from core import runtime_context
    monkeypatch.setattr(runtime_context, 'effective_target_language', lambda default='en': 'en')
    if request.node.name != 'test_optional_category_planning_filters_untyped_terms_and_never_bypasses_source_audit':
        monkeypatch.setattr(audit,'source_category_terms',lambda *_:[])


def budget(current="You need to pay."):
    return {"unit_id":"U1", "target_speech_seconds":1., "max_natural_speech_seconds":1.08,
            "max_speed_factor":1.08, "target_max_words":2,
            "duration_witness":{"text":current,"content_complete":True,"natural_seconds":1.8,
                                "natural_audio_sha256":"audio-sha","conditioning_signature":"voice-contract",
                                "readback_signature":"complete-readback"}}


def provider():
    return lp.LLMProviderConfig(name="omlx_homura", model="Index-Homura-9B-8bit",
        base_url="http://127.0.0.1:8000/v1", api_key="", temperature=0, max_tokens=512,
        chat_template_kwargs={"enable_thinking":False}, model_revision="pinned")


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_native_homura_rejects_invalid_integer_budgets(value):
    with pytest.raises(ValueError, match="positive integer"):
        ht.build_homura_prompt("我们去看电影。", value, "")


def test_homura_native_route_cannot_accidentally_replace_general_index_tasks(monkeypatch):
    config={'model':'Index-Homura-9B-8bit','base_url':'http://127.0.0.1:8000/v1','purpose':'speech_duration'}
    monkeypatch.setattr(lp,'load_key',lambda key,default=None:
        'omlx_homura' if key=='llm.provider' else config if key=='llm.providers.omlx_homura' else default)
    with pytest.raises(ValueError,match='general pipeline tasks'):
        lp.get_llm_provider_config()
    assert lp.get_llm_provider_config('omlx_homura').model=='Index-Homura-9B-8bit'


def test_native_prompt_translates_only_source_with_relevant_terms_and_role_hints():
    context=json.dumps({"parent_source":"我们去看电影。你出钱。",
                        "glossary_meanings":[{"src":"电影","tgt":"cinema"},{"src":"钱","tgt":"money"}],
                        "dialogue_role_hints":{"speaker":"friend","advisory":True}})
    prompt=ht.build_homura_prompt("我们去看电影。", 8, context)
    assert "严格控制在 8 个音节" in prompt
    assert "【电影 -> cinema】" in prompt and "钱 -> money" not in prompt
    assert prompt.endswith("\n\n我们去看电影。") and "friend" in prompt
    assert "JSON" not in prompt and "current_draft" not in prompt


def test_homura_omitted_payer_constraint_is_in_native_instructions():
    prompt=ht.build_homura_prompt('给你两千修他的车。',12,'')
    assert '不能添加 I/we give/pay' in prompt and 'you get/receive' in prompt
    explicit=ht.build_homura_prompt('我给你两千修他的车。',12,'')
    assert '不能添加 I/we give/pay' not in explicit


def test_homura_receives_short_question_and_reply_structure_before_speech():
    question = ht.build_homura_prompt('证据呢？', 3, '')
    assert '名词式短问句' in question and '不用扩成 Where' in question
    reply = ht.build_homura_prompt('惊喜嘛。', 2, '')
    assert '极短的回应' in reply and "It's..." in reply


def test_duration_prompts_offer_only_source_bound_ellipsis_and_preserve_glossary_meanings():
    from core.local_translation import spoken_translation_prompt
    context = json.dumps({'surrounding_source': ['请你登记。'],
                          'glossary_meanings': [{'src': '不用登记', 'tgt': 'no need to register'}]})
    native = ht.build_homura_prompt('不用登记，这已经办完了。', 7, context)
    primary = spoken_translation_prompt(['不用登记，这已经办完了。'], context, 'en', [budget()])
    assert '不能为缩短音节退回字面词义' in native
    for prompt in [native, primary]:
        assert '可用上下文明确的 No need' in prompt
        assert '独立断言和重复仍须完整保留' in prompt
    assert '可用上下文明确的 No need' not in ht.build_homura_prompt('不用付钱。', 5, context)
    assert '可用上下文明确的 No need' not in ht.build_homura_prompt('不用登记。', 5, '{}')


def test_copular_replies_and_definition_frames_can_shorten_without_erasing_obligations():
    reply = ht.build_homura_prompt('不用登记，就是合格。', 5, '{}')
    assert "不要增加 It's..., just" in reply
    assert '其他分句的独立意思' in reply
    definition = ht.build_homura_prompt('奖励就是三天以后，我送你两千元，懂了吗？', 18, '{}')
    assert '简短标签加冒号' in definition
    assert '数量和结尾问句仍完整保留' in definition
    assert '简短标签加冒号' not in ht.build_homura_prompt('我就是付钱的人。', 8, '{}')


def test_factual_validation_failure_is_fed_back_at_the_same_length(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    current='You will receive the money to fix his car.'
    replies=iter(['I will pay to fix his car.','You get money to fix his car.'])
    prompts=[]
    monkeypatch.setattr(ht,'ask_gpt',lambda prompt,**kw:prompts.append(prompt) or next(replies))
    monkeypatch.setattr(ht,'english_syllables',lambda _:12)
    def validate(text):
        return {'status':'error','message':'Duration translation changed explicit source facts: unconfirmed_first_person_payer'} if text.startswith('I ') else {'status':'success'}
    result=ht.propose_homura_translation('给你两千修他的车。',current,budget(current),'',provider(),
        validate_text=validate,audit_candidate=lambda *args:{'equivalent':True,'issues':[]})
    assert result=='You get money to fix his car.'
    assert all('7 个音节' in prompt for prompt in prompts)
    assert 'unconfirmed_first_person_payer' in prompts[1]


@pytest.mark.parametrize("change", ["partial", "other_text", "missing_audio", "missing_condition", "missing_readback"])
def test_unbound_or_incomplete_speech_cannot_calibrate_homura(change):
    data=budget()
    key,value={"partial":("content_complete",False),"other_text":("text","I pay."),
               "missing_audio":("natural_audio_sha256",None),"missing_condition":("conditioning_signature",None),
               "missing_readback":("readback_signature",None)}[change]
    data["duration_witness"][key]=value
    with pytest.raises(ValueError, match="complete, bound"):
        ht.syllable_targets("You need to pay.",data)


def test_homura_tries_finite_longest_first_targets_from_complete_speech(monkeypatch):
    monkeypatch.setattr(ht,"english_syllables",lambda text:12)
    assert ht.syllable_targets("You need to pay.",budget()) == [7,6,5]


def test_chinese_homura_calibrates_actual_spoken_names_and_requires_complete_witness():
    current = '至于Epidemic sounds，也一样。'
    data = budget(current)
    data['max_natural_speech_seconds'] = 1.9
    data['duration_witness']['natural_seconds'] = 2.4
    assert ht.syllable_targets(current, data, target_language='zh') == [9, 8, 7]
    data['duration_witness']['content_complete'] = False
    with pytest.raises(ValueError, match='complete, bound'):
        ht.syllable_targets(current, data, target_language='zh')
    assert '翻译为简体中文' in ht.build_homura_prompt('It is the same.', 9, '', target_language='zh')


@pytest.mark.parametrize('alteration', ['none', 'missing', 'duplicate', 'wrong_meaning'])
def test_chinese_homura_name_identity_and_semantics_still_gate_candidates(tmp_path, monkeypatch, alteration):
    import re
    monkeypatch.chdir(tmp_path)
    current = '至于Epidemic sounds，也一样。'
    data = budget(current)
    audited = []
    def reply(prompt, **kw):
        assert '翻译为简体中文' in prompt and 'Epidemic sounds' not in prompt
        assert '"spoken_units": 7' in prompt and kw['response_json'] is False
        token = re.search(r'__VLNAME_[a-f0-9]+_1_1__', prompt).group()
        return token + '也一样。' if alteration in {'none', 'wrong_meaning'} else '也一样。' if alteration == 'missing' else token + token + '也一样。'
    monkeypatch.setattr(ht, 'ask_gpt', reply)
    def audit(source, anchor, candidate, context):
        audited.append(candidate)
        assert '__VLNAME_' not in candidate and 'Epidemic sounds' in candidate
        return {'equivalent': alteration != 'wrong_meaning', 'issues': ['different action']}
    args = ('And for Epidemic sounds, it is the same.', current, data, '', provider())
    if alteration == 'none':
        assert ht.propose_homura_translation(*args, target_language='zh',
            validate_text=lambda _: {'status': 'success'}, audit_candidate=audit) == 'Epidemic sounds也一样。'
    else:
        with pytest.raises(ht.NoSafeHomuraCandidate):
            ht.propose_homura_translation(*args, target_language='zh',
                validate_text=lambda _: {'status': 'success'}, audit_candidate=audit)
    assert bool(audited) == (alteration in {'none', 'wrong_meaning'})


def test_source_changed_homura_candidate_is_rejected_before_semantic_or_audio_acceptance(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sb,"effective_target_language",lambda *_:"en")
    monkeypatch.setattr(sb,"load_key",lambda key,default=None:
        "omlx_homura" if key=="translation.speech_budget.candidate_provider" else default)
    monkeypatch.setattr(lp,"get_llm_provider_config",lambda *_:provider())
    replies=iter(["I pay.","You pay."])
    calls=[]
    def reply(prompt, **kwargs):
        calls.append((prompt,kwargs))
        assert kwargs["response_json"] is False
        return next(replies)
    monkeypatch.setattr(ht,"ask_gpt",reply)
    checked=[]
    monkeypatch.setattr(sb,"audit_duration_candidate",lambda source,anchor,candidate,context:
        checked.append(candidate) or {"equivalent":True,"issues":[]})
    result=sb.retranslate_unit("你出钱。","You need to pay.",budget(),"",measured_seconds=1.8)
    assert result=="You pay." and checked==["You pay."]
    assert len(calls)==2 and all("你出钱。" in p for p,_ in calls)
    records=json.loads((tmp_path/"output/log/translation_duration_candidate_provenance.json").read_text())["candidates"]
    assert records[0]["status"]=="rejected" and records[1]["status"]=="awaiting_measured_speech"
    assert records[1]["model"]=="Index-Homura-9B-8bit"


def test_contextually_wrong_homura_draft_uses_primary_source_fallback_not_json_homura(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    current="Do you need me to explain what it means? It is a surprise."
    data=budget(current);data.update(target_speech_seconds=3.,max_natural_speech_seconds=3.24,target_max_words=8)
    data["duration_witness"]["natural_seconds"]=6.
    monkeypatch.setattr(sb,"effective_target_language",lambda *_:"en")
    monkeypatch.setattr(sb,"load_key",lambda key,default=None:
        {"translation.speech_budget.candidate_provider":"omlx_homura",
         "translation.speech_budget.semantic_candidate_gate":True}.get(key,default))
    monkeypatch.setattr(lp,"get_llm_provider_config",lambda *_:provider())
    monkeypatch.setattr(ht,"ask_gpt",lambda *a,**kw:"No translation needed. It's a surprise.")
    monkeypatch.setattr(sb,"audit_duration_candidate",lambda s,a,c,ctx:
        {"equivalent":"translation" not in c,"issues":["wrong speech act"] if "translation" in c else []})
    primary=[]
    def fallback(prompt,**kwargs):
        assert kwargs.get("provider_override") is None and kwargs["response_json"] is True
        primary.append(prompt)
        return {"1":"Need explaining? It's a surprise."}
    monkeypatch.setattr(sb,"ask_gpt",fallback)
    assert sb.retranslate_unit("这还用翻译？就是惊喜。",current,data,"",measured_seconds=6.) == "Need explaining? It's a surprise."
    assert len(primary)==1
    records=json.loads((tmp_path/"output/log/translation_duration_candidate_provenance.json").read_text())["candidates"]
    assert records and all(r["status"]=="rejected" for r in records)


def test_semantic_failure_adds_bound_meaning_constraints_without_tightening_syllable_pressure(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ht,'english_syllables',lambda text:12)
    replies=iter(['They kidnapped the rich.','They kidnapped the gentry.'])
    prompts=[]
    monkeypatch.setattr(ht,'ask_gpt',lambda prompt,**kw:prompts.append(prompt) or next(replies))
    def audit(source,anchor,candidate,context):
        if 'rich' in candidate:
            return {'equivalent':False,'issues':['Lost social rank'],
                    'source_constraints':[{'src':'豪绅','tgt':'gentry'}]}
        return {'equivalent':True,'issues':[],'source_constraints':[]}
    current='They kidnapped the wealthy gentry.'
    result=ht.propose_homura_translation('绑架豪绅。',current,budget(current),'',provider(),
        validate_text=lambda text:{'status':'success'},audit_candidate=audit)
    assert result=='They kidnapped the gentry.'
    assert all('7 个音节' in prompt for prompt in prompts)
    assert '豪绅 -> gentry' not in prompts[0] and '豪绅 -> gentry' in prompts[1]


def test_homura_reuses_source_category_plan_from_before_synthesis(tmp_path,monkeypatch):
    from core import duration_candidate_audit as audit
    monkeypatch.chdir(tmp_path)
    current='They kidnapped the powerful landed gentry.'
    data=budget(current);data['source_category_terms']=[{'src':'豪绅','tgt':'gentry'}]
    monkeypatch.setattr(sb,'effective_target_language',lambda *_:'en')
    monkeypatch.setattr(sb,'load_key',lambda key,default=None:
        'omlx_homura' if key=='translation.speech_budget.candidate_provider' else default)
    monkeypatch.setattr(lp,'get_llm_provider_config',lambda *_:provider())
    monkeypatch.setattr(audit,'source_category_terms',lambda *_:pytest.fail('Bound source terms should be reused'))
    prompts=[]
    monkeypatch.setattr(ht,'ask_gpt',lambda prompt,**kw:prompts.append(prompt) or 'They kidnapped the gentry.')
    monkeypatch.setattr(sb,'audit_duration_candidate',lambda *args:{'equivalent':True,'issues':[]})
    assert sb.retranslate_unit('绑架豪绅。',current,data,'',measured_seconds=1.8)=='They kidnapped the gentry.'
    assert '豪绅 -> gentry' in prompts[0]


def test_semantic_meaning_feedback_cannot_import_neighbouring_facts_or_overwrite_glossary():
    from core.duration_candidate_audit import validate_verdict
    verdict={'equivalent':False,'issues':['Lost meaning'],
             'source_constraints':[{'src':'邻句里他出钱','tgt':'He pays.'}]}
    assert validate_verdict(verdict,source='你出钱。')['status']=='error'
    prompt=ht.build_homura_prompt('鹅城百姓。',8,json.dumps({'glossary_meanings':[{'src':'鹅城','tgt':'Goose Town'}]}),
        source_constraints=[{'src':'鹅城','tgt':'other place'},{'src':'黄老爷','tgt':'Master Huang'}])
    assert '鹅城 -> Goose Town' in prompt and 'other place' not in prompt and '黄老爷 ->' not in prompt


def test_glossary_meaning_loss_cannot_pass_as_style():
    from core.duration_candidate_audit import validate_verdict
    meanings = [{'src': '铁饭碗', 'tgt': 'secure job'}]
    verdict = {'equivalent': True, 'issues': [], 'style_notes': ['Iron bowl can be misleading.'],
               'meaning_checks': [{'src': '铁饭碗', 'meaning': 'secure job', 'preserved': False,
                                   'translation_quote': 'iron bowl'}]}
    assert validate_verdict(verdict, source='他有铁饭碗。', candidate='He has an iron bowl.',
                            meanings=meanings)['status'] == 'error'
    verdict['equivalent'] = False
    verdict['issues'] = ['Lost the stable employment meaning.']
    assert validate_verdict(verdict, source='他有铁饭碗。', candidate='He has an iron bowl.',
                            meanings=meanings)['status'] == 'success'


@pytest.mark.parametrize('change', ['missing', 'unquoted', 'non_boolean'])
def test_glossary_audit_requires_bound_candidate_evidence(change):
    from core.duration_candidate_audit import validate_verdict
    meanings = [{'src': '铁饭碗', 'tgt': 'secure job'}]
    checks = [{'src': '铁饭碗', 'meaning': 'secure job', 'preserved': True,
               'translation_quote': 'stable job'}]
    verdict = {'equivalent': True, 'issues': [], 'meaning_checks': checks}
    assert validate_verdict(verdict, source='他有铁饭碗。', candidate='He has a stable job.',
                            meanings=meanings)['status'] == 'success'
    if change == 'missing':
        verdict['meaning_checks'] = []
    elif change == 'unquoted':
        checks[0]['translation_quote'] = 'secure job'
    else:
        checks[0]['preserved'] = 'true'
    assert validate_verdict(verdict, source='他有铁饭碗。', candidate='He has a stable job.',
                            meanings=meanings)['status'] == 'error'


def test_meaning_checks_ignore_neighbor_terms_and_preserve_canonical_mapping():
    from core.duration_candidate_audit import current_source_meanings
    context = json.dumps({'glossary_meanings': [
        {'src': '铁饭碗', 'tgt': 'secure job', 'note': 'Stable employment.'},
        {'src': '铁饭碗', 'tgt': 'iron bowl'}, {'src': '电影', 'tgt': 'cinema'}]})
    assert current_source_meanings('他有铁饭碗。', context) == [
        {'src': '铁饭碗', 'tgt': 'secure job', 'note': 'Stable employment.'}]


def test_proposed_source_categories_cannot_redefine_approved_meanings():
    from core.duration_candidate_audit import current_source_meanings
    context = sb.with_source_category_terms(json.dumps({'glossary_meanings': [
        {'src': '铁饭碗', 'tgt': 'secure job'}]}), [{'src': '铁饭碗', 'tgt': 'iron bowl'},
                                              {'src': '你这个', 'tgt': 'this one'}])
    assert current_source_meanings('你这个铁饭碗。', context) == [
        {'src': '铁饭碗', 'tgt': 'secure job', 'note': ''}]
    prompt = ht.build_homura_prompt('你这个铁饭碗。', 10, context)
    assert '铁饭碗 -> secure job' in prompt and '铁饭碗 -> iron bowl' not in prompt


def test_optional_category_planning_filters_untyped_terms_and_never_bypasses_source_audit(tmp_path, monkeypatch):
    from core import duration_candidate_audit as audit
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(audit, 'ask_gpt', lambda *args, **kwargs: {'terms': [
        {'src': '医生', 'tgt': 'doctor', 'category_type': 'occupation'},
        {'src': '三天', 'tgt': 'three days'}, {'src': '你', 'tgt': 'you', 'category_type': 'pronoun'}]})
    terms = audit.source_category_terms('你当医生三天。', '')
    assert [term['src'] for term in terms] == ['医生']
    def unavailable(*args, **kwargs):
        raise RuntimeError('Vocabulary planner unavailable')
    monkeypatch.setattr(audit, 'ask_gpt', unavailable)
    assert audit.source_category_terms('你当医生三天。', '') == []
    warnings = json.loads((tmp_path/'output/log/source_category_guidance_warnings.json').read_text())
    assert warnings[-1]['fallback'] == 'whole_source_semantic_audit'


def test_source_semantic_feedback_cannot_cycle_the_same_draft(monkeypatch):
    monkeypatch.setattr(sb, 'effective_target_language', lambda *_: 'en')
    monkeypatch.setattr(sb, 'load_key', lambda key, default=None: default)
    monkeypatch.setattr(sb, 'ask_gpt', lambda *args, **kwargs: {'1': "I'll pay!"})
    data = {'target_speech_seconds': 2., 'max_speed_factor': 1.08, 'target_max_words': 6,
            'source_semantic_feedback': {'equivalent': False, 'issues': ['Wrong payer.']}}
    with pytest.raises(ValueError, match='same draft or punctuation-only'):
        sb.retranslate_unit('你出钱。', "I'll pay.", data, '')


def test_canonical_term_cannot_be_denied_but_does_not_waive_other_source_errors():
    from core.duration_candidate_audit import validate_verdict
    meanings = [{'src': '医生', 'tgt': 'doctor'}]
    verdict = {'equivalent': False, 'issues': ['Changed helping to killing.'],
               'meaning_checks': [{'src': '医生', 'meaning': 'doctor', 'preserved': False,
                                   'translation_quote': 'doctor'}]}
    assert validate_verdict(verdict, source='帮医生。', candidate='Kill the doctor.',
                            meanings=meanings)['status'] == 'error'
    verdict['meaning_checks'][0]['preserved'] = True
    assert validate_verdict(verdict, source='帮医生。', candidate='Kill the doctor.',
                            meanings=meanings)['status'] == 'success'


def test_term_meaning_loss_is_rejected_even_if_whole_sentence_model_would_call_it_style(tmp_path, monkeypatch):
    from core import duration_candidate_audit as audit
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(audit, 'load_key', lambda key, default=None: default)
    calls = []
    def response(prompt, **kwargs):
        calls.append(prompt)
        return {'preserved': False, 'translation_quote': 'iron bowl', 'issue': 'The stable employment meaning was replaced by a physical bowl.'}
    monkeypatch.setattr(audit, 'ask_gpt', response)
    context = json.dumps({'glossary_meanings': [{'src': '铁饭碗', 'tgt': 'secure job', 'note': 'Stable employment.'}]})
    verdict = audit.audit_duration_candidate('他有铁饭碗。', 'old draft', 'He has an iron bowl.', context)
    assert verdict['equivalent'] is False and len(calls) == 1
    assert verdict['source_constraints'] == [{'src': '铁饭碗', 'tgt': 'secure job'}]


def test_exact_terms_are_bound_without_model_echo_and_do_not_certify_the_whole_sentence(tmp_path, monkeypatch):
    from core import duration_candidate_audit as audit
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(audit, 'load_key', lambda key, default=None: default)
    calls = []
    monkeypatch.setattr(audit, 'ask_gpt', lambda prompt, **kwargs:
        calls.append(prompt) or {'equivalent': False, 'issues': ['Changed help to kill.'], 'source_constraints': []})
    context = json.dumps({'glossary_meanings': [{'src': '医生', 'tgt': 'doctor'}]})
    verdict = audit.audit_duration_candidate('帮医生。', 'old', 'Kill the doctor.', context)
    assert not verdict['equivalent'] and len(calls) == 1
    assert verdict['meaning_checks'][0]['preserved'] is True
    assert verdict['meaning_checks'][0]['translation_quote'] == 'doctor'


def test_missing_second_clause_cannot_be_accepted_by_a_whole_sentence_style_verdict(tmp_path, monkeypatch):
    from core import duration_candidate_audit as audit
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(audit, 'load_key', lambda key, default=None: default)
    calls = []
    def response(prompt, **kwargs):
        calls.append(kwargs['log_title'])
        if kwargs['log_title'] == 'duration_source_meaning_reference':
            return "We'll have dinner tonight." if prompt.endswith('今晚吃饭') else 'No explanation needed.'
        return {'preserved': False, 'translation_quote': '', 'issue': 'The dinner promise was deleted.'}
    monkeypatch.setattr(audit, 'ask_gpt', response)
    verdict = audit.audit_duration_candidate('不用解释。今晚吃饭。', 'old', 'No explanation needed.', '{}')
    assert not verdict['equivalent']
    assert verdict['source_constraints'] == [{'src': '今晚吃饭', 'tgt': "We'll have dinner tonight."}]
    assert 'duration_candidate_semantics' not in calls


def test_repeated_source_clauses_need_distinct_candidate_occurrences():
    from core import duration_candidate_audit as audit
    source = '人呢？人呢？人呢？'
    assert audit.source_clause_occurrences(source) == [('人呢', 3)]
    check = {'preserved': True, 'translation_quotes': ['The people?'], 'issue': '', 'repair_meaning': ''}
    assert not audit.bind_clause_occurrences('人呢', 3, check, 'The people? The people?')['preserved']
    assert audit.bind_clause_occurrences('人呢', 3, check, 'The people? The people? The people?')['preserved']
    check['translation_quotes'] = ['The people?', 'People?', 'Where are the people?']
    assert audit.bind_clause_occurrences('人呢', 3, check, 'The people? People? Where are the people?')['preserved']
    check['translation_quotes'] = ['The people? The people? The people?']
    assert audit.bind_clause_occurrences('人呢', 3, check, 'The people? The people? The people?')['preserved']
    check['translation_quotes'] = ['The people? The people?']
    assert not audit.bind_clause_occurrences('人呢', 3, check, 'The people? The people?')['preserved']


def test_bare_hesitation_fragments_do_not_create_object_or_repetition_facts():
    from core import duration_candidate_audit as audit
    assert audit.source_clause_occurrences('这个，你这个……') == []
    assert audit.source_clause_occurrences('走，走，走！') == [('走', 3)]


def test_embedded_repeated_questions_keep_the_occurrence_inside_another_statement():
    from core import duration_candidate_audit as audit
    source = '所有证人都在这里，警卫呢？警卫呢？警卫呢？'
    assert dict(audit.source_clause_occurrences(source))['警卫呢'] == 3
    check = {'preserved': True, 'translation_quotes': ['The guard?'], 'issue': '', 'repair_meaning': ''}
    assert not audit.bind_clause_occurrences('警卫呢', 3, check, 'All witnesses are here. The guard? The guard?')['preserved']
    assert audit.bind_clause_occurrences('警卫呢', 3, check, 'All witnesses are here. The guard? The guard? The guard?')['preserved']


def test_fast_mixed_nominal_calls_remain_compact_before_any_speech_is_generated():
    from core.spoken_length import source_form_conflicts
    source = '全是自己人，警卫呢？警卫呢？警卫呢？'
    assert source_form_conflicts(source, 'All our men. Where are the guards? Where are the guards? Where are the guards?', 3.8)
    assert not source_form_conflicts(source, 'All our men. Guards? Guards? Guards?', 3.8)
    assert not source_form_conflicts(source, 'All our men. Where are the guards? Where are the guards? Where are the guards?', 8.)


def test_breath_commas_and_unfinished_questions_are_not_independent_claims():
    from core import duration_candidate_audit as audit
    assert audit.source_clause_occurrences('什么他妈的叫，他妈的惊喜？') == []
    assert audit.source_clause_occurrences('张师傅，他是你的恩人，你来解释一下，什么叫惊喜？') == []
    assert audit.source_clause_occurrences('不用解释，就是惊喜。') == [('不用解释', 1), ('就是惊喜', 1)]
    assert audit.source_clause_occurrences('不是你的，是我的。') == [('不是你的', 1), ('是我的', 1)]


def test_source_clause_reference_is_derived_without_seeing_the_duration_draft(tmp_path, monkeypatch):
    from core import duration_candidate_audit as audit
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(audit, 'load_key', lambda key, default=None: default)
    candidate = 'UNRELATED_DURATION_DRAFT'
    prompts = []
    def response(prompt, **kwargs):
        prompts.append((kwargs['log_title'], prompt))
        if kwargs['log_title'] == 'duration_source_meaning_reference':
            return 'It is a surprise.'
        return {'preserved': False, 'translation_quote': '', 'issue': 'The surprise meaning is absent.'}
    monkeypatch.setattr(audit, 'ask_gpt', response)
    result = audit._source_clause_check('不用解释，就是惊喜。', candidate, '就是惊喜', 1, '{}', {})
    assert not result['preserved'] and result['source_meaning_reference'] == 'It is a surprise.'
    assert candidate not in prompts[0][1] and 'It is a surprise.' in prompts[1][1]


def test_source_reference_cannot_change_a_verified_source_payer(monkeypatch):
    from core import duration_candidate_audit as audit
    monkeypatch.setattr(audit, 'ask_gpt', lambda *args, **kwargs: 'I will pay.')
    with pytest.raises(ValueError, match='source facts'):
        audit._source_clause_reference('你出钱。', '你出钱', '{}')


def test_source_reference_cannot_import_repetitions_from_the_parent(monkeypatch):
    from core import duration_candidate_audit as audit
    monkeypatch.setattr(audit, 'ask_gpt', lambda *args, **kwargs: 'Where is everyone? Where is everyone?')
    with pytest.raises(ValueError, match='once'):
        audit._source_clause_reference('人呢？人呢？人呢？', '人呢', '{}')


def test_source_repetition_uses_one_lexical_reference_despite_final_punctuation(monkeypatch):
    from core import duration_candidate_audit as audit
    monkeypatch.setattr(audit, 'ask_gpt', lambda *args, **kwargs: "You're pointing a gun at me.")
    source = '你拿枪指着我，你拿枪指着我，你想跟我火拼？'
    candidate = "You're pointing a gun at me! You're pointing a gun at me! You wanna go head-to-head with me?"
    verdict = audit._source_clause_check(source, candidate, '你拿枪指着我', 2, '{}', {})
    assert verdict['preserved'] and verdict['candidate_occurrences'] == 2
    assert verdict['translation_quotes'] == ["You're pointing a gun at me"]


def test_a_repeated_source_span_cannot_be_proven_by_a_whole_candidate_quote(monkeypatch):
    from core import duration_candidate_audit as audit
    def response(prompt, **kwargs):
        value = {'preserved': True, 'translation_quote': 'The men? The men?', 'issue': ''}
        assert kwargs['valid_def'](value)['status'] == 'error'
        return value
    monkeypatch.setattr(audit, 'ask_gpt', response)
    with pytest.raises(ValueError, match='one matching English clause'):
        audit._glossary_meaning_check('人呢', 'The men? The men?',
                                      {'src': '人呢', 'tgt': 'Where is everyone?'}, {}, single_clause=True)


def test_source_span_review_sees_only_its_target_scope_and_cannot_borrow_neighboring_claims(monkeypatch):
    from core import duration_candidate_audit as audit
    candidate = "Big brother, this is a hell of a surprise! I'm willing to wait three days for you."
    reference = "I'm willing to wait for you for three days."
    assert audit._candidate_meaning_excerpt(reference, candidate) == "I'm willing to wait three days for you."
    assert audit._candidate_meaning_excerpt('This is a fucking surprise.', candidate) == 'this is a hell of a surprise!'
    calls = []
    def response(prompt, **kwargs):
        if kwargs['log_title'] == 'duration_source_meaning_reference':
            return reference
        calls.append(prompt)
        assert 'Big brother' not in prompt
        assert 'a hell of a surprise' not in prompt
        return {'preserved': True, 'translation_quote': "I'm willing to wait three days for you.", 'issue': ''}
    monkeypatch.setattr(audit, 'ask_gpt', response)
    verdict = audit._source_clause_check('大哥，这他妈是惊喜啊，小弟我愿意等你三天。',
                                        candidate, '小弟我愿意等你三天', 1, '{}', {})
    assert verdict['preserved'] and len(calls) == 1


def test_target_excerpt_routing_alone_never_certifies_an_omitted_meaning(monkeypatch):
    from core import duration_candidate_audit as audit
    def response(prompt, **kwargs):
        if kwargs['log_title'] == 'duration_source_meaning_reference':
            return "It's a surprise."
        return {'preserved': False, 'translation_quote': '', 'issue': 'The surprise meaning is missing.'}
    monkeypatch.setattr(audit, 'ask_gpt', response)
    verdict = audit._source_clause_check('不用解释，就是惊喜。', 'No need to explain.', '就是惊喜', 1, '{}', {})
    assert not verdict['preserved']


def test_target_scope_preserves_decimal_amounts_as_one_clause():
    from core import duration_candidate_audit as audit
    candidate = 'A surprise! You get 1.8 million taels of silver. Ready?'
    assert audit._candidate_meaning_excerpt('You receive 1.8 million taels of silver.', candidate) == 'You get 1.8 million taels of silver.'


def test_spoken_repetitions_ignore_punctuation_and_case_but_bind_actual_occurrences():
    from core import duration_candidate_audit as audit
    value = {'preserved': True, 'translation_quotes': ['Got it,'], 'issue': '', 'repair_meaning': ''}
    candidate = 'Got it, got it.'
    result = audit.bind_clause_occurrences('了然', 2, value, candidate)
    assert result['preserved'] and result['candidate_occurrences'] == 2
    assert [row['quote'] for row in result['quoted_occurrences']] == ['Got it', 'got it']
    assert not audit.bind_clause_occurrences('了然', 2, value, 'Forgot it. Got it.')['preserved']


def test_a_target_em_dash_can_separate_source_meanings_without_changing_the_candidate():
    from core import duration_candidate_audit as audit
    candidate = "No need—it's a surprise."
    assert audit._candidate_meaning_excerpt('No need to explain.', candidate) == 'No need'
    assert audit._candidate_meaning_excerpt("It's a surprise.", candidate) == "it's a surprise"


def test_need_ellipsis_requires_the_same_explicit_action_in_actual_neighboring_dialogue():
    from core import duration_candidate_audit as audit
    context = json.dumps({'surrounding_source': ['请你登记。']})
    result = audit._contextual_need_ellipsis('不用登记', 'No need to register.', 'No need', context)
    assert result['preserved'] and result['context_witnesses'] == ['请你登记。']
    assert audit._contextual_need_ellipsis('不用登记', 'No need to register.', 'No need', '{}') is None
    assert audit._contextual_need_ellipsis('不用付钱', 'No need to pay.', 'No need', context) is None
    assert audit._contextual_need_ellipsis('不要登记', 'Do not register.', 'No need', context) is None
    assert audit._contextual_need_ellipsis('不用登记', 'No need to register.', 'No need to pay', context) is None
    assert audit._contextual_need_ellipsis('不用登记', 'No need to register.', 'No need?', context) is None


def test_a_neighboring_assertion_cannot_certify_an_omitted_negated_need(monkeypatch):
    from core import duration_candidate_audit as audit
    from core.translation_invariants import source_fact_conflicts
    monkeypatch.setattr(audit, 'ask_gpt', lambda *a, **kw: pytest.fail('Absent negation must be caught before model review'))
    result = audit._glossary_meaning_check('不用登记，就是成功。', 'Success.',
                                         {'src': '不用登记', 'tgt': 'no need to register'}, {})
    assert not result['preserved'] and result['translation_quote'] == ''
    assert 'explicit_negated_need_lost' in source_fact_conflicts('不用登记，就是成功。', 'Success.', 'en')
    for reply in ['No need. Success.', "Don't bother registering. Success.", 'Skip registration. Success.']:
        assert 'explicit_negated_need_lost' not in source_fact_conflicts('不用登记，就是成功。', reply, 'en')


def test_expanding_a_kinship_abbreviation_does_not_erase_its_hierarchy_modifier(monkeypatch):
    from core import duration_candidate_audit as audit
    monkeypatch.setattr(audit, 'ask_gpt', lambda *args, **kwargs: pytest.fail('Exact kinship variant needs no style judgement'))
    term = {'src': '大哥', 'tgt': 'Big Bro'}
    assert audit._glossary_meaning_check('大哥', 'Big brother', term, {})['preserved']
    assert audit._canonical_quote('Bro', 'Big Brother') is None


def test_an_unused_repair_echo_is_not_a_loss_but_a_changed_repair_still_blocks():
    from core import duration_candidate_audit as audit
    value = {'preserved': True, 'translation_quotes': ['You pay!'], 'issue': '', 'repair_meaning': 'You pay.'}
    normalized = audit._single_repeat_quotes(value)
    assert normalized['unused_repair_meaning_echo'] == 'You pay.'
    assert audit.validate_clause_check(normalized, 'You pay!')['status'] == 'success'
    value['repair_meaning'] = 'I pay.'
    assert audit.validate_clause_check(value, 'You pay!')['status'] == 'error'
    value.update(repair_meaning='You pay.', issue='The payer changed.')
    assert audit.validate_clause_check(value, 'You pay!')['status'] == 'error'


def test_source_audit_keeps_old_wording_only_in_lineage_not_in_the_review_prompt(tmp_path,monkeypatch):
    from core import duration_candidate_audit as audit
    monkeypatch.chdir(tmp_path)
    prompts=[]
    monkeypatch.setattr(audit,'load_key',lambda key,default=None:default)
    def response(prompt,**kwargs):
        prompts.append(prompt)
        return {'equivalent':True,'issues':[],'style_notes':['Compact oral phrasing'],'source_constraints':[]}
    monkeypatch.setattr(audit,'ask_gpt',response)
    context=json.dumps({'parent_source':'你出钱。','glossary_meanings':[],
                        'dialogue_role_hints':{'speaker':'UNVERIFIED_ACTOR'},'speech_act':'UNVERIFIED_ACTION'})
    verdict=audit.audit_duration_candidate('你出钱。','UNIQUE_OLD_WORDING','You pay.',context)
    assert verdict['equivalent'] and 'UNIQUE_OLD_WORDING' not in prompts[0]
    assert 'UNVERIFIED_ACTOR' not in prompts[0] and 'UNVERIFIED_ACTION' not in prompts[0]
    record=json.loads((tmp_path/'output/log/duration_candidate_semantic_audit.json').read_text())['candidates'][0]
    assert record['anchor']=='UNIQUE_OLD_WORDING'


def test_same_model_id_with_changed_local_weights_invalidates_requests_and_preflight(tmp_path,monkeypatch):
    from core import atomic_dubbing as ad
    (tmp_path/"config.json").write_text('{}')
    (tmp_path/"tokenizer.json").write_text('{}')
    weights=tmp_path/"model.safetensors";weights.write_bytes(b'first weights')
    config={"model":"Index-Homura-9B-8bit","base_url":"http://127.0.0.1:8000/v1",
            "model_path":str(tmp_path),"model_revision":"same-source"}
    monkeypatch.setattr(lp,"load_key",lambda key,default=None:
        config if key=="llm.providers.omlx_homura" else default)
    monkeypatch.setattr(sb,"load_key",lambda key,default=None:
        "omlx_homura" if key=="translation.speech_budget.candidate_provider" else default)
    before=lp.get_llm_provider_config("omlx_homura")
    before_policy=ad.candidate_policy_signature()
    stat=weights.stat()
    weights.write_bytes(b'other weights')
    assert weights.stat().st_size==stat.st_size
    os.utime(weights,ns=(stat.st_atime_ns,stat.st_mtime_ns))
    after=lp.get_llm_provider_config("omlx_homura")
    assert before.model==after.model and before.model_revision!=after.model_revision
    assert lp.completion_fingerprint(before,False)!=lp.completion_fingerprint(after,False)
    assert before_policy!=ad.candidate_policy_signature()


def test_homura_candidate_reports_failed_then_complete_speech_without_claiming_final_review(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ht._record({'unit_id':'U1', 'target':'You pay.', 'strategy':'homura_native_source_syllables',
                'semantic_verdict':{'equivalent':True}, 'status':'awaiting_measured_speech'})
    ht.record_homura_speech_result('U1', 'You pay.', {'content_complete':False,'fits':True})
    path=tmp_path/'output/log/translation_duration_candidate_provenance.json'
    assert json.loads(path.read_text())['candidates'][0]['status']=='speech_rejected'
    ht.record_homura_speech_result('U1', 'You pay.', {'content_complete':True,'fits':True})
    record=json.loads(path.read_text())['candidates'][0]
    assert record['status']=='speech_pass_before_final_review' and len(record['measured_speech'])==2


@pytest.mark.parametrize('review_status',['failed','needs_review',None])
def test_translation_export_cannot_publish_homura_speech_with_unresolved_final_review(tmp_path,monkeypatch,review_status):
    from types import SimpleNamespace
    from pathlib import Path
    import pandas as pd
    from core import step4_2_translate_all as step
    from core import dialogue_roles, atomic_dubbing
    monkeypatch.chdir(tmp_path)
    Path('output/log').mkdir(parents=True)
    Path('output/log/terminology.json').write_text(json.dumps({'topic':'dialogue','speaker_profiles':[{'speaker_id':'S01'}]}))
    rows=pd.DataFrame([{'line_id':'L1.01','unit_id':'L1.01','speaker':'S01','text':'你出钱。',
                        'parent_source':'你出钱。','parent_line_id':'L1','start':0.,'end':1.,'source_end':1.}])
    monkeypatch.setattr(step,'load_speaker_aware_rows',lambda:(rows,SimpleNamespace(speakers=['S01'])))
    monkeypatch.setattr(dialogue_roles,'attach_dialogue_roles',lambda rows,*_:rows)
    monkeypatch.setattr(step,'split_speaker_aware_batches',lambda rows,**kw:[{'rows':rows}])
    monkeypatch.setattr(step,'translate_chunk',lambda *args:(0,['You pay.']))
    monkeypatch.setattr(atomic_dubbing,'translation_speech_preflight',lambda frame,*_:frame)
    monkeypatch.setattr(step,'review_and_repair_translation',lambda frame,*a,**kw:
        (frame,{'status':review_status,'error':'Unresolved source meaning'}))
    monkeypatch.setattr(atomic_dubbing,'verify_preflight_identity',lambda *_:
        pytest.fail('Unresolved semantic review must stop before translation/audio lock'))
    with pytest.raises(ValueError,match='Translation review gate failed'):
        step.translate_all()
    assert Path('output/log/translation_candidate.xlsx').is_file()
    assert not Path('output/log/translation_results.xlsx').exists()
