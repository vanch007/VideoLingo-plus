"""Regression witnesses for wrong-language drafts and bound speech recovery."""
from types import SimpleNamespace
from pathlib import Path
import json

import pytest

from core import translation_review as review
from core import tts_pronunciation as pronunciation
from core import speech_budget as budget


@pytest.mark.parametrize('text', ['初稿が来たら、', '中文ｶﾅ', '初稿입니다', '初稿カット'])
def test_shared_han_is_not_proof_of_chinese(text):
    from core.local_translation import validate_translation_rows
    assert not review.is_valid_target_language(text, 'zh')
    assert validate_translation_rows({'1':text}, ['When the first edit arrives.'], None, 'zh')['status'] == 'error'


@pytest.mark.parametrize('text', ['初稿出来时，', 'Claude Code能在本地工作。', '第一版剪輯出來時。'])
def test_chinese_keeps_latin_display_names_and_traditional_han(text):
    assert review.is_valid_target_language(text, 'zh-TW')


def test_standard_pinyin_only_encodes_registered_sounds_and_keeps_display_spelling():
    text = '用Claude Code检查Parakeet。'
    plan = pronunciation.spoken_plan(text, 'zh')
    spoken, receipts = pronunciation.annotate_standard_mandarin_spans(plan)
    assert 'KE4 LAO2 DE2 DAI4 MA3' in spoken
    assert 'PA4 LA1 JI1 TE4' in spoken
    request = SimpleNamespace(text=text, metadata={'synthesis_text':spoken,
        'spoken_plan':plan, 'spoken_transliterations':plan['matched_transliterations'],
        'spoken_pinyin':receipts, 'standard_mandarin_guidance':True})
    assert pronunciation.synthesis_text(request).strip() == spoken.strip()
    assert request.text == text
    request.metadata['synthesis_text'] = spoken.replace('DE2', 'DE5')
    with pytest.raises(ValueError, match='name sounds'):
        pronunciation.synthesis_text(request)


def test_standard_guidance_cannot_edit_words_outside_names():
    plan = pronunciation.spoken_plan('Claude保存文件。', 'zh')
    spoken, receipts = pronunciation.annotate_standard_mandarin_spans(plan)
    request = SimpleNamespace(text=plan['display_text'], metadata={'synthesis_text':spoken.replace('保存', '删除'),
        'spoken_plan':plan, 'spoken_transliterations':plan['matched_transliterations'],
        'spoken_pinyin':receipts, 'standard_mandarin_guidance':True})
    with pytest.raises(ValueError, match='other wording'):
        pronunciation.synthesis_text(request)


def test_measured_chinese_compaction_rejects_a_costlier_new_script(monkeypatch):
    monkeypatch.setattr(budget, 'effective_target_language', lambda *_:'zh')
    monkeypatch.setattr(budget, 'audit_duration_candidate', lambda *a:{'equivalent':True})
    from core import bilingual_alignment
    monkeypatch.setattr(bilingual_alignment, 'prepare_candidate_alignment', lambda *a:[])
    def reply(prompt, **kwargs):
        assert kwargs['valid_def']({'1':'所以我当时做的是，让它来保存文件。'})['status'] == 'error'
        assert kwargs['valid_def']({'1':'我让它保存文件。'})['status'] == 'success'
        return {'1':'我让它保存文件。'}
    monkeypatch.setattr(budget, 'ask_gpt', reply)
    assert budget.retranslate_unit('I got it to save the file.', '所以我做的是，让它保存文件。',
        {'unit_id':'U1','target_speech_seconds':2.,'target_max_words':4,'max_speed_factor':1.08}, '',
        measured_seconds=3.) == '我让它保存文件。'


def test_complete_duration_witness_can_recover_from_the_whole_source_without_failed_draft(monkeypatch):
    from core import bilingual_alignment
    monkeypatch.setattr(budget, 'effective_target_language', lambda *_: 'zh')
    monkeypatch.setattr(budget, 'load_key', lambda key, default=None:
                        True if key == 'translation.speech_budget.semantic_candidate_gate' else default)
    monkeypatch.setattr(budget, 'audit_duration_candidate', lambda *_: {'equivalent': True})
    monkeypatch.setattr(bilingual_alignment, 'prepare_candidate_alignment', lambda *_: [])
    old = '剪辑、缩放、布局、排版，等等。'
    short = '剪辑、缩放、布局、排版等。'
    calls = []
    def reply(prompt, **kwargs):
        calls.append(kwargs['log_title'])
        if kwargs['response_json']:
            verdict = kwargs['valid_def']({'1': old})
            assert verdict['status'] == 'error'
            raise ValueError('Validation still failed: ' + verdict['message'])
        assert old not in prompt
        assert '真实配音实测' in prompt and '保留每个列举项' in prompt
        assert kwargs['valid_def'](short)['status'] == 'success'
        return short
    monkeypatch.setattr(budget, 'ask_gpt', reply)
    result = budget.retranslate_unit('The cuts and zooms and layout and formatting, yada yada.', old,
        {'unit_id': 'U1', 'target_speech_seconds': 2.94, 'target_max_words': 4,
         'duration_witness': {'content_complete': True}}, '{}', measured_seconds=3.634)
    assert result == short
    assert calls == ['translate_duration_feedback', 'translate_source_only_recovery']


@pytest.mark.parametrize('source,current,short', [
    ('The cuts, zooms, layout, formatting, yada yada.', '剪辑、缩放、布局、排版，等等。', '剪辑、缩放、布局、排版等。'),
    ("And for Brand Name, it's the same thing.", '至于Brand Name，也是一样的操作。', 'Brand Name也一样。'),
])
def test_source_bound_compact_framing_keeps_all_items_and_names(source, current, short):
    assert budget.compact_chinese_framing(source, current) == [short]
    assert not budget.compact_chinese_framing('Wait, wait, I am not done.', current)
    assert not budget.compact_chinese_framing('For Brand Name it is a different operation.', current)


def test_compact_framing_is_only_a_proposal_and_cannot_skip_source_or_display_review(monkeypatch):
    from core import bilingual_alignment
    monkeypatch.setattr(budget, 'effective_target_language', lambda *_: 'zh')
    old, short = '剪辑、缩放、布局、排版，等等。', '剪辑、缩放、布局、排版等。'
    calls = []
    monkeypatch.setattr(budget, 'audit_duration_candidate',
                        lambda *_: calls.append('source') or {'equivalent': True})
    monkeypatch.setattr(bilingual_alignment, 'prepare_candidate_alignment',
                        lambda *_: calls.append('display') or [])
    monkeypatch.setattr(budget, 'ask_gpt', lambda *_args, **_kwargs: pytest.fail('No translation request is needed'))
    assert budget.retranslate_unit('The cuts, zooms, layout, formatting, yada yada.', old,
        {'unit_id': 'U1', 'target_speech_seconds': 2.94, 'target_max_words': 4,
         'duration_witness': {'content_complete': True}}, '{}', measured_seconds=3.634) == short
    assert calls == ['source', 'display']


def test_rejected_compact_framing_never_returns_as_accepted(monkeypatch):
    from core import bilingual_alignment
    monkeypatch.setattr(budget, 'effective_target_language', lambda *_: 'zh')
    monkeypatch.setattr(budget, 'load_key', lambda _key, default=None: default)
    monkeypatch.setattr(budget, 'audit_duration_candidate', lambda *_: {'equivalent': False})
    monkeypatch.setattr(bilingual_alignment, 'prepare_candidate_alignment', lambda *_: pytest.fail('Rejected source cannot reach display'))
    monkeypatch.setattr(budget, 'ask_gpt', lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError('Fallback unavailable')))
    with pytest.raises(RuntimeError, match='Fallback unavailable'):
        budget.retranslate_unit('The cuts, zooms, layout, formatting, yada yada.', '剪辑、缩放、布局、排版，等等。',
            {'unit_id': 'U1', 'target_speech_seconds': 2.94, 'target_max_words': 4,
             'duration_witness': {'content_complete': True}}, '{}', measured_seconds=3.634)


def test_repeated_name_failure_allows_source_rewording_but_not_duration_calibration(monkeypatch):
    from core import atomic_dubbing, bilingual_alignment
    reasons = ['readback_below_threshold', 'spoken_name_mismatch:registered-name']
    recent = [{'readback': text, 'readback_signature': 'current-audio', 'content_issues': reasons}
              for text in ['Named tool does the same operation.', 'Named tools does the same operation.']]
    assert atomic_dubbing._repeated_delivery_failure(recent, reasons)
    for bad in ('empty_or_unavailable_readback', 'reference_leak', 'unexpected_boundary_speech'):
        assert not atomic_dubbing._repeated_delivery_failure(recent, reasons + [bad])
    assert not atomic_dubbing._repeated_delivery_failure(recent[:1], reasons)
    monkeypatch.setattr(budget, 'effective_target_language', lambda *_: 'zh')
    monkeypatch.setattr(budget, 'audit_duration_candidate', lambda *_: {'equivalent': True})
    monkeypatch.setattr(bilingual_alignment, 'prepare_candidate_alignment', lambda *_: [])
    monkeypatch.setattr(budget, 'ask_gpt', lambda *_a, **_kw: pytest.fail('Source framing proposal needs no translation request'))
    assert budget.retranslate_unit("And for Brand Name, it's the same thing.",
        '至于Brand Name，也是一样的操作。',
        {'unit_id': 'U1', 'target_speech_seconds': 1.76, 'target_max_words': 4,
         'speech_delivery_feedback': {'issues': reasons, 'repeated_readbacks': recent}}, '{}') == 'Brand Name也一样。'


def test_lexical_readback_repair_cannot_recycle_the_script_or_assume_duration(monkeypatch):
    monkeypatch.setattr(budget, 'effective_target_language', lambda *_:'zh')
    monkeypatch.setattr(budget, 'audit_duration_candidate', lambda *a:{'equivalent':True})
    from core import bilingual_alignment
    monkeypatch.setattr(bilingual_alignment, 'prepare_candidate_alignment', lambda *a:[])
    def reply(prompt, **kwargs):
        assert '回读不是正确译文' in prompt
        assert kwargs['valid_def']({'1':'探看剪辑。'})['status'] == 'error'
        return {'1':'检查片段。'}
    monkeypatch.setattr(budget, 'ask_gpt', reply)
    assert budget.retranslate_unit('Inspect the clip.', '探看剪辑。',
        {'unit_id':'U1','target_speech_seconds':2.,'target_max_words':4,'max_speed_factor':1.08,
         'speech_delivery_feedback':{'expected':'探看剪辑。','repeated_readbacks':['看看剪辑。']*2}}, '') == '检查片段。'


@pytest.mark.parametrize('text,value', [('一千四百','1400'),('兩萬零三','20003'),('三点一四','3.14'),
                                      ('两亿三万','200030000'),('1400','1400')])
def test_number_readings_are_exact_across_cjk_and_digit_boundaries(text,value):
    from decimal import Decimal
    from core.providers.quality import spoken_numeric_values
    assert spoken_numeric_values('大概有'+text+'行。') == [Decimal(value)]


def test_number_normalization_does_not_excuse_a_wrong_value():
    from core.providers.quality import content_similarity, spoken_numeric_values
    wanted='我的这个叫优兔剪辑，大概有1400行。'
    heard='我的这个叫优兔剪辑，大概有一千四百行。'
    assert content_similarity(wanted,heard) == 1.0
    assert spoken_numeric_values(wanted) == spoken_numeric_values(heard)
    assert spoken_numeric_values(wanted) != spoken_numeric_values(heard.replace('一千四百','一千五百'))


def test_resume_keeps_the_fastest_complete_same_script_take_for_fresh_gates(tmp_path):
    import pandas as pd
    from core.atomic_dubbing import _preferred_natural_takes
    from core.pipeline.artifact_manifest import hash_path
    paths = []
    for i in range(5):
        path = tmp_path / f'{i}.wav'
        path.write_bytes(f'audio-{i}'.encode())
        paths.append(path)
    def take(i, seconds, *, text='完整列举。', complete=True):
        return {'unit_id':'U1','text':text,'natural_audio_path':str(paths[i]),
                'candidate_sha256':hash_path(paths[i]),'natural_seconds':seconds,
                'content_complete':complete}
    attempts = [take(0,3.1),take(1,2.,complete=False),take(2,4.2),
                take(3,1.,text='删掉一个列举项。'),take(4,.5)]
    paths[4].write_bytes(b'changed audio')
    tasks = pd.DataFrame([{'unit_id':'U1','text':'完整列举。'}])
    assert _preferred_natural_takes(tasks,attempts)['U1'] == attempts[0]
    # No passing readback is invented for a retained, unverified voice.
    fallback = _preferred_natural_takes(tasks,[attempts[1]])['U1']
    assert fallback['content_complete'] is False


def test_duration_calibration_uses_a_complete_bound_current_conditioning_witness(tmp_path):
    from core.atomic_dubbing import _duration_calibration_take
    from core.pipeline.artifact_manifest import hash_path
    row = SimpleNamespace(unit_id='U1',text='完整列举。')
    voices = []
    for i,seconds in enumerate([3.2,4.1,1.,.9,.8]):
        path = tmp_path/f'{i}.wav'
        path.write_bytes(f'audio-{i}'.encode())
        voices.append({'unit_id':row.unit_id,'text':row.text,'natural_seconds':seconds,
            'natural_audio_path':str(path),'candidate_sha256':hash_path(path),
            'content_complete':True,'readback_signature':'bound-content',
            'conditioning_signature':'current-reference-and-policy'})
    voices[2]['content_complete'] = False
    voices[3]['conditioning_signature'] = 'different-reference'
    Path(voices[4]['natural_audio_path']).write_bytes(b'changed')
    assert _duration_calibration_take(row,voices[1],voices) == voices[0]
    voices[0]['readback_signature'] = None
    assert _duration_calibration_take(row,voices[1],voices) == voices[1]


def _resume_draft_fixture(tmp_path, monkeypatch):
    import pandas as pd
    from core import atomic_dubbing as ad
    from core.utterance_plan import signature
    rows = pd.DataFrame([{'line_id':'U1','unit_id':'U1','text':'Save the file.'}])
    baseline = [{'LineID':'U1','Source':'Save the file.','Translation':'保存文件。'}]
    state = {'input_rows':baseline,'input_signature':signature(baseline),
             'unit_clock_signature':'same-clock',
             'candidates':[{'unit_id':'U1','target':'将文件保存。'}]}
    path = tmp_path/'drafts.json'
    path.write_text(json.dumps(state))
    monkeypatch.setattr(ad,'CANDIDATE_STATE',path)
    monkeypatch.setattr(ad,'load_plan',lambda:{'units':[{'unit_id':'U1','source':'Save the file.'}]})
    monkeypatch.setattr(ad,'unit_clock_signature',lambda _: 'same-clock')
    return ad, rows, state, path


def test_translation_resume_restores_the_bound_initial_anchor_not_acceptance(tmp_path, monkeypatch):
    ad,rows,state,path = _resume_draft_fixture(tmp_path,monkeypatch)
    assert ad.resume_translation_lines(rows) == ['保存文件。']
    assert json.loads(path.read_text()) == state


@pytest.mark.parametrize('change',['source','clock','input_hash','missing_candidate'])
def test_translation_resume_rejects_changed_or_corrupt_lineage(tmp_path,monkeypatch,change):
    ad,rows,state,path = _resume_draft_fixture(tmp_path,monkeypatch)
    if change == 'source':
        rows.loc[0,'text']='Delete the file.'
    elif change == 'clock':
        state['unit_clock_signature']='different-clock'
    elif change == 'input_hash':
        state['input_rows'][0]['Translation']='删除文件。'
    else:
        state['candidates']=[]
    path.write_text(json.dumps(state))
    with pytest.raises(ValueError, match='Cannot resume translation drafts'):
        ad.resume_translation_lines(rows)


def test_normal_translation_resume_skips_primary_calls_but_keeps_every_gate(tmp_path,monkeypatch):
    import pandas as pd
    from core import step4_2_translate_all as step,atomic_dubbing as ad,speech_budget,dialogue_roles,utterance_plan
    monkeypatch.chdir(tmp_path)
    Path('output/log').mkdir(parents=True)
    Path(step.TERMINOLOGY_FILE).write_text(json.dumps({'topic':'save','speaker_profiles':[{'speaker_id':'S01'}]}))
    rows=pd.DataFrame([{'line_id':'U1','unit_id':'U1','text':'Save the file.','speaker':'S01'}])
    metrics=SimpleNamespace(speakers=['S01'],speaker_count=1,coverage=1.)
    monkeypatch.setattr(step,'load_speaker_aware_rows',lambda:(rows,metrics))
    monkeypatch.setattr(step,'asdict',lambda _: {})
    monkeypatch.setattr(step,'load_key',lambda key,default=None:default)
    monkeypatch.setattr(step,'get_translation_strategy',lambda:'localized')
    monkeypatch.setattr(dialogue_roles,'attach_dialogue_roles',lambda rows,*_:rows)
    monkeypatch.setattr(step,'split_speaker_aware_batches',lambda rows,**kw:[{'rows':rows}])
    monkeypatch.setattr(ad,'resume_translation_lines',lambda _:['保存文件。'])
    monkeypatch.setattr(step,'translate_chunk',lambda *_:pytest.fail('Bound resumed drafts must not be translated again'))
    calls=[]
    def gate(name,frame):
        calls.append(name)
        assert frame.Source.tolist()==['Save the file.']
        assert frame.Translation.tolist()==['保存文件。']
        return frame
    monkeypatch.setattr(speech_budget,'guard_translation_context',lambda frame,*_:gate('context',frame))
    monkeypatch.setattr(ad,'translation_speech_preflight',lambda frame,*_:gate('speech_semantics_display',frame))
    monkeypatch.setattr(step,'review_and_repair_translation',lambda frame,*a,**kw:(gate('publication_review',frame),{'status':'pass'}))
    monkeypatch.setattr(ad,'verify_preflight_identity',lambda frame:gate('identity',frame))
    monkeypatch.setattr(utterance_plan,'bind_translation',lambda frame:gate('bind',frame))
    monkeypatch.setattr(step,'is_valid_target_language',lambda *_:True)
    monkeypatch.setattr(step,'record_llm_stage',lambda *a,**kw:None)
    step.translate_all()
    assert calls == ['context','speech_semantics_display','publication_review','identity','bind']
    assert pd.read_excel(step.TRANSLATION_RESULTS_FILE).Translation.tolist()==['保存文件。']
    assert json.loads(Path(step.WORKFLOW_REPORT_FILE).read_text())['primary_draft_reused']


def test_each_name_requires_its_own_exact_successful_asr_witness():
    wanted='hyperframes免费，Parakeet也免费。'
    primary='海珀真免费，帕拉吉特也免费。'
    secondary='海婆针免费，帕拉基特也免费。'
    assert pronunciation.spoken_name_issues(wanted,primary,'zh')
    assert pronunciation.spoken_name_issues(wanted,secondary,'zh')
    witnesses=pronunciation.spoken_name_witnesses(wanted,[
        {'backend':'qwen','status':'ok','transcript':primary},
        {'backend':'whisper','status':'ok','transcript':secondary}], 'zh')
    assert all(w['passed'] and len(w['confirmed_by'])==1 for w in witnesses)
    assert not all(w['passed'] for w in pronunciation.spoken_name_witnesses(wanted,[
        {'backend':'qwen','status':'ok','transcript':primary},
        {'backend':'whisper','status':'fail','transcript':secondary}], 'zh'))
    assert not all(w['passed'] for w in pronunciation.spoken_name_witnesses(wanted,[], 'zh'))
    assert not pronunciation.spoken_name_witnesses('Claude和Claude。',[
        {'backend':'asr','status':'ok','transcript':'克劳德。'}], 'zh')[0]['passed']


def test_name_consumer_rechecks_audio_identity_and_each_raw_witness(tmp_path,monkeypatch):
    import json
    import pandas as pd
    from core import atomic_dubbing, runtime_context
    from core.pipeline.artifact_manifest import hash_path
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runtime_context,'effective_target_language',lambda *_:'zh')
    path=tmp_path/'output/audio/segs/1_0.wav'
    path.parent.mkdir(parents=True)
    path.write_bytes(b'bound current audio')
    wanted='hyperframes免费，Parakeet也免费。'
    primary='海珀真免费，帕拉吉特也免费。'
    evidence={'status':'ok','expected':wanted,'audio_path':'output/audio/segs/1_0.wav',
              'audio_sha256':hash_path(path),'primary_transcript':primary,
              'secondary_readback':{'status':'ok','transcript':'海婆针免费，帕拉基特也免费。'}}
    tasks=pd.DataFrame([{'number':1,'text':wanted,'lines':[wanted],'asr_status':'ok',
                        'asr_transcript':primary,'asr_content_score':.99,
                        'asr_line_results':json.dumps([evidence],ensure_ascii=False)}])
    assert not atomic_dubbing._content_issues(tasks)
    path.write_bytes(b'another audio')
    assert atomic_dubbing._content_issues(tasks)[0] == ['spoken_name_mismatch:帕拉基特']


def test_scoped_objection_retains_enumeration_context_and_exact_quotes(monkeypatch):
    from core import duration_candidate_audit as audit
    source='The cuts, zooms, layout, formatting, yada, yada.'
    candidate='剪辑、缩放、布局、排版，等等。'
    defect={'source_quote':'yada, yada','candidate_quote':'等等','difference':'Possible changed meaning'}
    def reply(prompt,**kwargs):
        assert source in prompt and candidate in prompt
        value={'equivalent':True,'source_quote':'yada, yada','candidate_quote':'等等',
               'candidate_meaning':'The enumeration continues with unspecified additional examples.',
               'explanation':'Both leave the enumeration open in the complete sentence.'}
        assert kwargs['valid_def'](value)['status']=='success'
        assert kwargs['valid_def']({**value,'candidate_quote':'Wait'})['status']=='error'
        return value
    monkeypatch.setattr(audit,'ask_gpt',reply)
    assert audit.review_scoped_objection(source,candidate,defect,'{}',{})['equivalent']
