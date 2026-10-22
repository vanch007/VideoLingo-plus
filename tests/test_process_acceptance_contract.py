"""Regression cases discovered by the 2026-10-12 independent code audit."""
import json
from types import SimpleNamespace
import pytest
from pydub import AudioSegment
from pydub.generators import Sine
from core import atomic_dubbing as ad, tts_pronunciation as tp


@pytest.mark.parametrize('text', ['Claude和克劳德都在这儿。',
    'Epidemic Sound和Epidemic Sounds都在这儿。'])
def test_forward_pronunciation_allows_aliases_and_existing_chinese(text):
    spoken = tp.spoken_plan(text, 'zh')
    request = SimpleNamespace(text=text, metadata={'synthesis_text': spoken['spoken_text'],
        'spoken_transliterations': spoken['matched_transliterations']})
    assert tp.synthesis_text(request) == spoken['spoken_text']
    request.metadata['synthesis_text'] += '新增内容'
    with pytest.raises(ValueError):
        tp.synthesis_text(request)


def test_names_are_checked_independently_of_sentence_similarity():
    expected = '请用Claude完成这里的所有工作，完成后记得检查结果。'
    assert tp.spoken_name_issues(expected, '请用克洛德完成这里的所有工作，完成后记得检查结果。', 'zh')
    assert not tp.spoken_name_issues(expected, '请用克劳德完成这里的所有工作，完成后记得检查结果。', 'zh')
    assert tp.original_name_conflicts('Claude writes.', '克劳德写。', 'zh')
    assert not tp.original_name_conflicts('Claude writes.', 'Claude写。', 'zh')
    normalized, proof = tp.canonical_name_spellings('Claude writes.', '克劳德写。', 'zh')
    assert normalized == 'Claude写。' and proof
    assert tp.canonical_name_spellings('Claude和克劳德写。', '克劳德和克劳德写。', 'zh')[1] == []
    assert tp.canonical_name_spellings('Epidemic Sound and Epidemic Sounds.', '艾皮戴米克桑德和艾皮戴米克桑德。', 'zh')[1] == []


def test_registered_name_case_is_restored_from_source_before_strict_validation():
    from core import local_translation as lt
    source = 'Hyperframes works with Claude Code and Nvidia.'
    draft = 'HyperFrames配合claude code和NVIDIA使用。'
    normalized, receipts = tp.canonical_name_spellings(source, draft, 'zh')
    assert normalized == 'Hyperframes配合Claude Code和Nvidia使用。'
    assert len(receipts) == 3
    assert all('case_only' in r['basis'] for r in receipts)
    assert tp.original_name_conflicts(source, draft, 'zh')
    assert not tp.original_name_conflicts(source, normalized, 'zh')
    assert lt.validate_translation_rows({'1': draft}, [source], None, 'zh')['status'] == 'success'
    budgets = [{'target_speech_seconds': 4., 'max_natural_speech_seconds': 4.32,
                'max_speed_factor': 1.08, 'target_max_words': 12}]
    prompt = lt.spoken_translation_prompt([source], 'Glossary: HyperFrames', 'zh', budgets)
    assert '源文拼写优先于上下文术语表' in prompt


@pytest.mark.parametrize('source,draft', [
    ('Claude and Claude.', 'Claude来了。'),
    ('Claude.', 'Claude和Claude来了。'),
    ('Hyperframes.', 'Hyperframe来了。'),
    ('Hyperframes.', 'HyperframesPlus来了。'),
    ('Hyperframes and HyperFrames.', 'HyperFrames和HyperFrames。'),
    ('Claude.', 'Claude和HeyGen来了。'),
])
def test_name_canonicalization_cannot_manufacture_missing_or_ambiguous_names(source, draft):
    normalized, receipts = tp.canonical_name_spellings(source, draft, 'zh')
    assert normalized == draft
    assert not receipts
    assert tp.original_name_conflicts(source, normalized, 'zh')


def test_longest_registered_names_do_not_create_false_component_occurrences():
    source = 'Claude Code uses Claude.'
    draft = 'CLAUDE CODE使用CLAUDE。'
    normalized, _ = tp.canonical_name_spellings(source, draft, 'zh')
    assert normalized == 'Claude Code使用Claude。'
    assert not tp.original_name_conflicts(source, normalized, 'zh')


def test_translation_name_placeholders_bind_each_occurrence_and_source_spelling(tmp_path, monkeypatch):
    from core import local_translation as lt
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(lt, 'effective_target_language', lambda *_: 'zh')
    lines = ['Hyperframes works.', 'Epidemic sounds and Epidemic sounds.']
    names = lt.TranslationNameGuard(lines, 'zh')
    first = names.bindings['1'][0]['token']
    a, b = [x['token'] for x in names.bindings['2']]
    raw = {'1': first + '有用了。', '2': a + '和' + b + '。'}
    decoded = names.restore(raw)
    assert decoded == {'1': 'Hyperframes有用了。', '2': 'Epidemic sounds和Epidemic sounds。'}
    def infer(prompt, **kwargs):
        assert first in prompt and a in prompt and b in prompt
        assert 'HyperFrames' not in prompt and 'Epidemic Sound' not in prompt
        assert '__VLREF_' in prompt  # context keeps referents without a copyable alias
        assert kwargs['valid_def'](raw)['status'] == 'success'
        assert kwargs['valid_def']({'1': 'HyperFrames有用了。', '2': 'Epidemic Sound和Epidemic Sound。'})['status'] == 'error'
        return raw
    monkeypatch.setattr(lt, 'ask_gpt', infer)
    target, original = lt.translate_localized('\n'.join(lines), 'Glossary: HyperFrames; Epidemic Sound')
    assert target == '\n'.join(decoded.values()) and original == '\n'.join(lines)
    assert '__VLNAME_' not in target
    assert names.validate({'1': first + first, '2': a + b}, lambda _: {'status': 'success'})['status'] == 'error'
    assert names.validate({'1': a, '2': first + b}, lambda _: {'status': 'success'})['status'] == 'error'
    assert names.validate({'1': first, '2': a + b + '__VLNAME_forged__'}, lambda _: {'status': 'success'})['status'] == 'error'
    assert names.validate({'1': first, '2': a}, lambda _: {'status': 'success'})['status'] == 'error'
    assert names.validate({'1': first + '__VLREF_1__', '2': a + b}, lambda _: {'status': 'success'})['status'] == 'error'
    assert names.validate({'1': first + '删除文件。', '2': a + b},
                          lambda _: {'status': 'error', 'message': 'semantic mismatch'})['status'] == 'error'


def test_name_guard_masks_context_and_rejected_drafts_without_changing_restoration():
    from core import local_translation as lt
    source = 'So for this, I have a sound base called Epidemic Sound.'
    guard = lt.TranslationNameGuard([source], 'zh')
    token = guard.bindings['1'][0]['token']
    context = json.dumps({'source_text': source, 'quote': source,
        'terms': [{'src': 'Epidemic Sounds', 'tgt': 'Epidemic Sound',
                   'meaning': 'Paid music and sound-effect library'}],
        'rejected_draft': '我用了Epidemic Sound音乐库。'}, ensure_ascii=False)
    prompt = guard.prompt(lt.localized_translation_prompt(guard.lines, context, 'zh'))
    assert 'Epidemic Sound' not in prompt
    assert 'Paid music and sound-effect library' in prompt
    assert source.replace('Epidemic Sound', '__VLREF_1__') in prompt
    assert token in prompt
    assert guard.restore({'1': f'为此，我用一个叫{token}的音效库。'}) == {
        '1': '为此，我用一个叫Epidemic Sound的音效库。'}
    assert guard.validate({'1': '我用了Epidemic Sound音乐库。'},
                          lambda _: {'status': 'success'})['status'] == 'error'


def test_prompt_reference_aliases_cannot_merge_distinct_source_occurrences():
    from core.local_translation import TranslationNameGuard
    guard = TranslationNameGuard(['Epidemic Sound and Epidemic Sounds.'], 'zh')
    a, b = [entry['token'] for entry in guard.bindings['1']]
    prompt = guard.prompt('Both Epidemic Sound and Epidemic Sounds are discussed.')
    assert '__VLREF_1__' in prompt and '__VLREF_2__' in prompt
    assert guard.restore({'1': a+'和'+b}) == {'1': 'Epidemic Sound和Epidemic Sounds'}
    assert guard.validate({'1': a+a}, lambda _: {'status': 'success'})['status'] == 'error'


def test_registered_source_names_do_not_license_untranslated_component_words():
    from core.local_translation import source_copy_candidates
    terms = [{'src': 'Epidemic sounds', 'tgt': 'Epidemic Sound'}]
    assert source_copy_candidates('Epidemic sounds works.', 'Epidemic sounds有用。', 'zh', terms) == []
    assert source_copy_candidates('Use hyperframes.', '用hyperframes。', 'zh') == []
    assert source_copy_candidates('Epidemic sounds sounds good.', 'Epidemic sounds的sounds不错。', 'zh', terms) == ['sounds']


def test_protected_names_expose_spoken_cost_without_copyable_original_alias():
    from core.local_translation import TranslationNameGuard
    source = 'Epidemic Sound and Epidemic Sound.'
    guard = TranslationNameGuard([source], 'zh', [{'src': 'Epidemic Sound', 'spoken': '甲乙丙丁戊己庚辛壬'}])
    prompt = guard.prompt(source)
    assert 'Epidemic Sound' not in prompt
    assert all(binding['spoken_units'] == 9 for binding in guard.bindings['1'])
    assert prompt.count('"spoken_units": 9') == 2
    assert guard.restore({'1': '和'.join(b['token'] for b in guard.bindings['1'])})['1'] == 'Epidemic Sound和Epidemic Sound'


def test_duration_search_counts_actual_name_transliteration_not_latin_spelling(tmp_path, monkeypatch):
    from core import speech_budget as sb, bilingual_alignment as ba
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sb, 'effective_target_language', lambda *_: 'zh')
    monkeypatch.setattr(sb, 'load_key', lambda key, default=None: '' if key == 'translation.speech_budget.candidate_provider' else default)
    monkeypatch.setattr(sb, 'audit_duration_candidate', lambda *a: {'equivalent': True})
    monkeypatch.setattr(ba, 'prepare_candidate_alignment', lambda *a: [])
    def reply(prompt, **kwargs):
        assert '约 17 个读音单元' in prompt
        assert '包括专名音译' in prompt and '"spoken_units": 7' in prompt
        assert '专名已占 7 个' in prompt and '约 10 个汉语读音单元' in prompt
        assert '不超过17个汉字' not in prompt
        import re
        token = re.search(r'__VLNAME_[a-f0-9]+_1_1__', prompt).group()
        data = {'1': '所以我有个' + token + '音效库。'}
        assert kwargs['valid_def'](data)['status'] == 'success'
        return data
    monkeypatch.setattr(sb, 'ask_gpt', reply)
    assert sb.retranslate_unit('So for this, I have a sound base called Epidemic Sound.',
        '所以，我用了一个叫Epidemic Sound的声音库。',
        {'unit_id': 'U1', 'target_speech_seconds': 3.04, 'max_speed_factor': 1.08, 'target_max_words': 9},
        '', measured_seconds=3.575) == '所以我有个Epidemic Sound音效库。'


@pytest.mark.parametrize('recovery', ['valid', 'repeated', 'wrong_meaning', 'unavailable'])
def test_duration_no_progress_recovers_from_source_but_never_waives_gates(tmp_path, monkeypatch, recovery):
    from core import speech_budget as sb, bilingual_alignment as ba
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sb, 'effective_target_language', lambda *_: 'zh')
    monkeypatch.setattr(sb, 'load_key', lambda key, default=None: '' if key == 'translation.speech_budget.candidate_provider' else default)
    old, fresh = '打开并存储这个文件。', '打开并保存。'
    def anchored_reply(prompt, **kwargs):
        status = kwargs['valid_def']({'1': old})
        assert status['status'] == 'error'
        if recovery == 'unavailable':
            raise RuntimeError('API unavailable')
        raise RuntimeError('Validation still failed: ' + status['message'])
    monkeypatch.setattr(sb, 'ask_gpt', anchored_reply)
    translated, audited, aligned = [], [], []
    def source_translate(source, target, context, budget):
        translated.append(dict(budget))
        if len(translated) == 1:
            assert 'source_semantic_feedback' not in budget and 'meaning_anchor' not in budget
        assert budget['max_speed_factor'] == 1.08
        return old if recovery == 'repeated' else fresh
    monkeypatch.setattr(ba, 'translate_source_partition', source_translate)
    def audit(*args):
        audited.append(args[2])
        return {'equivalent': recovery == 'valid', 'issues': ['changed meaning']}
    monkeypatch.setattr(sb, 'audit_duration_candidate', audit)
    monkeypatch.setattr(ba, 'prepare_candidate_alignment', lambda unit, context: aligned.append(unit['target']))
    args = ('Open and save this file.', old,
        {'unit_id': 'U1', 'target_speech_seconds': 1., 'max_speed_factor': 1.08, 'target_max_words': 3}, '')
    if recovery == 'valid':
        assert sb.retranslate_unit(*args, measured_seconds=2., measured_candidates=[{'text': old, 'natural_seconds': 2.}]) == fresh
        assert audited == [fresh] and aligned == [fresh]
    else:
        with pytest.raises((ValueError, RuntimeError)):
            sb.retranslate_unit(*args, measured_seconds=2., measured_candidates=[{'text': old, 'natural_seconds': 2.}])
        assert not aligned
    if recovery == 'unavailable':
        assert not translated and not audited
    else:
        assert 1 <= len(translated) <= 3


def test_spoken_policy_changes_invalidate_both_acceptance_and_readback(tmp_path, monkeypatch):
    from core import config_utils, speech_budget
    from core.providers import asr_readback
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ad, 'native_tts_runtime_signature', lambda: None)
    monkeypatch.setattr(speech_budget, 'candidate_model_contract', lambda: None)
    def signatures(name):
        getter = lambda key, default=None: {'Claude': name} if key == 'translation.spoken_transliterations' else default
        monkeypatch.setattr(config_utils, 'load_key', getter)
        monkeypatch.setattr(ad, 'load_key', getter)
        return ad.synthesis_contract_signature(), asr_readback._readback_fingerprint(
            number=1, lines=['Claude写。'], src_lines=['Claude writes.'], language='zh')
    a, b = signatures('克劳德'), signatures('克洛德')
    assert a[0] != b[0] and a[1] != b[1]


def test_spoken_words_map_back_to_exact_display_occurrences():
    plan = tp.spoken_plan('用Claude和克劳德。', 'zh')
    words = [{'word': c, 'start': i*.1, 'end': (i+1)*.1}
             for i,c in enumerate('用克劳德和克劳德')]
    mapped = tp.display_aligned_words(plan, words)
    assert ''.join(w['word'] for w in mapped) == '用Claude和克劳德'
    assert mapped[1] == {'word': 'Claude', 'start': .1, 'end': .4}


def test_non_homophone_mandarin_does_not_get_toneless_fuzzy_boost():
    from core.providers.quality import content_similarity
    assert content_similarity('请你购买这家公司发行的股票。', '请你古迈这家公司发行的股票。') < .88


def test_audible_dsp_overhead_cannot_be_cropped(tmp_path, monkeypatch):
    from core import audio_speed
    raw, fitted = tmp_path/'raw.wav', tmp_path/'fitted.wav'
    Sine(300).to_audio_segment(duration=1080).export(raw, format='wav')
    monkeypatch.setattr(audio_speed, 'adjust_audio_speed', lambda source, output, speed:
        Sine(300).to_audio_segment(duration=1010).export(output, format='wav'))
    measured = ad.fit_take(raw, fitted, 1., max_speed=1.08)
    assert measured['fits'] is False
    assert len(AudioSegment.from_file(fitted)) == 1010


def test_subthreshold_preroll_is_removed_before_readback_and_onset_acceptance(tmp_path):
    raw, fitted = tmp_path/'raw.wav', tmp_path/'fitted.wav'
    preroll = Sine(300).to_audio_segment(duration=290).apply_gain(-40)
    (preroll + Sine(400).to_audio_segment(duration=700)).export(raw, format='wav')
    measured = ad.fit_take(raw, fitted, 1., max_speed=1.08)
    assert measured['fits'] and measured['acoustic_onset_seconds'] <= .12
    assert measured['natural_seconds'] < .8


def test_wide_quote_requires_independent_objection_evidence(tmp_path, monkeypatch):
    from core import duration_candidate_audit as audit, runtime_context
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runtime_context, 'effective_source_language', lambda *_: 'en')
    monkeypatch.setattr(runtime_context, 'effective_target_language', lambda *_: 'zh')
    candidate = '我今天在这里边唱歌边停下来继续解释这个流程。'
    rejected = {'equivalent': False, 'issues': ['Added singing'], 'defects': [
        {'source_quote': 'stop', 'candidate_quote': candidate, 'difference': 'Added singing'}]}
    monkeypatch.setattr(audit, 'ask_gpt', lambda *a, **kw: rejected)
    monkeypatch.setattr(audit, '_resolve_source_meanings', lambda *a: None)
    monkeypatch.setattr(audit, 'reverse_semantic_check', lambda *a:
        {'equivalent': True, 'issues': [], 'defects': []})
    calls = []
    def scoped(*args):
        calls.append(args)
        return rejected
    monkeypatch.setattr(audit, 'review_scoped_objection', scoped)
    result = audit.audit_duration_candidate('I stop.', '', candidate, '{}')
    assert not result['equivalent'] and len(calls) == 1


def test_semantic_receipt_is_stable_when_acoustic_clocks_are_added(tmp_path, monkeypatch):
    from core import bilingual_alignment as ba
    from core.llm_provider import LLMProviderConfig
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ba, 'get_llm_provider_config', lambda: LLMProviderConfig('local', 'index', 'http://local', ''))
    u = {'unit_id': 'L1', 'parent_line_id':'L1', 'parent_source':'Claude writes.', 'source': 'Claude writes.', 'target': 'Claude写。',
         'start': 0., 'end': 1., 'duration':1., 'source_duration':1.}
    pair = {'source': u['source'], 'target': u['target'], 'semantic_review': {'equivalent': True, 'issues': []}}
    monkeypatch.setattr(ba, 'prepare_candidate_alignment', lambda *a, **k: [pair])
    ba.ensure_bilingual_alignment({'units': [u]})
    before = ba.ALIGNMENT_PATH.read_bytes()
    u.update(target_words=[{'word':'Claude写', 'start':0.,'end':1.}],
        source_start=0., source_end=1., source_words=[{'word':'Claude writes', 'start':0.,'end':1.}])
    ba.ensure_bilingual_alignment({'units': [u]})
    assert ba.ALIGNMENT_PATH.read_bytes() == before


def test_after_effects_uses_full_mandarin_name_without_ambiguous_initials():
    from core.tts_pronunciation import spoken_plan, spoken_name_issues
    text = '以前都用After Effects做。'
    assert spoken_plan(text, 'zh')['spoken_text'] == '以前都用阿夫特伊费克茨做。'
    assert spoken_name_issues(text, '以前都用阿夫特一费克词做。', 'zh') == []
    assert spoken_name_issues(text, '以前都用A一做。', 'zh')


def test_native_pinyin_annotations_preserve_spoken_surface_and_tones():
    from core.tts_pronunciation import spoken_plan, annotate_mandarin_spans, synthesis_text
    from core.providers.contracts import TTSRequest
    text = '用After Effects做。'
    plan = spoken_plan(text, 'zh')
    annotated, receipts = annotate_mandarin_spans(plan)
    assert receipts[0]['phones'] == ['A1','FU1','TE4','YI1','FEI4','KE4','CI2']
    req = TTSRequest(text=text, output_path='test.wav', metadata={
        'synthesis_text':annotated, 'spoken_transliterations':plan['matched_transliterations']})
    assert '<费|FEI4>' in synthesis_text(req)


def test_chinese_profile_is_shared_without_changing_explicit_or_other_languages():
    from core.providers.mlx_tts import resolve_indextts_config
    cfg = {'profile':'auto','chinese_profile':'standard'}
    assert resolve_indextts_config(cfg,'zh')['profile'] == 'standard'
    assert resolve_indextts_config(cfg,'en')['profile'] == 'auto'
    assert resolve_indextts_config({**cfg,'profile':'v25'},'zh')['profile'] == 'v25'
    assert resolve_indextts_config({**cfg,'model':'chosen-model'},'zh')['profile'] == 'auto'


def test_selected_chinese_checkpoint_is_required_by_environment_check(tmp_path, monkeypatch):
    from core.providers.mlx_tts import IndexTTS2Backend
    from core import runtime_context
    monkeypatch.setattr(runtime_context,'effective_target_language',lambda *_:'zh')
    backend = IndexTTS2Backend({'root':str(tmp_path),'profile':'auto','chinese_profile':'standard'})
    assert not backend.health().ok
    model = tmp_path/'models/mlx-indexTTS2-standard-8bit'
    model.mkdir(parents=True)
    for n in ('config.json','config.yaml','gpt.safetensors','s2mel.safetensors','bigvgan.safetensors'):
        (model/n).write_bytes(b'presence-only-health-fixture')
    assert backend.health().ok
