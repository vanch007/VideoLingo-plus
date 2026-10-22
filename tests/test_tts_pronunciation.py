from types import SimpleNamespace

import pytest

from core import tts_pronunciation as pronunciation


def test_only_complete_source_bound_transliterations_with_known_readings_get_guidance(monkeypatch):
    monkeypatch.setattr(pronunciation, '_dictionary', lambda: {
        'hu': [['HH', 'UW1']], 'wan': [['W', 'AA1', 'N']]})
    terms = [{'src':'胡万','tgt':'Hu Wan'}, {'src':'黄老爷','tgt':'Master Huang'},
             {'src':'李明','tgt':'Li Ming'}, {'src':'胡万','tgt':'a henchman'}]
    plan = pronunciation.source_pronunciations('胡万在这里。', terms)
    assert len(plan) == 1 and plan[0]['tgt'] == 'Hu Wan'
    assert pronunciation.source_pronunciations('别的人。', terms) == []
    assert pronunciation.source_pronunciations('黄老爷在这里。', terms) == []


def test_annotations_preserve_surface_words_and_do_not_match_inside_another_name():
    entry = {'src':'胡万', 'tgt':'Hu Wan', 'words':['Hu','Wan'], 'phones':[['HH','UW1'],['W','AA1','N']]}
    text = 'Hu Wan saw Huwan and HU WAN.'
    annotated = pronunciation.annotate_pronunciations(text, [entry])
    assert annotated == '<Hu|HH UW1> <Wan|W AA1 N> saw Huwan and <HU|HH UW1> <WAN|W AA1 N>.'
    assert pronunciation.synthesis_text(SimpleNamespace(text=text,metadata={'synthesis_text':annotated})) == annotated


def test_source_bound_compound_guidance_preserves_inflection_and_full_canonical_text(monkeypatch):
    monkeypatch.setattr(pronunciation, '_dictionary', lambda: {
        'four': [['F', 'AO1', 'R']], 'dot': [['D', 'AA1', 'T']], 'masks': [['M', 'AE1', 'S', 'K', 'S']]})
    terms = [{'src': '四点面具', 'tgt': 'four-dot mask'}]
    text = "Four-dot masks! All our men. Bandits? Bandits? Bandits?"
    entries = pronunciation.source_compound_pronunciations('四点面具！都是自己人。', text, terms)
    assert entries[0]['words'] == ['Four', 'dot', 'masks']
    annotated = pronunciation.annotate_pronunciations(text, entries)
    assert annotated.startswith('<Four|F AO1 R>-<dot|D AA1 T> <masks|M AE1 S K S>!')
    assert pronunciation.synthesis_text(SimpleNamespace(text=text,metadata={'synthesis_text':annotated})) == annotated
    assert pronunciation.source_compound_pronunciations('别的东西。', text, terms) == []


def test_joined_romanization_can_use_exact_source_syllables_without_inventing_phones(monkeypatch):
    monkeypatch.setattr(pronunciation, '_dictionary', lambda: {
        'zhang': [['ZH', 'AA1', 'NG']], 'ma': [['M', 'AA1']], 'zi': [['Z', 'IY1']]})
    plan = pronunciation.source_pronunciations('张麻子来了。', [{'src': '张麻子', 'tgt': 'Zhang Mazi'}])
    assert plan[0]['phones'] == [['ZH', 'AA1', 'NG'], ['M', 'AA1', 'Z', 'IY1']]
    text = pronunciation.annotate_pronunciations('Zhang Mazi!', plan)
    assert text == '<Zhang|ZH AA1 NG> <Mazi|M AA1 Z IY1>!'
    assert pronunciation.source_pronunciations('张麻子来了。', [{'src': '张麻子', 'tgt': 'John Mazi'}]) == []
    monkeypatch.setattr(pronunciation, '_dictionary', lambda: {'zhang': [['ZH', 'AA1', 'NG']], 'ma': [['M', 'AA1']]})
    assert pronunciation.source_pronunciations('张麻子来了。', [{'src': '张麻子', 'tgt': 'Zhang Mazi'}]) == []


def test_guidance_cannot_silently_rewrite_the_accepted_translation():
    with pytest.raises(ValueError, match='canonical translation'):
        pronunciation.synthesis_text(SimpleNamespace(text='You pay.',metadata={'synthesis_text':'I pay.'}))
    direct='<word|W ER1 D>'
    assert pronunciation.synthesis_text(SimpleNamespace(text=direct,metadata={})) == direct


def test_unspoken_quotes_do_not_enter_tts_but_canonical_script_stays_bound():
    from core.providers.contracts import TTSRequest
    request = TTSRequest(text='点“新建”，然后说「Hello, I’m here.」。', output_path='unused')
    assert pronunciation.synthesis_text(request) == '点新建，然后说 Hello, I’m here.。'
    assert request.text == '点“新建”，然后说「Hello, I’m here.」。'
    with pytest.raises(ValueError, match='canonical translation'):
        pronunciation.synthesis_text(SimpleNamespace(text='点“新建”。', metadata={'synthesis_text': '点新建。'}))


def test_dictionary_hints_stabilize_short_words_and_skip_ambiguous_homographs():
    plan = pronunciation.short_english_pronunciations('Bandits?', [])
    annotated = pronunciation.annotate_pronunciations('Bandits?', plan)
    assert annotated == '<Bandits|B AE1 N D AH0 T S>?'
    assert pronunciation.synthesis_text(SimpleNamespace(text='Bandits?', metadata={'synthesis_text': annotated})) == annotated
    assert pronunciation.short_english_pronunciations('Read?', []) == []
    assert pronunciation.short_english_pronunciations('You should read it.', []) == []
    assert pronunciation.short_english_pronunciations('18?', []) == []


def test_same_text_with_changed_pronunciation_invalidates_a_speech_receipt(tmp_path):
    from pydub.generators import Sine
    from core.tts_reference_plan import write_conditioning_receipt, conditioning_receipt_matches
    from core.providers.contracts import TTSRequest
    path=tmp_path/'audio.wav';Sine(300).to_audio_segment(duration=300).export(path,format='wav')
    request=TTSRequest(text='Hu Wan.',output_path=str(path),language='en',
                       metadata={'reference_plan':{'fingerprint':'same'},'backend':'indextts2',
                                 'synthesis_text':'<Hu|HH UW1> <Wan|W AA1 N>.'})
    write_conditioning_receipt(request,backend='indextts2')
    assert conditioning_receipt_matches(request)
    request.metadata['synthesis_text']='<Hu|HH UW1> <Wan|W AE1 N>.'
    assert not conditioning_receipt_matches(request)


def test_indextts_receipt_binds_actual_quote_normalized_input(tmp_path):
    import json
    from pydub.generators import Sine
    from core.tts_reference_plan import write_conditioning_receipt, conditioning_receipt_matches
    from core.providers.contracts import TTSRequest
    path = tmp_path/'audio.wav'
    Sine(300).to_audio_segment(duration=300).export(path, format='wav')
    request = TTSRequest(text='点“新建”。', output_path=str(path), language='zh',
        metadata={'reference_plan': {'fingerprint': 'same'}, 'backend': 'indextts2'})
    write_conditioning_receipt(request, backend='indextts2')
    receipt_path = path.with_suffix('.conditioning.json')
    receipt = json.loads(receipt_path.read_text())
    assert receipt['text'] == '点“新建”。'
    assert receipt['synthesis_text'] == '点新建。'
    assert conditioning_receipt_matches(request)
    receipt['synthesis_text'] = request.text  # a take made under the old frontend
    receipt_path.write_text(json.dumps(receipt))
    assert not conditioning_receipt_matches(request)


def test_named_contrast_options_keep_source_spelling_and_chinese_speech():
    from core.tts_pronunciation import original_name_conflicts, spoken_plan, canonical_name_spellings
    source = "No, not Doritos or Lay's, I mean chips for AI."
    display = "不是Doritos或Lay's，是AI用的芯片。"
    assert not original_name_conflicts(source,display,'zh',[])
    assert original_name_conflicts(source,'不是Doritos，是AI用的芯片。','zh',[])
    actual = spoken_plan(display,'zh',[])['spoken_text']
    assert '多力多滋' in actual and '乐事' in actual
    assert 'Doritos' not in actual and "Lay's" not in actual
    restored, proof = canonical_name_spellings(source,'不是多力多滋或乐事，是AI用的芯片。','zh',[])
    assert restored == display and len(proof) == 2


def test_mixed_language_pronunciation_keeps_source_words_and_chinese_boundaries(monkeypatch):
    monkeypatch.setattr(pronunciation, '_dictionary', lambda: {
        'claude': [['K', 'L', 'AO1', 'D']], 'code': [['K', 'OW1', 'D']],
        'hey': [['HH', 'EY1']], 'gen': [['JH', 'EH1', 'N']],
        'read': [['R', 'IY1', 'D'], ['R', 'EH1', 'D']]})
    monkeypatch.setattr(pronunciation, '_local_g2p', lambda: None)
    text = '把Claude Code连上HeyGen，read和Unknown不变。'
    entries = pronunciation.mixed_english_pronunciations('Claude Code uses HeyGen; read Unknown.', text)
    assert [entry['tgt'] for entry in entries] == ['Claude', 'Code', 'HeyGen']
    annotated = pronunciation.annotate_pronunciations(text, entries)
    assert '把<Claude|K L AO1 D> <Code|K OW1 D>连上<HeyGen|HH EY1 JH EH1 N>' in annotated
    assert pronunciation.synthesis_text(SimpleNamespace(text=text, metadata={'synthesis_text': annotated})) == (
        '把 <Claude|K L AO1 D> <Code|K OW1 D> 连上 <HeyGen|HH EY1 JH EH1 N>，read 和 Unknown 不变。')
    assert pronunciation.mixed_english_pronunciations('No named tool here.', text) == []
    assert pronunciation.mixed_english_pronunciations('Claude Code.', 'Claude Code.') == []


def test_mixed_language_tts_surface_preserves_latin_word_spaces_and_canonical_text():
    request = SimpleNamespace(text='Epidemic Sound也一样。', metadata={})
    assert pronunciation.synthesis_text(request) == 'Epidemic Sound 也一样。'
    assert request.text == 'Epidemic Sound也一样。'


def test_unknown_source_name_uses_only_local_g2p_as_generation_hint(monkeypatch):
    monkeypatch.setattr(pronunciation, '_dictionary', lambda: {})
    monkeypatch.setattr(pronunciation, '_local_g2p', lambda: SimpleNamespace(predict=lambda _: ['T', 'EH1', 'L', 'AH0']))
    entries = pronunciation.mixed_english_pronunciations('I recorded in Tella.', '我用Tella录制。')
    assert len(entries) == 1
    assert entries[0]['basis'].endswith('local_G2P_prediction_requires_readback')
    assert entries[0]['words'] == ['Tella']
