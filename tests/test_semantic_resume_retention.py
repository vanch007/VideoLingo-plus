import json
from pathlib import Path

import pytest

from core import atomic_dubbing as dubbing
from core import duration_candidate_audit as audit
from core.pipeline.artifact_manifest import hash_path
from core.speech_budget import duration_context


def test_current_source_context_exposes_only_explicit_anaphora_antecedents():
    context = json.dumps({'topic': 'editing', 'preceding_source': ['Create an account.'],
                          'surrounding_source': ['Also sing a song.'],
                          'dialogue_role_hints': {'speaker': 'UNVERIFIED'}})
    plain = audit._current_source_context('The cuts, zooms and formatting.', context)
    assert 'source_anaphora' not in plain
    assert 'song' not in json.dumps(plain) and 'UNVERIFIED' not in json.dumps(plain)
    bound = audit._current_source_context('Use Google to do that.', context)['source_anaphora']
    assert bound['current_source_quotes'] == ['do that']
    assert bound['preceding_source'] == ['Create an account.']
    assert 'song' not in json.dumps(bound)
    # An unordered legacy neighborhood does not establish a preceding action.
    assert 'source_anaphora' not in audit._current_source_context(
        'Use Google to do that.', json.dumps({'surrounding_source': ['Create an account.']}))


def test_antecedent_source_comes_from_exact_source_lineage():
    unit = {'unit_id': 'L2.01', 'parent_line_id': 'L2', 'source': 'Use Google to do that.',
            'parent_source': 'Use Google to do that.'}
    terms = {'dialogue_units': [{'line_id': 'L1', 'source_text': 'Create an account.'},
                               {'line_id': 'L2', 'source_text': unit['source']},
                               {'line_id': 'L3', 'source_text': 'Also sing a song.'}]}
    context = json.loads(duration_context(unit, terms))
    assert context['preceding_source'] == ['Create an account.']
    terms['dialogue_units'][1]['source_text'] = 'Different source text.'
    assert json.loads(duration_context(unit, terms))['preceding_source'] == []


def test_scoped_objection_must_interpret_the_actual_target_not_repeat_an_allegation(monkeypatch):
    source, candidate = 'It is something I pay for.', '我付钱的事。'
    allegation = {'source_quote': source, 'candidate_quote': candidate, 'difference': 'FALSE_ALLEGATION_SENTINEL'}
    def reply(prompt, **kwargs):
        assert source in prompt and candidate in prompt
        assert 'The matter of me paying.' not in prompt
        assert 'FALSE_ALLEGATION_SENTINEL' not in prompt
        value = {'equivalent': True, 'source_quote': source, 'candidate_quote': candidate,
                 'candidate_meaning': 'Something I pay for.', 'explanation': '付钱 expresses pay.'}
        assert kwargs['valid_def'](value)['status'] == 'success'
        assert kwargs['valid_def']({**value, 'candidate_meaning': ''})['status'] == 'error'
        assert kwargs['valid_def']({**value, 'candidate_quote': '花钱'})['status'] == 'error'
        return value
    monkeypatch.setattr(audit, 'ask_gpt', reply)
    result = audit.review_scoped_objection(source, candidate, allegation, '{}', {},
                                          'The matter of me paying.')
    assert result['equivalent'] and result['defects'] == []


def test_duration_compaction_cannot_delete_an_explicit_frequency():
    from core.translation_invariants import source_fact_conflicts
    source = 'The edits usually go back and forth.'
    assert source_fact_conflicts(source,'修改来来回回，反复不断。','zh') == ['explicit_frequency_lost']
    assert not source_fact_conflicts(source,'修改通常来来回回。','zh')
    assert source_fact_conflicts(source,'修改总是来来回回。','zh') == ['explicit_frequency_strengthened']


def test_audible_draft_retention_requires_complete_audio_bound_evidence(tmp_path):
    audio = tmp_path / 'voice.wav'
    audio.write_bytes(b'original audio')
    passed = {'unit_id': 'U1', 'text': '保存文件。', 'natural_audio_path': str(audio),
              'candidate_sha256': hash_path(audio), 'fits': True, 'content_complete': True,
              'content_issues': [], 'readback_signature': 'measured-current-audio'}
    state = {'candidates': [{'unit_id': 'U1', 'target': 'An unsynthesized audit rewrite'}],
             'attempts': [passed, {**passed, 'text': 'Incomplete later take', 'content_complete': False}]}
    assert dubbing._audible_resume_drafts(state) == {'U1': '保存文件。'}
    assert audio.read_bytes() == b'original audio'
    for change in ({'readback_signature': None}, {'fits': False}, {'content_issues': ['name mismatch']}):
        assert dubbing._audible_resume_drafts({'attempts': [{**passed, **change}]}) == {}
    audio.write_bytes(b'changed audio')
    assert dubbing._audible_resume_drafts(state) == {}


@pytest.mark.parametrize('target', ['你大概也不懂技术。','你多半也不懂技术。','你应该也不是技术人员。'])
def test_explicit_probability_survives_natural_chinese_forms(target):
    from core.translation_invariants import source_fact_conflicts
    assert not source_fact_conflicts("You probably aren't technical either.",target,'zh')


def test_subject_probability_cannot_become_a_categorical_assertion():
    from core.translation_invariants import source_fact_conflicts
    assert source_fact_conflicts("You probably aren't technical either.",'你也不懂技术。','zh')
    assert not source_fact_conflicts('You are not technical.','你不懂技术。','zh')
    assert not source_fact_conflicts('The word probably is an adverb.','probably是个副词。','zh')


def test_probability_repair_guidance_preserves_modality_and_negation():
    from core.translation_invariants import source_fact_conflicts, source_fact_guidance
    conflicts = source_fact_conflicts('You probably do not want this.', '你肯定不想要这个。', 'zh')
    guidance = source_fact_guidance(conflicts)
    assert 'probably' in guidance and '不确定性' in guidance and '否定' in guidance
    assert source_fact_guidance([]) == ''


def test_usual_frequency_cannot_be_strengthened_into_always():
    from core.translation_invariants import source_fact_conflicts
    assert source_fact_conflicts('The revisions usually went back and forth.','修改总是来回反复。','zh')
    assert not source_fact_conflicts('The revisions usually went back and forth.','修改通常来回反复。','zh')
    assert not source_fact_conflicts('The revisions always went back and forth.','修改总是来回反复。','zh')


def test_explicit_source_owner_and_independent_goals_are_not_waived_by_a_model():
    from core.translation_invariants import source_fact_conflicts
    assert source_fact_conflicts('inside a folder on my computer.','电脑文件夹里的内容。','zh')==['explicit_possession:my']
    assert not source_fact_conflicts('inside a folder on my computer.','在我电脑的一个文件夹里。','zh')
    assert not source_fact_conflicts('Oh my goodness!','天哪！','zh')
    source='Teach people to use the tools and to make money with them.'
    assert source_fact_conflicts(source,'教人用工具赚钱。','zh')==['independent_teaching_goals_collapsed']
    assert not source_fact_conflicts(source,'教人用工具，还教人靠它们赚钱。','zh')
    assert not source_fact_conflicts('Use the tools to make money.','用工具赚钱。','zh')
def test_shared_candidate_discovery_reuses_shorter_bound_seed_without_accepting_it(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    from pydub.generators import Sine
    from core import atomic_dubbing as atomic
    from core.pipeline.artifact_manifest import hash_path
    root = tmp_path / 'candidates' / 'U1'
    root.mkdir(parents=True)
    for stem, seconds, text in [('slow', 2.0, '完整译文'), ('fast', 1.0, '完整译文'), ('other', .5, '另一译文')]:
        path = root / (stem + '.wav')
        Sine(440).to_audio_segment(duration=int(seconds * 1000)).export(path, format='wav')
        path.with_suffix('.conditioning.json').write_text(json.dumps({'text': text,
            'generation_settings': {'seed': 45}, 'audio': {'sha256': hash_path(path)}}))
    request = SimpleNamespace(output_path=str(tmp_path / 'U1.wav'), task_row={'unit_id': 'U1'})
    def retained(_, previous):
        if previous['text'] != '完整译文' or hash_path(previous['natural_audio_path']) != previous['candidate_sha256']:
            return None
        return SimpleNamespace(output_path=previous['natural_audio_path'])
    monkeypatch.setattr(atomic, '_resume_natural_take', retained)
    selected = atomic._cached_natural_take(request)
    assert selected.output_path == str(root / 'fast.wav')
    # A valid generation receipt only supplies input to current fit/readback.
    monkeypatch.setattr(atomic, '_resume_natural_take', lambda *_: None)
    assert atomic._cached_natural_take(request) is None


@pytest.mark.parametrize('source', ['Alright, my friends, enough talking.',
                                   'Hey, my dear friends, let us begin.'])
def test_direct_friendly_address_does_not_assert_ownership(source):
    from core.translation_invariants import source_fact_conflicts, source_vocative
    assert source_vocative(source)
    assert not source_fact_conflicts(source, '好了，各位，开始吧。', 'zh')
    context = audit._current_source_context(source, '{}')
    assert context['source_vocative']['source_quote'] in source


@pytest.mark.parametrize('source', ['My friends saved the file.',
                                   'Alright, my friends saved the file.',
                                   'Alright, my friends, save my file.'])
def test_friendly_address_exception_does_not_waive_other_possession(source):
    from core.translation_invariants import source_fact_conflicts
    assert 'explicit_possession:my' in source_fact_conflicts(source, '朋友保存文件。', 'zh')


def test_retained_take_accepts_digest_serialization_but_not_changed_bytes(tmp_path, monkeypatch):
    from core import tts_reference_plan
    from core.providers.contracts import TTSRequest
    path = tmp_path / 'voice.wav'
    path.write_bytes(b'bound natural audio')
    settings = {'seed': 43, 'runtime': 'current'}
    request = TTSRequest(text='完整译文。', output_path=str(tmp_path / 'U1.wav'), language='zh',
                         metadata={'generation_settings': {**settings, 'seed': 42}})
    previous = {'text': request.text, 'generation_settings': settings,
                'natural_audio_path': str(path),
                'candidate_sha256': hash_path(path).removeprefix('sha256:')}
    monkeypatch.setattr(tts_reference_plan, 'conditioning_receipt_matches', lambda _: True)
    assert dubbing._resume_natural_take(request, previous).output_path == str(path)
    path.write_bytes(b'changed')
    assert dubbing._resume_natural_take(request, previous) is None


def test_managed_short_candidates_are_discoverable_and_rejected_takes_are_excluded(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from pydub.generators import Sine
    repo = tmp_path / 'repo'
    root = repo / 'run_workspaces' / 'short' / 'output/audio/preflight/candidates/U1'
    root.mkdir(parents=True)
    path = root / 'fast.wav'
    Sine(440).to_audio_segment(duration=700).export(path, format='wav')
    path.with_suffix('.conditioning.json').write_text(json.dumps({'text': '完整译文',
        'generation_settings': {'seed': 45}, 'audio': {'sha256': hash_path(path)}}))
    monkeypatch.setattr(dubbing, '__file__', str(repo / 'core/atomic_dubbing.py'))
    request = SimpleNamespace(output_path=str(repo / 'full/output/audio/preflight/U1.wav'),
                              task_row={'unit_id': 'U1'})
    monkeypatch.setattr(dubbing, '_resume_natural_take', lambda _, previous:
                        SimpleNamespace(output_path=previous['natural_audio_path']))
    assert dubbing._cached_natural_take(request).output_path == str(path)
    assert dubbing._cached_natural_take(request, [hash_path(path)]) is None


def test_bound_cached_duration_draft_still_requires_current_semantics_and_display(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from pydub.generators import Sine
    from core import speech_budget, bilingual_alignment, runtime_context
    from core.providers import mlx_tts
    from core.tts_pronunciation import spoken_plan, annotate_standard_mandarin_spans, synthesis_text
    from core.utterance_plan import signature
    text = 'Claude也一样。'
    plan = spoken_plan(text, 'zh', [])
    spoken, guidance = annotate_standard_mandarin_spans(plan)
    rendered = synthesis_text(SimpleNamespace(text=text, metadata={
        'synthesis_text': spoken, 'standard_mandarin_guidance': True, 'spoken_plan': plan,
        'spoken_transliterations': plan['matched_transliterations'], 'spoken_pinyin': guidance}))
    path = tmp_path / 'bound.wav'
    Sine(440).to_audio_segment(duration=700).export(path, format='wav')
    receipt_path = path.with_suffix('.conditioning.json')
    receipt_path.write_text(json.dumps({'text': text, 'backend': 'indextts2', 'language': 'zh',
        'target_duration': None, 'reference_plan': {'fingerprint': 'same-reference'},
        'audio': {'sha256': hash_path(path)}, 'synthesis_text': rendered,
        'pronunciation_guidance': guidance, 'generation_settings': {'runtime_signature': 'runtime',
            'backend_config_signature': signature({'fit_duration': False}),
            'pronunciation_signature': signature(guidance)}}))
    monkeypatch.setattr(dubbing, '_candidate_receipts', lambda *_: [receipt_path])
    monkeypatch.setattr(dubbing, 'native_tts_runtime_signature', lambda: 'runtime')
    monkeypatch.setattr(dubbing, 'load_key', lambda key, default=None:
                        'mlx_indextts2' if key == 'tts_method' else default)
    monkeypatch.setattr(mlx_tts, 'resolve_indextts_config', lambda *_: {})
    monkeypatch.setattr(runtime_context, 'effective_target_language', lambda *_: 'zh')
    calls = []
    monkeypatch.setattr(speech_budget, 'audit_duration_candidate',
                        lambda *_: calls.append('source-review') or {'equivalent': True})
    monkeypatch.setattr(bilingual_alignment, 'prepare_candidate_alignment',
                        lambda *_: calls.append('display-review') or [])
    row = SimpleNamespace(unit_id='U1', text='至于Claude，也是一样的操作。',
                          duration=1., reference_plan_fingerprint='same-reference')
    unit = {'unit_id': 'U1', 'source': 'As for Claude, it is the same thing.'}
    assert dubbing._cached_duration_draft(row, None, unit, '{}', []) == text
    assert calls == ['source-review', 'display-review']
    monkeypatch.setattr(speech_budget, 'audit_duration_candidate', lambda *_: {'equivalent': False})
    assert dubbing._cached_duration_draft(row, None, unit, '{}', []) is None
    assert dubbing._cached_duration_draft(row, None, unit, '{}', [
        {'unit_id': 'U1', 'candidate_sha256': hash_path(path), 'content_complete': False}]) is None
