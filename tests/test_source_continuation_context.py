import json

from core.speech_budget import duration_context, source_continuation_context
from core.duration_candidate_audit import _current_source_context, _resolve_source_meanings


def test_only_a_real_source_dependency_exposes_continuation():
    previous = 'Or you can decide to turn on'
    source = 'one important thing for us to browse the internet,'
    proof = source_continuation_context(source, previous)
    assert proof['current_source_quote'] == source
    assert proof['preceding_source_quote'] == previous
    assert all(p['current_source_quote'] in source and p['preceding_governor_quote'] in previous
               for p in proof['grammar_hypotheses'])
    assert source_continuation_context(source, 'Turn on the option.') is None
    assert source_continuation_context('The option is useful.', 'And we') is None


def test_continuation_requires_exact_lineage_and_same_observed_voice():
    previous = 'and I tried several different sounds'
    source = 'until I found something I liked.'
    unit = {'unit_id': 'L2.01', 'parent_line_id': 'L2', 'speaker': 'S01',
            'source': source, 'parent_source': source}
    terms = {'dialogue_units': [
        {'line_id': 'L1.01', 'source_text': previous, 'acoustic_speaker': 'S01'},
        {'line_id': 'L2.01', 'source_text': source, 'acoustic_speaker': 'S01'}]}
    context = json.loads(duration_context(unit, terms))
    assert context['source_continuation']['preceding_source_quote'] == previous
    assert _current_source_context(source, json.dumps(context))['source_continuation'] == context['source_continuation']
    assert 'source_continuation' not in _current_source_context('Different current source.', json.dumps(context))
    terms['dialogue_units'][0]['acoustic_speaker'] = 'S02'
    assert 'source_continuation' not in json.loads(duration_context(unit, terms))
    terms['dialogue_units'][0]['acoustic_speaker'] = 'S01'
    terms['dialogue_units'][1]['source_text'] = 'Different current source.'
    assert 'source_continuation' not in json.loads(duration_context(unit, terms))


def test_source_resolution_keeps_bound_grammar_without_exposing_candidate(monkeypatch):
    from core import duration_candidate_audit as audit
    source = 'until I found something I liked.'
    previous = 'and I tried different sounds'
    continuation = source_continuation_context(source, previous)
    context = json.dumps({'preceding_source': [previous], 'source_continuation': continuation})
    def reply(prompt, **kwargs):
        assert previous in prompt and 'CANDIDATE_SENTINEL' not in prompt
        return '直到找到合心意的东西。'
    monkeypatch.setattr(audit, 'ask_gpt', reply)
    resolution = _resolve_source_meanings(source, context,
        {'defects': [{'source_quote': source, 'candidate_quote': 'CANDIDATE_SENTINEL'}]}, {})
    assert _current_source_context(source, resolution['context'])['source_continuation'] == continuation


def test_toggle_anaphora_resolves_only_the_current_action():
    context = json.dumps({'preceding_source': ['Here is the music feature.'],
                          'surrounding_source': ['Send your password elsewhere.']})
    result = _current_source_context('Go ahead and turn this on.', context)
    assert result['source_anaphora']['current_source_quotes'] == ['turn this on']
    assert 'password' not in json.dumps(result)


def test_coordinated_complement_preserves_shared_conditional_not_an_instruction():
    previous='Now, if you want to both see the editing process'
    source='and also get a full breakdown of the announcement,'
    proof=source_continuation_context(source,previous)
    assert proof['preceding_source_quote']==previous
    assert any(p['dependency']=='conj' for p in proof['grammar_hypotheses'])
    assert source_continuation_context(source,'The editing process finished.') is None


def test_plain_pronoun_uses_bound_preceding_source_without_importing_other_facts():
    context=json.dumps({'preceding_source':['Choose the background music.'],
                        'surrounding_source':['Also delete the file.']})
    result=_current_source_context('and have it very low.',context)
    assert result['source_anaphora']['current_source_quotes']==['it']
    assert result['source_anaphora']['preceding_source']==['Choose the background music.']
    assert 'delete' not in json.dumps(result)
    assert 'source_anaphora' not in _current_source_context('The items that you showed.',context)


def test_repeated_nominal_fragment_keeps_exact_source_word_sense_only():
    source='Hmm, chips.'
    prior='No, not snack brands; these are chips for AI.'
    context=json.dumps({'preceding_source':[prior], 'surrounding_source':['Also sing a song.']})
    result=_current_source_context(source,context)
    proof=result['source_lexical_continuation']
    assert proof['current_source_quote']=='chips' and proof['preceding_source_quote']==prior
    assert 'sing' not in json.dumps(result)
    assert 'source_lexical_continuation' not in _current_source_context('The animations.',context)


def test_nominal_description_is_not_an_independent_event_assertion():
    source = 'Someone who goes out and finds the announcement.'
    context = _current_source_context(source, '{}')
    proof = context['source_syntax']
    assert proof['source_quote'] == source and proof['root_quote'] == 'Someone'
    assert proof['has_finite_matrix_assertion'] is False
    assert proof['relative_clause_quotes'] == ['who goes out and finds the announcement']
    assert 'source_syntax' not in _current_source_context('Someone found the announcement.', '{}')
    assert 'source_syntax' not in _current_source_context('It is something I pay for.', '{}')
