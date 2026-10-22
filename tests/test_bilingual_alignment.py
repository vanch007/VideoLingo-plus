import copy
import json
from pathlib import Path

import pytest

from core import bilingual_alignment as ba
from core import plan_subtitles as ps
from core.llm_provider import LLMProviderConfig
from scripts.align_dubbing_units import alignment_text


def test_source_partition_keeps_words_names_and_original_order(monkeypatch):
    from core import runtime_context
    monkeypatch.setattr(runtime_context, 'effective_source_language', lambda *_: 'en')
    source = 'Everything in this recording was processed by my Media Studio system while I was away.'
    parts = ba.source_display_parts(source, 42)
    assert len(parts) > 1 and all(len(part) <= 42 for part in parts)
    assert ' '.join(parts) == source
    assert any('Media Studio' in part for part in parts)
    assert any(part.startswith('while') for part in parts)


def test_source_partition_prefers_clauses_and_does_not_merge_next_sentence(monkeypatch):
    from core import runtime_context
    monkeypatch.setattr(runtime_context, 'effective_source_language', lambda *_: 'en')
    source = "the exact cost, and how I'm saving thousands of dollars a month. Let's go."
    parts = ba.source_display_parts(source, 42)
    assert parts[0] == 'the exact cost,'
    assert parts[-1] == "Let's go."
    assert ' '.join(parts) == source
    assert all(len(part) <= 42 for part in parts)


@pytest.mark.parametrize('relation', ['before', 'after', 'until'])
def test_source_partition_keeps_wrappable_temporal_predicate_together(monkeypatch, relation):
    from core import runtime_context
    from core.subtitle_layout import split_display_text
    monkeypatch.setattr(runtime_context, 'effective_source_language', lambda *_: 'en')
    source = f'A skill is simply just a file of instructions that Claude reads {relation} it does the job.'
    parts = ba.source_display_parts(source, 84, line_cap=42, max_lines=2)
    assert ' '.join(parts) == source
    assert any(f'Claude reads {relation} it does the job' in part for part in parts)
    assert all(len(split_display_text(part, 42)) <= 2 for part in parts)


def test_temporal_preference_cannot_exceed_subtitle_capacity(monkeypatch):
    from core import runtime_context
    from core.subtitle_layout import split_display_text
    monkeypatch.setattr(runtime_context, 'effective_source_language', lambda *_: 'en')
    source = 'The system reads the complete instruction file before it starts the lengthy editing task.'
    parts = ba.source_display_parts(source, 42)
    assert ' '.join(parts) == source
    assert all(len(split_display_text(part, 42)) == 1 for part in parts)


@pytest.mark.parametrize('source', [
    'The cuts, the zooms, the layout, the formatting, yada, yada.',
    'The colors, the fonts, the margins, and the sizes.',
])
def test_native_recovery_keeps_a_wrappable_lexical_list_together(monkeypatch, source):
    from core import runtime_context
    from core.subtitle_layout import split_display_text
    monkeypatch.setattr(runtime_context, 'effective_source_language', lambda *_: 'en')
    assert len(split_display_text(source, 42)) <= 2
    assert ba.source_display_parts(source, 84, line_cap=42, max_lines=2) == [source]


def test_lexical_list_grouping_still_respects_real_display_capacity_and_sentence_boundary(monkeypatch):
    from core import runtime_context
    from core.subtitle_layout import split_display_text
    monkeypatch.setattr(runtime_context, 'effective_source_language', lambda *_: 'en')
    source = 'The colors, the fonts, the margins, and the sizes. Save it.'
    parts = ba.source_display_parts(source, 84, line_cap=42, max_lines=2)
    assert parts[-1] == 'Save it.' and ' '.join(parts) == source
    limited = ba.source_display_parts(source, 24, line_cap=24, max_lines=1)
    assert len(limited) > 2 and ' '.join(limited) == source
    assert all(len(split_display_text(part, 24)) == 1 for part in limited)


@pytest.mark.parametrize('source', [
    'Claude should be able to download and install everything that it needs and it can.',
    'It basically says that it cut down 2 minutes and 46 of raw footage into 31 seconds,',
])
def test_source_partition_uses_real_word_wrapping_not_total_capacity(monkeypatch, source):
    from core import runtime_context
    from core.subtitle_layout import split_display_text
    monkeypatch.setattr(runtime_context, 'effective_source_language', lambda *_: 'en')
    assert len(source) <= 84 and len(split_display_text(source, 42)) > 2
    parts = ba.source_display_parts(source, 84, line_cap=42, max_lines=2)
    assert len(parts) > 1 and ' '.join(parts) == source
    assert all(len(split_display_text(part, 42)) <= 2 for part in parts)


@pytest.mark.parametrize('duration', ['2 minutes and 46', 'two minutes and forty-six seconds', '1 hour 20 minutes'])
def test_source_partition_keeps_compound_durations_intact(monkeypatch, duration):
    from core import runtime_context
    monkeypatch.setattr(runtime_context, 'effective_source_language', lambda *_: 'en')
    source = f'It cut down {duration} of raw footage into 31 seconds.'
    parts = ba.source_display_parts(source, 84, line_cap=42, max_lines=2)
    assert ' '.join(parts) == source
    assert any(duration in part for part in parts)
    assert any(source[a:b] == duration for a, b in ba._target_phrase_spans(source))


def test_source_duration_cannot_be_split_across_separate_semantic_pairs():
    with pytest.raises(ValueError, match='source word, name or quantity'):
        ba.validate_pairs({'source': '2 minutes and 46 seconds', 'target': '两分钟加四十六秒'},
            {'pairs': [{'source': '2 minutes', 'target': '两分钟'},
                       {'source': 'and 46 seconds', 'target': '加四十六秒'}]},
            chars=44, source_chars=84, check_timing=False)


def test_semantic_span_uses_two_visual_lines_without_widening_either_line(monkeypatch):
    from core.subtitle_layout import split_display_text
    monkeypatch.setattr(ba, 'display_line_limit', lambda: 2)
    source = "and how I'm saving thousands of dollars a month."
    target = '以及我如何每月省下数千美元。'
    u = {'source': source, 'target': target}
    pairs = ba.validate_display_pairs(u, {'pairs': [{'source': source, 'target': target}]},
                                     chars=22, source_chars=42, check_timing=False)
    assert pairs[0]['source'] == source
    assert max(map(len, split_display_text(source, 42))) <= 42
    monkeypatch.setattr(ba, 'display_line_limit', lambda: 1)
    with pytest.raises(ValueError, match='oversized'):
        ba.validate_display_pairs(u, {'pairs': pairs}, chars=22, source_chars=42, check_timing=False)


def test_two_line_capacity_still_rejects_three_wrapped_lines(monkeypatch):
    monkeypatch.setattr(ba, 'display_line_limit', lambda: 2)
    source = 'abcdefghij abcdefghij abcdefghij'
    # 32 total characters fit the arithmetic 2 * 20 budget, but complete words
    # require three real visual lines. Character totals alone cannot accept it.
    with pytest.raises(ValueError, match='further semantic segmentation'):
        ba.validate_display_pairs({'source': source, 'target': '打开。'},
            {'pairs': [{'source': source, 'target': '打开。'}]}, chars=22, source_chars=20, check_timing=False)


def test_ordered_translation_cannot_publish_a_failed_pair(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ba, 'display_limits', lambda: (22, 42))
    monkeypatch.setattr(ba, 'source_display_parts', lambda *a, **k: ['Open it.', 'Save it.'])
    monkeypatch.setattr(ba, 'ask_gpt', lambda *a, **k: {'1': '打开它。', '2': '删除它。'})
    monkeypatch.setattr(ba, '_review_pairs', lambda *_: [
        {'equivalent': True, 'issues': []}, {'equivalent': False, 'issues': ['Save became delete']}])
    with pytest.raises(ValueError, match='lost fragment meaning'):
        ba.translate_source_partition('Open it. Save it.', 'zh', '',
                                      {'target_speech_seconds': 2., 'max_speed_factor': 1.08, 'target_max_words': 6})
    assert not ba.CANDIDATE_ALIGNMENT.exists()


@pytest.mark.parametrize('review_passes', [True, False])
def test_old_partition_is_revalidated_before_regenerating_boundaries(tmp_path, monkeypatch, review_passes):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ba, 'display_limits', lambda: (22, 42))
    monkeypatch.setattr(ba, '_candidate_key', lambda *a: 'current-policy')
    u = {'unit_id': 'L1', 'source': 'Open it. Save it.', 'target': '打开它。保存它。'}
    old = {'source': u['source'], 'target': u['target'], 'pairs': [
        {'source': 'Open it.', 'target': '打开它。'}, {'source': 'Save it.', 'target': '保存它。'}]}
    ba.CANDIDATE_ALIGNMENT.parent.mkdir(parents=True)
    ba.CANDIDATE_ALIGNMENT.write_text(json.dumps({'old-policy': old}))
    reviews = []
    def review(unit, pairs, context):
        reviews.append((pairs, context))
        return [{'equivalent': review_passes, 'issues': [] if review_passes else ['wrong meaning']}
                for pair in pairs]
    monkeypatch.setattr(ba, '_review_pairs', review)
    def regenerate(*args, **kwargs):
        assert not review_passes, 'Valid boundaries should be reviewed, not needlessly regenerated'
        raise ValueError('No accepted partition')
    monkeypatch.setattr(ba, 'align_unit', regenerate)
    if review_passes:
        result = ba.prepare_candidate_alignment(u, 'current context')
        assert [p['source'] for p in result] == ['Open it.', 'Save it.']
        assert 'current-policy' in json.loads(ba.CANDIDATE_ALIGNMENT.read_text())
    else:
        with pytest.raises(ValueError, match='No accepted partition'):
            ba.prepare_candidate_alignment(u, 'current context')
    assert len(reviews) == 1 and reviews[0][1] == 'current context'


def test_old_partition_cannot_bypass_new_display_contract(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ba, 'display_limits', lambda: (22, 20))
    monkeypatch.setattr(ba, '_candidate_key', lambda *a: 'new-policy')
    u = {'unit_id': 'L1', 'source': 'abcdefghij abcdefghij abcdefghij', 'target': '打开它。'}
    ba.CANDIDATE_ALIGNMENT.parent.mkdir(parents=True)
    ba.CANDIDATE_ALIGNMENT.write_text(json.dumps({'old-policy': {**u, 'pairs': [
        {'source': u['source'], 'target': u['target']}]}}))
    monkeypatch.setattr(ba, '_review_pairs', lambda *a: pytest.fail('Reject three visual lines before any model call'))
    monkeypatch.setattr(ba, 'align_unit', lambda *a, **kw: (_ for _ in ()).throw(ValueError('New boundaries required')))
    with pytest.raises(ValueError, match='New boundaries required'):
        ba.prepare_candidate_alignment(u)


def test_ordered_name_fragment_requires_exact_source_name_and_whole_target_language(tmp_path, monkeypatch):
    from core import runtime_context
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runtime_context, 'effective_source_language', lambda *_: 'en')
    monkeypatch.setattr(ba, 'display_limits', lambda: (22, 42))
    monkeypatch.setattr(ba, 'source_display_parts', lambda *a, **k: ['Acme Studio,', 'cost me money.'])
    monkeypatch.setattr(ba, '_review_pairs', lambda _, pairs, context:
        [{'equivalent': True, 'issues': []} for pair in pairs])
    def reply(prompt, **kwargs):
        validate = kwargs['valid_def']
        assert validate({'1': 'Acme Studio，', '2': '花了我的钱。'})['status'] == 'success'
        assert validate({'1': 'Another Service，', '2': '花了我的钱。'})['status'] == 'error'
        assert validate({'1': 'Acme Studio，', '2': 'cost me money.'})['status'] == 'error'
        return {'1': 'Acme Studio，', '2': '花了我的钱。'}
    monkeypatch.setattr(ba, 'ask_gpt', reply)
    target = ba.translate_source_partition('Acme Studio, cost me money.', 'zh', '',
        {'target_speech_seconds': 3., 'max_speed_factor': 1.08, 'target_max_words': 8})
    assert target == 'Acme Studio，花了我的钱。'


def test_long_subtitle_cannot_silently_drop_the_third_line():
    text = ' '.join(f'word{i:02}' for i in range(30))
    with pytest.raises(ValueError, match='timed segmentation'):
        ps._format_srt_text(text)
    assert ps._format_srt_text(text, max_lines=None).split() == text.split()


def test_failed_whole_utterance_fallback_is_not_accepted_as_review():
    u = unit()
    with pytest.raises(ValueError, match='passing semantic'):
        ba.validate_pairs(u, {'pairs': [{
            'source': u['source'], 'target': u['target'],
            'semantic_review': {'equivalent': True, 'issues': [], 'basis': 'fallback_complete_locked_utterance'},
        }]}, chars=100, require_review=True)


def test_chinese_boundary_ids_respect_actual_character_timing():
    u = {'unit_id': 'C1', 'source': 'Open it. Save it.', 'target': '打开。保存。',
         'start': 0., 'end': 2., 'target_words': [
             {'word': w, 'start': i * .4, 'end': (i + 1) * .4} for i, w in enumerate('打开保存')]}
    result = ba._decode_pairs(u, {'pairs': [
        {'source_end': 8, 'target_end': 3}, {'source_end': 17, 'target_end': 6}]})
    assert result['pairs'][0] == {'source': 'Open it.', 'target': '打开。'}


def test_alignment_clock_is_not_scaled_to_hide_out_of_window_words():
    u = unit()
    u['target_words'][-1]['start'] = 3.
    u['target_words'][-1]['end'] = 3.5
    assert max(ba.target_boundaries(u).values()) > u['end']


def unit():
    return {'unit_id': 'L1', 'parent_line_id': 'L1', 'parent_source': '我出钱，你剿匪。', 'source': '我出钱，你剿匪。',
            'target': 'I pay. You fight the bandits.', 'start': 10., 'end': 12.4, 'duration': 2.4,
            'target_words': [{'word': w, 'start': i * .3, 'end': (i+1) * .3}
                             for i, w in enumerate('I pay. You fight the bandits.'.split())]}


def pairs():
    return {'pairs': [{'source': '我出钱，', 'target': 'I pay.'},
                      {'source': '你剿匪。', 'target': 'You fight the bandits.'}]}


def boundary_ids():
    return {'pairs': [{'source_end': 4, 'target_end': 6}, {'source_end': 8, 'target_end': len(unit()['target'])}]}


def test_boundary_ids_materialize_canonical_punctuation_without_model_copying():
    assert ba._decode_pairs(unit(), boundary_ids()) == pairs()
    assert ba._decode_pairs(unit(), pairs()) == pairs()


def test_marker_punctuation_is_never_used_as_the_canonical_subtitle():
    markers = pairs()
    markers['pairs'][0]['target'] = 'I pay,'
    assert ba._decode_pairs(unit(), markers) == pairs()
    markers['pairs'][0]['target'] = 'You pay,'
    with pytest.raises(ValueError, match='locked words'):
        ba._decode_pairs(unit(), markers)


def test_boundary_marker_cannot_hide_a_decimal_value_change():
    with pytest.raises(ValueError, match='numeric value'):
        ba._materialize_markers('1.8 million.', ['18 million.'])


def test_alignment_punctuation_keeps_words_distinct_without_changing_spoken_letters():
    text = "Four-dot masks!Goose Town's people—it was 1.8 million."
    aligned = alignment_text(text)
    assert "people— it" in aligned and "masks! Goose" in aligned
    assert "Four-dot" in aligned and "1.8" in aligned
    assert ba.normalize_transcript(aligned) == ba.normalize_transcript(text)


def test_caption_partitions_preserve_source_compounds_and_target_titles():
    u = {'unit_id': 'L1', 'source': '我的亲爹也得吃饭。', 'target': 'My own father has to eat.',
         'start': 0., 'end': 3., 'target_words': [
             {'word': w, 'start': i*.3, 'end': (i+1)*.3} for i,w in enumerate('My own father has to eat.'.split())]}
    with pytest.raises(ValueError, match='source word'):
        ba.validate_pairs(u, {'pairs': [{'source': '我的亲', 'target': 'My own father'},
                                      {'source': '爹也得吃饭。', 'target': 'has to eat.'}]})
    u.update(source='给县长惊喜。', target='Give the County Magistrate a surprise.', target_words=[
        {'word': w, 'start': i*.3, 'end': (i+1)*.3} for i,w in enumerate('Give the County Magistrate a surprise.'.split())])
    with pytest.raises(ValueError, match='target name'):
        ba.validate_pairs(u, {'pairs': [{'source': '给县长', 'target': 'Give the County'},
                                      {'source': '惊喜。', 'target': 'Magistrate a surprise.'}]})


def test_target_hints_preserve_amount_units_and_departure_phrases():
    text = "I'll give you 1.8 million taels to leave town, and fight the bandits."
    parts = ba._target_parts(text, 38)
    assert all(len(p) <= 38 for p in parts)
    assert any('1.8 million taels' in p for p in parts)
    assert any('leave town' in p for p in parts)
    assert ''.join(ba.normalize_transcript(p) for p in parts) == ba.normalize_transcript(text)


def test_native_anchors_choose_actual_correspondence_instead_of_character_proportions():
    source = '我买米请大家吃饭，你去修电脑。'
    result = ba._source_parts_from_anchors(source, ['买米', '请大家吃饭', '你去修电脑'], 22)
    assert result == ['我买米', '请大家吃饭，', '你去修电脑。']
    ba._spans(source, result)


@pytest.mark.parametrize('change', ['rewrite', 'omit', 'repeat', 'reorder', 'long', 'word', 'number'])
def test_locked_alignment_rejects_changed_coverage_and_unsafe_cuts(change):
    u, data = unit(), pairs()
    if change == 'rewrite':
        data['pairs'][0]['target'] = 'You pay.'
    elif change == 'omit':
        data['pairs'] = data['pairs'][:1]
    elif change == 'repeat':
        data['pairs'].append(data['pairs'][0])
    elif change == 'reorder':
        data['pairs'].reverse()
    elif change == 'long':
        data = {'pairs': [{'source': u['source'], 'target': u['target']}]}
    elif change == 'word':
        data['pairs'][1]['target'] = 'You fight the ban'
        data['pairs'].append({'source': '。', 'target': 'dits.'})
    else:
        u['target'] = '1.8 million.'
        data = {'pairs': [{'source': '我出钱，', 'target': '1.'}, {'source': '你剿匪。', 'target': '8 million.'}]}
    with pytest.raises(ValueError):
        ba.validate_pairs(u, data, chars=24)


def test_repeated_source_and_target_occurrences_are_not_deduplicated():
    u = {'unit_id': 'L1', 'source': '人呢？人呢？', 'target': 'People? People?', 'start': 0., 'end': 1.2,
         'target_words': [{'word': 'People?', 'start': 0., 'end': .5}, {'word': 'People?', 'start': .6, 'end': 1.1}]}
    result = ba.validate_pairs(u, {'pairs': [{'source': '人呢？', 'target': 'People?'}] * 2}, chars=8)
    assert [p['source_char_span'] for p in result] == [[0, 3], [3, 6]]
    assert [p['target_char_span'] for p in result] == [[0, 7], [8, 15]]


def test_pair_review_cannot_certify_a_missing_pair(monkeypatch):
    monkeypatch.setattr(ba, 'ask_gpt', lambda *a, **k: {'checks': [{'pair_index': 1, 'equivalent': True, 'issues': []}]})
    with pytest.raises(ValueError, match='every pair'):
        ba._review_pairs(unit(), pairs()['pairs'], '')


def test_zero_based_review_normalization_keeps_every_verdict_and_rejects_missing_checks(monkeypatch):
    data = {'checks': [{'pair_index': 0, 'equivalent': False, 'issues': ['meaning mismatch']},
                       {'pair_index': 1, 'equivalent': True, 'issues': []}]}
    normalized = ba._normalize_pair_checks(data)
    assert normalized['checks'] == [{'pair_index': 1, 'equivalent': False, 'issues': ['meaning mismatch']},
                                    {'pair_index': 2, 'equivalent': True, 'issues': []}]
    assert data['checks'][0]['pair_index'] == 0
    incomplete = ba._normalize_pair_checks({'checks': data['checks'][:1]})
    monkeypatch.setattr(ba, 'ask_gpt', lambda *a, **k: incomplete)
    with pytest.raises(ValueError, match='every pair'):
        ba._review_pairs(unit(), pairs()['pairs'], '')


def test_semantic_mismatch_retries_only_boundaries_and_never_edits_the_script(monkeypatch):
    # Recovery must change the incorrect correspondence, not keep asking the
    # same reviewer until the unchanged partition receives a different vote.
    responses = iter([{'pairs': [
        {'source': '我出钱，', 'target': 'I pay. You'},
        {'source': '你剿匪。', 'target': 'fight the bandits.'}]}, boundary_ids()])
    prompts = []
    def request(prompt, **options):
        prompts.append(prompt)
        return next(responses)
    monkeypatch.setattr(ba, 'ask_gpt', request)
    native_feedback = []
    def native(*args):
        native_feedback.append(args[-1])
        return ba._decode_pairs(unit(), boundary_ids())
    monkeypatch.setattr(ba, '_native_anchor_pairs', native)
    reviews = iter([[{'pair_index': 1, 'equivalent': False, 'issues': ['Amounts and actions are mixed']},
                     {'pair_index': 2, 'equivalent': True, 'issues': []}],
                    [{'pair_index': i, 'equivalent': True, 'issues': []} for i in (1, 2)]])
    monkeypatch.setattr(ba, '_review_pairs', lambda *a: next(reviews))
    u = unit(); original = copy.deepcopy(u)
    result = ba.align_unit(u, chars=24)
    assert u == original
    assert [p['target'] for p in result] == ['I pay.', 'You fight the bandits.']
    assert 'Amounts and actions are mixed' in json.dumps(native_feedback)
    assert '不翻译、不改写' in prompts[0]


def test_failed_pair_review_never_falls_back_to_proportional_source_slicing(monkeypatch):
    monkeypatch.setattr(ba, 'ask_gpt', lambda *a, **k: boundary_ids())
    monkeypatch.setattr(ba, '_native_anchor_pairs', lambda *a: ba._decode_pairs(unit(), boundary_ids()))
    monkeypatch.setattr(ba, '_review_pairs', lambda *a: [
        {'pair_index': i, 'equivalent': False, 'issues': ['wrong paired meaning']} for i in (1, 2)])
    with pytest.raises(ValueError, match='semantic alignment failed'):
        ba.align_unit(unit(), chars=24)


def test_layout_uses_only_reviewed_meaning_pairs_and_the_same_clock_for_both_languages():
    u = unit(); original = copy.deepcopy(u)
    parts = [{**p, 'semantic_review': {'equivalent': True, 'issues': []}} for p in pairs()['pairs']]
    alignment = {'fingerprint': 'bound', 'units': [{**u, 'status': 'pass', 'pairs': parts}]}
    frame, report = ps.layout_plan({'units': [u]}, chars=24, bilingual_alignment=alignment)
    assert frame.Source.tolist() == ['我出钱，', '你剿匪。']
    assert frame.Translation.tolist() == ['I pay.', 'You fight the bandits.']
    assert frame.timestamp.tolist() == [(10., 10.6), (10.6, 12.4)]
    assert report['bilingual_semantic_alignment'] == 'pass'
    assert u == original
    with pytest.raises(ValueError, match='semantic alignment is required'):
        ps.layout_plan({'units': [u]}, chars=24)


def test_original_video_subtitles_use_source_speech_clock_and_dub_uses_target_clock():
    u = unit()
    u.update(source_start=10., source_end=12.4, source_words=[
        {'word': '我出钱', 'start': 10., 'end': 11.},
        {'word': '你剿匪', 'start': 11.2, 'end': 12.4}])
    parts = [{**p, 'semantic_review': {'equivalent': True, 'issues': []}} for p in pairs()['pairs']]
    alignment = {'fingerprint': 'bound', 'units': [{**u, 'status': 'pass', 'pairs': parts}]}
    original, source_report = ps.layout_plan({'units': [u]}, chars=24, bilingual_alignment=alignment, dubbed=False)
    dubbed, target_report = ps.layout_plan({'units': [u]}, chars=24, bilingual_alignment=alignment, dubbed=True)
    assert original.timestamp.tolist() == [(10., 11.2), (11.2, 12.4)]
    assert dubbed.timestamp.tolist() == [(10., 10.6), (10.6, 12.4)]
    assert source_report['display_clock'] == 'source_speech'
    assert target_report['display_clock'] == 'target_speech'
    assert original[['Source', 'Translation']].equals(dubbed[['Source', 'Translation']])


def test_cached_pairs_are_invalidated_by_changed_locked_text(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ba, 'get_llm_provider_config', lambda: LLMProviderConfig('local', 'index', 'http://local', ''))
    monkeypatch.setattr(ba, 'align_unit', lambda u, **kw: [
        {**p, 'semantic_review': {'equivalent': True, 'issues': []}} for p in pairs()['pairs']])
    first = ba.ensure_bilingual_alignment({'units': [unit()]}, chars=24)
    assert first['status'] == 'pass'
    changed = unit(); changed['source'] = '你出钱，我剿匪。'
    monkeypatch.setattr(ba, 'align_unit', lambda *a, **k: (_ for _ in ()).throw(ValueError('fresh review required')))
    with pytest.raises(ValueError, match='fresh review required'):
        ba.ensure_bilingual_alignment({'units': [changed]}, chars=24)
    # Failed new evidence cannot overwrite the last committed semantic receipt.
    assert json.loads(ba.ALIGNMENT_PATH.read_text()) == first
    failed = json.loads(ba.ALIGNMENT_PATH.with_suffix('.progress.json').read_text())
    assert failed['status'] == 'fail' and failed['fingerprint'] != first['fingerprint']


def test_export_rejects_modified_srt_even_when_the_pair_review_passed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ba, 'get_llm_provider_config', lambda: LLMProviderConfig('local', 'index', 'http://local', ''))
    u = {'unit_id': 'L1', 'parent_line_id': 'L1', 'parent_source': '走。', 'source': '走。', 'target': 'Go.',
         'start': 0., 'end': 1., 'duration': 1., 'target_words': [{'word': 'Go.', 'start': 0., 'end': .5}]}
    plan = {'units': [u]}
    monkeypatch.setattr(ps, 'load_plan', lambda: plan)
    monkeypatch.setattr(ps, 'align_target_words', lambda _: {'units': [{'unit_id': 'L1', 'words': u['target_words']}]})
    ps.render_plan_subtitles(dubbed=True)
    ba.verify_bilingual_display(plan)
    receipt = json.loads(ba.ALIGNMENT_PATH.read_text())
    receipt['units'][0]['pairs'][0]['semantic_review']['basis'] = 'altered review'
    ba.ALIGNMENT_PATH.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='stale'):
        ba.verify_bilingual_display(plan)
    ps.render_plan_subtitles(dubbed=True)
    path = Path('output/dub_bilingual.srt')
    path.write_text(path.read_text().replace('Go.', 'Stop.'))
    with pytest.raises(ValueError, match='files changed'):
        ba.verify_bilingual_display(plan)
