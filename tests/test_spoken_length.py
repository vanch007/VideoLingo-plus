from core.spoken_length import english_syllables, english_spoken_words, source_form_hint, source_form_conflicts


def test_chinese_duration_guidance_counts_pronounced_units_not_clauses():
    from core.spoken_length import chinese_spoken_units
    assert chinese_spoken_units('接着它还能高亮某些行，') == 10
    assert chinese_spoken_units('它用Claude Code。') == 2 + english_syllables('Claude Code')
    assert chinese_spoken_units('它用Claude Code。') == chinese_spoken_units('它 用 Claude Code')


def test_only_tight_nominal_questions_require_the_compact_spoken_form():
    assert source_form_conflicts('人呢？', 'Where are the people?', .7)
    assert not source_form_conflicts('人呢？', 'The people?', .7)
    assert not source_form_conflicts('人呢？', 'Where are the people?', 3.)
    assert not source_form_conflicts('你为什么在这里呢？', 'Why are you here?', .7)


def test_spoken_length_uses_pronunciation_instead_of_written_word_count():
    assert english_syllables("Understood.") > english_syllables("Got it.")
    assert english_syllables("family") > english_syllables("kin")


def test_numeric_notation_is_expanded_before_length_search():
    assert english_spoken_words("1.8 million") == ["one","point","eight","million"]
    assert english_syllables("1.8 million") == english_syllables("one point eight million")
    assert english_syllables("1,800,000") == english_syllables("one million eight hundred thousand")


def test_nominal_question_hint_does_not_strip_a_complete_verb_or_wh_question():
    assert "名词式" in source_form_hint("人呢？人呢？")
    assert not source_form_hint("你为什么在这里呢？")
    assert not source_form_hint("难道你听不懂什么叫惊喜？")
