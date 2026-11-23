import json

from core.wordplay import find_wordplay, translator_note

LINES = ["你要什么呀", "腿", "什么腿", "侠胆柔肠之大腿", "你的腿就是我的腿", "有了腿便有了路", "张麻子来了", "麻子"]


def reply(*items):
    return lambda prompt, **kwargs: {"wordplay": list(items)}


def item(src, literal_line, figurative_line, **extra):
    return {"src": src, "literal": "leg", "figurative": "a powerful patron", "joke": "both senses",
            "rendering": "a leg up", "literal_line": literal_line, "figurative_line": figurative_line,
            "understood_word_for_word": False, **extra}


def test_keeps_recurring_wordplay_with_two_quoted_lines_even_when_a_quote_is_padded():
    report = find_wordplay(LINES, ask=reply(item("腿", "有了腿便有了路", "要什么腿")))
    assert [w["src"] for w in report["wordplay"]] == ["腿"] and report["wordplay"][0]["occurrences"] == 5


def test_drops_items_without_transcript_evidence_or_that_are_names():
    report = find_wordplay(LINES, known_terms=["张麻子"], ask=reply(
        item("麻子", "麻子", "张麻子来了"),
        item("腿", "腿", "腿"),
        item("腿", "他的腿断了", "你的腿就是我的腿"),
        item("路", "有了腿便有了路", "有了腿便有了路"),
    ))
    assert report["wordplay"] == []
    assert [d["reason"] for d in report["dropped"]] == [
        "name_or_glossary_term", "evidence_not_two_nearby_lines",
        "evidence_not_in_transcript", "not_recurring"]


def test_empty_reply_and_empty_transcript():
    assert find_wordplay(LINES, ask=reply())["wordplay"] == []
    assert find_wordplay([], ask=None) == {"wordplay": [], "dropped": []}


def test_translator_gets_wordplay_note_only_for_batches_that_contain_it(tmp_path, monkeypatch):
    from core import step4_1_summarize
    wordplay = find_wordplay(LINES, ask=reply(item("腿", "有了腿便有了路", "你的腿就是我的腿")))["wordplay"]
    path = tmp_path / "terminology.json"
    path.write_text(json.dumps({"terms": [{"src": "师爷", "tgt": "Advisor", "note": "a counselor"}],
                                "wordplay": wordplay}, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(step4_1_summarize, "TERMINOLOGY_JSON_PATH", str(path))

    note = step4_1_summarize.search_things_to_note_in_prompt("师爷，你的腿就是我的腿")
    assert note.splitlines() == ['1. "师爷": "Advisor", meaning: a counselor', "2. " + translator_note(wordplay[0])]
    assert 'Suggested rendering: "a leg up"' in note and "same short image in every line" in note
    assert step4_1_summarize.search_things_to_note_in_prompt("师爷好") == '1. "师爷": "Advisor", meaning: a counselor'
    assert step4_1_summarize.search_things_to_note_in_prompt("你好") is None


def test_prompt_names_languages_in_english():
    from core.wordplay import english_language_name
    prompts = []
    find_wordplay(LINES, source_language="zh", target_language="en",
                  ask=lambda prompt, **kw: prompts.append(prompt) or {"wordplay": []})
    assert "a Simplified Chinese dialogue for translation into English" in prompts[0]
    assert english_language_name("zh-TW") == "Traditional Chinese" and english_language_name("pl") == "Polish"


def test_drops_wordplay_a_literal_translation_already_conveys():
    report = find_wordplay(LINES, ask=reply(item("腿", "有了腿便有了路", "你的腿就是我的腿", understood_word_for_word=True)))
    assert report["wordplay"] == [] and report["dropped"][0]["reason"] == "literal_translation_already_works"


def test_glossary_rows_marked_wordplay_become_translator_instructions():
    from core.wordplay import split_glossary_wordplay
    plain, wordplay = split_glossary_wordplay([
        {"src": "腿", "tgt": "a powerful ally", "note": "wordplay: 抱大腿 — a powerful patron to rely on."},
        {"src": "师爷", "tgt": "Advisor", "note": "a counselor"},
    ])
    assert [t["src"] for t in plain] == ["师爷"]
    assert wordplay == [{"src": "腿", "literal": "", "joke": "", "figurative": "抱大腿 — a powerful patron to rely on.",
                         "rendering": "a powerful ally", "origin": "glossary"}]
    assert translator_note(wordplay[0]).startswith(
        '"腿" is recurring wordplay: 抱大腿 — a powerful patron to rely on. Suggested rendering: "a powerful ally".')
