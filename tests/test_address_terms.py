import json

from core.address_terms import address_form, batch_note, find_exchanges

SOURCES = ["大哥", "美女我也不能要", "侠胆柔肠之大腿", "哥", "弟", "你的腿就是我的腿", "大哥，我还能改吗", "老弟！"]


def test_detects_whole_line_and_vocative_forms_only():
    assert [address_form(s) for s in SOURCES] == ["大哥", None, None, "哥", "弟", None, "大哥", "老弟"]
    assert address_form("哥们儿你好") is None and address_form("弟弟呀") == "弟弟"


def test_pairs_elder_and_younger_forms_said_close_together():
    found = find_exchanges(SOURCES)
    assert [(SOURCES[x["elder"]], SOURCES[x["younger"]]) for x in found] == [("哥", "弟"), ("大哥，我还能改吗", "老弟！")]
    assert find_exchanges(["大哥", "a", "b", "c", "弟"]) == []
    assert find_exchanges(["大哥", "哥"]) == []


def test_batch_note_names_both_lines():
    note = batch_note(["哥", "弟"])
    assert '"哥" and "弟" are two people addressing each other' in note and "never render both the same" in note
    assert batch_note(["大哥", "你好"]) is None


def test_translator_note_reaches_batches_with_the_exchange(tmp_path, monkeypatch):
    from core import step4_1_summarize
    path = tmp_path / "terminology.json"
    path.write_text(json.dumps({"terms": []}), encoding="utf-8")
    monkeypatch.setattr(step4_1_summarize, "TERMINOLOGY_JSON_PATH", str(path))
    note = step4_1_summarize.search_things_to_note_in_prompt("江湖豪情\n哥\n弟")
    assert note.startswith('1. "哥" and "弟" are two people addressing each other')
    assert step4_1_summarize.search_things_to_note_in_prompt("哥\n你好") is None
