import json

from core.address_terms import address_form, batch_note, find_exchanges, repair_exchanges

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


def test_english_whole_line_pair_uses_the_fixed_table_without_an_llm():
    sources = ["哥", "弟", "大哥，我还能改吗", "老弟！", "姐姐", "妹妹！"]
    drafts = ["Brother.", "brother!", "Brother, can I?", "Brother, can I?", "Sister", "Sister!"]

    def ask(*a, **k):
        raise AssertionError("no LLM for English whole-line addresses")

    fixed, report = repair_exchanges(sources, drafts, "English", ask)
    assert fixed == ["Big brother.", "Little brother!", drafts[2], drafts[3], "Big sister.", "Little sister!"]
    assert len(report["exchanges"]) == 3 and len(report["repaired"]) == 2
    assert report["kept"][0]["reason"].startswith("address inside a longer line")


def test_other_targets_ask_for_the_two_words_alone_and_reject_anything_longer():
    prompts = []

    def ask(prompt, **kwargs):
        prompts.append(prompt)
        return {"elder": "Anh ơi", "younger": "Em ơi"}

    fixed, report = repair_exchanges(["江湖豪情", "哥", "弟"], ["Giang hồ", "Anh.", "Anh."], "Vietnamese", ask)
    assert fixed == ["Giang hồ", "Anh ơi.", "Em ơi."] and "江湖豪情" not in prompts[0]

    drafts = ["Anh.", "Anh."]
    fixed, report = repair_exchanges(["哥", "弟"], drafts, "Vietnamese",
                                     lambda p, **k: {"elder": "Anh, chỗ dựa của em", "younger": "Em"})
    assert fixed == drafts and report["kept"]

    def boom(*a, **k):
        raise RuntimeError("LLM unavailable")

    fixed, report = repair_exchanges(["哥", "弟"], drafts, "Vietnamese", boom)
    assert fixed == drafts and "LLM unavailable" in report["kept"][0]["reason"]


def test_translator_note_reaches_batches_with_the_exchange(tmp_path, monkeypatch):
    from core import step4_1_summarize
    path = tmp_path / "terminology.json"
    path.write_text(json.dumps({"terms": []}), encoding="utf-8")
    monkeypatch.setattr(step4_1_summarize, "TERMINOLOGY_JSON_PATH", str(path))
    note = step4_1_summarize.search_things_to_note_in_prompt("江湖豪情\n哥\n弟")
    assert note.startswith('1. "哥" and "弟" are two people addressing each other')
    assert step4_1_summarize.search_things_to_note_in_prompt("哥\n你好") is None
