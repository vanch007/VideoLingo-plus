import pytest

from core import translation_constraints as tc

TERMS = [{"src": "黄老爷", "tgt": "Master Huang", "note": ""}]
WORDPLAY = [{"src": "腿", "rendering": "a powerful backer", "figurative": "a patron"}]


def test_english_pair_is_checked_by_seniority_and_sex_not_by_equality():
    sources = ["哥", "弟"]
    cons = tc.build_constraints(sources, "en", [], [])
    assert [(c.line, c.kind, c.strict) for c in cons] == [(0, "address", True), (1, "address", True)]

    def ask(*a, **k):
        raise AssertionError("whole-line English address needs no model")

    # The old repair skipped this pair because the two lines differ; 弟 is not a sister.
    fixed, report = tc.review(sources, ["Brother.", "Sister!"], cons, "English", ask)
    assert fixed == ["Big brother.", "Little brother!"] and len(report["repaired"]) == 2
    fixed, report = tc.review(sources, ["Big brother.", "Little bro."], cons, "English", ask)
    assert fixed == ["Big brother.", "Little bro."] and report["satisfied"] == 2


def test_female_forms_require_sister():
    cons = tc.build_constraints(["师姐", "师妹"], "en", [], [])
    fixed, _ = tc.review(["师姐", "师妹"], ["Senior brother.", "Junior brother."], cons, "English",
                         lambda *a, **k: None)
    assert fixed == ["Senior sister.", "Junior sister."]


def test_only_violating_lines_go_to_review_with_context_and_requirements():
    sources = ["你的腿就是我的腿", "黄老爷来了", "大哥，我还能改吗", "老弟！", "江湖"]
    drafts = ["Your backer is my backer.", "Huang is here.", "Brother, can I change?", "Little brother!", "Jianghu"]
    cons = tc.build_constraints(sources, "en", TERMS, WORDPLAY)
    seen = []

    def ask(prompt, valid_def=None, **kwargs):
        seen.append(prompt)
        reply = {"items": [{"id": "1", "action": "fix", "translation": "Master Huang is here."},
                           {"id": "2", "action": "fix", "translation": "Big brother, can I still change?"}]}
        assert valid_def(reply)["status"] == "success"
        return reply

    fixed, report = tc.review(sources, drafts, cons, "English", ask)
    assert fixed == [drafts[0], "Master Huang is here.", "Big brother, can I still change?", drafts[3], drafts[4]]
    assert len(seen) == 1 and '"id": "0"' not in seen[0] and "STRICT" in seen[0]
    assert report["satisfied"] == 2 and not report["unresolved"]


def test_reviewer_cannot_waive_strict_or_invent_content_but_can_keep_soft_terms():
    sources = ["大哥，我还能改吗", "老弟！", "黄老爷，您说呢"]
    drafts = ["Brother, can I change?", "Little brother!", "What do you say, sir?"]
    cons = tc.build_constraints(sources, "en", TERMS, [])

    def ask(prompt, **kwargs):
        return {"items": [{"id": "0", "action": "keep", "reason": "fine"},
                          {"id": "2", "action": "keep", "reason": "vocative rendered as sir"}]}

    fixed, report = tc.review(sources, drafts, cons, "English", ask)
    assert fixed == drafts
    assert [k["line"] for k in report["kept"]] == [2] and [u["line"] for u in report["unresolved"]] == [0]

    long = {"items": [{"id": "0", "action": "fix",
                       "translation": "Big brother, your backer is my lifeline and can I still change it now?"},
                      {"id": "2", "action": "keep"}]}
    fixed, report = tc.review(sources, drafts, cons, "English", lambda p, **k: long)
    assert fixed[0] == drafts[0] and report["unresolved"][0]["line"] == 0


def test_other_targets_are_judged_by_the_reviewer_and_failures_keep_drafts():
    sources = ["哥", "弟"]
    cons = tc.build_constraints(sources, "vi", [], [])
    fixed, report = tc.review(sources, ["Anh.", "Chị."], cons, "Vietnamese",
                              lambda p, **k: {"items": [{"id": "0", "action": "keep"},
                                                        {"id": "1", "action": "fix", "translation": "Em trai."}]})
    assert fixed == ["Anh.", "Em trai."]

    def boom(*a, **k):
        raise RuntimeError("LLM unavailable")

    fixed, report = tc.review(sources, ["Anh.", "Anh."], cons, "Vietnamese", boom)
    assert fixed == ["Anh.", "Anh."] and "LLM unavailable" in report["unresolved"][0]["reason"]


def test_wordplay_needle_is_the_head_word():
    cons = tc.build_constraints(["有了腿"], "en", [], WORDPLAY)
    assert cons[0].expect["needle"] == "backer" and tc.check(cons[0], "With a backer...") is True
    assert tc.check(cons[0], "With legs...") is False


def test_glossary_gloss_in_parentheses_accepts_either_word():
    """A summary glossary may say "leg (backing)"; requiring that literal string made the reviewer paste it in."""
    assert tc._alternatives("leg (backing)") == ["leg", "backing"]
    assert tc._alternatives("Counselor") == ["Counselor"]
    assert tc._alternatives("leg / backing") == ["leg", "backing"]
    cons = tc.build_constraints(["你的腿就是我的腿", "什么腿"], "English",
                                [{"src": "腿", "tgt": "leg (backing)"}], [])
    assert tc.check(cons[0], "Your backing is my backing.") is True
    assert tc.check(cons[1], "What leg?") is True
    assert tc.check(cons[1], "What?") is False
    assert "never several at once" in cons[0].requirement
    fixed, report = tc.review(["你的腿就是我的腿"], ["Your backing is my backing."], cons[:1], "English",
                              ask=lambda *a, **k: pytest.fail("satisfied lines must not reach the reviewer"))
    assert fixed == ["Your backing is my backing."] and report["satisfied"] == 1
