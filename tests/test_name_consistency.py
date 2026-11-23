from core.name_consistency import check_names, find_candidates, name_pool, run_name_consistency

LINES = [
    "又是一个",
    "现在收钱太早",
    "大玉",
    "添温",
    "如花似玉",
    "哎，我带夫人来了，再收姑娘不方便。",
    "不是，黛玉和晴雯吗？",
    "莫非她就叫黛玉晴雯？",
    "黛玉晴雯子。",
]
TERMS = [{"src": "师爷", "tgt": "Advisor", "note": ""},
         {"src": "黛玉晴雯子", "tgt": "Daiyu Qingwenzi", "note": ""},
         {"src": "Seedream", "tgt": "Seedream", "note": ""}]


def test_compound_name_parts_count_only_where_spoken_on_their_own():
    names = name_pool(TERMS, LINES)
    assert {"黛玉", "晴雯", "黛玉晴雯", "黛玉晴雯子", "师爷"} <= set(names)
    assert "玉晴" not in names and "Seedream" not in names


def test_only_misheard_names_become_candidates():
    found = find_candidates(LINES, name_pool(TERMS, LINES))
    assert [(c["heard"], c["name"]) for c in found] == [("大玉", "黛玉"), ("添温", "晴雯")]


def test_replaces_only_confirmed_spans_and_logs_rejections():
    calls = []

    def ask(prompt, **kwargs):
        calls.append(prompt)
        assert "不是，黛玉和晴雯吗？" in prompt
        return {"decisions": [{"case": 1, "is_name": True, "reason": "same name later"},
                              {"case": 2, "is_name": False, "reason": "unclear"}]}

    corrected, report = check_names(LINES, TERMS, "topic", ask)
    assert len(calls) == 1
    assert corrected[2] == "黛玉" and corrected[3] == "添温"
    assert [len(a) for a in corrected] == [len(b) for b in LINES]
    assert [a["heard"] for a in report["applied"]] == ["大玉"]
    assert [r["heard"] for r in report["rejected"]] == ["添温"]


def test_no_candidates_means_no_llm_call(tmp_path):
    path = tmp_path / "s.txt"
    path.write_text("你好\n黛玉晴雯子。", encoding="utf-8")

    def ask(*args, **kwargs):
        raise AssertionError("LLM must not be called")

    report = run_name_consistency(str(path), TERMS, ask=ask, report_path=str(tmp_path / "r.json"))
    assert report["candidates"] == [] and path.read_text(encoding="utf-8") == "你好\n黛玉晴雯子。"
    assert (tmp_path / "r.json").exists()


def test_run_rewrites_sentence_file_with_confirmed_names(tmp_path):
    path = tmp_path / "s.txt"
    path.write_text("\n".join(LINES), encoding="utf-8")

    def ask(prompt, **kwargs):
        return {"decisions": [{"case": 1, "is_name": True}, {"case": 2, "is_name": True}]}

    run_name_consistency(str(path), TERMS, ask=ask, report_path=str(tmp_path / "r.json"))
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[2:4] == ["黛玉", "晴雯"] and lines[5] == LINES[5]
