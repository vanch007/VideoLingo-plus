import json
from pathlib import Path
import pandas as pd
import pytest

from core import translation_review


def test_clean_suggested_fix():
    assert translation_review.clean_suggested_fix("Assuming the missing character, the translation should be: 'Surprise'.") == "Surprise"
    assert translation_review.clean_suggested_fix('Translation: "What is this?"') == "What is this?"
    assert translation_review.clean_suggested_fix("Note: 'Good morning'") == "Good morning"
    assert translation_review.clean_suggested_fix("“Hello world”") == "Hello world"
    assert translation_review.clean_suggested_fix("Clean text") == "Clean text"


def test_review_and_repair_applies_critical_fixes_with_re_review(tmp_path: Path, monkeypatch):
    report_path = tmp_path / "review_report.json"

    df = pd.DataFrame({
        "LineID": ["L00001", "L00002"],
        "Speaker": ["S01", "S02"],
        "Source": ["为什么不出钱剿匪", "三天之后给你一百八十万两银子"],
        "Translation": ["Why don't you spend money?", "I will give you 1.8 million silver taels in three days."],
    })

    call_count = 0
    def mock_ask_gpt(prompt, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return {
                "overall_summary": "One critical issue found.",
                "issues": [
                    {
                        "line_id": "L00002",
                        "speaker": "S02",
                        "issue_type": "agent_patient_confusion",
                        "severity": "critical",
                        "problem": "S02 is relaying S01's offer, not promising to pay himself.",
                        "suggested_fix": "In three days he'll give you 1.8 million silver taels.",
                    }
                ],
            }
        else:
            return {
                "overall_summary": "All clear after repair.",
                "issues": [],
            }

    monkeypatch.setattr(translation_review, "ask_gpt", mock_ask_gpt)

    repaired_df, report = translation_review.review_and_repair_translation(
        df,
        terminology={"topic": "Test video", "speaker_profiles": []},
        report_path=report_path,
        max_repair_rounds=2,
    )

    assert call_count == 2
    assert report["status"] == "pass"
    assert report["issues_found_count"] == 1
    assert report["repairs_applied_count"] == 1
    assert repaired_df.loc[repaired_df["LineID"] == "L00002", "Translation"].values[0] == "In three days he'll give you 1.8 million silver taels."
    assert report_path.exists()


def test_review_fails_when_repair_introduces_new_unresolvable_issue(tmp_path: Path, monkeypatch):
    report_path = tmp_path / "review_report.json"
    df = pd.DataFrame([{"LineID": "L1", "Speaker": "S07", "Source": "不要打开门", "Translation": "Do not open the door."}])

    call_count = 0
    def mock_ask_gpt(prompt, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return {
                "issues": [{
                    "line_id": "L1", "speaker": "S07", "severity": "critical",
                    "suggested_fix": "Open the door.", "problem": "Synthetic bad repair"
                }]
            }
        else:
            return {
                "issues": [{
                    "line_id": "L1", "speaker": "S07", "severity": "critical",
                    "suggested_fix": "", "problem": "Dropped negation"
                }]
            }

    monkeypatch.setattr(translation_review, "ask_gpt", mock_ask_gpt)
    monkeypatch.setattr(translation_review, "load_key", lambda k, default=None: "en" if k == "target_language" else default)

    repaired, report = translation_review.review_and_repair_translation(
        df, terminology={}, report_path=report_path, max_repair_rounds=2
    )
    assert report["status"] == "failed"


def test_review_fails_on_unknown_line_id(tmp_path: Path, monkeypatch):
    report_path = tmp_path / "review_report.json"
    df = pd.DataFrame([{"LineID": "L1", "Speaker": "S07", "Source": "测试", "Translation": "Test"}])
    monkeypatch.setattr(
        translation_review, "ask_gpt",
        lambda *args, **kwargs: {
            "issues": [{"line_id": "MISSING_LINE", "speaker": "S07", "severity": "critical", "suggested_fix": "Fixed", "problem": "Bad ID"}]
        }
    )
    monkeypatch.setattr(translation_review, "load_key", lambda k, default=None: "en" if k == "target_language" else default)
    repaired, report = translation_review.review_and_repair_translation(
        df, terminology={}, report_path=report_path
    )
    assert report["status"] == "failed"
    assert "unresolvable critical issue" in report["error"]


def test_review_fails_on_empty_fix_critical(tmp_path: Path, monkeypatch):
    report_path = tmp_path / "review_report.json"
    df = pd.DataFrame([{"LineID": "L1", "Speaker": "S07", "Source": "测试", "Translation": "Test"}])
    monkeypatch.setattr(
        translation_review, "ask_gpt",
        lambda *args, **kwargs: {
            "issues": [{"line_id": "L1", "speaker": "S07", "severity": "critical", "suggested_fix": "", "problem": "No fix given"}]
        }
    )
    repaired, report = translation_review.review_and_repair_translation(
        df, terminology={}, report_path=report_path
    )
    assert report["status"] == "failed"


def test_review_fails_on_explicit_failed_response(tmp_path: Path, monkeypatch):
    report_path = tmp_path / "review_report.json"
    df = pd.DataFrame([{"LineID": "L1", "Speaker": "S07", "Source": "测试", "Translation": "Test"}])
    monkeypatch.setattr(
        translation_review, "ask_gpt",
        lambda *args, **kwargs: {"status": "failed", "overall_summary": "LLM review crashed"}
    )
    repaired, report = translation_review.review_and_repair_translation(
        df, terminology={}, report_path=report_path
    )
    assert report["status"] == "failed"
    assert "LLM review crashed" in report["error"]


def test_review_and_repair_handles_exception_as_failed_status(tmp_path: Path, monkeypatch):
    report_path = tmp_path / "review_report.json"
    df = pd.DataFrame({
        "LineID": ["L00001"],
        "Speaker": ["S01"],
        "Source": ["测试"],
        "Translation": ["Test"],
    })
    monkeypatch.setattr(
        translation_review,
        "ask_gpt",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("LLM rate limit"))
    )
    repaired_df, report = translation_review.review_and_repair_translation(
        df,
        terminology={"topic": "Test", "speaker_profiles": []},
        report_path=report_path,
    )
    assert report["status"] == "failed"
    assert "LLM rate limit" in report["error"]
    assert report["issues_found_count"] == 0


def test_review_and_repair_rejects_wrong_target_language_fix(tmp_path: Path, monkeypatch):
    report_path = tmp_path / "review_report.json"
    df = pd.DataFrame([
        {
            "LineID": "L00001",
            "Speaker": "S09",
            "Source": "明天给你五十块",
            "Translation": "I will give you fifty tomorrow.",
        }
    ])
    fake_issues = [
        {
            "line_id": "L00001",
            "speaker": "S09",
            "severity": "critical",
            "suggested_fix": "明天给你五十块",
            "problem": "test wrong language fix",
        }
    ]
    monkeypatch.setattr(
        translation_review,
        "ask_gpt",
        lambda *args, **kwargs: {"overall_summary": "Issues found", "issues": fake_issues},
    )
    monkeypatch.setattr(
        translation_review,
        "load_key",
        lambda k, default=None: "en" if k == "target_language" else default,
    )
    repaired, report = translation_review.review_and_repair_translation(
        df,
        terminology={"topic": "test", "speaker_profiles": []},
        report_path=report_path,
    )
    assert report["status"] == "failed"
    assert repaired.iloc[0]["Translation"] == "I will give you fifty tomorrow."

def test_clean_suggested_fix_rejects_meta_explanations():
    meta1 = "Confirm speaker assignment. If indeed S01: 'I said explain what surprise means.' If S02: 'No need.'"
    assert translation_review.clean_suggested_fix(meta1) == ""
    assert translation_review.is_meta_explanation(meta1) is True

    meta2 = "'Tell me what a fucking surprise is!' (maintains profanity but highlights character inconsistency)"
    assert translation_review.clean_suggested_fix(meta2) == "Tell me what a fucking surprise is!"


def test_review_fails_when_final_round_repairs_remain_unverified(tmp_path: Path, monkeypatch):
    report_path = tmp_path / "review_report.json"
    df = pd.DataFrame([{"LineID": "L00001", "Speaker": "S01", "Source": "你好", "Translation": "Hello."}])
    call_idx = 0
    def mock_ask(*args, **kwargs):
        nonlocal call_idx
        call_idx += 1
        return {"issues": [{"line_id": "L00001", "severity": "critical", "suggested_fix": f"Fix attempt {call_idx}", "problem": "Still incorrect."}]}

    monkeypatch.setattr(translation_review, "load_key", lambda k, default=None: "en" if k == "target_language" else default)
    monkeypatch.setattr(translation_review, "ask_gpt", mock_ask)

    result, report = translation_review.review_and_repair_translation(
        df, terminology={}, report_path=report_path, max_repair_rounds=2
    )
    assert report["status"] == "failed"
    assert report["status"] == "failed" and "unresolved issues remain" in report["error"]


def test_final_audit_explicit_failure_sets_failed_status(tmp_path: Path, monkeypatch):
    report_path = tmp_path / 'review_report.json'
    df = pd.DataFrame([{'LineID': 'L00001', 'Speaker': 'S01', 'Source': '你好', 'Translation': 'Hello.'}])
    calls = []
    def mock_ask(prompt, **kwargs):
        calls.append(prompt)
        if len(calls) == 1:
            return {'issues': [{'line_id': 'L00001', 'severity': 'critical', 'suggested_fix': 'Hi.', 'problem': 'Use informal'}]}
        return {'status': 'failed', 'issues': [], 'overall_summary': 'Service unavailable'}

    monkeypatch.setattr(translation_review, 'load_key', lambda k, default=None: 'en' if k == 'target_language' else default)
    monkeypatch.setattr(translation_review, 'ask_gpt', mock_ask)
    _, report = translation_review.review_and_repair_translation(df, terminology={}, report_path=report_path, max_repair_rounds=1)
    assert report['status'] == 'failed'
    assert 'Service unavailable' in report['error']


def test_unresolved_warning_without_fix_sets_failed_status(tmp_path: Path, monkeypatch):
    report_path = tmp_path / 'review_report.json'
    df = pd.DataFrame([{'LineID': 'L00001', 'Speaker': 'S01', 'Source': '你好', 'Translation': 'Hello.'}])
    monkeypatch.setattr(translation_review, 'load_key', lambda k, default=None: 'en' if k == 'target_language' else default)
    monkeypatch.setattr(
        translation_review, 'ask_gpt',
        lambda *a, **kw: {'issues': [{'line_id': 'L00001', 'severity': 'warning', 'suggested_fix': '', 'problem': 'Tone ambiguity'}]}
    )
    _, report = translation_review.review_and_repair_translation(df, terminology={}, report_path=report_path, max_repair_rounds=2)
    assert report['status'] == 'failed'


def test_unreviewed_final_warning_rewrite_fails_status(tmp_path: Path, monkeypatch):
    report_path = tmp_path / "report.json"
    df = pd.DataFrame([{"LineID": "L1", "Speaker": "S01", "Source": "不要开门", "Translation": "Don't open the door."}])
    responses = [
        {"issues": [{"line_id": "L1", "severity": "critical", "problem": "wording", "suggested_fix": "Do not open the door."}]},
        {"issues": [{"line_id": "L1", "severity": "warning", "problem": "untrusted proposed rewrite", "suggested_fix": "Open the door."}]},
    ]
    monkeypatch.setattr(translation_review, "load_key", lambda k, default=None: "en" if k == "target_language" else default)
    monkeypatch.setattr(translation_review, "ask_gpt", lambda *a, **kw: responses.pop(0))
    result, report = translation_review.review_and_repair_translation(df, {}, report_path=report_path, max_repair_rounds=1)
    assert report["status"] == "failed"
    assert "unresolved issues remain in final verification audit" in report["error"]
    assert result.iloc[0]["Translation"] == "Do not open the door."


def test_final_audit_clean_pass_without_issues(tmp_path: Path, monkeypatch):
    report_path = tmp_path / "report.json"
    df = pd.DataFrame([{"LineID": "L1", "Speaker": "S01", "Source": "不要开门", "Translation": "Don't open the door."}])
    responses = [
        {"issues": [{"line_id": "L1", "severity": "critical", "problem": "wording", "suggested_fix": "Do not open the door."}]},
        {"issues": []},
    ]
    monkeypatch.setattr(translation_review, "load_key", lambda k, default=None: "en" if k == "target_language" else default)
    monkeypatch.setattr(translation_review, "ask_gpt", lambda *a, **kw: responses.pop(0))
    result, report = translation_review.review_and_repair_translation(df, {}, report_path=report_path, max_repair_rounds=1)
    assert report["status"] == "pass"
    assert report["error"] is None
    assert result.iloc[0]["Translation"] == "Do not open the door."
