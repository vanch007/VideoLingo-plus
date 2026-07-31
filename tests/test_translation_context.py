import json

import pandas as pd
import pytest

from core import translation_context
from core import prompts_storage
from core.prompts_storage import generate_shared_prompt, get_summary_prompt
from core.step4_1_summarize import apply_stt_exact_replacements, validate_stt_correction
from core import step4_2_translate_all
from core.step4_2_translate_all import (
    apply_translation_exact_replacements,
    assemble_translation_results,
)


def _rows():
    return pd.DataFrame(
        {
            "line_id": ["L00001", "L00002", "L00003", "L00004"],
            "text": ["Same line", "Reply", "Same line", "Closing"],
            "speaker": ["S01", "S02", "S01", "S02"],
            "start": [0.0, 1.0, 2.0, 3.0],
            "end": [0.9, 1.9, 2.9, 3.9],
        }
    )


def test_load_speaker_aware_rows_writes_coverage_report(tmp_path, monkeypatch):
    sentences = tmp_path / "sentences.txt"
    words = tmp_path / "words.xlsx"
    report = tmp_path / "speaker_context.json"
    sentences.write_text("Hello\nThere\n", encoding="utf-8")
    pd.DataFrame({"text": ["Hello", "There"]}).to_excel(words, index=False)
    monkeypatch.setattr(
        translation_context,
        "get_sentence_timestamps",
        lambda *_: [(0.0, 0.8, "S01"), (0.9, 1.7, "S02")],
    )
    monkeypatch.setattr(
        translation_context,
        "load_key",
        lambda key, default=None: {
            "translation_context.require_speakers": True,
            "translation_context.min_line_coverage": 1.0,
            "translation_context.previous_lines": 4,
            "translation_context.next_lines": 3,
        }.get(key, default),
    )

    rows, metrics = translation_context.load_speaker_aware_rows(sentences, words, report)

    assert rows[["line_id", "speaker"]].values.tolist() == [
        ["L00001", "S01"],
        ["L00002", "S02"],
    ]
    assert metrics.coverage == 1.0
    assert metrics.speaker_transitions == 1
    assert json.loads(report.read_text(encoding="utf-8"))["known_speaker_lines"] == 2


def test_load_speaker_aware_rows_fails_closed_below_required_coverage(tmp_path, monkeypatch):
    sentences = tmp_path / "sentences.txt"
    words = tmp_path / "words.xlsx"
    sentences.write_text("Hello\nThere\n", encoding="utf-8")
    pd.DataFrame({"text": ["Hello", "There"]}).to_excel(words, index=False)
    monkeypatch.setattr(
        translation_context,
        "get_sentence_timestamps",
        lambda *_: [(0.0, 0.8, "S01"), (0.9, 1.7, None)],
    )
    monkeypatch.setattr(
        translation_context,
        "load_key",
        lambda key, default=None: {
            "translation_context.require_speakers": True,
            "translation_context.min_line_coverage": 1.0,
        }.get(key, default),
    )

    with pytest.raises(RuntimeError, match="speaker coverage"):
        translation_context.load_speaker_aware_rows(sentences, words, tmp_path / "report.json")


def test_batches_and_async_results_preserve_duplicate_line_identity(monkeypatch):
    rows = _rows()
    monkeypatch.setattr(translation_context, "load_key", lambda key, default=None: 1)
    batches = translation_context.split_speaker_aware_batches(rows, max_lines=2)

    output = assemble_translation_results(
        [(1, ["T3", "T4"]), (0, ["T1", "T2"])],
        batches,
        rows,
    )

    assert output["LineID"].tolist() == ["L00001", "L00002", "L00003", "L00004"]
    assert output["Source"].tolist() == ["Same line", "Reply", "Same line", "Closing"]
    assert output["Translation"].tolist() == ["T1", "T2", "T3", "T4"]
    assert "[S01]" in batches[0]["speaker_context"]


def test_verified_translation_replacement_is_keyed_by_exact_source(monkeypatch):
    monkeypatch.setattr(
        step4_2_translate_all,
        "load_key",
        lambda key, default=None: {
            "translation_exact_replacements": {
                "敢问九筒大哥何方神圣": "May I ask, Brother Nine Dots, who exactly are you?"
            }
        }.get(key, default),
    )
    output = pd.DataFrame(
        {
            "Source": ["敢问九筒大哥何方神圣", "鄙人张麻子"],
            "Translation": ["Who are you, Nine Bamboo?", "I'm Zhang Mazi."],
        }
    )

    corrected, count = apply_translation_exact_replacements(output)

    assert count == 1
    assert corrected["Translation"].tolist() == [
        "May I ask, Brother Nine Dots, who exactly are you?",
        "I'm Zhang Mazi.",
    ]


def test_prompts_use_speaker_roles_but_forbid_speaker_ids_in_output():
    summary = get_summary_prompt("[L00001][S01] Hello", speaker_ids=["S01", "S02"])
    shared = generate_shared_prompt(
        "[L00000][S02] Before",
        "[L00003][S02] After",
        '{"speaker_profiles": [{"speaker_id": "S01", "role": "official"}]}',
        None,
        "[L00001][S01] Hello",
    )

    assert "speaker_profiles" in summary
    assert "S01" in summary and "S02" in summary
    assert "never invent names, gender, or biography" in summary
    assert "Never output speaker IDs" in shared
    assert "pronouns" in shared and "honorifics" in shared


def test_expressive_prompt_forbids_dropping_core_predicates(monkeypatch):
    monkeypatch.setattr(
        prompts_storage,
        "load_key",
        lambda key, default=None: {
            "target_language": "en",
            "whisper.detected_language": "zh",
        }.get(key, default),
    )
    prompt = prompts_storage.get_prompt_expressiveness(
        {
            "1": {
                "origin": "不刮穷鬼的钱你收谁的呀",
                "direct": "If you don't tax the poor, who do you tax?",
            }
        },
        "不刮穷鬼的钱你收谁的呀",
        "",
    )

    assert "MEANING PRESERVATION OVERRIDES BREVITY" in prompt
    assert "core proposition" in prompt
    assert "who do you tax?" in prompt


def test_existing_translation_requires_speaker_columns_and_profiles(tmp_path, monkeypatch):
    result_path = tmp_path / "translation_results.xlsx"
    terminology_path = tmp_path / "terminology.json"
    monkeypatch.setattr(step4_2_translate_all, "TRANSLATION_RESULTS_FILE", str(result_path))
    monkeypatch.setattr(step4_2_translate_all, "TERMINOLOGY_FILE", str(terminology_path))
    terminology_path.write_text(json.dumps({"speaker_profiles": [{"speaker_id": "S01"}]}))
    pd.DataFrame({"Source": ["Hello"], "Translation": ["你好"]}).to_excel(result_path, index=False)

    assert step4_2_translate_all.existing_translation_is_speaker_aware() is False

    pd.DataFrame(
        {
            "LineID": ["L00001"],
            "Speaker": ["S01"],
            "Source": ["Hello"],
            "Translation": ["你好"],
        }
    ).to_excel(result_path, index=False)
    assert step4_2_translate_all.existing_translation_is_speaker_aware() is True


def test_stt_correction_rejects_character_count_drift():
    original = ["犀利二零一零", "站着正前"]

    invalid = validate_stt_correction(
        {"corrected_lines": ["零一零", "站着挣钱"]}, original
    )
    valid = validate_stt_correction(
        {"corrected_lines": ["西历二零一零", "站着挣钱"]}, original
    )

    assert invalid["status"] == "error"
    assert "Character count mismatch" in invalid["message"]
    assert valid["status"] == "success"


def test_verified_stt_replacements_can_restore_missing_transcript_text(monkeypatch):
    monkeypatch.setattr(
        "core.step4_1_summarize.load_key",
        lambda key, default=None: {
            "stt_correction_exact_replacements": {
                "巧立明目": "巧立名目",
                "鄙人张麻": "鄙人张麻子",
            }
        }.get(key, default),
    )

    corrected, count = apply_stt_exact_replacements(["还得巧立明目", "鄙人张麻"])

    assert corrected == ["还得巧立名目", "鄙人张麻子"]
    assert count == 2
