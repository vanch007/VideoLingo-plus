import json
from types import SimpleNamespace

import pandas as pd
import pytest

from core.dialogue_segmentation import group_dialogue_words, is_lexically_joined_span
from core.step6_generate_final_timeline import get_sentence_timestamps
from core.subtitle_quality import subtitle_quality_report


def words():
    return [{"text": "什么叫惊", "speaker": "S01", "start": 0.0, "end": 0.8},
            {"text": "喜", "speaker": "S02", "start": 0.8, "end": 1.0}]


def tokenizer(text="惊喜", idx=3):
    return SimpleNamespace(make_doc=lambda source: [SimpleNamespace(text=text, idx=idx)])


@pytest.mark.parametrize("gap", [0.0, 0.16])
def test_lexical_join_keeps_word_labels_and_identity_uncertainty(tmp_path, monkeypatch, gap):
    rows = words()
    rows[1]["start"] += gap
    rows[1]["end"] += gap
    before = json.dumps(rows)
    groups, report = group_dialogue_words(rows, tokenizer(), "")
    assert json.dumps(rows) == before
    assert groups[0]["words"] == ["什么叫惊", "喜"]
    assert groups[0]["speaker"] is None
    assert groups[0]["acoustic_speakers"] == ["S01", "S02"]
    assert report["status"] == "needs_review"
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "output/log/lexical_boundary_review.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(report))
    assert get_sentence_timestamps(pd.DataFrame(rows), pd.DataFrame({"Source": ["什么叫惊喜"]}))[0] == (0., 1. + gap, None)
    rows[1]["text"] = "欢"
    with pytest.raises(ValueError, match="source changed"):
        is_lexically_joined_span(rows, 0, 1, path)


@pytest.mark.parametrize("scenario", ["pause", "long_token", "latin"])
def test_unverified_or_real_turn_boundaries_remain_separate(scenario):
    rows = words()
    nlp, joiner = tokenizer(), ""
    if scenario == "pause":
        rows[1].update(start=2., end=2.2)
    elif scenario == "long_token":
        nlp = tokenizer("什么叫惊喜", 0)
    else:
        joiner = " "
    groups, report = group_dialogue_words(rows, nlp, joiner)
    assert len(groups) == 2
    assert not report["joined_boundaries"]


def test_missing_or_pending_quality_is_never_accepted(tmp_path):
    report = subtitle_quality_report(tmp_path)
    assert report["status"] == "pending"
    assert report["automatic_gate_passed"] is False
    assert report["accepted"] is False
    (tmp_path / "translation_review.json").write_text('{"status":"failed"}')
    assert subtitle_quality_report(tmp_path)["status"] == "fail"
