import json
from pathlib import Path

import pandas as pd
import pytest

from core.pipeline import checkpoint
from core.pipeline.runner import PipelineRun, PipelineStep, run_pipeline


def _table(rows):
    Path("output/log").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["Source", "Translation"]).to_excel(checkpoint.TRANSLATION_FILE, index=False)


@pytest.fixture
def flow(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(checkpoint, "load_key", lambda k, d=None: True if k == "pause_after_translate" else d)
    monkeypatch.setattr("core.pipeline.runner._record_step_manifest", lambda *a, **k: None)
    ran = []

    def translate():
        _table([("hello", "你好"), ("bye", "再见")])
        ran.append("translate")

    steps = [PipelineStep("translate", "t", action=translate),
             PipelineStep("split_subtitle", "s", action=lambda: ran.append("split"))]
    run = PipelineRun(run_id="r", profile="cinematic", source="en", target="zh", steps=[])
    return steps, run, ran


def test_pauses_after_translate_then_resumes_with_edits_recorded(flow):
    steps, run, ran = flow
    out = run_pipeline(run, steps, resume=False)
    assert out.status == "paused" and ran == ["translate"] and "Translation column" in out.error
    _table([("hello", "大家好"), ("bye", "再见")])
    # Resume: the translate step is already complete, only the later step is pending.
    steps2 = [PipelineStep("split_subtitle", "s", action=lambda: ran.append("split"))]
    run2 = PipelineRun(run_id="r", profile="cinematic", source="en", target="zh", steps=[])
    out2 = run_pipeline(run2, steps2, resume=False)
    assert ran == ["translate", "split"] and out2.status != "paused"
    edits = json.loads(checkpoint.EDITS_FILE.read_text())
    assert edits["edited_rows"] == [1]
    assert any(e["event"] == "manual_translation_review" for e in out2.events)


def test_changed_source_or_row_count_is_rejected_and_keeps_the_pause(flow):
    steps, run, ran = flow
    run_pipeline(run, steps, resume=False)
    _table([("hello", "你好"), ("CHANGED", "再见")])
    with pytest.raises(ValueError, match="Source column changed"):
        checkpoint.consume_pause()
    _table([("hello", "你好")])
    with pytest.raises(ValueError, match="number of rows"):
        checkpoint.consume_pause()
    _table([("hello", "你好"), ("bye", "")])
    with pytest.raises(ValueError, match="empty translation"):
        checkpoint.consume_pause()
    assert checkpoint.PAUSE_FILE.is_file()  # still paused until the file is right


def test_no_pause_when_disabled_or_last_step(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(checkpoint, "load_key", lambda k, d=None: d)
    assert not checkpoint.pause_requested("translate", True)
    monkeypatch.setattr(checkpoint, "load_key", lambda k, d=None: True if k == "pause_after_translate" else d)
    assert checkpoint.pause_requested("translate", True)
    assert not checkpoint.pause_requested("translate", False)
    assert not checkpoint.pause_requested("summarize", True)


def test_translation_style_is_added_to_prompts(monkeypatch):
    from core import local_translation as lt
    monkeypatch.setattr("core.config_utils.load_key", lambda k, d=None: "口语化，短句" if k == "translation_style" else d)
    assert "口语化，短句" in lt.translation_style_note()
    monkeypatch.setattr("core.config_utils.load_key", lambda k, d=None: "" if k == "translation_style" else d)
    assert lt.translation_style_note() == ""
