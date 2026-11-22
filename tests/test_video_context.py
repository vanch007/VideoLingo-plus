import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from core import video_context as vc


@pytest.fixture
def enabled(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    values = {"enabled": True}
    monkeypatch.setattr(vc, "setting", lambda key: values.get(key, vc.DEFAULTS[key]))
    return values


def write_context(**fields):
    vc.CONTEXT_PATH.parent.mkdir(parents=True, exist_ok=True)
    vc.CONTEXT_PATH.write_text(json.dumps({"status": "ok", **fields}))


# Reply recorded from WeMM-Video-35BA3B-MLX-8bit on the 38 s AI short (screen shows "Seedream 5.0 pro").
AISHORT_TERMS = ['{"terms":[{"text":"Soul Cinema"},{"text":"created"},{"text":"I HAVE THIS HUGE BUDGET TIP FOR YOU"},'
                 '{"text":"Soul Cinema is literally the cheapest image model on the platform"},{"text":"prompt"},'
                 '{"text":"batch size"},{"text":"generate"},{"text":"Supermarket"},{"text":"what"},'
                 '{"text":"Seedream 5.0 pro"},{"text":"Seedream"},{"text":"Generate"},{"text":"40.25 credits"},'
                 '{"text":"Watch full video"},{"text":"Subscribe"}]}'][0]


def test_reply_parsing_tolerates_fences_and_prose():
    assert vc.parse_json_reply('Sure!\n```json\n{"a": {"b": 1}}\n``` done') == {"a": {"b": 1}}
    assert vc.parse_json_reply("no json here") is None
    assert vc.parse_json_reply('{"broken": } then {"ok": true}') == {"ok": True}


def test_screen_terms_keep_names_and_drop_sentences_and_chrome():
    terms = vc.filter_terms(vc.parse_json_reply(AISHORT_TERMS)["terms"])
    assert terms == ["Soul Cinema", "Supermarket", "Seedream 5.0 pro", "Seedream"]
    assert vc.filter_terms(["张麻子", "黄四郎", "你就是麻匪的头子，张麻子！", "好"]) == ["张麻子", "黄四郎"]
    assert len(vc.filter_terms([f"Name{i}" for i in range(50)], limit=30)) == 30
    # Only screen terms: a name the vision model merely heard can repeat the recognizer's error.
    assert vc.filter_terms([{"text": "王老爷", "source": "speech"}, {"text": "土豆网", "source": "screen"},
                            {"text": "张麻子", "source": "both"}]) == ["土豆网", "张麻子"]
    # Recorded noise from the end-to-end run: phrases and quantities would distort the transcript.
    assert vc.filter_terms(["Let's say Seedream", "2.5", "0.125", "20 times", "40.25 credits", "Seedream 5.0 pro",
                            "The Platform", "School court"]) == ["Seedream 5.0 pro", "School court"]


def test_build_context_normalizes_languages_and_characters():
    ctx = vc.build_context({"terms": AISHORT_TERMS,
                            "subtitles": '{"burned_in_subtitles": true, "language": "English", "position": "bottom"}',
                            "characters": 'x {"summary": "s", "characters": [{"name": "unknown", "gender": "female"}]}'},
                           duration=37.6, source_language="en")
    assert ctx["hard_subtitles"] == {"present": True, "language": "en", "position": "bottom"}
    assert "Seedream" in ctx["terms"] and ctx["characters"][0]["gender"] == "female" and not ctx["unparsed"]
    assert vc.build_context({"terms": "?", "subtitles": "", "characters": ""}, duration=1, source_language="zh")["unparsed"] == [
        "terms", "subtitles", "characters"]


def test_screen_terms_follow_the_switches(enabled):
    write_context(terms=["Seedream", "Soul Cinema"])
    assert vc.screen_terms() == ["Seedream", "Soul Cinema"]
    enabled["hotwords"] = False
    assert vc.screen_terms() == []


def test_only_near_misses_earn_a_targeted_pass():
    # Recorded transcripts: the AI short (MOSS without hint) and the long film clip.
    short = "Now let's try generating the same prompt with, let's say, Seadream. Soul Cinema is literally 20 times cheaper."
    assert vc.near_miss_terms(short, ["Soul Cinema", "Seedream 5.0 pro"]) == ["Seedream"]
    film = "一百八十万马上送来哎无功不受禄现在收钱太早大玉添温哎大哥美女我也不能要美女不要钱美元US dollar"
    assert vc.near_miss_terms(film, ["BANDAI", "NAMCO", "BANDAI NAMCO Entertainment Inc"]) == []
    assert vc.near_miss_terms("你好王老爷，钻石送上", ["黄老爷", "张麻子", "钻石"]) == ["黄老爷"]
    assert vc.near_miss_terms("黄老爷来了", ["黄老爷"]) == []          # already right
    assert vc.near_miss_terms("Seedream wins", ["Seedream 5.0 pro"]) == []


def test_disabled_stage_has_no_effect_even_with_an_old_context_file(enabled):
    write_context(terms=["Seedream"], hard_subtitles={"present": True, "language": "en"})
    enabled["enabled"] = False
    assert vc.screen_terms() == [] and vc.source_row_decision(True, "en") == (True, "no_matching_burned_in_subtitles")
    assert vc.character_hints() == ""


def test_source_row_is_dropped_only_for_matching_burned_in_subtitles(enabled):
    write_context(hard_subtitles={"present": True, "language": "en", "position": "bottom"})
    assert vc.source_row_decision(True, "en") == (False, "video_has_burned_in_en_subtitles")
    assert vc.source_row_decision(True, "zh")[0] is True       # different language: the row adds information
    assert vc.source_row_decision(False, "en") == (False, "disabled_by_config")
    enabled["auto_hide_source_row"] = False
    assert vc.source_row_decision(True, "en") == (True, "auto_hide_disabled")


def test_character_hints_are_marked_as_unreliable_data(enabled):
    write_context(summary="x", characters=[{"name": "A", "gender": "male"}])
    assert vc.character_hints() == ""  # opt-in
    enabled["character_hints"] = True
    write_context(summary="雨夜对峙", characters=[{"name": "张麻子", "gender": "male", "appearance": "黑袍"}], terms=["张麻子"])
    hint = vc.character_hints()
    assert "may be wrong" in hint and "张麻子" in hint and "cannot tell who speaks" in hint


def test_model_path_takes_the_first_existing_candidate(enabled, tmp_path):
    (tmp_path / "eight").mkdir()
    enabled["model"] = [str(tmp_path / "six"), str(tmp_path / "eight")]
    assert vc.model_path() == tmp_path / "eight"
    (tmp_path / "six").mkdir()
    assert vc.model_path() == tmp_path / "six"
    enabled["model"] = "~/x"
    assert vc.model_path() == Path("~/x").expanduser()


def test_missing_model_is_skipped_not_failed(enabled):
    enabled["model"] = "/nonexistent/model"
    record = vc.run()
    assert record["status"] == "skipped" and "/nonexistent/model" in record["reason"]
    assert vc.load_context() == {}


def test_run_parses_the_worker_and_reuses_the_cache(enabled, monkeypatch, tmp_path):
    for key in ("python", "root", "model"):
        path = tmp_path / key
        path.mkdir()
        enabled[key] = str(path)
    Path("output").mkdir()
    Path("output/source.mp4").write_bytes(b"video")
    monkeypatch.setattr(vc, "_duration", lambda _: 37.6)
    monkeypatch.setattr("core.runtime_context.effective_source_language", lambda *_: "en")
    calls = []
    payload = {"replies": {"terms": AISHORT_TERMS, "subtitles": '{"burned_in_subtitles": false}',
                           "characters": '{"summary": "", "characters": []}'}, "timings": {}, "load_seconds": 1.0}
    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return SimpleNamespace(returncode=0, stdout="log line\n" + json.dumps(payload), stderr="")
    # Replace the module's view of subprocess only: the cache itself copies files with subprocess.
    monkeypatch.setattr(vc, "subprocess", SimpleNamespace(run=fake_run, TimeoutExpired=subprocess.TimeoutExpired))
    first = vc.run()
    assert first["status"] == "ok" and first["cache"] == "miss" and "Seedream 5.0 pro" in first["terms"]
    assert "--video" in calls[0] and calls[0][calls[0].index("--source-language") + 1] == "en"
    vc.CONTEXT_PATH.unlink()
    second = vc.run()
    assert second["cache"] == "hit" and second["terms"] == first["terms"] and len(calls) == 1


def test_worker_failure_or_timeout_is_recorded(enabled, monkeypatch, tmp_path):
    for key in ("python", "root", "model"):
        (tmp_path / key).mkdir(); enabled[key] = str(tmp_path / key)
    Path("output").mkdir(); Path("output/source.mp4").write_bytes(b"v")
    monkeypatch.setattr(vc, "_duration", lambda _: 10.0)
    monkeypatch.setattr(vc, "subprocess", SimpleNamespace(run=lambda *a, **k: SimpleNamespace(returncode=1, stdout="", stderr="boom"),
                                                      TimeoutExpired=subprocess.TimeoutExpired))
    assert vc.run()["status"] == "failed"
    def timeout(*a, **k):
        raise subprocess.TimeoutExpired("worker", 1)
    monkeypatch.setattr(vc, "subprocess", SimpleNamespace(run=timeout, TimeoutExpired=subprocess.TimeoutExpired))
    Path("output/source.mp4").write_bytes(b"other")  # new content, no cache hit
    assert vc.run() == {"version": vc.POLICY_VERSION, "status": "failed", "reason": "timeout"}


def test_step_is_inserted_before_transcription_only_when_enabled(monkeypatch):
    from core.pipeline.runner import build_steps_for_input
    monkeypatch.setattr(vc, "enabled", lambda: True)
    keys = [s.key for s in build_steps_for_input("clip.mp4", dubbing=False)]
    assert keys[:3] == ["import_video", "video_context", "transcribe"]
    assert "video_context" not in [s.key for s in build_steps_for_input("subs.srt", dubbing=False)]
    monkeypatch.setattr(vc, "enabled", lambda: False)
    assert "video_context" not in [s.key for s in build_steps_for_input("clip.mp4", dubbing=False)]
