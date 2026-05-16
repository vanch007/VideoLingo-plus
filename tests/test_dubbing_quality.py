import pandas as pd

from core import dubbing_quality
from core.dubbing_quality import actual_expand_needed, apply_dubbing_budget_columns, evaluate_dubbing, natural_speed_rewrite_needed, parse_list
from core.step10_gen_audio import build_atempo_filter, retry_fast_speech_rows


def test_parse_list_handles_excel_repr():
    assert parse_list("[[1.0, 2.0], [2.0, 3.0]]") == [[1.0, 2.0], [2.0, 3.0]]
    assert parse_list("not a list") == []


def test_apply_dubbing_budget_columns_adds_duration_contract():
    df = pd.DataFrame([{
        "number": 1,
        "start_time": "00:00:01.000",
        "end_time": "00:00:03.000",
        "duration": 2.0,
        "tolerance": 0.5,
        "text": "xin chao",
    }])
    out = apply_dubbing_budget_columns(df)
    assert out.loc[0, "available_duration"] == 2.5
    assert out.loc[0, "target_duration"] == 2.5
    assert out.loc[0, "quality_mode"] in {"subtitle", "dubbing", "high_sync"}
    assert out.loc[0, "max_speed_factor"] > 0
    assert out.loc[0, "max_natural_speed_factor"] > 0


def test_build_atempo_filter_chains_out_of_range_values():
    assert build_atempo_filter(1.25) == "atempo=1.25"
    assert build_atempo_filter(2.5) == "atempo=2,atempo=1.25"
    assert build_atempo_filter(0.25) == "atempo=0.5,atempo=0.5"


def test_actual_expand_needed_detects_short_measured_audio(monkeypatch):
    def fake_load_key(key, default=None):
        if key == "dubbing_quality.mode":
            return "high_sync"
        if key == "dubbing_quality.min_duration_ratio":
            return 0.9
        if key == "speed_factor.min":
            return 0.8
        return default

    monkeypatch.setattr(dubbing_quality, "load_key", fake_load_key)

    assert actual_expand_needed({"real_dur": 2.0, "available_duration": 4.0})
    assert not actual_expand_needed({"real_dur": 3.0, "available_duration": 4.0})


def test_natural_speed_rewrite_needed_detects_fast_speech(monkeypatch):
    def fake_load_key(key, default=None):
        if key == "dubbing_quality.mode":
            return "high_sync"
        if key == "dubbing_quality.max_natural_speed_factor":
            return 1.12
        return default

    monkeypatch.setattr(dubbing_quality, "load_key", fake_load_key)

    assert natural_speed_rewrite_needed({"real_dur": 4.6, "available_duration": 4.0})
    assert not natural_speed_rewrite_needed({"real_dur": 4.4, "available_duration": 4.0})


def test_evaluate_dubbing_reports_under_duration_without_end_drift(monkeypatch, tmp_path):
    audio_dir = tmp_path / "segs"
    audio_dir.mkdir()
    wav = audio_dir / "1_0.wav"
    wav.write_bytes(b"fake")

    monkeypatch.setattr(dubbing_quality, "SEGS_DIR", str(audio_dir))
    monkeypatch.setattr(dubbing_quality, "get_audio_duration", lambda _: 2.0)

    def fake_load_key(key, default=None):
        if key == "dubbing_quality.mode":
            return "high_sync"
        if key == "dubbing_quality.min_duration_ratio":
            return 0.9
        if key == "dubbing_quality.max_end_drift":
            return 0.18
        if key == "dubbing_quality.min_audio_size":
            return 1
        return default

    monkeypatch.setattr(dubbing_quality, "load_key", fake_load_key)
    tasks = pd.DataFrame(
        [
            {
                "number": 1,
                "start_time": "00:00:00.000",
                "end_time": "00:00:04.000",
                "duration": 4.0,
                "available_duration": 4.0,
                "tolerance": 0.0,
                "text": "xin chao",
                "lines": ["xin chao"],
                "new_sub_times": [[0.0, 2.0]],
                "real_dur": 2.0,
            }
        ]
    )

    eval_df, summary = evaluate_dubbing(tasks)

    assert eval_df.loc[0, "reason"] == "under_duration"
    assert summary["quality_gate"]["reasons"]["under_duration"] == 1
    assert summary["asr_readback"] is False
    assert summary["asr_scored_rows"] == 0
    assert summary["avg_content_score"] is None


def test_evaluate_dubbing_marks_asr_readback_effective_only_with_scores(monkeypatch, tmp_path):
    audio_dir = tmp_path / "segs"
    audio_dir.mkdir()
    wav = audio_dir / "1_0.wav"
    wav.write_bytes(b"fake")

    monkeypatch.setattr(dubbing_quality, "SEGS_DIR", str(audio_dir))
    monkeypatch.setattr(dubbing_quality, "get_audio_duration", lambda _: 2.0)

    def fake_load_key(key, default=None):
        if key == "dubbing_quality.mode":
            return "high_sync"
        if key == "dubbing_quality.asr_readback":
            return True
        if key == "dubbing_quality.min_audio_size":
            return 1
        return default

    monkeypatch.setattr(dubbing_quality, "load_key", fake_load_key)
    base_task = {
        "number": 1,
        "start_time": "00:00:00.000",
        "end_time": "00:00:02.000",
        "duration": 2.0,
        "available_duration": 2.0,
        "tolerance": 0.0,
        "text": "xin chao",
        "lines": ["xin chao"],
        "new_sub_times": [[0.0, 2.0]],
        "real_dur": 2.0,
    }

    _, without_scores = evaluate_dubbing(pd.DataFrame([base_task]))
    assert without_scores["asr_readback_configured"] is True
    assert without_scores["asr_readback"] is False

    with_scores = dict(base_task, asr_content_score=0.95, asr_leakage_score=0.02)
    _, summary = evaluate_dubbing(pd.DataFrame([with_scores]))
    assert summary["asr_readback"] is True
    assert summary["asr_scored_rows"] == 1
    assert summary["avg_content_score"] == 0.95
    assert summary["max_leakage_score"] == 0.02


def test_evaluate_dubbing_warns_on_fast_speech_factor(monkeypatch, tmp_path):
    audio_dir = tmp_path / "segs"
    audio_dir.mkdir()
    wav = audio_dir / "1_0.wav"
    wav.write_bytes(b"fake")

    monkeypatch.setattr(dubbing_quality, "SEGS_DIR", str(audio_dir))
    monkeypatch.setattr(dubbing_quality, "get_audio_duration", lambda _: 2.0)

    def fake_load_key(key, default=None):
        if key == "dubbing_quality.mode":
            return "high_sync"
        if key == "dubbing_quality.max_natural_speed_factor":
            return 1.12
        if key == "dubbing_quality.min_audio_size":
            return 1
        return default

    monkeypatch.setattr(dubbing_quality, "load_key", fake_load_key)
    tasks = pd.DataFrame(
        [
            {
                "number": 1,
                "start_time": "00:00:00.000",
                "end_time": "00:00:02.000",
                "duration": 2.0,
                "available_duration": 2.0,
                "tolerance": 0.0,
                "text": "xin chao",
                "lines": ["xin chao"],
                "new_sub_times": [[0.0, 2.0]],
                "real_dur": 2.0,
                "speed_factor": 1.2,
            }
        ]
    )

    eval_df, summary = evaluate_dubbing(tasks)

    assert eval_df.loc[0, "reason"] == "speech_rate_fast"
    assert summary["quality_gate"]["reasons"]["speech_rate_fast"] == 1
    assert summary["max_speed_factor"] == 1.2


def test_retry_fast_speech_rows_rewrites_and_regenerates(monkeypatch):
    class Quality:
        enabled = True
        max_natural_speed_factor = 1.12
        max_rewrite_rounds = 2

    calls = {}
    monkeypatch.setattr("core.step10_gen_audio.get_quality_config", lambda: Quality())
    monkeypatch.setattr("core.step10_gen_audio.load_key", lambda key, default=None: True if key == "rewrite_text_for_dubbing" else default)
    monkeypatch.setattr("core.step10_gen_audio.rewrite_task_lines", lambda row, reason, direction="shorten": ["short text"])
    monkeypatch.setattr("core.step10_gen_audio._delete_temp_files_for_row", lambda row: None)

    def fake_process(row, tasks_df):
        calls["text"] = row["text"]
        return int(row["number"]), 1.8

    monkeypatch.setattr("core.step10_gen_audio.process_row", fake_process)
    tasks = pd.DataFrame(
        [
            {
                "number": 1,
                "start_time": "00:00:00.000",
                "end_time": "00:00:02.000",
                "duration": 2.0,
                "text": "long text",
                "lines": ["long text"],
                "real_dur": 2.6,
                "speed_factor": 1.25,
                "dubbing_rewrite_rounds": 0,
            }
        ]
    )

    out, changed = retry_fast_speech_rows(tasks)

    assert changed
    assert calls["text"] == "short text"
    assert out.loc[0, "rewrite_reason"] == "speech_rate_fast"
    assert out.loc[0, "real_dur"] == 1.8
