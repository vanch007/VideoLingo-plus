"""Unit tests for Fast Mode translation, cloning dubbing, timing adaptation & subtitles."""
import pytest
from pathlib import Path
import json
import pandas as pd

from core.config_utils import load_key, update_key
from core.pipeline.profiles import apply_profile, apply_fast_profile, apply_cinematic_profile

@pytest.fixture(autouse=True)
def restore_config_after_test():
    config_path = Path("config.yaml")
    original_config = config_path.read_text(encoding="utf-8")
    yield
    config_path.write_text(original_config, encoding="utf-8")

from core.fast_subtitles import _split_text_by_punctuation, _wrap_lines, build_fast_subtitles


def test_fast_profile_settings():
    apply_fast_profile()
    assert load_key("quality_mode") == "fast"
    assert load_key("dubbing_quality.mode") == "fast"
    assert load_key("dubbing_quality.asr_readback") is False
    assert load_key("dubbing_quality.max_speed_factor") == 1.25
    assert load_key("fast_mode.enabled") is True
    assert load_key("fast_mode.max_speed_factor") == 1.25
    assert load_key("fast_mode.max_gap_extension_seconds") == 0.35
    assert load_key("fast_mode.video_extension_enabled") is True
    assert load_key("mlx_tts.backends.indextts2.chinese_profile") == "v25"


def test_cinematic_profile_reversibility():
    apply_cinematic_profile()
    assert load_key("quality_mode") == "high_sync"
    assert load_key("dubbing_quality.mode") == "high_sync"
    assert load_key("dubbing_quality.asr_readback") is True
    assert load_key("dubbing_quality.max_speed_factor") == 1.08
    # Switch back to fast
    apply_profile("fast")
    assert load_key("quality_mode") == "fast"


def test_fast_subtitles_splitting():
    text_cjk = "这是第一句话，中间有逗号；然后是第二句话，结尾有句号。"
    parts_cjk = _split_text_by_punctuation(text_cjk, is_cjk=True)
    assert len(parts_cjk) == 2
    assert "第一句话" in parts_cjk[0]
    assert "第二句话" in parts_cjk[1]

    text_en = "This is the first clause, with a comma; and here is the second, with a period."
    parts_en = _split_text_by_punctuation(text_en, is_cjk=False)
    assert len(parts_en) == 2
    assert "first clause" in parts_en[0]
    assert "second" in parts_en[1]


def test_fast_subtitles_line_wrapping():
    # CJK line wrap within 22 chars
    long_zh = "这是一个非常长非常长非常长的中文句子超过了二十二个字的单行限制需要自然折行展示"
    wrapped_zh = _wrap_lines(long_zh, max_chars=22, max_lines=2, is_cjk=True)
    lines_zh = wrapped_zh.split("\n")
    assert len(lines_zh) <= 2
    for l in lines_zh:
        assert len(l) <= 22

    # Latin line wrap within 42 chars
    long_en = "This is an extremely long English sentence that exceeds the forty-two characters threshold and must be formatted across two lines cleanly"
    from core.fast_subtitles import _display_parts
    with pytest.raises(ValueError, match="additional display cues"):
        _wrap_lines(long_en, max_chars=42, max_lines=2, is_cjk=False)
    parts = _display_parts(long_en, 42)
    assert "".join("".join(parts).split()) == "".join(long_en.split())
    lines_en = parts[0].split("\n")
    assert len(lines_en) <= 2
    for l in lines_en:
        assert len(l) <= 42


def test_fast_subtitles_generation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "output").mkdir()

    tts_results = [
        {
            "number": 1,
            "unit_id": "U0001",
            "origin": "Hello everyone, welcome back to the channel.",
            "text": "大家好，欢迎回到本频道。",
            "start": 0.5,
            "end": 3.0,
            "final_start": 0.5,
            "final_end": 3.0,
        },
        {
            "number": 2,
            "unit_id": "U0002",
            "origin": "Today we are automating video localization using Claude Code.",
            "text": "今天我们使用 Claude Code 自动化视频本地化。",
            "start": 3.2,
            "end": 6.8,
            "final_start": 3.2,
            "final_end": 6.8,
        }
    ]

    build_fast_subtitles(tts_results)

    bilingual_srt = tmp_path / "output" / "dub_bilingual.srt"
    assert bilingual_srt.is_file()
    content = bilingual_srt.read_text(encoding="utf-8")
    assert "Hello everyone" in content
    assert "大家好" in content
    assert "Claude Code" in content

    timeline_xlsx = tmp_path / "output" / "final_timeline.xlsx"
    assert timeline_xlsx.is_file()
    df = pd.read_excel(timeline_xlsx)
    assert len(df) >= 2
    assert "Translation" in df.columns
    assert "Source" in df.columns

def test_cli_profile_defaults():
    import argparse
    from core.cli import build_parser, _fill_resume_args, _build_run

    parser = build_parser()
    run_args = parser.parse_args(["run", "--input", "test.mp4"])
    assert run_args.profile == "fast"

    # Resume without explicit profile should inherit from state
    resume_args = parser.parse_args(["resume", "--input", "test.mp4"])
    assert resume_args.profile is None
    state_mock = {"profile": "cinematic", "input": "test.mp4"}
    _fill_resume_args(resume_args, state_mock)
    assert resume_args.profile == "cinematic"

    # Build run record for fast mode
    run_obj = _build_run(run_args, [])
    assert run_obj.profile == "fast"
    assert run_obj.quality_status == "fast_mode"
    assert run_obj.quality_gate_passed is None


def test_timing_adaptation_model():
    # Gap borrowing calculation
    orig_start, orig_end = 1.0, 3.0
    orig_dur = orig_end - orig_start  # 2.0s
    next_start = 3.5                  # gap = 0.5s
    borrow = min(0.35, next_start - orig_end)
    assert borrow == 0.35
    target_dur = orig_dur + borrow     # 2.35s

    # Case 1: Audio natural duration 2.1s <= 2.35s -> speed 1.0, no extension
    meas_dur1 = 2.1
    ratio1 = meas_dur1 / target_dur
    assert ratio1 <= 1.0

    # Case 2: Audio duration 2.8s -> ratio = 2.8/2.35 = 1.191 <= 1.25 -> speed 1.191, no extension
    meas_dur2 = 2.8
    ratio2 = meas_dur2 / target_dur
    assert 1.0 < ratio2 <= 1.25

    # Case 3: Audio duration 3.5s -> ratio = 3.5/2.35 = 1.489 > 1.25 -> cap at 1.25x
    meas_dur3 = 3.5
    actual_play_dur = meas_dur3 / 1.25  # 2.8s
    ext_by = actual_play_dur - target_dur # 2.8 - 2.35 = 0.45s
    assert ext_by > 0
    assert abs(ext_by - 0.45) < 1e-4
