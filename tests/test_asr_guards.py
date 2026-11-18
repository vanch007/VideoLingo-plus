import shutil
import subprocess

import pytest

from core.providers.asr_guards import assess, is_repetitive, is_sparse, script_mismatch


def test_repetition_loop_detected_but_ordinary_repeats_pass():
    loop = "thank you for watching " * 12
    assert is_repetitive(loop)
    assert "repetition_loop" in assess(loop, 60, "en")
    assert not is_repetitive("no no no no, that is not what I meant at all, " + "x" * 5)
    normal = "The quick brown fox jumps over the lazy dog while the band plays on and on tonight."
    assert not is_repetitive(normal)


def test_chinese_loop_detected():
    assert is_repetitive("我们开始吧" * 20)


def test_sparse_needs_active_audio():
    assert is_sparse("hi", 60, 50)
    assert not is_sparse("hi", 60, 5)          # mostly quiet: little text is expected
    assert not is_sparse("hi", 10, 10)          # too short to judge
    assert not is_sparse("hi", 60, None)        # unmeasurable
    assert not is_sparse("x" * 400, 60, 50)


def test_script_mismatch():
    assert script_mismatch("这是中文" * 40, "en") == "wrong_script"
    assert script_mismatch("hello world " * 20, "zh") == "wrong_script"
    assert script_mismatch("这是中文的句子" * 30, "ja") == "japanese_without_kana"
    assert script_mismatch("これは日本語のぶんしょうです" * 20, "zh") == "chinese_with_kana"
    assert script_mismatch("hello world " * 20, "en") is None
    assert script_mismatch("short", "zh") is None  # too few letters to judge


FFMPEG = shutil.which("ffmpeg")


@pytest.mark.skipif(not FFMPEG, reason="ffmpeg required")
def test_cut_point_finds_pause_under_background_music(tmp_path):
    from core.all_whisper_methods.audio_preprocess import find_cut_point, active_seconds
    import numpy as np
    import soundfile as sf
    wav = tmp_path / "a.wav"
    t = np.arange(9 * 16000) / 16000
    # 0-4s and 5-9s loud, 4-5s a quiet pause; a constant music bed underneath would defeat
    # silencedetect -30dB, but the pause is still the quietest moment.
    gain = np.where((t >= 4) & (t < 5), 0.02, 0.3)
    sf.write(str(wav), gain * np.sin(2 * np.pi * 300 * t) + 0.005 * np.sin(2 * np.pi * 90 * t), 16000)
    cut = find_cut_point(str(wav), 2.0, 7.0)
    assert 4.0 <= cut <= 5.0
    assert active_seconds(str(wav), 0, 9) == pytest.approx(9.0, abs=0.3)


def test_guarded_transcribe_retries_in_halves_and_flags_final(monkeypatch):
    from core import step2_whisperX as s2
    monkeypatch.setattr("core.all_whisper_methods.audio_preprocess.active_seconds", lambda *a: None)
    monkeypatch.setattr("core.all_whisper_methods.audio_preprocess.find_cut_point", lambda a, lo, hi: (lo + hi) / 2)
    monkeypatch.setattr("core.runtime_context.effective_source_language", lambda d="auto": "en")
    calls = []

    def ts(audio, start, end):
        calls.append((start, end))
        looping = (end - start) > 100  # the long window loops, halves are fine
        text = "thank you for watching " * 12 if looping else "A normal sentence about the weather today."
        return {"segments": [{"text": text}]}

    report = []
    out = s2.guarded_transcribe(ts, "x", 0.0, 180.0, report)
    assert calls == [(0.0, 180.0), (0.0, 90.0), (90.0, 180.0)]
    assert len(out) == 2 and report[0]["reasons"] == ["repetition_loop"] and not report[0]["final"]

    def always_bad(audio, start, end):
        return {"segments": [{"text": "thank you for watching " * 12}]}

    report = []
    out = s2.guarded_transcribe(always_bad, "x", 0.0, 100.0, report)
    assert len(out) == 1 and report[-1]["final"] is True  # kept, flagged, never dropped
