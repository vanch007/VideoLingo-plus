"""Signal-chain regressions with real FFmpeg and synthetic media (no models, no network)."""
import os
import shutil
import subprocess

import pytest

from core.all_whisper_methods import audio_preprocess as ap
from core.step12_merge_dub_to_vid import (
    loudness_gain_db, measure_final_audio_loudness, normalize_audio_volume,
)

FFMPEG = shutil.which("ffmpeg")
pytestmark = pytest.mark.skipif(not FFMPEG, reason="ffmpeg required")


def _run(*args):
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", *args], check=True)


def _duration(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "csv=p=0", path], capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


def test_gain_respects_true_peak_ceiling_and_silence():
    assert loudness_gain_db(-30.0, -10.0, -18.0) == pytest.approx(9.0)    # loudness target wins
    assert loudness_gain_db(-30.0, -3.0, -18.0) == pytest.approx(2.0)     # peak ceiling wins
    assert loudness_gain_db(-10.0, -1.0, -18.0) == pytest.approx(-8.0)    # attenuation uncapped
    assert loudness_gain_db(None, None, -18.0) == 0.0                     # unmeasurable


def test_normalize_is_fixed_gain_and_ignores_silence(tmp_path):
    tone = tmp_path / "tone.wav"
    padded = tmp_path / "padded.wav"
    _run("-f", "lavfi", "-i", "sine=f=440:d=4", "-af", "volume=-30dB", str(tone))
    _run("-i", str(tone), "-af", "apad=pad_dur=20", str(padded))
    a, b = tmp_path / "a.wav", tmp_path / "b.wav"
    normalize_audio_volume(str(tone), str(a))
    normalize_audio_volume(str(padded), str(b))
    la, lb = measure_final_audio_loudness(str(a)), measure_final_audio_loudness(str(b))
    assert abs(la["integrated_lufs"] - lb["integrated_lufs"]) < 0.3
    assert la["true_peak_dBTP"] <= -0.9


def test_normalize_silence_is_passed_through(tmp_path):
    silent = tmp_path / "s.wav"
    out = tmp_path / "o.wav"
    _run("-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono", "-t", "1", str(silent))
    assert normalize_audio_volume(str(silent), str(out)) == str(out)


def test_normalize_raises_on_ffmpeg_failure(tmp_path):
    with pytest.raises(Exception):
        normalize_audio_volume(str(tmp_path / "missing.wav"), str(tmp_path / "o.wav"))


def test_aligned_extraction_follows_presentation_clock(tmp_path):
    """Audio with repeated timestamp gaps: plain decode is shorter/longer than the clock."""
    seg = tmp_path / "seg.wav"
    _run("-f", "lavfi", "-i", "sine=f=300:d=1", "-ar", "16000", "-ac", "1", str(seg))
    # Two 1 s chunks 1 s apart on the timeline (gap), muxed with explicit timestamps.
    gap = tmp_path / "gap.mkv"
    _run("-i", str(seg), "-itsoffset", "2", "-i", str(seg),
         "-filter_complex", "[0:a][1:a]amix=inputs=2:duration=longest", "-c:a", "pcm_s16le", str(gap))
    aligned = tmp_path / "aligned.wav"
    _run("-i", str(gap), "-vn", "-af", ap.ALIGN_FILTER, str(aligned))
    assert _duration(str(aligned)) == pytest.approx(_duration(str(gap)), abs=0.05)
    assert ap.ALIGN_FILTER == "aresample=async=1:first_pts=0"


def test_all_extraction_commands_carry_align_filter(tmp_path, monkeypatch):
    calls = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ap, "AUDIO_DIR", str(tmp_path / "audio"))
    monkeypatch.setattr(ap, "RAW_AUDIO_FILE", str(tmp_path / "audio" / "raw.mp3"))
    monkeypatch.setattr(ap.subprocess, "run", lambda cmd, **k: calls.append(cmd))
    ap.convert_video_to_audio("in.mp4")
    ap.compress_audio("in.wav", str(tmp_path / "out.mp3"))
    assert len(calls) == 3
    assert all(ap.ALIGN_FILTER in c for c in calls)
