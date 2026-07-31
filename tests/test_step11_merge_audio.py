import json
from types import SimpleNamespace

import pytest
from pydub import AudioSegment

from core import step11_merge_full_audio


def test_merge_audio_segments_fails_fast_when_audio_missing(monkeypatch, tmp_path):
    issues_path = tmp_path / "merge_issues.json"
    missing_audio = tmp_path / "missing.wav"
    monkeypatch.setattr(step11_merge_full_audio, "DUBBING_MERGE_ISSUES_JSON", str(issues_path))
    monkeypatch.setattr(
        step11_merge_full_audio,
        "get_quality_config",
        lambda: SimpleNamespace(allow_silence_fallback=False),
    )

    with pytest.raises(FileNotFoundError):
        step11_merge_full_audio.merge_audio_segments([str(missing_audio)], [[0.0, 1.0]], 16000)

    issues = json.loads(issues_path.read_text(encoding="utf-8"))
    assert issues[0]["audio_file"] == str(missing_audio)
    assert issues[0]["silence_fallback"] is False


def test_process_audio_segment_does_not_create_silence_on_decode_failure(monkeypatch, tmp_path):
    broken_audio = tmp_path / "broken.wav"
    broken_audio.write_bytes(b"not real wav data" * 100)

    def fail_ffmpeg(*args, **kwargs):
        raise step11_merge_full_audio.subprocess.CalledProcessError(returncode=1, cmd="ffmpeg", stderr=b"bad")

    monkeypatch.setattr(step11_merge_full_audio.subprocess, "run", fail_ffmpeg)
    monkeypatch.setattr(step11_merge_full_audio.AudioSegment, "from_file", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("decode failed")))

    with pytest.raises(RuntimeError, match="decode failed"):
        step11_merge_full_audio.process_audio_segment(str(broken_audio), allow_silence_fallback=False)


def test_merge_audio_segments_rejects_cardinality_mismatch(monkeypatch):
    monkeypatch.setattr(
        step11_merge_full_audio,
        "get_quality_config",
        lambda: SimpleNamespace(allow_silence_fallback=False),
    )
    with pytest.raises(ValueError, match="cardinality mismatch"):
        step11_merge_full_audio.merge_audio_segments(["one.wav"], [], 16000)


def test_merge_audio_segments_refuses_to_truncate_spoken_content(monkeypatch, tmp_path):
    issues_path = tmp_path / "merge_issues.json"
    monkeypatch.setattr(step11_merge_full_audio, "DUBBING_MERGE_ISSUES_JSON", str(issues_path))
    monkeypatch.setattr(
        step11_merge_full_audio,
        "get_quality_config",
        lambda: SimpleNamespace(allow_silence_fallback=False),
    )
    monkeypatch.setattr(
        step11_merge_full_audio,
        "process_audio_segment",
        lambda *_args, **_kwargs: AudioSegment.silent(duration=1500, frame_rate=16000),
    )

    with pytest.raises(RuntimeError, match="Refusing to truncate spoken content"):
        step11_merge_full_audio.merge_audio_segments(
            ["first.wav", "second.wav"],
            [[0.0, 1.5], [1.0, 2.0]],
            16000,
        )
    issues = json.loads(issues_path.read_text(encoding="utf-8"))
    assert "Refusing to truncate spoken content" in issues[0]["error"]


def test_merge_audio_segments_only_trims_encoding_rounding_tail(monkeypatch, tmp_path):
    issues_path = tmp_path / "merge_issues.json"
    monkeypatch.setattr(step11_merge_full_audio, "DUBBING_MERGE_ISSUES_JSON", str(issues_path))
    monkeypatch.setattr(
        step11_merge_full_audio,
        "get_quality_config",
        lambda: SimpleNamespace(allow_silence_fallback=False),
    )
    monkeypatch.setattr(
        step11_merge_full_audio,
        "process_audio_segment",
        lambda *_args, **_kwargs: AudioSegment.silent(duration=1004, frame_rate=16000),
    )

    merged = step11_merge_full_audio.merge_audio_segments(["line.wav"], [[0.0, 1.0]], 16000)

    assert len(merged) == 1000
    issues = json.loads(issues_path.read_text(encoding="utf-8"))
    assert issues[0]["error"] == "trimmed_encoding_rounding_tail"
    assert issues[0]["overflow_ms"] == 4
