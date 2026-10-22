from copy import deepcopy

from core.all_whisper_methods.stable_ts_local import adjust_nonempty_gaps
from core.step2_whisperX import transcribe_recovery_audio


def test_silent_stable_ts_probe_remains_empty_without_gap_adjustment_crash():
    from stable_whisper import WhisperResult
    silent = WhisperResult({"segments": [], "language": "zh"})
    assert adjust_nonempty_gaps(silent) is silent
    assert silent.segments == []


def test_recovery_prioritizes_original_audio_and_preserves_native_timing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    audio = tmp_path / "output/audio"
    audio.mkdir(parents=True)
    for name in ["raw_master.wav", "enhanced.wav", "vocal.wav"]:
        (audio / name).write_bytes(name.encode())
    native = {"segments": [{"start": 5.1, "end": 5.5, "words": [{"word": "等等", "start": 5.1, "end": 5.5}]}]}
    calls = []
    def recognize(path, start, end):
        calls.append(path)
        return deepcopy(native)
    result = transcribe_recovery_audio(recognize, "output/audio/enhanced.wav", 5., 6.)
    assert calls == ["output/audio/raw_master.wav"]
    assert result["segments"] == native["segments"]
    assert result["recovery_audio"]["sha256"].startswith("sha256:")


def test_recovery_can_try_another_audio_view_but_never_invent_words(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    audio = tmp_path / "output/audio"
    audio.mkdir(parents=True)
    for name in ["raw_master.wav", "vocal.wav"]:
        (audio / name).write_bytes(name.encode())
    calls = []
    def recognize(path, start, end):
        calls.append(path)
        return {"segments": []}
    result = transcribe_recovery_audio(recognize, "output/audio/raw_master.wav", 5., 6.)
    assert calls == ["output/audio/raw_master.wav", "output/audio/vocal.wav"]
    assert result["segments"] == []
