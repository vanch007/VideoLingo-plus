import numpy as np
import soundfile as sf

from core.source_pause_budget import quiet_source_spans


def test_quiet_evidence_uses_real_audio_and_invalidates_changed_source(tmp_path):
    sr = 16000
    audio = np.zeros((sr * 2, 2), dtype=np.float32)
    # A voice present on one channel must not become a quiet reserve.
    audio[:sr // 2, 0] = .1
    audio[sr:sr * 3 // 2] = .1
    path, report = tmp_path / "source.wav", tmp_path / "pauses.json"
    sf.write(path, audio, sr)
    before = quiet_source_spans(path, report)
    assert before["quiet_spans"] == [[.5,1.], [1.5,2.]]
    assert quiet_source_spans(path, report) == before
    audio[sr // 2:sr] = .1
    sf.write(path, audio, sr)
    after = quiet_source_spans(path, report)
    assert before["identity"]["audio_sha256"] != after["identity"]["audio_sha256"]
    assert after["quiet_spans"] == [[1.5,2.]]
