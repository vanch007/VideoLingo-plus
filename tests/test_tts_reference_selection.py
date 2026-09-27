import pandas as pd
import pytest
from pydub import AudioSegment
from pydub.generators import Sine

from core.all_tts_functions.tts_utils import get_reference_audio_path


def test_speaker_anchor_prefers_stable_duration_over_short_row_reference(tmp_path):
    refers = tmp_path / "refers"
    refers.mkdir()
    for number, duration in ((1, 800), (2, 4900), (3, 7000)):
        Sine(240).to_audio_segment(duration=duration).export(
            refers / f"{number}.wav", format="wav"
        )
    tasks = pd.DataFrame([
        {"number": 1, "speaker": "S01"},
        {"number": 2, "speaker": "S01"},
        {"number": 3, "speaker": "S01"},
    ])

    path, fallback_number = get_reference_audio_path(
        1,
        refers_dir=str(refers),
        task_df=tasks,
        speaker="S01",
        prefer_speaker_anchor=True,
        speaker_anchor_target_ms=5000,
    )

    assert path.endswith("2.wav")
    assert fallback_number == 2


def test_anchor_scoring_does_not_penalize_high_pitch(tmp_path):
    from core.all_tts_functions.tts_utils import _score_reference_candidate

    for pitch in (120, 300):
        Sine(pitch).to_audio_segment(duration=3000).apply_gain(-12).export(
            tmp_path / f"{pitch}.wav", format="wav"
        )
    assert abs(_score_reference_candidate(str(tmp_path / "120.wav"), 3000)
               - _score_reference_candidate(str(tmp_path / "300.wav"), 3000)) < 30


def test_strict_speaker_reference_rejects_cross_speaker_borrowing(tmp_path):
    refers = tmp_path / "refers"
    refers.mkdir()
    # Only S01 has a valid audio file (1.wav)
    AudioSegment.silent(duration=2000, frame_rate=16000).export(
        refers / "1.wav", format="wav"
    )
    tasks = pd.DataFrame([
        {"number": 1, "speaker": "S01"},
        {"number": 2, "speaker": "S02"},
    ])

    # For S02 (row 2), it should NOT borrow S01's 1.wav when strict_speaker_match=True
    with pytest.raises(FileNotFoundError, match="未找到属于说话人 S02 的有效参考音频"):
        get_reference_audio_path(
            2,
            refers_dir=str(refers),
            task_df=tasks,
            speaker="S02",
            strict_speaker_match=True,
        )
