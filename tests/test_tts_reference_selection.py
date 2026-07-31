import pandas as pd
from pydub import AudioSegment

from core.all_tts_functions.tts_utils import get_reference_audio_path


def test_speaker_anchor_prefers_stable_duration_over_short_row_reference(tmp_path):
    refers = tmp_path / "refers"
    refers.mkdir()
    for number, duration in ((1, 800), (2, 4900), (3, 7000)):
        AudioSegment.silent(duration=duration, frame_rate=16000).export(
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
