from pathlib import Path

import pytest
from pydub.generators import Sine

from core import step12_merge_dub_to_vid


def test_merge_checks_locked_tasks_and_preflight_before_rendering(monkeypatch, tmp_path):
    import pandas as pd
    from core import atomic_dubbing, utterance_plan

    tasks = pd.DataFrame([{'unit_id': 'L1', 'text': 'Go.'}])
    plan = {'units': [{'unit_id': 'L1', 'source': '走。', 'target': 'Go.'}]}
    validated = []
    identities = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(utterance_plan, 'plan_active', lambda: True)
    monkeypatch.setattr(utterance_plan, 'load_plan', lambda: plan)
    monkeypatch.setattr(utterance_plan, 'validate_tasks',
                        lambda frame, actual, **options: validated.append((frame, actual, options)))
    monkeypatch.setattr(atomic_dubbing, 'verify_preflight_identity', lambda frame: identities.append(frame))
    monkeypatch.setattr(step12_merge_dub_to_vid.pd, 'read_excel', lambda _: tasks)
    monkeypatch.setattr(step12_merge_dub_to_vid, 'SOURCE_VIDEO', str(tmp_path / 'source.mp4'))

    with pytest.raises(FileNotFoundError, match='Canonical source video is missing'):
        step12_merge_dub_to_vid.merge_video_audio()

    assert validated[0][0] is tasks and validated[0][1] is plan
    assert validated[0][2] == {'generated': True}
    assert identities[0].to_dict('records') == [{'LineID': 'L1', 'Source': '走。', 'Translation': 'Go.'}]


def test_merge_video_requires_canonical_source_instead_of_derivative(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "source.mp4"
    (tmp_path / "AI字幕.mp4").write_bytes(b"already subtitled")
    monkeypatch.setattr(step12_merge_dub_to_vid, "SOURCE_VIDEO", str(source))

    with pytest.raises(FileNotFoundError, match="Canonical source video is missing"):
        step12_merge_dub_to_vid.merge_video_audio()


def test_source_video_constant_is_not_a_generated_derivative():
    assert Path(step12_merge_dub_to_vid.SOURCE_VIDEO).name == "source.mp4"


def test_dub_windows_are_merged_and_clamped_to_source_duration():
    assert step12_merge_dub_to_vid._normalise_dub_windows(
        [[0.2, 0.6], [0.5, 1.2], [-1, 0.1], [1.5, 1.5]], 1000
    ) == [(0, 100), (200, 1000)]


def test_audio_context_preserves_original_only_outside_dub_windows():
    source = Sine(440).to_audio_segment(duration=1000).apply_gain(-8).set_channels(2)
    background = Sine(880).to_audio_segment(duration=1000).apply_gain(-18).set_channels(2)

    original_context, background_context = step12_merge_dub_to_vid._build_audio_context_tracks(
        source, background, [(200, 600)]
    )

    assert original_context[:200].rms > 0
    assert original_context[200:600].rms == 0
    assert original_context[600:].rms > 0
    assert background_context[:200].rms == 0
    assert background_context[200:600].rms > 0
    assert background_context[600:].rms == 0


def test_validate_merged_video_rejects_missing_or_empty_file(tmp_path):
    missing = tmp_path / "missing.mp4"
    with pytest.raises(RuntimeError, match="does not exist or is empty"):
        step12_merge_dub_to_vid.validate_merged_video(str(missing))

    empty = tmp_path / "empty.mp4"
    empty.write_bytes(b"")
    with pytest.raises(RuntimeError, match="does not exist or is empty"):
        step12_merge_dub_to_vid.validate_merged_video(str(empty))



def test_validate_merged_video_checks_source_duration(monkeypatch, tmp_path):
    test_vid = tmp_path / "vid.mp4"
    test_vid.write_bytes(b"fake")
    src_vid = tmp_path / "src.mp4"
    src_vid.write_bytes(b"src")
    import json
    fake_json = json.dumps({"streams": [{"codec_type": "video"}, {"codec_type": "audio"}], "format": {"duration": "10.0"}})
    monkeypatch.setattr(
        step12_merge_dub_to_vid.subprocess,
        "run",
        lambda cmd, **kw: type("Res", (), {
            "returncode": 0,
            "stdout": fake_json,
            "stderr": ""
        })()
    )
    info = step12_merge_dub_to_vid.validate_merged_video(str(test_vid), str(src_vid))
    assert info["format"]["duration"] == "10.0"
