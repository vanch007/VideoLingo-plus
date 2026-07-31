from pathlib import Path

import pytest

from core import step12_merge_dub_to_vid


def test_merge_video_requires_canonical_source_instead_of_derivative(monkeypatch, tmp_path):
    source = tmp_path / "source.mp4"
    (tmp_path / "AI字幕.mp4").write_bytes(b"already subtitled")
    monkeypatch.setattr(step12_merge_dub_to_vid, "SOURCE_VIDEO", str(source))

    with pytest.raises(FileNotFoundError, match="Canonical source video is missing"):
        step12_merge_dub_to_vid.merge_video_audio()


def test_source_video_constant_is_not_a_generated_derivative():
    assert Path(step12_merge_dub_to_vid.SOURCE_VIDEO).name == "source.mp4"
