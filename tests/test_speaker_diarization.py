import pytest

from core.providers.speaker_diarization import (
    SpeakerDiarizationError,
    apply_speaker_overrides_to_rows,
    align_speakers_to_words,
    apply_speaker_diarization,
    group_word_rows_by_speaker,
    same_known_speaker,
    speaker_coverage,
    validate_cached_speaker_rows,
)


def _result():
    return {
        "segments": [
            {
                "start": 0.0,
                "end": 2.0,
                "text": "hello there general kenobi",
                "words": [
                    {"word": "hello", "start": 0.0, "end": 0.4},
                    {"word": "there", "start": 0.45, "end": 0.9},
                    {"word": "general", "start": 1.05, "end": 1.4},
                    {"word": "kenobi", "start": 1.45, "end": 1.9},
                ],
            }
        ]
    }


def test_moss_turns_are_attached_to_native_words_without_changing_timestamps():
    result = _result()
    original_times = [(word["start"], word["end"]) for word in result["segments"][0]["words"]]

    report = align_speakers_to_words(
        result,
        [
            {"start": 0.0, "end": 0.95, "speaker": "S00"},
            {"start": 1.0, "end": 2.0, "speaker": "S01"},
        ],
    )

    words = result["segments"][0]["words"]
    assert [word["speaker"] for word in words] == ["S00", "S00", "S01", "S01"]
    assert [(word["start"], word["end"]) for word in words] == original_times
    assert report["coverage"] == 1.0
    assert report["speaker_transitions"] == 1
    assert result["segments"][0]["speaker_mixed"] is True


def test_invalid_moss_turn_is_rejected():
    with pytest.raises(SpeakerDiarizationError, match="valid speaker interval"):
        align_speakers_to_words(
            _result(),
            [{"start": 1.0, "end": 0.0, "speaker": "S00"}],
        )


def test_speaker_groups_never_merge_unknown_or_different_speakers():
    groups = group_word_rows_by_speaker(
        [
            {"text": "A", "speaker": "S00"},
            {"text": "B", "speaker": "S00"},
            {"text": "C", "speaker": "S01"},
            {"text": "D", "speaker": None},
            {"text": "E", "speaker": None},
        ]
    )

    assert groups == [
        {"speaker": "S00", "words": ["A", "B"]},
        {"speaker": "S01", "words": ["C"]},
        {"speaker": None, "words": ["D"]},
        {"speaker": None, "words": ["E"]},
    ]
    assert same_known_speaker("S00", "S00")
    assert not same_known_speaker(None, None)
    assert not same_known_speaker("S00", "S01")


def test_speaker_coverage_counts_only_known_labels():
    report = speaker_coverage(
        [
            {"speaker": "S00"},
            {"speaker": " unknown "},
            {"speaker": float("nan")},
            {"speaker": "S01"},
        ]
    )
    assert report["coverage"] == 0.5
    assert report["speakers"] == ["S00", "S01"]


def test_old_cache_without_speakers_fails_closed(monkeypatch):
    monkeypatch.setattr(
        "core.providers.speaker_diarization.load_key",
        lambda key, default=None: {
            "enabled": True,
            "required": True,
            "min_word_coverage": 0.98,
        },
    )
    with pytest.raises(SpeakerDiarizationError, match="archive/clean"):
        validate_cached_speaker_rows([{"text": "hello", "speaker": None}])


def test_isolated_unassigned_filler_is_retained_as_unknown_instead_of_dropping():
    result = {
        "segments": [
            {
                "start": 1.0,
                "end": 3.0,
                "text": "好嗯",
                "words": [
                    {"word": "好", "start": 1.0, "end": 1.2},
                    {"word": "嗯", "start": 2.5, "end": 3.0},
                ],
            }
        ]
    }
    report = align_speakers_to_words(
        result,
        [{"start": 0.9, "end": 1.3, "speaker": "S01"}],
        max_gap_seconds=0.1,
    )
    assert [word["word"] for word in result["segments"][0]["words"]] == ["好", "嗯"]
    assert result["segments"][0]["words"][1]["speaker"] is None
    assert report["word_count"] == 2
    assert report["assigned_words"] == 1


def test_verified_word_override_snaps_a_boundary_word_without_changing_timing():
    rows = [{"text": '"百"', "start": 99.72, "end": 100.02, "speaker": "S02"}]
    original_times = (rows[0]["start"], rows[0]["end"])

    applied = apply_speaker_overrides_to_rows(
        rows,
        [{"text": "百", "start": 99.72, "speaker": "S01", "tolerance_seconds": 0.03}],
    )

    assert applied == 1
    assert rows[0]["speaker"] == "S01"
    assert rows[0]["speaker_source"] == "verified-config-override"
    assert (rows[0]["start"], rows[0]["end"]) == original_times


def test_unsupported_diarization_backend_raises(monkeypatch):
    monkeypatch.setattr(
        "core.providers.speaker_diarization.load_key",
        lambda key, default=None: {"enabled": True, "backend": "unsupported-engine"},
    )
    with pytest.raises(SpeakerDiarizationError, match="supported: moss-mlx, nemotron-mlx"):
        apply_speaker_diarization(_result(), "dummy.wav")


def test_align_speakers_with_probabilities_tensor(tmp_path):
    import numpy as np

    probs_file = tmp_path / "probs.npz"
    # Create 200 frames (2.0s) of probabilities for 8 speakers
    # S01 (index 0) active for 0.0 - 0.9s (frames 0 to 90)
    # S02 (index 1) active for 1.0 - 2.0s (frames 100 to 200)
    probs = np.zeros((200, 8), dtype=np.float32)
    probs[0:90, 0] = 0.95
    probs[105:200, 1] = 0.92
    np.savez_compressed(probs_file, probs=probs, frame_stride=0.01)

    result = _result()
    report = align_speakers_to_words(
        result,
        [
            {"start": 0.0, "end": 0.9, "speaker": "S01"},
            {"start": 1.05, "end": 1.9, "speaker": "S02"},
        ],
        backend_name="nemotron-mlx",
        probabilities_path=str(probs_file),
    )
    words = result["segments"][0]["words"]
    assert [w["speaker"] for w in words] == ["S01", "S01", "S02", "S02"]
    assert words[0]["speaker_source"] == "nemotron-mlx"
    assert words[0]["speaker_score"] >= 0.9
    assert report["coverage"] == 1.0


def test_apply_speaker_diarization_routes_nemotron_and_shadow(monkeypatch, tmp_path):
    from dataclasses import dataclass

    @dataclass
    class DummyRun:
        segments: list
        output_dir: str
        probabilities_path: str = None

    monkeypatch.setattr(
        "core.providers.speaker_diarization.load_key",
        lambda key, default=None: {
            "enabled": True,
            "backend": "nemotron-mlx",
            "shadow_backend": "moss-mlx",
            "report_path": str(tmp_path / "speaker_diarization.json"),
            "source_coverage": {"enabled": False},
            "min_word_coverage": 0.5,
        },
    )
    monkeypatch.setattr(
        "core.providers.nemotron_diarization.run_nemotron_diarization",
        lambda audio_path, **kwargs: DummyRun(
            segments=[
                {"start": 0.0, "end": 0.9, "speaker": "S01"},
                {"start": 1.0, "end": 2.0, "speaker": "S02"},
            ],
            output_dir=str(tmp_path / "nemotron_out"),
        ),
    )
    monkeypatch.setattr(
        "core.providers.moss_asr.run_moss_asr",
        lambda audio_path, **kwargs: DummyRun(
            segments=[
                {"start": 0.0, "end": 0.9, "speaker": "S01", "text": "hello there"},
                {"start": 1.0, "end": 2.0, "speaker": "S02", "text": "general kenobi"},
            ],
            output_dir=str(tmp_path / "moss_out"),
        ),
    )

    result = _result()
    report = apply_speaker_diarization(result, "test.wav")
    assert report["status"] == "pass"
    assert report["backend"] == "nemotron-mlx"
    assert "shadow" in report
    assert report["shadow"]["status"] == "pass"
    assert report["shadow"]["backend"] == "moss-mlx"
    assert report["shadow"]["report_path"] == str(tmp_path / "speaker_diarization_shadow.json")
    assert (tmp_path / "speaker_diarization_shadow.json").is_file()
