from core.providers.moss_asr import recover_segments, transcript_from_segments


def test_moss_transcript_joins_nonempty_segments():
    assert transcript_from_segments(
        [{"text": "hello"}, {"text": ""}, {"text": "world"}]
    ) == "hello world"


def test_moss_tolerant_parser_recovers_extra_speaker_bracket():
    segments = recover_segments("[0.00][[S01] Xin chào, đây là thử nghiệm.[2.70]")

    assert len(segments) == 1
    assert segments[0]["speaker"] == "S01"
    assert segments[0]["start"] == 0.0
    assert segments[0]["end"] == 2.7
    assert segments[0]["text"] == "Xin chào, đây là thử nghiệm."
