import json

import pytest

from core.providers.source_coverage import (
    SourceCoverageError,
    analyze_source_coverage,
    reconcile_source_coverage,
)


def _native(words):
    return {
        "segments": [
            {
                "start": words[0][1],
                "end": words[-1][2],
                "text": "".join(word[0] for word in words),
                "words": [
                    {"word": text, "start": start, "end": end}
                    for text, start, end in words
                ],
            }
        ]
    }


def test_coverage_detects_temporal_gap_even_when_existing_words_are_valid():
    result = _native([("寒碜", 110.2, 110.7), ("很", 121.6, 124.6)])
    report = analyze_source_coverage(
        result,
        [
            {"start": 112.0, "end": 113.1, "speaker": "S02", "text": "很他妈寒碜"},
            {"start": 115.6, "end": 116.8, "speaker": "S02", "text": "我是想站着"},
        ],
    )
    assert report["status"] == "fail"
    assert report["missing_turns"] == 2


def test_coverage_detects_unrelated_long_word_masking_a_turn():
    result = _native([("很", 121.6, 124.6)])
    report = analyze_source_coverage(
        result,
        [{"start": 123.5, "end": 124.3, "speaker": "S02", "text": "能不能挣钱"}],
    )
    assert report["turns"][0]["overlap_ratio"] == 1.0
    assert report["turns"][0]["status"] == "missing"


def test_equal_density_text_disagreement_is_not_misclassified_as_missing_speech():
    result = _native([("必然", 140.7, 141.2)])
    report = analyze_source_coverage(
        result,
        [{"start": 140.7, "end": 141.2, "speaker": "S02", "text": "鄙人"}],
    )
    assert report["turns"][0]["text_similarity"] == 0.0
    assert report["status"] == "pass"


def test_recovery_uses_native_word_timestamps_and_writes_audit_report(tmp_path):
    result = _native([("前", 10.0, 10.2), ("后", 14.0, 14.2)])
    witness = [{"start": 11.0, "end": 13.0, "speaker": "S01", "text": "我是想站着"}]
    calls = []

    def recover(start, end):
        calls.append((start, end))
        return _native(
            [("我是", 11.1, 11.5), ("想", 11.5, 11.8), ("站着", 11.8, 12.4)]
        )

    path = tmp_path / "coverage.json"
    report = reconcile_source_coverage(
        result,
        witness,
        recover,
        backend="stable-ts",
        report_path=path,
    )

    assert calls == [(10.5, 13.5)]
    assert report["status"] == "pass"
    assert [word["word"] for segment in result["segments"] for word in segment["words"]] == [
        "前",
        "我是",
        "想",
        "站着",
        "后",
    ]
    assert json.loads(path.read_text(encoding="utf-8"))["timestamp_contract"] == (
        "native-word-timestamps-only"
    )


def test_recovery_fails_closed_when_native_asr_still_returns_no_words(tmp_path):
    result = _native([("前", 10.0, 10.2), ("后", 14.0, 14.2)])
    witness = [{"start": 11.0, "end": 13.0, "speaker": "S01", "text": "漏掉台词"}]

    with pytest.raises(SourceCoverageError, match="still misses"):
        reconcile_source_coverage(
            result,
            witness,
            lambda _start, _end: {"segments": []},
            backend="stable-ts",
            report_path=tmp_path / "coverage.json",
        )
