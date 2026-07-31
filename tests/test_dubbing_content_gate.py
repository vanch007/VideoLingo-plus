import pandas as pd
import pytest

from core import dubbing_content_gate
from core.providers.asr_readback import ReadbackSummary


def _readback(status="ok"):
    return ReadbackSummary(
        status=status,
        checked=1,
        skipped=0,
        failed=0,
        cached=0,
        min_content_score=0.2,
        max_leakage_score=0.0,
        output_file="tasks.xlsx",
    )


def test_probable_tail_truncation_distinguishes_homophone_from_missing_tail():
    assert dubbing_content_gate.probable_tail_truncation(
        "Stand tall or make money?", "Stand tall."
    )


@pytest.mark.parametrize(
    ("expected", "transcript"),
    [
        ("If not the poor, then who do you tax?", "If not the poor, then who do you?"),
        ("The peasants' share? Seven to three.", "The peasants share seven to"),
        (
            "To the common folk, you're the magistrate. To Huang Sillang, you're just begging "
            "on your knees. But making money? That's no shame.",
            "To the common folk, you're the magistrate. To Huang Xilong, you're just begging "
            "on your knees, but making money. That",
        ),
        ("It's humiliating. Fuck, it's humiliating.", "It's humiliating. Fuck."),
    ],
)
def test_probable_tail_truncation_catches_high_similarity_and_repeated_tails(expected, transcript):
    assert dubbing_content_gate.probable_tail_truncation(expected, transcript)


def test_high_score_tail_is_repaired_even_above_content_threshold():
    tasks = pd.DataFrame([{
        "number": 6,
        "text": "If not the poor, then who do you tax?",
        "asr_transcript": "If not the poor, then who do you?",
        "asr_content_score": 0.94,
    }])
    low, tails = dubbing_content_gate._failing_indices(tasks, 0.88)
    assert low == []
    assert tails == [0]


def test_probable_prefix_truncation_catches_missing_opening_clause():
    assert dubbing_content_gate.probable_prefix_truncation(
        "You can. On your knees.", "On your knees."
    )


def test_high_score_missing_prefix_is_repaired_even_above_severe_threshold():
    tasks = pd.DataFrame([{
        "number": 42,
        "text": "You can. On your knees.",
        "asr_transcript": "On your knees.",
        "asr_content_score": 0.7857,
    }])

    low, incomplete = dubbing_content_gate._failing_indices(tasks, 0.88)

    assert low == [0]
    assert incomplete == [0]
    assert dubbing_content_gate._repair_indices(tasks, low, incomplete) == [0]


def test_content_gate_does_not_regenerate_homophone_only_warning(monkeypatch, tmp_path):
    monkeypatch.setattr(dubbing_content_gate, "CONTENT_GATE_REPORT", tmp_path / "gate.json")
    monkeypatch.setattr(
        dubbing_content_gate,
        "load_key",
        lambda key, default=None: {
            "dubbing_quality.content_completion_score_min": 0.88,
            "dubbing_quality.content_completion_regenerate_score_min": 0.65,
            "dubbing_quality.content_completion_max_rounds": 2,
            "dubbing_quality.content_completion_fail_on_unverified": True,
            "dubbing_quality.content_completion_fail_on_incomplete": True,
            "dubbing_quality.content_completion_fail_on_unreadable": True,
        }.get(key, default),
    )

    def verify(df, force=False):
        out = df.copy()
        out["asr_transcript"] = "Too late. Chiefs text 90 years ahead."
        out["asr_content_score"] = 0.82
        return out, _readback()

    repairs = []
    tasks = pd.DataFrame([{
        "number": 1,
        "text": "Too late. Chiefs taxed ninety years ahead.",
    }])
    _, summary = dubbing_content_gate.enforce_content_completion(
        tasks,
        repair_rows=lambda df, indices, rewrite: (repairs.append(indices) or df, []),
        verify=verify,
    )

    assert repairs == []
    assert summary.status == "warn"
    assert summary.remaining_tail_truncation_rows == []
    assert not dubbing_content_gate.probable_tail_truncation(
        "Too late. Previous chiefs taxed ninety years ahead.",
        "Too late. Previous chiefs text 90 years ahead.",
    )


def test_single_word_nonempty_readback_is_warning_not_unreadable(monkeypatch, tmp_path):
    monkeypatch.setattr(dubbing_content_gate, "CONTENT_GATE_REPORT", tmp_path / "gate.json")
    monkeypatch.setattr(
        dubbing_content_gate,
        "load_key",
        lambda key, default=None: {
            "dubbing_quality.content_completion_score_min": 0.88,
            "dubbing_quality.content_completion_regenerate_score_min": 0.65,
            "dubbing_quality.content_completion_max_rounds": 2,
            "dubbing_quality.content_completion_fail_on_unverified": True,
            "dubbing_quality.content_completion_fail_on_incomplete": True,
            "dubbing_quality.content_completion_fail_on_unreadable": True,
        }.get(key, default),
    )

    def verify(df, force=False):
        out = df.copy()
        out["asr_transcript"] = "can."
        out["asr_content_score"] = 1 / 3
        return out, _readback()

    repairs = []
    _, summary = dubbing_content_gate.enforce_content_completion(
        pd.DataFrame([{"number": 38, "text": "Nah."}]),
        repair_rows=lambda df, indices, rewrite: (repairs.append(indices) or df, []),
        verify=verify,
    )
    assert repairs == []
    assert summary.status == "warn"
    assert summary.remaining_severely_unreadable_rows == []


def test_full_length_accented_readback_is_warning_not_missing_audio():
    row = pd.Series(
        {
            "text": "Past mayors till 2090.",
            "asr_transcript": "host mares till twenty ninety.",
        }
    )
    assert dubbing_content_gate._has_substantive_spoken_readback(row)


def test_content_gate_regenerates_uncapped_then_rewrites_remaining_tail(monkeypatch, tmp_path):
    monkeypatch.setattr(dubbing_content_gate, "CONTENT_GATE_REPORT", tmp_path / "gate.json")
    monkeypatch.setattr(
        dubbing_content_gate,
        "load_key",
        lambda key, default=None: {
            "dubbing_quality.content_completion_score_min": 0.88,
            "dubbing_quality.content_completion_max_rounds": 2,
            "dubbing_quality.content_completion_fail_on_unverified": True,
            "dubbing_quality.content_completion_fail_on_incomplete": True,
            "dubbing_quality.content_completion_fail_on_unreadable": True,
        }.get(key, default),
    )
    states = [
        ("Stand tall.", 0.5),
        ("Stand tall.", 0.5),
        ("Stand tall or earn?", 1.0),
    ]
    verify_calls = []

    def verify(df, force=False):
        transcript, score = states[len(verify_calls)]
        verify_calls.append(force)
        out = df.copy()
        out["asr_transcript"] = transcript
        out["asr_content_score"] = score
        return out, _readback()

    repairs = []

    def repair(df, indices, rewrite):
        repairs.append((indices, rewrite))
        out = df.copy()
        if rewrite:
            out.at[indices[0], "text"] = "Stand tall or earn?"
        return out, [1] if rewrite else []

    tasks = pd.DataFrame([{"number": 1, "text": "Stand tall or make money?"}])
    checkpoints = []
    result, summary = dubbing_content_gate.enforce_content_completion(
        tasks,
        repair_rows=repair,
        verify=verify,
        checkpoint=lambda df: checkpoints.append(df.at[0, "asr_content_score"]),
    )

    assert repairs == [([0], False), ([0], True)]
    assert verify_calls == [False, False, False]
    assert result.at[0, "asr_content_score"] == 1.0
    assert summary.status == "ok"
    assert summary.regenerated_rows == [1]
    assert summary.rewritten_rows == [1]
    assert checkpoints == [0.5, 0.5, 1.0]


def test_content_gate_blocks_pipeline_when_tail_remains(monkeypatch, tmp_path):
    monkeypatch.setattr(dubbing_content_gate, "CONTENT_GATE_REPORT", tmp_path / "gate.json")
    monkeypatch.setattr(
        dubbing_content_gate,
        "load_key",
        lambda key, default=None: {
            "dubbing_quality.content_completion_score_min": 0.88,
            "dubbing_quality.content_completion_max_rounds": 1,
            "dubbing_quality.content_completion_fail_on_unverified": True,
            "dubbing_quality.content_completion_fail_on_incomplete": True,
            "dubbing_quality.content_completion_fail_on_unreadable": True,
        }.get(key, default),
    )

    def verify(df, force=False):
        out = df.copy()
        out["asr_transcript"] = "Stand tall."
        out["asr_content_score"] = 0.5
        return out, _readback()

    tasks = pd.DataFrame([{"number": 7, "text": "Stand tall or make money?"}])
    with pytest.raises(RuntimeError, match=r"rows \[7\]"):
        dubbing_content_gate.enforce_content_completion(
            tasks,
            repair_rows=lambda df, _indices, _rewrite: (df, []),
            verify=verify,
        )


def test_content_gate_blocks_persistently_unreadable_audio(monkeypatch, tmp_path):
    monkeypatch.setattr(dubbing_content_gate, "CONTENT_GATE_REPORT", tmp_path / "gate.json")
    monkeypatch.setattr(
        dubbing_content_gate,
        "load_key",
        lambda key, default=None: {
            "dubbing_quality.content_completion_score_min": 0.88,
            "dubbing_quality.content_completion_regenerate_score_min": 0.65,
            "dubbing_quality.content_completion_max_rounds": 1,
            "dubbing_quality.content_completion_fail_on_unverified": True,
            "dubbing_quality.content_completion_fail_on_incomplete": True,
            "dubbing_quality.content_completion_fail_on_unreadable": True,
        }.get(key, default),
    )

    def verify(df, force=False):
        out = df.copy()
        out["asr_transcript"] = "unrelated noise"
        out["asr_content_score"] = 0.2
        return out, _readback()

    tasks = pd.DataFrame([{"number": 9, "text": "Were you chief?"}])
    with pytest.raises(RuntimeError, match=r"rows \[9\]"):
        dubbing_content_gate.enforce_content_completion(
            tasks,
            repair_rows=lambda df, _indices, _rewrite: (df, []),
            verify=verify,
        )


def test_empty_readback_is_severe_instead_of_treating_none_as_spoken_text():
    tasks = pd.DataFrame([{
        "number": 20,
        "text": "Yep",
        "asr_transcript": None,
        "asr_content_score": None,
    }])

    assert dubbing_content_gate._repair_indices(tasks, [0], []) == [0]


def test_content_gate_repairs_partial_asr_failures_instead_of_treating_backend_as_down(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(dubbing_content_gate, "CONTENT_GATE_REPORT", tmp_path / "gate.json")
    monkeypatch.setattr(
        dubbing_content_gate,
        "load_key",
        lambda key, default=None: {
            "dubbing_quality.content_completion_score_min": 0.88,
            "dubbing_quality.content_completion_regenerate_score_min": 0.65,
            "dubbing_quality.content_completion_max_rounds": 1,
            "dubbing_quality.content_completion_fail_on_unverified": True,
            "dubbing_quality.content_completion_fail_on_incomplete": True,
            "dubbing_quality.content_completion_fail_on_unreadable": True,
        }.get(key, default),
    )
    calls = []

    def verify(df, force=False):
        out = df.copy()
        if not calls:
            out["asr_transcript"] = ["Too late.", ""]
            out["asr_content_score"] = [1.0, None]
            summary = ReadbackSummary("fail", 2, 0, 1, 0, 1.0, 0.0, "tasks.xlsx")
        else:
            out["asr_transcript"] = ["Too late.", "Right."]
            out["asr_content_score"] = [1.0, 1.0]
            summary = ReadbackSummary("ok", 2, 0, 0, 0, 1.0, 0.0, "tasks.xlsx")
        calls.append(True)
        return out, summary

    repairs = []
    tasks = pd.DataFrame([
        {"number": 1, "text": "Too late."},
        {"number": 2, "text": "Right."},
    ])
    result, summary = dubbing_content_gate.enforce_content_completion(
        tasks,
        repair_rows=lambda df, indices, _rewrite: (repairs.append(indices) or df, []),
        verify=verify,
    )

    assert repairs == [[1]]
    assert summary.status == "ok"
    assert result.at[1, "asr_transcript"] == "Right."
