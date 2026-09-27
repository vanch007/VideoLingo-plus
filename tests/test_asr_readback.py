import pandas as pd

from core.providers import asr_readback
from core.providers.asr_readback import extract_transcript, verify_tasks_df
from core.providers.contracts import ASRVerificationResult
from core.providers.quality import content_similarity
from core.providers.quality_gate import summarize_quality_gate


def test_extract_transcript_from_common_payloads():
    assert extract_transcript("plain transcript") == "plain transcript"
    assert extract_transcript('{"text": "hello"}') == "hello"
    assert extract_transcript('{"segments": [{"text": "xin"}, {"text": "chao"}]}') == "xin chao"


def test_cross_script_proper_name_uses_phonetic_similarity():
    assert content_similarity("Exactly. Jang Mah-zuh.", "Exactly. 江马斯族") > 0.8


def test_verify_tasks_df_skips_when_command_not_configured():
    df = pd.DataFrame([{
        "number": 999999,
        "start_time": "00:00:01.000",
        "end_time": "00:00:02.000",
        "duration": 1.0,
        "text": "xin chao",
        "lines": ["xin chao"],
    }])
    out, summary = verify_tasks_df(df, limit=1, force=True)
    assert summary.status == "fail"
    assert summary.failed == 1
    assert out.loc[0, "asr_status"] == "fail"


def test_verify_tasks_df_skips_existing_audio_without_command(monkeypatch, tmp_path):
    segs = tmp_path / "segs"
    segs.mkdir()
    (segs / "1_0.wav").write_bytes(b"fake wav")
    monkeypatch.setattr(asr_readback, "SEGS_DIR", str(segs))
    monkeypatch.setattr(asr_readback, "load_key", lambda key, default=None: [] if key == "dubbing_quality.asr_readback_command" else default)

    df = pd.DataFrame([{"number": 1, "text": "xin chao", "lines": ["xin chao"]}])
    out, summary = verify_tasks_df(df, limit=1, force=True)

    assert summary.status == "skipped"
    assert summary.skipped == 1
    assert out.loc[0, "asr_status"] == "skipped"


def test_verify_tasks_df_scores_builtin_readback(monkeypatch, tmp_path):
    segs = tmp_path / "segs"
    segs.mkdir()
    (segs / "1_0.wav").write_bytes(b"fake wav")
    monkeypatch.setattr(asr_readback, "SEGS_DIR", str(segs))
    monkeypatch.setattr(asr_readback, "load_key", lambda key, default=None: "vi" if key == "target_language" else default)
    monkeypatch.setattr(asr_readback, "effective_target_language", lambda default="auto": "vi")
    monkeypatch.setattr(
        asr_readback,
        "_run_asr",
        lambda audio_path, language: ASRVerificationResult(status="ok", transcript="xin chao"),
    )

    df = pd.DataFrame([{"number": 1, "text": "xin chao", "lines": ["xin chao"], "origin": "你好"}])
    out, summary = verify_tasks_df(df, limit=1, force=True)

    assert summary.status == "ok"
    assert summary.checked == 1
    assert out.loc[0, "asr_status"] == "ok"
    assert out.loc[0, "asr_content_score"] >= 0.99
    assert out.loc[0, "asr_leakage_score"] == 0.0
    assert isinstance(out.loc[0, "asr_fingerprint"], str)
    assert out.loc[0, "asr_language"] == "vi"


def test_verify_tasks_df_reuses_matching_fingerprint_cache(monkeypatch, tmp_path):
    segs = tmp_path / "segs"
    segs.mkdir()
    (segs / "1_0.wav").write_bytes(b"fake wav")
    monkeypatch.setattr(asr_readback, "SEGS_DIR", str(segs))
    monkeypatch.setattr(asr_readback, "load_key", lambda key, default=None: "vi" if key == "target_language" else default)
    monkeypatch.setattr(
        asr_readback,
        "_run_asr",
        lambda audio_path, language: ASRVerificationResult(status="ok", transcript="xin chao"),
    )

    df = pd.DataFrame([{"number": 1, "text": "xin chao", "lines": ["xin chao"], "origin": "你好"}])
    out, first = verify_tasks_df(df, limit=1, force=True)
    assert first.checked == 1

    def fail_if_called(audio_path, language):
        raise AssertionError("matching ASR fingerprint should reuse cached scores")

    monkeypatch.setattr(asr_readback, "_run_asr", fail_if_called)
    cached_out, summary = verify_tasks_df(out, limit=1, force=False)

    assert summary.status == "ok"
    assert summary.checked == 0
    assert summary.cached == 1
    assert summary.skipped == 1
    assert cached_out.loc[0, "asr_content_score"] == out.loc[0, "asr_content_score"]


def test_verify_tasks_df_reruns_when_fingerprint_changes(monkeypatch, tmp_path):
    segs = tmp_path / "segs"
    segs.mkdir()
    (segs / "1_0.wav").write_bytes(b"fake wav")
    monkeypatch.setattr(asr_readback, "SEGS_DIR", str(segs))
    monkeypatch.setattr(asr_readback, "load_key", lambda key, default=None: "vi" if key == "target_language" else default)
    monkeypatch.setattr(
        asr_readback,
        "_run_asr",
        lambda audio_path, language: ASRVerificationResult(status="ok", transcript="xin chao"),
    )

    df = pd.DataFrame([{"number": 1, "text": "xin chao", "lines": ["xin chao"], "origin": "你好"}])
    out, _ = verify_tasks_df(df, limit=1, force=True)
    out.at[0, "text"] = "xin chao moi"
    out.at[0, "lines"] = ["xin chao moi"]
    calls = {"count": 0}

    def rerun(audio_path, language):
        calls["count"] += 1
        return ASRVerificationResult(status="ok", transcript="xin chao moi")

    monkeypatch.setattr(asr_readback, "_run_asr", rerun)
    refreshed, summary = verify_tasks_df(out, limit=1, force=False)

    assert calls["count"] == 1
    assert summary.checked == 1
    assert summary.cached == 0
    assert refreshed.loc[0, "asr_transcript"] == "xin chao moi"


def test_run_asr_uses_builtin_backend(monkeypatch):
    monkeypatch.setattr(
        asr_readback,
        "load_key",
        lambda key, default=None: "builtin" if key == "dubbing_quality.asr_readback_backend" else default,
    )
    monkeypatch.setattr(
        asr_readback,
        "_run_builtin_asr",
        lambda audio_path, language: ASRVerificationResult(status="ok", transcript="noi dung"),
    )

    result = asr_readback._run_asr("sample.wav", "vi")
    assert result.status == "ok"
    assert result.transcript == "noi dung"


def test_run_asr_uses_moss_backend(monkeypatch):
    monkeypatch.setattr(
        asr_readback,
        "load_key",
        lambda key, default=None: "moss-mlx" if key == "dubbing_quality.asr_readback_backend" else default,
    )
    monkeypatch.setattr(
        asr_readback,
        "_run_moss_asr",
        lambda audio_path, language: ASRVerificationResult(status="ok", transcript="noi dung moss"),
    )

    result = asr_readback._run_asr("sample.wav", "vi")
    assert result.status == "ok"
    assert result.transcript == "noi dung moss"


def test_quality_gate_summary_counts_reasons():
    eval_df = pd.DataFrame([
        {"status": "ok", "reason": ""},
        {"status": "warn", "reason": "low_content_score"},
        {"status": "fail", "reason": "missing_audio,reference_leak"},
    ])
    gate = summarize_quality_gate(eval_df)
    assert not gate.passed
    assert gate.fail == 1
    assert gate.reasons["missing_audio"] == 1
