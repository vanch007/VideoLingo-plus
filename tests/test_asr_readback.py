import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from core.providers import asr_readback
from core.providers.asr_readback import extract_transcript, verify_tasks_df
from core.providers.contracts import ASRVerificationResult
from core.providers.quality import content_similarity
from core.providers.quality_gate import summarize_quality_gate


def test_extract_transcript_from_common_payloads():
    assert extract_transcript("plain transcript") == "plain transcript"
    assert extract_transcript('{"text": "hello"}') == "hello"
    assert extract_transcript('{"segments": [{"text": "xin"}, {"text": "chao"}]}') == "xin chao"


@pytest.mark.parametrize('missing', [None, '', '  ', float('nan'), pd.NA])
def test_missing_fresh_readback_clears_old_evidence_and_stays_rejected(monkeypatch, tmp_path, missing):
    from core.atomic_dubbing import _content_issues
    monkeypatch.setattr(asr_readback, 'SEGS_DIR', str(tmp_path))
    (tmp_path / '1_0.wav').write_bytes(b'fixture')
    monkeypatch.setattr(asr_readback, 'load_key', lambda key, default=None: default)
    monkeypatch.setattr(asr_readback, '_run_asr', lambda *_: ASRVerificationResult(status='ok', transcript=missing))
    tasks = pd.DataFrame([{'number':1, 'text':'Keep the source.', 'lines':['Keep the source.'],
        'asr_transcript':'Keep the source.', 'asr_status':'ok', 'asr_content_score':1.,
        'asr_leakage_score':0., 'asr_fingerprint':'old'}])
    actual, summary = verify_tasks_df(tasks, force=True)
    assert summary.failed == 1
    assert actual.loc[0, 'asr_status'] == 'fail'
    assert pd.isna(actual.loc[0, 'asr_transcript'])
    assert pd.isna(actual.loc[0, 'asr_content_score'])
    assert pd.isna(actual.loc[0, 'asr_fingerprint'])
    assert _content_issues(actual) == {0: ['empty_or_unavailable_readback']}
    actual.to_excel(tmp_path / 'roundtrip.xlsx', index=False)
    assert _content_issues(pd.read_excel(tmp_path / 'roundtrip.xlsx')) == {0: ['empty_or_unavailable_readback']}


def test_partial_readback_never_caches_a_whole_row_as_success(monkeypatch, tmp_path):
    monkeypatch.setattr(asr_readback, 'SEGS_DIR', str(tmp_path))
    for i in range(2):
        (tmp_path / f'1_{i}.wav').write_bytes(b'fixture')
    monkeypatch.setattr(asr_readback, 'load_key', lambda key, default=None: default)
    responses = iter(['Keep it.', ''])
    monkeypatch.setattr(asr_readback, '_run_asr', lambda *_: ASRVerificationResult(status='ok', transcript=next(responses)))
    actual, summary = verify_tasks_df(pd.DataFrame([{'number':1, 'text':'Keep it. Then go.',
        'lines':['Keep it.', 'Then go.']}]))
    assert summary.failed == 1
    assert actual.loc[0, 'asr_status'] == 'fail'
    assert pd.isna(actual.loc[0, 'asr_content_score'])
    assert len(json.loads(actual.loc[0, 'asr_line_results'])) == 2


@pytest.mark.parametrize('missing', [None, float('nan'), '', ' '])
def test_matching_fingerprint_with_missing_text_is_not_a_cache_hit(missing):
    row = pd.Series({'asr_status':'ok', 'asr_transcript':missing,
                     'asr_content_score':1., 'asr_fingerprint':'bound'})
    assert not asr_readback._is_cached_row(row, 'bound')


def test_cross_script_proper_name_uses_phonetic_similarity():
    assert content_similarity("Exactly. Jang Mah-zuh.", "Exactly. 江马斯族") > 0.8


def test_readback_equal_numbers_and_exact_homophones_are_not_missing_speech():
    assert content_similarity("You'll receive 1.8 million taels.", "You'll receive one point eight million taels.") == 1.
    assert content_similarity("Hey, Hu Wan!", "Hey, who won?") == 1.
    assert content_similarity("Fine.", "Fight.") < .88
    assert content_similarity("You'll receive 1.8 million.", "You'll receive 1.9 million.") < 1.


def test_mandarin_readback_accepts_only_equal_tone_bearing_sound_sequences():
    from core.dubbing_content_gate import probable_prefix_truncation, probable_tail_truncation, unexpected_boundary_speech
    expected, heard = '等它给出初版剪辑后。', '等他给出出版剪辑后。'
    assert content_similarity(expected, heard) == 1.
    assert not probable_prefix_truncation('它接着完成工作。', '他接着完成工作。')
    assert not probable_tail_truncation('我信任它。', '我信任他。')
    assert not unexpected_boundary_speech(expected, heard)
    assert content_similarity('买。', '卖。') < 1.  # different tone
    assert content_similarity('初版。', '新版。') < 1.  # different sound
    assert content_similarity('我能完成。', '我不能完成。') < 1.  # extra negation
    assert probable_tail_truncation('它能完成全部工作。', '它能完成全部。')
    assert unexpected_boundary_speech('它能完成工作。', '它能完成工作王。')


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


@pytest.mark.parametrize("wrong_order", [False, True])
def test_local_batch_readback_preserves_audio_identity_and_cache(monkeypatch, tmp_path, wrong_order):
    segs = tmp_path / "segs"
    segs.mkdir()
    for number in (1, 2):
        (segs / f"{number}_0.wav").write_bytes(b"audio")
    script = Path(asr_readback.__file__).resolve().parents[2] / "scripts/qwen3_asr_readback.py"
    settings = {"dubbing_quality.asr_readback_backend": "command",
                "dubbing_quality.asr_readback_command": ["python", str(script)],
                "dubbing_quality.asr_readback_batch": True}
    monkeypatch.setattr(asr_readback, "load_key", lambda key, default=None: settings.get(key, default))
    monkeypatch.setattr(asr_readback, "effective_target_language", lambda *_: "en")
    monkeypatch.setattr(asr_readback, "SEGS_DIR", str(segs))
    calls = []
    def run(cmd, **kwargs):
        calls.append(cmd)
        items = json.loads(Path(cmd[-1]).read_text())
        output = [{**item, "text": text} for item, text in zip(items, ["You pay.", "I'll go."])]
        if wrong_order:
            output.reverse()
        return SimpleNamespace(returncode=0, stdout=json.dumps(output), stderr="")
    monkeypatch.setattr(asr_readback.subprocess, "run", run)
    monkeypatch.setattr(asr_readback, "_run_asr", lambda *_: pytest.fail("Batch results must be reused"))
    frame = pd.DataFrame([{"number": n, "text": text, "lines": [text]}
                          for n, text in [(1, "You pay."), (2, "I'll go.")]])
    if wrong_order:
        with pytest.raises(ValueError, match="identity/cardinality"):
            verify_tasks_df(frame)
    else:
        output, report = verify_tasks_df(frame)
        assert report.checked == 2 and report.failed == 0
        assert output.asr_content_score.min() == 1.
        _, report = verify_tasks_df(output)
        assert report.cached == 2
        assert len(calls) == 1


def test_secondary_local_witness_retains_both_transcripts(monkeypatch, tmp_path):
    segs = tmp_path / 'segs'; segs.mkdir()
    (segs / '1_0.wav').write_bytes(b'audio')
    monkeypatch.setattr(asr_readback, 'SEGS_DIR', str(segs))
    monkeypatch.setattr(asr_readback, 'effective_target_language', lambda *_: 'en')
    monkeypatch.setattr(asr_readback, 'load_key', lambda k, d=None: 'local-witness' if k == 'dubbing_quality.asr_readback_secondary_model' else d)
    monkeypatch.setattr(asr_readback, '_run_asr', lambda *_: ASRVerificationResult(status='ok', transcript='How much?'))
    monkeypatch.setattr(asr_readback, '_run_builtin_asr', lambda *a, **k: ASRVerificationResult(status='ok', transcript='Hu Wan.'))
    frame, report = verify_tasks_df(pd.DataFrame([{'number':1,'text':'Hu Wan.','lines':['Hu Wan.']}]))
    assert frame.at[0,'asr_content_score'] == 1.
    evidence = json.loads(frame.at[0,'asr_line_results'])[0]
    assert evidence['primary_transcript'] == 'How much?'
    assert evidence['secondary_readback']['transcript'] == 'Hu Wan.'


def test_high_similarity_missing_name_tail_still_invokes_independent_witness(monkeypatch,tmp_path):
    segs=tmp_path/'segs';segs.mkdir();(segs/'1_0.wav').write_bytes(b'audio')
    monkeypatch.setattr(asr_readback,'SEGS_DIR',str(segs))
    monkeypatch.setattr(asr_readback,'effective_target_language',lambda *_:'en')
    monkeypatch.setattr(asr_readback,'load_key',lambda key,default=None:
        'local-witness' if key=='dubbing_quality.asr_readback_secondary_model' else default)
    expected='Tomorrow we will welcome our special guest Alex Morgan.'
    primary='Tomorrow we will welcome our special guest Alex.'
    assert asr_readback.content_similarity(expected,primary)>.88
    monkeypatch.setattr(asr_readback,'_run_asr',lambda *_:ASRVerificationResult(status='ok',transcript=primary))
    calls=[]
    monkeypatch.setattr(asr_readback,'_run_builtin_asr',lambda *a,**k:
        calls.append(k) or ASRVerificationResult(status='ok',transcript=expected))
    frame,_=verify_tasks_df(pd.DataFrame([{'number':1,'text':expected,'lines':[expected]}]))
    evidence=json.loads(frame.at[0,'asr_line_results'])[0]
    assert calls and frame.at[0,'asr_transcript']==expected
    assert evidence['secondary_readback']['trigger']==['possible_tail_loss']
    assert evidence['primary_transcript']==primary


def test_changed_readback_acceptance_threshold_cannot_reuse_older_selection(monkeypatch,tmp_path):
    segs=tmp_path/'segs';segs.mkdir();(segs/'1_0.wav').write_bytes(b'audio')
    monkeypatch.setattr(asr_readback,'SEGS_DIR',str(segs))
    monkeypatch.setattr(asr_readback,'effective_target_language',lambda *_:'en')
    state={'threshold':.88}
    monkeypatch.setattr(asr_readback,'load_key',lambda key,default=None:
        state['threshold'] if key=='dubbing_quality.content_score_min' else
        'local-witness' if key=='dubbing_quality.asr_readback_secondary_model' else default)
    expected='Hello there, our guest has arrived.'
    primary='Hi there, our guest has arrived.'
    calls=[]
    monkeypatch.setattr(asr_readback,'_run_asr',lambda *_:
        calls.append(True) or ASRVerificationResult(status='ok',transcript=primary))
    monkeypatch.setattr(asr_readback,'_run_builtin_asr',lambda *a,**k:
        ASRVerificationResult(status='ok',transcript=expected))
    first,_=verify_tasks_df(pd.DataFrame([{'number':1,'text':expected,'lines':[expected]}]))
    assert first.at[0,'asr_transcript']==primary
    state['threshold']=.99
    second,_=verify_tasks_df(first)
    assert len(calls)==2 and second.at[0,'asr_transcript']==expected
    assert first.at[0,'asr_fingerprint']!=second.at[0,'asr_fingerprint']


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


def test_numeric_readback_conflict_triggers_independent_witness_even_with_high_similarity(monkeypatch, tmp_path):
    segs = tmp_path/'segs'; segs.mkdir(); (segs/'1_0.wav').write_bytes(b'audio')
    monkeypatch.setattr(asr_readback, 'SEGS_DIR', str(segs))
    monkeypatch.setattr(asr_readback, 'effective_target_language', lambda *_: 'en')
    monkeypatch.setattr(asr_readback, 'load_key', lambda key, default=None:
        'local-witness' if key == 'dubbing_quality.asr_readback_secondary_model' else default)
    expected = "Four-dot masks! All our men. Bandits? Bandits? Bandits?"
    primary = expected.replace('Four', 'For')
    monkeypatch.setattr(asr_readback, '_run_asr', lambda *_: ASRVerificationResult(status='ok', transcript=primary))
    monkeypatch.setattr(asr_readback, 'content_similarity', lambda *_: .99)
    calls = []
    monkeypatch.setattr(asr_readback, '_run_builtin_asr', lambda *a, **kw:
        calls.append(kw) or ASRVerificationResult(status='ok', transcript=expected))
    frame, _ = verify_tasks_df(pd.DataFrame([{'number': 1, 'text': expected, 'lines': [expected]}]))
    assert calls and frame.at[0, 'asr_transcript'] == expected
    evidence = json.loads(frame.at[0, 'asr_line_results'])[0]
    assert 'spoken_numeric_value_mismatch' in evidence['secondary_readback']['trigger']
    assert evidence['primary_transcript'] == primary
