from types import SimpleNamespace

from core import doctor


def _tts_config(monkeypatch, **overrides):
    values = {"tts_method": "mlx_indextts2", "mlx_tts.default_backend": "auto", "target_language": "en"}
    values.update(overrides)
    monkeypatch.setattr(doctor, "load_key", lambda key, default=None: values.get(key, default))
    import core.providers.mlx_tts as tts
    monkeypatch.setattr(tts, "load_key", lambda key, default=None: values.get(key, default))
    monkeypatch.setattr(tts, "list_backend_status", lambda: [
        {"name": "indextts2", "ok": True, "detail": "runtime ready"},
        {"name": "omnivoice", "ok": True, "detail": "runtime ready"},
        {"name": "higgs", "ok": False, "detail": "missing executable"},
    ])
    return tts


def test_inactive_missing_backend_is_visible_warning(monkeypatch, capsys):
    _tts_config(monkeypatch)
    checks = doctor._check_mlx_tts()
    monkeypatch.setattr(doctor, "run_checks", lambda: checks)
    assert doctor.print_report() == 0
    output = capsys.readouterr().out
    assert "[OK] mlx-tts:indextts2" in output
    assert "[WARN] mlx-tts:higgs" in output


def test_selected_missing_backend_is_fatal(monkeypatch, capsys):
    _tts_config(monkeypatch, tts_method="mlx_higgs_audio")
    checks = doctor._check_mlx_tts()
    monkeypatch.setattr(doctor, "run_checks", lambda: checks)
    assert doctor.print_report() == 1
    assert "[FAIL] mlx-tts:higgs" in capsys.readouterr().out


def test_forced_router_backend_remains_required(monkeypatch):
    _tts_config(monkeypatch, tts_method="mlx_router", **{"mlx_tts.default_backend": "higgs"})
    checks = {c.name: c for c in doctor._check_mlx_tts()}
    assert checks["mlx-tts:higgs"].required
    assert not checks["mlx-tts:higgs"].ok


def test_automatic_chinese_and_emotion_routes_are_checked(monkeypatch):
    _tts_config(monkeypatch, tts_method="mlx_router", target_language="zh")
    required = {c.name for c in doctor._check_mlx_tts() if c.required}
    # Chinese and emotion-reference rows both route to IndexTTS 2.x, so it is the only required backend.
    assert required == {"mlx-tts:indextts2"}


def test_selected_disabled_backend_fails(monkeypatch):
    tts = _tts_config(monkeypatch)
    monkeypatch.setattr(tts, "list_backend_status", lambda: [
        {"name": "indextts2", "ok": True, "detail": "disabled by config"},
    ])
    check = doctor._check_mlx_tts()[0]
    assert check.required and not check.ok


def test_missing_selected_health_result_cannot_pass(monkeypatch):
    tts = _tts_config(monkeypatch)
    monkeypatch.setattr(tts, "list_backend_status", lambda: [])
    check = doctor._check_mlx_tts()[0]
    assert check.name == "mlx-tts:indextts2" and not check.ok


def test_dependency_report_keeps_all_conflicts(monkeypatch):
    monkeypatch.setattr(doctor.subprocess, "run", lambda *a, **k: SimpleNamespace(
        returncode=1, stdout="demucs requires torchaudio\nopencv requires numpy>=2\n", stderr=""))
    ok, detail = doctor._run(["python", "-m", "pip", "check"], all_output=True)
    assert not ok and "demucs" in detail and "opencv" in detail


def test_dependency_conflict_is_never_exempted(monkeypatch):
    monkeypatch.setattr(doctor, "_run", lambda *a, **k: (False, "demucs requires torchaudio\nopencv requires numpy>=2"))
    monkeypatch.setattr(doctor, "load_key", lambda key, default=None: default)
    monkeypatch.setattr(doctor, "_check_import", lambda name: doctor.CheckResult(name, True, "stub"))
    monkeypatch.setattr(doctor, "_check_moss_asr", lambda: doctor.CheckResult("moss", True, "stub"))
    monkeypatch.setattr(doctor, "is_step_completed", lambda name: False)
    pip_check = next(c for c in doctor.run_checks(include_services=False) if c.name == "pip-check")
    assert not pip_check.ok and pip_check.required
