import argparse
import json
from pathlib import Path

from core import cli


def _run_args(**overrides):
    values = {
        "input": "sample.mp4",
        "source": "zh",
        "target": "vi",
        "profile": "cinematic",
        "llm": "config",
        "tts": "auto",
        "smoke_seconds": None,
        "run_id": "test-run",
        "subtitle_only": False,
        "no_subtitles": False,
        "no_resume": False,
        "auto_archive_stale_translation": False,
        "dry_run": True,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_run_dry_run_does_not_mutate_config_or_state(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    calls = []
    monkeypatch.setattr(cli, "update_key", lambda key, value: calls.append((key, value)))
    monkeypatch.setattr(cli, "apply_profile", lambda name: calls.append(("profile", name)))

    import core.translation_state as translation_state

    monkeypatch.setattr(
        translation_state,
        "guard_translation_artifacts_for_steps",
        lambda steps, auto_archive=False: {"ok": True, "action": "not_applicable", "steps": steps},
    )

    assert cli._cmd_run(_run_args()) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["run"]["status"] == "dry_run"
    assert payload["planned_config"]["tts_method"] == "mlx_indextts2"
    assert calls == []
    assert not Path("output/pipeline_state.json").exists()


def test_resume_reuses_stored_input_when_not_passed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("output").mkdir()
    Path("output/pipeline_state.json").write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "input": "stored.mp4",
                "source": "zh",
                "target": "vi",
                "profile": "cinematic",
                "llm": "config",
                "tts": "auto",
                "subtitle_only": True,
                "no_subtitles": False,
            }
        ),
        encoding="utf-8",
    )
    seen = {}

    def fake_run(args):
        seen["input"] = args.input
        return 0

    monkeypatch.setattr(cli, "_cmd_run", fake_run)

    args = _run_args(input=None, run_id="run-1", source=None, target=None, subtitle_only=False)

    assert cli._cmd_resume(args) == 0
    assert seen["input"] == "stored.mp4"
    assert args.subtitle_only is True


def test_status_reports_artifacts_without_existing_state(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)

    assert cli._cmd_status(argparse.Namespace(include_dubbing_eval=False)) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["pipeline_state"] is None
    assert payload["pending_steps"]


def test_status_include_dubbing_eval_does_not_write_eval(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "evaluate_dubbing", lambda: (None, {"ok": 1}))

    def fail_if_written():
        raise AssertionError("status must not rewrite dubbing eval artifacts")

    monkeypatch.setattr(cli, "write_dubbing_eval", fail_if_written)

    assert cli._cmd_status(argparse.Namespace(include_dubbing_eval=True)) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["dubbing_eval"] == {"ok": 1}


def test_apply_run_config_keeps_explicit_tts_after_profile(monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "apply_profile", lambda name: calls.append(("profile", name)))
    monkeypatch.setattr(cli, "update_key", lambda key, value: calls.append((key, value)))

    cli._apply_run_config(_run_args(tts="edge_tts", dry_run=False))

    assert calls[0] == ("profile", "cinematic")
    assert calls[-1] == ("tts_method", "edge_tts")


def test_run_passes_smoke_seconds_to_step_builder(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    seen = {}

    def fake_build_steps(input_value, *, dubbing=True, subtitles=True, smoke_seconds=None):
        seen["input"] = input_value
        seen["smoke_seconds"] = smoke_seconds
        return []

    import core.translation_state as translation_state

    monkeypatch.setattr(cli, "build_steps_for_input", fake_build_steps)
    monkeypatch.setattr(cli, "_apply_run_config", lambda args: None)
    monkeypatch.setattr(
        translation_state,
        "guard_translation_artifacts_for_steps",
        lambda steps, auto_archive=False: {"ok": True, "action": "not_applicable", "steps": steps},
    )

    assert cli._cmd_run(_run_args(dry_run=True, smoke_seconds=60)) == 0

    payload = json.loads(capsys.readouterr().out)
    assert seen == {"input": "sample.mp4", "smoke_seconds": 60}
    assert payload["planned_config"]["smoke_seconds"] == 60


def test_local_video_step_uses_smoke_seconds(monkeypatch):
    from core.pipeline import runner

    seen = {}
    monkeypatch.setattr(
        runner,
        "prepare_local_video",
        lambda input_path, *, smoke_seconds=None: seen.update(input=input_path, smoke_seconds=smoke_seconds) or "output/source.mp4",
    )

    steps = runner.build_steps_for_input("sample.mp4", smoke_seconds=60)
    steps[0].action()

    assert seen == {"input": "sample.mp4", "smoke_seconds": 60}
