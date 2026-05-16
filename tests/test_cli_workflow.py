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
    assert payload["planned_config"]["tts_method"] == "mlx_router"
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
