from pathlib import Path

import pandas as pd

from core.providers.dubbing_repair import (
    OVER_DURATION_MANUAL,
    OVER_DURATION_REWRITE,
    OVER_DURATION_SPEED_FIT,
    UNDER_DURATION_SLOW_FIT,
    _under_duration_target,
    apply_repair_plan,
    append_repair_history,
    build_repair_plan,
    classify_over_duration,
    classify_under_duration,
    choose_repair_action,
    summarize_over_duration,
    write_repair_plan,
)


def _task_row(**overrides):
    row = {
        "number": 1,
        "start_time": "00:00:01.000",
        "end_time": "00:00:03.000",
        "duration": 2.0,
        "tolerance": 0.0,
        "text": "xin chao",
        "lines": ["xin chao"],
        "ref_text": "source words",
    }
    row.update(overrides)
    return row


def test_over_duration_action_rewrites_and_prefers_duration_backend():
    eval_row = {"number": 1, "status": "warn", "reason": "over_duration", "text": "xin chao"}
    action = choose_repair_action(eval_row, _task_row())
    assert action.action == "rewrite_and_regenerate"
    assert action.rewrite
    assert action.backend == "indextts2"


def test_under_duration_action_expands_and_prefers_duration_backend():
    eval_row = {"number": 1, "status": "warn", "reason": "under_duration", "text": "xin chao"}
    action = choose_repair_action(eval_row, _task_row())
    assert action.action == "expand_and_regenerate"
    assert action.rewrite
    assert action.backend == "indextts2"


def test_under_duration_action_slow_fits_when_existing_audio_can_stretch():
    eval_row = {
        "number": 1,
        "status": "warn",
        "reason": "under_duration",
        "text": "xin chao",
        "available_duration": 4.0,
        "final_audio_dur": 3.6,
        "speed_factor": 0.95,
    }
    action = choose_repair_action(eval_row, _task_row(speed_factor=0.95))
    assert classify_under_duration(eval_row, _task_row(speed_factor=0.95)) == UNDER_DURATION_SLOW_FIT
    assert action.action == "slow_fit_existing_audio"
    assert not action.rewrite
    assert action.backend is None


def test_under_duration_target_keeps_early_end_margin(monkeypatch):
    def fake_quality_load_key(key, default=None):
        values = {
            "dubbing_quality.mode": "high_sync",
            "dubbing_quality.min_duration_ratio": 0.5,
            "dubbing_quality.max_early_end_drift": 0.46,
        }
        return values.get(key, default)

    def fake_repair_load_key(key, default=None):
        values = {
            "dubbing_quality.under_duration_fit_margin_seconds": 0.06,
            "dubbing_quality.slow_fit_target_ratio": 0.5,
        }
        return values.get(key, default)

    monkeypatch.setattr("core.dubbing_quality.load_key", fake_quality_load_key)
    monkeypatch.setattr("core.providers.dubbing_repair_rules.load_key", fake_repair_load_key)

    assert _under_duration_target({"available_duration": 4.0}) == 3.6


def test_mixed_duration_action_prefers_shortening_over_expansion():
    eval_row = {
        "number": 1,
        "status": "warn",
        "reason": "over_duration,under_duration",
        "text": "xin chao",
        "duration_ratio": 1.2,
    }
    action = choose_repair_action(eval_row, _task_row())
    assert action.action == "rewrite_and_regenerate"
    assert action.rewrite
    assert action.backend == "indextts2"


def test_over_duration_speed_fit_uses_existing_audio_action():
    eval_row = {
        "number": 1,
        "status": "warn",
        "reason": "over_duration",
        "text": "xin chao",
        "duration_ratio": 1.2,
    }
    action = choose_repair_action(eval_row, _task_row())
    assert action.action == "speed_fit_existing_audio"
    assert not action.rewrite
    assert action.backend is None


def test_fast_speech_action_rewrites_instead_of_accepting_acceleration():
    eval_row = {
        "number": 1,
        "status": "warn",
        "reason": "speech_rate_fast",
        "text": "xin chao",
        "speed_factor": 1.22,
    }
    action = choose_repair_action(eval_row, _task_row())
    assert action.action == "rewrite_and_regenerate"
    assert action.rewrite
    assert action.backend == "indextts2"


def test_over_duration_extreme_routes_to_manual_timeline_review():
    eval_row = {
        "number": 1,
        "status": "warn",
        "reason": "over_duration",
        "text": "xin chao",
        "duration_ratio": 3.5,
    }
    action = choose_repair_action(eval_row, _task_row())
    assert action.action == "timeline_or_manual_review"
    assert not action.delete_audio


def test_reference_leak_prefers_clean_ref_backend():
    eval_row = {"number": 1, "status": "warn", "reason": "reference_leak", "text": "xin chao"}
    action = choose_repair_action(eval_row, _task_row(ref_text="clean reference text"))
    assert action.action == "regenerate_with_backend"
    assert action.backend == "qwen3_tts"


def test_vietnamese_low_content_prefers_indextts2(monkeypatch):
    monkeypatch.setattr("core.providers.dubbing_repair.load_key", lambda key, default=None: "vi" if key == "target_language" else default)
    eval_row = {"number": 1, "status": "warn", "reason": "low_content_score", "text": "xin chao"}
    action = choose_repair_action(eval_row, _task_row(ref_text="clean reference text"))
    assert action.action == "regenerate_with_backend"
    assert action.backend == "indextts2"


def test_non_vietnamese_low_content_falls_back_to_edge_tts(monkeypatch, tmp_path):
    def fake_load_key(key, default=None):
        if key == "target_language":
            return "en"
        if key == "dubbing_repair.low_content_fallback_backend":
            return "edge_tts"
        return default

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("core.providers.dubbing_repair.load_key", fake_load_key)
    eval_row = {"number": 1, "status": "warn", "reason": "low_content_score", "text": "hello"}
    action = choose_repair_action(eval_row, _task_row(ref_text="clean reference text"))
    assert action.action == "regenerate_with_backend"
    assert action.backend == "edge_tts"
    assert "Edge TTS" in " ".join(action.notes)


def test_low_content_uses_run_state_target_over_stale_config(monkeypatch, tmp_path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "pipeline_state.json").write_text('{"target": "en"}', encoding="utf-8")

    def fake_load_key(key, default=None):
        if key == "target_language":
            return "vi"
        if key == "dubbing_repair.low_content_fallback_backend":
            return "edge_tts"
        return default

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("core.providers.dubbing_repair.load_key", fake_load_key)
    eval_row = {"number": 1, "status": "warn", "reason": "low_content_score", "text": "hello"}
    action = choose_repair_action(eval_row, _task_row(ref_text="clean reference text"))
    assert action.backend == "edge_tts"


def test_repeated_asr_quality_repair_routes_to_manual_review(monkeypatch):
    def fake_load_key(key, default=None):
        if key == "dubbing_repair.max_auto_regeneration_attempts":
            return 2
        return default

    monkeypatch.setattr("core.providers.dubbing_repair.load_key", fake_load_key)
    eval_row = {"number": 1, "status": "warn", "reason": "low_content_score", "text": "hello"}
    task_row = _task_row(repair_attempts=2, repair_status="rewrite_failed")
    action = choose_repair_action(eval_row, task_row)
    assert action.action == "manual_review"
    assert not action.delete_audio


def test_repair_preserves_edge_fallback_for_later_speed_repairs(monkeypatch):
    monkeypatch.setattr("core.providers.dubbing_repair.load_key", lambda key, default=None: default)
    eval_row = {
        "number": 1,
        "status": "warn",
        "reason": "speech_rate_fast",
        "text": "hello",
        "real_dur": 2.0,
        "duration": 1.0,
        "speed_factor": 1.2,
    }
    task_row = _task_row(tts_method="edge_tts", speed_factor=1.2, real_dur=2.0, duration=1.0)
    action = choose_repair_action(eval_row, task_row)
    assert action.action == "rewrite_and_regenerate"
    assert action.backend == "edge_tts"


def test_repair_routes_exhausted_rewrite_rows_to_manual_review(monkeypatch):
    def fake_load_key(key, default=None):
        if key == "dubbing_repair.max_rewrite_rounds":
            return 2
        return default

    monkeypatch.setattr("core.providers.dubbing_repair.load_key", fake_load_key)
    eval_row = {
        "number": 1,
        "status": "warn",
        "reason": "speech_rate_fast",
        "text": "hello",
        "real_dur": 2.0,
        "duration": 1.0,
        "speed_factor": 1.2,
    }
    action = choose_repair_action(eval_row, _task_row(repair_rewrite_rounds=2))
    assert action.action == "manual_review"
    assert not action.delete_audio


def test_build_repair_plan_can_write_json(tmp_path: Path):
    tasks_df = pd.DataFrame([_task_row()])
    eval_df = pd.DataFrame([{"number": 1, "status": "fail", "reason": "missing_audio", "text": "xin chao"}])
    plan = build_repair_plan(tasks_df, eval_df, limit=1)
    path = write_repair_plan(plan, str(tmp_path / "plan.json"))
    assert plan["planned"] == 1
    assert plan["reason_counts"]["missing_audio"] == 1
    assert Path(path).exists()


def test_build_repair_plan_prioritizes_missing_audio_before_warning():
    tasks_df = pd.DataFrame([_task_row(number=1), _task_row(number=2)])
    eval_df = pd.DataFrame([
        {"number": 1, "status": "warn", "reason": "over_duration", "text": "too long"},
        {"number": 2, "status": "fail", "reason": "missing_audio", "text": "missing"},
    ])
    plan = build_repair_plan(tasks_df, eval_df, limit=1)
    assert plan["items"][0]["number"] == 2


def test_build_repair_plan_can_filter_by_reason():
    tasks_df = pd.DataFrame([_task_row(number=1), _task_row(number=2)])
    eval_df = pd.DataFrame([
        {"number": 1, "status": "warn", "reason": "over_duration", "text": "too long"},
        {"number": 2, "status": "fail", "reason": "missing_audio", "text": "missing"},
    ])
    plan = build_repair_plan(tasks_df, eval_df, reasons="over_duration")
    assert plan["reason_filter"] == ["over_duration"]
    assert [item["number"] for item in plan["items"]] == [1]


def test_build_repair_plan_can_filter_by_action():
    tasks_df = pd.DataFrame([_task_row(number=1), _task_row(number=2, speed_factor=0.95)])
    eval_df = pd.DataFrame([
        {"number": 1, "status": "warn", "reason": "under_duration", "text": "too short"},
        {
            "number": 2,
            "status": "warn",
            "reason": "under_duration",
            "text": "stretchable",
            "available_duration": 4.0,
            "final_audio_dur": 3.6,
            "speed_factor": 0.95,
        },
    ])
    plan = build_repair_plan(tasks_df, eval_df, actions="slow_fit_existing_audio")
    assert plan["action_filter"] == ["slow_fit_existing_audio"]
    assert [item["number"] for item in plan["items"]] == [2]


def test_over_duration_report_classifies_triage_buckets():
    assert classify_over_duration({"duration_ratio": 1.2}) == OVER_DURATION_SPEED_FIT
    assert classify_over_duration({"duration_ratio": 1.6}) == OVER_DURATION_REWRITE
    assert classify_over_duration({"duration_ratio": 3.0}) == OVER_DURATION_MANUAL

    tasks_df = pd.DataFrame([_task_row(number=1), _task_row(number=2), _task_row(number=3)])
    eval_df = pd.DataFrame([
        {"number": 1, "status": "warn", "reason": "over_duration", "text": "a", "duration_ratio": 1.2},
        {"number": 2, "status": "warn", "reason": "over_duration", "text": "b", "duration_ratio": 1.6},
        {"number": 3, "status": "warn", "reason": "over_duration", "text": "c", "duration_ratio": 3.0},
    ])
    report = summarize_over_duration(tasks_df, eval_df)
    assert report["counts"][OVER_DURATION_SPEED_FIT] == 1
    assert report["counts"][OVER_DURATION_REWRITE] == 1
    assert report["counts"][OVER_DURATION_MANUAL] == 1


def test_apply_repair_plan_dry_run_does_not_regenerate(tmp_path: Path):
    tasks_path = tmp_path / "tasks.xlsx"
    pd.DataFrame([_task_row()]).to_excel(tasks_path, index=False)
    plan = {
        "plan_path": str(tmp_path / "plan.json"),
        "items": [
            {
                "number": 1,
                "action": "regenerate",
                "reasons": ["missing_audio"],
                "delete_audio": True,
            }
        ],
    }
    summary = apply_repair_plan(plan, dry_run=True, tasks_path=str(tasks_path))
    assert summary.planned == 1
    assert summary.applied == 0
    assert summary.dry_run


def test_apply_repair_plan_executes_expand_regenerate(monkeypatch, tmp_path: Path):
    tasks_path = tmp_path / "tasks.xlsx"
    pd.DataFrame([_task_row(asr_content_score=0.5, asr_transcript="old", asr_status="ok")]).to_excel(tasks_path, index=False)
    calls = {}

    def fake_rewrite(row, reason, **kwargs):
        calls["direction"] = kwargs.get("direction")
        return ["xin chao them"]

    def fake_process(row, tasks_df):
        calls["processed_text"] = row["text"]
        return int(row["number"]), 2.0

    monkeypatch.setattr("core.dubbing_rewrite.rewrite_task_lines", fake_rewrite)
    monkeypatch.setattr("core.step10_gen_audio.process_row", fake_process)
    monkeypatch.setattr("core.step10_gen_audio.merge_chunks", lambda df: df)
    monkeypatch.setattr("core.providers.dubbing_repair._delete_audio_for_row", lambda row: None)
    monkeypatch.setattr("core.providers.dubbing_repair.write_dubbing_eval", lambda df: {"ok": 1, "warn": 0, "fail": 0})

    plan = {
        "plan_path": str(tmp_path / "plan.json"),
        "items": [
            {
                "number": 1,
                "action": "expand_and_regenerate",
                "reasons": ["under_duration"],
                "rewrite": True,
                "delete_audio": True,
                "backend": "indextts2",
            }
        ],
    }

    summary = apply_repair_plan(plan, dry_run=False, full_remap=True, tasks_path=str(tasks_path))

    assert summary.applied == 1
    assert calls["direction"] == "expand"
    assert calls["processed_text"] == "xin chao them"
    updated = pd.read_excel(tasks_path)
    assert pd.isna(updated.loc[0, "asr_content_score"])
    assert pd.isna(updated.loc[0, "asr_transcript"])


def test_apply_repair_plan_uses_independent_rewrite_rounds(monkeypatch, tmp_path: Path):
    tasks_path = tmp_path / "tasks.xlsx"
    pd.DataFrame([_task_row(dubbing_rewrite_rounds=2, repair_rewrite_rounds=0, speed_factor=1.8)]).to_excel(tasks_path, index=False)
    calls = {}

    def fake_load_key(key, default=None):
        if key == "dubbing_repair.max_rewrite_rounds":
            return 1
        return default

    def fake_rewrite(row, reason, **kwargs):
        calls["direction"] = kwargs.get("direction")
        return ["short text"]

    def fake_process(row, tasks_df):
        calls["processed_text"] = row["text"]
        calls["speed_factor"] = row["speed_factor"]
        return int(row["number"]), 1.8

    monkeypatch.setattr("core.providers.dubbing_repair.load_key", fake_load_key)
    monkeypatch.setattr("core.dubbing_rewrite.rewrite_task_lines", fake_rewrite)
    monkeypatch.setattr("core.step10_gen_audio.process_row", fake_process)
    monkeypatch.setattr("core.step10_gen_audio.merge_chunks", lambda df: df)
    monkeypatch.setattr("core.providers.dubbing_repair._delete_audio_for_row", lambda row: None)
    monkeypatch.setattr("core.providers.dubbing_repair.write_dubbing_eval", lambda df: {"ok": 1, "warn": 0, "fail": 0})

    plan = {
        "plan_path": str(tmp_path / "plan.json"),
        "items": [
            {
                "number": 1,
                "action": "rewrite_and_regenerate",
                "reasons": ["speech_rate_fast"],
                "rewrite": True,
                "delete_audio": True,
                "backend": "indextts2",
            }
        ],
    }

    summary = apply_repair_plan(plan, dry_run=False, full_remap=True, tasks_path=str(tasks_path))

    assert summary.applied == 1
    assert calls["direction"] == "shorten"
    assert calls["processed_text"] == "short text"
    assert calls["speed_factor"] == 1.0
    updated = pd.read_excel(tasks_path)
    assert updated.loc[0, "repair_rewrite_rounds"] == 1
    assert updated.loc[0, "dubbing_rewrite_rounds"] == 3


def test_apply_repair_plan_skips_unchanged_rewrite(monkeypatch, tmp_path: Path):
    tasks_path = tmp_path / "tasks.xlsx"
    pd.DataFrame([_task_row(text="same text", lines=["same text"], dubbing_rewrite_rounds=2)]).to_excel(tasks_path, index=False)

    def fake_rewrite(row, reason, **kwargs):
        return ["same text"]

    def fail_process(row, tasks_df):
        raise AssertionError("unchanged rewrite must not regenerate audio")

    monkeypatch.setattr("core.dubbing_rewrite.rewrite_task_lines", fake_rewrite)
    monkeypatch.setattr("core.step10_gen_audio.process_row", fail_process)
    monkeypatch.setattr("core.providers.dubbing_repair._delete_audio_for_row", lambda row: None)
    monkeypatch.setattr("core.providers.dubbing_repair.write_dubbing_eval", lambda df: {"ok": 0, "warn": 1, "fail": 0})

    plan = {
        "plan_path": str(tmp_path / "plan.json"),
        "items": [
            {
                "number": 1,
                "action": "rewrite_and_regenerate",
                "reasons": ["speech_rate_fast"],
                "rewrite": True,
                "delete_audio": True,
                "backend": "indextts2",
            }
        ],
    }

    summary = apply_repair_plan(plan, dry_run=False, full_remap=True, tasks_path=str(tasks_path))

    assert summary.applied == 0
    assert summary.skipped == 1
    updated = pd.read_excel(tasks_path)
    assert updated.loc[0, "repair_action"] == "manual_review"
    assert updated.loc[0, "repair_status"] == "rewrite_failed"


def test_append_repair_history_writes_jsonl(tmp_path: Path):
    path = tmp_path / "history.jsonl"
    append_repair_history({"batch": 1, "after": {"fail": 3}}, str(path))
    append_repair_history({"batch": 2, "after": {"fail": 2}}, str(path))
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert '"batch": 2' in lines[1]
