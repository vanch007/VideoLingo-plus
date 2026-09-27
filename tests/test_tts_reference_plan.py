import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from core import tts_reference_plan as refs
from core.all_tts_functions import mlx_router
from core.providers.contracts import TTSRequest
from core.providers.mlx_tts import IndexTTS2Backend, MlxTTSRouter


@pytest.fixture
def scene(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "output/audio"
    (root / "refers").mkdir(parents=True)
    sr = 16000
    samples = 0.2 * np.sin(2 * np.pi * 250 * np.arange(sr * 5) / sr)
    sf.write(root / "vocal.wav", samples, sr)
    for number in (1, 2):
        sf.write(root / "refers" / f"{number}.wav", samples[:sr], sr)
    config = {
        "tts_method": "mlx_indextts2",
        "mlx_tts.reference_strategy": "speaker_anchor",
        "mlx_tts.backends.indextts2": {"emotion_reference_strategy": "source_row", "emo_alpha": 1.0},
    }
    monkeypatch.setattr(refs, "load_key", lambda k, d=None: config.get(k, d))
    monkeypatch.setattr(mlx_router, "load_key", lambda k, d=None: config.get(k, d))
    monkeypatch.setattr(mlx_router, "effective_target_language", lambda _: "en")
    monkeypatch.setattr(mlx_router, "effective_source_language", lambda _: "zh")
    tasks = pd.DataFrame([
        {"number": 1, "speaker": "S01", "start_time": "00:00:01.000", "end_time": "00:00:02.000", "origin": "原句一"},
        {"number": 2, "speaker": "S01", "start_time": "00:00:02.000", "end_time": "00:00:02.460", "origin": "啊"},
    ])
    return root, tasks, config


def test_same_speaker_keeps_anchor_but_changes_emotion_per_utterance(scene):
    root, tasks, _ = scene
    a, b = [refs.resolve_reference_plan(r.to_dict(), tasks) for _, r in tasks.iterrows()]
    assert a["speaker_reference"] == b["speaker_reference"]
    assert a["emotion_reference"] != b["emotion_reference"]
    assert sf.info(b["emotion_reference"]["path"]).duration == pytest.approx(0.46)
    assert b["emo_alpha"] == 1
    assert "short_emotion_reference_review_required" in b["warnings"]
    assert b["emotion_source_window"]["start"] == 2


def test_source_or_timing_change_invalidates_content_addressed_emotion(scene):
    root, tasks, _ = scene
    row = tasks.iloc[0].to_dict()
    before = refs.resolve_reference_plan(row, tasks)
    changed = refs.resolve_reference_plan({**row, "start_time": "00:00:01.100"}, tasks)
    assert before["fingerprint"] != changed["fingerprint"]
    audio, sr = sf.read(root / "vocal.wav")
    sf.write(root / "vocal.wav", audio * 0.8, sr)
    after = refs.resolve_reference_plan(row, tasks)
    assert before["emotion_reference"] != after["emotion_reference"]


def test_explicit_reference_and_weight_survive_rewrite(scene):
    root, tasks, _ = scene
    row = {**tasks.iloc[1].to_dict(), "emotion_ref": str(root / "refers/2.wav"), "emo_alpha": 0.85}
    first = mlx_router.build_mlx_tts_request("First wording", "first.wav", 2, tasks, row)
    retry = mlx_router.build_mlx_tts_request("Short wording", "retry.wav", 2, tasks, row)
    assert retry.ref_audio == first.ref_audio
    assert retry.emotion_ref == first.emotion_ref == row["emotion_ref"]
    assert retry.emo_alpha == 0.85
    assert retry.metadata["reference_plan"]["emotion_source"] == "explicit"


def test_no_emotion_reference_is_not_silent_neutral_fallback(scene):
    root, tasks, _ = scene
    (root / "vocal.wav").unlink()
    with pytest.raises(FileNotFoundError, match="Separated vocal"):
        refs.resolve_reference_plan(tasks.iloc[0].to_dict(), tasks)


def test_overlapping_rows_are_not_used_as_emotion_reference(scene):
    _, tasks, _ = scene
    tasks.at[1, "start_time"] = "00:00:01.900"
    with pytest.raises(ValueError, match="Overlapping"):
        refs.resolve_reference_plan(tasks.iloc[0].to_dict(), tasks)


def test_unknown_speaker_does_not_borrow_known_voice(scene):
    _, tasks, _ = scene
    tasks.at[1, "speaker"] = None
    plan = refs.resolve_reference_plan(tasks.iloc[1].to_dict(), tasks)
    assert plan["speaker_reference"]["path"].endswith("2.wav")
    assert "speaker_identity_unknown" in plan["warnings"]


def test_single_adapter_passes_separate_reference_and_weight(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(IndexTTS2Backend, "_run", lambda self, cmd, request: captured.update(cmd=cmd))
    request = TTSRequest(text="Hello", output_path="line.wav", language="en",
                         ref_audio="speaker.wav", emotion_ref="angry.wav", emo_alpha=0.9)
    IndexTTS2Backend({"root": str(tmp_path)}).synthesize(request)
    cmd = captured["cmd"]
    assert cmd[cmd.index("--emotion-ref-audio") + 1].endswith("angry.wav")
    assert cmd[cmd.index("--emo-alpha") + 1] == "0.9"
    assert cmd[cmd.index("--language") + 1] == "en"
    with pytest.raises(ValueError, match="emo_alpha"):
        IndexTTS2Backend({}).synthesize(replace(request, emo_alpha=float("nan")))


def test_emotion_required_blocks_speaker_only_backend(monkeypatch):
    monkeypatch.setattr(MlxTTSRouter, "__init__", lambda self: None)
    monkeypatch.setattr(MlxTTSRouter, "select_backend", lambda self, request: "dots")
    with pytest.raises(ValueError, match="silent speaker-only fallback"):
        MlxTTSRouter().synthesize(TTSRequest(text="Hello", output_path="line.wav",
                                           metadata={"requires_separate_emotion": True}))


def test_content_repair_backend_override_cannot_disable_emotion(scene):
    _, tasks, _ = scene
    row = {**tasks.iloc[0].to_dict(), "tts_method": "mlx_dots_tts", "tts_backend": "dots"}
    request = mlx_router.build_mlx_tts_request("Retry", "retry.wav", 1, tasks, row)
    assert request.emotion_ref
    assert request.metadata["requires_separate_emotion"] is True


def test_content_repair_skips_incompatible_backend_before_deleting_audio(monkeypatch):
    from core import step10_gen_audio as step10

    monkeypatch.setattr(step10, "load_key", lambda key, default=None: {
        "dubbing_repair.low_content_fallback_backend": "dots",
        "mlx_tts.backends.indextts2.emotion_reference_strategy": "source_row",
    }.get(key, default))
    assert step10._content_fallback_method() == ("", "")


def test_manifest_is_plan_not_perceptual_acceptance(scene):
    root, tasks, _ = scene
    planned = refs.prepare_reference_plan(tasks)
    assert "reference_plan_fingerprint" in planned
    manifest = json.loads((root / "reference_manifest.json").read_text())
    assert manifest["stage"] == "planned"
    assert all(r["emotion_fidelity"] == "missing evidence" for r in manifest["rows"])


def test_resume_invalidates_changed_reference_without_writing_clips(scene):
    root, tasks, _ = scene
    planned = refs.prepare_reference_plan(tasks)
    assert refs.reference_plan_is_current(planned)
    files_before = sorted((root / "emotion_refers").iterdir())
    planned.at[0, "start_time"] = "00:00:01.100"
    assert not refs.reference_plan_is_current(planned)
    assert sorted((root / "emotion_refers").iterdir()) == files_before


def test_stale_generation_also_reschedules_merges(monkeypatch):
    from core.pipeline import artifacts

    monkeypatch.setattr(artifacts, "is_step_completed", lambda step: step != "gen_audio")
    assert artifacts.pending_steps(["transcribe", "gen_audio", "merge_audio", "merge_video"]) == [
        "gen_audio", "merge_audio", "merge_video"
    ]


def test_content_gate_pass_is_not_overall_perceptual_acceptance(tmp_path, monkeypatch):
    from core.pipeline.runner import PipelineRun, run_pipeline

    monkeypatch.chdir(tmp_path)
    target = tmp_path / "output/audio/dubbing_eval.json"
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps({"summary": {
        "quality_gate": {"passed": True}, "perceptual_quality": {"status": "pending"}
    }}))
    run = run_pipeline(PipelineRun("pilot", "cinematic", "zh", "en", []), [], resume=False)
    assert run.status == "rendered"
    assert run.quality_status == "pending"


def test_generation_checkpoint_binds_emotion_contents_and_weight(scene, monkeypatch):
    from core import step10_gen_audio as step10

    root, tasks, _ = scene
    monkeypatch.setattr(step10, "load_key", lambda key, default=None: default)
    monkeypatch.setattr(step10, "TEMP_FILE_TEMPLATE", str(root / "{}_temp.wav"))
    planned = refs.prepare_reference_plan(tasks)
    planned["lines"] = [["Hello"], ["Hi"]]
    planned["real_dur"] = 1.0
    for number in (1, 2):
        sf.write(root / f"{number}_0_temp.wav", np.ones(16000) * 0.01, 16000)
    saved = step10.mark_tts_generation_checkpoint(planned)
    assert step10.tts_generation_checkpoint_valid(saved)
    path = saved.iloc[0]["resolved_emotion_ref"]
    audio, sr = sf.read(path)
    sf.write(path, audio * 0.5, sr)
    assert not step10.tts_generation_checkpoint_valid(saved)


def test_receipt_rejects_old_audio_when_emotion_changes(scene):
    root, tasks, _ = scene
    row = tasks.iloc[0].to_dict()
    out = str(root / "generated.wav")
    request = mlx_router.build_mlx_tts_request("Hello", out, 1, tasks, row)
    sf.write(out, np.ones(16000) * 0.01, 16000)
    assert not refs.conditioning_receipt_matches(request)
    refs.write_conditioning_receipt(request)
    assert refs.conditioning_receipt_matches(request)
    changed = mlx_router.build_mlx_tts_request("Hello", out, 1, tasks, {**row, "emo_alpha": 0.8})
    assert not refs.conditioning_receipt_matches(changed)


def test_hybrid_long_utterance_uses_source_for_both_without_anchor(scene, monkeypatch):
    _, tasks, config = scene
    config["mlx_tts.reference_strategy"] = "hybrid"
    tasks.at[0, "start_time"] = "00:00:00.000"
    tasks.at[0, "origin"] = "这是一句完整的原始台词"
    monkeypatch.setattr(refs, "get_reference_audio_path", lambda *a, **k: pytest.fail("Long row should not select an anchor"))
    request = mlx_router.build_mlx_tts_request("Long translated line", "long.wav", 1, tasks, tasks.iloc[0].to_dict())
    plan = request.metadata["reference_plan"]
    assert request.ref_audio == request.emotion_ref
    assert plan["speaker_strategy"] == "row_shared"
    assert plan["speaker_reference_number"] == 1


@pytest.mark.parametrize("origin,fragment,reason", [
    ("这是一句完整的原始台词", True, "fragment"),
    ("啊", False, "short_source_text"),
    ("飞飞飞飞", False, "short_source_text"),
    ("Wait wait wait", False, "short_source_text"),
])
def test_hybrid_fragments_and_held_replies_keep_anchor(scene, origin, fragment, reason):
    _, tasks, config = scene
    config["mlx_tts.reference_strategy"] = "hybrid"
    tasks.at[0, "start_time"] = "00:00:00.000"
    tasks.at[0, "origin"] = origin
    row = {**tasks.iloc[0].to_dict(), "reference_is_fragment": fragment}
    plan = refs.resolve_reference_plan(row, tasks)
    assert plan["speaker_strategy"] == "speaker_anchor"
    assert plan["speaker_reference"] != plan["emotion_reference"]
    assert plan["reference_policy"]["decision_reason"] == reason


def test_hybrid_short_source_is_independent_of_target_text_length(scene):
    _, tasks, config = scene
    config["mlx_tts.reference_strategy"] = "hybrid"
    row = {**tasks.iloc[1].to_dict(), "text": "A very long translation cannot turn a short source into a stable voice reference."}
    plan = refs.resolve_reference_plan(row, tasks)
    assert plan["speaker_strategy"] == "speaker_anchor"


def test_hybrid_manual_reference_wins_and_policy_changes_invalidate_cache(scene):
    root, tasks, config = scene
    config["mlx_tts.reference_strategy"] = "hybrid"
    tasks.at[0, "start_time"] = "00:00:00.000"
    tasks.at[0, "origin"] = "这是一句完整的原始台词"
    row = {**tasks.iloc[0].to_dict(), "ref_audio": str(root / "refers/2.wav")}
    plan = refs.resolve_reference_plan(row, tasks)
    assert plan["speaker_strategy"] == "explicit"
    assert plan["speaker_reference"]["path"] == row["ref_audio"]
    config["mlx_tts.hybrid_short_threshold_seconds"] = 3.0
    assert refs.resolve_reference_plan(row, tasks)["fingerprint"] != plan["fingerprint"]
