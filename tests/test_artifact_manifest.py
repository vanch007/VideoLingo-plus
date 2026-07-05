from pathlib import Path

from core.pipeline.artifact_manifest import (
    build_artifact_manifest,
    inspect_artifact_manifest,
    manifest_path_for,
    validate_artifact_manifest,
    write_artifact_manifest,
)


def _context(input_path: Path) -> dict:
    return {
        "run_id": "run-1",
        "input_path": str(input_path),
        "profile": "cinematic",
        "source_language": "zh",
        "target_language": "vi",
    }


def test_artifact_manifest_validates_matching_sidecar(tmp_path):
    source = tmp_path / "source.mp4"
    source.write_text("video", encoding="utf-8")
    artifact = tmp_path / "tts_tasks.xlsx"
    artifact.write_text("rows", encoding="utf-8")
    manifest = build_artifact_manifest(
        artifact=artifact,
        step="gen_audio_task",
        run_context=_context(source),
        code_paths=["core/step8_1_gen_audio_task.py"],
        models={"tts": {"method": "mlx_indextts2"}},
        quality={"config_hash": "sha256:quality"},
        schema={"required_columns": ["number", "start_time", "end_time", "text"]},
    )
    write_artifact_manifest(manifest)

    check = validate_artifact_manifest(artifact=artifact, expected=manifest)

    assert check.ok is True
    assert check.status == "pass"
    assert check.reasons == []


def test_artifact_manifest_requires_sidecar(tmp_path):
    source = tmp_path / "source.mp4"
    source.write_text("video", encoding="utf-8")
    artifact = tmp_path / "tts_tasks.xlsx"
    artifact.write_text("rows", encoding="utf-8")
    expected = build_artifact_manifest(artifact=artifact, step="gen_audio_task", run_context=_context(source))

    check = validate_artifact_manifest(artifact=artifact, expected=expected)

    assert check.ok is False
    assert "manifest_missing" in check.reasons


def test_artifact_manifest_reports_quality_config_mismatch(tmp_path):
    source = tmp_path / "source.mp4"
    source.write_text("video", encoding="utf-8")
    artifact = tmp_path / "tts_tasks.xlsx"
    artifact.write_text("rows", encoding="utf-8")
    manifest = build_artifact_manifest(
        artifact=artifact,
        step="gen_audio_task",
        run_context=_context(source),
        quality={"config_hash": "sha256:old"},
    )
    write_artifact_manifest(manifest)
    expected = build_artifact_manifest(
        artifact=artifact,
        step="gen_audio_task",
        run_context=_context(source),
        quality={"config_hash": "sha256:new"},
    )

    check = validate_artifact_manifest(artifact=artifact, expected=expected)

    assert check.ok is False
    assert "manifest_mismatch:quality.config_hash" in check.reasons


def test_artifact_manifest_keeps_structural_validator_in_gate(tmp_path):
    source = tmp_path / "source.mp4"
    source.write_text("video", encoding="utf-8")
    artifact = tmp_path / "tts_tasks.xlsx"
    artifact.write_text("rows", encoding="utf-8")
    manifest = build_artifact_manifest(artifact=artifact, step="gen_audio_task", run_context=_context(source))
    write_artifact_manifest(manifest)

    check = validate_artifact_manifest(
        artifact=artifact,
        expected=manifest,
        structural_validator=lambda path: path.name != "tts_tasks.xlsx",
    )

    assert check.ok is False
    assert "artifact_invalid" in check.reasons


def test_artifact_manifest_sidecar_name_matches_schema(tmp_path):
    artifact = tmp_path / "tts_tasks.xlsx"

    assert manifest_path_for(artifact) == tmp_path / "tts_tasks.xlsx.manifest.json"


def test_artifact_manifest_audit_reports_step_mismatch(tmp_path):
    source = tmp_path / "source.mp4"
    source.write_text("video", encoding="utf-8")
    artifact = tmp_path / "tts_tasks.xlsx"
    artifact.write_text("rows", encoding="utf-8")
    manifest = build_artifact_manifest(artifact=artifact, step="gen_audio_task", run_context=_context(source))
    write_artifact_manifest(manifest)

    audit = inspect_artifact_manifest(artifact, expected_step="translate")

    assert audit.ok is False
    assert audit.status == "pending"
    assert "manifest_step_mismatch" in audit.reasons
