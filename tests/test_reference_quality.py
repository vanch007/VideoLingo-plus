import json

from core.pipeline.artifact_manifest import hash_path
from core.subtitle_quality import subtitle_quality_report


def test_verified_reference_failure_overrides_pending_automatic_checks(tmp_path):
    artifact = tmp_path / "translation_results.xlsx"
    artifact.write_bytes(b"assessed candidate")
    report = tmp_path / "reference_translation_assessment.json"
    report.write_text(json.dumps({"status": "fail", "translation_sha256": hash_path(artifact)}))
    quality = subtitle_quality_report(tmp_path)
    assert quality["status"] == "fail"
    assert quality["reference_translation_quality"] == "fail"
    assert quality["accepted"] is False
    artifact.write_bytes(b"a changed candidate")
    assert subtitle_quality_report(tmp_path)["reference_translation_quality"] == "stale_evidence"


def test_reference_pass_never_establishes_missing_other_quality_evidence(tmp_path):
    artifact = tmp_path / "translation_results.xlsx"
    artifact.write_bytes(b"assessed translation")
    (tmp_path / "reference_translation_assessment.json").write_text(json.dumps(
        {"status": "pass", "translation_sha256": hash_path(artifact)}))
    quality = subtitle_quality_report(tmp_path)
    assert quality["reference_translation_quality"] == "pass"
    assert quality["automatic_gate_passed"] is False
    assert quality["accepted"] is False


def test_delivery_rejection_survives_semantic_pass_only_for_assessed_artifact(tmp_path):
    artifact = tmp_path / "translation_results.xlsx"
    artifact.write_bytes(b"readable meaning but unreadable display")
    (tmp_path / "reference_translation_assessment.json").write_text(json.dumps({
        "status": "pass", "translation_sha256": hash_path(artifact),
        "formal_delivery_status": "fail",
    }))
    quality = subtitle_quality_report(tmp_path)
    assert quality["reference_translation_quality"] == "pass"
    assert quality["reference_delivery_quality"] == "fail"
    assert quality["status"] == "fail" and quality["accepted"] is False
    artifact.write_bytes(b"new candidate needs new assessment")
    updated = subtitle_quality_report(tmp_path)
    assert updated["reference_delivery_quality"] == "missing_evidence"
    assert updated["status"] == "pending"
