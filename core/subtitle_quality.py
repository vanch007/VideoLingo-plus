"""Shared subtitle evidence status for the pipeline and rendered candidate."""
import json
from pathlib import Path


def bound_assessment_status(path, artifact, hash_key):
    from core.pipeline.artifact_manifest import hash_path
    data = json.loads(path.read_text(encoding="utf-8"))
    expected, actual = data.get(hash_key), hash_path(artifact)
    if not expected or not actual or expected.removeprefix("sha256:") != actual.removeprefix("sha256:"):
        return "stale_evidence"
    return data.get("status", "missing_evidence")


def subtitle_quality_report(log_dir="output/log"):
    components = {}
    for name in ("source_verification", "lexical_boundary_review", "dialogue_role_review",
                 "translation_review", "subtitle_layout"):
        path = Path(log_dir) / f"{name}.json"
        components[name] = json.loads(path.read_text(encoding="utf-8")).get("status", "missing_evidence") if path.is_file() else "missing_evidence"
    source_reference = Path(log_dir) / "source_reference_assessment.json"
    if source_reference.is_file():
        components["source_reference_assessment"] = bound_assessment_status(
            source_reference, Path(log_dir) / "cleaned_chunks.xlsx", "transcript_sha256")
    reference = Path(log_dir) / "reference_translation_assessment.json"
    reference_quality = "missing_evidence"
    delivery_quality = "missing_evidence"
    if reference.is_file():
        reference_quality = bound_assessment_status(reference, Path(log_dir) / "translation_results.xlsx", "translation_sha256")
        components["reference_translation_assessment"] = reference_quality
        # An explicit delivery rejection must not disappear behind a semantic
        # subgate pass. Reuse it only for the exact assessed translation file.
        if reference_quality in {"pass", "fail"}:
            delivery_quality = json.loads(reference.read_text(encoding="utf-8")).get("formal_delivery_status", "missing_evidence")
            if delivery_quality != "missing_evidence":
                components["reference_delivery_assessment"] = delivery_quality
    return {"automatic_checks": components,
            "automatic_gate_passed": all(value == "pass" for value in components.values()),
            "human_quality": "pending", "reference_translation_quality": reference_quality,
            "reference_delivery_quality": delivery_quality, "accepted": False,
            "status": "fail" if any(value in {"failed", "fail"} for value in components.values()) else "pending"}
