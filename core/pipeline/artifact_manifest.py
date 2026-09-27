from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable


SUPPORTED_MANIFEST_VERSION = 1
DEFAULT_COMPARE_PATHS = (
    "input.sha256",
    "output_sha256",
    "profile",
    "source_language",
    "target_language",
    "step",
    "models",
    "quality.config_hash",
    "step_version.code_hashes",
)


@dataclass(frozen=True)
class ManifestCheck:
    ok: bool
    status: str
    artifact: str
    manifest_path: str
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ManifestAudit:
    ok: bool
    status: str
    artifact: str
    manifest_path: str
    artifact_exists: bool
    manifest_exists: bool
    reasons: list[str] = field(default_factory=list)
    version: int | None = None
    step: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def manifest_path_for(artifact: str | Path) -> Path:
    return Path(f"{artifact}.manifest.json")


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

def hash_path(path: str | Path) -> str | None:
    target = Path(path)
    if not target.exists() and not target.is_absolute():
        alt = PROJECT_ROOT / path
        if alt.exists():
            target = alt
    if not target.exists():
        return None
    digest = hashlib.sha256()
    if target.is_file():
        _update_digest_from_file(digest, target)
    elif target.is_dir():
        for child in sorted(item for item in target.rglob("*") if item.is_file()):
            digest.update(str(child.relative_to(target)).encode("utf-8"))
            _update_digest_from_file(digest, child)
    else:
        return None
    return f"sha256:{digest.hexdigest()}"


def _update_digest_from_file(digest: "hashlib._Hash", path: Path) -> None:
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)


def build_artifact_manifest(
    *,
    artifact: str | Path,
    step: str,
    run_context: dict[str, Any],
    code_paths: list[str] | None = None,
    models: dict[str, Any] | None = None,
    quality: dict[str, Any] | None = None,
    schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    input_path = run_context.get("input_path") or run_context.get("input")
    git_commit = run_context.get("git_commit")
    if not git_commit:
        try:
            import subprocess
            res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
            if res.returncode == 0:
                git_commit = res.stdout.strip()
        except Exception:
            pass
    producing_steps = run_context.get("producing_steps") or [step]
    if step not in producing_steps:
        producing_steps = list(dict.fromkeys(list(producing_steps) + [step]))
    return {
        "version": SUPPORTED_MANIFEST_VERSION,
        "artifact": str(artifact),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "run_id": run_context.get("run_id"),
        "input": {
            "path": str(input_path) if input_path else None,
            "sha256": hash_path(input_path) if input_path else None,
        },
        "output_sha256": hash_path(artifact),
        "profile": run_context.get("profile"),
        "source_language": run_context.get("source_language") or run_context.get("source"),
        "target_language": run_context.get("target_language") or run_context.get("target"),
        "step": step,
        "producing_steps": producing_steps,
        "step_version": {
            "git_commit": git_commit,
            "code_paths": code_paths or [],
            "code_hashes": {str(p): hash_path(p) for p in (code_paths or []) if hash_path(p)},
        },
        "models": models or run_context.get("models") or {},
        "quality": quality or run_context.get("quality") or {},
        "schema": schema or {},
    }


def write_artifact_manifest(manifest: dict[str, Any], path: str | Path | None = None) -> str:
    output = Path(path) if path else manifest_path_for(manifest["artifact"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(output)


def read_artifact_manifest(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def inspect_artifact_manifest(artifact: str | Path, *, expected_step: str | None = None) -> ManifestAudit:
    artifact_path = Path(artifact)
    sidecar = manifest_path_for(artifact_path)
    reasons: list[str] = []
    version = None
    step = None
    if not artifact_path.exists():
        reasons.append("artifact_missing")
    if not sidecar.exists():
        reasons.append("manifest_missing")
        return _audit_result(artifact_path, sidecar, reasons, version=version, step=step)
    try:
        manifest = read_artifact_manifest(sidecar)
    except Exception as exc:
        reasons.append(f"manifest_unreadable:{type(exc).__name__}")
        return _audit_result(artifact_path, sidecar, reasons, version=version, step=step)
    version = manifest.get("version")
    step = manifest.get("step")
    if version != SUPPORTED_MANIFEST_VERSION:
        reasons.append("unsupported_manifest_version")
    if manifest.get("artifact") not in {str(artifact_path), str(artifact)}:
        reasons.append("manifest_artifact_mismatch")
    recorded_steps = set(manifest.get("producing_steps") or [])
    if step:
        recorded_steps.add(step)
    if expected_step and expected_step not in recorded_steps:
        reasons.append("manifest_step_mismatch")
    recorded_output_sha = manifest.get("output_sha256")
    if not recorded_output_sha:
        reasons.append("manifest_missing_output_hash")
    elif artifact_path.exists():
        current_sha = hash_path(artifact_path)
        if current_sha != recorded_output_sha:
            reasons.append("artifact_content_mismatch")

    input_info = manifest.get("input", {})
    input_path = input_info.get("path")
    recorded_input_sha = input_info.get("sha256")
    if input_path:
        curr_input_sha = hash_path(input_path)
        if not curr_input_sha or not recorded_input_sha:
            reasons.append("input_missing")
        elif curr_input_sha != recorded_input_sha:
            reasons.append("input_content_mismatch")

    code_paths = manifest.get("step_version", {}).get("code_paths", [])
    code_hashes = manifest.get("step_version", {}).get("code_hashes", {})
    if code_paths and not code_hashes:
        reasons.append("code_hashes_missing")
    for path_str in code_paths:
        expected_hash = code_hashes.get(path_str)
        curr_hash = hash_path(path_str)
        if not expected_hash or not curr_hash:
            reasons.append(f"code_missing:{path_str}")
        elif curr_hash != expected_hash:
            reasons.append(f"code_content_mismatch:{path_str}")

    return _audit_result(artifact_path, sidecar, reasons, version=version, step=step)


def validate_artifact_manifest(
    *,
    artifact: str | Path,
    expected: dict[str, Any],
    structural_validator: Callable[[Path], bool] | None = None,
    compare_paths: tuple[str, ...] = DEFAULT_COMPARE_PATHS,
) -> ManifestCheck:
    artifact_path = Path(artifact)
    sidecar = manifest_path_for(artifact_path)
    reasons: list[str] = []
    if not artifact_path.exists():
        reasons.append("artifact_missing")
    if artifact_path.exists() and structural_validator and not structural_validator(artifact_path):
        reasons.append("artifact_invalid")
    if not sidecar.exists():
        reasons.append("manifest_missing")
        return _check_result(artifact_path, sidecar, reasons)
    try:
        manifest = read_artifact_manifest(sidecar)
    except Exception as exc:
        reasons.append(f"manifest_unreadable:{type(exc).__name__}")
        return _check_result(artifact_path, sidecar, reasons)
    if manifest.get("version") != SUPPORTED_MANIFEST_VERSION:
        reasons.append("unsupported_manifest_version")
    recorded_output_sha = manifest.get("output_sha256")
    if not recorded_output_sha:
        reasons.append("manifest_missing_output_hash")
    elif artifact_path.exists():
        current_sha = hash_path(artifact_path)
        if current_sha != recorded_output_sha:
            reasons.append("artifact_content_mismatch")
    for path in compare_paths:
        if _nested_get(manifest, path) != _nested_get(expected, path):
            reasons.append(f"manifest_mismatch:{path}")
    return _check_result(artifact_path, sidecar, reasons)


def _nested_get(data: dict[str, Any], path: str) -> Any:
    current: Any = data
    for part in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _check_result(artifact: Path, sidecar: Path, reasons: list[str]) -> ManifestCheck:
    return ManifestCheck(
        ok=not reasons,
        status="pass" if not reasons else "pending",
        artifact=str(artifact),
        manifest_path=str(sidecar),
        reasons=reasons,
    )


def _audit_result(
    artifact: Path,
    sidecar: Path,
    reasons: list[str],
    *,
    version: int | None,
    step: str | None,
) -> ManifestAudit:
    return ManifestAudit(
        ok=not reasons,
        status="pass" if not reasons else "pending",
        artifact=str(artifact),
        manifest_path=str(sidecar),
        artifact_exists=artifact.exists(),
        manifest_exists=sidecar.exists(),
        reasons=reasons,
        version=version,
        step=step,
    )
