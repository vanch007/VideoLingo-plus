from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.step_checker import STEP_CHECKPOINTS, is_step_completed


@dataclass(frozen=True)
class ArtifactStatus:
    step: str
    path: str
    completed: bool
    manifest: dict[str, Any] | None = None


def list_artifacts(*, include_manifests: bool = False) -> list[ArtifactStatus]:
    statuses: list[ArtifactStatus] = []
    for step, path in STEP_CHECKPOINTS.items():
        manifest = None
        if include_manifests:
            from core.pipeline.artifact_manifest import inspect_artifact_manifest

            manifest = inspect_artifact_manifest(path, expected_step=step).to_dict()
        statuses.append(ArtifactStatus(step=step, path=path, completed=is_step_completed(step), manifest=manifest))
    return statuses


def pending_steps(step_order: list[str]) -> list[str]:
    return [step for step in step_order if not is_step_completed(step)]
