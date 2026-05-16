from __future__ import annotations

from dataclasses import dataclass

from core.step_checker import STEP_CHECKPOINTS, is_step_completed


@dataclass(frozen=True)
class ArtifactStatus:
    step: str
    path: str
    completed: bool


def list_artifacts() -> list[ArtifactStatus]:
    return [
        ArtifactStatus(step=step, path=path, completed=is_step_completed(step))
        for step, path in STEP_CHECKPOINTS.items()
    ]


def pending_steps(step_order: list[str]) -> list[str]:
    return [step for step in step_order if not is_step_completed(step)]
