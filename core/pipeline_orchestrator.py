from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineStep:
    name: str
    action: Callable[[], object]


def run_pipeline(steps: list[PipelineStep], attempts: int = 3, on_step=None):
    """Run a recoverable ordered pipeline shared by UI and batch callers."""
    for step in steps:
        if on_step:
            on_step(step.name, 1, attempts)
        for attempt in range(1, attempts + 1):
            try:
                if on_step and attempt > 1:
                    on_step(step.name, attempt, attempts)
                result = step.action()
                if result is not None and isinstance(result, dict):
                    globals().update(result)
                break
            except Exception:
                if attempt == attempts:
                    raise
