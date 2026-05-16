from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from core.pipeline.artifacts import pending_steps


STATE_FILE = Path("output/pipeline_state.json")


@dataclass(frozen=True)
class PipelineStep:
    key: str
    label: str
    module: str | None = None
    action: Callable[[], object] | None = None


@dataclass
class PipelineRun:
    run_id: str
    profile: str
    source: str
    target: str
    steps: list[str]
    input: str = ""
    subtitle_only: bool = False
    no_subtitles: bool = False
    llm: str | None = None
    tts: str | None = None
    status: str = "created"
    started_at: float = field(default_factory=time.time)
    completed_steps: list[str] = field(default_factory=list)
    failed_step: str | None = None

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "profile": self.profile,
            "source": self.source,
            "target": self.target,
            "input": self.input,
            "subtitle_only": self.subtitle_only,
            "no_subtitles": self.no_subtitles,
            "llm": self.llm,
            "tts": self.tts,
            "steps": self.steps,
            "status": self.status,
            "started_at": self.started_at,
            "completed_steps": self.completed_steps,
            "failed_step": self.failed_step,
        }


def save_state(run: PipelineRun) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(run.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def load_state() -> dict:
    if not STATE_FILE.exists():
        raise FileNotFoundError(f"No pipeline state found at {STATE_FILE}")
    return json.loads(STATE_FILE.read_text(encoding="utf-8"))


def load_state_or_none() -> dict | None:
    if not STATE_FILE.exists():
        return None
    return load_state()


def _run_module(module: str) -> None:
    subprocess.run([sys.executable, "-m", module], check=True)


def run_pipeline(run: PipelineRun, steps: list[PipelineStep], *, resume: bool = True, dry_run: bool = False) -> PipelineRun:
    step_keys = [step.key for step in steps]
    todo = pending_steps(step_keys) if resume else step_keys
    run.steps = step_keys

    if dry_run:
        run.status = "dry_run"
        run.completed_steps = [key for key in step_keys if key not in todo]
        return run

    run.status = "running"
    save_state(run)

    for step in steps:
        if step.key not in todo:
            run.completed_steps.append(step.key)
            save_state(run)
            continue
        try:
            if step.action:
                step.action()
            elif step.module:
                _run_module(step.module)
            else:
                raise ValueError(f"Pipeline step has no action or module: {step.key}")
            run.completed_steps.append(step.key)
            save_state(run)
        except Exception:
            run.status = "failed"
            run.failed_step = step.key
            save_state(run)
            raise

    run.status = "completed"
    save_state(run)
    return run


def prepare_local_video(input_path: str) -> str:
    src = Path(input_path).expanduser()
    if not src.exists():
        raise FileNotFoundError(input_path)
    output = Path("output")
    output.mkdir(exist_ok=True)
    target = output / f"source{src.suffix.lower()}"
    if src.resolve() != target.resolve():
        shutil.copy2(src, target)
    return str(target)


TEXT_STEPS = [
    PipelineStep("split_spacy", "SpaCy split", "core.step3_1_spacy_split"),
    PipelineStep("split_meaning", "Semantic split", "core.step3_2_splitbymeaning"),
    PipelineStep("summarize", "Summary and terminology", "core.step4_1_summarize"),
    PipelineStep("translate", "Translate", "core.step4_2_translate_all"),
    PipelineStep("split_subtitle", "Subtitle split", "core.step5_splitforsub"),
    PipelineStep("timeline", "Timeline", "core.step6_generate_final_timeline"),
]

SUBTITLE_STEPS = [PipelineStep("merge_subtitle", "Burn subtitles", "core.step7_merge_sub_to_vid")]

DUBBING_STEPS = [
    PipelineStep("gen_audio_task", "Generate audio tasks", "core.step8_1_gen_audio_task"),
    PipelineStep("gen_dub_chunks", "Optimize dubbing chunks", "core.step8_2_gen_dub_chunks"),
    PipelineStep("extract_refer", "Extract reference audio", "core.step9_extract_refer_audio"),
    PipelineStep("gen_audio", "Generate TTS audio", "core.step10_gen_audio"),
    PipelineStep("merge_audio", "Merge dubbing audio", "core.step11_merge_full_audio"),
    PipelineStep("merge_video", "Merge final video", "core.step12_merge_dub_to_vid"),
]


def build_steps_for_input(input_value: str, *, dubbing: bool = True, subtitles: bool = True) -> list[PipelineStep]:
    suffix = Path(input_value).suffix.lower()
    steps: list[PipelineStep] = []
    if suffix == ".srt":
        from core.step2_prepare_from_srt import prepare_from_srt

        steps.append(PipelineStep("transcribe", "Prepare from SRT", action=lambda: prepare_from_srt(input_value)))
    elif input_value.startswith(("http://", "https://")):
        from core.config_utils import load_key
        from core.step1_ytdlp import download_video_ytdlp

        steps.append(
            PipelineStep(
                "download",
                "Download video",
                action=lambda: download_video_ytdlp(
                    input_value,
                    resolution=load_key("ytb_resolution", "1080"),
                    cutoff_time=load_key("smoke_test.cutoff_seconds", 60) if load_key("smoke_test.enabled", False) else None,
                ),
            )
        )
        steps.append(PipelineStep("transcribe", "Transcribe", "core.step2_whisperX"))
    else:
        steps.append(PipelineStep("import_video", "Import local video", action=lambda: prepare_local_video(input_value)))
        steps.append(PipelineStep("transcribe", "Transcribe", "core.step2_whisperX"))

    steps.extend(TEXT_STEPS)
    if subtitles:
        steps.extend(SUBTITLE_STEPS)
    if dubbing:
        steps.extend(DUBBING_STEPS)
    return steps
