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
    smoke_seconds: int | None = None
    status: str = "created"
    quality_status: str | None = None
    quality_gate_passed: bool | None = None
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
            "smoke_seconds": self.smoke_seconds,
            "steps": self.steps,
            "status": self.status,
            "quality_status": self.quality_status,
            "quality_gate_passed": self.quality_gate_passed,
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


STEP_CODE_MAP = {
    "import_video": ["core/step1_ytdlp.py"],
    "transcribe": ["core/step2_whisperX.py", "core/all_whisper_methods/stable_ts_local.py", "core/all_whisper_methods/audio_preprocess.py", "core/all_whisper_methods/demucs_vl.py", "core/providers/speaker_diarization.py", "core/providers/source_coverage.py"],
    "split_spacy": ["core/step3_1_spacy_split.py"],
    "split_meaning": ["core/step3_2_splitbymeaning.py"],
    "summarize": ["core/step4_1_summarize.py", "core/translation_context.py"],
    "translate": ["core/step4_2_translate_all.py", "core/translation_review.py", "core/translation_context.py"],
    "split_subtitle": ["core/step5_splitforsub.py"],
    "timeline": ["core/step6_generate_final_timeline.py"],
    "merge_subtitle": ["core/step7_merge_sub_to_vid.py"],
    "gen_audio_task": ["core/step8_1_gen_audio_task.py"],
    "gen_dub_chunks": ["core/step8_2_gen_dub_chunks.py"],
    "extract_refer": ["core/step9_extract_refer_audio.py", "core/tts_reference_plan.py"],
    "gen_audio": ["core/step10_gen_audio.py", "core/all_tts_functions/mlx_router.py", "core/dubbing_rewrite.py", "core/dubbing_content_gate.py", "core/tts_reference_plan.py", "core/audio_speed.py"],
    "merge_audio": ["core/step11_merge_full_audio.py", "core/dubbing_quality.py", "core/audio_speed.py"],
    "merge_video": ["core/step12_merge_dub_to_vid.py", "core/dubbing_quality.py"],
    "dubbing_eval": ["core/dubbing_quality.py", "core/providers/asr_readback.py", "core/providers/quality.py"],
}

def _record_step_manifest(step_key: str, run: PipelineRun) -> None:
    try:
        from core.step_checker import STEP_CHECKPOINTS
        from core.pipeline.artifact_manifest import build_artifact_manifest, write_artifact_manifest, manifest_path_for, read_artifact_manifest
        from core.config_utils import load_key
        checkpoint = STEP_CHECKPOINTS.get(step_key)
        if checkpoint and os.path.exists(checkpoint):
            sidecar = manifest_path_for(checkpoint)
            producing_steps = [step_key]
            if sidecar.exists():
                try:
                    prev = read_artifact_manifest(sidecar)
                    prev_steps = prev.get("producing_steps", [prev.get("step")])
                    producing_steps = list(dict.fromkeys(prev_steps + [step_key]))
                except Exception:
                    pass
            run_dict = run.to_dict()
            run_dict["producing_steps"] = producing_steps
            code_paths = STEP_CODE_MAP.get(step_key, [])
            models_info = {}
            if step_key == "transcribe":
                models_info = {"whisper": load_key("whisper.model", "large-v3-turbo"), "diarization": load_key("speaker_diarization.backend", "nemotron-mlx")}
            elif step_key in ("summarize", "translate", "split_meaning", "split_subtitle"):
                models_info = {"llm": load_key("api.model", "zai-org/GLM-4.5-Air")}
            elif step_key in ("gen_audio", "gen_dub_chunks", "gen_audio_task"):
                models_info = {"tts": load_key("tts_method", "mlx_indextts2")}
            manifest = build_artifact_manifest(
                artifact=checkpoint,
                step=step_key,
                run_context=run_dict,
                code_paths=code_paths,
                models=models_info,
            )
            write_artifact_manifest(manifest)
            if step_key == "gen_audio" and os.path.exists("output/audio/tts_tasks.xlsx"):
                tasks_sidecar = manifest_path_for("output/audio/tts_tasks.xlsx")
                tasks_steps = ["gen_audio_task", "gen_dub_chunks", "gen_audio"]
                run_dict_tasks = run.to_dict()
                run_dict_tasks["producing_steps"] = tasks_steps
                tasks_manifest = build_artifact_manifest(
                    artifact="output/audio/tts_tasks.xlsx",
                    step="gen_dub_chunks",
                    run_context=run_dict_tasks,
                    code_paths=STEP_CODE_MAP.get("gen_audio", []),
                    models={"tts": load_key("tts_method", "mlx_indextts2")},
                )
                write_artifact_manifest(tasks_manifest)
    except Exception as e:
        import sys
        print(f"Warning: failed to record manifest for {step_key}: {e}", file=sys.stderr)


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
            # Retain original provenance for skipped steps; do not restamp
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
            _record_step_manifest(step.key, run)
            save_state(run)
        except Exception:
            run.status = "failed"
            run.quality_status = "fail"
            run.failed_step = step.key
            save_state(run)
            raise

    eval_file = Path("output/audio/dubbing_eval.json")
    if eval_file.is_file():
        try:
            eval_data = json.loads(eval_file.read_text(encoding="utf-8"))
            gate = eval_data.get("summary", {}).get("quality_gate", {})
            passed = bool(gate.get("passed", False))
            run.quality_gate_passed = passed
            perceptual_status = eval_data.get("summary", {}).get("perceptual_quality", {}).get("status")
            perceptual_pending = (perceptual_status != "accepted")
            run.quality_status = ("pending" if perceptual_pending else "pass") if passed else "fail"
            run.status = "accepted" if passed and not perceptual_pending else "rendered"
        except Exception:
            run.quality_status = "pending"
            run.status = "rendered"
    else:
        run.quality_status = "not_applicable" if run.subtitle_only else "pending"
        run.status = "accepted" if run.subtitle_only else "completed"
    _record_step_manifest("dubbing_eval", run)
    save_state(run)
    return run


def prepare_local_video(input_path: str, *, smoke_seconds: int | None = None) -> str:
    src = Path(input_path).expanduser()
    if not src.exists():
        raise FileNotFoundError(input_path)
    output = Path("output")
    output.mkdir(exist_ok=True)
    target = output / f"source{src.suffix.lower()}"
    if src.resolve() == target.resolve():
        return str(target)
    if smoke_seconds and smoke_seconds > 0:
        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-i",
            str(src),
            "-t",
            str(smoke_seconds),
            "-map",
            "0",
            "-c",
            "copy",
            str(target),
        ]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError:
            fallback_cmd = [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "warning",
                "-i",
                str(src),
                "-t",
                str(smoke_seconds),
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-c:a",
                "aac",
                str(target),
            ]
            subprocess.run(fallback_cmd, check=True)
    else:
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


def build_steps_for_input(
    input_value: str,
    *,
    dubbing: bool = True,
    subtitles: bool = True,
    smoke_seconds: int | None = None,
) -> list[PipelineStep]:
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
        steps.append(
            PipelineStep(
                "import_video",
                "Import local video",
                action=lambda: prepare_local_video(input_value, smoke_seconds=smoke_seconds),
            )
        )
        steps.append(PipelineStep("transcribe", "Transcribe", "core.step2_whisperX"))

    steps.extend(TEXT_STEPS)
    if subtitles:
        steps.extend(SUBTITLE_STEPS)
    if dubbing:
        steps.extend(DUBBING_STEPS)
    return steps
