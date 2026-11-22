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
    stop_after: str | None = None
    status: str = "created"
    quality_status: str | None = None
    quality_gate_passed: bool | None = None
    started_at: float = field(default_factory=time.time)
    completed_steps: list[str] = field(default_factory=list)
    failed_step: str | None = None
    current_step: str | None = None
    error: str | None = None
    events: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "profile": self.profile,
            "policy_version": "fast-v2" if self.profile == "fast" else "cinematic-v1",
            "source": self.source,
            "target": self.target,
            "input": self.input,
            "subtitle_only": self.subtitle_only,
            "no_subtitles": self.no_subtitles,
            "llm": self.llm,
            "tts": self.tts,
            "smoke_seconds": self.smoke_seconds,
            "stop_after": self.stop_after,
            "steps": self.steps,
            "status": self.status,
            "quality_status": self.quality_status,
            "quality_gate_passed": self.quality_gate_passed,
            "started_at": self.started_at,
            "completed_steps": self.completed_steps,
            "failed_step": self.failed_step,
            "current_step": self.current_step,
            "error": self.error,
            "events": self.events,
            "pid": os.getpid(),
            "updated_at": time.time(),
        }


def save_state(run: PipelineRun) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATE_FILE.with_suffix('.tmp')
    temporary.write_text(json.dumps(run.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(STATE_FILE)


def load_state() -> dict:
    if not STATE_FILE.exists():
        raise FileNotFoundError(f"No pipeline state found at {STATE_FILE}")
    return json.loads(STATE_FILE.read_text(encoding="utf-8"))


def load_state_or_none() -> dict | None:
    if not STATE_FILE.exists():
        return None
    return load_state()


STEP_CODE_MAP = {
    "import_video": ["core/step1_ytdlp.py", "core/pipeline/runner.py"],
    "video_context": ["core/video_context.py", "scripts/video_context_worker.py"],
    "transcribe": ["core/step2_whisperX.py", "core/asr_schema.py", "core/source_accuracy.py", "core/all_whisper_methods/stable_ts_local.py", "core/all_whisper_methods/qwen3_aligned.py", "core/all_whisper_methods/local_aligned.py", "scripts/recognize_source_audio.py", "scripts/align_source_audio.py", "core/all_whisper_methods/audio_preprocess.py", "core/all_whisper_methods/demucs_vl.py", "core/providers/speaker_diarization.py", "core/providers/source_coverage.py"],
    "split_spacy": ["core/step3_1_spacy_split.py", "core/spacy_utils/split_by_mark.py", "core/dialogue_segmentation.py"],
    "split_meaning": ["core/step3_2_splitbymeaning.py"],
    "summarize": ["core/step4_1_summarize.py", "core/translation_context.py", "core/source_verification.py", "core/dialogue_roles.py", "core/translation_invariants.py"],
    "translate": ["core/step4_2_translate_all.py", "core/fast_text.py", "core/translate_once.py", "core/local_translation.py", "core/translation_review.py", "core/translation_context.py", "core/translation_invariants.py", "core/dialogue_roles.py"],
    "split_subtitle": ["core/step5_splitforsub.py"],
    "timeline": ["core/step6_generate_final_timeline.py", "core/subtitle_layout.py", "core/dialogue_segmentation.py"],
    "merge_subtitle": ["core/step7_merge_sub_to_vid.py", "core/subtitle_quality.py"],
    "gen_audio_task": ["core/step8_1_gen_audio_task.py"],
    "gen_dub_chunks": ["core/step8_2_gen_dub_chunks.py"],
    "extract_refer": ["core/step9_extract_refer_audio.py", "core/tts_reference_plan.py"],
    "gen_audio": ["core/step10_gen_audio.py", "core/all_tts_functions/mlx_router.py", "core/dubbing_rewrite.py", "core/dubbing_content_gate.py", "core/tts_reference_plan.py", "core/audio_speed.py"],
    "merge_audio": ["core/step11_merge_full_audio.py", "core/dubbing_quality.py", "core/audio_speed.py"],
    "merge_video": ["core/step12_merge_dub_to_vid.py", "core/dubbing_quality.py"],
    "dubbing_eval": ["core/dubbing_quality.py", "core/providers/asr_readback.py", "core/providers/quality.py"],
}

# These shared contracts affect both translation acceptance and every consumer.
STEP_CODE_MAP["summarize"] += ["core/utterance_plan.py", "core/source_pause_budget.py"]
for _step in ("translate", "split_subtitle", "timeline", "gen_audio_task", "gen_dub_chunks",
              "extract_refer", "gen_audio", "merge_audio", "merge_video"):
    STEP_CODE_MAP[_step] += ["core/utterance_plan.py", "core/atomic_dubbing.py", "core/speech_budget.py",
                            "core/homura_translation.py", "core/model_artifacts.py", "core/duration_candidate_audit.py",
                            "core/spoken_length.py", "core/tts_pronunciation.py"]
for _step in ('split_subtitle', 'timeline', 'merge_subtitle', 'merge_audio', 'merge_video'):
    STEP_CODE_MAP[_step] += ['core/plan_subtitles.py', 'core/bilingual_alignment.py', 'scripts/align_dubbing_units.py']
for _step in ("gen_audio", "merge_audio", "merge_video"):
    STEP_CODE_MAP[_step] += ["core/fast_dubbing.py", "core/fast_subtitles.py", "core/fast_video_merge.py", "core/fast_audio_cache.py", "core/fast_timing.py"]
STEP_CODE_MAP["dubbing_eval"] += ["core/atomic_dubbing.py", "core/utterance_plan.py"]

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
                if load_key("whisper.runtime") == "local-aligned":
                    models_info = {"recognizer": load_key("source_asr.recognizer"), "source_model": load_key("source_asr.model"),
                                   "aligner": load_key("source_asr.aligner"), "timing_source": "forced_alignment_80ms",
                                   "diarization": load_key("speaker_diarization.backend", "nemotron-mlx")}
            elif step_key in ("summarize", "translate", "split_meaning", "split_subtitle"):
                from core.llm_provider import get_llm_provider_config
                provider = get_llm_provider_config()
                models_info = {"llm": provider.model, "llm_provider": provider.name}
                code_paths = list(code_paths) + ["core/ask_gpt.py", "core/llm_provider.py", "core/prompts_storage.py", "core/translation_language.py",
                                               "core/local_mlx_chat.py","scripts/local_mlx_chat_worker.py"]
            elif step_key in ("gen_audio", "gen_dub_chunks", "gen_audio_task"):
                models_info = {"tts": load_key("tts_method", "mlx_indextts2")}
            from core.pipeline.artifact_manifest import hash_path
            from core.config_utils import CONFIG_PATH
            manifest = build_artifact_manifest(
                artifact=checkpoint,
                step=step_key,
                run_context=run_dict,
                code_paths=code_paths,
                models=models_info,
                quality={"config_path": str(Path(CONFIG_PATH).resolve()), "config_hash": hash_path(CONFIG_PATH)},
            )
            dependencies = {
                "transcribe": ["output/source.mp4", "output/log/video_context.json"],
                "split_spacy": ["output/log/cleaned_chunks.xlsx"],
                "split_meaning": ["output/log/sentence_splitbynlp.txt"],
                "summarize": ["output/log/sentence_splitbymeaning.txt", "output/log/cleaned_chunks.xlsx", "output/log/lexical_boundary_review.json", "output/log/video_context.json"],
                "translate": ["output/log/terminology.json", "output/log/sentence_splitbymeaning.txt", "output/log/source_verification.json",
                              "output/log/translation_speech_preflight.json", "output/log/translation_duration_candidate_provenance.json"],
                "split_subtitle": ["output/log/translation_results.xlsx", "output/log/translation_review.json", "output/log/translation_fact_normalizations.json"],
                "timeline": ["output/log/translation_results_for_subtitles.xlsx", "output/log/translation_results_remerged.xlsx"],
                "merge_subtitle": ["output/source.mp4", "output/trans.srt", "output/src.srt", "output/log/subtitle_quality.json", "output/log/video_context.json"],
            }.get(step_key, [])
            if step_key in ('split_subtitle', 'timeline', 'merge_subtitle', 'merge_audio', 'merge_video'):
                dependencies.append('output/log/bilingual_alignment.json')
                if step_key != 'split_subtitle':
                    dependencies.append('output/log/dub_subtitle_layout.json' if step_key in ('merge_audio', 'merge_video')
                                        else 'output/log/subtitle_layout.json')
            if step_key == "translate" and Path("output/log/translation_speech_preflight.json").is_file():
                speech = json.loads(Path("output/log/translation_speech_preflight.json").read_text(encoding="utf-8"))
                dependencies.extend(take["path"] for take in speech.get("audio", []))
                if Path("output/log/duration_candidate_semantic_audit.json").is_file():
                    dependencies.append("output/log/duration_candidate_semantic_audit.json")
            manifest["dependencies"] = {path: hash_path(path) for path in dependencies if Path(path).exists()}
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
        raise RuntimeError(f"Failed to record manifest for {step_key}: {e}") from e


def _run_module(module: str) -> None:
    subprocess.run([sys.executable, "-m", module], check=True)


def _recover_code_stale_text_checkpoint(key: str, checkpoint: str, run: PipelineRun) -> bool:
    """Replay a code-invalid text producer while preserving its historical output.

    Code-only invalidation is recoverable; changed source/config/output or
    missing provenance is not. Media and measured speech caches are untouched.
    The producer must really run before it can receive a current manifest.
    """
    from core.pipeline.artifact_manifest import inspect_artifact_manifest, manifest_path_for, hash_path
    if key not in {'split_meaning', 'summarize', 'translate', 'split_subtitle'}:
        return False
    audit = inspect_artifact_manifest(checkpoint, expected_step=key)
    if not audit.reasons or any(not reason.startswith('code_content_mismatch:') for reason in audit.reasons):
        return False
    artifact = Path(checkpoint)
    sidecar = manifest_path_for(artifact)
    history = Path('output/history/code_stale_checkpoints') / run.run_id / key
    history.mkdir(parents=True, exist_ok=False)
    record = {'run_id': run.run_id, 'step': key, 'reasons': audit.reasons,
              'artifact': checkpoint, 'sha256': hash_path(artifact),
              'history': str(history), 'action': 'archive_then_execute_producer'}
    shutil.move(str(artifact), history / artifact.name)
    shutil.move(str(sidecar), history / sidecar.name)
    (history / 'recovery.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    return True


def run_pipeline(run: PipelineRun, steps: list[PipelineStep], *, resume: bool = True, dry_run: bool = False) -> PipelineRun:
    step_keys = [step.key for step in steps]
    todo = pending_steps(step_keys) if resume else step_keys
    run.steps = step_keys

    if dry_run:
        run.status = "dry_run"
        run.completed_steps = [key for key in step_keys if key not in todo]
        return run

    # Modules have legacy file-exists shortcuts. Never restamp a stale shortcut
    # as a freshly executed step; require an isolated run or explicit recovery.
    from core.step_checker import STEP_CHECKPOINTS
    for key in todo:
        checkpoint = STEP_CHECKPOINTS.get(key)
        if key in {"transcribe", "split_spacy", "split_meaning", "summarize", "translate", "split_subtitle"} and checkpoint and Path(checkpoint).exists():
            if resume and _recover_code_stale_text_checkpoint(key, checkpoint, run):
                continue
            # Resuming: the partial artifact of the step the previous run was interrupted in (or that
            # failed) may be cleared so it reruns cleanly. Any other existing checkpoint without a valid
            # manifest is unproven (another input, configuration or code): it is never reused or restamped.
            if resume:
                previous = load_state_or_none() or {}
                interrupted = (previous.get('failed_step') == key
                               or (previous.get('status') in {'failed', 'interrupted', 'paused'}
                                   and key not in (previous.get('completed_steps') or [])
                                   and key in (previous.get('steps') or [])))
                if interrupted:
                    try:
                        from core.pipeline.artifact_manifest import manifest_path_for
                        c_path = Path(checkpoint)
                        sidecar = manifest_path_for(c_path)
                        if c_path.is_file():
                            c_path.unlink()
                        if sidecar.exists():
                            sidecar.unlink()
                        continue
                    except Exception:
                        pass
            run.status = 'failed'
            run.quality_status = 'fail'
            run.failed_step = key
            run.error = f"Stale {key} checkpoint: {checkpoint}. Use a fresh run directory; no existing artifacts were changed."
            save_state(run)
            raise RuntimeError(run.error)

    try:
        from core.pipeline.checkpoint import consume_pause
        reviewed = consume_pause()
    except ValueError as exc:
        run.status = "paused"
        run.error = str(exc)
        save_state(run)
        raise
    if reviewed:
        run.events.append({'step': 'translate', 'event': 'manual_translation_review', 'time': time.time(), **reviewed})
        if reviewed["edited_rows"]:
            _record_step_manifest("translate", run)  # the edited file is the artifact later stages use
    run.error = None
    run.status = "running"
    save_state(run)

    for step in steps:
        if step.key not in todo:
            run.completed_steps.append(step.key)
            # Retain original provenance for skipped steps; do not restamp
            save_state(run)
            continue
        try:
            run.current_step = step.key
            run.events.append({'step': step.key, 'event': 'started', 'time': time.time()})
            save_state(run)
            if step.action:
                step.action()
            elif step.module:
                _run_module(step.module)
            else:
                raise ValueError(f"Pipeline step has no action or module: {step.key}")
            _record_step_manifest(step.key, run)
            run.completed_steps.append(step.key)
            run.events.append({'step': step.key, 'event': 'completed', 'time': time.time()})
            save_state(run)
            from core.pipeline.checkpoint import pause_requested, write_pause
            if pause_requested(step.key, steps.index(step) + 1 < len(steps)):
                run.error = write_pause()
                run.status = "paused"
                run.current_step = None
                run.quality_status = "not_evaluated"
                run.quality_gate_passed = False
                save_state(run)
                return run
        except KeyboardInterrupt:
            run.status = 'interrupted'
            run.quality_status = 'not_evaluated'
            run.quality_gate_passed = False
            run.failed_step = step.key
            run.error = 'KeyboardInterrupt: execution stopped before this step completed'
            run.events.append({'step': step.key, 'event': 'interrupted', 'time': time.time(), 'error': run.error})
            save_state(run)
            raise
        except Exception as exc:
            run.status = "failed"
            run.quality_status = "fail"
            run.failed_step = step.key
            run.error = f'{type(exc).__name__}: {exc}'
            run.events.append({'step': step.key, 'event': 'failed', 'time': time.time(), 'error': run.error})
            save_state(run)
            raise

    run.current_step = None
    if run.stop_after:
        run.status = "stopped_after_step"
        run.quality_status = "not_evaluated"
        run.quality_gate_passed = False
        save_state(run)
        return run

    eval_file = Path("output/audio/dubbing_eval.json")
    if run.profile == "fast" and not run.subtitle_only:
        from core.fast_video_merge import validate_fast_delivery
        try:
            report = validate_fast_delivery()
            run.quality_status = "fast_basic_pass"
            run.quality_gate_passed = None  # strict quality gate was not run
            run.status = "completed"
        except Exception as exc:
            run.quality_status = "fail"
            run.quality_gate_passed = False
            run.status = "failed"
            run.error = str(exc)
    elif run.subtitle_only:
        from core.subtitle_quality import subtitle_quality_report
        quality = subtitle_quality_report()
        run.quality_gate_passed = quality["automatic_gate_passed"]
        run.quality_status = quality["status"]
        run.status = "rendered"
        Path("output/log/subtitle_quality.json").write_text(json.dumps(quality, ensure_ascii=False, indent=2))
    elif eval_file.is_file():
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
    from core.step1_ytdlp import ensure_canonical_source_video
    src = Path(input_path).expanduser()
    if not src.exists():
        raise FileNotFoundError(input_path)
    output = Path("output")
    output.mkdir(exist_ok=True)
    target = output / f"source{src.suffix.lower()}"
    if src.resolve() == target.resolve():
        return ensure_canonical_source_video(str(output))
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
    return ensure_canonical_source_video(str(output))


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
    profile: str | None = None,
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

    if suffix != ".srt":
        from core.video_context import enabled as video_context_enabled
        if video_context_enabled():
            steps.insert(len(steps) - 1, PipelineStep("video_context", "Visual context (screen terms, hard subtitles, people)",
                                                      "core.video_context"))
    steps.extend(TEXT_STEPS)
    if subtitles and not (profile == "fast" and dubbing):
        steps.extend(SUBTITLE_STEPS)
    if dubbing:
        steps.extend(step for step in DUBBING_STEPS if profile != "fast" or step.key not in {"gen_dub_chunks", "merge_audio"})
    return steps
