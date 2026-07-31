# VideoLingo-plus Workflow Registry

**Date**: 2026-07-05
**Auditor**: Workflow Architect
**Status**: Draft; Phase 3 top workflow specs started

This registry records workflows discovered from the current codebase. Rows marked `Missing` exist in code but do not yet have a dedicated workflow spec.

Phase 0 baseline reference: `docs/reports/baseline-2026-07-05.md`.

## Workflows

| Workflow | Spec file | Status | Trigger | Primary actor | Last reviewed |
|---|---|---|---|---|---|
| Shared CLI full run | WORKFLOW-cli-full-run.md | Draft | `python -m core.cli run --input ...` | CLI runner | 2026-07-05 |
| Streamlit guided run | Missing | Missing | User actions in `st.py` | Streamlit UI | 2026-07-05 |
| Batch video processing | WORKFLOW-batch-run.md | Draft | `batch/utils/batch_processor.py` | Batch runner | 2026-07-05 |
| Input URL download | Missing | Missing | URL input to CLI/UI | `core.step1_ytdlp` | 2026-07-05 |
| Local video import | Missing | Missing | Local video input to CLI/UI | `core.pipeline.runner` | 2026-07-05 |
| SRT import | Missing | Missing | `.srt` input to CLI/UI | `core.step2_prepare_from_srt` | 2026-07-05 |
| Audio preparation and source ASR | Missing | Missing | `transcribe` step | `core.step2_whisperX` | 2026-07-05 |
| Semantic sentence split | Missing | Missing | `split_meaning` step | `core.step3_2_splitbymeaning` | 2026-07-05 |
| Summary and terminology extraction | Missing | Missing | `summarize` step | `core.step4_1_summarize` | 2026-07-05 |
| Translation | Missing | Missing | `translate` step | `core.step4_2_translate_all` | 2026-07-05 |
| Subtitle split and remerge | Missing | Missing | `split_subtitle` step | `core.step5_splitforsub` | 2026-07-05 |
| Final timeline alignment | Missing | Missing | `timeline` step | `core.step6_generate_final_timeline` | 2026-07-05 |
| Subtitle burn-in | Missing | Missing | `merge_subtitle` step | `core.step7_merge_sub_to_vid` | 2026-07-05 |
| Dubbing task generation | Missing | Missing | `gen_audio_task` step | `core.step8_1_gen_audio_task` | 2026-07-05 |
| Dubbing chunk optimization | Missing | Missing | `gen_dub_chunks` step | `core.step8_2_gen_dub_chunks` | 2026-07-05 |
| Reference audio extraction | Missing | Missing | `extract_refer` step | `core.step9_extract_refer_audio` | 2026-07-05 |
| TTS segment generation | Missing | Missing | `gen_audio` step | `core.step10_gen_audio` | 2026-07-05 |
| Full dubbing audio merge | Missing | Missing | `merge_audio` step | `core.step11_merge_full_audio` | 2026-07-05 |
| Final video mux | Missing | Missing | `merge_video` step | `core.step12_merge_dub_to_vid` | 2026-07-05 |
| Dubbing quality evaluation | WORKFLOW-dubbing-quality-eval.md | Draft | `python -m core.cli eval dubbing` | `core.dubbing_quality` | 2026-07-05 |
| ASR readback verification | Missing | Missing | `eval dubbing --readback` | `core.providers.asr_readback` | 2026-07-05 |
| Dubbing repair planning | WORKFLOW-dubbing-repair-apply.md | Draft | `python -m core.cli repair dubbing` | `core.providers.dubbing_repair` | 2026-07-05 |
| Dubbing repair apply and rebuild | WORKFLOW-dubbing-repair-apply.md | Draft | `repair dubbing --apply` | Repair provider + merge steps | 2026-07-05 |
| Timeline rescue | Missing | Missing | `python -m core.cli rescue timeline` | `core.providers.timeline_rescue` | 2026-07-05 |
| Translation provenance guard | WORKFLOW-translation-provenance-guard.md | Draft | CLI run/resume before translation-dependent steps | `core.translation_state` | 2026-07-05 |
| Translation artifact archive | Missing | Missing | `translation archive --apply` | `core.translation_state` | 2026-07-05 |
| Provider health/model inspection | Missing | Missing | `python -m core.cli models list` / `doctor` | Provider registries | 2026-07-05 |
| Provider model update | WORKFLOW-provider-model-update.md | Draft | Provider/model promotion request | Provider governance | 2026-07-05 |

## Components

| Component | File(s) | Workflows it participates in |
|---|---|---|
| Streamlit UI | `st.py`, `st_components/*` | Streamlit guided run, input import/download, subtitle/dubbing workflows |
| CLI | `core/cli.py` | Shared CLI full run, resume/status, eval, repair, rescue, translation guard, provider inspection |
| Pipeline runner | `core/pipeline/runner.py`, `core/pipeline/artifacts.py`, `core/step_checker.py` | Shared CLI full run, resume, artifact checkpointing |
| Profiles | `core/pipeline/profiles.py` | Shared CLI full run, quality profile application |
| Source input | `core/step1_ytdlp.py`, `core/step2_prepare_from_srt.py` | URL download, SRT import, local import |
| ASR | `core/step2_whisperX.py`, `core/all_whisper_methods/*`, `core/asr_schema.py` | Audio preparation and source ASR, ASR readback |
| LLM provider | `core/llm_provider.py`, `core/ask_gpt.py`, `core/prompts_storage.py` | Semantic split, summary, correction, translation, repair rewrite |
| Translation provenance | `core/translation_state.py` | Translation provenance guard, archive, adopt-current |
| Subtitle/timeline | `core/step5_splitforsub.py`, `core/step6_generate_final_timeline.py`, `core/providers/timeline_rescue.py` | Subtitle split, timeline alignment, timeline rescue |
| TTS registry | `core/all_tts_functions/tts_registry.py`, `core/all_tts_functions/tts_main.py` | TTS segment generation, repair apply |
| Local MLX TTS router | `core/providers/mlx_tts.py`, `core/all_tts_functions/mlx_router.py` | TTS segment generation, provider health/model inspection |
| Quality and repair | `core/dubbing_quality.py`, `core/providers/asr_readback.py`, `core/providers/dubbing_repair*.py`, `core/providers/quality_gate.py` | Eval, readback, repair planning/apply |
| Audio/video mux | `core/step11_merge_full_audio.py`, `core/step12_merge_dub_to_vid.py` | Audio merge, final video mux, repair rebuild |
| Batch runner | `batch/utils/*` | Batch video processing |
| Config | `config.yaml`, `.env.example`, `core/config_utils.py` | All model/provider/runtime workflows |

## User Journeys

### Creator Journeys

| What the creator experiences | Underlying workflow(s) | Entry point |
|---|---|---|
| Translate and dub a local video | Local video import -> ASR -> text/translation -> timeline -> subtitle burn -> TTS -> audio/video merge -> eval | `core.cli run --input local.mp4 --profile cinematic` or Streamlit |
| Translate and dub a URL video | URL download -> ASR -> text/translation -> timeline -> subtitle burn -> TTS -> audio/video merge -> eval | CLI/UI URL input |
| Build subtitles only | Input workflow -> ASR/text/translation -> timeline -> subtitle burn | `--subtitle-only` or UI subtitle path |
| Start from an SRT file | SRT import -> text/translation -> timeline -> optional subtitle/dubbing | `.srt` input |
| Repair failed dubbing rows | Eval/readback -> repair plan -> optional apply -> rebuild output -> re-eval | `core.cli repair dubbing` |
| Rescue a broken timeline | Timeline rescue report -> candidate tasks -> manual/apply decision | `core.cli rescue timeline` |

### Operator Journeys

| What the operator does | Underlying workflow(s) | Entry point |
|---|---|---|
| Inspect current run state | Artifact checkpointing + pipeline state | `core.cli status --include-dubbing-eval` |
| Inspect model/provider health | oMLX discovery + MLX TTS backend health | `core.cli models list`, `core.cli doctor` |
| Prevent stale model artifacts | Translation provenance guard | `core.cli translation status` |
| Archive stale translation artifacts | Translation archive plan/apply | `core.cli translation archive` |

### System-to-System Journeys

| What happens automatically | Underlying workflow(s) | Trigger |
|---|---|---|
| Pipeline resume skips completed artifacts | Artifact checkpointing | `run_pipeline(..., resume=True)` |
| Translation-dependent run blocks stale artifacts | Translation provenance guard | CLI run/resume before downstream steps |
| Repair clears stale ASR score columns | Dubbing repair apply | Mutating selected audio/task rows |
| Quality gate derives pass/warn/fail | Dubbing quality evaluation | Eval command, repair planning, output check |

## State Map

| Entity/state | Entered by | Exited by | Workflows that can trigger exit |
|---|---|---|---|
| `PipelineRun.created` | CLI run construction | `running`, `dry_run` | Shared CLI full run |
| `PipelineRun.dry_run` | `--dry-run` | Terminal | Shared CLI full run |
| `PipelineRun.running` | `run_pipeline` starts | `completed`, `failed` | Shared CLI full run |
| `PipelineRun.failed` | Any step exception | New run/resume after fix | Shared CLI full run, resume |
| `PipelineRun.completed` | All steps succeed | Terminal for run id | Shared CLI full run |
| Translation stage `current` | Stage manifest/provider matches current config | `stale` if provider/model/log mismatch | Translation provenance guard |
| Translation stage `warning` | Artifact exists without manifest or mixed logs | Adopt/archive/rerun | Translation status/adopt/archive |
| Translation stage `stale` | Provider/log/bad-file mismatch | Archive or rerun | Translation archive, run guard |
| Dubbing row `ok` | Eval thresholds pass | `warn` or `fail` if thresholds change/artifacts mutate | Eval/readback/repair |
| Dubbing row `warn` | Duration/drift/ASR quality threshold warning | `ok`, `manual_review`, `fail` | Repair apply, eval |
| Dubbing row `fail` | Missing/tiny/corrupt audio | `ok`, `manual_review` | Repair apply, regen/rebuild |
| Dubbing row `manual_review` | Auto repair exhausted or unsafe | Human edit/regenerate | Repair planning/apply |
| Provider `healthy` | Health check/root/model discovery succeeds | `unhealthy` | Doctor/models list |
| Provider `unhealthy` | Root missing, service down, key missing, model invalid | `healthy` after configuration fix | Doctor/models list |

## Discovery Audit Checklist

| Area | Status | Evidence |
|---|---|---|
| API route files | Not applicable | No web API routes discovered; app is Streamlit + CLI |
| UI entry points | Pass | `st.py`, `st_components/*` scanned |
| CLI entry points | Pass | `core/cli.py` scanned |
| Background workers/jobs | Partial | Batch runner exists; no queue/worker service spec |
| Scheduled jobs/cron | Not found | No scheduled job workflow found in scan |
| Event/message consumers | Not found | No event bus/message consumer found |
| Service orchestration | Partial | `Dockerfile` exists; no docker-compose/k8s workflow spec |
| Data migrations | Not applicable | No database migrations; state is file/artifact based |
| State enums/status | Pass | `PipelineRun.status`, translation stale/warning, eval ok/warn/fail found |
| Config/env | Pass | `config.yaml` and `.env.example` scanned; secrets not copied |

## Red Flags

| # | Finding | Severity | Notes |
|---|---|---|---|
| RF-1 | Many discovered workflows are still `Missing` as formal specs | High | Top workflow specs now exist for CLI full run, batch run, translation guard, dubbing eval, dubbing repair, and provider-model update |
| RF-2 | Artifact completion mostly checks existence/columns, not full input/model/profile fingerprints | High | Translation has provenance guard; ASR/timeline/TTS/video artifacts still need stronger run binding; schema drafted in `ARTIFACT-MANIFEST-SCHEMA.md` |
| RF-3 | Quality pass depends on current config threshold | High | Current `content_score_min=0.55`; cinematic profile code sets `0.88` when applied |
| RF-4 | Some model presets in config are stale or unverified | High | `gemini-3-pro-preview` is shut down per Gemini docs; current `DeepSeek-V3.2` route is provider-specific |
| RF-5 | Experimental MLX clone service may be offline | Medium | ZONOS2 requires its configured local HTTP service; health checks report it unavailable until started |
