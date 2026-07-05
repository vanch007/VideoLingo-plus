# Workflow: CLI Full Run

**Date**: 2026-07-05
**Status**: Draft
**Owner role**: Workflow architect / backend maintainer
**Entry point**: `python -m core.cli run --input ...`

## Purpose

Run the shared VideoLingo pipeline from URL, local video, or SRT input while recording run state and guarding stale translation artifacts.

## Actors

| Actor | Responsibility |
|---|---|
| CLI | Parses run options and applies profile/config overrides |
| Pipeline runner | Builds step list, resumes completed steps, saves `output/pipeline_state.json` |
| Step modules | Execute ASR, text, translation, timeline, subtitle, TTS, and merge work |
| Translation guard | Blocks downstream reuse when translation artifacts do not match current LLM config |

## Preconditions

- Environment passes required runtime checks for the selected route.
- `config.yaml` has non-secret defaults; provider secrets come from environment variables.
- Input is a URL, existing local media file, or `.srt`.
- Existing dirty runtime assets are intentionally preserved unless an explicit archive/cleanup command is used.

## Flow

1. Build step list with `core.pipeline.runner.build_steps_for_input`.
2. Apply profile and CLI config overrides unless `--dry-run` is set.
3. Run `guard_translation_artifacts_for_steps` for translation-dependent steps.
4. Compute `pending_steps` from current checkpoint validators when resume is enabled.
5. Save `output/pipeline_state.json` before execution and after each step.
6. Execute action-backed steps or `python -m <module>` step modules.
7. Mark run `completed` when all steps finish, or `failed` with `failed_step` on exception.

## Observable States

| State | Evidence |
|---|---|
| `dry_run` | CLI JSON contains `planned_config` and `pending_steps`; no config/state write |
| `running` | `output/pipeline_state.json` has `status: running` |
| `failed` | `output/pipeline_state.json` has `failed_step` |
| `completed` | `output/pipeline_state.json` has `status: completed` |

## Failure Modes

| Failure | Handling |
|---|---|
| Local input missing | `prepare_local_video` raises `FileNotFoundError` |
| Translation artifacts stale | CLI returns blocked translation guard payload unless auto-archive is explicit |
| Step module fails | Runner records `failed_step` and re-raises |
| Resume checkpoint is stale by content | Pending: general sidecar manifest enforcement is not wired yet |

## Recovery

- Re-run the same command after fixing the failed step.
- Use `--no-resume` to force all selected steps.
- Use `translation archive --apply` or `--auto-archive-stale-translation` only when moving stale translation artifacts is intended.

## Tests

| Case | Current evidence |
|---|---|
| Dry run does not mutate config/state | `tests/test_cli_workflow.py` |
| Resume reuses stored input/options | `tests/test_cli_workflow.py` |
| Smoke seconds passed to local video import | `tests/test_cli_workflow.py` |

## Open Gaps

- General artifact sidecar manifests are helper-only; resume enforcement is pending.
- Batch resource locking is not applied to CLI runs yet.
