# Workflow: Dubbing Quality Evaluation

**Date**: 2026-07-05
**Status**: Draft
**Owner role**: Audio quality maintainer
**Entry point**: `python -m core.cli eval dubbing`

## Purpose

Score generated dubbing rows for timing, missing/tiny audio, natural speed, ASR content match, and reference leakage.

## Actors

| Actor | Responsibility |
|---|---|
| Dubbing quality module | Loads tasks, applies thresholds, writes JSON/XLSX reports |
| ASR readback verifier | Optionally writes content/leak scores before evaluation |
| Quality gate summarizer | Produces aggregate pass/warn/fail summary |

## Preconditions

- `output/audio/tts_tasks.xlsx` exists and has required task columns.
- Generated segment audio should exist under `output/audio/segs`.
- Optional ASR readback can be run with `--readback`.

## Flow

1. Load effective `DubbingQualityConfig` from config.
2. Optionally run ASR readback when CLI receives `--readback`.
3. Evaluate each task row against duration, drift, speed, missing audio, content score, and leak thresholds.
4. Write `output/audio/dubbing_eval.xlsx`.
5. Write `output/audio/dubbing_eval.json` with sanitized summary and rows.

## Observable States

| State | Evidence |
|---|---|
| `ok` row | Row has no failed/warn reasons |
| `warn` row | Row violates timing/content/leak/speed warning thresholds |
| `fail` row | Row has missing or silent/tiny audio |
| ASR scored | Summary has `asr_scored_rows > 0` |

## Failure Modes

| Failure | Handling |
|---|---|
| Missing task file | Pandas read fails; CLI returns exception |
| Missing segment audio | Row status becomes `fail` with `missing_audio` |
| Tiny/silent audio | Row status becomes `fail` with `silent_or_tiny_audio` |
| Readback unavailable | ASR fields remain unverified unless scores exist |

## Recovery

- Run `python -m core.cli repair dubbing` to plan repairs.
- Re-run `eval dubbing --readback --force` after repair or scoring logic changes.

## Tests

| Case | Current evidence |
|---|---|
| Status command can include eval without writing files | `tests/test_cli_workflow.py` |
| ASR readback cache/fingerprint behavior | `tests/test_asr_readback.py` |
| Quality text scoring helpers | `tests/test_provider_contracts.py` |

## Open Gaps

- Evaluation artifacts do not yet include a general sidecar manifest with quality config hash.
- Strict cinematic threshold comparison currently lives in a report, not a first-class eval mode.
