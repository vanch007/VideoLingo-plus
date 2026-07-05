# Workflow: Batch Run

**Date**: 2026-07-05
**Status**: Draft
**Owner role**: Batch operator / workflow maintainer
**Entry point**: `python -m batch.utils.batch_processor`

## Purpose

Process rows from `batch/tasks_setting.xlsx`, retry failed rows, and preserve per-task status while restoring language config after each row.

## Actors

| Actor | Responsibility |
|---|---|
| Batch processor | Iterates rows, updates language config, records task status |
| Settings check | Validates batch settings before processing |
| Video processor | Executes one video task |
| Operator | Maintains task sheet and reviews `batch/output` |

## Preconditions

- `batch/tasks_setting.xlsx` exists with expected columns.
- `check_settings()` passes.
- Each task row includes `Video File`, source/target language fields, `Dubbing`, and `Status`.

## Flow

1. Run settings check.
2. Load task sheet.
3. For each blank or error status row, record current source/target config.
4. Update source/target language for the task.
5. If retrying an error, restore previous task files from `batch/output/ERROR/<video-name>` when present.
6. Call `process_video(video_file, dubbing, is_retry)`.
7. Write `Done` or `Error: <step> - <message>` into the row status.
8. Restore original source/target config and save the task sheet.

## Observable States

| State | Evidence |
|---|---|
| Pending row | Empty `Status` or `Status` containing `Error` |
| Done row | `Status` equals `Done` |
| Error row | `Status` starts with `Error:` |
| Skipped row | Console prints skip message for non-error status |

## Failure Modes

| Failure | Handling |
|---|---|
| Settings check fails | Batch raises exception before processing |
| Error restore folder missing | Warning is printed; retry continues |
| Per-video exception | Row status records unhandled exception |
| Local resources contend across jobs | Pending: no batch resource lock is implemented |

## Recovery

- Fix the row, config, or input asset.
- Leave `Status` blank or containing `Error` to retry.
- Inspect `batch/output` and `batch/output/ERROR`.

## Tests

| Case | Current evidence |
|---|---|
| Batch processor workflow | missing evidence |
| Resource lock behavior | pending |

## Open Gaps

- No formal lock prevents concurrent local MLX/ASR/TTS contention.
- The batch flow still mutates global config between rows and relies on finally-block restoration.
