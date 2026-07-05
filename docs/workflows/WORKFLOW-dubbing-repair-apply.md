# Workflow: Dubbing Repair Apply

**Date**: 2026-07-05
**Status**: Draft
**Owner role**: Audio repair maintainer
**Entry point**: `python -m core.cli repair dubbing`

## Purpose

Plan and optionally apply targeted repairs for failed or warned dubbing rows while requiring explicit `--apply` before mutating task/audio artifacts.

## Actors

| Actor | Responsibility |
|---|---|
| Repair planner | Sorts non-ok rows by repair priority and action |
| Repair applier | Regenerates, rewrites, speed-fits, slow-fits, or marks manual review |
| Merge steps | Rebuild `output/dub.mp3` and `output/AI配音.mp4` after applied repairs unless disabled |
| ASR readback verifier | Optional pre-repair readback |

## Preconditions

- `output/audio/tts_tasks.xlsx` exists.
- `output/audio/dubbing_eval.*` can be generated from current tasks.
- Mutating repairs require explicit `--apply`.

## Flow

1. Optionally run ASR readback.
2. Build repair plan from current eval results, filters, and backend fallback.
3. Write `output/audio/dubbing_repair_plan.json`.
4. Without `--apply`, return dry-run payload.
5. With `--apply`, mutate selected task rows/audio according to plan.
6. Clear stale ASR score columns for mutated rows.
7. Re-evaluate dubbing and optionally rebuild merged audio/video.
8. For batched apply, append batch records to repair history.

## Observable States

| State | Evidence |
|---|---|
| Planned only | CLI payload has `apply.dry_run: true` |
| Applied | Apply summary has `applied > 0` |
| Skipped/manual | Apply summary has `skipped > 0` or task row `manual_review` |
| Rebuilt | CLI payload has `rebuilt_output: true` |

## Failure Modes

| Failure | Handling |
|---|---|
| Missing existing audio for speed fit | Repair action raises file error |
| Rewrite limit reached | Row marked manual review and skipped |
| Wrong/unchanged rewrite | Row becomes manual review or skipped |
| Merge rebuild disabled | Output video/audio may remain stale until manually rebuilt |

## Recovery

- Re-run repair with narrower `--reasons` or `--actions`.
- Use `--batches N` for repeated small repair passes.
- Re-run `eval dubbing --readback --force` after repair when content scores matter.

## Tests

| Case | Current evidence |
|---|---|
| Repair planning/apply behavior | `tests/test_dubbing_repair.py` |
| Tests avoid deleting real output audio | `.ai_memory.md` notes and repair tests |

## Open Gaps

- Applied repair artifacts do not yet write general sidecar manifests.
- Batch repair and CLI run do not share a resource lock for local model contention.
