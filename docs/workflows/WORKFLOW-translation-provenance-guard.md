# Workflow: Translation Provenance Guard

**Date**: 2026-07-05
**Status**: Draft
**Owner role**: Backend maintainer
**Entry points**: `python -m core.cli translation status`, `archive`, `adopt-current`, and CLI run/resume guard

## Purpose

Prevent downstream dubbing, subtitle, and merge steps from silently reusing translation artifacts created by a different LLM provider/model.

## Actors

| Actor | Responsibility |
|---|---|
| Translation state module | Reads logs, artifacts, and `output/log/llm_artifacts_manifest.json` |
| CLI run/resume | Calls guard before translation-dependent steps |
| Operator | Archives stale artifacts or adopts trusted current artifacts |

## Preconditions

- Current LLM provider/model can be read from config.
- Translation logs and stage artifacts may or may not exist.
- No secrets are copied into manifests.

## Flow

1. Build current provider fingerprint from configured provider, base URL, model, and capability fields.
2. Inspect translation stage artifacts and GPT logs.
3. Compare stage manifest provider fingerprint against current provider fingerprint.
4. Compare log model evidence against current configured model.
5. Return `ok=true` when no stale stage exists.
6. Return blocked guard payload when stale stages would feed downstream steps.
7. Optionally archive stale derived artifacts when explicit archive/apply behavior is requested.

## Observable States

| State | Evidence |
|---|---|
| Current | `stale: false`, no blocking reasons |
| Warning | Artifact exists without manifest, or logs contain other models |
| Stale | Manifest/log provider mismatch or bad log files |
| Archived | `output/history/translation_<timestamp>/archive_manifest.json` |

## Failure Modes

| Failure | Handling |
|---|---|
| Manifest missing but artifacts exist | Warning; next action recommends rerun or adopt |
| Manifest provider mismatch | Blocking stale state |
| Log model mismatch | Blocking stale state |
| Bad log JSON | Blocking stale state |

## Recovery

- `python -m core.cli translation status`
- `python -m core.cli translation adopt-current --apply` only when logs match current config and artifacts are trusted.
- `python -m core.cli translation archive --apply` to move stale artifacts into history.

## Tests

| Case | Current evidence |
|---|---|
| Artifact without manifest warns | `tests/test_translation_state.py` |
| Current stage manifest matches provider | `tests/test_translation_state.py` |
| Adopt current writes manifest when logs match | `tests/test_translation_state.py` |

## Open Gaps

- Translation manifest schema is not yet unified with general `core/pipeline/artifact_manifest.py`.
- General resume checks still depend on step-specific checkpoint validators.
