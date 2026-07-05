# WORKFLOW: Provider Model Update

**Version**: 0.1
**Date**: 2026-07-05
**Author**: Workflow Architect
**Status**: Draft
**Implements**: Final optimization blueprint Phase 1-4 gate for model upgrades

## Overview

This workflow governs adding, verifying, benchmarking, and promoting an LLM, ASR, or TTS provider/model route. It prevents default model replacement without baseline evidence, provider lifecycle metadata, smoke checks, benchmark comparison, and rollback.

## Actors

| Actor | Role |
|---|---|
| Operator | Requests provider/model update |
| Provider governance registry | Stores lifecycle and adapter status |
| Benchmark runner | Produces comparison evidence |
| CLI | Runs `models verify`, `benchmark`, and smoke commands |
| Config maintainer | Applies default route change only after gate pass |

## Prerequisites

- Phase 0 baseline report exists.
- Provider route has metadata: provider, model, lifecycle, adapter implementation state, API key/service requirements.
- No secrets are stored in config or reports.
- Benchmark fixtures exist for the target route type.

## Trigger

Provider/model update request, for example:

```text
Promote DeepSeek V4 Pro for cinematic zh->vi translation.
```

## Workflow Tree

### STEP 1: Register Candidate Metadata

**Actor**: Config maintainer
**Action**: Add or update provider lifecycle metadata without changing defaults.
**Timeout**: 10 minutes
**Success**: Candidate appears in provider governance output.
**Failure**:
- `FAILURE(missing_metadata)`: reject update until required fields exist.
- `FAILURE(secret_in_config)`: stop and move secret to environment variable.

### STEP 2: Verify Provider Readiness

**Actor**: CLI
**Action**: Run provider lifecycle verification.
**Command**:

```bash
python -m core.cli models verify --fail-on-error
```

**Success**: No deprecated, unavailable, or unimplemented provider errors.
**Failure**:
- `FAILURE(deprecated)`: provider cannot be promoted.
- `FAILURE(unimplemented)`: implement adapter before benchmark.
- `FAILURE(missing_key)`: configure environment secret before smoke.

### STEP 3: Run Target Benchmark

**Actor**: Benchmark runner
**Action**: Run benchmark against baseline and candidate providers.
**Success**: JSON and Markdown comparison report written.
**Failure**:
- `FAILURE(missing_dataset)`: freeze fixture first.
- `FAILURE(invalid_dataset)`: fix fixture schema before any model call.
- `FAILURE(seed_schema_only)`: replace placeholder media/input paths before ASR, TTS-with-reference, or E2E promotion.
- `FAILURE(provider_error)`: candidate remains `candidate` or `experimental`.
- `FAILURE(format_error)`: candidate cannot be promoted for JSON/line-preserving tasks.

### STEP 4: Run End-to-End Smoke

**Actor**: CLI
**Action**: Run short E2E route on fixture or approved local input.
**Success**: Quality gate passes with effective thresholds recorded.
**Failure**:
- `FAILURE(quality_gate)`: do not promote; keep fallback.
- `FAILURE(stale_artifact)`: archive or regenerate artifacts with matching manifests.

### STEP 5: Promotion Decision

**Actor**: Config maintainer
**Action**: Compare candidate against baseline and write decision.
**Success**: Candidate can be promoted only if all gates pass.
**Failure**:
- `FAILURE(regression)`: reject candidate and document regression.
- `FAILURE(cost_unbounded)`: reject candidate until cost bound exists.

### STEP 6: Apply Default Route Change

**Actor**: Config maintainer
**Action**: Change default route only after approval.
**Success**: Default config updated with rollback route documented.
**Failure**:
- `FAILURE(no_rollback)`: stop; do not change default.

## Handoff Contracts

### Provider Governance -> Benchmark Runner

Payload:

```json
{
  "provider": "string",
  "model": "string",
  "kind": "llm|asr|tts",
  "status": "candidate|experimental|stable",
  "implemented": true
}
```

Success response:

```json
{
  "ok": true,
  "eligible_for_benchmark": true
}
```

Failure response:

```json
{
  "ok": false,
  "code": "UNAVAILABLE_OR_UNIMPLEMENTED",
  "retryable": false
}
```

## Test Cases

| Test | Expected behavior |
|---|---|
| Candidate provider missing API key | Verify reports warning/error, no promotion |
| Deprecated provider | Verify fails with `--fail-on-error` |
| Unimplemented TTS provider | Verify fails; UI hides normal selection |
| Missing benchmark dataset | Benchmark plan returns `missing_dataset` |
| Placeholder ASR/E2E media fixture | Benchmark plan returns `seed_schema_only` |
| Malformed benchmark fixture | Benchmark plan returns `invalid_dataset` |
| Candidate benchmark worse than baseline | Candidate rejected |
| Candidate passes benchmark but E2E fails | Candidate rejected |
| Candidate passes all gates | Default change allowed with rollback |

## Assumptions

| # | Assumption | Where verified | Risk if wrong |
|---|---|---|---|
| A1 | Baseline report exists before promotion | Phase 0 report | No comparator for quality/cost |
| A2 | Provider smoke can run without storing secrets | Environment variables | Secret leakage |
| A3 | Benchmark fixtures represent priority language pairs | Fixture inventory | Overfitting to weak samples |

## Open Questions

- What cost per finished minute is acceptable for premium cloud TTS?
- Should strict cinematic semantic threshold stay at `0.88` for every language pair?
