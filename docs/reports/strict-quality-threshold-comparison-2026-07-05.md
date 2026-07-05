# Strict Quality Threshold Comparison

**Date**: 2026-07-05
**Input artifact**: `output/audio/tts_tasks.xlsx`
**Rows**: 102
**Score column**: `asr_content_score`
**Purpose**: Clarify what the current `102/102 ok` result means under different semantic ASR content thresholds.

## Summary

The current baseline passes under the active config threshold `content_score_min=0.55`. It does not provide enough evidence to claim pass under a stricter cinematic semantic threshold of `0.88`.

| Threshold | Rows below | Rows at or above | Interpretation |
|---:|---:|---:|---|
| 0.55 | 0 | 102 | pass under current config |
| 0.80 | 25 | 77 | quality review recommended |
| 0.88 | 47 | 55 | pending/fail risk under strict cinematic threshold |
| 0.90 | 54 | 48 | stricter candidate threshold, not current policy |

## Score Distribution

| Metric | Value |
|---|---:|
| Scored rows | 102 |
| Minimum score | 0.6078431372549019 |
| Average score | 0.8727833418188853 |
| Maximum score | 1.0 |

## Decision

Use this run as the Phase 0 baseline only with a precise label:

```text
pass under current config content_score_min=0.55
pending under strict cinematic semantic threshold content_score_min=0.88
```

## Required Follow-Up

1. Phase 1 should make the active quality threshold visible in every quality report.
2. Phase 2 benchmark reports must show results at both the current threshold and the strict cinematic threshold.
3. No model should be promoted based only on the current `102/102 ok` count unless the benchmark states which threshold produced it.

