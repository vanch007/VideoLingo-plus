# Benchmark Fixture Inventory

**Date**: 2026-07-05
**Phase**: Phase 0 - Baseline Freeze

This directory is reserved for small benchmark fixtures used by future LLM, ASR, TTS, and end-to-end route comparisons.

## Current Inventory

| File | Status | Notes |
|---|---|---|
| `translation_zh_vi.jsonl` | seed fixture | Text-only LLM line-count and terminology fixture |
| `tts_lines.yaml` | seed fixture | Text-only TTS duration/readback fixture definition |
| `asr_samples.yaml` | seed schema only | Audio path is intentionally `null` until a tiny clip is approved |
| `e2e_smoke.yaml` | seed schema only | Input path is intentionally `null` until a tiny smoke input is approved |

The benchmark planner validates fixture schemas before returning `ready`. Files marked `seed schema only` are expected to return `seed_schema_only` and must not be used for provider promotion.

Phase 0 intentionally does not copy large runtime assets from `output/`, `redub_runs/`, `redub_assets/`, or `run_inputs/`.

## Fixture Requirements

Future fixture files should be small, deterministic, and safe to commit.

| Fixture type | Suggested format | Required fields |
|---|---|---|
| LLM translation | `translation_zh_vi.jsonl` | source lines, target language, terminology expectations, line-count expectation |
| ASR | `asr_samples.yaml` | audio path or fixture id, source language, expected transcript, expected time windows |
| TTS | `tts_lines.yaml` | text, language, target duration, optional reference audio fixture id |
| E2E smoke | `e2e_smoke.yaml` | input fixture id, source/target language, profile, expected artifact gates |

## Commit Policy

- Do not commit secrets.
- Do not commit large videos or generated output directories.
- Prefer short clips, short text fixtures, and tiny reference audio files.
- If a fixture cannot be committed because it is large or private, document its external path and checksum in a report instead.
