# Phase 4 Model Upgrade Gates

**Date**: 2026-07-05
**Status**: Draft gate report

## Decision

Phase 4 model upgrades are not eligible to change defaults yet. The project now has a baseline and initial provider governance/benchmark scaffolding, but no candidate has produced benchmark and E2E smoke evidence that beats the baseline.

## Required Gates

| Gate | Required evidence | Current status |
|---|---|---|
| Baseline report | `docs/reports/baseline-2026-07-05.md` | pass |
| Strict threshold report | `docs/reports/strict-quality-threshold-comparison-2026-07-05.md` | pass |
| Provider lifecycle verification | `python -m core.cli models verify` | implemented; current report expected to flag unavailable/deprecated providers |
| Benchmark fixture exists | Target-specific fixture file plus schema validation | partial: LLM/TTS seed fixtures validate; ASR/E2E remain `seed_schema_only` |
| LLM/ASR/TTS benchmark report | JSON/Markdown comparison | pending |
| E2E smoke report | 60s route with quality gate | pending |
| Artifact provenance protection | Sidecar manifests | helper and CLI audit implemented; resume enforcement pending |
| Rollback route | Previous default route documented | pending per candidate |

## Default Route Policy

Current defaults remain unchanged:

| Route | Current default |
|---|---|
| LLM | `openai_compatible` using `api.base_url` / `api.model` |
| LLM model | `deepseek-ai/DeepSeek-V3.2` via current config |
| ASR | `stable-ts` with `large-v3-turbo` and MLX |
| TTS | `mlx_indextts2` |

## Phase 4 Stop Condition

Do not change defaults until all of the following are true:

1. Candidate provider is not deprecated, unavailable, or unimplemented.
2. Candidate has account-level smoke evidence.
3. Candidate fixture validates as `ready`; `seed_schema_only` placeholders are not eligible.
4. Candidate benchmark report exists and beats baseline on required metrics.
5. Candidate E2E smoke passes current and strict-threshold reporting.
6. Artifact provenance protects against mixed old/new model outputs.
7. Rollback route is documented and tested.
