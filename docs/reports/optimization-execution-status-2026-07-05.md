# Optimization Execution Status

**Date**: 2026-07-05
**Goal**: Execute all phases of the final optimization blueprint while preserving current defaults until gates pass.

## Summary

Phase 0 is complete. Phase 1, Phase 2, and Phase 3 now have executable or reviewable foundations. Phase 4 is intentionally blocked from changing defaults because candidate benchmark reports, E2E smoke reports, real ASR/E2E media fixtures, and artifact sidecar enforcement are not complete yet.

## Phase Status

| Phase | Status | Evidence |
|---|---|---|
| Phase 0: Baseline Freeze | complete | `docs/reports/baseline-2026-07-05.md`, strict threshold report |
| Phase 1: Provider Hygiene | foundation complete | `core/providers/provider_governance.py`, `python -m core.cli models verify`, TTS selectable route filtering |
| Phase 2: Benchmark System | schema-gated skeleton complete | `core/providers/benchmarking.py`, `python -m core.cli benchmark ...`, benchmark plan reports |
| Phase 3: Workflow Hardening | top specs drafted; manifest audit complete; enforcement pending | `core/pipeline/artifact_manifest.py`, `python -m core.cli status --include-manifests`, top `WORKFLOW-*.md` specs |
| Phase 4: Model Upgrade | gated/pending | `docs/reports/phase4-model-upgrade-gates-2026-07-05.md` |

## New Commands

### Provider Lifecycle Verification

```bash
conda run -n videolingo python -m core.cli models verify
```

Current expected result: `ok=false` until unavailable/deprecated providers are resolved and required credentials are configured.

Current notable findings:

- `gemini_3_pro` is marked `deprecated`.
- `openai_tts`, `elevenlabs_tts`, and `cosyvoice3_tts` are marked `unavailable` and unimplemented.
- Candidate cloud LLM routes warn when required API key environment variables are not configured.
- Experimental routes are not eligible for default/recommended use unless explicitly included.

### Benchmark Plan

```bash
conda run -n videolingo python -m core.cli benchmark llm \
  --dataset tests/fixtures/translation_zh_vi.jsonl \
  --providers deepseek_v4_pro,gpt_5_5 \
  --fail-on-missing
```

Current expected result: `ready` for the seed LLM text fixture. ASR/E2E fixture plans return `seed_schema_only` until placeholder media paths are replaced.

Current plan reports:

| Target | Status | Report |
|---|---|---|
| LLM | `ready` | `docs/reports/benchmark-plan-llm-2026-07-05.md` |
| TTS | `ready` | `docs/reports/benchmark-plan-tts-2026-07-05.md` |
| ASR | `seed_schema_only` | `docs/reports/benchmark-plan-asr-2026-07-05.md` |
| E2E | `seed_schema_only` | `docs/reports/benchmark-plan-e2e-2026-07-05.md` |

### Artifact Manifest Audit

```bash
conda run -n videolingo python -m core.cli status --include-manifests
```

Current behavior: non-mutating audit only. It reports artifact existence, sidecar path, sidecar existence, supported version, artifact path mismatch, and step mismatch. It does not change resume behavior yet.

## UI Behavior

Streamlit TTS selection now uses selectable TTS methods only. Registered but unimplemented routes are not shown as normal choices. If `config.yaml` is manually set to an unavailable method, the UI keeps it visible with a warning instead of silently dropping it.

## Default Route Changes

No defaults were changed.

| Route | Current default |
|---|---|
| LLM provider | `openai_compatible` |
| LLM model | `deepseek-ai/DeepSeek-V3.2` |
| ASR | `stable-ts` + `large-v3-turbo` + MLX |
| TTS | `mlx_indextts2` |

## Validation

| Check | Status |
|---|---|
| `python -m py_compile core/all_tts_functions/tts_registry.py core/providers/provider_governance.py core/providers/benchmarking.py core/cli.py` | pass |
| `conda run -n videolingo python -m py_compile core/pipeline/artifact_manifest.py core/pipeline/artifacts.py core/providers/benchmarking.py core/cli.py` | pass |
| `conda run -n videolingo pytest tests/test_artifact_manifest.py tests/test_provider_contracts.py tests/test_cli_workflow.py` | pass, 33 tests |
| `conda run -n videolingo python -m core.cli status --include-manifests` | pass; current completed artifacts report `manifest_missing` sidecars |
| `conda run -n videolingo python -m core.cli models verify` | pass command execution; expected `ok=false` provider findings reported |
| `git diff --check` | pass |

## Remaining Blockers

| Blocker | Phase | Required next action |
|---|---|---|
| Real audio/video benchmark fixtures not frozen | Phase 2 | Replace seed-schema `null` ASR/E2E paths with tiny approved media fixtures or documented external paths |
| Candidate benchmark reports missing | Phase 4 | Run real benchmarks after fixtures and credentials exist |
| E2E smoke matrix missing | Phase 4 | Define and run 60-second smoke routes |
| Artifact sidecar enforcement not wired into resume | Phase 3 | Use `core/pipeline/artifact_manifest.py` from artifact skip logic after migration plan is approved |
| Some lower-priority workflows still lack dedicated specs | Phase 3 | Continue from `docs/workflows/REGISTRY.md` rows still marked `Missing` |
| Provider lifecycle currently code-defined | Phase 1 | Optionally move lifecycle metadata to config or YAML registry after format stabilizes |

## Next Recommended Step

Replace seed-schema placeholders with tiny approved media fixtures:

1. Fill `tests/fixtures/asr_samples.yaml` audio fields.
2. Fill `tests/fixtures/e2e_smoke.yaml` input fields.
3. Add real benchmark execution runners behind explicit provider/account gates.

The current benchmark skeleton validates target-specific fixture schemas and records planned metrics, but still does not call external model providers or score candidate outputs.
