# Artifact Manifest Schema

**Date**: 2026-07-05
**Status**: Draft schema; helper and CLI audit implemented
**Phase**: Phase 3 - Workflow Hardening

## Purpose

Major artifacts must not be reused only because a file exists. A step may be skipped only when the artifact exists, passes its structural validator, and has a sidecar manifest that matches the current run context.

## Sidecar Naming

For an artifact path:

```text
output/audio/tts_tasks.xlsx
```

The sidecar path should be:

```text
output/audio/tts_tasks.xlsx.manifest.json
```

Helper module: `core/pipeline/artifact_manifest.py`

Current implementation scope:

- Builds manifest payloads with input hashes, run context, model metadata, quality config, and schema metadata.
- Writes and reads `*.manifest.json` sidecars.
- Validates artifact existence, optional structural validators, supported manifest version, input hash, profile/language, step, model metadata, and quality config hash.
- Exposes non-mutating status audit through `python -m core.cli status --include-manifests`.
- Does not yet enforce pipeline resume decisions; existing skip logic still uses the current artifact validators until migration is wired explicitly.

## Required Fields

```json
{
  "version": 1,
  "artifact": "output/audio/tts_tasks.xlsx",
  "created_at": "2026-07-05T00:00:00+0700",
  "run_id": "fullrun-20260516-012152",
  "input": {
    "path": "output/source.mp4",
    "sha256": "sha256:..."
  },
  "profile": "cinematic",
  "source_language": "zh",
  "target_language": "vi",
  "step": "gen_audio_task",
  "step_version": {
    "git_commit": "2854f94",
    "code_paths": [
      "core/step8_1_gen_audio_task.py"
    ]
  },
  "models": {
    "llm": {
      "provider": "openai_compatible",
      "model": "deepseek-ai/DeepSeek-V3.2",
      "fingerprint": "openai_compatible|openai_compatible|https://api.siliconflow.cn/v1|deepseek-ai/DeepSeek-V3.2|False"
    },
    "asr": {
      "runtime": "stable-ts",
      "model": "large-v3-turbo",
      "use_mlx": true
    },
    "tts": {
      "method": "mlx_indextts2",
      "backend": "indextts2"
    }
  },
  "quality": {
    "mode": "high_sync",
    "config_hash": "sha256:...",
    "content_score_min": 0.55,
    "leak_score_max": 0.12
  },
  "schema": {
    "required_columns": [
      "number",
      "start_time",
      "end_time",
      "text"
    ]
  }
}
```

## Resume Rule

A pipeline step can be skipped only when:

1. The artifact exists.
2. The existing structural validator passes.
3. The sidecar manifest exists.
4. Manifest `version` is supported.
5. Manifest input hash matches the current input.
6. Profile, source language, target language, step, model fingerprints, and quality config hash match.

If any condition fails, the step is `pending`.

## Initial Adoption

Implement manifests in this order:

1. Translation artifacts already have a partial manifest and should be aligned with this schema.
2. `output/log/cleaned_chunks.xlsx`
3. `output/audio/trans_subs_for_audio.srt`
4. `output/audio/tts_tasks.xlsx`
5. `output/audio/segs`
6. `output/dub.mp3`
7. `output/AI配音.mp4`

## Test Cases

| Test | Expected |
|---|---|
| Matching artifact and manifest | Step is skipped after resume enforcement is wired |
| Artifact exists but manifest missing | Step is pending after resume enforcement is wired; status audit reports `manifest_missing` now |
| Manifest model differs | Step is pending after resume enforcement is wired |
| Manifest input hash differs | Step is pending after resume enforcement is wired |
| Manifest quality config differs | Step is pending after resume enforcement is wired |
| Manifest version unsupported | Step is pending with actionable warning |

Implemented unit coverage currently checks matching sidecars, missing sidecars, quality hash mismatch, structural validator failure, sidecar naming, audit step mismatch, and CLI status manifest audit.
