# VideoLingo-plus Project Optimization Plan 2026

**Date**: 2026-07-05
**Author**: Workflow Optimizer x Workflow Architect
**Status**: Draft
**Scope**: Existing workflow analysis, current model/solution research, and optimization roadmap.

## Executive Summary

VideoLingo-plus is not a completely stale one-year-old project locally: this branch already added a shared CLI, provider registries, translation provenance checks, MLX TTS routing, dubbing quality gates, ASR readback, and repair workflows in April-June 2026. The main gap is governance: workflows exist in code but were not formalized as specs, model presets are not continuously verified, and there is no benchmark harness that proves when a new model should replace the current default.

Current baseline is usable under current config: `core.cli status --include-dubbing-eval` reports full run `fullrun-20260516-012152` completed, 102/102 dubbing rows ok, 0 warn, 0 fail, no missing audio, ASR readback configured and scored. However, this pass uses current `dubbing_quality.content_score_min=0.55`; the cinematic profile function sets `0.88`, and 47/102 rows are below 0.88. Therefore the accurate conclusion is: **pass under current config, pending under stricter cinematic content-score policy**.

The highest-leverage plan is:

1. Create a workflow registry and gradually add `WORKFLOW-*.md` specs for run, repair, translation reset, and provider update paths.
2. Add a provider/model benchmark harness before changing defaults.
3. Replace stale/unverified model presets, especially Gemini 3 Pro preview, and add lifecycle states for all providers.
4. Make every artifact resume decision depend on input + profile + model + step-version fingerprints, not only file existence.
5. Promote cloud TTS/ASR adapters only after they have implementation, smoke tests, and cost/quality metrics.

## Local Evidence Snapshot

| Area | Status | Evidence |
|---|---|---|
| Git state | Pass with dirty worktree | Branch `codex/videolingo-upgrade-20260426`; dirty/untracked voice and redub assets exist and were not touched |
| Full pipeline status | Pass | `core.cli status --include-dubbing-eval`: run completed, pending steps `[]` |
| Dubbing timing gate | Pass under current config | 102 total, 102 ok, 0 warn, 0 fail, max duration ratio 1.003, max speed factor 1.061 |
| ASR readback | Pass under current config | 102 scored rows, avg content score 0.8728, max leakage 0.0 |
| Strict cinematic threshold | Pending/fail risk | Current `content_score_min=0.55`; 47/102 rows are below 0.88 |
| Translation provenance | Pass | `stale_stage_count=0`, `warning_stage_count=0`, logs match `deepseek-ai/DeepSeek-V3.2` |
| Local provider health | Pass | oMLX exposes 3 models; MLX TTS roots ok for IndexTTS2, OmniVoice, Qwen3-TTS, VoxCPM2 |
| Runtime package snapshot | Pass | OpenAI 2.32.0, Streamlit 1.56.0, WhisperX 3.8.5, stable-ts 2.19.1, FunASR 1.2.7, yt-dlp 2026.3.17 |

## Current Workflow Map

### Main Pipeline

```text
Input
  -> URL download | local import | SRT import
  -> audio preparation + Demucs optional vocal separation
  -> source ASR
  -> SpaCy split
  -> semantic split by LLM
  -> summary + terminology + optional Chinese STT correction
  -> faithful/expressive translation
  -> subtitle split and remerge
  -> timeline alignment
  -> subtitle burn-in
  -> dubbing task generation
  -> dubbing chunk optimization
  -> reference audio extraction
  -> TTS segment generation
  -> full audio merge
  -> final video mux
  -> dubbing eval + optional ASR readback
```

### Repair and Recovery Loops

```text
Eval/readback
  -> classify missing_audio | over_duration | under_duration | speech_rate_fast | low_content_score | reference_leak
  -> write non-mutating repair plan
  -> optional apply
  -> rewrite/expand/speed-fit/slow-fit/regenerate
  -> clear stale ASR score columns
  -> rebuild dub.mp3 + AI配音.mp4
  -> re-evaluate
```

```text
Translation model/provider changes
  -> compare logs + manifest + current provider fingerprint
  -> pass, warning, or stale
  -> block downstream run if stale
  -> archive or adopt artifacts explicitly
```

## Model Stack: Current vs. Recommended Direction

### LLM Translation and Text Workflow

| Stage | Current | Issue | Recommended 2026 direction |
|---|---|---|---|
| Default LLM route | OpenAI-compatible `api.base_url` + `deepseek-ai/DeepSeek-V3.2` via SiliconFlow | Provider-specific model; not official DeepSeek current route evidence | Benchmark official DeepSeek V4 Flash/Pro, OpenAI GPT-5.5/GPT-5.4, Gemini 3.1 Pro/3.5 Flash, and optionally Anthropic via native adapter |
| JSON-heavy prompts | `json_repair` + retries, no structured output for current model | More retries/cost; error recovery after malformed output | Add model capability metadata: structured JSON supported, max context, cost tier, latency tier |
| Provider abstraction | OpenAI-compatible only | Anthropic native API cannot be used without proxy/adapter | Either add native provider adapters or integrate a lightweight model gateway such as LiteLLM-style routing |
| Prompt stages | Same provider for split/summary/translation/repair | Expensive or weak models are used uniformly | Use stage-specific routing: cheap structured model for split, high-quality model for translation, fast model for repair rewrite |

External findings:

- OpenAI model catalog lists GPT-5.5 as newest flagship and GPT-5.4/GPT-5.1 families as available API options: [OpenAI models](https://developers.openai.com/api/docs/models).
- Anthropic model overview lists Claude Fable/Opus/Sonnet/Haiku families with specific model aliases and lifecycle dates: [Anthropic model overview](https://docs.anthropic.com/en/docs/about-claude/models/overview).
- Gemini docs list Gemini 3.1 Pro Preview, Gemini 3.5 Flash, and mark Gemini 3 Pro Preview as shut down on 2026-03-09: [Gemini models](https://ai.google.dev/gemini-api/docs/models).
- DeepSeek official pricing/model table lists DeepSeek V4 Flash/Pro and notes older DeepSeek-V3.2 aliases are deprecated with retirement dates: [DeepSeek pricing](https://api-docs.deepseek.com/quick_start/pricing).

### ASR and Alignment

| Stage | Current | Issue | Recommended 2026 direction |
|---|---|---|---|
| Source ASR default | stable-ts + MLX `large-v3-turbo` | Fast on Apple Silicon, but MLX path has reduced advanced alignment features and can lose word-level data in manual splitting | Keep as Mac default, but add a forced-alignment fallback when word timestamps are missing |
| Chinese ASR candidate | FunASR Nano/SenseVoice | Good candidate, but no project benchmark matrix | Add zh/vi/en fixture set and compare CER/WER, timestamp drift, diarization availability |
| Cloud ASR | Not integrated as provider route | Useful fallback for noisy audio and validation | Add optional OpenAI `gpt-4o-transcribe` / `gpt-4o-mini-transcribe` and ElevenLabs Scribe-class route after cost benchmark |
| Readback | Builtin stable-whisper/MLX | Useful and cached; threshold drift found | Split timing gate and semantic gate thresholds by profile; store threshold version in eval artifact |

External findings:

- OpenAI speech-to-text guide documents `gpt-4o-transcribe` and `gpt-4o-mini-transcribe`: [OpenAI speech-to-text](https://developers.openai.com/api/docs/guides/speech-to-text).
- Current local ASR stack includes stable-ts, WhisperX, FunASR, and MLX Whisper; keep local routes because video alignment quality depends on timestamps, not just transcript text.

### TTS and Voice Cloning

| Stage | Current | Issue | Recommended 2026 direction |
|---|---|---|---|
| Default TTS | `mlx_indextts2` | Good local cloning and duration route, but benchmark is ad hoc | Keep as default for zh->vi cinematic runs until a measured contender beats it |
| Local router | IndexTTS2, OmniVoice, Qwen3-TTS, VoxCPM2 | Root health exists; no per-language scorecard | Add router benchmark with RTF, duration ratio, ASR content, leak score, human review |
| Cloud TTS | Edge works; OpenAI/ElevenLabs/CosyVoice3 registered but not implemented | UI/config can imply unavailable routes | Hide or mark unavailable until adapters pass smoke tests |
| Premium voice | Not implemented | High quality route missing | Implement ElevenLabs v3 and/or OpenAI TTS as optional premium provider, not default |

External findings:

- OpenAI text-to-speech guide documents `gpt-4o-mini-tts` and voice/instruction controls: [OpenAI text-to-speech](https://developers.openai.com/api/docs/guides/text-to-speech).
- ElevenLabs model docs list Eleven v3 as a high-quality TTS model and Scribe v2 for speech-to-text: [ElevenLabs models](https://elevenlabs.io/docs/models).

## Workflow Optimization Findings

| # | Finding | Impact | Recommended fix |
|---|---|---|---|
| WF-1 | No formal workflow specs exist for code workflows | Engineers/agents must infer recovery paths from code | Use `docs/workflows/REGISTRY.md` as the index, then write one `WORKFLOW-*.md` per high-risk workflow |
| WF-2 | Artifact resume uses existence/columns more than full provenance | Wrong input/model/profile can silently reuse old artifacts | Add artifact fingerprint: input hash, source/target, profile, provider, model, prompt version, step code version |
| WF-3 | Quality profile threshold drift | Pass/fail meaning changes depending on whether profile was applied | Store effective quality config inside `dubbing_eval.json`; make `status` show threshold version |
| WF-4 | Provider presets have no lifecycle status | Invalid/deprecated models can stay in config | Add `status: stable/candidate/experimental/deprecated`, `last_verified`, `source_url`, `smoke_status` |
| WF-5 | TTS registry includes not-yet-implemented providers | Users can select routes that intentionally fail | Split registry state into `registered`, `implemented`, `healthy`, `recommended` |
| WF-6 | LLM retry throttling uses rough word-count TPM estimate | May under/over-throttle and hurt cost/latency | Capture real token usage when provider returns it; add per-provider concurrency limits |
| WF-7 | Batch workflow has no queue/resource lock spec | Multiple TTS/ASR jobs can compete for local MLX/GPU/ffmpeg | Add local resource lock and batch scheduler states |
| WF-8 | Model upgrade path lacks benchmark acceptance criteria | New model changes can reduce translation/dubbing quality | Add model benchmark command and require before/after report before default changes |

## Target Architecture

### 1. Workflow Governance Layer

Add formal specs in this order:

1. `WORKFLOW-cli-full-run.md`
2. `WORKFLOW-translation-provenance-guard.md`
3. `WORKFLOW-dubbing-quality-eval.md`
4. `WORKFLOW-dubbing-repair-apply.md`
5. `WORKFLOW-provider-model-update.md`
6. `WORKFLOW-batch-run.md`

Each spec should include happy path, validation failures, timeouts, transient failures, permanent failures, partial cleanup, concurrent conflict behavior, observable states, and test cases.

### 2. Model Capability Registry

Create a versioned model registry, for example:

```yaml
llm_models:
  openai_gpt_5_5:
    provider: openai
    api_style: openai
    model: gpt-5.5
    status: candidate
    last_verified: 2026-07-05
    capabilities:
      json_object: true
      long_context: true
      translation_quality: pending
    recommended_for:
      - premium_translation
      - final_review
```

Required fields: `provider`, `api_style`, `model`, `status`, `last_verified`, `source_url`, `cost_tier`, `latency_tier`, `json_support`, `context_window`, `languages_tested`, `smoke_status`, `benchmark_status`.

### 3. Benchmark Harness

Add a non-mutating benchmark command:

```bash
python -m core.cli benchmark llm --dataset tests/fixtures/translation_zh_vi.jsonl --providers deepseek_v4_pro,gpt_5_5,gemini_3_1_pro
python -m core.cli benchmark asr --dataset tests/fixtures/asr_samples.yaml --providers stable_ts_mlx,funasr_nano,openai_transcribe
python -m core.cli benchmark tts --dataset tests/fixtures/tts_lines.yaml --providers mlx_indextts2,mlx_qwen3_tts,edge_tts,elevenlabs_v3
```

Minimum metrics:

| Area | Metrics |
|---|---|
| LLM | JSON validity, line-count preservation, terminology accuracy, style score, retry count, latency, cost |
| ASR | CER/WER, timestamp drift, word timestamp coverage, diarization availability, RTF |
| TTS | Duration ratio, ASR content score, reference leak score, RTF, natural speed factor, missing/tiny audio rate |
| End-to-end | Total runtime, cost/minute, repair count, final gate pass rate, human review count |

### 4. Artifact Provenance

Extend every major artifact with a sidecar manifest:

```json
{
  "artifact": "output/audio/tts_tasks.xlsx",
  "run_id": "fullrun-20260516-012152",
  "input_hash": "sha256:...",
  "source_language": "zh",
  "target_language": "vi",
  "profile": "cinematic",
  "step": "gen_audio_task",
  "step_version": "git:...",
  "model_fingerprints": {
    "llm": "...",
    "asr": "...",
    "tts": "..."
  },
  "quality_config_hash": "sha256:..."
}
```

Resume should skip a step only when both the artifact validator and manifest match.

## Recommended Model Routing Policy

| Use case | Default | Candidate | Fallback |
|---|---|---|---|
| zh->vi cinematic translation | DeepSeek V4 Pro after benchmark | GPT-5.5 / Gemini 3.1 Pro | Current DeepSeek V3.2 route |
| Fast draft subtitles | DeepSeek V4 Flash / Gemini 3.5 Flash | GPT-5.1 mini-class | Current config route |
| Strict JSON split/repair | Model with verified JSON output | GPT-5.1 mini-class / Gemini Flash | `json_repair` route |
| Chinese source ASR | stable-ts MLX large-v3-turbo | FunASR Nano/SenseVoice | WhisperX |
| Noisy multilingual ASR | stable-ts non-MLX or WhisperX | OpenAI transcribe / ElevenLabs Scribe | Manual review |
| Vietnamese cloned TTS | MLX IndexTTS2 | ElevenLabs v3 / OpenAI TTS after adapter | Edge TTS |
| Chinese dialogue TTS | MLX OmniVoice | Qwen3-TTS / VoxCPM2 after benchmark | IndexTTS2 |
| Low-content ASR repair fallback | Edge TTS for non-Vietnamese | OpenAI/ElevenLabs after adapter | Manual review |

## Implementation Roadmap

### Phase 0: Stabilize the Baseline (1-2 days)

| Task | Output | Acceptance |
|---|---|---|
| Add workflow registry | `docs/workflows/REGISTRY.md` | All discovered workflows are listed with status |
| Capture strict quality baseline | Eval under 0.55 and 0.88 thresholds | Report shows rows below each threshold |
| Mark stale model presets | Config/model registry issue list | Gemini 3 Pro preview flagged deprecated |
| Freeze benchmark fixtures | Small zh/vi/en ASR, LLM, TTS fixtures | Fixtures do not include secrets or large assets |

### Phase 1: Provider Hygiene (1-2 weeks)

| Task | Output | Acceptance |
|---|---|---|
| Add provider lifecycle metadata | `model_registry.yaml` or config section | Every provider has status/source/last_verified |
| Add model smoke verifier | `core.cli models verify` | Invalid/deprecated model fails before run |
| Hide unimplemented TTS routes | UI/CLI only show implemented routes by default | `openai_tts`, `elevenlabs_tts`, `cosyvoice3_tts` cannot be mistaken for working |
| Add structured JSON capability tests | Unit + smoke tests | JSON tasks use `response_format` only when verified |

### Phase 2: Benchmark Before Switching Defaults (2-4 weeks)

| Task | Output | Acceptance |
|---|---|---|
| LLM benchmark | Provider comparison JSON/MD | Line-count preservation >= 99%, JSON validity >= 99%, terminology score target defined |
| ASR benchmark | ASR comparison report | Timestamp drift and CER/WER reported per language |
| TTS benchmark | TTS scorecard | Duration ratio, ASR content, leak score, RTF reported |
| End-to-end smoke matrix | 60s smoke run per route | No route becomes default without E2E smoke pass |

### Phase 3: Workflow Hardening (4-8 weeks)

| Task | Output | Acceptance |
|---|---|---|
| Artifact sidecar manifests | Manifest per major artifact | Resume rejects mismatched input/model/profile |
| Formal workflow specs | Top 6 `WORKFLOW-*.md` files | Each has failure branches, cleanup, test cases |
| Batch scheduler/resource locks | Local lock files or job queue | Two batch jobs cannot compete for same MLX/TTS resource |
| Quality config versioning | Eval embeds thresholds/profile | `status` explains exact pass criteria |

### Phase 4: Strategic Model Upgrades (8-12 weeks)

| Task | Output | Acceptance |
|---|---|---|
| Replace default LLM if benchmark wins | Config/profile update | New default beats baseline on quality/cost/latency target |
| Implement premium cloud TTS | OpenAI/ElevenLabs adapter + tests | Smoke, duration, and readback pass |
| Add cloud ASR fallback | Provider adapter + benchmark | Improves noisy/multilingual samples |
| Add operator report | `core.cli report run` | One command summarizes run, cost, models, gates, repair history |

## Success Metrics

These are targets, not current measured results.

| Metric | Current | Target |
|---|---|---|
| Invalid/deprecated model used in config | Present risk | 0 after provider verification |
| Workflow specs for high-risk flows | 0 | 6 specs in first pass |
| Artifact stale-reuse risk | Partial guard only | All major artifacts fingerprinted |
| JSON parse/validation retry rate | Unknown | < 1% on benchmark fixtures |
| End-to-end smoke pass matrix | Unknown | 100% for recommended routes |
| Manual dubbing review rows | Project-dependent | 30-50% reduction after TTS routing benchmark |
| Batch run collision risk | Unknown | Resource lock prevents concurrent local model contention |

## Immediate Action Items

1. Treat the current 102/102 pass as a current-config pass, not a strict cinematic pass.
2. Remove or deprecate `gemini_3_pro` preset; replace with currently documented Gemini models after account smoke.
3. Add official DeepSeek V4 Flash/Pro presets and benchmark against current SiliconFlow DeepSeek V3.2 route before changing default.
4. Mark `openai_tts`, `elevenlabs_tts`, and `cosyvoice3_tts` as unavailable until adapters are implemented.
5. Add `models verify` and `benchmark` commands before broad model changes.
6. Write the first workflow spec: `WORKFLOW-cli-full-run.md`.

## Risks and Assumptions

| # | Assumption | Evidence | Risk if wrong |
|---|---|---|---|
| A1 | The local full-run output remains a valid baseline | CLI status and eval succeeded on 2026-07-05 | Benchmark comparisons may target an outdated artifact |
| A2 | User wants quality over minimum cost for cinematic dubbing | Existing `high_sync`, readback, repair loops | Wrong routing if cost is primary |
| A3 | Provider docs reflect account-level API availability | Public docs checked 2026-07-05 | Some model names may be unavailable to this account/region |
| A4 | Local MLX TTS repos are usable, not just present | `models list` root health passed only | Smoke generation could still fail |
| A5 | No production deployment exists | No server/IaC found | Operational roadmap may need adjustment if production exists elsewhere |

## Source Links Checked

- [OpenAI model catalog](https://developers.openai.com/api/docs/models)
- [OpenAI speech-to-text guide](https://developers.openai.com/api/docs/guides/speech-to-text)
- [OpenAI text-to-speech guide](https://developers.openai.com/api/docs/guides/text-to-speech)
- [Anthropic model overview](https://docs.anthropic.com/en/docs/about-claude/models/overview)
- [Gemini API model list](https://ai.google.dev/gemini-api/docs/models)
- [DeepSeek API pricing/model table](https://api-docs.deepseek.com/quick_start/pricing)
- [ElevenLabs model list](https://elevenlabs.io/docs/models)

