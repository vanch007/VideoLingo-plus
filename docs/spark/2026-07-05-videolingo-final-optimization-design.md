# VideoLingo-plus Final Optimization Blueprint

**Date**: 2026-07-05
**Status**: Approved design draft
**Author**: Spark Design
**Source material**:
- `docs/PROJECT_OPTIMIZATION_PLAN_2026.md`
- `docs/workflows/REGISTRY.md`
- `.ai_project.md`
- `.ai_memory.md`

## 1. Executive Decision

The next optimization cycle should not start by directly replacing default models. The project already has useful local improvements: shared CLI workflow, provider registries, translation provenance checks, MLX TTS routing, dubbing quality gates, ASR readback, and repair workflows. The highest-risk gap is that the system lacks a complete evidence loop for deciding when a workflow or model is safe to promote.

The final optimization direction is:

1. Freeze and document the current baseline.
2. Add governance around workflow specs, provider lifecycle, artifact provenance, and quality thresholds.
3. Build benchmark and smoke-test gates for LLM, ASR, TTS, and end-to-end routes.
4. Upgrade default models only after benchmark evidence beats the current baseline.
5. Keep cloud ASR/TTS and premium models as candidates until adapters, cost evidence, and quality gates pass.

Current output evidence is useful but must be labeled precisely: the existing full run reports `102/102` dubbing rows ok under current `content_score_min=0.55`. Under the stricter cinematic target of `0.88`, `47/102` rows are below threshold. Therefore this baseline is **pass under current config** and **pending under strict cinematic content threshold**.

## 2. Design Goals

### Goals

- Produce an implementation-ready optimization blueprint for the next work cycle.
- Preserve working local routes while making model upgrades safer.
- Make workflow state, model choice, artifact reuse, and quality gates auditable.
- Separate provider registration from provider implementation and provider recommendation.
- Define explicit decision gates before default LLM, ASR, or TTS routes change.
- Reduce silent reuse of stale artifacts when input, model, profile, or quality config changes.

### Non-Goals

- Do not implement benchmark commands in this spec.
- Do not change `config.yaml` defaults in this spec.
- Do not remove existing local MLX or Edge routes.
- Do not delete existing output, history, voice samples, redub assets, or dirty worktree files.
- Do not mark any new model as production-ready without local account-level smoke evidence.

## 3. Current Baseline

| Area | Current state | Design implication |
|---|---|---|
| Main app | Streamlit UI plus shared CLI | Keep both paths using shared core modules |
| Main pipeline | Input -> ASR -> split/summary/translation -> timeline -> subtitle/TTS -> merge -> eval | Treat every step as a workflow with artifacts and failure modes |
| Current default LLM | OpenAI-compatible route using `api.base_url` / `api.model`, currently DeepSeek V3.2 through SiliconFlow | Do not replace default until benchmark evidence exists |
| Current ASR | stable-ts + MLX `large-v3-turbo`; WhisperX and FunASR available | Keep stable-ts MLX as Mac default, benchmark alternatives |
| Current TTS | `mlx_indextts2`; router can reach IndexTTS2, OmniVoice, Qwen3-TTS, VoxCPM2 | Keep IndexTTS2 default for current zh->vi route until measured competitor wins |
| Quality loop | Dubbing eval, ASR readback, repair planning/apply, timeline rescue | Promote into formal workflow specs and quality threshold manifests |
| Workflow docs | Registry exists, individual workflow specs missing | Create top workflow specs before broad refactor |
| Artifact safety | Translation provenance guard exists; other artifacts mostly rely on file existence/columns | Add sidecar manifests and fingerprints across major artifacts |
| Provider health | `models list` checks oMLX and local MLX TTS roots | Add provider lifecycle and route verification |

## 4. Target Operating Model

The optimized project should operate as five coordinated layers.

### 4.1 Evidence Layer

Purpose: make the current system measurable before changing it.

Required outputs:

- Baseline report for current full run.
- Strict threshold comparison report for `content_score_min=0.55` and `0.88`.
- Runtime dependency snapshot.
- Current provider/model snapshot.
- Source fixture list for future benchmarks.

Acceptance:

- Any future model upgrade can compare against a named baseline.
- Quality status always states which thresholds produced the result.

### 4.2 Governance Layer

Purpose: turn implicit workflows and model choices into governed contracts.

Required outputs:

- `docs/workflows/REGISTRY.md` maintained as the index.
- Top workflow specs:
  - `WORKFLOW-cli-full-run.md`
  - `WORKFLOW-translation-provenance-guard.md`
  - `WORKFLOW-dubbing-quality-eval.md`
  - `WORKFLOW-dubbing-repair-apply.md`
  - `WORKFLOW-provider-model-update.md`
  - `WORKFLOW-batch-run.md`
- Provider lifecycle metadata with `stable`, `candidate`, `experimental`, `deprecated`, and `unavailable` states.
- TTS route state split into `registered`, `implemented`, `healthy`, and `recommended`.

Acceptance:

- New contributors can see which workflows exist and which specs are missing.
- Unimplemented providers cannot look equivalent to working providers.
- Deprecated or unavailable model presets are visible before a run starts.

### 4.3 Verification Layer

Purpose: make model and workflow upgrades evidence-driven.

Required outputs:

- LLM benchmark specification.
- ASR benchmark specification.
- TTS benchmark specification.
- End-to-end smoke matrix specification.
- `models verify` behavior spec.

Minimum metrics:

| Area | Metrics |
|---|---|
| LLM | JSON validity, line-count preservation, terminology accuracy, retry count, latency, estimated cost |
| ASR | CER/WER, timestamp drift, word timestamp coverage, language handling, RTF |
| TTS | duration ratio, ASR content score, reference leak score, natural speed factor, RTF, missing/tiny audio rate |
| End-to-end | total runtime, repair count, final quality pass rate, manual-review count, cost per finished minute |

Acceptance:

- No default model changes without benchmark output.
- No provider is marked recommended without smoke evidence.

### 4.4 Upgrade Layer

Purpose: introduce new models without destabilizing the working pipeline.

Candidate routing policy:

| Use case | Preferred baseline | Candidate direction | Promotion rule |
|---|---|---|---|
| Cinematic zh->vi translation | Current DeepSeek V3.2 route | DeepSeek V4 Pro, GPT-5.x, Gemini current Pro route | Must beat baseline on line preservation, terminology, quality, retry rate, and cost/latency target |
| Fast draft subtitles | Current config route | DeepSeek V4 Flash or Gemini Flash class | Must pass JSON and line-count gates |
| Chinese source ASR | stable-ts MLX | FunASR Nano/SenseVoice, WhisperX, cloud ASR | Must improve CER/WER or timestamp drift on fixtures |
| Noisy/multilingual ASR | stable-ts or WhisperX | cloud ASR fallback | Must improve difficult fixture set and have cost bound |
| Vietnamese cloned TTS | MLX IndexTTS2 | ElevenLabs/OpenAI/premium route after adapter | Must pass duration/readback/leak/cost gates |
| Chinese dialogue TTS | MLX OmniVoice candidate | Qwen3-TTS/VoxCPM2 | Must pass language-specific TTS fixture |

Acceptance:

- Default route changes happen only through a provider-model update workflow.
- Every promoted model keeps an explicit fallback to the previous baseline.

### 4.5 Operations Layer

Purpose: make repeated runs and batch workflows reliable.

Required outputs:

- Batch resource lock design for local MLX/TTS/ASR jobs.
- Run report design summarizing input, models, artifacts, quality gates, and repair history.
- Provider retirement check cadence.
- Maintenance rule for adding new providers through registries/adapters, not UI branches first.

Acceptance:

- Concurrent runs cannot silently corrupt shared local resources.
- Operators can inspect a finished run without reading logs one by one.

## 5. Execution Roadmap

### Phase 0: Baseline Freeze

**Objective**: lock down what is known before changing defaults.

Tasks:

1. Generate a baseline report from current run state, translation status, model list, dependency versions, and dubbing eval.
2. Record strict quality threshold comparison for `0.55` and `0.88`.
3. Mark current workflow registry as the starting index.
4. Freeze small benchmark fixture candidates for zh, vi, and en.

Deliverables:

- `docs/reports/baseline-2026-07-05.md`
- fixture inventory under `tests/fixtures/` or documented external paths
- registry update with reviewed status

Acceptance:

- Baseline report can be used as the comparator for future model tests.
- No default model is changed in Phase 0.

Stop conditions:

- Existing output cannot be read.
- Quality metrics cannot be reproduced.
- Fixtures include secrets or large generated assets.

### Phase 1: Provider Hygiene

**Objective**: make provider/model availability explicit.

Tasks:

1. Add model/provider lifecycle metadata.
2. Mark stale, deprecated, candidate, experimental, and unavailable presets.
3. Split TTS route state into registered/implemented/healthy/recommended.
4. Define `models verify` output and failure states.
5. Hide or warn on unimplemented TTS routes in UI/CLI.

Deliverables:

- provider lifecycle registry
- `models verify` spec or command behavior doc
- UI/CLI provider display rules

Acceptance:

- Deprecated model presets cannot be mistaken for stable defaults.
- `openai_tts`, `elevenlabs_tts`, and `cosyvoice3_tts` are not presented as working unless adapters pass smoke tests.

Stop conditions:

- Provider state cannot be represented without exposing secrets.
- A provider requires live billing-resource creation before basic metadata can be stored.

### Phase 2: Benchmark System

**Objective**: compare LLM, ASR, TTS, and E2E routes on repeatable fixtures.

Tasks:

1. Define LLM benchmark fixtures and scoring.
2. Define ASR benchmark fixtures and scoring.
3. Define TTS benchmark fixtures and scoring.
4. Define a 60-second E2E smoke matrix.
5. Store benchmark output in JSON plus Markdown summary.

Deliverables:

- benchmark command specs
- fixture schema docs
- benchmark report format

Acceptance:

- Each benchmark records model, provider, config, timestamp, metrics, errors, and cost estimate when available.
- A route cannot be promoted if benchmark output is missing or worse than baseline on required metrics.

Stop conditions:

- Provider account cannot access the model.
- Model output cannot be legally or safely cached in benchmark artifacts.

### Phase 3: Workflow Hardening

**Objective**: reduce stale artifact reuse and undocumented recovery behavior.

Tasks:

1. Add artifact sidecar manifest design for major outputs.
2. Define resume skip rule: artifact validator and manifest must both match.
3. Write top six workflow specs from the registry.
4. Define batch resource lock workflow.
5. Embed effective quality config into eval artifacts.

Deliverables:

- artifact manifest schema
- workflow specs for top six workflows
- quality config versioning design
- batch lock design

Acceptance:

- Resume rejects mismatched input, language, profile, provider, model, prompt version, step version, or quality config.
- Each high-risk workflow includes happy path, validation failures, timeout behavior, retry behavior, partial cleanup, observable states, and tests.

Stop conditions:

- Existing artifact formats cannot support sidecar linking without breaking current users.
- Workflow specs reveal behavior that needs user decision before implementation.

### Phase 4: Model Upgrade

**Objective**: update defaults only after evidence gates pass.

Tasks:

1. Benchmark DeepSeek V4, current OpenAI/Gemini candidates, and current baseline route for translation.
2. Benchmark ASR candidates against stable-ts MLX.
3. Implement and benchmark premium cloud TTS only if provider hygiene marks adapters available.
4. Run E2E smoke matrix for recommended routes.
5. Update defaults through provider-model update workflow with rollback.

Deliverables:

- model comparison reports
- selected route decision records
- updated config/profile defaults only after approval
- rollback instructions

Acceptance:

- New default beats baseline on required metrics.
- Fallback to previous baseline is documented and tested.
- Artifact provenance prevents mixed old/new model outputs.

Stop conditions:

- Candidate improves one metric but regresses required quality gates.
- Candidate access, pricing, or quota is unstable.
- Adapter lacks smoke evidence.

## 6. Decision Gates

### Default Model Replacement Gate

A default LLM, ASR, or TTS route may change only when all are true:

- Baseline report exists.
- Candidate benchmark report exists.
- Candidate passes JSON/format gates where relevant.
- Candidate meets or exceeds quality thresholds.
- Candidate latency and cost are within accepted bounds.
- E2E smoke run passes.
- Rollback route is documented.
- Artifact provenance prevents mixed artifacts.

### Provider Recommendation Gate

A provider may be marked `recommended` only when all are true:

- Adapter is implemented.
- API key or service requirements are documented without secrets.
- Health/smoke check passes.
- At least one benchmark fixture passes.
- Failure modes and retry behavior are documented.

### Artifact Resume Gate

A step may be skipped only when all are true:

- Expected artifact exists and passes structural validator.
- Sidecar manifest exists.
- Manifest matches input hash, run profile, source/target language, provider, model, prompt version, step version, and quality config.

### Quality Gate

Every quality report must include:

- Effective threshold values.
- Profile name.
- Quality config hash or version.
- Count of rows passing current threshold.
- Count of rows failing stricter cinematic threshold when applicable.

## 7. Backlog

### P0: Must Do First

| Task | Owner role | Output | Acceptance |
|---|---|---|---|
| Baseline report | Project assistant | Markdown report | Reproduces current status/eval/model/dependency evidence |
| Provider lifecycle registry | Backend/config engineer | Registry data | Every provider has lifecycle state |
| Hide unavailable TTS routes | Frontend/backend engineer | UI/CLI route filtering | Unimplemented routes no longer appear as normal choices |
| Quality threshold manifest | Backend engineer | Eval metadata | Eval explains threshold source |
| First workflow spec | Workflow architect | `WORKFLOW-cli-full-run.md` | Includes branches, recovery, observable states |

### P1: Build Verification

| Task | Owner role | Output | Acceptance |
|---|---|---|---|
| LLM benchmark spec and runner | AI/backend engineer | JSON/MD report | Preserves lines and validates JSON |
| ASR benchmark spec and runner | AI/audio engineer | JSON/MD report | Reports CER/WER and timestamp drift |
| TTS benchmark spec and runner | AI/audio engineer | JSON/MD report | Reports duration, readback, leak, RTF |
| E2E smoke matrix | QA/backend engineer | Smoke report | 60s routes pass or fail with reason |
| Provider verify command | Backend engineer | CLI command | Invalid/deprecated providers fail clearly |

### P2: Harden and Upgrade

| Task | Owner role | Output | Acceptance |
|---|---|---|---|
| Artifact sidecar manifests | Backend engineer | Manifest schema and validators | Resume rejects mismatches |
| Batch resource lock | Backend engineer | Lock workflow | Concurrent local model contention prevented |
| Run report command | Project assistant/backend engineer | CLI report | Summarizes models, artifacts, quality, repair history |
| Cloud ASR adapter | AI/backend engineer | Candidate provider | Passes benchmark and cost gate |
| Premium TTS adapter | AI/audio engineer | Candidate provider | Passes smoke, benchmark, and fallback gate |

## 8. Test Strategy

### Unit Tests

- Provider lifecycle parsing.
- TTS route filtering by implementation state.
- Artifact manifest matching and mismatch cases.
- Quality config hash/version generation.
- Benchmark fixture schema validation.

### Integration Tests

- `models verify` against mocked providers.
- Benchmark runner with tiny local fixtures.
- Resume behavior when manifest matches.
- Resume behavior when model/profile/input/quality config differs.
- Repair workflow after eval threshold changes.

### Manual Smoke Tests

- Current baseline route.
- One fast draft subtitle route.
- One local MLX TTS route.
- One candidate cloud provider only after credentials are configured.

## 9. Reporting Requirements

Every optimization cycle should produce a short report containing:

- Run id.
- Input type and duration.
- Source/target language.
- LLM/ASR/TTS provider and model fingerprints.
- Profile and quality config.
- Runtime and package snapshot.
- Dubbing eval summary.
- ASR readback summary.
- Repair history summary.
- Benchmark comparison if model changed.
- Decision: keep baseline, promote candidate, or reject candidate.

## 10. Risks

| Risk | Severity | Mitigation |
|---|---|---|
| Stale model preset stays in config | High | Provider lifecycle registry plus `models verify` |
| Quality pass is misread because threshold drifted | High | Embed effective quality config in every eval |
| Resume reuses artifacts from old input/model/profile | High | Artifact sidecar manifests |
| Cloud model works in docs but not account/region | Medium | Account-level smoke before candidate status |
| Premium TTS improves quality but raises cost too much | Medium | Cost per finished minute metric |
| Benchmark fixtures are too small to represent real videos | Medium | Start small, then expand fixture suite after first pass |
| Batch jobs compete for local MLX resources | Medium | Resource lock and batch workflow spec |

## 11. Open Questions

- What is the acceptable cost per finished video minute for premium cloud routes?
- Should strict cinematic content threshold remain `0.88`, or should it be recalibrated per language pair?
- Which language pairs are highest priority after zh->vi?
- Should workflow specs live only under `docs/workflows/`, or should they also be surfaced in the docs website?

## 12. Approval Criteria for This Blueprint

This blueprint is ready for implementation planning when:

- The user accepts the staged execution model.
- Phase 0 and Phase 1 are treated as prerequisites for broad model upgrades.
- Default model replacement remains gated by benchmark evidence.
- Existing working local routes remain available until measured replacements are proven.

