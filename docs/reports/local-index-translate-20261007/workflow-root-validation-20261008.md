# Translation-to-dubbing workflow repair evidence

## Scope and acceptance

The requested delivery is a newly generated full 170.85-second source-video dub using local Index translation, source-grounded character context, compact spoken language, at most 1.08x speed, synchronized readable subtitles, and complete audio through the end. A generated file, passing code tests, or a model-generated semantic verdict alone cannot establish this acceptance.

No movie-specific exact translation replacement is part of the current workflow repair. Index remains the primary translation model. Candidate edits happen before translation acceptance; downstream dubbing and display stages must reuse the accepted wording and measured audio.

## Implemented workflow boundaries

- Source dialogue and measured words form an auditable utterance plan. Spoken sentence budgets are independent of display subtitle lengths. Short adjacent fragments share a budget only within the same known source voice; limited quiet lead-in requires acoustic evidence.
- Translation requests carry allowed natural-speech duration and native Index source/constraints syntax. A validation retry rebuilds that same syntax rather than feeding failed assistant text into a generic repair prompt.
- Natural candidate audio is measured before accepting translation. Candidate seeds and conditioning receipts are distinct; content readback and inclusive 1.08x fitting remain required. Short complete takes retain natural speed with silence padding.
- Duration candidates retain failed measured drafts, pronunciation guidance, deterministic source-fact checks and a source-semantic gate. Model edits cannot silently remove facts to pass timing.
- Accepted text, source clocks, synthesis policy and audio hashes bind the preflight receipt. Subtitle splitting is display-only. Steps 8–12 cannot rewrite or regroup accepted dialogue.
- A local MLX auxiliary-provider adapter is prepared. It is lazy, cannot download weights, and fails closed if its installed runtime/model is absent. Enabling a new auxiliary model requires a real capability check and another full run.

## Current evidence

| Check | Result | Evidence |
| --- | --- | --- |
| Full code regression | pass | 481 tests passed in 84.72 seconds, 2026-10-08; conda environment `videolingo` |
| Whitespace validation | pass | `git diff --check`, 2026-10-08 |
| V30 source-to-translation trial | fail | `run_workspaces/index_quality_v30_turn_budget_workflow_20261008/output/log/translation_speech_preflight.json` |
| V30 speech acceptance | fail | 41 units; 8 units still fail content/duration; cap 1.08x |
| New repaired full-video delivery | missing evidence | No accepted final video has been produced in V30 |
| Qwen3.5-9B auxiliary proposal | withdrawn | No capability comparison supports making this untested model a dependency; weights are not installed |

V30 is a diagnostic run using its saved configuration, not the final repaired delivery. Subsequent code changes invalidate treating that run as evidence for the current implementation. Its failed examples include unnecessarily expanded short questions, a changed departure/action direction, and a rhetorical question rendered as a command. These motivate general context/editor/semantic contracts rather than per-sentence patches.

## Model-selection correction

Official Index-Translate documentation explicitly supports context/style/length-constrained translation and reports stronger translation and instruction-following benchmarks than its Qwen3.5 baselines. Generic instruction-model capabilities do not demonstrate better dubbing edits or role understanding in this case. The untested Qwen3.5-9B download proposal is withdrawn; continue diagnosing the existing Index route first. Official target-syllable control is a distinct Index-Homura task and does not certify actual audio duration. Sources: https://huggingface.co/IndexTeam/Index-Translate-35B-A3B-preview and https://github.com/bilibili/Index-Translate/blob/main/docs/prompts.md (read 2026-10-08).

A same-model text-only diagnostic compared the current native JSON prompt with a compact official-style plain-output prompt on eight failed units: `run_workspaces/index_native_prompt_comparison_20261008_120226/report.json`. There was no uniform winner: compact output shortened one brief response, but expanded several longer lines and did not reliably honor the estimated syllable target. This is prompt-behavior evidence, not speech-duration acceptance, a model comparison, or an independent semantic-quality result. No alternate model was downloaded or enabled; the inactive Qwen9B preset was removed from root configuration.

## Next full-run checks

After any auxiliary-model installation and capability check, create a fresh workspace and rerun the original source with empty exact replacements. Require full speech preflight, translation semantic assessment, target word alignment, subtitle layout/timing checks, complete segment audio, media stream duration, full dubbing readback, and current artifact manifests. Report each gate separately; inferred role identity and perceptual voice/emotion quality must not be inferred from duration or readback checks.
