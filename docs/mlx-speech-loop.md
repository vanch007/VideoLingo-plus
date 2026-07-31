# Local MLX speech loop

VideoLingo-plus uses a local Apple Silicon speech loop by default:

1. WhisperX or stable-ts transcribes the source audio and must emit native word timestamps.
2. MOSS-Transcribe-Diarize supplies speaker turns. VideoLingo overlays only the speaker IDs onto native ASR words and rejects insufficient coverage.
3. The MLX TTS router selects IndexTTS2 for Vietnamese/emotion-reference work and OmniVoice for Chinese dialogue. Qwen3-TTS, VoxCPM2, Higgs Audio, dots.tts, ZONOS2, and MOSS-TTS remain forceable voice-cloning candidates.
   When a subtitle target duration is available, IndexTTS2 uses its native duration-fit mode before any generic audio post-processing.
4. Speaker changes are hard boundaries in NLP sentence detection, subtitle timeline gap extension, and dubbing-task merging.
5. MOSS reads generated segment audio back into text.
6. Existing content-similarity, reference-leakage, timing, and repair gates decide whether the row passes or must be regenerated/reviewed.

## Configuration

```yaml
whisper:
  runtime: stable-ts

moss_asr:
  root: /Users/vanch/mlx-MOSS-Transcribe-Diarize
  python: /Users/vanch/.codex/envs/bluestone-mlx-audio/bin/python
  model: /Users/vanch/mlx-MOSS-Transcribe-Diarize/pretrained/mlx-moss-transcribe-diarize-8bit
  diarization_max_new_tokens: 8192

speaker_diarization:
  enabled: true
  backend: moss-mlx
  required: true
  min_word_coverage: 0.98
  max_gap_seconds: 0.35

tts_method: mlx_router

dubbing_quality:
  asr_readback: true
  asr_readback_backend: moss-mlx
```

`moss_asr.model` may be the published `vanch007/mlx-MOSS-Transcribe-Diarize-8bit` repo id when the converted local directory is unavailable. The configured isolated Python runtime prevents MOSS dependencies from colliding with the main VideoLingo environment.

## Evidence and artifacts

Each MOSS call writes an isolated directory under `output/log/moss_asr` containing:

- `raw_transcript.txt`
- native `segments.json`, `subtitle.srt`, and `subtitle.ass`
- `videolingo_run.json` with the effective model, command, and recovery status
- `videolingo_recovered_segments.json` only when the model text was usable but strict native parsing rejected a near-valid timestamp/speaker bracket

MOSS segment timing is not promoted to source word timing. It is used only to assign speaker IDs to the native stable-ts/WhisperX word intervals and for TTS readback. Evidence is written to `output/log/speaker_diarization.json`; sentence and final-timeline boundary reports are written beside their respective outputs.

## MOSS versus pyannote.audio

The default is [`moss-mlx`](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize) because this project targets Apple Silicon, already has an isolated MLX runtime, and MOSS returns timestamped speaker turns without introducing another PyTorch model stack or access token. [`pyannote.audio`](https://github.com/pyannote/pyannote-audio) is a mature specialist candidate with public diarization-error-rate benchmarks and overlap/change-detection components, but its official `community-1` setup requires PyTorch/torchcodec, accepting Hugging Face model conditions, and an access token. Its published speed figures use NVIDIA H100 hardware, so they do not establish performance on this Mac.

The word-to-turn overlay contract in `core/providers/speaker_diarization.py` is backend-neutral. A pyannote adapter can be added later without changing subtitle/TTS boundary rules, but selecting it as default remains `pending` until both backends are evaluated on the same representative project audio.

## Commands

```bash
python -m core.cli models list
python -m core.cli doctor
python -m core.cli run --input /path/video.mp4 --source zh --target vi --tts auto
python -m core.cli eval dubbing --readback
```

Forced MLX methods are `mlx_indextts2`, `mlx_omnivoice`, `mlx_qwen3_tts`, `mlx_voxcpm2`, `mlx_higgs_audio`, `mlx_dots_tts`, `mlx_zonos2`, and `mlx_moss_tts`. The router also accepts canonical backend names in row-level metadata.
