<div align="center">

<img src="/docs/logo.png" alt="VideoLingo Logo" height="140">

# Connect the World, Frame by Frame

<a href="https://trendshift.io/repositories/12200" target="_blank"><img src="https://trendshift.io/api/badge/repositories/12200" alt="Huanshere%2FVideoLingo | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>

[**English**](/README.md)｜[**简体中文**](/translations/README.zh.md)｜[**繁體中文**](/translations/README.zh-TW.md)｜[**日本語**](/translations/README.ja.md)｜[**Español**](/translations/README.es.md)｜[**Русский**](/translations/README.ru.md)｜[**Français**](/translations/README.fr.md)

</div>

## 🌟 Overview ([Try VL Now!](https://videolingo.io))

VideoLingo is an all-in-one video translation, localization, and dubbing tool aimed at generating Netflix-quality subtitles. It eliminates stiff machine translations and multi-line subtitles while adding high-quality dubbing, enabling global knowledge sharing across language barriers.

Key features:
- 🎥 YouTube video download via yt-dlp

- **🎙️ Word-level and Low-illusion subtitle recognition with WhisperX**

- **📝 NLP and AI-powered subtitle segmentation**

- **📚 Custom + AI-generated terminology for coherent translation**

- **🔄 3-step Translate-Reflect-Adaptation for cinematic quality**

- **✅ Readable subtitles with natural line wrapping for long dubbed sentences**

- **🗣️ Local MLX voice-cloning dubbing with nine selectable backends**

- 🚀 One-click startup and processing in Streamlit

- 🌍 Multi-language support in Streamlit UI

- 📝 Detailed logging with progress resumption

This fork includes speaker-aware translation, local voice cloning, and explicit dubbing quality reports.

## 🎥 Demo

<table>
<tr>
<td width="33%">

### Dual Subtitles
---
https://github.com/user-attachments/assets/a5c3d8d1-2b29-4ba9-b0d0-25896829d951

</td>
<td width="33%">

### Voice Clone Demo
---
https://github.com/user-attachments/assets/e065fe4c-3694-477f-b4d6-316917df7c0a

</td>
<td width="33%">

### Voice Clone Demo
---
https://github.com/user-attachments/assets/47d965b2-b4ab-4a0b-9d08-b49a7bf3508c

</td>
</tr>
</table>

### Language Support

**Input Language Support(more to come):**

🇺🇸 English 🤩 | 🇷🇺 Russian 😊 | 🇫🇷 French 🤩 | 🇩🇪 German 🤩 | 🇮🇹 Italian 🤩 | 🇪🇸 Spanish 🤩 | 🇯🇵 Japanese 😐 | 🇨🇳 Chinese* 😊

> *Chinese uses a separate punctuation-enhanced whisper model, for now...

**Translation supports all languages, while dubbing language depends on the chosen TTS method.**

## Installation

You don't have to read the whole docs, [**here**](https://share.fastgpt.in/chat/share?shareId=066w11n3r9aq6879r4z0v9rh) is an online AI agent to help you.

> **Note:** For Windows users with NVIDIA GPU, follow these steps before installation:
> 1. Install [CUDA Toolkit 12.6](https://developer.download.nvidia.com/compute/cuda/12.6.0/local_installers/cuda_12.6.0_560.76_windows.exe)
> 2. Install [CUDNN 9.3.0](https://developer.download.nvidia.com/compute/cudnn/9.3.0/local_installers/cudnn_9.3.0_windows.exe)
> 3. Add `C:\Program Files\NVIDIA\CUDNN\v9.3\bin\12.6` to your system PATH
> 4. Restart your computer

> **Note:** FFmpeg is required. Please install it via package managers:
> - Windows: ```choco install ffmpeg``` (via [Chocolatey](https://chocolatey.org/))
> - macOS: ```brew install ffmpeg``` (via [Homebrew](https://brew.sh/))
> - Linux: ```sudo apt install ffmpeg``` (Debian/Ubuntu)

1. Clone the repository

```bash
git clone https://github.com/Huanshere/VideoLingo.git
cd VideoLingo
```

2. Install dependencies(requires `python=3.10`)

```bash
conda create -n videolingo python=3.10.0 -y
conda activate videolingo
python install.py
```

3. Start the application

```bash
streamlit run st.py
```

### Docker
Alternatively, you can use Docker (requires CUDA 12.4 and NVIDIA Driver version >550), see [Docker docs](/docs/pages/docs/docker.en-US.md):

```bash
docker build -t videolingo .
docker run -d -p 8501:8501 --gpus all videolingo
```

## APIs
VideoLingo supports OpenAI-compatible LLM APIs and local MLX speech backends:
- LLM: `claude-3-5-sonnet-20240620`, `deepseek-chat(v3)`, `gemini-2.0-flash-exp`, `gpt-4o`, ... (sorted by performance)
- WhisperX: Run whisperX locally or use 302.ai API
- TTS: the local MLX router with IndexTTS2, OmniVoice, Qwen3-TTS, VoxCPM2, Higgs Audio, dots.tts, ZONOS2, and MOSS-TTS.

> **Note:** Translation can still use an OpenAI-compatible API; dubbing in this fork is local MLX only.

## 2026 Local Upgrade Notes

This fork keeps the original workflow but adds provider-based configuration and environment checks:

- Final V13 code is integrated in `core/` with the default configuration in `config.yaml`. See [integration record and remaining evidence gaps](docs/reports/final-code-integration-20260927/README.md). Run `python -m pytest -q` for the maintained regression suite; historical run directories are excluded from default discovery.
- Secrets are no longer expected in `config.yaml`. Set keys through `.env` / shell variables based on `.env.example`.
- Run `conda run -n videolingo python -m core.doctor` before long jobs to check FFmpeg, Python packages, services, secrets, and recoverable step status.
- LLM routing is provider-based. The configured default is the OpenAI-compatible SiliconFlow `zai-org/GLM-4.5-Air` route; local oMLX remains selectable.
- Source text and word timing use stable-ts or WhisperX. The default speaker sidecar is `nemotron-mlx` with `mlx-community/Nemotron-3-Diarization-8bit`; it labels native words without replacing their timestamps. Required word coverage is 95%, and translation requires speaker information. MOSS remains an optional sidecar.
- The configured TTS default is local `mlx_indextts2`. Short or fragmented utterances use a same-speaker anchor for timbre plus the original utterance for emotion; long utterances use their original source clip for both. Use `--tts mlx_indextts2` explicitly with the cinematic profile; `--tts auto` selects the MLX router.
- Mixing retains a continuous Demucs background track, preserves source context outside speech windows, writes a lossless master, and checks encoded audio peak and stream timing. These checks do not establish perceptual identity or emotion fidelity.
- High-sync dubbing now uses duration budgets, optional LLM shortening/retry, IndexTTS2 `target_duration` passthrough, absolute-timeline audio overlay, ASR/leak score fields when available, and `output/audio/dubbing_eval.json` / `.xlsx` metrics.
- Shared CLI entrypoint: `conda run -n videolingo python -m core.cli doctor`, `python -m core.cli models list`, `python -m core.cli run --input <video-or-srt> --source zh --target vi --profile cinematic --tts auto`, and `python -m core.cli eval dubbing`.
- Preview a local run with `python -m core.cli run --input <local-video> --source zh --target en --profile cinematic --tts mlx_indextts2 --dry-run`; omit `--dry-run` to execute against the current workspace. Local smoke runs support `--smoke-seconds 60`.
- ASR readback defaults to the local Qwen3-ASR command in `scripts/qwen3_asr_readback.py`; run `python -m core.cli eval dubbing --readback`. MOSS and builtin stable-whisper remain selectable alternatives.
- Dubbing repair loop: `python -m core.cli repair dubbing --limit 20` writes `output/audio/dubbing_repair_plan.json`; add `--reasons missing_audio,over_duration` to target specific failure classes, `--apply` to mutate selected rows, `--batches 3` to repeat several plan/apply batches, and `--full-remap` only when chunk timing should be recomputed globally. Applied repairs rebuild `output/dub.mp3` and `output/AI配音.mp4` by default; use `--no-rebuild-output` only for intermediate tuning. Batch runs append summaries to `output/audio/dubbing_repair_history.jsonl`, and over-duration triage writes `output/audio/dubbing_over_duration_report.json`.
- Low-content fallback is configured through `dubbing_repair.low_content_fallback_backend`. Rows requiring separate emotion conditioning reject backends that cannot preserve it. Repeated ASR-quality failures become explicit `manual_review` warnings.
- Repair/rewrite/readback/TTS use the current `output/pipeline_state.json` target language before `config.yaml`, so a resumed English smoke is not rewritten or scored as Vietnamese when config defaults drift.
- ASR readback stores a fingerprint of target text, source reference, language, backend/model, and segment audio signatures. Cached scores are reused only when the fingerprint still matches.
- Audio merge fails fast on missing or corrupt segment audio unless silence fallback is explicitly enabled, preventing a broken repair run from producing a silent dubbed video.
- Timeline rescue loop: `python -m core.cli rescue timeline` writes `output/audio/timeline_rescue_report.json` / `.xlsx`; add `--write-candidate-tasks` to create non-destructive `output/audio/tts_tasks_timeline_rescue.xlsx` before deciding whether to regenerate all dubbing audio.
- Translation provenance check: `python -m core.cli translation status` reports whether LLM-generated translation artifacts match the current configured model. Use `translation adopt-current --apply` to write a manifest for trusted existing artifacts. `translation archive` is dry-run by default; add `--apply` to move translation-derived artifacts and GPT logs into `output/history/translation_*` before a clean retranslation. CLI `run` / `resume` blocks stale translation artifacts unless `--auto-archive-stale-translation` is explicit.
- Repair-time LLM rewrite uses `dubbing_repair.llm_timeout_seconds` and `dubbing_repair.llm_retry_attempts`, separate from the longer global translation retry settings.

Common modes:

1. Subtitle-only: download/import video -> ASR or SRT/subtitle extraction -> translation -> SRT/video subtitle output.
2. Subtitle + dubbing: run subtitle pipeline, then TTS task generation -> reference extraction -> TTS -> audio/video merge.
3. Existing SRT / embedded subtitles: skip ASR with Mode 2 or Mode 3, then continue from segmentation/translation.

For quick tuning, enable `smoke_test.enabled` in `config.yaml` or the Streamlit sidebar to trim newly downloaded videos to a short sample before running the full dubbing chain.

For detailed installation, API configuration, and batch mode instructions, please refer to the documentation: [English](/docs/pages/docs/start.en-US.md) | [中文](/docs/pages/docs/start.zh-CN.md)

## Current Limitations

1. WhisperX transcription performance may be affected by video background noise, as it uses wav2vac model for alignment. For videos with loud background music, please enable Voice Separation Enhancement. Additionally, subtitles ending with numbers or special characters may be truncated early due to wav2vac's inability to map numeric characters (e.g., "1") to their spoken form ("one").

2. Using weaker models can lead to errors during intermediate processes due to strict JSON format requirements for responses. If this error occurs, please delete the `output` folder and retry with a different LLM, otherwise repeated execution will read the previous erroneous response causing the same error.

3. The dubbing feature may not be 100% perfect due to differences in speech rates and intonation between languages, as well as the impact of the translation step. However, this project has implemented extensive engineering processing for speech rates to ensure the best possible dubbing results.

4. **Multilingual video transcription recognition will only retain the main language**. This is because whisperX uses a specialized model for a single language when forcibly aligning word-level subtitles, and will delete unrecognized languages.

5. Speaker IDs are acoustic clusters, not verified character names. Context-derived roles remain inferences; unknown or tied identities must not authorize a different speaker's clone reference.

## 📄 License

This project is licensed under the Apache 2.0 License. Special thanks to the following open source projects for their contributions:

[whisperX](https://github.com/m-bain/whisperX), [yt-dlp](https://github.com/yt-dlp/yt-dlp), [json_repair](https://github.com/mangiucugna/json_repair), [BELLE](https://github.com/LianjiaTech/BELLE)

## 📬 Contact Me

- Submit [Issues](https://github.com/Huanshere/VideoLingo/issues) or [Pull Requests](https://github.com/Huanshere/VideoLingo/pulls) on GitHub
- DM me on Twitter: [@Huanshere](https://twitter.com/Huanshere)
- Email me at: team@videolingo.io

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Huanshere/VideoLingo&type=Timeline)](https://star-history.com/#Huanshere/VideoLingo&Timeline)

---

<p align="center">If you find VideoLingo helpful, please give me a ⭐️!</p>
