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

- **✅ Netflix-standard, Single-line subtitles Only**

- **🗣️ Dubbing with GPT-SoVITS, Azure, OpenAI, and more**

- 🚀 One-click startup and processing in Streamlit

- 🌍 Multi-language support in Streamlit UI

- 📝 Detailed logging with progress resumption

Difference from similar projects: **Single-line subtitles only, superior translation quality, seamless dubbing experience**

## 🎥 Demo

<table>
<tr>
<td width="33%">

### Dual Subtitles
---
https://github.com/user-attachments/assets/a5c3d8d1-2b29-4ba9-b0d0-25896829d951

</td>
<td width="33%">

### Cosy2 Voice Clone
---
https://github.com/user-attachments/assets/e065fe4c-3694-477f-b4d6-316917df7c0a

</td>
<td width="33%">

### GPT-SoVITS with my voice
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
VideoLingo supports OpenAI-Like API format and various TTS interfaces:
- LLM: `claude-3-5-sonnet-20240620`, `deepseek-chat(v3)`, `gemini-2.0-flash-exp`, `gpt-4o`, ... (sorted by performance)
- WhisperX: Run whisperX locally or use 302.ai API
- TTS: `azure-tts`, `openai-tts`, `siliconflow-fishtts`, **`fish-tts`**, `GPT-SoVITS`, `edge-tts`, `*custom-tts`(You can modify your own TTS in custom_tts.py!)

> **Note:** VideoLingo works with **[302.ai](https://gpt302.saaslink.net/C2oHR9)** - one API key for all services (LLM, WhisperX, TTS). Or run locally with Ollama and Edge-TTS for free, no API needed!

## 2026 Local Upgrade Notes

This fork keeps the original workflow but adds provider-based configuration and environment checks:

- Secrets are no longer expected in `config.yaml`. Set keys through `.env` / shell variables based on `.env.example`.
- Run `conda run -n videolingo python -m core.doctor` before long jobs to check FFmpeg, Python packages, services, secrets, and recoverable step status.
- LLM routing is provider-based. Default translation uses `llm.provider: openai_compatible`, which reads the existing `api.base_url` / `api.model` block. oMLX remains an optional provider for local experiments, but it is not the default translation route.
- ASR routes remain configurable: stable-ts + MLX for Mac, FunASR Nano for Chinese-heavy content, and WhisperX 3.8.5 for alignment/diarization scenarios.
- TTS routes are registry-based. Current working routes remain Edge/Cloudflare Edge, SiliconFlow IndexTTS2, VoxCPM, GPT-SoVITS, Piper, and Indonesian VITS. `tts_method: mlx_router` adds local MLX routing across `/Users/vanch/mlx-indextts2`, `/Users/vanch/mlx-omnivoice`, `/Users/vanch/mlx-qwen3-tts`, and `/Users/vanch/mlx-voxcpm2`.
- High-sync dubbing now uses duration budgets, optional LLM shortening/retry, IndexTTS2 `target_duration` passthrough, absolute-timeline audio overlay, ASR/leak score fields when available, and `output/audio/dubbing_eval.json` / `.xlsx` metrics.
- Shared CLI entrypoint: `conda run -n videolingo python -m core.cli doctor`, `python -m core.cli models list`, `python -m core.cli run --input <video-or-srt> --source zh --target vi --profile cinematic --tts auto`, and `python -m core.cli eval dubbing`.
- Optional ASR readback can be enabled by setting `dubbing_quality.asr_readback_command` in `config.yaml`, then running `python -m core.cli eval dubbing --readback`. The command receives `{audio}` and `{language}` placeholders and should print raw transcript text or JSON with `text` / `transcript` / `segments`.
- Dubbing repair loop: `python -m core.cli repair dubbing --limit 20` writes `output/audio/dubbing_repair_plan.json`; add `--reasons missing_audio,over_duration` to target specific failure classes, `--apply` to mutate selected rows, `--batches 3` to repeat several plan/apply batches, and `--full-remap` only when chunk timing should be recomputed globally. Batch runs append summaries to `output/audio/dubbing_repair_history.jsonl`, and over-duration triage writes `output/audio/dubbing_over_duration_report.json`.
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

5. **Cannot dub multiple characters separately**, as whisperX's speaker distinction capability is not sufficiently reliable.

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
