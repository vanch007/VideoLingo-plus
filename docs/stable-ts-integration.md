# Stable-TS Integration for VideoLingo

This document describes the integration of [stable-ts](https://github.com/jianfch/stable-ts) (Stable Whisper) into VideoLingo for improved local transcription.

## Overview

Stable-TS is a fork of OpenAI's Whisper that focuses on generating more accurate timestamps and stable transcriptions. It includes several improvements:

- **Enhanced Timestamp Accuracy**: Utilizes advanced techniques like VAD (Voice Activity Detection), timestamp refinement, and segment regrouping for more precise word-level and segment-level timestamps.
- **Audio Cleaning**: The project's main workflow supports `demucs` for music and noise removal from the audio before passing it to any transcription model, leading to higher quality transcriptions.
- **Silence Handling**: Improved silence handling for cleaner transcripts.
- **Apple Silicon Support**: Leverages MLX acceleration on Apple Silicon devices for faster processing.

## Installation

To use `stable-ts`, you need to install it and its dependencies. The provided script automates this process.

1.  Run the installation script:
    ```bash
    python install_stable_ts.py
    ```

    This script will:
    - Install or upgrade `stable-whisper` directly from the [official GitHub repository](https://github.com/jianfch/stable-ts) to ensure you have the latest version.
    - Install required dependencies like `torch`, `librosa`, etc.
    - Install `mlx` for Apple Silicon devices to enable hardware acceleration.

2.  If the script fails, you can manually install the packages:
    ```bash
    # Install stable-ts from GitHub (recommended)
    pip install --upgrade git+https://github.com/jianfch/stable-ts.git

    # Install other dependencies
    pip install torch librosa rich numpy ffmpeg-python

    # For Apple Silicon users
    pip install mlx
    ```

## Usage

To use `stable-ts` in VideoLingo:

1.  Set the `whisper.runtime` option to `stable-ts` in the UI settings or in your `config.yaml` file:
    ```yaml
    whisper:
      runtime: 'stable-ts'
    ```

2.  The global `demucs` option in the main settings will handle audio cleaning for all transcription runtimes, including `stable-ts`.

3.  Run the transcription process as usual.

## Features

- **Improved Timestamps**: The integration now uses `refine()` and `regroup()` to post-process the transcription, resulting in more accurate timestamps.
- **Centralized Demucs Support**: The main workflow handles `demucs` processing, providing cleaned audio to the `stable-ts` engine.
- **Better Silence Handling**: Uses a more sensitive VAD threshold for better detection of speech and non-speech segments.
- **Apple Silicon Support**: Uses MLX acceleration on Apple Silicon devices for faster processing.
- **Seamless Integration**: The output is compatible with the existing pipeline.

## Requirements

The `stable-ts` integration requires the following dependencies:

- `stable-whisper`
- `torch`
- `librosa`
- `ffmpeg`

For Apple Silicon users, it will automatically use MLX acceleration when available.

## Technical Details

The integration works by:

1.  The main workflow in `step2_whisperX.py` prepares the audio, optionally running `demucs` for audio separation.
2.  The cleaned audio is then passed in segments to the `transcribe_audio` function in `core/all_whisper_methods/stable_ts_local.py`.
3.  This function processes each segment with `stable-ts`.
4.  It then post-processes the results with `refine()` and `regroup()` to improve accuracy (for non-MLX devices).
5.  The results are converted to a compatible format and combined into a single transcript.

The implementation is located in `core/all_whisper_methods/stable_ts_local.py`.
