# Stable-TS Integration for VideoLingo

This document describes the integration of [stable-ts](https://github.com/jianfch/stable-ts) (Stable Whisper) into VideoLingo for improved local transcription.

## Overview

Stable-TS is a fork of OpenAI's Whisper that focuses on generating more accurate timestamps and stable transcriptions. It includes several improvements:

- Better word-level timestamps
- Voice activity detection (VAD) for improved silence handling
- Silence suppression for cleaner transcripts
- Support for Apple Silicon via MLX acceleration (automatically uses whisper-large-v3-turbo model)

## Installation

Before using stable-ts, you need to install it and its dependencies:

1. Run the installation script:
   ```bash
   python install_stable_ts.py
   ```

   This script will:
   - Install stable-whisper from the local directory or from PyPI
   - Install required dependencies (torch, librosa, etc.)
   - Install MLX support for Apple Silicon devices (if applicable)

2. If the installation script fails, you can manually install the required packages:
   ```bash
   # Option 1: Install from local directory
   pip install -e ./stable-ts

   # Option 2: Install directly from GitHub (recommended)
   pip install git+https://github.com/jianfch/stable-ts.git

   # Install other dependencies
   pip install torch librosa rich numpy ffmpeg-python

   # For Apple Silicon users
   pip install mlx mlx-whisper
   ```

   Note: Installing from GitHub is recommended as it ensures you get the latest version with all bug fixes.

## Usage

To use stable-ts in VideoLingo:

1. Set the `whisper.runtime` option to `stable-ts` in the UI settings or in your `config.yaml` file:
   ```yaml
   whisper:
     runtime: 'stable-ts'
   ```

2. Run the transcription process as usual.

## Features

- **Improved Timestamps**: stable-ts provides more accurate word-level timestamps compared to standard WhisperX
- **Better Silence Handling**: Automatically detects and suppresses silence for cleaner transcripts
- **Apple Silicon Support**: Uses MLX acceleration on Apple Silicon devices for faster processing
  - Automatically uses the optimized `whisper-large-v3-turbo` model on Apple Silicon
  - Up to 2-4x faster transcription compared to standard WhisperX
- **Compatible Output**: Generates output in the same format as WhisperX for seamless integration

## Requirements

The stable-ts integration requires the following dependencies:

- stable-whisper
- torch
- librosa
- ffmpeg

For Apple Silicon users, it will automatically use MLX acceleration when available.

## Technical Details

The integration works by:

1. Extracting audio segments from the input file
2. Processing each segment with stable-ts
3. Converting the stable-ts output format to match WhisperX's format
4. Combining the results into a single transcript

The implementation is located in `core/all_whisper_methods/stable_ts_local.py`.
