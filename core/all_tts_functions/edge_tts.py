from pathlib import Path
import os, sys, time
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from core.config_utils import load_key

# Cloudflare Edge TTS Worker API (DIYgod/cloudflare-edge-tts)
# Local: http://127.0.0.1:5566/tts
# Contract: POST /tts, Content-Type: application/json, body: {"text": "...", "voice": "..."}
# Returns: audio/mpeg stream

CLOUDFLARE_EDGE_TTS_URL = os.environ.get("CLOUDFLARE_EDGE_TTS_URL", "http://127.0.0.1:5566/tts")
MAX_RETRIES = int(os.environ.get("CLOUDFLARE_EDGE_TTS_RETRIES", "1"))

def _convert_mp3_to_wav(mp3_path, wav_path):
    import subprocess
    subprocess.run(
        ["ffmpeg", "-y", "-v", "warning", "-i", str(mp3_path), str(wav_path)],
        check=True,
    )


def _native_edge_tts(text, voice, save_path):
    import asyncio
    import tempfile
    import edge_tts as native_edge_tts

    async def synthesize():
        communicate = native_edge_tts.Communicate(text, voice)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            temp_mp3 = f.name
        await communicate.save(temp_mp3)
        try:
            _convert_mp3_to_wav(temp_mp3, save_path)
        finally:
            if os.path.exists(temp_mp3):
                os.remove(temp_mp3)

    asyncio.run(synthesize())


def edge_tts(text, save_path):
    """Synthesize speech via cloudflare-edge-tts Worker, fallback to native edge-tts, and save as WAV."""
    import requests
    import tempfile

    # Load voice from config
    edge_set = load_key("edge_tts")
    voice = edge_set.get("voice", "en-US-AvaMultilingualNeural")

    # Create output directory
    speech_file_path = Path(save_path)
    speech_file_path.parent.mkdir(parents=True, exist_ok=True)

    # Call cloudflare-edge-tts Worker with retry
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(
                CLOUDFLARE_EDGE_TTS_URL,
                json={"text": text, "voice": voice},
                headers={"Content-Type": "application/json"},
                timeout=60,
            )
            resp.raise_for_status()

            # Save MP3 to temp file
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                f.write(resp.content)
                temp_mp3 = f.name

            _convert_mp3_to_wav(temp_mp3, speech_file_path)
            os.remove(temp_mp3)
            print(f"Audio saved to {speech_file_path}")
            return
        except Exception as e:
            last_err = e
            print(f"⚠️ edge_tts attempt {attempt}/{MAX_RETRIES} failed: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(5 * attempt)

    print(f"⚠️ cloudflare-edge-tts unavailable after {MAX_RETRIES} retries: {last_err}")
    print("↪ Falling back to native edge-tts package.")
    _native_edge_tts(text, voice, speech_file_path)
    print(f"Audio saved to {speech_file_path}")

if __name__ == "__main__":
    edge_tts("Xin chào, thế giới!", "edge_tts_test.wav")
