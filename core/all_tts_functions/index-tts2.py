import requests
import json
import os
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.config_utils import load_key

# The URL of your local index-tts service
BASE_URL = "http://127.0.0.1:8000/tts"

def custom_tts(text: str, save_as: str, number: int, task_df, attempt: int):
    """
    Calls the local index-tts service to generate audio using dynamic reference audio.
    It uses different payloads for different retry attempts.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    spk_audio_prompt = os.path.join(project_root, 'output', 'audio', 'refers', f'{number}.wav')

    if not os.path.exists(spk_audio_prompt):
        raise FileNotFoundError(f"Reference audio not found for number {number} at {spk_audio_prompt}")
    if os.path.getsize(spk_audio_prompt) < 20000:
        raise ValueError(f"Reference audio for number {number} is too small and likely invalid.")

    # Default payload: use audio for both speaker and emotion
    payload = {
        "text": text,
        "spk_audio_prompt": spk_audio_prompt,
        "emo_audio_prompt": spk_audio_prompt,
        "output_path": "outputs/api_generated.wav",
        "verbose": True
    }
    headers = {'Content-Type': 'application/json'}
    Path(save_as).parent.mkdir(parents=True, exist_ok=True)

    print(f"Calling custom_tts API for text: '{text}' with audio emotion payload")

    try:
        response = requests.post(BASE_URL, headers=headers, data=json.dumps(payload))
        
        if response.status_code == 500:
            print("[yellow]TTS failed with audio emotion (500 error), retrying with text emotion...[/yellow]")
            payload = {
                "text": text,
                "spk_audio_prompt": spk_audio_prompt,
                "output_path": "outputs/api_generated.wav",
                "use_emo_text": True,
                "emo_text": text,
                "verbose": True
            }
            print(f"Retrying custom_tts API for text: '{text}' with text emotion payload")
            response = requests.post(BASE_URL, headers=headers, data=json.dumps(payload))

        if response.status_code == 200:
            with open(save_as, 'wb') as f:
                f.write(response.content)
            print(f"Custom TTS audio successfully saved to {save_as}")
        else:
            print(f"Error calling custom_tts API: {response.status_code}")
            print(f"Response: {response.text}")
            raise Exception(f"Custom TTS API returned non-200 status code: {response.status_code}. Response: {response.text}")

    except requests.exceptions.RequestException as e:
        print(f"Failed to connect to custom_tts service at {BASE_URL}. Please ensure the service is running.")
        raise e
    except Exception as e:
        print(f"An error occurred during custom_tts generation: {e}")
        raise

if __name__ == '__main__':
    # This is a test block to run the function directly for debugging
    print("Running custom_tts test...")
    
    # Before running, make sure you have a config file that `load_key` can read from,
    # and it contains the [custom_tts] section with `spk_audio_prompt`.
    # For this test, we'll mock the config dependency or assume it's configured.
    
    test_text = "你好，这是一个通过自定义TTS API生成的语音。"
    test_save_path = "test_custom_tts.wav"
    
    # You would need to ensure your config is set up for this to run.
    # For example, create a temporary config.toml with:
    # [custom_tts]
    # spk_audio_prompt = "examples/voice_01.wav"
    
    try:
        custom_tts(test_text, test_save_path)
        print(f"Test successful. Audio saved to {test_save_path}")
    except Exception as e:
        print(f"Test failed: {e}")
        print("Please ensure your local TTS service is running and the 'custom_tts.spk_audio_prompt' is set in your config.")
