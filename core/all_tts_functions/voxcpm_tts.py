"""
VoxCPM TTS Module for VideoLingo-plus
Integrates with local VoxCPM Gradio API for dynamic voice cloning TTS.
"""
import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.config_utils import load_key
from rich import print as rprint

# Default VoxCPM API endpoint
DEFAULT_BASE_URL = "http://127.0.0.1:7860"

def get_base_url():
    """Get VoxCPM API base URL from config or use default"""
    try:
        return load_key("voxcpm_tts.api_url")
    except (KeyError, TypeError):
        return DEFAULT_BASE_URL


def voxcpm_tts(text: str, save_as: str, number: int, task_df, attempt: int = 0):
    """
    Calls the local VoxCPM service to generate audio using dynamic reference audio.
    Similar to gpt_sovits, uses per-segment reference audio and original text for voice cloning.
    
    Args:
        text: Text to synthesize (translated text)
        save_as: Output file path
        number: Segment number to find corresponding reference audio
        task_df: Task dataframe containing 'origin' column with original text
        attempt: Retry attempt number
    """
    from gradio_client import Client, handle_file
    import shutil
    
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    refers_dir = os.path.join(project_root, 'output', 'audio', 'refers')
    spk_audio_prompt = os.path.join(refers_dir, f'{number}.wav')
    
    # Minimum valid audio file size (20KB)
    MIN_AUDIO_SIZE = 20000
    use_fallback = False
    fallback_number = None

    # Check if the reference audio is valid, otherwise use fallback
    if not os.path.exists(spk_audio_prompt) or os.path.getsize(spk_audio_prompt) < MIN_AUDIO_SIZE:
        rprint(f"[yellow]⚠️ Reference audio for {number} is invalid, searching for fallback...[/yellow]")
        use_fallback = True
        
        # Find the first valid reference audio
        for i in range(1, 100):  # Check first 100 segments
            fallback_path = os.path.join(refers_dir, f'{i}.wav')
            if os.path.exists(fallback_path) and os.path.getsize(fallback_path) >= MIN_AUDIO_SIZE:
                spk_audio_prompt = fallback_path
                fallback_number = i
                rprint(f"[yellow]📌 Using fallback reference audio: {i}.wav[/yellow]")
                break
        else:
            raise FileNotFoundError(f"No valid reference audio found in {refers_dir}")

    # Get prompt_text from task_df (original text corresponding to the reference audio)
    # This is required for voice cloning - the model needs to know what's being said in the reference
    # Use fallback_number if we're using a fallback audio, otherwise use the original number
    ref_number = fallback_number if use_fallback else number
    try:
        prompt_text = task_df.loc[task_df['number'] == ref_number, 'origin'].values[0]
        if not prompt_text or len(prompt_text.strip()) == 0:
            raise ValueError("Empty prompt_text")
    except (KeyError, IndexError, ValueError, TypeError) as e:
        rprint(f"[yellow]⚠️ Could not get original text for number {ref_number}, using synthesized text as fallback[/yellow]")
        prompt_text = text  # Fallback to the text being synthesized

    # Load configuration options
    try:
        use_prompt_enhancement = load_key("voxcpm_tts.use_prompt_enhancement")
    except (KeyError, TypeError):
        use_prompt_enhancement = False
    
    try:
        normalize = load_key("voxcpm_tts.normalize")
    except (KeyError, TypeError):
        normalize = False
        
    try:
        denoise = load_key("voxcpm_tts.denoise")
    except (KeyError, TypeError):
        denoise = False

    Path(save_as).parent.mkdir(parents=True, exist_ok=True)

    rprint(f"[cyan]🎙️ VoxCPM: '{text[:40]}...' (ref: '{prompt_text[:30]}...')[/cyan]")

    try:
        base_url = get_base_url()
        client = Client(base_url)
        
        # Call the voice cloning API endpoint
        # API: /generate_speech_1
        # Parameters: text, prompt_audio, prompt_text, use_prompt_enhancement, normalize, denoise
        result = client.predict(
            text=text,
            prompt_audio=handle_file(spk_audio_prompt),
            prompt_text=prompt_text,  # Original text from the reference audio
            use_prompt_enhancement=use_prompt_enhancement,
            normalize=normalize,
            denoise=denoise,
            api_name="/generate_speech_1"
        )
        
        # Result is (audio_file_path, status_text)
        if result and len(result) >= 1:
            audio_path = result[0]
            status = result[1] if len(result) > 1 else ""
            
            if audio_path is None:
                raise Exception(f"VoxCPM returned error: {status}")
            
            if isinstance(audio_path, str) and os.path.exists(audio_path):
                # Copy the generated audio to the target path
                shutil.copy2(audio_path, save_as)
                rprint(f"[green]✅ VoxCPM saved: {save_as}[/green]")
            else:
                raise Exception(f"Invalid audio path returned: {audio_path}")
        else:
            raise Exception(f"Unexpected result format: {result}")

    except Exception as e:
        rprint(f"[red]❌ VoxCPM error: {e}[/red]")
        raise


if __name__ == '__main__':
    # Test block
    print("Running VoxCPM TTS test...")
    test_text = "你好，这是一个通过VoxCPM语音克隆API生成的语音测试。"
    test_save_path = "test_voxcpm_tts.wav"
    
    try:
        voxcpm_tts(test_text, test_save_path, 0, None)
        print(f"Test successful. Audio saved to {test_save_path}")
    except Exception as e:
        print(f"Test failed: {e}")
        print("Please ensure VoxCPM service is running and reference audio exists.")

