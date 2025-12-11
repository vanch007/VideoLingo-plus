"""
VoxCPM TTS Module for VideoLingo-plus
Integrates with local VoxCPM Gradio API for dynamic voice cloning TTS.
"""
import os
import sys
from pathlib import Path

# 使用统一的路径设置（替代 sys.path.append）
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.config_utils import load_key
from core.all_tts_functions.tts_utils import (
    get_reference_audio_path,
    get_prompt_text_from_df,
    ensure_output_dir
)
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
    
    # 使用 tts_utils 获取参考音频（自动处理 fallback）
    spk_audio_prompt, fallback_number = get_reference_audio_path(number)
    
    # 确定要用哪个编号获取原始文本
    ref_number = fallback_number if fallback_number is not None else number
    
    # 使用 tts_utils 获取原始文本
    prompt_text = get_prompt_text_from_df(task_df, ref_number, fallback_text=text)

    # Load configuration options
    use_prompt_enhancement = load_key("voxcpm_tts.use_prompt_enhancement", False)
    normalize = load_key("voxcpm_tts.normalize", False)
    denoise = load_key("voxcpm_tts.denoise", False)

    # 确保输出目录存在
    ensure_output_dir(save_as)

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
