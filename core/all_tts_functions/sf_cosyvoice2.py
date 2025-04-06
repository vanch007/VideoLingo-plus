from openai import OpenAI
from pathlib import Path
import base64
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from core.config_utils import load_key

def wav_to_base64(wav_file_path):
    with open(wav_file_path, 'rb') as audio_file:
        audio_content = audio_file.read()
    base64_audio = base64.b64encode(audio_content).decode('utf-8')
    return base64_audio

def cosyvoice_tts_for_videolingo(text, save_as, number, task_df):
    """
    使用 CosyVoice 进行 TTS 转换，支持参考音频
    """
    prompt_text = task_df.loc[task_df['number'] == number, 'origin'].values[0]
    API_KEY = load_key("sf_cosyvoice2.api_key")
    # 设置参考音频路径 (先尝试带line_index后缀的格式)
    current_dir = Path.cwd()
    ref_audio_path_new = current_dir / f"output/audio/refers/{number}_0.wav"  # 假设line_index=0
    ref_audio_path_old = current_dir / f"output/audio/refers/{number}.wav"
    
    # 优先使用新格式，其次旧格式
    if ref_audio_path_new.exists():
        ref_audio_path = ref_audio_path_new
    elif ref_audio_path_old.exists():
        ref_audio_path = ref_audio_path_old
    else:
        # 如果都不存在，尝试提取参考音频
        try:
            from core.step9_extract_refer_audio import extract_refer_audio_main
            print(f"参考音频文件不存在，尝试提取: {ref_audio_path_new}")
            extract_refer_audio_main()
            # 再次检查新格式是否存在
            if ref_audio_path_new.exists():
                ref_audio_path = ref_audio_path_new
            elif ref_audio_path_old.exists():
                ref_audio_path = ref_audio_path_old
            else:
                raise Exception("无法获取参考音频")
        except Exception as e:
            print(f"提取参考音频失败: {str(e)}")
            raise

    # 转换参考音频为 base64
    reference_base64 = wav_to_base64(ref_audio_path)
    
    client = OpenAI(
        api_key=API_KEY,
        base_url="https://api.siliconflow.cn/v1"
    )

    save_path = Path(save_as)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with client.audio.speech.with_streaming_response.create(
        model="FunAudioLLM/CosyVoice2-0.5B",
        voice="",
        input=text,
        response_format="wav",
        extra_body={
            "references": [
                {
                    "audio": f"data:audio/wav;base64,{reference_base64}",
                    "text": prompt_text
                }
            ]
        }
    ) as response:
        response.stream_to_file(save_path)
    
    print(f"音频已成功保存至: {save_path}")
    return True
