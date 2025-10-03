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

import random

def indextts2_tts_for_videolingo(text, save_as, number, task_df, clone_mode="dynamic", fixed_voice_name=None, use_emo_text=False, emo_text=None, verbose=True, speed=1.0):
    """
    使用 index-tts2 进行 TTS 转换，支持参考音频、文本情感等高级功能
    """
    API_KEY = load_key("sf_indextts2.api_key")

    if clone_mode == "fixed" and fixed_voice_name:
        # 固定克隆模式
        voice_dir = Path(__file__).parent / "voice" / fixed_voice_name
        if not voice_dir.is_dir():
            raise ValueError(f"指定的固定克隆声音 '{fixed_voice_name}' 不存在于: {voice_dir}")
        
        ref_audios = list(voice_dir.glob("*.wav"))
        if not ref_audios:
            raise ValueError(f"在声音目录 '{voice_dir}' 中没有找到任何 .wav 参考音频")
        
        # 随机选择一个参考音频
        ref_audio_path = random.choice(ref_audios)
        prompt_text = ref_audio_path.stem  # 文件名（不带扩展名）即为参考文本
        print(f"使用固定克隆声音: {fixed_voice_name}, 参考音频: {ref_audio_path.name}")
    else:
        # 动态克隆模式（原始逻辑）
        prompt_text = task_df.loc[task_df['number'] == number, 'origin'].values[0]
        current_dir = Path.cwd()
        ref_audio_path = current_dir / f"output/audio/refers/{number}.wav"
        
        if not ref_audio_path.exists():
            ref_audio_path = current_dir / "output/audio/refers/1.wav"
            if not ref_audio_path.exists():
                try:
                    from core.step9_extract_refer_audio import extract_refer_audio_main
                    print(f"参考音频文件不存在，尝试提取: {ref_audio_path}")
                    extract_refer_audio_main()
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
    
    # 将文件扩展名从.wav改为.mp3，因为API实际返回的是MP3格式
    if save_path.suffix.lower() == '.wav':
        mp3_save_path = save_path.with_suffix('.mp3')
    else:
        mp3_save_path = save_path

    # 构建请求体
    extra_body = {
        "references": [
            {
                "audio": f"data:audio/wav;base64,{reference_base64}",
                "text": prompt_text
            }
        ],
        "verbose": verbose,
        "speed": speed
    }

    # 如果启用了文本情感，则添加到请求体
    if use_emo_text and emo_text:
        extra_body["use_emo_text"] = True
        extra_body["emo_text"] = emo_text

    with client.audio.speech.with_streaming_response.create(
        model="IndexTeam/IndexTTS-2",
        voice="",
        input=text,
        response_format="mp3",  # 改为mp3格式
        extra_body=extra_body
    ) as response:
        response.stream_to_file(mp3_save_path)
    
    print(f"音频已成功保存至: {mp3_save_path}")
    return True