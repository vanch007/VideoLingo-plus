from openai import OpenAI
from pathlib import Path
import base64
import os
import sys
import random

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from core.config_utils import load_key
from core.all_tts_functions.tts_utils import (
    get_reference_audio_path,
    get_prompt_text_from_df,
    ensure_output_dir,
    extract_refer_audio_if_needed
)



def wav_to_base64(wav_file_path):
    """将 WAV 文件转换为 base64 编码"""
    with open(wav_file_path, 'rb') as audio_file:
        audio_content = audio_file.read()
    return base64.b64encode(audio_content).decode('utf-8')


def indextts2_tts_for_videolingo(
    text, save_as, number, task_df, 
    clone_mode="dynamic", fixed_voice_name=None, 
    use_emo_text=False, emo_text=None, 
    verbose=True, speed=1.0, target_duration=None
):
    """
    使用 index-tts2 进行 TTS 转换，支持参考音频、文本情感等高级功能
    
    Args:
        text: 要合成的文本
        save_as: 保存路径
        number: 片段编号
        task_df: 包含原始文本的任务 DataFrame
        clone_mode: "dynamic" (动态克隆) 或 "fixed" (固定声音)
        fixed_voice_name: 固定模式下的声音名称
        use_emo_text: 是否使用文本情感
        emo_text: 情感文本
        verbose: 详细输出
        speed: 语速
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
        # 动态克隆模式 - 使用增强版 fallback 逻辑
        ref_audio_path, fallback_num = get_reference_audio_path(number)
        
        # 获取参考文本（使用实际的参考音频编号）
        actual_number = fallback_num if fallback_num else number
        prompt_text = get_prompt_text_from_df(task_df, actual_number, fallback_text=text)
        
        if fallback_num:
            print(f"使用备选参考音频 {fallback_num}.wav (原: {number}.wav)")


    
    # 转换参考音频为 base64
    reference_base64 = wav_to_base64(ref_audio_path)
    
    client = OpenAI(
        api_key=API_KEY,
        base_url="https://api.siliconflow.cn/v1"
    )

    save_path = Path(save_as)
    ensure_output_dir(save_as)
    
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
    if target_duration is not None:
        extra_body["target_duration"] = target_duration

    # 如果启用了文本情感，则添加到请求体
    if use_emo_text and emo_text:
        extra_body["use_emo_text"] = True
        extra_body["emo_text"] = emo_text

    # Use a temporary path for the MP3 download
    temp_mp3_path = save_path.with_suffix('.mp3')

    with client.audio.speech.with_streaming_response.create(
        model="IndexTeam/IndexTTS-2",
        voice="",
        input=text,
        response_format="mp3",
        extra_body=extra_body
    ) as response:
        response.stream_to_file(temp_mp3_path)
    
    # If the requested save path is .wav, convert the MP3 to WAV
    if save_path.suffix.lower() == '.wav':
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_mp3(temp_mp3_path)
            audio.export(save_path, format="wav")
            # Optionally remove the temp mp3 file
            if temp_mp3_path != save_path:
                os.remove(temp_mp3_path)
            print(f"音频已成功保存并转换为: {save_path}")
        except Exception as e:
            print(f"转换 MP3 到 WAV 失败: {e}")
            # Fallback: keep the mp3 file
            print(f"保留 MP3 文件: {temp_mp3_path}")
    else:
        # If not .wav (e.g. .mp3), and we saved to temp_mp3_path which is .mp3
        # If save_path was .mp3, temp_mp3_path is same as save_path
        if temp_mp3_path != save_path:
            os.rename(temp_mp3_path, save_path)
        print(f"音频已成功保存至: {save_path}")

    return True
