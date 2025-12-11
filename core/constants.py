"""
项目常量定义

集中定义所有文件路径常量，避免在各模块中重复定义。

使用方法：
    from core.constants import SRC_SRT, TRANS_SRT, TTS_TASKS_FILE
"""

# ========== 输出目录 ==========
OUTPUT_DIR = "output"
LOG_DIR = "output/log"
AUDIO_DIR = "output/audio"
REFERS_DIR = "output/audio/refers"
SEGS_DIR = "output/audio/segs"
TEMP_DIR = "output/audio/temp"

# ========== 日志/中间文件 ==========
CLEANED_CHUNKS_FILE = "output/log/cleaned_chunks.xlsx"
SENTENCE_SPLIT_NLP_FILE = "output/log/sentence_splitbynlp.txt"
SENTENCE_SPLIT_MEANING_FILE = "output/log/sentence_splitbymeaning.txt"
TERMINOLOGY_FILE = "output/log/terminology.json"
TRANSLATION_RESULTS_FILE = "output/log/translation_results.xlsx"
TRANSLATION_FOR_SUBTITLES_FILE = "output/log/translation_results_for_subtitles.xlsx"
STEP_TIMINGS_FILE = "output/log/step_timings.json"
GPT_LOG_DIR = "output/gpt_log"

# ========== 音频文件 ==========
RAW_AUDIO_FILE = "output/audio/raw_audio.mp3"
VOCAL_AUDIO_FILE = "output/audio/vocals.mp3"
WHISPER_AUDIO_FILE = "output/audio/for_whisper.mp3"
ENHANCED_VOCAL_FILE = "output/audio/enhanced_vocals.mp3"
DUB_VOCAL_FILE = "output/dub.mp3"
FULL_AUDIO_FILE = "output/audio/full_audio.wav"

# ========== 任务文件 ==========
TTS_TASKS_FILE = "output/audio/tts_tasks.xlsx"
AUDIO_TASK_FILE = "output/audio/audio_task.xlsx"
DUB_CHUNKS_FILE = "output/audio/dub_chunks.xlsx"

# ========== 字幕文件 ==========
SRC_SRT = "output/src.srt"
TRANS_SRT = "output/trans.srt"
DUB_SRT = "output/dub.srt"
SRC_SUBS_FOR_AUDIO_FILE = "output/audio/src_subs_for_audio.srt"
TRANS_SUBS_FOR_AUDIO_FILE = "output/audio/trans_subs_for_audio.srt"

# ========== 输出视频 ==========
SUB_VIDEO = "output/AI字幕.mp4"
DUB_VIDEO = "output/AI配音.mp4"

# ========== TTS 相关常量 ==========
MIN_VALID_AUDIO_SIZE = 20000  # 最小有效音频文件大小 (bytes)
DEFAULT_WARMUP_SIZE = 5  # TTS 预热批次大小

# ========== 模型缓存目录 ==========
MODEL_CACHE_DIR = "_model_cache"
