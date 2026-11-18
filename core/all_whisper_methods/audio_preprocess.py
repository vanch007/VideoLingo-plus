import os, sys, subprocess
import pandas as pd
from typing import Dict, List, Tuple
from rich import print
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from core.config_utils import update_key

AUDIO_DIR = "output/audio"
RAW_AUDIO_FILE = "output/audio/raw.mp3"
RAW_MASTER_AUDIO_FILE = "output/audio/raw_master.wav"
CLEANED_CHUNKS_EXCEL_PATH = "output/log/cleaned_chunks.xlsx"
# Follow the container presentation clock: packet-timestamp gaps/overlaps must not stretch the
# decoded timeline, or subtitles drift later and later.
ALIGN_FILTER = "aresample=async=1:first_pts=0"

def compress_audio(input_file: str, output_file: str):
    """将输入音频文件压缩为低质量音频文件，用于转录"""
    if not os.path.exists(output_file):
        print(f"🗜️ Converting to low quality audio with FFmpeg ......")
        # 16000 Hz, 1 channel, (Whisper default) , 96kbps to keep more details as well as smaller file size
        subprocess.run([
            'ffmpeg', '-y', '-i', input_file, '-vn', '-af', ALIGN_FILTER, '-b:a', '96k',
            '-ar', '16000', '-ac', '1', '-metadata', 'encoding=UTF-8',
            '-f', 'mp3', output_file
        ], check=True, stderr=subprocess.PIPE)
        print(f"🗜️ Converted <{input_file}> to <{output_file}> with FFmpeg")
    return output_file

def convert_video_to_audio(video_file: str):
    os.makedirs(AUDIO_DIR, exist_ok=True)
    raw_master = os.path.join(AUDIO_DIR, "raw_master.wav")
    if not os.path.exists(raw_master):
        print(f"🎬➡️🎵 Extracting lossless master audio with FFmpeg ......")
        subprocess.run([
            'ffmpeg', '-y', '-i', video_file, '-vn',
            '-af', ALIGN_FILTER,
            '-c:a', 'pcm_s16le',
            raw_master
        ], check=True, stderr=subprocess.PIPE)
        print(f"🎬➡️🎵 Extracted lossless master <{raw_master}>\n")

    if not os.path.exists(RAW_AUDIO_FILE):
        print(f"🎬➡️🎵 Converting to ASR analysis audio with FFmpeg ......")
        input_audio = raw_master if os.path.exists(raw_master) else video_file
        subprocess.run([
            'ffmpeg', '-y', '-i', input_audio, '-vn',
            '-af', ALIGN_FILTER,
            '-c:a', 'libmp3lame', '-b:a', '128k',
            '-ar', '16000',
            '-ac', '1',
            '-metadata', 'encoding=UTF-8', RAW_AUDIO_FILE
        ], check=True, stderr=subprocess.PIPE)
        print(f"🎬➡️🎵 Converted <{video_file}> to <{RAW_AUDIO_FILE}> with FFmpeg\n")

def _detect_silence(audio_file: str, start: float, end: float) -> List[float]:
    """Detect silence points in the given audio segment"""
    cmd = ['ffmpeg', '-y', '-i', audio_file,
           '-ss', str(start), '-to', str(end),
           '-af', 'silencedetect=n=-30dB:d=0.5',
           '-f', 'null', '-']

    output = subprocess.run(cmd, capture_output=True, text=True,
                          encoding='utf-8').stderr

    return [float(line.split('silence_end: ')[1].split(' ')[0])
            for line in output.split('\n')
            if 'silence_end' in line]

def get_audio_duration(audio_file: str) -> float:
    """Get the duration of an audio file using ffmpeg."""
    # Check if file exists and has valid size
    if not os.path.exists(audio_file):
        print(f"[red]❌ Error: Audio file does not exist: {audio_file}[/red]")
        return 0

    file_size = os.path.getsize(audio_file)
    if file_size < 1000:  # Less than 1KB is likely invalid
        print(f"[red]❌ Error: Audio file too small ({file_size} bytes): {audio_file}[/red]")
        return 0

    cmd = ['ffmpeg', '-i', audio_file]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    _, stderr = process.communicate()
    output = stderr.decode('utf-8', errors='ignore')

    try:
        duration_str = [line for line in output.split('\n') if 'Duration' in line]
        if not duration_str:  # No duration found
            print(f"[red]❌ Error: No duration information found in file: {audio_file}[/red]")
            return 0

        duration_str = duration_str[0]
        if 'N/A' in duration_str:  # Handle N/A duration
            print(f"[red]❌ Error: Duration is N/A for file: {audio_file}[/red]")
            return 0

        duration_parts = duration_str.split('Duration: ')[1].split(',')[0].split(':')
        duration = float(duration_parts[0])*3600 + float(duration_parts[1])*60 + float(duration_parts[2])

        # Sanity check - if duration is unreasonably long or short
        if duration > 3600 or duration <= 0:  # More than 1 hour or negative/zero
            print(f"[yellow]⚠️ Warning: Unusual duration ({duration:.2f}s) for file: {audio_file}[/yellow]")
    except Exception as e:
        print(f"[red]❌ Error: Failed to get audio duration: {e}[/red]")
        duration = 0
    return duration


FRAME_SECONDS = 0.05
SPEECH_DBFS = -42.0  # frames above this level count as active (speech or loud background)


def _frame_levels(audio_file: str, start: float, end: float):
    """Per-frame RMS in dBFS (16 kHz mono decode of [start, end]); None if unreadable."""
    import numpy as np
    proc = subprocess.run(
        ['ffmpeg', '-v', 'error', '-ss', str(start), '-to', str(end), '-i', audio_file,
         '-vn', '-ac', '1', '-ar', '16000', '-f', 's16le', '-'],
        capture_output=True)
    if proc.returncode != 0 or not proc.stdout:
        return None
    pcm = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    size = int(16000 * FRAME_SECONDS)
    frames = len(pcm) // size
    if frames < 2:
        return None
    rms = np.sqrt((pcm[:frames * size].reshape(frames, size) ** 2).mean(axis=1) + 1e-12)
    return 20 * np.log10(rms)


def active_seconds(audio_file: str, start: float, end: float) -> float | None:
    """Seconds of [start, end] louder than SPEECH_DBFS; None when it cannot be measured."""
    levels = _frame_levels(audio_file, start, end)
    if levels is None:
        return None
    return float((levels > SPEECH_DBFS).sum()) * FRAME_SECONDS


def find_cut_point(audio_file: str, start: float, end: float):
    """Quietest moment in [start, end], smoothed over ~0.4 s so a lone gap between words in
    running speech does not beat a real pause. Works with background music, where
    silencedetect finds no silence at all. Returns None when the audio cannot be read."""
    import numpy as np
    levels = _frame_levels(audio_file, start, end)
    if levels is None:
        return None
    width = max(1, int(0.4 / FRAME_SECONDS))
    smooth = np.convolve(levels, np.ones(width) / width, mode='same')
    margin = width  # keep cuts away from the window edges
    if len(smooth) <= 2 * margin:
        return None
    index = margin + int(np.argmin(smooth[margin:len(smooth) - margin]))
    return start + index * FRAME_SECONDS

def split_audio(audio_file: str, target_len: int = 25, win: int = 5) -> List[Tuple[float, float]]:
    # Natural dialogue segments (15-25s) matching Whisper's 30s attention window
    print("[bold blue]🔪 Starting audio segmentation...[/]")

    duration = get_audio_duration(audio_file)

    segments = []
    pos = 0
    while pos < duration:
        if duration - pos < target_len:
            segments.append((pos, duration))
            break
        win_start = pos + target_len - win
        win_end = min(win_start + 2 * win, duration)
        silences = _detect_silence(audio_file, win_start, win_end)

        if silences:
            target_pos = target_len - (win_start - pos)
            split_at = next((t for t in silences if t - win_start > target_pos), None)
            if split_at:
                segments.append((pos, split_at))
                pos = split_at
                continue
        # No silence in the window (background music): cut at its quietest moment instead
        # of the arbitrary target length, which lands inside a word.
        valley = find_cut_point(audio_file, win_start, win_end)
        cut = valley if valley is not None and valley > pos + 1.0 else pos + target_len
        segments.append((pos, cut))
        pos = cut

    print(f"🔪 Audio split into {len(segments)} segments")
    return segments

def process_transcription(result: Dict) -> pd.DataFrame:
    all_words = []
    for segment in result['segments']:
        # Get speaker for the segment if available
        segment_speaker = segment.get('speaker', None)

        if 'words' in segment and segment['words']:
            for word in segment['words']:
                # Use word-level speaker if available, otherwise fallback to segment speaker
                current_speaker = word.get('speaker', segment_speaker)

                if 'start' not in word and 'end' not in word:
                    if all_words:
                        # Assign the end time of the previous word as the start and end time of the current word
                        word_dict = {
                            'text': word["word"].strip(),
                            'start': all_words[-1]['end'],
                            'end': all_words[-1]['end'],
                            'speaker': current_speaker
                        }
                        all_words.append(word_dict)
                    else:
                        # If it's the first word, look next for a timestamp then assign it to the current word
                        next_word = next((w for w in segment['words'] if 'start' in w and 'end' in w), None)
                        if next_word:
                            word_dict = {
                                'text': word["word"].strip(),
                                'start': next_word["start"],
                                'end': next_word["end"],
                                'speaker': current_speaker
                            }
                            all_words.append(word_dict)
                        else:
                            raise Exception(f"No next word with timestamp found for the current word : {word}")
                else:
                    # Normal case, with start and end times
                    word_dict = {
                        'text': f'{word["word"].strip()}',
                        'start': word.get('start', all_words[-1]['end'] if all_words else 0),
                        'end': word['end'],
                        'speaker': current_speaker
                    }

                    all_words.append(word_dict)
                for key in ("recognition_segment_id", "recognition_text", "recognition_start", "recognition_end"):
                    if key in word:
                        all_words[-1][key] = word[key]
        elif 'text' in segment:
            word_dict = {
                'text': segment['text'].strip(),
                'start': segment['start'],
                'end': segment['end'],
                'speaker': segment_speaker
            }
            all_words.append(word_dict)

    return pd.DataFrame(all_words)

def save_results(df: pd.DataFrame):
    os.makedirs('output/log', exist_ok=True)

    # Remove rows where 'text' is empty or only isolated punctuation
    from core.source_verification import normalize_transcript
    initial_rows = len(df)
    df = df[df['text'].astype(str).apply(lambda x: len(normalize_transcript(x.strip().strip('"'))) > 0)]
    removed_rows = initial_rows - len(df)
    if removed_rows > 0:
        print(f"ℹ️ Removed {removed_rows} row(s) with empty or punctuation-only text.")

    # Normalize spaces in text
    df['text'] = df['text'].apply(lambda x: ' '.join(x.split()))

    # Token length is not evidence of hallucination. Aligned CJK phrases,
    # URLs and technical names must survive ASR export intact.

    df['text'] = df['text'].apply(lambda x: f'"{x}"')

    # Ensure speaker column exists if not present
    if 'speaker' not in df.columns:
        df['speaker'] = None

    # Ensure word timestamps are strictly monotonic and non-overlapping
    if "start" in df.columns and "end" in df.columns:
        df["start"] = df["start"].astype(float)
        df["end"] = df["end"].astype(float)
        for i in range(1, len(df)):
            prev_end = df.iat[i - 1, df.columns.get_loc("end")]
            curr_start = df.iat[i, df.columns.get_loc("start")]
            if curr_start < prev_end - 0.001:
                prev_start = df.iat[i - 1, df.columns.get_loc("start")]
                curr_end = df.iat[i, df.columns.get_loc("end")]
                # Smooth overlapping word boundaries monotonically
                if curr_end > prev_start + 0.02:
                    mid = round((prev_end + curr_start) / 2.0, 4)
                    mid = max(prev_start + 0.01, min(curr_end - 0.01, mid))
                    df.iat[i - 1, df.columns.get_loc("end")] = mid
                    df.iat[i, df.columns.get_loc("start")] = mid
                else:
                    df.iat[i, df.columns.get_loc("start")] = prev_end
                    df.iat[i, df.columns.get_loc("end")] = max(prev_end + 0.01, curr_end)

    df.to_excel(CLEANED_CHUNKS_EXCEL_PATH, index=False)
    print(f"📊 Excel file saved to {CLEANED_CHUNKS_EXCEL_PATH}")


def save_sentences_from_segments(result: dict, output_file: str = "output/log/sentence_splitbynlp.txt"):
    """
    从 ASR 结果的 segments 直接生成分句文件，跳过 spacy 分句步骤。

    适用于 stable-ts 等已经能按句子分割且保留词级时间戳的 ASR 引擎。

    Args:
        result: ASR 返回的结果字典，包含 segments 列表
        output_file: 输出的分句文件路径
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    sentences = []
    for segment in result.get('segments', []):
        text = segment.get('text', '').strip()
        if text:
            # 清理文本：去除多余空格
            text = ' '.join(text.split())
            sentences.append(text)

    if sentences:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(sentences))
        print(f"[green]📝 直接输出分句文件 ({len(sentences)} 句) → {output_file}[/green]")
        print(f"[cyan]💡 已跳过 spacy 分句步骤，ASR 模型已完成句子分割[/cyan]")
        return True
    return False


def save_language(language: str):
    update_key("whisper.detected_language", language)
