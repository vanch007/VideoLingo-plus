import pandas as pd
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import re
from rich.panel import Panel
from rich.console import Console
import autocorrect_py as autocorrect

console = Console()

CLEANED_CHUNKS_FILE = 'output/log/cleaned_chunks.xlsx'
TRANSLATION_RESULTS_FOR_SUBTITLES_FILE = 'output/log/translation_results_for_subtitles.xlsx'
TRANSLATION_RESULTS_REMERGED_FILE = 'output/log/translation_results_remerged.xlsx'

OUTPUT_DIR = 'output'
AUDIO_OUTPUT_DIR = 'output/audio'

SUBTITLE_OUTPUT_CONFIGS = [ 
    ('src.srt', ['Source']),
    ('trans.srt', ['Translation']),
    ('src_trans.srt', ['Source', 'Translation']),
    ('trans_src.srt', ['Translation', 'Source'])
]

AUDIO_SUBTITLE_OUTPUT_CONFIGS = [
    ('src_subs_for_audio.srt', ['Source']),
    ('trans_subs_for_audio.srt', ['Translation'])
]

def convert_to_srt_format(start_time, end_time):
    """Convert time (in seconds) to the format: hours:minutes:seconds,milliseconds"""
    def seconds_to_hmsm(seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        milliseconds = int(seconds * 1000) % 1000
        return f"{hours:02d}:{minutes:02d}:{int(seconds):02d},{milliseconds:03d}"

    start_srt = seconds_to_hmsm(start_time)
    end_srt = seconds_to_hmsm(end_time)
    return f"{start_srt} --> {end_srt}"

def remove_punctuation(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

def show_difference(str1, str2):
    """Show the difference positions between two strings"""
    min_len = min(len(str1), len(str2))
    diff_positions = []
    
    for i in range(min_len):
        if str1[i] != str2[i]:
            diff_positions.append(i)
    
    if len(str1) != len(str2):
        diff_positions.extend(range(min_len, max(len(str1), len(str2))))
    
    print("Difference positions:")
    print(f"Expected sentence: {str1}")
    print(f"Actual match: {str2}")
    print("Position markers: " + "".join("^" if i in diff_positions else " " for i in range(max(len(str1), len(str2)))))
    print(f"Difference indices: {diff_positions}")

    
    return time_stamp_list

def get_sentence_timestamps(df_words, df_sentences):
    """Original text-matching alignment logic for ASR workflow (Mode 1).
    
    Note: This function uses fuzzy matching to handle text differences that may occur
    due to LLM-based STT error correction in step4_1_summarize.py.
    """
    time_stamp_list = []
    full_words_str = ''
    position_to_word_idx = {}
    
    # Check if speaker column exists
    has_speaker = 'speaker' in df_words.columns
    
    for idx, word in enumerate(df_words['text']):
        clean_word = remove_punctuation(str(word).lower())
        start_pos = len(full_words_str)
        full_words_str += clean_word
        for pos in range(start_pos, len(full_words_str)):
            position_to_word_idx[pos] = idx
    
    current_pos = 0
    for idx, sentence in df_sentences['Source'].items():
        clean_sentence = remove_punctuation(str(sentence).lower()).replace(" ", "")
        sentence_len = len(clean_sentence)
        
        match_found = False
        # Increase search window to handle LLM corrections that may shift positions
        search_range = min(len(full_words_str) - sentence_len + 1, current_pos + 800)
        
        # Find the best match in the search window
        best_match_pos = -1
        # Lower threshold to accommodate LLM corrections (was 0.8)
        highest_similarity = 0.6
        
        # Import difflib at function level for fuzzy matching
        from difflib import SequenceMatcher

        temp_pos = current_pos
        while temp_pos < search_range:
            substring = full_words_str[temp_pos:temp_pos+sentence_len]
            similarity = SequenceMatcher(None, substring, clean_sentence).ratio()

            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match_pos = temp_pos
            
            # If very high similarity, lock it in
            if similarity > 0.95:
                break
            temp_pos += 1

        if best_match_pos != -1:
            start_word_idx = position_to_word_idx[best_match_pos]
            end_word_idx = position_to_word_idx[best_match_pos + sentence_len - 1]
            
            speaker = df_words.iloc[start_word_idx]['speaker'] if has_speaker else None

            time_stamp_list.append((
                float(df_words['start'][start_word_idx]),
                float(df_words['end'][end_word_idx]),
                speaker
            ))
            
            current_pos = best_match_pos + sentence_len
            match_found = True
            
            # Log low similarity matches for debugging
            if highest_similarity < 0.8:
                console.print(f"[yellow]⚠️ Low similarity match ({highest_similarity:.2f}) for: {sentence[:30]}...[/yellow]")

        if not match_found:
            console.print(f"\n⚠️ Warning: No exact match found for sentence: {sentence}")
            show_difference(clean_sentence, 
                          full_words_str[current_pos:current_pos+len(clean_sentence)])
            console.print("\nOriginal sentence:", df_sentences['Source'][idx])
            raise ValueError("❎ No match found for sentence.")
    
    return time_stamp_list

def get_sentence_timestamps_by_index(df_words, df_sentences):
    """Patched index-based alignment logic for provided SRT workflow (Mode 3)."""
    if len(df_words) != len(df_sentences):
        raise Exception(f"FATAL ERROR in step6: Row count mismatch. Word file has {len(df_words)} rows, Sentence file has {len(df_sentences)} rows. Cannot align.")

    time_stamp_list = []
    has_speaker = 'speaker' in df_words.columns
    
    for i in range(len(df_words)):
        try:
            start_time = float(df_words.iloc[i]['start'])
            end_time = float(df_words.iloc[i]['end'])
            speaker = df_words.iloc[i]['speaker'] if has_speaker else None
            time_stamp_list.append((start_time, end_time, speaker))
        except (ValueError, TypeError) as e:
            # Fallback for non-convertible timestamp data
            console.print(f"\n❌ WARNING: Could not convert timestamp to float at row {i}. Data: start='{df_words.iloc[i]['start']}', end='{df_words.iloc[i]['end']}'. Error: {e}")
            last_valid_end = time_stamp_list[-1][1] if time_stamp_list else 0
            # Use None for speaker in fallback
            time_stamp_list.append((last_valid_end, last_valid_end, None))
            continue
    return time_stamp_list

def align_timestamp(df_text, df_translate, subtitle_output_configs: list, output_dir: str, for_display: bool = True):
    """Dispatcher function to select the correct alignment logic."""
    df_trans_time = df_translate.copy()

    # Check if Mode 3 is active by looking for the unique file it creates.
    if os.path.exists('output/log/srt_chunks.xlsx'):
        time_stamp_list = get_sentence_timestamps_by_index(df_text, df_translate)
    else:
        time_stamp_list = get_sentence_timestamps(df_text, df_translate)

    df_trans_time['timestamp'] = [t[:2] for t in time_stamp_list]
    df_trans_time['speaker'] = [t[2] for t in time_stamp_list]
    df_trans_time['duration'] = df_trans_time['timestamp'].apply(lambda x: x[1] - x[0])

    for i in range(len(df_trans_time)-1):
        delta_time = df_trans_time.loc[i+1, 'timestamp'][0] - df_trans_time.loc[i, 'timestamp'][1]
        if 0 < delta_time < 1:
            df_trans_time.at[i, 'timestamp'] = (df_trans_time.loc[i, 'timestamp'][0], df_trans_time.loc[i+1, 'timestamp'][0])

    df_trans_time['timestamp'] = df_trans_time['timestamp'].apply(lambda x: convert_to_srt_format(x[0], x[1]))

    if for_display:
        df_trans_time['Translation'] = df_trans_time['Translation'].apply(lambda x: re.sub(r'[，。]', ' ', x).strip())

    def generate_subtitle_string(df, columns):
        return ''.join([f"{i+1}\n{row['timestamp']}\n{row[columns[0]].strip()}\n{row[columns[1]].strip() if len(columns) > 1 else ''}\n\n" for i, row in df.iterrows()]).strip()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save Excel with speaker info for internal use
        excel_path = os.path.join(output_dir, "final_timeline.xlsx")
        df_trans_time.to_excel(excel_path, index=False)
        console.print(f"[green]💾 Saved timeline with speaker info to {excel_path}[/green]")
        
        for filename, columns in subtitle_output_configs:
            subtitle_str = generate_subtitle_string(df_trans_time, columns)
            with open(os.path.join(output_dir, filename), 'w', encoding='utf-8') as f:
                f.write(subtitle_str)
    
    return df_trans_time

# ✨ Beautify the translation
def clean_translation(x):
    if pd.isna(x):
        return ''
    cleaned = str(x).strip('。').strip('，')
    return autocorrect.format(cleaned)

def align_timestamp_main(df_text, df_translate):
    df_translate['Translation'] = df_translate['Translation'].apply(clean_translation)
    
    align_timestamp(df_text, df_translate, SUBTITLE_OUTPUT_CONFIGS, OUTPUT_DIR)
    console.print(Panel("[bold green]🎉📝 Subtitles generation completed! Please check in the `output` folder 👀[/bold green]"))

    # for audio
    df_translate_for_audio = pd.read_excel(TRANSLATION_RESULTS_REMERGED_FILE) # use remerged file to avoid unmatched lines when dubbing
    df_translate_for_audio['Translation'] = df_translate_for_audio['Translation'].apply(clean_translation)
    
    align_timestamp(df_text, df_translate_for_audio, AUDIO_SUBTITLE_OUTPUT_CONFIGS, AUDIO_OUTPUT_DIR)
    console.print(Panel("[bold green]🎉📝 Audio subtitles generation completed! Please check in the `output/audio` folder 👀[/bold green]"))
    

if __name__ == '__main__':
    align_timestamp_main()