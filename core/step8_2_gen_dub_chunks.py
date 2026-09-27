import pandas as pd
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.config_utils import load_key
from core.all_whisper_methods.audio_preprocess import get_audio_duration
from core.step8_1_gen_audio_task import time_diff_seconds
import datetime
import re
from core.all_tts_functions.estimate_duration import init_estimator, estimate_duration
from rich import print as rprint
from core.constants import SRC_SUBS_FOR_AUDIO_FILE, TRANS_SUBS_FOR_AUDIO_FILE
from core.dubbing_quality import (
    apply_dubbing_budget_columns,
    estimated_rewrite_needed,
    get_quality_config,
)
from core.dubbing_rewrite import rewrite_task_lines
from core.providers.speaker_diarization import same_known_speaker

INPUT_EXCEL = "output/audio/tts_tasks.xlsx"
OUTPUT_EXCEL = "output/audio/tts_tasks.xlsx"
SRC_SRT = SRC_SUBS_FOR_AUDIO_FILE
TRANS_SRT = TRANS_SUBS_FOR_AUDIO_FILE
MAX_MERGE_COUNT = 5
AUDIO_FILE = 'output/audio/raw.mp3'
ESTIMATOR = None

def calc_if_too_fast(est_dur, tol_dur, duration, tolerance):
    accept = load_key("speed_factor.accept") # Maximum acceptable speed factor
    if est_dur / accept > tol_dur:  # Even max speed factor cannot adapt
        return 2
    elif est_dur > tol_dur:  # Speed adjustment needed within acceptable range
        return 1
    elif est_dur < duration - tolerance:  # Speaking speed too slow
        return -1
    else:  # Normal speaking speed
        return 0

def merge_rows(df, start_idx, merge_count):
    """Merge multiple rows and calculate cumulative values"""
    merged = {
        'est_dur': df.iloc[start_idx]['est_dur'],
        'tol_dur': df.iloc[start_idx]['tol_dur'],
        'duration': df.iloc[start_idx]['duration']
    }

    has_speaker = 'speaker' in df.columns

    while merge_count < MAX_MERGE_COUNT and (start_idx + merge_count) < len(df):
        next_row = df.iloc[start_idx + merge_count]

        # Check speaker consistency
        if has_speaker:
            curr_spk = df.iloc[start_idx]['speaker']
            next_spk = next_row['speaker']
            # Unknown or different speakers are both hard merge boundaries.
            if not same_known_speaker(curr_spk, next_spk):
                df.at[start_idx + merge_count - 1, 'cut_off'] = 1
                return merge_count

        merged['est_dur'] += next_row['est_dur']
        merged['tol_dur'] += next_row['tol_dur']
        merged['duration'] += next_row['duration']

        speed_flag = calc_if_too_fast(
            merged['est_dur'],
            merged['tol_dur'],
            merged['duration'],
            df.iloc[start_idx + merge_count]['tolerance']
        )

        if speed_flag <= 0 or merge_count >= MAX_MERGE_COUNT:
            df.at[start_idx + merge_count, 'cut_off'] = 1
            return merge_count + 1

        merge_count += 1

    # If no suitable merge point is found
    if merge_count >= MAX_MERGE_COUNT or (start_idx + merge_count) >= len(df):
        df.at[start_idx + merge_count - 1, 'cut_off'] = 1
    return merge_count

def analyze_subtitle_timing_and_speed(df):
    rprint("[🔍 Analyzing] Calculating subtitle timing and speed...")
    global ESTIMATOR
    if ESTIMATOR is None:
        ESTIMATOR = init_estimator()
    TOLERANCE = load_key("tolerance")
    whole_dur = get_audio_duration(AUDIO_FILE)
    df['gap'] = 0.0  # Initialize gap column
    df['raw_gap'] = 0.0
    for i in range(len(df) - 1):
        current_end = datetime.datetime.strptime(df.loc[i, 'end_time'], '%H:%M:%S.%f').time()
        next_start = datetime.datetime.strptime(df.loc[i + 1, 'start_time'], '%H:%M:%S.%f').time()
        raw_gap = time_diff_seconds(current_end, next_start, datetime.date.today())
        df.loc[i, 'raw_gap'] = raw_gap
        df.loc[i, 'gap'] = max(0.0, raw_gap)

    # Set the gap for the last line
    last_end = datetime.datetime.strptime(df.iloc[-1]['end_time'], '%H:%M:%S.%f').time()
    last_end_seconds = (last_end.hour * 3600 + last_end.minute * 60 +
                       last_end.second + last_end.microsecond / 1000000)
    last_gap = whole_dur - last_end_seconds
    df.iloc[-1, df.columns.get_loc('raw_gap')] = last_gap
    df.iloc[-1, df.columns.get_loc('gap')] = max(0.0, last_gap)

    df['tolerance'] = df['gap'].apply(lambda x: max(0.0, TOLERANCE if x > TOLERANCE else x))
    df['tol_dur'] = df['duration'] + df['tolerance']
    df['est_dur'] = df.apply(lambda x: estimate_duration(x['text'], ESTIMATOR), axis=1)


    ## Calculate speed indicators
    accept = load_key("speed_factor.accept") # Maximum acceptable speed factor

    def calc_if_too_fast(row):
        est_dur = row['est_dur']
        tol_dur = row['tol_dur']
        duration = row['duration']
        tolerance = row['tolerance']

        if est_dur / accept > tol_dur:  # Even max speed factor cannot adapt
            return 2
        elif est_dur > tol_dur:  # Speed adjustment needed within acceptable range
            return 1
        elif est_dur < duration - tolerance:  # Speaking speed too slow
            return -1
        else:  # Normal speaking speed
            return 0

    df['if_too_fast'] = df.apply(calc_if_too_fast, axis=1)
    return df

def process_cutoffs(df):
    rprint("[✂️ Processing] Generating cutoff points...")
    df['cut_off'] = 0  # Initialize cut_off column
    df.loc[df['gap'] >= load_key("tolerance"), 'cut_off'] = 1  # Set to 1 when gap is greater than TOLERANCE
    idx = 0
    while idx < len(df):
        # Process marked split points
        if df.iloc[idx]['cut_off'] == 1:
            if df.iloc[idx]['if_too_fast'] == 2:
                rprint(f"[⚠️ Warning] Line {idx} is too fast and cannot be fixed by speed adjustment")
            idx += 1
            continue

        # Process the last line
        if idx + 1 >= len(df):
            df.at[idx, 'cut_off'] = 1
            break

        # Process normal or slow lines
        if df.iloc[idx]['if_too_fast'] <= 0:
            if df.iloc[idx + 1]['if_too_fast'] <= 0:
                df.at[idx, 'cut_off'] = 1
                idx += 1
            else:
                idx += merge_rows(df, idx, 1)
        # Process fast lines
        else:
            idx += merge_rows(df, idx, 1)

    return df


def rewrite_estimated_overlong_rows(df):
    """Shorten rows that are already predicted to exceed their dubbing window."""
    if not load_key("rewrite_text_for_dubbing", True):
        return df
    df = apply_dubbing_budget_columns(df)
    if "rewritten_for_dubbing" not in df.columns:
        df["rewritten_for_dubbing"] = False
    if "dubbing_rewrite_rounds" not in df.columns:
        df["dubbing_rewrite_rounds"] = 0
    if "rewrite_reason" not in df.columns:
        df["rewrite_reason"] = ""
    else:
        df["rewrite_reason"] = df["rewrite_reason"].astype(object)

    max_rounds = get_quality_config().max_rewrite_rounds
    for idx, initial_row in df.iterrows():
        row = initial_row.to_dict()
        rounds = int(row.get("dubbing_rewrite_rounds", 0) or 0)
        while estimated_rewrite_needed(row) and rounds < max_rounds:
            try:
                reason = (
                    f"Estimated speech duration {float(row.get('est_dur', 0) or 0):.2f}s exceeds "
                    f"available window {float(row.get('available_duration', 0) or 0):.2f}s."
                )
                rewritten = rewrite_task_lines(
                    row,
                    reason=reason,
                    max_retries=int(load_key("dubbing_repair.llm_retry_attempts", 1)),
                    retry_interval=int(load_key("dubbing_repair.llm_retry_interval", 1)),
                    timeout_seconds=float(load_key("dubbing_repair.llm_timeout_seconds", 60)),
                )
                if not rewritten:
                    break
                new_text = " ".join(rewritten)
                if new_text == str(row.get("text", "")).strip():
                    rprint(f"[yellow]⚠️ Rewrite produced identical text for row {row.get('number', idx)}, stopping rewrite loop.[/yellow]")
                    break
                df.at[idx, "lines"] = rewritten
                df.at[idx, "text"] = new_text
                df.at[idx, "est_dur"] = estimate_duration(new_text, ESTIMATOR)
                df.at[idx, "rewritten_for_dubbing"] = True
                rounds += 1
                df.at[idx, "dubbing_rewrite_rounds"] = rounds
                df.at[idx, "rewrite_reason"] = "estimated_over_duration"
                row = df.loc[idx].to_dict()
                rprint(
                    f"[green]✍️ Rewrote overlong dubbing row {row['number']} "
                    f"before TTS ({rounds}/{max_rounds})[/green]"
                )
            except Exception as exc:
                rprint(f"[yellow]⚠️ Dubbing rewrite skipped for row {row.get('number', idx)}: {exc}[/yellow]")
                break
    df['if_too_fast'] = df.apply(
        lambda x: calc_if_too_fast(x['est_dur'], x['tol_dur'], x['duration'], x['tolerance']),
        axis=1,
    )
    return df


MIN_CHUNK_DURATION = 2.0  # Minimum merged chunk duration in seconds

def pre_merge_short_chunks(df):
    """Pre-merge consecutive ultra-short chunks (gap=0) into groups of >= MIN_CHUNK_DURATION.

    Root fix for alignment: WhisperX produces many sub-0.3s fragments for dense speech.
    When each fragment gets its own TTS (min ~1s), the dubbing drifts massively.
    Merging them into reasonable groups (~2-4s) gives the speed system enough room.
    """
    rprint(f"[🔗 Pre-merge] Merging ultra-short chunks (target >= {MIN_CHUNK_DURATION}s per group)...")

    # First calculate gaps between consecutive rows
    for i in range(len(df) - 1):
        current_end = datetime.datetime.strptime(df.loc[i, 'end_time'], '%H:%M:%S.%f').time()
        next_start = datetime.datetime.strptime(df.loc[i + 1, 'start_time'], '%H:%M:%S.%f').time()
        gap = time_diff_seconds(current_end, next_start, datetime.date.today())
        df.loc[i, '_raw_gap'] = gap
        df.loc[i, '_gap'] = max(0.0, gap)
    df.loc[len(df) - 1, '_gap'] = 999.0  # Last row always breaks

    # Identify merge groups: consecutive rows with gap=0 (or very small gap < 0.05s)
    groups = []
    current_group = [0]
    cumulative_dur = df.loc[0, 'duration']

    for i in range(1, len(df)):
        gap = df.loc[i - 1, '_gap']

        # Check if speaker changes (if speaker column exists)
        # Note: NaN != NaN is True in Python, so we must handle NaN explicitly
        speaker_change = False
        if 'speaker' in df.columns:
            curr_spk = df.loc[current_group[0], 'speaker']
            next_spk = df.loc[i, 'speaker']
            speaker_change = not same_known_speaker(curr_spk, next_spk)

        # Merge conditions: small gap, same speaker, and cumulative duration < target
        if gap < 0.05 and not speaker_change and cumulative_dur < MIN_CHUNK_DURATION:
            current_group.append(i)
            cumulative_dur += df.loc[i, 'duration']
        else:
            groups.append(current_group)
            current_group = [i]
            cumulative_dur = df.loc[i, 'duration']
    groups.append(current_group)

    # Build merged DataFrame
    merged_rows = []
    for group in groups:
        if len(group) == 1:
            idx = group[0]
            merged_rows.append(df.loc[idx].to_dict())
        else:
            first = df.loc[group[0]]
            last = df.loc[group[-1]]

            # Merge text with space separator
            merged_text = ' '.join(str(df.loc[i, 'text']) for i in group if pd.notna(df.loc[i, 'text']) and str(df.loc[i, 'text']).strip())
            merged_origin = ' '.join(str(df.loc[i, 'origin']) for i in group if pd.notna(df.loc[i, 'origin']) and str(df.loc[i, 'origin']).strip())

            merged_row = first.to_dict()
            merged_row['end_time'] = last['end_time']
            merged_row['duration'] = sum(df.loc[i, 'duration'] for i in group)
            merged_row['text'] = merged_text
            merged_row['origin'] = merged_origin
            merged_row['sub_times'] = [eval(first['sub_times'])[0] if isinstance(first['sub_times'], str) else first['sub_times'][0],
                                        eval(last['sub_times'])[1] if isinstance(last['sub_times'], str) else last['sub_times'][1]]
            merged_rows.append(merged_row)

    new_df = pd.DataFrame(merged_rows).reset_index(drop=True)
    new_df['number'] = range(1, len(new_df) + 1)

    # Remove temp column
    if '_gap' in new_df.columns:
        new_df = new_df.drop(columns=['_gap'])
    if '_raw_gap' in new_df.columns:
        new_df = new_df.drop(columns=['_raw_gap'])

    rprint(f"[🔗 Pre-merge] {len(df)} chunks → {len(new_df)} chunks (merged {len(df) - len(new_df)} ultra-short fragments)")
    return new_df

def gen_dub_chunks():
    rprint("[🎬 Starting] Generating dubbing chunks...")
    df = pd.read_excel(INPUT_EXCEL)
    # Note: Filtering is now done in step8_1, so df is already clean

    # Pre-merge ultra-short chunks before timing analysis
    df = pre_merge_short_chunks(df)

    rprint("[📊 Processing] Analyzing timing and speed...")
    df = analyze_subtitle_timing_and_speed(df)

    rprint("[📝 Reading] Loading transcript files...")
    content = open(TRANS_SRT, "r", encoding="utf-8").read()
    ori_content = open(SRC_SRT, "r", encoding="utf-8").read()

    # Process subtitle content
    content_lines = []
    ori_content_lines = []

    # Process translated subtitles
    for block in content.strip().split('\n\n'):
        lines = [line.strip() for line in block.split('\n') if line.strip()]
        if len(lines) >= 3:
            text = ' '.join(lines[2:])
            text = re.sub(r'\([^)]*\)|（[^）]*）', '', text).strip()
            # Remove only '-' character, keep other punctuation
            text = text.replace('-', '')
            content_lines.append(text)

    # Process source subtitles (same structure)
    for block in ori_content.strip().split('\n\n'):
        lines = [line.strip() for line in block.split('\n') if line.strip()]
        if len(lines) >= 3:
            text = ' '.join(lines[2:])
            text = re.sub(r'\([^)]*\)|（[^）]*）', '', text).strip().replace('-', '')
            ori_content_lines.append(text)

    # Match processing
    df['lines'] = None
    df['src_lines'] = None
    last_idx = 0

    def clean_text(text):
        """clean space and punctuation"""
        if not text or not isinstance(text, str):
            return ''
        # First remove all punctuation and whitespace
        return re.sub(r'[^\w\s]|[\s]', '', text)

    for idx, row in df.iterrows():
        target = clean_text(row['text'])
        matches = []
        current = ''
        match_indices = []  # Store indices for matching lines

        # Special handle for empty target (e.g. pure punctuation/space)
        if not target:
            # Try to consume one line from content_lines if it's also empty/negligible
            if last_idx < len(content_lines):
                line = content_lines[last_idx]
                if not clean_text(line):
                    df.at[idx, 'lines'] = [line]
                    df.at[idx, 'src_lines'] = [ori_content_lines[last_idx]]
                    last_idx += 1
                    continue
                else:
                     # If SRT has content but Excel says empty, we might have a sync issue
                     # But we can just set empty for this row and NOT advance last_idx
                     # to let the next Excel row match this SRT line.
                     df.at[idx, 'lines'] = [""]
                     df.at[idx, 'src_lines'] = [""]
                     continue
            else:
                df.at[idx, 'lines'] = [""]
                df.at[idx, 'src_lines'] = [""]
                continue

        for i in range(last_idx, len(content_lines)):
            line = content_lines[i]
            cleaned_line = clean_text(line)
            current += cleaned_line
            matches.append(line)  # 存储原始文本
            match_indices.append(i)

            if current == target:
                df.at[idx, 'lines'] = matches
                df.at[idx, 'src_lines'] = [ori_content_lines[i] for i in match_indices]
                last_idx = i + 1
                break
        else:  # If no match is found
            rprint(f"[yellow]⚠️ Subtitle matching fell back to direct row text at line {idx}: '{row['text']}'[/yellow]")
            df.at[idx, 'lines'] = [row['text']]
            df.at[idx, 'src_lines'] = [row['origin']]

    df = apply_dubbing_budget_columns(df)
    df = rewrite_estimated_overlong_rows(df)
    rprint("[✂️ Processing] Processing cutoffs...")
    df = process_cutoffs(df)

    # Save results
    df.to_excel(OUTPUT_EXCEL, index=False)
    rprint("[✅ Complete] Matching completed successfully!")

if __name__ == "__main__":
    gen_dub_chunks()
