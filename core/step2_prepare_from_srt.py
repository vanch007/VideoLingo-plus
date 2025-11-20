
import os
import sys
import re
import pandas as pd
from rich import print as rprint
from rich.panel import Panel

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Define output paths
OUTPUT_LOG_DIR = 'output/log'
SRT_CHUNKS_PATH = os.path.join(OUTPUT_LOG_DIR, 'srt_chunks.xlsx')
CLEANED_CHUNKS_PATH = os.path.join(OUTPUT_LOG_DIR, 'cleaned_chunks.xlsx')
SENTENCE_MARK_PATH = os.path.join(OUTPUT_LOG_DIR, 'sentence_by_mark.txt')

def time_str_to_seconds(time_str):
    """Converts an SRT time string (HH:MM:SS,ms) to seconds."""
    h, m, s_ms = time_str.split(':')
    s, ms = s_ms.split(',')
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0

def parse_srt_to_dataframe(srt_path: str):
    """
    Parses an SRT file and converts it into a Pandas DataFrame with start, end, and text columns.
    This format mimics the output of the ASR (Whisper) process.
    """
    if not os.path.exists(srt_path):
        raise FileNotFoundError(f"SRT file not found at: {srt_path}")

    with open(srt_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Regex to capture subtitle blocks: index, start_time, end_time, text
    pattern = re.compile(
        r'''(\d+)\r?\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\r?\n(.*?)(?=\r?\n\r?\n|\Z)''',
        re.DOTALL
    )
    
    matches = pattern.findall(content)
    
    subtitles = []
    for match in matches:
        # index = int(match[0])
        start_time = time_str_to_seconds(match[1])
        end_time = time_str_to_seconds(match[2])
        text = match[3].replace('\r', '').replace('\n', ' ').strip()
        
        # For compatibility with cleaned_chunks.xlsx, text is wrapped in quotes
        subtitles.append({
            'start': start_time,
            'end': end_time,
            'text': f'"{text}"'
        })
    
    if not subtitles:
        rprint(Panel("[bold yellow]Warning: No subtitle entries were parsed from the SRT file. The file might be empty or in an incorrect format.[/bold yellow]", title="Parsing Warning"))

    return pd.DataFrame(subtitles)

def prepare_from_srt(user_srt_path: str):
    """
    Main function for this module. It orchestrates the process of converting an SRT file
    into the intermediate formats required by the downstream processing steps.
    """
    rprint(Panel(f"[bold green]🚀 Starting Mode 3: Preparing from provided SRT file: {os.path.basename(user_srt_path)}[/bold green]", title="New Process Start"))

    # Ensure the output directory exists
    os.makedirs(OUTPUT_LOG_DIR, exist_ok=True)

    # 1. Parse the user-provided SRT file into a DataFrame
    df = parse_srt_to_dataframe(user_srt_path)
    
    if df.empty:
        rprint(Panel("[bold red]Error: DataFrame is empty after parsing SRT. Cannot proceed.[/bold red]", title="Error"))
        return

    # 2. Save the DataFrame to srt_chunks.xlsx
    df.to_excel(SRT_CHUNKS_PATH, index=False)
    df.to_excel(CLEANED_CHUNKS_PATH, index=False)
    rprint(f"✅ Successfully created intermediate timeline file: [cyan]{SRT_CHUNKS_PATH}[/cyan] and [cyan]{CLEANED_CHUNKS_PATH}[/cyan]")

    # 3. Create the sentence_by_mark.txt file for the NLP splitting step
    # This file contains the raw text content, which step3_1 expects.
    all_text = '\n'.join(df['text'].str.strip('"'))
    with open(SENTENCE_MARK_PATH, 'w', encoding='utf-8') as f:
        f.write(all_text)
    rprint(f"✅ Successfully created intermediate text file: [cyan]{SENTENCE_MARK_PATH}[/cyan]")
    
    rprint(Panel("[bold green]✨ Preparation from SRT complete. The system can now proceed with the standard translation and dubbing pipeline.[/bold green]", title="Success"))


if __name__ == '__main__':
    # This is a test block to demonstrate the module's functionality.
    # To run this test, create a dummy SRT file named 'test.srt' in the project root.
    
    dummy_srt_content = """1
00:00:01,000 --> 00:00:04,000
Hello, this is the first line of the subtitle.

2
00:00:05,500 --> 00:00:08,500
And this is the second line, demonstrating the process.

3
00:00:10,000 --> 00:00:12,000
The final line.
"""
    dummy_srt_path = 'test.srt'
    with open(dummy_srt_path, 'w', encoding='utf-8') as f:
        f.write(dummy_srt_content)
        
    rprint(f"Created a dummy SRT file for testing at: {dummy_srt_path}")

    # Run the main function
    prepare_from_srt(dummy_srt_path)

    # Verify outputs
    rprint("\n--- Verification ---")
    if os.path.exists(CLEANED_CHUNKS_PATH):
        rprint(f"✔️ [green]File created:[/green] {CLEANED_CHUNKS_PATH}")
        df_check = pd.read_excel(CLEANED_CHUNKS_PATH)
        rprint("Contents of cleaned_chunks.xlsx:")
        rprint(df_check)
    else:
        rprint(f"❌ [red]File not created:[/red] {CLEANED_CHUNKS_PATH}")

    if os.path.exists(SENTENCE_MARK_PATH):
        rprint(f"✔️ [green]File created:[/green] {SENTENCE_MARK_PATH}")
        with open(SENTENCE_MARK_PATH, 'r', encoding='utf-8') as f:
            rprint("Contents of sentence_by_mark.txt:")
            rprint(f.read())
    else:
        rprint(f"❌ [red]File not created:[/red] {SENTENCE_MARK_PATH}")
        
    # Clean up the dummy file
    os.remove(dummy_srt_path)
    rprint(f"\nCleaned up dummy file: {dummy_srt_path}")
