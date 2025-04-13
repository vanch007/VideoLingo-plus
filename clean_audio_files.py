import os
import sys
import shutil
import pandas as pd
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress
from pydub import AudioSegment

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from core.step10_gen_audio import clean_invalid_audio_files, gen_audio, process_row

console = Console()

def force_regenerate_problematic_files():
    """Force regenerate known problematic audio files"""
    rprint(Panel("Checking for known problematic audio files", title="[bold yellow]Targeted Cleanup[/bold yellow]"))

    # List of known problematic file numbers
    problematic_numbers = [693, 699, 700, 701, 702, 703, 704, 705, 698, 858, 653, 1069]

    # Check if task file exists
    tasks_file = "output/audio/tts_tasks.xlsx"
    if not os.path.exists(tasks_file):
        rprint(f"[yellow]⚠️ Task file not found: {tasks_file}[/yellow]")
        return

    # Load task file
    try:
        tasks_df = pd.read_excel(tasks_file)
        rprint(f"[green]✅ Loaded task file: {tasks_file}[/green]")
    except Exception as e:
        rprint(f"[red]❌ Failed to load task file: {str(e)}[/red]")
        return

    # Filter for problematic numbers
    problem_rows = tasks_df[tasks_df['number'].isin(problematic_numbers)]

    if len(problem_rows) == 0:
        rprint("[green]✅ No known problematic files found in the task file[/green]")
        return

    rprint(f"[yellow]⚠️ Found {len(problem_rows)} known problematic files. Regenerating...[/yellow]")

    # Regenerate each problematic file
    with Progress() as progress:
        task = progress.add_task("[cyan]Regenerating problematic files...", total=len(problem_rows))

        for _, row in problem_rows.iterrows():
            number = row['number']
            temp_file = f"output/audio/tmp/{number}_temp.wav"

            # Remove the file if it exists
            if os.path.exists(temp_file):
                os.remove(temp_file)
                rprint(f"[yellow]⚠️ Removed problematic file: {temp_file}[/yellow]")

            # Regenerate the file
            try:
                process_row(row, tasks_df)
                rprint(f"[green]✅ Regenerated file for number {number}[/green]")
            except Exception as e:
                rprint(f"[red]❌ Failed to regenerate file for number {number}: {str(e)}[/red]")

            progress.advance(task)

def verify_all_audio_files():
    """Verify all audio files in both temp and segs directories"""
    rprint(Panel("Verifying all audio files", title="[bold blue]Audio Verification[/bold blue]"))

    temp_dir = 'output/audio/tmp'
    segs_dir = 'output/audio/segs'
    invalid_count = 0

    for directory in [temp_dir, segs_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            continue

        rprint(f"[blue]Checking files in {directory}...[/blue]")

        for filename in os.listdir(directory):
            if not filename.endswith('.wav'):
                continue

            file_path = os.path.join(directory, filename)
            file_size = os.path.getsize(file_path)

            # Check if file is too small
            if file_size < 20000:
                rprint(f"[red]❌ Invalid file size: {file_path} ({file_size} bytes)[/red]")
                os.remove(file_path)
                invalid_count += 1
                continue

            # Try to load with pydub to verify it's a valid audio file
            try:
                audio = AudioSegment.from_wav(file_path)
                duration = len(audio) / 1000  # Convert to seconds

                if duration < 0.5 or duration > 10:
                    rprint(f"[red]❌ Invalid duration: {file_path} ({duration:.2f} seconds)[/red]")
                    os.remove(file_path)
                    invalid_count += 1
            except Exception as e:
                rprint(f"[red]❌ Corrupted audio file: {file_path} - {str(e)}[/red]")
                os.remove(file_path)
                invalid_count += 1

    if invalid_count > 0:
        rprint(f"[yellow]⚠️ Removed {invalid_count} invalid audio files[/yellow]")
    else:
        rprint("[green]✅ All audio files are valid[/green]")

def main():
    """Clean up invalid audio files and regenerate audio"""
    rprint(Panel("Starting audio cleanup and regeneration process", title="[bold blue]VideoLingo Audio Cleanup[/bold blue]"))

    # Step 1: Create necessary directories
    temp_dir = 'output/audio/tmp'
    segs_dir = 'output/audio/segs'
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(segs_dir, exist_ok=True)

    # Step 2: Clean up invalid audio files
    clean_invalid_audio_files()

    # Step 3: Force regenerate known problematic files
    force_regenerate_problematic_files()

    # Step 4: Verify all audio files
    verify_all_audio_files()

    # Step 5: Regenerate audio
    rprint(Panel("Starting audio regeneration process", title="[bold green]Regenerating Audio[/bold green]"))
    gen_audio()

    rprint(Panel("Audio cleanup and regeneration completed", title="[bold green]Process Completed[/bold green]"))

if __name__ == "__main__":
    main()
