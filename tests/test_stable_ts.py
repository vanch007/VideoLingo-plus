import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tempfile
import subprocess
from rich import print as rprint
from core.config_utils import update_key, load_key
from core.all_whisper_methods.stable_ts_local import transcribe_audio

def create_test_audio():
    """Create a short test audio file using ffmpeg"""
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
        temp_audio_path = temp_audio.name
    
    # Generate a 3-second test tone
    ffmpeg_cmd = f'ffmpeg -y -f lavfi -i "sine=frequency=440:duration=3" "{temp_audio_path}"'
    subprocess.run(ffmpeg_cmd, shell=True, check=True, capture_output=True)
    
    return temp_audio_path

def test_stable_ts_transcription():
    """Test the stable-ts transcription functionality"""
    rprint("[bold cyan]Testing stable-ts transcription...[/bold cyan]")
    
    # Save original runtime setting
    original_runtime = load_key("whisper.runtime")
    
    try:
        # Set runtime to stable-ts
        update_key("whisper.runtime", "stable-ts")
        
        # Create test audio
        test_audio_path = create_test_audio()
        rprint(f"[green]Created test audio file:[/green] {test_audio_path}")
        
        # Test transcription
        try:
            result = transcribe_audio(test_audio_path, 0.0, 3.0)
            
            # Check if result has expected structure
            if 'segments' in result and 'language' in result:
                rprint("[bold green]✓ Test passed! stable-ts transcription works correctly.[/bold green]")
                rprint(f"[cyan]Detected language:[/cyan] {result['language']}")
                rprint(f"[cyan]Number of segments:[/cyan] {len(result['segments'])}")
                
                # Print first segment if available
                if result['segments']:
                    rprint(f"[cyan]First segment text:[/cyan] {result['segments'][0]['text']}")
            else:
                rprint("[bold red]✗ Test failed! Result does not have expected structure.[/bold red]")
                rprint(f"Result: {result}")
        except Exception as e:
            rprint(f"[bold red]✗ Test failed with error:[/bold red] {str(e)}")
        
        # Clean up test audio
        if os.path.exists(test_audio_path):
            os.unlink(test_audio_path)
    
    finally:
        # Restore original runtime setting
        update_key("whisper.runtime", original_runtime)

if __name__ == "__main__":
    test_stable_ts_transcription()
