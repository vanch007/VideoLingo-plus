import os, sys
import glob
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.step1_ytdlp import find_video_files
import shutil

def cleanup(history_dir="history"):
    print(f"Starting cleanup process, moving files to {history_dir}")

    # Get video file name to create a unique history folder
    try:
        video_file = find_video_files()
        video_name = os.path.splitext(os.path.basename(video_file))[0]
        video_name = sanitize_filename(video_name)
    except Exception as e:
        print(f"Could not find a unique video file, using 'unknown' for history folder. Error: {e}")
        video_name = "unknown"

    # Define history paths
    video_history_dir = os.path.join(history_dir, video_name)
    log_dir = os.path.join(video_history_dir, "log")
    gpt_log_dir = os.path.join(video_history_dir, "gpt_log")

    # Create history directories
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(gpt_log_dir, exist_ok=True)

    # Move all files and subdirectories from output/log to the new log_dir
    if os.path.exists("output/log"):
        for item in os.listdir("output/log"):
            move_file(os.path.join("output/log", item), log_dir)

    # Move all files from output/gpt_log to the new gpt_log_dir
    if os.path.exists("output/gpt_log"):
        for item in os.listdir("output/gpt_log"):
            move_file(os.path.join("output/gpt_log", item), gpt_log_dir)

    # Move remaining files from the root of output to the video_history_dir
    if os.path.exists("output"):
        for item in os.listdir("output"):
            src_path = os.path.join("output", item)
            if item not in ['log', 'gpt_log']:
                move_file(src_path, video_history_dir)

    # Clean up empty source directories
    try:
        if os.path.exists("output/log") and not os.listdir("output/log"):
            os.rmdir("output/log")
        if os.path.exists("output/gpt_log") and not os.listdir("output/gpt_log"):
            os.rmdir("output/gpt_log")
        if os.path.exists("output") and not os.listdir("output"):
            os.rmdir("output")
    except OSError as e:
        print(f"Could not remove all source directories: {e}")

def move_file(src, dst):
    try:
        # Get the source file name
        src_filename = os.path.basename(src)
        # Use os.path.join to ensure correct path and include file name
        dst = os.path.join(dst, sanitize_filename(src_filename))

        # Add detailed logging for step_timings.json
        if src_filename == "step_timings.json":
            print(f"📊 Processing step_timings.json: {src}")
            print(f"📊 Source exists: {os.path.exists(src)}")
            print(f"📊 Source size: {os.path.getsize(src) if os.path.exists(src) else 'N/A'}")
            print(f"📊 Destination path: {dst}")

        if os.path.exists(dst):
            if os.path.isdir(dst):
                # If destination is a folder, try to delete its contents
                shutil.rmtree(dst, ignore_errors=True)
                if src_filename == "step_timings.json":
                    print(f"📊 Removed existing directory at destination: {dst}")
            else:
                # If destination is a file, try to delete it
                os.remove(dst)
                if src_filename == "step_timings.json":
                    print(f"📊 Removed existing file at destination: {dst}")

        shutil.move(src, dst, copy_function=shutil.copy2)

        # Verify the file was moved successfully
        if src_filename == "step_timings.json":
            print(f"📊 Destination exists after move: {os.path.exists(dst)}")
            print(f"📊 Destination size after move: {os.path.getsize(dst) if os.path.exists(dst) else 'N/A'}")
            print(f"📊 Source exists after move: {os.path.exists(src)}")

        print(f"✅ Moved: {src} -> {dst}")
    except PermissionError:
        print(f"⚠️ Permission error: Cannot delete {dst}, attempting to overwrite")
        try:
            shutil.copy2(src, dst)
            os.remove(src)
            print(f"✅ Copied and deleted source file: {src} -> {dst}")

            # Verify the file was copied successfully
            if src_filename == "step_timings.json":
                print(f"📊 Destination exists after copy: {os.path.exists(dst)}")
                print(f"📊 Destination size after copy: {os.path.getsize(dst) if os.path.exists(dst) else 'N/A'}")
                print(f"📊 Source exists after removal: {os.path.exists(src)}")
        except Exception as e:
            print(f"❌ Move failed: {src} -> {dst}")
            print(f"Error message: {str(e)}")
            if src_filename == "step_timings.json":
                print(f"📊 Error details for step_timings.json: {type(e).__name__}")
    except Exception as e:
        print(f"❌ Move failed: {src} -> {dst}")
        print(f"Error message: {str(e)}")
        if src_filename == "step_timings.json":
            print(f"📊 Error details for step_timings.json: {type(e).__name__}")

def sanitize_filename(filename):
    # Remove or replace disallowed characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename

if __name__ == "__main__":
    cleanup()