import os, sys
import glob
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.step1_ytdlp import find_video_files
import shutil

def cleanup(history_dir="history"):
    print(f"🔍 Starting cleanup process with history_dir={history_dir}")

    # Check if step_timings.json exists before cleanup
    step_timings_path = "output/log/step_timings.json"
    if os.path.exists(step_timings_path):
        print(f"🔍 step_timings.json exists before cleanup: {step_timings_path}")
        print(f"🔍 File size: {os.path.getsize(step_timings_path)} bytes")
        # Make a backup of step_timings.json content
        try:
            with open(step_timings_path, 'r', encoding='utf-8') as f:
                step_timings_content = f.read()
            print(f"🔍 Successfully read step_timings.json content: {len(step_timings_content)} bytes")
        except Exception as e:
            print(f"⚠️ Error reading step_timings.json: {str(e)}")
            step_timings_content = None
    else:
        print(f"🔍 step_timings.json does not exist before cleanup")
        step_timings_content = None

    # Get video file name
    try:
        video_file = find_video_files()
        video_name = video_file.split("/")[1]
        video_name = os.path.splitext(video_name)[0]
        video_name = sanitize_filename(video_name)
        print(f"🔍 Using video name for history folder: {video_name}")
    except Exception as e:
        print(f"⚠️ Error finding video file: {str(e)}")
        print("⚠️ Using 'unknown' as video name")
        video_name = "unknown"

    # Create required folders
    os.makedirs(history_dir, exist_ok=True)
    video_history_dir = os.path.join(history_dir, video_name)
    log_dir = os.path.join(video_history_dir, "log")
    gpt_log_dir = os.path.join(video_history_dir, "gpt_log")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(gpt_log_dir, exist_ok=True)
    print(f"🔍 Created history directories: {log_dir}, {gpt_log_dir}")

    # Move non-log files
    for file in glob.glob("output/*"):
        if not file.endswith(('log', 'gpt_log')):
            move_file(file, video_history_dir)

    # Special handling for step_timings.json - move it first
    if os.path.exists(step_timings_path):
        print(f"🔍 Special handling for step_timings.json")
        # Force copy instead of move to ensure it's properly transferred
        dst_path = os.path.join(log_dir, "step_timings.json")
        try:
            shutil.copy2(step_timings_path, dst_path)
            print(f"✅ Copied step_timings.json to {dst_path}")
            # Verify the copy was successful
            if os.path.exists(dst_path):
                print(f"✅ Verified step_timings.json exists at destination: {dst_path}")
                print(f"✅ Destination file size: {os.path.getsize(dst_path)} bytes")
                # Now remove the original file
                os.remove(step_timings_path)
                print(f"✅ Removed original step_timings.json at {step_timings_path}")
            else:
                print(f"❌ Failed to copy step_timings.json to {dst_path}")
        except Exception as e:
            print(f"❌ Error during special handling of step_timings.json: {str(e)}")

    # Move log files
    print(f"🔍 Moving log files from output/log/ to {log_dir}")
    log_files = glob.glob("output/log/*")
    print(f"🔍 Found {len(log_files)} log files: {log_files}")
    for file in log_files:
        # Skip step_timings.json as it's already handled
        if os.path.basename(file) != "step_timings.json":
            move_file(file, log_dir)

    # Move gpt_log files
    print(f"🔍 Moving gpt_log files from output/gpt_log/ to {gpt_log_dir}")
    for file in glob.glob("output/gpt_log/*"):
        move_file(file, gpt_log_dir)

    # Check if step_timings.json exists in history after moving
    history_step_timings_path = os.path.join(log_dir, "step_timings.json")
    if os.path.exists(history_step_timings_path):
        print(f"🔍 step_timings.json exists in history after cleanup: {history_step_timings_path}")
        print(f"🔍 File size in history: {os.path.getsize(history_step_timings_path)} bytes")
        # Verify the content matches
        if step_timings_content:
            try:
                with open(history_step_timings_path, 'r', encoding='utf-8') as f:
                    history_content = f.read()
                if history_content == step_timings_content:
                    print(f"✅ Content verification: History file content matches original")
                else:
                    print(f"⚠️ Content verification: History file content DOES NOT match original")
            except Exception as e:
                print(f"⚠️ Error verifying history file content: {str(e)}")
    else:
        print(f"🔍 step_timings.json does not exist in history after cleanup: {history_step_timings_path}")
        # If the file doesn't exist in history but we have the content, recreate it
        if step_timings_content:
            try:
                with open(history_step_timings_path, 'w', encoding='utf-8') as f:
                    f.write(step_timings_content)
                print(f"✅ Recreated step_timings.json in history from backup content")
            except Exception as e:
                print(f"❌ Failed to recreate step_timings.json in history: {str(e)}")

    # Delete empty output directories
    try:
        # Check if output/log directory is empty before removing
        if os.path.exists("output/log") and not os.listdir("output/log"):
            os.rmdir("output/log")
            print("🔍 Removed empty output/log directory")
        elif os.path.exists("output/log"):
            print(f"⚠️ output/log directory not empty, contains: {os.listdir('output/log')}")

        # Check if output/gpt_log directory is empty before removing
        if os.path.exists("output/gpt_log") and not os.listdir("output/gpt_log"):
            os.rmdir("output/gpt_log")
            print("🔍 Removed empty output/gpt_log directory")
        elif os.path.exists("output/gpt_log"):
            print(f"⚠️ output/gpt_log directory not empty, contains: {os.listdir('output/gpt_log')}")

        # Check if output directory is empty before removing
        if os.path.exists("output") and not os.listdir("output"):
            os.rmdir("output")
            print("🔍 Removed empty output directory")
        elif os.path.exists("output"):
            print(f"⚠️ output directory not empty, contains: {os.listdir('output')}")
    except OSError as e:
        print(f"⚠️ Could not remove output directories: {str(e)}")  # Log the error

    # Final check to see if step_timings.json was recreated after cleanup
    if os.path.exists(step_timings_path):
        print(f"⚠️ step_timings.json was recreated after cleanup: {step_timings_path}")
        print(f"⚠️ File size: {os.path.getsize(step_timings_path)} bytes")
        # Try to determine what recreated it
        import traceback
        print(f"⚠️ Current call stack:\n{traceback.format_stack()}")

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