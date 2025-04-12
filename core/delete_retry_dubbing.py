import os, sys
import shutil
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def delete_dubbing_files():
    files_to_delete = [
        os.path.join("output", "AI配音.mp4"),
        os.path.join("output", "dub.mp3"),
        os.path.join("output", "dub.srt"),
        os.path.join("output", "dub_orig.srt"),
        os.path.join("output", "audio", "tts_tasks.xlsx"),
        os.path.join("output", "normalized_dub.wav")
    ]
    
    for file_path in files_to_delete:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"Deleted: {file_path}")
            except Exception as e:
                print(f"Error deleting {file_path}: {str(e)}")
        else:
            print(f"File not found: {file_path}")
    
    # 清理音频处理中间文件
    folders_to_delete = [
        os.path.join("output", "audio", "segs"),
        os.path.join("output", "audio", "tmp"),
        os.path.join("output", "audio", "refers")
    ]
    
    for folder_path in folders_to_delete:
        if os.path.exists(folder_path):
            try:
                shutil.rmtree(folder_path)
                print(f"Deleted folder and contents: {folder_path}")
            except Exception as e:
                print(f"Error deleting folder {folder_path}: {str(e)}")
        else:
            print(f"Folder not found: {folder_path}")

if __name__ == "__main__":
    delete_dubbing_files()