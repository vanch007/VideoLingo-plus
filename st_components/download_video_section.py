import streamlit as st
import os, sys, shutil
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.config_utils import load_key
from core.step1_ytdlp import download_video_ytdlp, find_video_files
from core.timing_utils import save_timing, clear_timings
from time import sleep
import re
import subprocess
from translations.translations import translate as t

OUTPUT_DIR = "output"

def download_video_section():
    st.header(t("a. Download or Upload Video"))
    with st.container(border=True):
        # 检查是否已经有视频文件
        video_exists = False
        try:
            video_file = find_video_files()
            video_exists = True
            st.video(video_file)
            col1, col2 = st.columns(2)
            with col1:
                if st.button(t("Delete and Reselect"), key="delete_video_button"):
                    os.remove(video_file)
                    if os.path.exists(OUTPUT_DIR):
                        shutil.rmtree(OUTPUT_DIR)
                    sleep(1)
                    st.rerun()
            with col2:
                if st.button(t("清除耗时统计"), key="clear_timing_button"):
                    clear_timings()
                    st.success(t("耗时统计已清除"))
                    sleep(1)
                    st.rerun()
            return True
        except:
            # 如果没有找到视频文件，继续执行下面的代码
            col1, col2 = st.columns([3, 1])
            with col1:
                url = st.text_input(t("Enter YouTube link:"))
            with col2:
                res_dict = {
                    "360p": "360",
                    "1080p": "1080",
                    "Best": "best"
                }
                target_res = load_key("ytb_resolution")
                res_options = list(res_dict.keys())
                default_idx = list(res_dict.values()).index(target_res) if target_res in res_dict.values() else 0
                res_display = st.selectbox(t("Resolution"), options=res_options, index=default_idx)
                res = res_dict[res_display]
            if st.button(t("Download Video"), key="download_button", use_container_width=True):
                if url:
                    # 记录下载开始时间
                    download_start_time = time.time()
                    with st.spinner("Downloading video..."):
                        download_video_ytdlp(url, resolution=res)
                    # 记录下载耗时
                    download_elapsed = time.time() - download_start_time
                    save_timing("下载视频", download_elapsed)
                    st.rerun()

            if st.button(t("Download and Dub"), key="download_and_dub_button", use_container_width=True):
                if url:
                    # 记录下载开始时间
                    download_start_time = time.time()
                    with st.spinner("Downloading video..."):
                        download_video_ytdlp(url, resolution=res)
                    # 记录下载耗时
                    download_elapsed = time.time() - download_start_time
                    save_timing("下载视频", download_elapsed)

                    # 记录项目总开始时间
                    project_start_time = time.time()
                    save_timing("项目开始时间", project_start_time)

                    # 确保视频文件存在后再处理
                    try:
                        # 检查视频文件是否存在
                        find_video_files()
                        with st.spinner("Processing..."):
                            from st import process_text, process_audio
                            process_text(skip_merge_subtitles=True)
                            process_audio()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        st.info("Please try again or check if the video was downloaded correctly.")

                    # 记录项目总耗时
                    project_elapsed = time.time() - project_start_time
                    save_timing("项目总耗时", project_elapsed)
                    st.rerun()

            uploaded_file = st.file_uploader(t("Or upload video"), type=load_key("allowed_video_formats") + load_key("allowed_audio_formats"))
            if uploaded_file:
                # 记录上传开始时间
                upload_start_time = time.time()

                if os.path.exists(OUTPUT_DIR):
                    shutil.rmtree(OUTPUT_DIR)
                os.makedirs(OUTPUT_DIR, exist_ok=True)

                raw_name = uploaded_file.name.replace(' ', '_')
                name, ext = os.path.splitext(raw_name)
                clean_name = re.sub(r'[^\w\-_\.]', '', name) + ext.lower()

                with open(os.path.join(OUTPUT_DIR, clean_name), "wb") as f:
                    f.write(uploaded_file.getbuffer())

                if ext.lower() in load_key("allowed_audio_formats"):
                    convert_audio_to_video(os.path.join(OUTPUT_DIR, clean_name))

                # 记录上传耗时
                upload_elapsed = time.time() - upload_start_time
                save_timing("上传视频", upload_elapsed)

                # 记录项目总开始时间
                project_start_time = time.time()
                save_timing("项目开始时间", project_start_time)

                # 确保视频文件存在后再处理
                try:
                    # 检查视频文件是否存在
                    find_video_files()
                    with st.spinner("Processing..."):
                        from st import process_text, process_audio
                        process_text(skip_merge_subtitles=True)
                        process_audio()
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.info("Please try again or check if the video was uploaded correctly.")

                # 记录项目总耗时
                project_elapsed = time.time() - project_start_time
                save_timing("项目总耗时", project_elapsed)
                st.rerun()
            else:
                return False

def convert_audio_to_video(audio_file: str) -> str:
    output_video = os.path.join(OUTPUT_DIR, 'black_screen.mp4')
    if not os.path.exists(output_video):
        print(f"🎵➡️🎬 Converting audio to video with FFmpeg ......")
        ffmpeg_cmd = ['ffmpeg', '-y', '-f', 'lavfi', '-i', 'color=c=black:s=640x360', '-i', audio_file, '-shortest', '-c:v', 'libx264', '-c:a', 'aac', '-pix_fmt', 'yuv420p', output_video]
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True, encoding='utf-8')
        print(f"🎵➡️🎬 Converted <{audio_file}> to <{output_video}> with FFmpeg\n")
        # delete audio file
        os.remove(audio_file)
    return output_video
