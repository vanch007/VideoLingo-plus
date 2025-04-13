import streamlit as st
import os, sys
import time
import pandas as pd
from st_components.imports_and_utils import *
from core.config_utils import load_key
from core.timing_utils import time_it, get_formatted_timings, clear_timings, save_timing
from core.step1_ytdlp import find_video_files

# 确保set_page_config()只在主脚本中调用一次
if not hasattr(st, '_page_config_set'):
    st.set_page_config(page_title="VideoLingo", page_icon="docs/logo.svg")
    st._page_config_set = True

# SET PATH
current_dir = os.path.dirname(os.path.abspath(__file__))
os.environ['PATH'] += os.pathsep + current_dir
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

SUB_VIDEO = "output/AI字幕.mp4"
DUB_VIDEO = "output/AI配音.mp4"

def text_processing_section():
    st.header(t("b. Translate and Generate Subtitles"))
    with st.container(border=True):
        if not os.path.exists(SUB_VIDEO) and not os.path.exists(DUB_VIDEO):
            col1, col2 = st.columns(2)
            with col1:
                if st.button(t("Translate and Dub"), key="translate_and_dub_button"):
                    with st.spinner(t("Processing translation and dubbing...")):
                        # 调用process_text函数并跳过step7的字幕处理流程
                        try:
                            # 检查视频文件是否存在
                            find_video_files()
                            process_text(skip_merge_subtitles=True)
                            process_audio()
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                            st.info("Please make sure a video is available before processing.")
                    st.rerun()
            with col2:
                if st.button(t("Start Processing Subtitles"), key="text_processing_button"):
                    try:
                        # 检查视频文件是否存在
                        find_video_files()
                        process_text()
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        st.info("Please make sure a video is available before processing.")

        st.markdown(f"""
        <p style='font-size: 20px;'>
        {t("This stage includes the following steps:")}
        <p style='font-size: 20px;'>
            1. {t("WhisperX word-level transcription")}<br>
            2. {t("Sentence segmentation using NLP and LLM")}<br>
            3. {t("Summarization and multi-step translation")}<br>
            4. {t("Cutting and aligning long subtitles")}<br>
            5. {t("Generating timeline and subtitles")}<br>
            6. {t("Merging subtitles into the video")}
        """, unsafe_allow_html=True)

        if os.path.exists(SUB_VIDEO):
            if load_key("burn_subtitles"):
                st.video(SUB_VIDEO)
            download_subtitle_zip_button(text=t("Download All Srt Files"))

            if st.button(t("Archive to 'history'"), key="cleanup_in_text_processing"):
                cleanup()
                st.rerun()
            return True

def process_text(skip_merge_subtitles=False):
    # 记录整体字幕处理开始时间
    total_start_time = time.time()

    # 创建一个占位符来显示实时耗时统计
    timing_placeholder = st.empty()

    # 显示初始耗时统计
    with timing_placeholder.container():
        display_timing_statistics(key_suffix="text_init")

    # Check if stable-ts is selected but not installed
    if load_key("whisper.runtime") == "stable-ts":
        try:
            import stable_whisper
        except ImportError:
            st.error(t("stable-ts is not installed. Please run 'python install_stable_ts.py' to install it."))
            st.info(t("Alternatively, you can change the WhisperX Runtime to 'local' or 'cloud' in the settings."))
            return

    try:
        with st.spinner(t("Using Whisper for transcription...")):
            start_time = time.time()
            step2_whisperX.transcribe()
            elapsed = time.time() - start_time
            save_timing("转录", elapsed)
            # 更新耗时统计显示
            with timing_placeholder.container():
                display_timing_statistics(key_suffix="text_step1")

        with st.spinner(t("Splitting long sentences...")):
            start_time = time.time()
            step3_1_spacy_split.split_by_spacy()
            elapsed = time.time() - start_time
            save_timing("NLP分句", elapsed)
            # 更新耗时统计显示
            with timing_placeholder.container():
                display_timing_statistics(key_suffix="text_step2")

            start_time = time.time()
            step3_2_splitbymeaning.split_sentences_by_meaning()
            elapsed = time.time() - start_time
            save_timing("LLM分句", elapsed)
            # 更新耗时统计显示
            with timing_placeholder.container():
                display_timing_statistics(key_suffix="text_step3")

        with st.spinner(t("Summarizing and translating...")):
            start_time = time.time()
            step4_1_summarize.get_summary()
            elapsed = time.time() - start_time
            save_timing("摘要", elapsed)
            # 更新耗时统计显示
            with timing_placeholder.container():
                display_timing_statistics(key_suffix="text_step4")

            if load_key("pause_before_translate"):
                input(t("⚠️ PAUSE_BEFORE_TRANSLATE. Go to `output/log/terminology.json` to edit terminology. Then press ENTER to continue..."))

            start_time = time.time()
            step4_2_translate_all.translate_all()
            elapsed = time.time() - start_time
            save_timing("翻译", elapsed)
            # 更新耗时统计显示
            with timing_placeholder.container():
                display_timing_statistics(key_suffix="text_step5")

        with st.spinner(t("Processing and aligning subtitles...")):
            start_time = time.time()
            step5_splitforsub.split_for_sub_main()
            elapsed = time.time() - start_time
            save_timing("字幕分割", elapsed)
            # 更新耗时统计显示
            with timing_placeholder.container():
                display_timing_statistics(key_suffix="text_step6")

            start_time = time.time()
            step6_generate_final_timeline.align_timestamp_main()
            elapsed = time.time() - start_time
            save_timing("时间轴对齐", elapsed)
            # 更新耗时统计显示
            with timing_placeholder.container():
                display_timing_statistics(key_suffix="text_step7")

        if not skip_merge_subtitles:
            with st.spinner(t("Merging subtitles to video...")):
                start_time = time.time()
                step7_merge_sub_to_vid.merge_subtitles_to_video()
                elapsed = time.time() - start_time
                save_timing("字幕合并到视频", elapsed)
                # 更新耗时统计显示
                with timing_placeholder.container():
                    display_timing_statistics(key_suffix="text_step8")
        else:
            # 跳过字幕合并到视频步骤
            rprint = print if 'rprint' not in globals() else globals()['rprint']
            rprint("[bold yellow]Skipping step7_merge_sub_to_vid.py as requested[/bold yellow]")

        # 记录整体字幕处理耗时
        save_timing("整体字幕处理", time.time() - total_start_time)

        st.success(t("Subtitle processing complete! 🎉"))
        st.balloons()
    except Exception as e:
        st.error(f"Error during text processing: {str(e)}")
        raise e
    finally:
        # 无论是否出错，都显示耗时统计
        display_timing_statistics(key_suffix="text_final")

def audio_processing_section():
    st.header(t("c. Dubbing"))
    with st.container(border=True):
        st.markdown(f"""
        <p style='font-size: 20px;'>
        {t("This stage includes the following steps:")}
        <p style='font-size: 20px;'>
            1. {t("Generate audio tasks and chunks")}<br>
            2. {t("Extract reference audio")}<br>
            3. {t("Generate and merge audio files")}<br>
            4. {t("Merge final audio into video")}
        """, unsafe_allow_html=True)
        if not os.path.exists(DUB_VIDEO):
            if st.button(t("Start Audio Processing"), key="audio_processing_button"):
                try:
                    # 检查视频文件是否存在
                    find_video_files()
                    process_audio()
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.info("Please make sure a video is available before processing.")
        else:
            st.success(t("Audio processing is complete! You can check the audio files in the `output` folder."))
            if load_key("burn_subtitles"):
                st.video(DUB_VIDEO)
            if st.button(t("删除配音文件"), key="delete_dubbing_files"):
                delete_dubbing_files()
                st.rerun()
            if st.button(t("归档到'history'文件夹"), key="cleanup_in_audio_processing"):
                cleanup()
                st.rerun()

def process_audio():
    # 记录整体配音处理开始时间
    total_start_time = time.time()

    # 创建一个占位符来显示实时耗时统计
    timing_placeholder = st.empty()

    # 显示初始耗时统计
    with timing_placeholder.container():
        display_timing_statistics(key_suffix="audio_init")

    try:
        with st.spinner(t("Generate audio tasks")):
            start_time = time.time()
            step8_1_gen_audio_task.gen_audio_task_main()
            elapsed = time.time() - start_time
            save_timing("生成配音任务", elapsed)
            # 更新耗时统计显示
            with timing_placeholder.container():
                display_timing_statistics(key_suffix="audio_step1")

            start_time = time.time()
            step8_2_gen_dub_chunks.gen_dub_chunks()
            elapsed = time.time() - start_time
            save_timing("生成配音分块", elapsed)
            # 更新耗时统计显示
            with timing_placeholder.container():
                display_timing_statistics(key_suffix="audio_step2")

        with st.spinner(t("Extract refer audio")):
            start_time = time.time()
            step9_extract_refer_audio.extract_refer_audio_main()
            elapsed = time.time() - start_time
            save_timing("提取参考音频", elapsed)
            # 更新耗时统计显示
            with timing_placeholder.container():
                display_timing_statistics(key_suffix="audio_step3")

        with st.spinner(t("Generate all audio")):
            start_time = time.time()
            step10_gen_audio.gen_audio()
            elapsed = time.time() - start_time
            save_timing("生成配音", elapsed)
            # 更新耗时统计显示
            with timing_placeholder.container():
                display_timing_statistics(key_suffix="audio_step4")

        with st.spinner(t("Merge full audio")):
            start_time = time.time()
            step11_merge_full_audio.merge_full_audio()
            elapsed = time.time() - start_time
            save_timing("合并配音", elapsed)
            # 更新耗时统计显示
            with timing_placeholder.container():
                display_timing_statistics(key_suffix="audio_step5")

        with st.spinner(t("Merge dubbing to the video")):
            start_time = time.time()
            step12_merge_dub_to_vid.merge_video_audio()
            elapsed = time.time() - start_time
            save_timing("配音合并到视频", elapsed)
            # 更新耗时统计显示
            with timing_placeholder.container():
                display_timing_statistics(key_suffix="audio_step6")

        # 记录整体配音处理耗时
        save_timing("整体配音处理", time.time() - total_start_time)
        # 最终更新耗时统计显示
        with timing_placeholder.container():
            display_timing_statistics(key_suffix="audio_final")

        st.success(t("Audio processing complete! 🎇"))
        st.balloons()
    except Exception as e:
        st.error(f"Error during audio processing: {str(e)}")
        raise e
    finally:
        # 无论是否出错，都显示耗时统计
        display_timing_statistics(key_suffix="audio_error")

def display_timing_statistics(key_suffix="main"):
    """显示各步骤耗时统计

    Args:
        key_suffix (str): 按钮的key后缀，用于区分不同实例的按钮
    """
    try:
        # 确保计时文件存在
        from core.timing_utils import ensure_timing_file
        ensure_timing_file()

        # 使用高亮边框和颜色
        with st.container(border=True):
            # 使用更醒目的标题
            st.markdown("<h3 style='color:#FF4B4B;'>⏱️ 处理耗时统计</h3>", unsafe_allow_html=True)

            try:
                timings = get_formatted_timings()

                if timings:
                    # 分类耗时数据
                    project_timings = []
                    download_timings = []
                    text_timings = []
                    audio_timings = []
                    other_timings = []

                    # 直接使用已排序的耗时数据
                    for item in timings:
                        step_name = item['步骤']
                        if step_name in ['项目开始时间', '项目总耗时']:
                            project_timings.append(item)
                        elif step_name in ['下载视频', '上传视频']:
                            download_timings.append(item)
                        elif step_name in ['转录', 'NLP分句', 'LLM分句', '摘要', '翻译', '字幕分割', '时间轴对齐', '字幕合并到视频', '整体字幕处理']:
                            text_timings.append(item)
                        elif step_name in ['生成配音任务', '生成配音分块', '提取参考音频', '生成配音', '合并配音', '配音合并到视频', '整体配音处理']:
                            audio_timings.append(item)
                        else:
                            other_timings.append(item)

                    # 显示项目总耗时
                    if project_timings:
                        st.markdown("<h4 style='color:#4B8BF5;'>项目总耗时</h4>", unsafe_allow_html=True)
                        project_df = pd.DataFrame(project_timings)
                        st.dataframe(project_df, use_container_width=True, height=min(35 * (len(project_timings) + 1), 150))

                    # 显示下载/上传耗时
                    if download_timings:
                        st.markdown("<h4 style='color:#4B8BF5;'>下载/上传耗时</h4>", unsafe_allow_html=True)
                        download_df = pd.DataFrame(download_timings)
                        st.dataframe(download_df, use_container_width=True, height=min(35 * (len(download_timings) + 1), 150))

                    # 显示翻译阶段耗时
                    if text_timings:
                        st.markdown("<h4 style='color:#4B8BF5;'>翻译阶段耗时</h4>", unsafe_allow_html=True)
                        # 定义翻译阶段的步骤顺序
                        text_step_order = {
                            '整体字幕处理': 0,
                            '转录': 1,
                            '摘要': 2,
                            'LLM分句': 3,
                            'NLP分句': 4,
                            '翻译': 5,
                            '字幕分割': 6,
                            '时间轴对齐': 7,
                            '字幕合并到视频': 8
                        }
                        # 按步骤顺序排序
                        sorted_text_timings = sorted(text_timings, key=lambda x: text_step_order.get(x['步骤'], 100))
                        text_df = pd.DataFrame(sorted_text_timings)
                        st.dataframe(text_df, use_container_width=True, height=min(35 * (len(text_timings) + 1), 300))

                        # 创建翻译阶段的条形图
                        try:

                            chart_data = pd.DataFrame({
                                '步骤': [item['步骤'] for item in sorted_text_timings],
                                '耗时(秒)': [float(item['耗时'].split()[0]) if '秒' in item['耗时'] and '分' not in item['耗时'] and '小时' not in item['耗时'] else
                                          float(item['耗时'].split()[0]) * 60 + float(item['耗时'].split()[2]) if '分' in item['耗时'] and '小时' not in item['耗时'] else
                                          float(item['耗时'].split()[0]) * 3600 + float(item['耗时'].split()[2]) * 60 + float(item['耗时'].split()[4]) if '小时' in item['耗时'] else 0
                                         for item in sorted_text_timings]
                            })
                            st.bar_chart(chart_data.set_index('步骤'), use_container_width=True, height=200)
                        except Exception as e:
                            st.warning(f"无法生成翻译阶段图表: {str(e)}")

                    # 显示配音阶段耗时
                    if audio_timings:
                        st.markdown("<h4 style='color:#4B8BF5;'>配音阶段耗时</h4>", unsafe_allow_html=True)
                        # 定义配音阶段的步骤顺序
                        audio_step_order = {
                            '整体配音处理': 0,
                            '生成配音任务': 1,
                            '生成配音分块': 2,
                            '提取参考音频': 3,
                            '生成配音': 4,
                            '合并配音': 5,
                            '配音合并到视频': 6
                        }
                        # 按步骤顺序排序
                        sorted_audio_timings = sorted(audio_timings, key=lambda x: audio_step_order.get(x['步骤'], 100))
                        audio_df = pd.DataFrame(sorted_audio_timings)
                        st.dataframe(audio_df, use_container_width=True, height=min(35 * (len(audio_timings) + 1), 300))

                        # 创建配音阶段的条形图
                        try:

                            chart_data = pd.DataFrame({
                                '步骤': [item['步骤'] for item in sorted_audio_timings],
                                '耗时(秒)': [float(item['耗时'].split()[0]) if '秒' in item['耗时'] and '分' not in item['耗时'] and '小时' not in item['耗时'] else
                                          float(item['耗时'].split()[0]) * 60 + float(item['耗时'].split()[2]) if '分' in item['耗时'] and '小时' not in item['耗时'] else
                                          float(item['耗时'].split()[0]) * 3600 + float(item['耗时'].split()[2]) * 60 + float(item['耗时'].split()[4]) if '小时' in item['耗时'] else 0
                                         for item in sorted_audio_timings]
                            })
                            st.bar_chart(chart_data.set_index('步骤'), use_container_width=True, height=200)
                        except Exception as e:
                            st.warning(f"无法生成配音阶段图表: {str(e)}")

                    # 显示其他耗时
                    if other_timings:
                        st.markdown("<h4 style='color:#4B8BF5;'>其他耗时</h4>", unsafe_allow_html=True)
                        other_df = pd.DataFrame(other_timings)
                        st.dataframe(other_df, use_container_width=True, height=min(35 * (len(other_timings) + 1), 200))
                else:
                    # 使用更醒目的提示
                    st.info("暂无耗时数据，运行处理后将显示在这里")
            except Exception as e:
                st.error(f"获取耗时数据出错: {str(e)}")

            # 添加清除按钮
            col1, col2 = st.columns([3, 1])
            with col2:
                # 使用不同的key后缀来区分不同实例的按钮
                button_key = f"clear_timing_{key_suffix}"
                if st.button("清除耗时统计", key=button_key, type="primary"):
                    try:
                        from core.timing_utils import clear_timings
                        clear_timings()
                        st.success("耗时统计数据已清除")
                        st.rerun()
                    except Exception as e:
                        st.error(f"清除耗时统计出错: {str(e)}")
    except Exception as e:
        st.error(f"显示耗时统计出错: {str(e)}")

def main():
    logo_col, _ = st.columns([1,1])
    with logo_col:
        st.image("docs/logo.png", use_column_width=True)
    st.markdown(button_style, unsafe_allow_html=True)
    welcome_text = t("Hello, welcome to VideoLingo. If you encounter any issues, feel free to get instant answers with our Free QA Agent <a href=\"https://share.fastgpt.in/chat/share?shareId=066w11n3r9aq6879r4z0v9rh\" target=\"_blank\">here</a>! You can also try out our SaaS website at <a href=\"https://videolingo.io\" target=\"_blank\">videolingo.io</a> for free!")
    st.markdown(f"<p style='font-size: 20px; color: #808080;'>{welcome_text}</p>", unsafe_allow_html=True)
    # add settings
    with st.sidebar:
        page_setting()
        st.markdown(give_star_button, unsafe_allow_html=True)

        # 在侧边栏添加耗时统计开关
        st.divider()
        st.subheader("耗时统计设置")
        show_timing = st.toggle("显示耗时统计", value=True, key="show_timing_toggle")
        if st.button("清除耗时数据", key="clear_timing_in_sidebar"):
            from core.timing_utils import clear_timings
            clear_timings()
            st.success("耗时统计数据已清除")
            st.rerun()

    download_video_section()
    text_processing_section()
    audio_processing_section()

    # 根据开关状态显示耗时统计区域
    if st.session_state.get("show_timing_toggle", True):
        st.divider()
        display_timing_statistics(key_suffix="main_page")

if __name__ == "__main__":
    main()
