import streamlit as st
import os
import shutil
import json
from core.config_utils import get_work_dir
import os, sys
import time
import pandas as pd
from st_components.imports_and_utils import *
from st_components.timing_display import display_timing_statistics_component
from core.config_utils import load_key, update_key
from core.timing_utils import get_formatted_timings, save_timing
from core.step1_ytdlp import find_video_files
import core.step2_whisperX as step2_whisperX
import core.step3_1_spacy_split as step3_1_spacy_split
import core.step3_2_splitbymeaning as step3_2_splitbymeaning
import core.step4_1_summarize as step4_1_summarize
import core.step4_2_translate_all as step4_2_translate_all
import core.step5_splitforsub as step5_splitforsub
import core.step6_generate_final_timeline as step6_generate_final_timeline
import core.step7_merge_sub_to_vid as step7_merge_sub_to_vid
import core.step8_1_gen_audio_task as step8_1_gen_audio_task
import core.step8_2_gen_dub_chunks as step8_2_gen_dub_chunks
import core.step9_extract_refer_audio as step9_extract_refer_audio
import core.step10_gen_audio as step10_gen_audio
import core.step11_merge_full_audio as step11_merge_full_audio
import core.step12_merge_dub_to_vid as step12_merge_dub_to_vid

# 导入新功能模块
import core.step2_extract_subtitles as step2_extract_subtitles
import core.step3_3_process_extracted_subs as step3_3_process_extracted_subs
from core.step_checker import get_completed_steps, get_pending_steps, is_step_completed

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

import core.step2_prepare_from_srt as step2_prepare_from_srt

def text_processing_section():
    st.header(t("b. Translate and Generate Subtitles"))
    timing_placeholder = st.empty()
    with st.container(border=True):
        # 初始化 session_state
        if 'processing_mode' not in st.session_state:
            st.session_state.processing_mode = t("模式一：ASR语音识别")

        # 模式选择
        processing_mode = st.radio(
            t("字幕处理模式"),
            options=[
                t("模式一：ASR语音识别"),
                t("模式二：提取内嵌字幕"),
                t("模式三：提供视频和源字幕")
            ],
            captions=[
                t("从音频转录文字并翻译"),
                t("使用视频内已有的字幕轨道"),
                t("提供自己的SRT字幕文件进行翻译")
            ],
            horizontal=True,
            key='processing_mode' # 绑定到 session_state
        )

        # 为模式三添加文件上传组件
        uploaded_srt_file = None
        if st.session_state.processing_mode == t("模式三：提供视频和源字幕"):
            uploaded_srt_file = st.file_uploader(
                t("上传您的SRT源字幕文件"),
                type=['srt'],
                help=t("请上传一个UTF-8编码的SRT格式字幕文件")
            )
            # 模式三：上传文件后显示专门的开始按钮
            if uploaded_srt_file is not None:
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(t("开始翻译和配音"), key="mode3_translate_and_dub_button"):
                        project_start_time = time.time()
                        save_timing("项目开始时间", project_start_time)
                        with st.spinner(t("Processing translation and dubbing...")):
                            try:
                                find_video_files()
                                if run_text_processing_pipeline(st.session_state.processing_mode, uploaded_srt_file, timing_placeholder):
                                    process_audio(timing_placeholder)
                                save_timing("项目总耗时", time.time() - project_start_time)
                                timing_placeholder.empty()
                            except Exception as e:
                                st.error(f"Error: {str(e)}")
                                st.info("Please make sure a video is available before processing.")
                with col2:
                    if st.button(t("仅处理字幕"), key="mode3_text_processing_button"):
                        project_start_time = time.time()
                        save_timing("项目开始时间", project_start_time)
                        try:
                            find_video_files()
                            run_text_processing_pipeline(st.session_state.processing_mode, uploaded_srt_file, timing_placeholder)
                            save_timing("项目总耗时", time.time() - project_start_time)
                            timing_placeholder.empty()
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                            st.info("Please make sure a video is available before processing.")

        if not os.path.exists(SUB_VIDEO) and not os.path.exists(DUB_VIDEO) and st.session_state.processing_mode != t("模式三：提供视频和源字幕"):
            # 定义清理函数
            def cleanup_intermediate_files():
                st.toast("Cleaning up intermediate files from previous run...")
                files_to_delete = [
                    'output/log/cleaned_chunks.xlsx',
                    'output/log/srt_chunks.xlsx', # Add the new srt_chunks file to cleanup
                    'output/log/sentence_by_mark.txt',
                    'output/log/sentence_splitbynlp.txt',
                    'output/log/sentence_splitbymeaning.txt',
                    'output/log/translation_results.xlsx',
                    'output/log/translation_results_for_subtitles.xlsx',
                    'output/log/translation_results_remerged.xlsx'
                ]
                for file_path in files_to_delete:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                st.toast("Cleanup complete.")

            col1, col2 = st.columns(2)
            with col1:
                if st.button(t("Translate and Dub"), key="translate_and_dub_button"):
                    # cleanup_intermediate_files() # 移除自动清理以支持断点续传
                    project_start_time = time.time()
                    save_timing("项目开始时间", project_start_time)
                    with st.spinner(t("Processing translation and dubbing...")):
                        try:
                            find_video_files()
                            if st.session_state.processing_mode == t("模式一：ASR语音识别"):
                                from core.pipeline.job import start_pipeline_job
                                start_pipeline_job(os.path.abspath(find_video_files()))
                                st.rerun()
                            elif run_text_processing_pipeline(st.session_state.processing_mode, uploaded_srt_file, timing_placeholder):
                                process_audio(timing_placeholder)
                            save_timing("项目总耗时", time.time() - project_start_time)
                            timing_placeholder.empty()
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                            st.info("Please make sure a video is available before processing.")
            with col2:
                if st.button(t("Start Processing Subtitles"), key="text_processing_button"):
                    # cleanup_intermediate_files() # 移除自动清理以支持断点续传
                    project_start_time = time.time()
                    save_timing("项目开始时间", project_start_time)
                    try:
                        find_video_files()
                        run_text_processing_pipeline(st.session_state.processing_mode, uploaded_srt_file, timing_placeholder)
                        save_timing("项目总耗时", time.time() - project_start_time)
                        timing_placeholder.empty()
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
                import importlib, core.onekeycleanup
                importlib.reload(core.onekeycleanup)
                core.onekeycleanup.cleanup()
                st.rerun()
            return True

import core.step2_prepare_from_srt as step2_prepare_from_srt

def run_text_processing_pipeline(processing_mode, uploaded_srt_file, timing_placeholder):
    """根据选择的模式运行相应的文本处理流程。"""

    # 模式一：ASR
    if processing_mode == t("模式一：ASR语音识别"):
        with st.spinner(t("Using Whisper for transcription...")):
            start_time = time.time()
            step2_whisperX.transcribe()
            elapsed = time.time() - start_time
            save_timing("转录", elapsed)
        run_translation_pipeline(perform_splitting=True, timing_placeholder=timing_placeholder)
        return True

    # 模式二：提取内嵌字幕
    elif processing_mode == t("模式二：提取内嵌字幕"):
        process_extracted_subtitles(timing_placeholder)
        return True

    # 模式三：提供SRT文件
    elif processing_mode == t("模式三：提供视频和源字幕"):
        if uploaded_srt_file is None:
            st.error(t("请先上传一个SRT文件"))
            return False
        
        # 为配音流程准备音频文件
        with st.spinner(t("音频预处理中...")):
            from core.step2_whisperX import prepare_audio_only
            prepare_audio_only()

        # 保存上传的SRT文件
        srt_path = os.path.join("output", "user_provided.srt")
        with open(srt_path, "wb") as f:
            f.write(uploaded_srt_file.getbuffer())
        
        with st.spinner(t("正在从SRT文件准备时间轴...")):
            start_time = time.time()
            step2_prepare_from_srt.prepare_from_srt(srt_path)
            elapsed = time.time() - start_time
            save_timing("SRT字幕解析", elapsed)
        
        run_translation_pipeline(perform_splitting=False, timing_placeholder=timing_placeholder)
        return True
    
    return False

def run_translation_pipeline(perform_splitting: bool, timing_placeholder):
    """运行共享的翻译和字幕生成流程，支持断点续传。"""
    from core.translation_state import guard_translation_artifacts_for_steps
    total_start_time = time.time()
    with timing_placeholder.container():
        display_timing_statistics_component(key_suffix="text_init")

    try:
        # Step 3: Conditional Sentence Splitting
        guard = guard_translation_artifacts_for_steps(["translate"])
        if not guard["ok"]:
            raise RuntimeError("已有译文与当前模型/流程不匹配。请先审阅并归档旧译文，再运行新翻译。")
        if perform_splitting:
            if not is_step_completed("split_meaning"):
                with st.spinner(t("Splitting long sentences...")):
                    if not is_step_completed("split_spacy"):
                        start_time = time.time()
                        step3_1_spacy_split.split_by_spacy()
                        elapsed = time.time() - start_time
                        save_timing("NLP分句", elapsed)
                    else:
                        st.info("✓ NLP分句已完成，跳过")
                    with timing_placeholder.container():
                        display_timing_statistics_component(key_suffix="text_step2")

                    start_time = time.time()
                    step3_2_splitbymeaning.split_sentences_by_meaning()
                    elapsed = time.time() - start_time
                    save_timing("LLM分句", elapsed)
                    with timing_placeholder.container():
                        display_timing_statistics_component(key_suffix="text_step3")
            else:
                st.info("✓ 分句已完成，跳过")
        else:
            st.info(t("Skipping sentence splitting, using lines from provided SRT."))
            df_chunks = pd.read_excel('output/log/cleaned_chunks.xlsx')
            all_text = '\n'.join(df_chunks['text'].str.strip('"'))
            with open('output/log/sentence_splitbymeaning.txt', 'w', encoding='utf-8') as f:
                f.write(all_text)
            save_timing("NLP分句", 0.01)
            save_timing("LLM分句", 0.01)

        # Step 4: Summarize and Translate
        if not is_step_completed("translate"):
            with st.spinner(t("Summarizing and translating...")):
                if not is_step_completed("summarize"):
                    start_time = time.time()
                    # STT correction is only needed for ASR mode (perform_splitting=True)
                    # For Mode 2 (extracted subs) and Mode 3 (provided SRT), skip correction
                    step4_1_summarize.get_summary()
                    elapsed = time.time() - start_time
                    save_timing("摘要", elapsed)
                else:
                    st.info("✓ 摘要已完成，跳过")
                with timing_placeholder.container():
                    display_timing_statistics_component(key_suffix="text_step4")

                if load_key("pause_before_translate"):
                    input(t("⚠️ PAUSE_BEFORE_TRANSLATE. Go to `output/log/terminology.json` to edit terminology. Then press ENTER to continue..."))

                start_time = time.time()
                step4_2_translate_all.translate_all()
                elapsed = time.time() - start_time
                save_timing("翻译", elapsed)
                with timing_placeholder.container():
                    display_timing_statistics_component(key_suffix="text_step5")
        else:
            st.info("✓ 翻译已完成，跳过")

        # Step 5: Conditional Subtitle Splitting
        if not is_step_completed("split_subtitle"):
            if perform_splitting:
                with st.spinner(t("Processing and aligning subtitles...")):
                    start_time = time.time()
                    step5_splitforsub.split_for_sub_main()
                    elapsed = time.time() - start_time
                    save_timing("字幕分割", elapsed)
                    with timing_placeholder.container():
                        display_timing_statistics_component(key_suffix="text_step6")
            else:
                shutil.copy('output/log/translation_results.xlsx', 'output/log/translation_results_for_subtitles.xlsx')
                shutil.copy('output/log/translation_results.xlsx', 'output/log/translation_results_remerged.xlsx')
                st.info(t("Skipping subtitle line splitting."))
                save_timing("字幕分割", 0.01)
        else:
            st.info("✓ 字幕分割已完成，跳过")

        # --- Step 6: Generate Final Timeline (In-Memory) ---
        if not is_step_completed("timeline"):
            with st.spinner(t("Generating final timeline...")):
                start_time = time.time()
                # Load the necessary data into memory, with isolated paths
                if perform_splitting: # Mode 1
                    df_text = pd.read_excel('output/log/cleaned_chunks.xlsx')
                else: # Mode 3
                    df_text = pd.read_excel('output/log/srt_chunks.xlsx')

                df_text['text'] = df_text['text'].str.strip('"').str.strip()
                df_translate = pd.read_excel('output/log/translation_results_for_subtitles.xlsx')
                # Call the refactored function with DataFrames
                step6_generate_final_timeline.align_timestamp_main(df_text, df_translate)
                elapsed = time.time() - start_time
                save_timing("时间轴对齐", elapsed)
                with timing_placeholder.container():
                    display_timing_statistics_component(key_suffix="text_step7")
        else:
            st.info("✓ 时间轴对齐已完成，跳过")

        # Step 7: Merge Subtitles to Video
        if not is_step_completed("merge_subtitle"):
            with st.spinner(t("Merging subtitles to video...")):
                start_time = time.time()
                step7_merge_sub_to_vid.merge_subtitles_to_video()
                elapsed = time.time() - start_time
                save_timing("字幕合并到视频", elapsed)
                with timing_placeholder.container():
                    display_timing_statistics_component(key_suffix="text_step8")
        else:
            st.info("✓ 字幕视频已生成，跳过")

        save_timing("整体字幕处理", time.time() - total_start_time)
        st.success(t("Subtitle processing complete! 🎉"))
        st.balloons()

    except Exception as e:
        st.error(f"Error during text processing: {str(e)}")
        raise e
    finally:
        with timing_placeholder.container():
            display_timing_statistics_component(key_suffix="text_final")

def process_extracted_subtitles(timing_placeholder):
    """使用内嵌字幕的工作流程"""
    # 记录整体字幕处理开始时间
    total_start_time = time.time()

    # 创建一个占位符来显示实时耗时统计
    # timing_placeholder = st.empty() # Removed local placeholder creation

    # 显示初始耗时统计
    with timing_placeholder.container():
        display_timing_statistics_component(key_suffix="extracted_subs_init")

    try:
        # 确保必要的目录存在
        import os
        os.makedirs('output/log', exist_ok=True)
        os.makedirs('output/audio', exist_ok=True)
        
        # 首先执行音频预处理步骤（仅处理音频，不进行转录）
        with st.spinner(t("音频预处理...")):
            start_time = time.time()
            # 使用仅处理音频的函数而不是完整的转录函数
            from core.step2_whisperX import prepare_audio_only
            prepare_audio_only()  # 这只会处理音频文件，不会执行转录
            elapsed = time.time() - start_time
            save_timing("音频预处理", elapsed)
            with timing_placeholder.container():
                display_timing_statistics_component(key_suffix="extracted_subs_step0")

        with st.spinner(t("提取内嵌字幕...")):
            start_time = time.time()
            from core import step2_extract_subtitles
            success = step2_extract_subtitles.extract_subtitles_main()
            if not success:
                raise Exception("提取内嵌字幕失败")
            elapsed = time.time() - start_time
            save_timing("提取内嵌字幕", elapsed)
            with timing_placeholder.container():
                display_timing_statistics_component(key_suffix="extracted_subs_step1")

        with st.spinner(t("处理提取的字幕...")):
            start_time = time.time()
            from core import step3_3_process_extracted_subs
            success = step3_3_process_extracted_subs.process_extracted_subs_main()
            if not success:
                raise Exception("处理提取的字幕失败")
            elapsed = time.time() - start_time
            save_timing("处理提取的字幕", elapsed)
            with timing_placeholder.container():
                display_timing_statistics_component(key_suffix="extracted_subs_step2")

        # 记录整体字幕处理耗时
        save_timing("整体字幕处理(内嵌)", time.time() - total_start_time)

        st.success(t("内嵌字幕处理完成! 🎉"))
        st.balloons()
    except Exception as e:
        st.error(f"处理内嵌字幕时出错: {str(e)}")
        raise e
    finally:
        # 无论是否出错，都显示耗时统计
        with timing_placeholder.container():
            display_timing_statistics_component(key_suffix="extracted_subs_final")

def audio_processing_section():
    st.header(t("c. Dubbing"))
    timing_placeholder = st.empty()
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
                    project_start_time = time.time()
                    save_timing("项目开始时间", project_start_time)
                    # 检查视频文件是否存在
                    find_video_files()
                    if st.session_state.get('processing_mode') == t("模式一：ASR语音识别"):
                        from core.pipeline.job import start_pipeline_job
                        start_pipeline_job(os.path.abspath(find_video_files()))
                    else:
                        process_audio(timing_placeholder)
                    save_timing("项目总耗时", time.time() - project_start_time)
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
                import importlib, core.onekeycleanup
                importlib.reload(core.onekeycleanup)
                core.onekeycleanup.cleanup()
                st.rerun()

def process_audio(timing_placeholder):
    """处理配音流程，支持断点续传。"""
    from core.translation_state import guard_translation_artifacts_for_steps
    total_start_time = time.time()

    # 显示初始耗时统计
    with timing_placeholder.container():
        display_timing_statistics_component(key_suffix="audio_init")

    try:
        # Step 8.1: Generate audio tasks
        guard = guard_translation_artifacts_for_steps(["gen_audio_task"])
        if not guard["ok"]:
            raise RuntimeError("已有译文与当前模型/流程不匹配。请先审阅并归档旧译文，再开始配音。")
        if not is_step_completed("gen_audio_task"):
            with st.spinner(t("Generate audio tasks")):
                start_time = time.time()
                step8_1_gen_audio_task.gen_audio_task_main()
                elapsed = time.time() - start_time
                save_timing("生成配音任务", elapsed)
                with timing_placeholder.container():
                    display_timing_statistics_component(key_suffix="audio_step1")
        else:
            st.info("✓ 配音任务已生成，跳过")

        # Step 8.2: Generate dub chunks
        if not is_step_completed("gen_dub_chunks"):
            with st.spinner(t("Generate dub chunks")):
                start_time = time.time()
                step8_2_gen_dub_chunks.gen_dub_chunks()
                elapsed = time.time() - start_time
                save_timing("生成配音分块", elapsed)
                with timing_placeholder.container():
                    display_timing_statistics_component(key_suffix="audio_step2")
        else:
            st.info("✓ 配音分块已生成，跳过")

        # Step 9: Extract reference audio
        if not is_step_completed("extract_refer"):
            with st.spinner(t("Extract refer audio")):
                start_time = time.time()
                step9_extract_refer_audio.extract_refer_audio_main()
                elapsed = time.time() - start_time
                save_timing("提取参考音频", elapsed)
                with timing_placeholder.container():
                    display_timing_statistics_component(key_suffix="audio_step3")
        else:
            st.info("✓ 参考音频已提取，跳过")

        # Step 10: Generate all audio
        if not is_step_completed("gen_audio"):
            with st.spinner(t("Generate all audio")):
                start_time = time.time()
                step10_gen_audio.gen_audio()
                elapsed = time.time() - start_time
                save_timing("生成配音", elapsed)
                with timing_placeholder.container():
                    display_timing_statistics_component(key_suffix="audio_step4")
        else:
            st.info("✓ 配音已生成，跳过")

        # Step 11: Merge full audio
        if not is_step_completed("merge_audio"):
            with st.spinner(t("Merge full audio")):
                start_time = time.time()
                step11_merge_full_audio.merge_full_audio()
                elapsed = time.time() - start_time
                save_timing("合并配音", elapsed)
                with timing_placeholder.container():
                    display_timing_statistics_component(key_suffix="audio_step5")
        else:
            st.info("✓ 配音已合并，跳过")

        # Step 12: Merge dubbing to video
        if not is_step_completed("merge_video"):
            with st.spinner(t("Merge dubbing to the video")):
                start_time = time.time()
                step12_merge_dub_to_vid.merge_video_audio()
                elapsed = time.time() - start_time
                save_timing("配音合并到视频", elapsed)
                with timing_placeholder.container():
                    display_timing_statistics_component(key_suffix="audio_step6")
        else:
            st.info("✓ 配音视频已生成，跳过")

        # 记录整体配音处理耗时
        save_timing("整体配音处理", time.time() - total_start_time)
        # 最终更新耗时统计显示
        with timing_placeholder.container():
            display_timing_statistics_component(key_suffix="audio_final")

        st.success(t("Audio processing complete! 🎇"))
        st.balloons()
    except Exception as e:
        st.error(f"Error during audio processing: {str(e)}")
        raise e
    finally:
        # 无论是否出错，都显示耗时统计
        with timing_placeholder.container():
            display_timing_statistics_component(key_suffix="audio_error")



@st.fragment(run_every=5)
def pipeline_job_progress():
    from core.pipeline.job import job_status
    job = job_status()
    if not job.get('pid'):
        return
    # Fallback to current state even if run_id updated during resume
    state = job.get('state') or {}
    if job['active']:
        done, steps = len(state.get('completed_steps', [])), state.get('steps', [])
        step_name = state.get('current_step') or '环境检查'
        if step_name == 'translate':
            step_name = '翻译与逐句配音预检'
        st.progress(done / max(len(steps), 1), text=f"自动运行：{step_name}（{done}/{len(steps)}）")
        if state.get('current_step') == 'translate' and job.get('workspace'):
            audit_file = os.path.join(job['workspace'], 'output/log/duration_candidate_semantic_audit.json')
            try:
                with open(audit_file, encoding='utf-8') as handle:
                    candidates = json.load(handle)['candidates']
                source_count = len({item['source'] for item in candidates})
                st.caption(f"逐句预检：已写入 {len(candidates)} 条候选语义校验记录，涉及 {source_count} 种原句。记录数不等于验收通过数。")
                natural_dir = os.path.join(job['workspace'], 'output/audio/preflight')
                if os.path.isdir(natural_dir):
                    count = sum(entry.is_file() and entry.name.endswith('.wav') for entry in os.scandir(natural_dir))
                    st.caption(f"已生成自然配音文件：{count}；完整回读、时长与字幕检查通过后才交付。")
                takes_file = os.path.join(job['workspace'], 'output/log/translation_speech_candidate_state.json')
                if os.path.isfile(takes_file):
                    with open(takes_file, encoding='utf-8') as handle:
                        speech = json.load(handle)
                    latest = {take['unit_id']: take for take in speech.get('attempts', [])
                              if 'natural_seconds' in take}
                    accepted = sum(take.get('fits') is True and take.get('content_complete') is True
                                   for take in latest.values())
                    awaiting_readback = sum(take.get('content_complete') is None for take in latest.values())
                    total = len(speech.get('candidates', []))
                    if latest:
                        st.caption(f"最近一次逐句结果：{accepted}/{total} 句同时通过内容与时长检查；"
                                   f"{awaiting_readback} 句待回读，其余句子由流程局部重试。最终验收仍以完整报告为准。")
            except (OSError, ValueError, KeyError, TypeError):
                # The producer may be writing its next receipt. A partial
                # progress read must never interrupt the independent job.
                pass
    elif state.get('status') == 'failed' or (not job['active'] and state.get('status') not in {'completed', 'accepted', 'rendered'}):
        failed_step = state.get('failed_step') or '执行'
        err_msg = state.get('error') or '任务异常退出'
        st.error(f"流程在 {failed_step} 停止：{err_msg}")
        c1, c2 = st.columns([1, 1])
        with c1:
            if st.button('从检查点继续', key='resume_pipeline_job'):
                from core.pipeline.job import start_pipeline_job
                start_pipeline_job(job['input'], subtitle_only=state.get('subtitle_only', False),
                                   no_subtitles=state.get('no_subtitles', False), workspace=job.get('workspace'))
                st.rerun()
        with c2:
            if st.button('清除此任务状态 (重新在 output 运行)', key='clear_pipeline_job'):
                from pathlib import Path
                job_f = Path('output/pipeline_job.json')
                if job_f.exists(): job_f.unlink()
                st.rerun()
    elif state.get('quality_status') == 'fast_basic_pass':
        st.success('快速模式成片已生成，完整性检查通过。深度语义、回读及音色审核未执行。')
    elif state.get('quality_gate_passed') is True:
        st.success('成片已生成，自动质量检查通过。人工观感验收另行确认。')
    elif state.get('status') in {'rendered', 'completed', 'accepted'}:
        st.warning('成片已生成，但质量验收尚未通过，请查看报告。')
    else:
        st.warning('任务已停止，请查看运行日志。')
    if job.get('workspace') and state.get('profile') == 'fast':
        progress_path = os.path.join(job['workspace'], 'output/log/fast_progress.json')
        try:
            with open(progress_path, encoding='utf-8') as handle:
                progress = json.load(handle)
            st.caption(f"配音：新生成 {progress.get('generated', 0)}，复用 {progress.get('reused', 0)}，"
                       f"异常 {progress.get('failed', 0)}；耗时 {progress.get('elapsed_seconds', 0):.1f} 秒")
            manifest_path = os.path.join(job['workspace'], 'output/log/fast_dubbing_manifest.json')
            if os.path.isfile(manifest_path):
                with open(manifest_path, encoding='utf-8') as handle:
                    manifest = json.load(handle)
                st.caption(f"画面局部延长：{manifest.get('total_video_extension_seconds', 0):.3f} 秒")
        except (OSError, ValueError, TypeError):
            pass
    if job.get('workspace'):
        st.caption('任务目录：' + job['workspace'])
        video = os.path.join(job['workspace'], 'output', 'AI配音.mp4')
        if not job['active'] and os.path.isfile(video):
            st.video(video)
    if job.get('log') and os.path.isfile(job['log']):
        with st.expander('运行日志', expanded=False):
            st.code(open(job['log'], encoding='utf-8', errors='replace').read()[-6000:])
    if not job['active'] and st.button('刷新结果', key='refresh_pipeline_result'):
        st.rerun()


def main():
    logo_col, _ = st.columns([1,1])
    with logo_col:
        st.image("docs/logo.png", width="stretch")
    st.markdown(button_style, unsafe_allow_html=True)
    welcome_text = t("Hello, welcome to VideoLingo. If you encounter any issues, feel free to get instant answers with our Free QA Agent <a href=\"https://share.fastgpt.in/chat/share?shareId=066w11n3r9aq6879r4z0v9rh\" target=\"_blank\">here</a>! You can also try out our SaaS website at <a href=\"https://videolingo.io\" target=\"_blank\">videolingo.io</a> for free!")
    st.markdown(f"<p style='font-size: 20px; color: #808080;'>{welcome_text}</p>", unsafe_allow_html=True)
    # add settings
    with st.sidebar:
        page_setting()
        with st.expander("Workflow Status", expanded=False):
            completed_steps = get_completed_steps()
            pending_steps = get_pending_steps()
            st.caption(f"Completed: {len(completed_steps)} | Pending: {len(pending_steps)}")
            if completed_steps:
                st.write("✅ " + ", ".join(completed_steps))
            if pending_steps:
                st.write("⏳ " + ", ".join(pending_steps[:8]))
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

    from core.pipeline.job import job_status
    if job_status().get('pid'):
        with st.container(border=True):
            pipeline_job_progress()
        st.divider()
    download_video_section()
    text_processing_section()
    audio_processing_section()

    # 根据开关状态显示耗时统计区域
    if st.session_state.get("show_timing_toggle", True):
        st.divider()
        display_timing_statistics_component(key_suffix="main_page")

if __name__ == "__main__":
    main()
