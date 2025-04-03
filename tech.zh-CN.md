**Videolingo 视频翻译系统技术文档**

Videolingo 是一个高度集成的视频翻译系统，能够自动化执行视频下载、音频提取、语音识别、字幕生成、文本翻译，以及音视频合成等一系列复杂操作。该系统还提供了一个 Web 界面，用于任务管理和系统配置。

对于开发人员，可以单步执行 `core` 下的每一个 `step__.py` 文件并在 `output` 下检查每一步的输出。

以下是系统的核心技术模块、工作流程和主要函数：

1. **视频获取模块**:
   - `core/step1_ytdlp.py`: 集成`yt-dlp`库，实现从指定 URL 高效下载视频的功能，并清理文件名。
     - `sanitize_filename(filename)`: 清理文件名中的特殊字符
     - `download_video_ytdlp(url, save_path='output', resolution='1080', cutoff_time=None)`: 下载视频主函数
     - `find_video_files(save_path='output')`: 查找下载的视频文件

2. **音频处理与语音识别模块**:
   - `core/all_whisper_methods/whisperX.py`: 使用本地 WhisperX 模型进行转录。
     - `transcribe_audio(audio_file: str, start: float, end: float)`: 转录指定时间段的音频
   - `core/all_whisper_methods/whisperXapi.py`: 使用 replicate 的 whisperX 模型进行转录。
   - `core/step2_whisperX.py`: 利用 WhisperX 模型进行高精度的语音识别
     - `enhance_vocals(vocals_ratio=2.50)`: 增强人声音量
     - `transcribe()`: 执行转录主函数
   - `core/all_whisper_methods/audio_preprocess.py`: 音频预处理工具
     - `compress_audio(input_file: str, output_file: str)`: 压缩音频文件
     - `convert_video_to_audio(video_file: str)`: 视频转音频
     - `get_audio_duration(audio_file: str)`: 获取音频时长
     - `split_audio(audio_file: str, target_len: int)`: 分割长音频
     - `process_transcription(result: Dict)`: 处理转录结果
     - `save_results(df: pd.DataFrame)`: 保存转录结果
   - `core/all_whisper_methods/demucs_vl.py`: 音频分离
     - `demucs_main()`: 执行人声分离主函数

3. **文本处理与翻译模块**:
   - `core/step3_1_spacy_split.py`: 应用 SpaCy 自然语言处理工具进行初步的文本分割。
     - `split_by_spacy()`: 执行SpaCy分割主函数
   - `core/step3_2_splitbymeaning.py`: 结合 GPT 模型的语义理解能力，对长句进行更精确的分割。
     - `split_sentence(sentence, num_parts, word_limit=18)`: 分割单个句子
     - `parallel_split_sentences(sentences, max_length, max_workers, nlp)`: 并行分割多个句子
     - `split_sentences_by_meaning()`: 执行语义分割主函数
   - `core/step4_1_summarize.py`: 利用 GPT 模型对视频内容进行智能摘要，提取关键术语。
     - `combine_chunks()`: 合并文本块
     - `search_things_to_note_in_prompt(sentence)`: 搜索需要注意的内容
     - `get_summary()`: 获取摘要主函数
     - `valid_summary(response_data)`: 验证摘要结果
   - `core/step4_2_translate_all.py`: 实现批量化的字幕文本翻译处理。
     - `split_chunks_by_chars(chunk_size=400, max_i=8)`: 按字符分割文本块
     - `translate_chunk(chunk, chunks, theme_prompt, i)`: 翻译单个文本块
     - `translate_all()`: 执行批量翻译主函数
   - `core/translate_once.py`: 采用三步翻译法（直译、意译和润色）实现高质量的英文到中文的逐句翻译。
     - `valid_translate_result(result, required_keys, required_sub_keys)`: 验证翻译结果
     - `translate_lines(lines, previous_content_prompt, after_cotent_prompt, things_to_note_prompt, summary_prompt)`: 翻译多行文本
     - `retry_translation(prompt, step_name)`: 重试翻译
   - `core/spacy_utils/`: SpaCy工具集
     - `split_by_comma.py`: 按逗号分割
       - `split_by_comma(text, nlp)`: 按逗号分割文本
     - `split_by_connector.py`: 按连接词分割
       - `split_by_connectors(text, context_words=5, nlp=None)`: 按连接词分割文本
     - `split_long_by_root.py`: 按根节点分割长句
       - `split_long_sentence(doc)`: 分割长句子
     - `split_by_mark.py`: 按标点分割
       - `split_by_mark(nlp)`: 按标点分割文本
     - `load_nlp_model.py`: 加载NLP模型
       - `get_spacy_model(language: str)`: 获取指定语言的SpaCy模型

4. **字幕处理与合成模块**:
   - `core/step5_splitforsub.py`: 根据字幕格式规范，对翻译后的文本进行精确分割和时间对齐。
      - `calc_len(text: str)`: 计算文本长度
      - `align_subs(src_sub: str, tr_sub: str, src_part: str)`: 对齐源字幕和翻译字幕
      - `split_align_subs(src_lines: List[str], tr_lines: List[str])`: 分割和对齐字幕
      - `split_for_sub_main()`: 执行字幕分割主函数
   - `core/step6_generate_final_timeline.py`: 生成标准 SRT 格式的字幕文件，包含精确的时间轴信息。
      - `seconds_to_hmsm(seconds)`: 秒数转换为时间格式
      - `get_sentence_timestamps(df_words, df_sentences)`: 获取句子时间戳
      - `align_timestamp(df_text, df_translate, subtitle_output_configs, output_dir, for_display)`: 对齐时间戳
      - `align_timestamp_main()`: 执行时间轴对齐主函数
   - `core/step7_merge_sub_to_vid.py`: 实现字幕与视频的无缝集成，使用 ffmpeg 进行处理。
      - `check_gpu_available()`: 检查GPU可用性
      - `merge_subtitles_to_video()`: 合并字幕到视频主函数

5. **音频处理与配音模块**:
     - `core/step8_1_gen_audio_task.py`: 生成音频任务，处理字幕以确保与时间相符。
       - `check_len_then_trim(text, duration)`: 检查并修剪文本长度
       - `process_srt()`: 处理SRT字幕文件
       - `gen_audio_task_main()`: 生成音频任务主函数
     - `core/step8_2_gen_dub_chunks.py`: 生成配音片段
       - `calc_if_too_fast(est_dur, tol_dur, duration, tolerance)`: 计算是否语速过快
       - `merge_rows(df, start_idx, merge_count)`: 合并字幕行
       - `gen_dub_chunks()`: 生成配音片段主函数
     - `core/step9_extract_refer_audio.py`: 提取参考音频
       - `extract_refer_audio_main()`: 提取参考音频主函数
     - `core/step10_gen_audio.py`: 从文本生成音频文件
       - `adjust_audio_speed(input_file: str, output_file: str, speed_factor: float)`: 调整音频速度
       - `parse_df_srt_time(time_str: str)`: 转换SRT时间格式为秒数
       - `process_row(row: pd.Series, tasks_df: pd.DataFrame)`: 处理单行数据并检测音频文件异常(大小<30k或时长>10秒)
       - `generate_tts_audio(tasks_df: pd.DataFrame)`: 生成TTS音频并实现3次重试机制
       - `process_chunk(chunk_df: pd.DataFrame, accept: float, min_speed: float)`: 处理音频块并计算速度因子
       - `merge_chunks(tasks_df: pd.DataFrame)`: 合并音频块并调整时间轴
       - `gen_audio()`: 生成音频主函数
     - `core/step11_merge_full_audio.py`: 合并完整音频
       - `merge_full_audio()`: 合并音频主函数
     - `core/step12_merge_dub_to_vid.py`: 将配音合并到视频
       - `normalize_audio_volume(audio_path: str, output_path: str, target_db: float)`: 标准化音频音量
       - `merge_video_audio()`: 合并视频音频主函数
     - `core/all_tts_functions/`: 文本转语音功能
       - `azure_tts.py`: Azure TTS
         - `azure_tts(text: str, save_path: str)`: Azure文本转语音
       - `edge_tts.py`: Edge TTS
         - `edge_tts(text, save_path)`: Edge文本转语音
       - `fish_tts.py`: Fish TTS
         - `fish_tts(text: str, save_as: str)`: Fish文本转语音
       - `gpt_sovits_tts.py`: GPT-SoVITS TTS
         - `gpt_sovits_tts(text, text_lang, save_path, ref_audio_path, prompt_lang, prompt_text)`: GPT-SoVITS文本转语音
       - `openai_tts.py`: OpenAI TTS
         - `openai_tts(text, save_path)`: OpenAI文本转语音
       - `tts_main.py`: TTS主入口
         - `tts_main(text, save_as, number, task_df)`: TTS主函数
       - `estimate_duration.py`: 音频时长估计
         - `estimate_duration(text: str, lang: str)`: 估计文本转语音时长

6. **系统配置与工具模块**:
   - `core/step8_gen_audio_task.py`: 生成音频任务，处理字幕以确保与时间相符。
   - `core/step10_gen_audio.py`: 从文本生成音频文件，并根据时间调整语速。
   - `core/step11_merge_audio_to_vid.py`: 将生成的配音音频与视频进行专业级别的合成。
   - `core/delete_retry_dubbing.py`: 删除不必要的音频文件以清理生成过程中的多余文件。

6. **自然语言处理工具集**:
   - `core/ask_gpt.py`: 封装与 GPT 模型交互的标准化接口，用于各类文本生成和分析任务。
   - `core/prompts_storage.py`: 集中管理针对不同任务优化的提示模板。
   - `core/spacy_utils/`: 封装基于 SpaCy 的句子分割等高级文本处理功能。
     - `split_by_connector.py`: 根据连接词拆分文本句子，提高可读性。
     - `split_by_comma.py`: 根据逗号和冒号拆分文本处理。
     - `split_long_by_root.py`: 按句子根节点拆分长句子，增强文本的可读性。
     - `split_by_mark.py`: 用标点符号对文本进行句子拆分。
     - `load_nlp_model.py`: 加载和初始化所需的 NLP 模型，支持多种语言。

7. **文本转语音（TTS）模块**:
   - `core/all_tts_functions/fish_tts.py`: 使用外部 API 实现文本转语音功能，生成音频文件。
   - `core/all_tts_functions/openai_tts.py`: 使用 OpenAI 的 TTS 服务将文本转换为音频并保存。
   - `core/all_tts_functions/gpt_sovits_tts.py`: 使用 GPT-SoVITS 进行文本到语音转换，支持多语言。
   - `core/all_tts_functions/azure_tts.py`: 利用 Azure 语音服务将文本转换为音频，保存为 WAV 格式。

8. **系统配置与工具模块**:
   - `config.yaml`: 集中存储和管理系统的全局参数配置。
   - `install.py`: 自动化系统依赖包和模型的安装与配置过程。
   - `onekeycleanup.py`: 提供一键式中间文件清理功能，优化系统存储空间。
   - `core/config_utils.py`: 读取和更新 YAML 配置文件，确保多线程安全的读写操作。

9. **批量处理模块**:
   - `batch/utils/batch_processor.py`: 批量处理视频任务，通过 Excel 配置管理视频处理流程。
   - `batch/utils/video_processor.py`: 实现视频的下载、转录、分句、翻译和合成带字幕的视频。
   - `batch/utils/settings_check.py`: 检查输入文件与配置的一致性，确保视频处理设置的正确性。

10. **Streamlit 界面模块**:
    - `st.py`: 基于 Streamlit 框架构建的交互式 Web 应用，实现各处理模块的无缝集成。
    - `st_components/download_video_section.py`: 提供 YouTube 链接下载和本地文件上传两种视频获取方式。
    - `st_components/imports_and_utils.py`: 封装界面组件通用的工具函数库。
    - `st_components/sidebar_setting.py`: 实现基于侧边栏的系统设置界面，提供直观的配置管理。

Videolingo 系统通过这些模块的协同工作，实现了从视频下载到最终生成带有翻译字幕和配音的视频的全流程自动化。系统的模块化设计使得每个步骤都可以独立运行和调试，同时也为未来的功能扩展提供了便利。
