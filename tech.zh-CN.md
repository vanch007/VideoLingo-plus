**Videolingo 视频翻译系统技术文档**

## 系统工作流程概述

Videolingo 系统遵循以下核心处理流程：

1. **视频获取**
   - 通过yt-dlp下载视频
   - 清理文件名并保存到output目录

2. **音频处理**
   - 提取视频音频
   - 使用Demucs分离人声
   - 使用WhisperX或stable-ts进行语音识别

3. **文本处理**
   - 使用SpaCy进行初步文本分割
   - 结合GPT模型进行语义分割
   - 生成视频内容摘要和术语表

4. **翻译处理**
   - 三步翻译法(直译、意译、润色)
   - 批量处理字幕文本
   - 确保翻译质量和长度符合要求

5. **字幕处理**
   - 精确分割和对齐字幕
   - 生成SRT格式字幕文件
   - 合并字幕到视频

6. **音频配音**
   - 生成音频任务和时间轴
   - 智能处理语速和合并
   - 提取参考音频
   - 生成TTS音频并调整速度

7. **最终合成**
   - 合并完整配音音频
   - 标准化音量
   - 将配音合并到最终视频

整个流程高度自动化，各模块可独立运行或组合使用，支持批量处理和自定义配置。

### 新功能：使用内嵌字幕的工作流程

此工作流程适用于视频文件已包含高质量、时间轴准确的源语言和目标语言字幕的情况。它通过直接利用现有字幕，绕过了语音识别（Whisper）和机器翻译等耗时步骤，可以更快速、更精确地生成配音任务。

**核心处理流程如下：**

1.  **视频获取与字幕下载 (core/step1_ytdlp.py)**
    -   与原有流程相同，但下载时会配置 `yt-dlp` 以确保下载 MKV 格式并 **嵌入所有可用字幕轨道**。

2.  **音频预处理 (core/step2_whisperX.py)**
    -   此为通用步骤，无论使用何种模式，都需执行以确保后续配音流程所需的文件存在。
    -   `prepare_audio_and_vocals()`: 从视频中提取原始音频 (`raw.mp3`)，并使用 Demucs 分离人声和背景音乐。
    -   **输出文件**: `output/audio/raw.mp3`, `output/audio/vocal.mp3`, `output/audio/background.mp3` (背景音乐，配音合成时需要)。

3.  **字幕轨道提取 (core/step2_extract_subtitles.py)**
    -   当用户在界面选择“使用内嵌字幕”模式后，此模块启动。
    -   `get_subtitle_tracks()`: 使用 `ffprobe` 解析视频文件，向用户展示所有可用的内嵌字幕轨道。
    -   `extract_subtitles_main()`: 根据用户选择的轨道，使用 `ffmpeg` 提取源语言和目标语言的字幕。
    -   **输出文件**: `output/raw_src.srt`, `output/raw_trans.srt`。

4.  **字幕处理与时间轴生成 (core/step3_3_process_extracted_subs.py)**
    -   此模块负责清洗和对齐提取出的两条字幕轨道。
    -   `process_extracted_subs_main()`: 读取 SRT 文件，清理非对话内容，并生成配音所需文件
    -   **输出文件**:`output/src.srt`, `output/trans.srt`, `output/src_trans.srt`, `output/trans_src.srt` (各种格式的字幕文件)

通过以上步骤，新功能实现了对现有字幕的再利用，为后续的配音（Step 9, 10, 11, 12）提供了高质量的文本和时间轴输入，同时保证了音频文件也准备就绪。

### 新功能：使用外部 SRT 字幕的工作流程 (Mode 3)

此模式允许用户上传现有的 SRT 字幕文件，跳过语音识别步骤，直接进行后续的翻译或配音流程。

- **核心模块**: `core/step2_prepare_from_srt.py`
- **功能**: 
  - 解析用户提供的 SRT 文件
  - 生成中间格式文件 `srt_chunks.xlsx` 和 `cleaned_chunks.xlsx`
  - 生成 `sentence_by_mark.txt` 供后续 NLP 处理
- **适用场景**: 已有高质量人工字幕，仅需翻译或配音的情况。

Videolingo 是一个高度集成的视频翻译系统，能够自动化执行视频下载、音频提取、语音识别、字幕生成、文本翻译，以及音视频合成等一系列复杂操作。该系统还提供了一个 Web 界面，用于任务管理和系统配置。

对于开发人员，可以单步执行 `core` 下的每一个 `step__.py` 文件并在 `output` 下检查每一步的输出。

以下是系统的核心技术模块、主要函数以及输入输出文件：

1. **视频获取模块**:
   - `core/step1_ytdlp.py`: 集成`yt-dlp`库，实现从指定 URL 高效下载视频的功能，并清理文件名。
     —— `sanitize_filename(filename)`: 清理文件名中的特殊字符
     —— `download_video_ytdlp(url, save_path='output', resolution='1080', cutoff_time=None)`: 下载视频主函数
     —— `find_video_files(save_path='output')`: 查找下载的视频文件
     —— 输入文件: 无
     —— 输出文件: `output/[video_name].[ext]` (下载的视频文件)

2. **音频处理与语音识别模块**:
   - `core/all_whisper_methods/whisperX_local.py`: 使用本地 WhisperX 模型进行转录。
     —— `transcribe_audio(audio_file: str, start: float, end: float)`: 转录指定时间段的音频
     —— 输入文件: 音频文件片段
     —— 输出文件: 转录结果字典
   - `core/all_whisper_methods/whisperX_302.py`: 使用 302 API 的 whisperX 模型进行转录。
     —— `transcribe_audio_302(audio_file: str, start: float, end: float)`: 转录指定时间段的音频
     —— 输入文件: 音频文件片段
     —— 输出文件: 转录结果字典
   - `core/all_whisper_methods/stable_ts_local.py`: 使用本地 stable-ts 模型进行转录。
     —— `transcribe_audio(audio_file: str, start: float, end: float)`: 转录指定时间段的音频
     —— 输入文件: 音频文件片段
     —— 输出文件: 转录结果字典
   - `core/step2_whisperX.py`: 利用 WhisperX 或 stable-ts 模型进行高精度的语音识别
     —— `prepare_audio_and_vocals()`: 预处理音频，包括视频转音频、人声分离和压缩。
     —— `enhance_vocals(vocals_ratio=2.50)`: (可选)增强人声音量。
     —— `transcribe()`: 执行转录的主函数，它会调用 `split_audio` 对音频分段，然后使用所选的 `runtime` (local, stable-ts, or cloud) 进行并行转录，最后调用 `process_transcription` 和 `save_results` 保存结果。
     —— 输入文件: 
       - `output/[video_name].[ext]` (视频文件)
       - `config.yaml` (配置文件，包含模型路径、语言等参数)
     —— 输出文件: 
       - `output/log/cleaned_chunks.xlsx` (转录结果)
       - `output/audio/raw.mp3` (原始音频)
       - `output/audio/for_whisper.mp3` (处理后的音频)
       - `output/audio/vocal.mp3` (人声)
       - `output/audio/background.mp3` (背景声)
     —— 依赖模块:
       - `core/all_whisper_methods/audio_preprocess.py`
       - `core/all_whisper_methods/demucs_vl.py`
   - `core/all_whisper_methods/audio_preprocess.py`: 音频预处理工具
     —— `compress_audio(input_file: str, output_file: str)`: 压缩音频文件
     —— `convert_video_to_audio(video_file: str)`: 视频转音频
     —— `get_audio_duration(audio_file: str)`: 获取音频时长
     —— `split_audio(audio_file: str, target_len: int)`: 分割长音频
     —— `process_transcription(result: Dict)`: 处理转录结果
     —— `save_results(df: pd.DataFrame)`: 保存转录结果
     —— 输入文件: 视频或音频文件
     —— 输出文件: `output/audio/raw.mp3` (原始音频), `output/audio/for_whisper.mp3` (压缩后的音频)
   - `core/all_whisper_methods/demucs_vl.py`: 音频分离
     —— `demucs_main()`: 执行人声分离主函数
     —— 输入文件: `output/audio/raw.mp3` (原始音频)
     —— 输出文件: `output/audio/vocal.mp3` (人声), `output/audio/background.mp3` (背景音乐)
   - `core/step2_prepare_from_srt.py`: 从 SRT 文件准备上下文 (Mode 3)
     —— `prepare_from_srt(user_srt_path)`: 解析SRT并生成中间文件，跳过Whisper步骤
     —— `parse_srt_to_dataframe(srt_path)`: 解析SRT到DataFrame
     —— 输入文件: 用户提供的SRT文件
     —— 输出文件: `output/log/srt_chunks.xlsx`, `output/log/cleaned_chunks.xlsx`, `output/log/sentence_by_mark.txt`

3. **文本处理与翻译模块**:
   - `core/step3_1_spacy_split.py`: 应用 SpaCy 自然语言处理工具进行初步的文本分割。
     —— `split_by_spacy()`: 执行SpaCy分割主函数
     —— 输入文件: `output/log/cleaned_chunks.xlsx` (转录结果)
     —— 输出文件: `output/log/sentence_splitbynlp.txt` (初步分割的句子)
   - `core/step3_2_splitbymeaning.py`: 结合 GPT 模型的语义理解能力，对长句进行更精确的分割。
     —— `split_sentences_by_meaning()`: 执行语义分割主函数，它会调用 `parallel_split_sentences` 并行处理。
     —— `parallel_split_sentences(sentences, max_length, max_workers, nlp)`: 并行分割多个句子，支持多线程处理。
     —— `split_sentence(sentence, num_parts, word_limit)`: 调用LLM分割单个句子。
     —— 输入文件: `output/log/sentence_splitbynlp.txt` (初步分割的句子)
     —— 输出文件: `output/log/sentence_splitbymeaning.txt` (语义分割后的句子)
   - `core/step3_3_process_extracted_subs.py`: 处理提取的内嵌字幕
     —— `process_extracted_subs_main()`: 清洗和对齐内嵌字幕，生成标准SRT
     —— 输入文件: `output/raw_src.srt`, `output/raw_trans.srt`
     —— 输出文件: `output/src.srt`, `output/trans.srt` 等
   - `core/step4_1_summarize.py`: 利用 GPT 模型对视频内容进行智能摘要，提取关键术语。
     —— `get_summary()`: 获取摘要主函数
     —— `combine_chunks()`: 合并文本块
     —— `search_things_to_note_in_prompt(sentence)`: 搜索需要注意的内容
     —— `valid_summary(response_data)`: 验证摘要结果
     —— 输入文件: `output/log/sentence_splitbymeaning.txt` (语义分割后的句子)
     —— 输出文件: `output/log/terminology.json` (术语和摘要)
   - `core/step4_2_translate_all.py`: 实现批量化的字幕文本翻译处理。
     —— `translate_all()`: 执行批量翻译主函数。
     —— `split_chunks_by_chars(chunk_size=2000, max_i=10)`: 按字符分割文本块。
     —— `translate_chunk(chunk, chunks, theme_prompt, i)`: 翻译单个文本块。
     —— 输入文件: `output/log/sentence_splitbymeaning.txt` (语义分割后的句子), `output/log/terminology.json` (术语和摘要)
     —— 输出文件: `output/log/translation_results.xlsx` (翻译结果)
   - `core/translate_once.py`: 采用三步翻译法（直译、意译和润色）实现高质量的英文到中文的逐句翻译。
     —— `translate_lines(lines, ...)`: 翻译多行文本的核心函数，包含调用 `get_prompt_faithfulness` 和 `get_prompt_expressiveness` 的逻辑。
     —— `retry_translation(prompt, step_name)`: 包含重试逻辑，确保翻译结果的长度和格式正确。
     —— `valid_translate_result(result, ...)`: 验证翻译结果的完整性。
     —— 输入文件: 文本内容
     —— 输出文件: 翻译结果字典
   - `core/spacy_utils/`: SpaCy工具集
     - `split_by_comma.py`: 按逗号分割
     - `split_by_connector.py`: 按连接词分割
     - `split_long_by_root.py`: 按根节点分割长句
     - `split_by_mark.py`: 按标点分割
     - `load_nlp_model.py`: 加载NLP模型

4. **字幕处理与合成模块**:
   - `core/step5_splitforsub.py`: 根据字幕格式规范，对翻译后的文本进行精确分割和时间对齐。
      —— `split_for_sub_main()`: 执行字幕分割主函数，包含多达5次的重试机制以确保所有字幕行都符合长度规范。
      —— `split_align_subs(src_lines, tr_lines)`: 并行处理字幕分割和对齐，支持多线程。
      —— `align_subs(src_sub, tr_sub, src_part)`: 使用GPT模型对齐源字幕和翻译字幕。
      —— `calc_len(text: str)`: 计算文本长度，支持多语言字符权重计算。
      —— 输入文件: `output/log/translation_results.xlsx` (翻译结果)
      —— 输出文件: `output/log/translation_results_for_subtitles.xlsx` (分割后的字幕)
   - `core/step6_generate_final_timeline.py`: 生成标准 SRT 格式的字幕文件，包含精确的时间轴信息。
      —— `align_timestamp_main()`: 执行时间轴对齐主函数。
      —— `align_timestamp(df_text, df_translate, ...)`: 对齐时间戳。
      —— `get_sentence_timestamps(df_words, df_sentences)`: 获取句子时间戳。
      —— 输入文件: `output/log/cleaned_chunks.xlsx` (转录结果), `output/log/translation_results_for_subtitles.xlsx` (分割后的字幕)
      —— 输出文件: `output/src.srt`, `output/trans.srt`, `output/src_trans.srt`, `output/trans_src.srt` (各种格式的字幕文件)
   - `core/step7_merge_sub_to_vid.py`: 实现字幕与视频无缝集成，使用 ffmpeg 进行处理。
      —— `merge_subtitles_to_video()`: 合并字幕到视频主函数。
      —— `check_gpu_available()`: 检查GPU可用性。
      —— 输入文件: `output/[video_name].[ext]` (视频文件), `output/src.srt`, `output/trans.srt` (字幕文件)
      —— 输出文件: `output/AI字幕.mp4` (带字幕的视频)

5. **音频处理与配音模块**:
     - `core/step8_1_gen_audio_task.py`: 生成初始音频任务，解析SRT字幕。
       —— `gen_audio_task_main()`: 生成音频任务主函数，解析SRT并计算时长。
       —— `process_srt()`: 处理SRT字幕文件，生成包含开始时间、结束时间、文本和原始文本的任务DataFrame。
       —— `time_diff_seconds()`: 计算时间差。
       —— 输入文件: `output/trans.srt` (翻译字幕), `output/src.srt` (原始字幕)
       —— 输出文件: `output/audio/tts_tasks.xlsx` (初步配音任务文件)
     - `core/step8_2_gen_dub_chunks.py`: 生成配音切片，处理语速和断句。
       —— `gen_dub_chunks()`: 配音切片生成主函数。
       —— `analyze_subtitle_timing_and_speed(df)`: 分析字幕时序和语速，计算间隙和容忍度。
       —— `process_cutoffs(df)`: 根据间隙和语速标记切分点，智能合并行以优化配音节奏。
       —— `merge_rows(df, ...)`: 合并多行字幕。
       —— 输入文件: `output/audio/tts_tasks.xlsx`, `output/src.srt`, `output/trans.srt`
       —— 输出文件: `output/audio/tts_tasks.xlsx` (更新后的配音任务文件)
     - `core/step9_extract_refer_audio.py`: 提取参考音频，支持Apple Silicon加速。
       —— `extract_refer_audio_main()`: 提取参考音频主函数，它会为每个字幕任务，从人声音轨中提取一个稍长（前后各扩展1秒）的音频片段作为声音克隆的参考。
       —— `extract_audio(audio_data, sr, ...)`: 音频提取核心函数。
       —— 输入文件: `output/audio/tts_tasks.xlsx` (配音任务文件), `output/audio/vocal.mp3` (人声音频)
       —— 输出文件: `output/audio/refers/[number].wav` (参考音频片段)
     - `core/step10_gen_audio.py`: 从文本生成音频文件，支持多线程和异常处理。
       —— `gen_audio()`: 生成音频主函数，是整个模块的入口。
       —— `clean_invalid_audio_files()`: 在开始前清理临时目录中所有小于10KB的无效音频文件。
       —— `generate_tts_audio(tasks_df)`: 核心函数，使用 `ThreadPoolExecutor` 或 `ProcessPoolExecutor` 并行生成TTS音频。
       —— `process_row(row, tasks_df)`: 处理单个配音任务，包含复杂的重试（3次）、超时（60秒）验证逻辑。
       —— `merge_chunks(tasks_df)`: 根据 `step8_2` 划分的区块，计算统一的语速调整因子，并调用 `adjust_audio_speed` 处理每个音频片段。
       —— `adjust_audio_speed(input_file, output_file, speed_factor)`: 使用 `ffmpeg` 调整音频速度。
       —— 输入文件: `output/audio/tts_tasks.xlsx` (配音任务文件), `output/audio/refers/[number].wav` (参考音频片段)
       —— 输出文件: `output/audio/segs/[number].wav` (生成的配音片段)
     - `core/step11_merge_full_audio.py`: 合并完整音频
       —— `merge_full_audio()`: 合并音频主函数。
       —— `merge_audio_segments(audios, new_sub_times, sample_rate)`: 将所有在 `step10` 中生成的、经过速度调整的独立音频片段，根据精确的时间戳覆盖到一个静音的完整时长音轨上。
       —— `create_srt_subtitle()`: 创建配音对应的SRT字幕文件。
       —— 输入文件: `output/audio/tts_tasks.xlsx` (配音任务文件), `output/audio/segs/[number].wav` (配音片段)
       —— 输出文件: `output/dub.mp3` (合并后的完整配音), `output/dub.srt` (配音字幕)
     - `core/step12_merge_dub_to_vid.py`: 将配音合并到视频，支持多线程加速和音频标准化处理。
       —— `merge_video_audio()`: 合并视频音频主函数，它会混合原始背景音和新的配音音轨，并根据配置决定是否烧录双语字幕。
       —— `normalize_audio_volume(audio_path, ...)`: 标准化音频音量至-20dB。
       —— 输入文件: 
         - `output/[video_name].[ext]` (原始视频)
         - `output/dub.mp3` (完整配音)
         - `output/background.mp3` (背景音)
         - `output/src.srt`, `output/trans.srt` (字幕文件)
       —— 输出文件: 
         - `output/AI配音.mp4` (带配音的视频)
     - `core/all_tts_functions/`: 文本转语音功能集 (Azure, Edge, Fish, GPT-SoVITS, OpenAI, IndexTTS2 等)

6. **系统配置与工具模块**:
   - `core/delete_retry_dubbing.py`: 删除不必要的音频文件以清理生成过程中的多余文件。
   - `core/timing_utils.py`: 记录和管理各步骤的耗时统计。
   - `core/config_utils.py`: 读取和更新配置文件。
   - `core/onekeycleanup.py`: 提供一键式中间文件清理功能。
   - `core/pypi_autochoose.py`: 自动选择最快的 PyPI 镜像源。
     —— `test_mirror_speed(name, url)`: 测试镜像速度
     —— `set_pip_mirror(url)`: 设置 pip 全局镜像
   - `core/ask_gpt.py`: 封装与 GPT 模型交互的标准化接口。
   - `core/prompts_storage.py`: 集中管理针对不同任务优化的提示模板。

7. **自然语言处理工具集 (core/spacy_utils/)**:
   - `split_by_connector.py`: 根据连接词拆分文本句子。
   - `split_by_comma.py`: 根据逗号和冒号拆分文本处理。
   - `split_long_by_root.py`: 按句子根节点拆分长句子。
   - `split_by_mark.py`: 用标点符号对文本进行句子拆分。
   - `load_nlp_model.py`: 加载和初始化所需的 NLP 模型。

8. **文本转语音（TTS）模块 (core/all_tts_functions/)**:
   - `fish_tts.py`, `openai_tts.py`, `gpt_sovits_tts.py`, `azure_tts.py`, `edge_tts.py`, `sf_indextts2.py`: 各种TTS实现的封装。
   - `tts_main.py`: TTS主入口，统一调用各种TTS方法。
   - `estimate_duration.py`: 音频时长估计工具。

9. **批量处理模块 (batch/)**:
   - `batch/utils/batch_processor.py`: 批量处理视频任务配置和执行。
   - `batch/utils/video_processor.py`: 实现视频的下载、转录、分句、翻译和合成带字幕的视频。
   - `batch/utils/settings_check.py`: 检查批量任务设置的一致性。

10. **Streamlit 界面模块**:
    - `st.py`: Web 应用入口，集成各处理模块。
    - `st_components/download_video_section.py`: 视频下载和上传组件。
    - `st_components/imports_and_utils.py`: 通用工具函数。
    - `st_components/sidebar_setting.py`: 侧边栏设置组件。

Videolingo 系统通过这些模块的协同工作，实现了从视频下载到最终生成带有翻译字幕和配音的视频的全流程自动化。系统的模块化设计使得每个步骤都可以独立运行和调试，同时也为未来的功能扩展提供了便利。