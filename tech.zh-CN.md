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
       —— `split_by_comma(text, nlp)`: 按逗号分割文本
       —— 输入文件: `output/log/sentence_by_mark.txt`
       —— 输出文件: `output/log/sentence_by_comma.txt`
     - `split_by_connector.py`: 按连接词分割
       —— `split_by_connectors(text, context_words=5, nlp=None)`: 按连接词分割文本
       —— 输入文件: `output/log/sentence_by_comma.txt`
       —— 输出文件: `output/log/sentence_splitbyconnector.txt`
     - `split_long_by_root.py`: 按根节点分割长句
       —— `split_long_sentence(doc)`: 分割长句子
       —— 输入文件: `output/log/sentence_splitbyconnector.txt`
       —— 输出文件: `output/log/sentence_splitbynlp.txt`
     - `split_by_mark.py`: 按标点分割
       —— `split_by_mark(nlp)`: 按标点分割文本
       —— 输入文件: `output/log/cleaned_chunks.xlsx`
       —— 输出文件: `output/log/sentence_by_mark.txt`
     - `load_nlp_model.py`: 加载NLP模型
       —— `get_spacy_model(language: str)`: 获取指定语言的SpaCy模型

4. **字幕处理与合成模块**:
   - `core/step5_splitforsub.py`: 根据字幕格式规范，对翻译后的文本进行精确分割和时间对齐。
      —— `split_for_sub_main()`: 执行字幕分割主函数，包含多达5次的重试机制以确保所有字幕行都符合长度规范。
      —— `split_align_subs(src_lines, tr_lines)`: 并行处理字幕分割和对齐，支持多线程。
      —— `align_subs(src_sub, tr_sub, src_part)`: 使用GPT模型对齐源字幕和翻译字幕。
      —— `calc_len(text: str)`: 计算文本长度，支持多语言字符权重计算(中文/日文1.75，韩文1.5，其他1)。
      —— 输入文件: `output/log/translation_results.xlsx` (翻译结果)
      —— 输出文件: `output/log/translation_results_for_subtitles.xlsx` (分割后的字幕), `output/log/translation_results_remerged.xlsx` (重新合并的字幕)
   - `core/step6_generate_final_timeline.py`: 生成标准 SRT 格式的字幕文件，包含精确的时间轴信息。
      —— `align_timestamp_main()`: 执行时间轴对齐主函数。
      —— `align_timestamp(df_text, df_translate, ...)`: 对齐时间戳。
      —— `get_sentence_timestamps(df_words, df_sentences)`: 获取句子时间戳。
      —— `clean_translation(x)`: 清理翻译文本。
      —— 输入文件: `output/log/cleaned_chunks.xlsx` (转录结果), `output/log/translation_results_for_subtitles.xlsx` (分割后的字幕)
      —— 输出文件: `output/src.srt`, `output/trans.srt`, `output/src_trans.srt`, `output/trans_src.srt` (各种格式的字幕文件)
   - `core/step7_merge_sub_to_vid.py`: 实现字幕与视频无缝集成，使用 ffmpeg 进行处理。
      —— `merge_subtitles_to_video()`: 合并字幕到视频主函数。
      —— `check_gpu_available()`: 检查GPU可用性。
      —— 输入文件: `output/[video_name].[ext]` (视频文件), `output/src.srt`, `output/trans.srt` (字幕文件)
      —— 输出文件: `output/AI字幕.mp4` (带字幕的视频)

5. **音频处理与配音模块**:
     - `core/step8_1_gen_audio_task.py`: 生成音频任务，处理字幕以确保与时间相符。
       —— `gen_audio_task_main()`: 生成音频任务主函数，会调用 `process_srt`。
       —— `process_srt()`: 处理SRT字幕文件，生成DataFrame。
       —— `merge_short_subtitles(df)`: 智能合并时长过短（低于1.5秒）的字幕行，或在无法合并时延长其时间轴。
       —— `expand_short_text(text, ...)`: 对字符数过少（<=3）的文本，调用LLM进行自然语言扩展。
       —— `check_len_then_trim(text, duration)`: 检查文本预估阅读时长是否超过其在时间轴上的长度，如果超过则调用LLM进行缩减。
       —— 输入文件: `output/trans.srt` (翻译字幕), `output/src.srt` (原始字幕)
       —— 输出文件: `output/audio/tts_tasks.xlsx` (配音任务文件)
     - `core/step9_extract_refer_audio.py`: 提取参考音频，支持Apple Silicon加速。
       —— `extract_refer_audio_main()`: 提取参考音频主函数，它会为每个字幕任务，从人声音轨中提取一个稍长（前后各扩展1秒）的音频片段作为声音克隆的参考。
       —— `extract_audio(audio_data, sr, ...)`: 音频提取核心函数。
       —— 输入文件: `output/audio/tts_tasks.xlsx` (配音任务文件), `output/audio/vocal.mp3` (人声音频)
       —— 输出文件: `output/audio/refers/[number].wav` (参考音频片段)
     - `core/step10_gen_audio.py`: 从文本生成音频文件，支持多线程和异常处理。
       —— `gen_audio()`: 生成音频主函数，是整个模块的入口。
       —— `clean_invalid_audio_files()`: 在开始前清理临时目录中所有小于10KB的无效音频文件。
       —— `generate_tts_audio(tasks_df)`: 核心函数，使用 `ThreadPoolExecutor` 或 `ProcessPoolExecutor` 并行生成TTS音频，包含预热和失败重试逻辑。
       —— `process_row(row, tasks_df)`: 处理单个配音任务，包含复杂的重试（3次）、超时（60秒）、文件大小和时长的验证逻辑，并在生成后对长音频进行静音切除。
       —— `merge_chunks(tasks_df)`: 根据 `step8_2` 划分的区块，计算统一的语速调整因子，并调用 `adjust_audio_speed` 处理每个音频片段。
       —— `adjust_audio_speed(input_file, output_file, speed_factor)`: 使用 `ffmpeg` 调整音频速度。
       —— 输入文件: `output/audio/tts_tasks.xlsx` (配音任务文件), `output/audio/refers/[number].wav` (参考音频片段)
       —— 输出文件: `output/audio/segs/[number].wav` (生成的配音片段)
     - `core/step11_merge_full_audio.py`: 合并完整音频
       —— `merge_full_audio()`: 合并音频主函数。
       —— `merge_audio_segments(audios, new_sub_times, sample_rate)`: 将所有在 `step10` 中生成的、经过速度调整的独立音频片段，根据精确的时间戳覆盖到一个静音的完整时长音轨上。
       —— `create_srt_subtitle()`: 创建配音对应的SRT字幕文件。
       —— `create_orig_srt_subtitle()`: 创建原始文本的SRT字幕文件。
       —— 输入文件: `output/audio/tts_tasks.xlsx` (配音任务文件), `output/audio/segs/[number].wav` (配音片段)
       —— 输出文件: `output/dub.mp3` (合并后的完整配音), `output/dub.srt` (配音字幕)
     - `core/step12_merge_dub_to_vid.py`: 将配音合并到视频，支持多线程加速和音频标准化处理。
       —— `merge_video_audio()`: 合并视频音频主函数，它会混合原始背景音和新的配音音轨，并根据配置决定是否烧录双语字幕。
       —— `normalize_audio_volume(audio_path, ...)`: 标准化音频音量至-20dB。
       —— `check_gpu_available()`: 检查GPU可用性。
       —— 输入文件: 
         - `output/[video_name].[ext]` (原始视频)
         - `output/dub.mp3` (完整配音)
         - `output/background.mp3` (背景音)
         - `output/src.srt`, `output/trans.srt` (字幕文件)
       —— 输出文件: 
         - `output/AI配音.mp4` (带配音的视频)
     - `core/all_tts_functions/`: 文本转语音功能
       - `azure_tts.py`: Azure TTS
         —— `azure_tts(text: str, save_path: str)`: Azure文本转语音
       - `edge_tts.py`: Edge TTS
         —— `edge_tts(text, save_path)`: Edge文本转语音
       - `fish_tts.py`: Fish TTS
         —— `fish_tts(text: str, save_as: str)`: Fish文本转语音
       - `gpt_sovits_tts.py`: GPT-SoVITS TTS
         —— `gpt_sovits_tts(text, text_lang, save_path, ref_audio_path, prompt_lang, prompt_text)`: GPT-SoVITS文本转语音
       - `openai_tts.py`: OpenAI TTS
         —— `openai_tts(text, save_path)`: OpenAI文本转语音
       - `sf_indextts2.py`: IndexTTS2 TTS
         —— `indextts2_tts_for_videolingo(text, save_as, ...)`: IndexTTS2文本转语音
       - `tts_main.py`: TTS主入口
         —— `tts_main(text, save_as, number, task_df)`: TTS主函数
       - `estimate_duration.py`: 音频时长估计
         —— `estimate_duration(text: str, estimator)`: 估计文本转语音时长

6. **系统配置与工具模块**:
   - `core/delete_retry_dubbing.py`: 删除不必要的音频文件以清理生成过程中的多余文件。
     —— 输入文件: `output/audio/segs/` 目录
     —— 输出文件: 无（删除文件）
   - `core/timing_utils.py`: 记录和管理各步骤的耗时统计。
     —— `time_it(step_name)`: 装饰器，计算函数执行时间
     —— `save_timing(step_name, elapsed_time)`: 保存步骤耗时
     —— `get_all_timings()`: 获取所有耗时数据
     —— `get_formatted_timings()`: 获取格式化的耗时数据
     —— `clear_timings()`: 清除耗时数据
     —— 输入文件: 无
     —— 输出文件: `output/log/step_timings.json` (耗时统计数据)
   - `core/config_utils.py`: 读取和更新配置文件。
     —— `load_key(key)`: 读取配置项
     —— `update_key(key, new_value)`: 更新配置项
     —— `get_joiner(language)`: 获取语言的连接符
     —— 输入文件: `config.yaml` (配置文件)
     —— 输出文件: `config.yaml` (更新后的配置文件)
   - `core/onekeycleanup.py`: 提供一键式中间文件清理功能。
     —— `cleanup(history_dir="history")`: 清理中间文件并备份到历史目录
     —— 输入文件: `output/` 目录下的所有文件
     —— 输出文件: `history/[video_name]/` 目录（备份文件）

7. **自然语言处理工具集**:
   - `core/ask_gpt.py`: 封装与 GPT 模型交互的标准化接口，用于各类文本生成和分析任务。
     —— `ask_gpt(prompt, response_json=True, valid_def=None, log_title='default')`: 调用GPT模型并处理响应
     —— `save_log(model, prompt, response, log_title='default', message=None)`: 保存调用日志
     —— 输入文件: 无
     —— 输出文件: `output/gpt_log/[log_title].json` (GPT调用日志)
   - `core/prompts_storage.py`: 集中管理针对不同任务优化的提示模板。
     —— `get_split_prompt()`: 获取分割提示
     —— `get_summary_prompt()`: 获取摘要提示
     —— `get_align_prompt()`: 获取对齐提示
     —— `get_subtitle_trim_prompt()`: 获取字幕修剪提示
     —— 输入文件: 无
     —— 输出文件: 无
   - `core/spacy_utils/`: 封装基于 SpaCy 的句子分割等高级文本处理功能。
     - `split_by_connector.py`: 根据连接词拆分文本句子，提高可读性。
       —— 输入文件: `output/log/sentence_by_comma.txt`
       —— 输出文件: `output/log/sentence_splitbyconnector.txt`
     - `split_by_comma.py`: 根据逗号和冒号拆分文本处理。
       —— 输入文件: `output/log/sentence_by_mark.txt`
       —— 输出文件: `output/log/sentence_by_comma.txt`
     - `split_long_by_root.py`: 按句子根节点拆分长句子，增强文本的可读性。
       —— 输入文件: `output/log/sentence_splitbyconnector.txt`
       —— 输出文件: `output/log/sentence_splitbynlp.txt`
     - `split_by_mark.py`: 用标点符号对文本进行句子拆分。
       —— 输入文件: `output/log/cleaned_chunks.xlsx`
       —— 输出文件: `output/log/sentence_by_mark.txt`
     - `load_nlp_model.py`: 加载和初始化所需的 NLP 模型，支持多种语言。
       —— `get_spacy_model(language: str)`: 获取指定语言的SpaCy模型
       —— 输入文件: 无
       —— 输出文件: 无

8. **文本转语音（TTS）模块**:
   - `core/all_tts_functions/fish_tts.py`: 使用外部 API 实现文本转语音功能，生成音频文件。
     —— `fish_tts(text: str, save_as: str)`: Fish文本转语音
     —— 输入文件: 文本内容
     —— 输出文件: 指定路径的WAV音频文件
   - `core/all_tts_functions/openai_tts.py`: 使用 OpenAI 的 TTS 服务将文本转换为音频并保存。
     —— `openai_tts(text, save_path)`: OpenAI文本转语音
     —— 输入文件: 文本内容
     —— 输出文件: 指定路径的WAV音频文件
   - `core/all_tts_functions/gpt_sovits_tts.py`: 使用 GPT-SoVITS 进行文本到语音转换，支持多语言。
     —— `gpt_sovits_tts(text, text_lang, save_path, ref_audio_path, prompt_lang, prompt_text)`: GPT-SoVITS文本转语音
     —— 输入文件: 文本内容, 参考音频文件
     —— 输出文件: 指定路径的WAV音频文件
   - `core/all_tts_functions/azure_tts.py`: 利用 Azure 语音服务将文本转换为音频，保存为 WAV 格式。
     —— `azure_tts(text: str, save_path: str)`: Azure文本转语音
     —— 输入文件: 文本内容
     —— 输出文件: 指定路径的WAV音频文件
   - `core/all_tts_functions/edge_tts.py`: 使用 Microsoft Edge TTS 进行文本转语音。
     —— `edge_tts(text, save_path)`: Edge文本转语音
     —— 输入文件: 文本内容
     —— 输出文件: 指定路径的WAV音频文件
   - `core/all_tts_functions/sf_indextts2.py`: 使用 IndexTTS2 进行文本转语音，支持参考音频。
     —— `indextts2_tts_for_videolingo(text, save_as, ...)`: TTS主函数，自动处理参考音频获取和格式转换
     —— 输入文件: 文本内容, 参考音频文件
     —— 输出文件: 指定路径的WAV音频文件
   - `core/all_tts_functions/tts_main.py`: TTS主入口，统一调用各种TTS方法。
     —— `tts_main(text, save_as, number, task_df)`: TTS主函数
     —— 输入文件: 文本内容, 配音任务数据
     —— 输出文件: 指定路径的WAV音频文件
   - `core/all_tts_functions/estimate_duration.py`: 音频时长估计工具。
     —— `estimate_duration(text: str, estimator)`: 估计文本转语音时长
     —— 输入文件: 文本内容
     —— 输出文件: 无

9. **系统配置与工具模块**:
   - `config.yaml`: 集中存储和管理系统的全局参数配置。
   - `install.py`: 自动化系统依赖包和模型的安装与配置过程。
   - `onekeycleanup.py`: 提供一键式中间文件清理功能，优化系统存储空间。
   - `core/config_utils.py`: 读取和更新 YAML 配置文件，确保多线程安全的读写操作。

10. **批量处理模块**:
   - `batch/utils/batch_processor.py`: 批量处理视频任务，通过 Excel 配置管理视频处理流程。
     —— `process_batch()`: 批量处理主函数，读取Excel配置并执行任务
     —— `record_and_update_config(source_language, target_language)`: 记录并更新语言配置
     —— 输入文件: `batch/tasks_setting.xlsx` (任务配置文件)
     —— 输出文件: `batch/output/[task_name]/` (处理结果目录)
   - `batch/utils/video_processor.py`: 实现视频的下载、转录、分句、翻译和合成带字幕的视频。
     —— `process_video(file, dubbing=False, is_retry=False)`: 处理单个视频文件的主流程
     —— `prepare_output_folder(output_folder)`: 准备输出目录
     —— `process_input_file(file)`: 处理输入文件(下载或复制)
     —— `split_sentences()`: 执行句子分割
     —— `summarize_and_translate()`: 执行摘要和翻译
     —— `process_and_align_subtitles()`: 处理和对齐字幕
     —— `gen_audio_tasks()`: 生成音频任务
     —— 输入文件: 视频URL或本地文件路径
     —— 输出文件: `batch/output/[task_name]/` (处理结果目录)
   - `batch/utils/settings_check.py`: 检查输入文件与配置的一致性，确保视频处理设置的正确性。
     —— `check_settings()`: 检查批量任务设置的有效性
     —— 输入文件: `batch/tasks_setting.xlsx` (任务配置文件)
     —— 输出文件: 无

11. **Streamlit 界面模块**:
    - `st.py`: 基于 Streamlit 框架构建的交互式 Web 应用，实现各处理模块的无缝集成。
      —— `main()`: 应用主入口，初始化界面布局
      —— `text_processing_section()`: 文本处理界面，展示字幕生成流程
      —— `audio_processing_section()`: 音频处理界面，控制配音流程
      —— `process_text()`: 执行文本处理全流程
      —— `process_audio()`: 执行音频处理全流程
      —— `display_timing_statistics()`: 显示各步骤耗时统计
      —— 输入文件: 视频URL或本地文件路径
      —— 输出文件: `output/` 目录下的处理结果
    - `st_components/download_video_section.py`: 提供 YouTube 链接下载和本地文件上传两种视频获取方式。
      —— `download_video_section()`: 视频下载和上传主界面
      —— `convert_audio_to_video(audio_file: str)`: 将音频文件转换为带黑屏的视频
      —— 输入文件: 视频URL或本地文件路径
      —— 输出文件: `output/` 目录下的视频文件
    - `st_components/imports_and_utils.py`: 封装界面组件通用的工具函数库。
      —— `download_subtitle_zip_button(text: str)`: 打包下载所有字幕文件
      —— `check_api()`: 验证API密钥有效性
      —— 输入文件: `output/` 目录下的字幕文件
      —— 输出文件: 下载的字幕文件压缩包
    - `st_components/sidebar_setting.py`: 实现基于侧边栏的系统设置界面，提供直观的配置管理。
      —— `page_setting()`: 系统配置主界面
      —— `config_input(label, key, help=None)`: 通用配置输入组件
      —— `check_api()`: 验证API连接状态
      —— 输入文件: `config.yaml` (配置文件)
      —— 输出文件: `config.yaml` (更新后的配置文件)

Videolingo 系统通过这些模块的协同工作，实现了从视频下载到最终生成带有翻译字幕和配音的视频的全流程自动化。系统的模块化设计使得每个步骤都可以独立运行和调试，同时也为未来的功能扩展提供了便利。
