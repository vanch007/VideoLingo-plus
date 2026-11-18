# 上游 VideoLingo v3.2.0 借鉴分析与落地方案（2026-11-18）

> 范围：只读分析 + 方案，不修改代码。
> 上游：`Huanshere/VideoLingo` main @ `d2d5a04`（v3.2.0，2026-11-13），浅克隆于会话临时目录。
> 本地：`VideoLingo-plus` main @ `f5c550a`（基于上游 2.2.x 分叉，`config.yaml version: 2.2.2`）。
> 取舍原则：**Mac（Apple Silicon）本地运行优先、尽量不依赖 API**。云 ASR / 云 TTS / Windows 安装相关内容一律不建议引入。

---

## 0. 结论速览

本地在「配音」这条线上已大幅领先上游（utterance plan、实测时长的 atomic dubbing、9 个 MLX TTS 适配器、说话人分离、质量门禁、sidechain ducking、工作区隔离），上游在这些方面**没有值得回迁的东西**。

上游近半年的价值集中在**信号链与字幕排版的工程正确性**，恰好对应本地已知缺陷（`docs/reports/fast-plan-audit-20261019/README.md` 中的 P1 字幕截尾）：

| 优先级 | 借鉴点 | 上游位置 | 解决本地什么问题 | 改动量 |
|---|---|---|---|---|
| **P0** | 按渲染宽度排版字幕（HarfBuzz 测宽 + BudouX 断词 + 绝不截尾 + 双语显式堆叠 ASS） | `core/utils/typeset.py`、`subtitle_width.py`、`_5_split_sub.py` | Fast 模式 46 个英文单元截尾、中文专名被切（`Epidemic s`）、`FFmpeg` 跨行拆开、单行超 22 字 | 中-大 |
| **P0** | 提取音频时对齐时间戳 `aresample=async=1:first_pts=0` | `asr_backend/audio_preprocess.py`，`docs/audio-timeline.md` | 时间戳有间隙/重叠的容器（录屏、VFR、网页下载）字幕逐渐滞后 | 极小 |
| **P0** | 两遍响度：门限测量 → 固定增益（真峰值封顶），不做动态压缩 | `_12_dub_to_vid.py`，`docs/dubbing-loudness.md` | 本地单遍 `loudnorm` 为动态模式（会“抽吸”），失败回退到含静音的整文件 dBFS | 小 |
| **P1** | Demucs 产物不经 lameenc：用 FFmpeg libmp3lame 写（或下游全用 WAV） | `asr_backend/demucs_vl.py` | `vocal.mp3`/`background.mp3` 起点约晚 25 ms，影响参考音切片与背景混音对齐 | 小 |
| **P1** | Demucs 分块分离（300 s 块 + 5 s 上下文，流式写盘） | `asr_backend/demucs_vl.py` | 长视频在统一内存上与 oMLX/TTS 抢内存 | 小-中 |
| **P1** | ASR 退化检测：重复环、稀疏字符、文字体系不符 → 缩窗重识别 | `asr_backend/qwen_asr_local.py` | MOSS（LLM 式 ASR）可能复读/漏段；本地 `source_coverage` 只查缺口，不查复读 | 中 |
| **P1** | Silero VAD 只修窗口边缘（两侧留 1 s），无语音窗口保留不丢 | `asr_backend/speech_edges.py` | 本地 `silencedetect -30dB` 切分在有 BGM 时失效 | 小-中 |
| **P1** | 跨任务持久化 ASR 缓存（源字节哈希 + 模型/引擎/版本/schema，逐段落盘） | `asr_backend/transcription_cache.py` | 本地每个新工作区都重跑 MOSS + 对齐 + 分离 | 中 |
| **P2** | 翻译后人工检查点 `pause_after_translate` | `_4_2_translate.py`、`task_runner.py` | 目前只有 `pause_before_translate` | 小 |
| **P2** | `translation_style` 自然语言风格 | `prompts.py` | 无统一风格入口 | 极小 |
| **P2** | “降级但显式告知”的容错模式 | `translate_lines.py`、`_5_split_sub.py`、UI 截断提示 | 审计指出的“空译用源文代替还报 pass” | 小 |
| **P2** | 测试风格：真实 FFmpeg + 合成媒体、禁网、无模型 | `tests/test_audio_timeline.py` 等 | 审计指出 Fast 测试只断言配置值 | 持续 |
| 可选 | 本地 HTTP API（FastAPI，单任务） | `api.py` | 外部自动化接入；本地已有 CLI/批处理守护，非必需 | 中 |
| 可选 | audio.cpp 推理引擎（C++，独立进程） | `qwen_asr_local.py` `_audiocpp` | 与 oMLX 进程隔离、降内存；本地已有 MLX，非必需 | 中 |

**不建议引入**：ElevenLabs / MAI 云 ASR、Edge/OpenAI/Fish/SiliconFlow TTS、`static-ffmpeg`（本地 Homebrew FFmpeg 带 libass/x265/videotoolbox 更合适）、uv 安装器与 Windows 兼容代码、将 Qwen3-ASR 改为主识别（本地实测 MOSS CER 6.3% 优于 Whisper；Qwen 已承担对齐与回读）。

---

## 1. 上游现状概览

- 管线仍是 12 步（`core/_1_ytdlp.py` … `_12_dub_to_vid.py`），共享 `core/pipeline.py` + `core/task_runner.py`（线程 + 暂停/停止事件，`check_cancel()` 在长循环中协作取消）。
- ASR 默认 **Qwen3-ASR 1.7B + Qwen3-ForcedAligner-0.6B**；Apple Silicon 走 mlx-audio，其他平台走 transformers；可选 audio.cpp；WhisperX 降为手动安装。
- 字幕：v3.2.0 引入 1401 行的 `typeset.py`，多语言双语排版、按渲染宽度切行。
- 配音：仍为逐句 TTS + 变速 + 截断（`speed_factor.max` 封顶，放不下的截掉并在 UI 提示），远弱于本地。
- 工程：大量 `docs/*.md` 以“缺陷—修复—验证—不覆盖什么”格式记录信号链修复，测试 30 个文件，强调真实 FFmpeg、无网络、无模型。

---

## 2. 逐项深度分析与落地方案

### 2.1 [P0] 按渲染宽度排版字幕

**上游做法**
- `subtitle_width.py`（270 行，轻量）：按字符所属行（源/译）和字体估算**绘制宽度**，单位 = 1/80 横屏行宽；`wrap_line` 用断点代价 + 行平衡（`_balanced_rows`）。
- `typeset.py`（重量）：
  - 每种书写系统独立选字体和字号；用 **uharfbuzz 真实测量**每个 token 的宽度；
  - **BudouX** 对中/日/泰做词级断行（libass 本身不会在 CJK 词边界断）；连字符、缩写、英文整词不被拆；
  - 译文和原文显式上下堆叠，写成**实际分辨率的 ASS**，用 `ass=...:fontsdir=fonts` 渲染，两条轨永不重叠；复杂文字（印地/泰/阿拉伯）单独走 HarfBuzz shaper。
- `_5_split_sub.py`：判断“过长”用渲染宽度：源行 ≤ 1 行、译行 ≤ 1.65 行，超出才交 LLM 切分，LLM 失败则机械切分——**任何情况下不截尾**。

**本地现状**
- `core/subtitle_layout.py` 用 `len(text)` 计字符数，`compute_subtitle_line_limits` 按宽高比给 CJK/拉丁字符上限。
- `core/fast_subtitles.py:36–75, 98–122` 超长直接截尾（审计 P1）。
- `step7/step12` 用两个独立 `subtitles=...:force_style=` 滤镜叠加源/译 SRT，靠 MarginV 避让，长译文换行后可能与源字幕重叠。

**方案（分两期）**
1. **一期：测宽替换计数**（低风险，解决截尾）
   - 新增 `core/subtitle_measure.py`：参照 `subtitle_width.py`，以 uharfbuzz 加载**本机字体**（`/System/Library/Fonts/PingFang.ttc`、`Helvetica`/`SF`；与 `subtitle_layout.py` 选用的 `PingFang SC` 一致）测宽，BudouX 提供中文词边界。
   - `fast_subtitles.py` 和 `step5_splitforsub` 的“超长”判断改为宽度；超长时**生成新的显示片段**（按词边界 + 声学时钟分配时长），删除截尾分支。
   - 守恒断言：显示片段拼接后 == 原文（去除显示用标点后），写入现有 manifest。
2. **二期：双语 ASS 排版**（解决重叠、统一样式）
   - 由 SRT+force_style 改为生成实际分辨率 ASS（PlayResX/Y = 视频宽高），源/译在同一 Dialogue 块内显式定位堆叠。
   - 竖屏沿用本地 `test_portrait_subtitle_layout.py` 的规则作为几何输入。
   - 保留 `bilingual_alignment.py` 的配对结果作为输入，排版器只负责“画”，不改词。
- 依赖：`uharfbuzz`、`budoux`、`fonttools`，全部纯本地 wheel，无需联网；Mac 无需像上游那样下载字体。
- 验证：以审计中的 46 个英文单元 + 149 号中文单元为回归样本，断言全文守恒、无跨词断行、单行宽度 ≤ 上限；渲染后用 `ffmpeg -vf ass` 抽帧比对。

### 2.2 [P0] 音频提取时钟对齐

**上游**：`audio-timeline.md` 证明容器包时间戳有间隙/重叠时，直接拼接解码样本会让识别时钟“变长”，字幕越来越晚（实测 >0.8 s）。修复只是在所有提取命令加 `-af aresample=async=1:first_pts=0`。

**本地**：`core/all_whisper_methods/audio_preprocess.py:31–35` 提取 `raw_master.wav` 只用 `-c:a pcm_s16le`，没有时钟对齐；`raw.mp3` 由 master 派生，继承问题。`raw_master.wav` 同时是 MOSS、Demucs、源核验、背景混音的源头，影响面最大。

**方案**
- `convert_video_to_audio` 两条命令加 `-af aresample=async=1:first_pts=0`；`compress_audio` 同理。
- 因 `raw_master.wav` 哈希参与指纹（`step2_whisperX.py`），改动后旧缓存自动失效，符合预期；在 `pipeline_policy_version` 上加一次版本号以明示。
- 验证：移植上游 `tests/test_audio_timeline.py` 思路——用 FFmpeg 合成带重复间隙/重叠的 PCM 媒体，断言提取时长 == 源帧呈现时长（误差 ≤1 帧）。

### 2.3 [P0] 两遍响度标准化

**上游**：第一遍 `loudnorm=I=-20:TP=-1:LRA=11:print_format=json` 只测量；第二遍只施加**固定增益**，且增益受真峰值 -1 dBTP 封顶（峰值优先于响度目标）；静音/过短音频增益为 0；FFmpeg 失败直接抛错。

**本地**：`core/step12_merge_dub_to_vid.py:71–85` 单遍 `loudnorm=I=-18:TP=-1:LRA=11` 直接输出 → FFmpeg 进入**动态模式**，会压缩句间动态、产生“忽远忽近”；失败时回退到 pydub `audio.dBFS`（整文件 RMS，含静音）——正是上游 #533 已修掉的问题。

**方案**
- `normalize_audio_volume` 改为：测量 `input_i / input_tp` → `gain = min(target - input_i, -1 - input_tp)` → `volume={gain}dB`；无法测量时增益 0 并在 manifest 记录；删除 dBFS 回退，失败即报错。
- 仅作用于配音人声轨，在现有 EQ + sidechain ducking + alimiter 之前，ducking 链不变。
- 验证：参照上游 `tests/test_dubbing_loudness.py`——合成“语音+长静音”断言活动段电平差 <0.3 dB、12 dB 动态被保留、峰值 ≤ -1 dBTP。

### 2.4 [P1] Demucs 编码延迟与分块

**问题 1：MP3 编码延迟**。Demucs `save_audio` 用 lameenc 写 MP3，没有 LAME/Xing 头，解码器无法裁掉约 25 ms priming。本地仍在用这些 MP3：
- `core/atomic_dubbing.py:86`、`core/tts_reference_plan.py:79` 从 `vocal.mp3` 切参考音；
- `core/step12_merge_dub_to_vid.py` 用 `BACKGROUND_AUDIO_FILE`（MP3）混音；
- `core/step2_whisperX.py:108` 增强后的人声 MP3 可能进入 ASR。

**方案**：下游统一改读已生成的 `vocal.wav` / `background.wav`（`step9_extract_refer_audio.py:53` 已这样做）；若必须保留 MP3，改用 FFmpeg libmp3lame 写出。验证：互相关测 `vocal` 与 `raw_master` 偏移 ≤1 个 16 kHz 样本。

**问题 2：内存**。本地 `separator.separate_audio_file()` 一次处理全片。上游 `CHUNK_SECONDS=300`、`CONTEXT_SECONDS=5`：每块两侧多听 5 s、丢弃两侧输出、流式写盘。长片（>30 分钟）在 M 系列统一内存上与 oMLX 35B 模型同时驻留时收益明显。
**方案**：移植 `separate_in_chunks` + `StemWriter` 逻辑，块长做成 `demucs_chunk_seconds` 配置；校验分块与整段结果在块交界处的差异（SNR）后再默认开启。

### 2.5 [P1] ASR 退化检测与缩窗重试

**上游 `qwen_asr_local.py` 的防护（可与模型无关地复用）**
- 复读：`(.{1,40}?)\1{3,}` + 覆盖率 ≥50%（`REPEAT_MIN_CHARS=40`）；
- 漏识别：字符密度 < `SPARSE_CHARS_PER_SECOND=2.0` 而 VAD 显示有语音；
- 文字体系不符：按 Han/Kana/Hangul 加权统计，判断中日混淆或识别成错误语言；
- 处置：触发后以 `RETRY_WINDOW_SECONDS=60` 缩窗重识别；token 预算按 `13 token/s + 32` 动态给，避免长窗口截断或放任复读。

**本地**：MOSS 按 `source_asr.chunk_seconds: 180` 切块，`max_new_tokens` 固定 2048；`core/providers/source_coverage.py` 能发现**时间缺口**与时间戳幻觉，但不检测**文本复读**和**语种错乱**。

**方案**
- 新建 `core/providers/asr_guards.py`（纯函数：`is_repetitive / is_sparse / script_mismatch`），在 MOSS 每块结果返回后调用；不合格块缩到 60 s 重跑一次，仍失败则标记 `needs_review` 而非静默接受。
- `max_new_tokens` 改为按块时长计算（上限仍用配置值）。
- 验证：构造复读文本、空文本、日文假名比例异常的单元测试；用历史任务中已知问题段做回放。

### 2.6 [P1] VAD 修边

**上游 `speech_edges.py`**：Silero VAD（CPU、几 MB、随 pip 包分发）只裁每个窗口的**首尾**，两侧各留 1.0 s；窗口内部停顿不删；整窗无语音时**原样保留**（防止 VAD 漏检轻声）；返回的索引仍是原始时间轴。另外注意它对 `torch.set_num_threads` 的副作用做了恢复。

**本地**：`audio_preprocess.split_audio` 用 `silencedetect=n=-30dB:d=0.5` 找切点，背景音乐持续时找不到静音点。Nemotron 分离已有 VAD 能力，但没有用于切窗。

**方案**：切窗优先使用 Silero VAD 的语音区间找切点（在目标长度附近 ±30 s 内找最长非语音间隙），找不到再回退 silencedetect；首尾修边仅用于减少 MOSS 对片头音乐的幻觉。可先用 Nemotron 已产出的语音区间，避免新增依赖。

### 2.7 [P1] 跨任务持久 ASR 缓存

**上游**：`.cache/asr/` 位于 `output/` 之外；键 = 源文件 MD5 + 语言 + 后端 + 模型尺寸 + 解析后的引擎 + 是否 Demucs + 相关包版本 + schema；**逐段写入**，失败/取消/残缺 JSON 不复用；完整命中时不加载模型。

**本地**：指纹做得比上游更严（含 review 文件、模型权重哈希），但存在工作区内；`run_workspaces` 每个新任务都会对同一视频重跑 MOSS、对齐、分离、说话人分离（长片可达数十分钟）。

**方案**
- 新建全局 `~/.cache/videolingo-plus/asr/<key>/`（或仓库根 `.cache/asr/`），键沿用本地现有指纹字段（`source_sha256`、模型路径 + 权重哈希、aligner、chunk 配置、`raw_master` 提取参数、schema）。
- 缓存对象：MOSS 分段原文、forced-align 词时间、Demucs 产物、Nemotron 结果，各自独立键，逐块落盘。
- 只缓存“模型输出”，不缓存人工 review 后的结果；review 仍在工作区内叠加，保证可审计。
- 提供 `cli.py cache --stats/--purge`，无自动淘汰。

### 2.8 [P2] 翻译后人工检查点

**上游**：`pause_after_translate: true` 时翻译后暂停，用户改 `translation_results.xlsx` 的 Translation 列（不动 Source 列与行数），`/resume` 继续；`TaskRunner.pause_message` 告诉 UI 为什么停。

**本地适配**：在 `step4_2_translate_all` 之后、atomic dubbing 实测之前加检查点；恢复时对比每行文本哈希，**仅改动行**的实测时长/回读证据作废，其余复用（本地已有身份绑定机制，天然支持）。Streamlit 与 CLI `resume` 共用 `pipeline_state.json` 的 `paused_reason` 字段。

### 2.9 [P2] `translation_style`

一行配置，原样注入翻译 prompt 的共享部分（如“口语化、短句、保留英文术语”）。本地 Index 走 instTrans 规范 prompt，需在 `local_translation.py` 的约束段追加，并把该字段纳入请求缓存键与翻译 provenance。

### 2.10 [P2] “降级但显式告知”

上游的几个模式：润色失败保留直译、LLM 切分失败改机械切分、“字幕行两侧都切或都不切”、配音截断时 UI 显示被截断的行。共同点是**不让单点失败中断整片，但一定可见**。

本地严格模式应维持“失败即停”；Fast 模式可采用这一模式，但需修正审计中的反例：降级必须写入 manifest 的 `degraded[]` 并使 `quality_gate_passed=false / pending`，UI 显示降级数。

### 2.11 [P2] 测试方法

上游信号链测试的共同约束：真实 FFmpeg、合成媒体、禁用 socket、无模型、断言可测信号属性（时长、偏移、电平、频带），并在文档中写明“不覆盖什么”。建议为 2.2–2.4 各配一个这样的测试，补上审计指出的“只断言配置值”缺口。

### 2.12 可选项

- **本地 HTTP API**（`api.py`，FastAPI，`/input /run /status /resume /archive`，单任务、409 忙碌、仅 localhost）：如需让外部 Agent/脚本驱动，可在本地 `core/pipeline/job.py` 之上包一层；本地已有 `run_daemon_batch.sh` 与 `scripts/monitor_batch.py`，暂不必要。
- **audio.cpp 引擎**：独立 C++ 服务，可让 ASR 与 Python 进程、oMLX 内存隔离；本地 MOSS 已以独立 conda 子进程运行，收益有限，关注即可。
- **安装自检**：上游 `installer.py --check` 检查 FFmpeg 必需滤镜与编码器（libass、aac、libmp3lame、h264）。本地 `core/doctor.py` 可补“`ffmpeg -filters` 含 `ass/subtitles/loudnorm/sidechaincompress`”这一项，防止换 FFmpeg 后静默失败。

---

## 3. 实施路线

| 阶段 | 内容 | 预估 | 验收 |
|---|---|---|---|
| 1 | 2.2 时钟对齐、2.3 两遍响度、2.4 Demucs 下游改用 WAV | 0.5–1 天 | 合成媒体测试；以历史成片比较活动段电平与参考切片偏移 |
| 2 | 2.1 一期：测宽 + 不截尾（Fast 与严格共用） | 1–2 天 | 审计 46+3 个单元全文守恒、无跨词断行 |
| 3 | 2.5 ASR 退化防护、2.6 VAD 切窗 | 1–2 天 | 问题段回放；单测 |
| 4 | 2.7 全局 ASR/分离缓存、2.4 Demucs 分块 | 1–2 天 | 同视频二次任务跳过 MOSS/Demucs；长片峰值内存对比 |
| 5 | 2.1 二期双语 ASS、2.8/2.9/2.10 | 2–3 天 | 抽帧目检；检查点增量复用 |

阶段 1 与 2 互不依赖可并行；阶段 4 的缓存键依赖阶段 1 的提取参数定稿，应在其后。

## 4. 风险

- 2.2 会改变 `raw_master.wav` 哈希，使所有现存工作区的 ASR 指纹失效；已归档成片不受影响，进行中的任务需决定是否沿用旧音频。
- 2.3 响度结果会与历史交付不同（不再动态压缩），需要以听感确认 -18 LUFS 目标是否仍合适。
- 2.1 换测宽后字幕分段数变化，会触发显示层 manifest 重算；不应触发 TTS 重合成（本地已分离显示与 TTS 单元）。
- 上游与本地均为 Apache-2.0，移植代码需保留版权与出处说明。

## 5. 证据位置

- 上游关键文件：`core/utils/typeset.py`、`core/utils/subtitle_width.py`、`core/_5_split_sub.py`、`core/asr_backend/{qwen_asr_local,speech_edges,transcription_cache,demucs_vl,audio_preprocess}.py`、`core/_12_dub_to_vid.py`、`core/task_runner.py`、`docs/{audio-timeline,dubbing-loudness,transcription-cache,audio-extract-quality}.md`。
- 本地对照：`core/all_whisper_methods/audio_preprocess.py:26–48`、`core/all_whisper_methods/demucs_vl.py:76–92`、`core/step12_merge_dub_to_vid.py:71–85`、`core/subtitle_layout.py`、`core/fast_subtitles.py`、`core/providers/moss_asr.py:118–193`、`core/providers/source_coverage.py`、`core/step2_whisperX.py:164–178`。
