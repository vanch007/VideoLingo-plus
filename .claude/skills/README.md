# VideoLingo Claude Skills

这是 VideoLingo 项目的 Claude Code Skills 集合，提供智能化的视频翻译和配音工作流程辅助。

## 📋 可用 Skills

### 1. `/video-translate` - 完整视频翻译流程
完整的视频翻译和配音工作流程，从下载到最终输出。

**适用场景：**
- 翻译整个视频并生成配音
- 一站式处理所有步骤
- 首次处理新视频

**触发词：**
- "翻译视频"
- "translate video"
- "视频配音"
- "video dubbing"

### 2. `/subtitle-only` - 仅生成字幕
只生成翻译字幕，不进行配音处理。

**适用场景：**
- 快速生成字幕
- 先检查质量再决定是否配音
- 不需要配音的场景

**触发词：**
- "生成字幕"
- "generate subtitles"
- "只要字幕"

### 3. `/config-helper` - 配置助手
帮助配置系统参数、API keys 和处理选项。

**适用场景：**
- 首次设置系统
- 优化配置参数
- 切换 API 或模型
- 调试配置问题

**触发词：**
- "配置系统"
- "configure"
- "设置参数"
- "setup"

### 4. `/video-analyzer` - 视频分析器
分析处理状态、检查输出文件、诊断问题。

**适用场景：**
- 检查处理进度
- 分析输出质量
- 诊断错误
- 性能分析

**触发词：**
- "分析视频"
- "analyze video"
- "检查状态"
- "查看进度"

### 5. `/batch-process` - 批量处理
批量处理多个视频文件。

**适用场景：**
- 处理多个视频
- 大规模翻译任务
- 自动化工作流

**触发词：**
- "批量处理"
- "batch process"
- "批量翻译"
- "多个视频"

### 6. `/subtitle-extract` - 字幕提取
提取内嵌字幕或处理外部 SRT 文件。

**适用场景：**
- 视频已有字幕轨道
- 使用外部 SRT 文件
- 利用现有字幕节省时间

**触发词：**
- "提取字幕"
- "extract subtitles"
- "内嵌字幕"
- "embedded subtitles"

### 7. `/dubbing-only` - 配音生成
为已有字幕生成配音音频。

**适用场景：**
- 字幕已准备好
- 只需要配音
- 测试不同 TTS 引擎

**触发词：**
- "生成配音"
- "generate dubbing"
- "只配音"

## 🚀 快速开始

### 使用 Skill

在 Claude Code 中，你可以通过以下方式调用 skill：

```bash
# 直接使用斜杠命令
/video-translate

# 或在对话中提及触发词
"帮我翻译这个视频"
"生成字幕"
"配置系统参数"
```

### 典型工作流程

#### 场景 1：翻译新视频（完整流程）

```
1. 使用 /video-translate
2. 提供视频 URL 或本地路径
3. 确认源语言和目标语言
4. 等待处理完成
5. 检查输出文件
```

#### 场景 2：快速生成字幕

```
1. 使用 /subtitle-only
2. 上传或下载视频
3. 选择处理模式（ASR/内嵌/外部SRT）
4. 获得字幕文件
5. （可选）使用 /dubbing-only 生成配音
```

#### 场景 3：使用现有字幕

```
1. 使用 /subtitle-extract
2. 提取内嵌字幕或上传 SRT
3. 使用 /dubbing-only 生成配音
4. 获得最终视频
```

#### 场景 4：批量翻译

```
1. 准备 URLs 列表文件
2. 使用 /config-helper 优化配置
3. 使用 /batch-process 批量处理
4. 使用 /video-analyzer 检查结果
```

## 📁 项目结构

```
VideoLingo-plus/
├── .claude/
│   └── skills/
│       ├── skills.yaml              # Skills 配置文件
│       ├── README.md                # 本文档
│       ├── video-translate.md       # 完整翻译流程
│       ├── subtitle-only.md         # 字幕生成
│       ├── config-helper.md         # 配置助手
│       ├── video-analyzer.md        # 视频分析
│       ├── batch-process.md         # 批量处理
│       ├── subtitle-extract.md      # 字幕提取
│       └── dubbing-only.md          # 配音生成
├── config.yaml                      # 系统配置
├── core/                            # 核心功能模块
│   ├── step1_ytdlp.py              # 视频下载
│   ├── step2_whisperX.py           # 语音识别
│   ├── step3_1_spacy_split.py      # 文本分割
│   ├── step4_2_translate_all.py    # 批量翻译
│   ├── step5_splitforsub.py        # 字幕分割
│   ├── step6_generate_final_timeline.py  # 时间轴生成
│   ├── step8_1_gen_audio_task.py   # 音频任务生成
│   ├── step10_gen_audio.py         # TTS 生成
│   └── step12_merge_dub_to_vid.py  # 视频合成
├── output/                          # 输出目录
│   ├── *.srt                       # 字幕文件
│   ├── AI字幕.mp4                  # 带字幕视频
│   ├── AI配音.mp4                  # 最终配音视频
│   ├── audio/                      # 音频文件
│   └── log/                        # 处理日志
└── st.py                           # Streamlit Web 界面
```

## 🔧 核心配置

主要配置项（`config.yaml`）：

```yaml
# API 设置
api:
  key: 'your-api-key'
  base_url: 'https://api.openai.com'
  model: 'gpt-4'

# 语言设置
source_language: 'en'
target_language: 'zh'

# Whisper 配置
whisper:
  model: 'large-v3-turbo'
  language: 'en'
  runtime: 'stable-ts'

# TTS 配置
tts:
  method: 'azure'  # azure, edge, openai, fish, gpt_sovits
  voice_name: 'zh-CN-XiaoxiaoNeural'

# 音频处理
demucs: true
demucs_model: 'htdemucs'
background_volume: 1.5
```

## 📊 处理步骤概览

VideoLingo 的完整处理流程包括 12 个步骤：

| 步骤 | 模块 | 功能 | 输出 |
|------|------|------|------|
| 1 | step1_ytdlp | 下载视频 | 视频文件 |
| 2 | step2_whisperX | 语音识别 | 转录文本 |
| 3.1 | step3_1_spacy_split | NLP 分割 | 分割句子 |
| 3.2 | step3_2_splitbymeaning | 语义分割 | 优化句子 |
| 4.1 | step4_1_summarize | 生成摘要 | 术语表 |
| 4.2 | step4_2_translate_all | 批量翻译 | 翻译结果 |
| 5 | step5_splitforsub | 字幕分割 | 字幕文本 |
| 6 | step6_generate_final_timeline | 时间轴生成 | SRT 文件 |
| 7 | step7_merge_sub_to_vid | 字幕合并 | 带字幕视频 |
| 8.1 | step8_1_gen_audio_task | 音频任务 | 任务列表 |
| 8.2 | step8_2_gen_dub_chunks | 语速分析 | 切片任务 |
| 9 | step9_extract_refer_audio | 提取参考 | 参考音频 |
| 10 | step10_gen_audio | TTS 生成 | 配音片段 |
| 11 | step11_merge_full_audio | 合并音频 | 完整配音 |
| 12 | step12_merge_dub_to_vid | 视频合成 | 最终视频 |

## 💡 使用技巧

### 1. 优化处理速度

- 使用 `large-v3-turbo` 替代 `large-v3` Whisper 模型
- 选择 `gpt-3.5-turbo` 降低翻译成本
- 使用 Edge TTS（免费）替代 Azure TTS
- 启用 Apple Silicon MLX 加速

### 2. 提升质量

- 使用 `large-v3` 获得最佳转录精度
- 使用 `gpt-4` 或 `gpt-4-turbo` 提升翻译质量
- 编辑 `custom_terms.xlsx` 添加专业术语
- 使用 GPT-SoVITS 进行声音克隆

### 3. 处理大量视频

- 使用 `/batch-process` skill
- 准备 `batch/urls.txt` URL 列表
- 调整并发数和重试参数
- 分批处理以控制资源

### 4. 使用现有资源

- 提取视频内嵌字幕（模式二）
- 上传外部 SRT 文件（模式三）
- 节省 Whisper 转录时间
- 保持原有时间轴精度

## 🐛 故障排查

### 常见问题

**1. API 调用失败**
- 检查 API key 配置
- 验证网络连接
- 查看错误日志

**2. Whisper 转录错误**
- 确认音频文件存在
- 检查语言设置
- 尝试更大的模型

**3. TTS 生成失败**
- 验证 TTS 配置
- 检查参考音频质量
- 尝试其他 TTS 引擎

**4. 字幕不同步**
- 检查 Whisper word-level timestamps
- 验证 SRT 时间格式
- 使用 `/video-analyzer` 诊断

### 获取帮助

使用 `/video-analyzer` skill 进行诊断：
```
/video-analyzer

"分析视频处理状态"
"检查输出质量"
"诊断错误"
```

## 📚 更多资源

- **技术文档**: `tech.zh-CN.md`
- **项目文档**: `docs/`
- **配置模板**: `config.yaml`
- **Web 界面**: `streamlit run st.py`

## 🎯 下一步

1. **首次使用**: 运行 `/config-helper` 设置配置
2. **处理视频**: 使用 `/video-translate` 或 `/subtitle-only`
3. **批量处理**: 准备 URLs 并使用 `/batch-process`
4. **优化质量**: 使用 `/video-analyzer` 分析并优化

## 📝 注意事项

- 确保有足够的磁盘空间（视频文件和中间文件）
- API 调用会产生费用（OpenAI/Azure）
- 处理时间取决于视频长度和配置
- 首次运行会下载模型文件（~1-5GB）
- 建议先用短视频测试配置

---

**VideoLingo** - 智能视频翻译和配音系统
使用 Claude Code Skills 让视频处理更简单！
