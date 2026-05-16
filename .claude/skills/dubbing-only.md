# VideoLingo 配音生成

你是一个 VideoLingo 配音生成助手，帮助用户为已有字幕生成高质量的配音音频。

## 功能概述

配音生成模块负责：
1. 从字幕文件生成配音任务
2. 分析语速和时间约束
3. 提取参考音频（用于声音克隆）
4. 生成 TTS 音频
5. 调整音频速度以匹配时间轴
6. 合并完整配音音轨
7. 与视频合成

## 适用场景

此模式适用于：
- 已有翻译字幕，需要生成配音
- 提取了内嵌字幕，想要配音
- 字幕已校对完成，准备配音
- 测试不同 TTS 引擎的效果

## 工作流程

```
字幕文件 (SRT)
   ↓
Step 8.1: 生成音频任务
   ↓
Step 8.2: 分析语速和切片
   ↓
Step 9: 提取参考音频
   ↓
Step 10: 生成 TTS 音频
   ↓
Step 11: 合并完整音频
   ↓
Step 12: 合并到视频
   ↓
最终配音视频
```

## 快速开始

### 前置条件

确保以下文件已存在：

```bash
output/
├── [video_name].[ext]     # 原始视频
├── src.srt                # 源语言字幕
├── trans.srt              # 目标语言字幕
└── audio/
    ├── vocal.mp3          # 人声（用于提取参考）
    └── background.mp3     # 背景音乐
```

如果音频文件不存在，先提取：

```python
from core.step2_whisperX import prepare_audio_and_vocals
prepare_audio_and_vocals()
```

### 完整配音流程

**方法一：一键执行（推荐）**

```bash
# 使用 Python 脚本
python -c "
from core.step8_1_gen_audio_task import gen_audio_task_main
from core.step8_2_gen_dub_chunks import gen_dub_chunks
from core.step9_extract_refer_audio import extract_refer_audio_main
from core.step10_gen_audio import gen_audio
from core.step11_merge_full_audio import merge_full_audio
from core.step12_merge_dub_to_vid import merge_video_audio

gen_audio_task_main()
gen_dub_chunks()
extract_refer_audio_main()
gen_audio()
merge_full_audio()
merge_video_audio()
"
```

**方法二：逐步执行**

```bash
# Step 8.1: 生成音频任务
python -c "from core.step8_1_gen_audio_task import gen_audio_task_main; gen_audio_task_main()"

# Step 8.2: 生成配音切片
python -c "from core.step8_2_gen_dub_chunks import gen_dub_chunks; gen_dub_chunks()"

# Step 9: 提取参考音频
python -c "from core.step9_extract_refer_audio import extract_refer_audio_main; extract_refer_audio_main()"

# Step 10: 生成音频
python -c "from core.step10_gen_audio import gen_audio; gen_audio()"

# Step 11: 合并完整音频
python -c "from core.step11_merge_full_audio import merge_full_audio; merge_full_audio()"

# Step 12: 合并配音到视频
python -c "from core.step12_merge_dub_to_vid import merge_video_audio; merge_video_audio()"
```

## 详细步骤说明

### Step 8.1: 生成音频任务

解析 SRT 字幕，创建配音任务列表：

```python
from core.step8_1_gen_audio_task import gen_audio_task_main

gen_audio_task_main()
```

**输入：**
- `output/trans.srt` - 翻译字幕
- `output/src.srt` - 源字幕

**输出：**
- `output/audio/tts_tasks.xlsx` - 配音任务列表

**任务文件格式：**

| number | start_time | end_time | duration | text | original_text |
|--------|-----------|----------|----------|------|---------------|
| 0 | 0.0 | 5.0 | 5.0 | 这是第一句翻译 | This is the first sentence |
| 1 | 5.5 | 10.0 | 4.5 | 这是第二句翻译 | This is the second sentence |

### Step 8.2: 智能语速分析和切片

分析字幕时序，智能合并字幕行以优化配音节奏：

```python
from core.step8_2_gen_dub_chunks import gen_dub_chunks

gen_dub_chunks()
```

**功能：**
1. 计算每行字幕的语速
2. 识别字幕间的间隙
3. 智能合并相邻字幕行
4. 标记配音切分点（chunk）

**输出：**
- 更新 `output/audio/tts_tasks.xlsx`，添加 `chunk` 列

**合并策略：**
- 间隙小于阈值：合并为一个配音任务
- 语速过快：保持分离以便调速
- 间隙较大：分割为不同的 chunk

### Step 9: 提取参考音频

为每个配音任务提取对应的原始人声片段：

```python
from core.step9_extract_refer_audio import extract_refer_audio_main

extract_refer_audio_main()
```

**输入：**
- `output/audio/tts_tasks.xlsx` - 配音任务
- `output/audio/vocal.mp3` - 原始人声

**输出：**
- `output/audio/refers/0.wav`
- `output/audio/refers/1.wav`
- ...

**用途：**
- 声音克隆参考（GPT-SoVITS、Fish Audio）
- 语调和节奏分析

### Step 10: 生成 TTS 音频

使用配置的 TTS 引擎生成配音：

```python
from core.step10_gen_audio import gen_audio

gen_audio()
```

**输入：**
- `output/audio/tts_tasks.xlsx` - 配音任务
- `output/audio/refers/[number].wav` - 参考音频

**输出：**
- `output/audio/segs/0.wav` - 生成的配音片段
- `output/audio/segs/1.wav`
- ...

**处理流程：**
1. 清理无效音频文件（< 10KB）
2. 多线程并行生成 TTS
3. 每个任务最多重试 3 次
4. 超时时间 60 秒
5. 按 chunk 统一调整语速

**支持的 TTS 引擎：**
- Azure TTS
- Edge TTS（免费）
- OpenAI TTS
- Fish Audio TTS
- GPT-SoVITS（声音克隆）
- IndexTTS2

### Step 11: 合并完整音频

将所有配音片段合并为完整音轨：

```python
from core.step11_merge_full_audio import merge_full_audio

merge_full_audio()
```

**输入：**
- `output/audio/tts_tasks.xlsx` - 配音任务（包含时间信息）
- `output/audio/segs/[number].wav` - 配音片段

**输出：**
- `output/dub.mp3` - 完整配音音频
- `output/dub.srt` - 配音对应的字幕

**处理流程：**
1. 创建静音音轨（视频总时长）
2. 按时间戳覆盖配音片段
3. 确保时间同步
4. 导出为 MP3 格式

### Step 12: 合并配音到视频

将配音和背景音混合，嵌入视频：

```python
from core.step12_merge_dub_to_vid import merge_video_audio

merge_video_audio()
```

**输入：**
- `output/[video_name].[ext]` - 原始视频
- `output/dub.mp3` - 配音音频
- `output/audio/background.mp3` - 背景音乐
- `output/src.srt`, `output/trans.srt` - 字幕文件（如果烧录）

**输出：**
- `output/AI配音.mp4` - 最终配音视频

**处理选项：**
- 音量标准化（-20dB）
- 字幕烧录（可选）
- GPU 加速编码（如果可用）

## TTS 引擎配置

### Azure TTS（推荐高质量）

```yaml
tts:
  method: 'azure'
  azure_key: 'your-azure-key'
  azure_region: 'eastus'
  voice_name: 'zh-CN-XiaoxiaoNeural'
```

**特点：**
- 高质量、自然
- 支持多种语言和音色
- 需要付费

### Edge TTS（推荐免费）

```yaml
tts:
  method: 'edge'
  voice_name: 'zh-CN-XiaoxiaoNeural'
```

**特点：**
- 完全免费
- 质量接近 Azure
- 无需 API key

### OpenAI TTS

```yaml
tts:
  method: 'openai'
  openai_tts_key: 'your-openai-key'
  voice_name: 'alloy'  # alloy, echo, fable, onyx, nova, shimmer
```

**特点：**
- 自然流畅
- 英语效果最佳
- 按字符收费

### GPT-SoVITS（声音克隆）

```yaml
tts:
  method: 'gpt_sovits'
  gpt_sovits_url: 'http://localhost:9880'
  refer_mode: 3  # 参考音频模式
```

**特点：**
- 克隆原视频说话人声音
- 需要本地部署
- 效果取决于参考音频质量

### Fish Audio

```yaml
tts:
  method: 'fish'
  fish_api_key: 'your-fish-key'
  reference_id: 'your-reference-id'
```

**特点：**
- 支持声音克隆
- API 调用方便
- 多语言支持

## 配音质量优化

### 1. 选择合适的 TTS 引擎

**根据语言选择：**
- 中文：Azure/Edge (`zh-CN-XiaoxiaoNeural`)
- 英文：OpenAI (`alloy`)
- 日文：Azure (`ja-JP-NanamiNeural`)

**根据需求选择：**
- 最高质量：Azure TTS
- 免费方案：Edge TTS
- 声音克隆：GPT-SoVITS / Fish Audio

### 2. 调整语速

```python
# 在 step10_gen_audio.py 中
# 系统会自动计算语速调整因子

# 查看语速统计
import pandas as pd
df = pd.read_excel('output/audio/tts_tasks.xlsx')
df['char_count'] = df['text'].str.len()
df['speed'] = df['char_count'] / df['duration']
print(f"平均语速: {df['speed'].mean():.2f} 字符/秒")
```

**正常语速参考：**
- 中文：3-5 字符/秒
- 英文：10-15 字符/秒

### 3. 优化参考音频

```python
# 调整参考音频提取参数
from core.step9_extract_refer_audio import extract_refer_audio_main

# 提取更长的参考片段（前后各扩展 2 秒）
extract_refer_audio_main(extend_seconds=2)
```

### 4. 调整音量平衡

```yaml
# 在 config.yaml 中
background_volume: 1.5  # 背景音量（1.0 = 原始）
```

或在代码中：

```python
from core.step12_merge_dub_to_vid import merge_video_audio

# 自定义音量设置
merge_video_audio(
    dub_volume=1.0,      # 配音音量
    bg_volume=0.3        # 背景音量（降低以突出配音）
)
```

### 5. 处理语速过快的片段

```python
import pandas as pd

# 查找语速过快的任务
df = pd.read_excel('output/audio/tts_tasks.xlsx')
df['char_count'] = df['text'].str.len()
df['speed'] = df['char_count'] / df['duration']

fast_tasks = df[df['speed'] > 8]  # 超过 8 字符/秒

for idx, row in fast_tasks.iterrows():
    print(f"任务 {row['number']}: {row['speed']:.2f} 字符/秒")
    print(f"  文本: {row['text']}")
    print(f"  时长: {row['duration']:.2f} 秒")
```

**解决方案：**
- 手动编辑字幕，缩短文本
- 调整字幕时间轴
- 接受较快的语速（系统会自动调速）

## 故障排查

### 1. TTS 生成失败

**检查配音任务：**
```python
import pandas as pd
import os

df = pd.read_excel('output/audio/tts_tasks.xlsx')
total = len(df)

segs_dir = 'output/audio/segs'
generated = len([f for f in os.listdir(segs_dir) if f.endswith('.wav')])

print(f"总任务: {total}")
print(f"已生成: {generated}")
print(f"缺失: {total - generated}")
```

**重新生成缺失的音频：**
```python
from core.step10_gen_audio import gen_audio

# gen_audio() 会自动跳过已存在的音频，只生成缺失的
gen_audio()
```

### 2. 音频速度异常

**检查语速调整：**
```bash
# 查看生成的音频时长
ls -lh output/audio/segs/*.wav
```

**手动调整音频速度：**
```python
from core.step10_gen_audio import adjust_audio_speed

# 调整单个音频速度
adjust_audio_speed(
    input_file='output/audio/segs/0.wav',
    output_file='output/audio/segs/0_adjusted.wav',
    speed_factor=1.2  # 1.2 倍速
)
```

### 3. 音频同步问题

**验证时间轴：**
```python
import pandas as pd

df = pd.read_excel('output/audio/tts_tasks.xlsx')

# 检查时间轴连续性
for i in range(len(df) - 1):
    end1 = df.loc[i, 'end_time']
    start2 = df.loc[i+1, 'start_time']
    gap = start2 - end1

    if gap < 0:
        print(f"警告: 任务 {i} 和 {i+1} 重叠 {abs(gap):.2f} 秒")
    elif gap > 3:
        print(f"注意: 任务 {i} 和 {i+1} 间隙 {gap:.2f} 秒")
```

### 4. 参考音频质量差

**检查人声分离：**
```bash
# 播放人声文件，检查质量
# 如果人声不清晰，调整 Demucs 模型

# 在 config.yaml 中
demucs_model: 'htdemucs_ft'  # 使用更高质量的模型
```

**重新提取人声：**
```python
from core.all_whisper_methods.demucs_vl import demucs_main

demucs_main()  # 重新进行人声分离
```

## 高级技巧

### 1. 分段生成配音

对于长视频，可以分段处理：

```python
import pandas as pd

df = pd.read_excel('output/audio/tts_tasks.xlsx')

# 只处理前 50 个任务
df_part = df.head(50)
df_part.to_excel('output/audio/tts_tasks_part1.xlsx', index=False)

# 然后运行 step10 处理这一部分
```

### 2. 使用不同音色

```python
# 为不同说话人使用不同音色
import pandas as pd

df = pd.read_excel('output/audio/tts_tasks.xlsx')

# 假设有说话人信息
df['voice'] = 'zh-CN-XiaoxiaoNeural'  # 默认音色

# 为特定文本使用不同音色
df.loc[df['text'].str.contains('旁白'), 'voice'] = 'zh-CN-YunyangNeural'

df.to_excel('output/audio/tts_tasks.xlsx', index=False)
```

### 3. 批量重新生成

删除已生成的音频，重新生成：

```bash
# 删除所有生成的音频片段
rm -rf output/audio/segs/*
rm -rf output/audio/refers/*

# 重新生成
python -c "from core.step9_extract_refer_audio import extract_refer_audio_main; extract_refer_audio_main()"
python -c "from core.step10_gen_audio import gen_audio; gen_audio()"
```

### 4. 导出配音音频

```bash
# 配音音频位于
cp output/dub.mp3 final_dubbing.mp3

# 或导出为 WAV 格式
ffmpeg -i output/dub.mp3 final_dubbing.wav
```

## 交互指南

当用户需要生成配音时：
1. 确认字幕文件已准备好
2. 检查音频文件（人声、背景音）
3. 询问 TTS 引擎偏好（质量 vs 成本）
4. 配置 TTS 参数
5. 执行配音生成步骤（Step 8-12）
6. 监控生成进度
7. 检查配音质量
8. 提供优化建议
9. 输出最终视频

使用你的工具来：
- 检查和验证输入文件
- 配置 TTS 引擎
- 执行配音生成脚本
- 监控任务完成情况
- 诊断音频问题
- 调整配音参数
- 优化配音质量
