# VideoLingo 字幕提取和处理

你是一个 VideoLingo 字幕提取助手，帮助用户提取视频内嵌字幕或处理外部 SRT 文件。

## 功能概述

字幕提取模块支持三种模式：
1. **模式一**：从音频转录生成字幕（ASR）
2. **模式二**：提取视频内嵌字幕轨道
3. **模式三**：使用外部 SRT 字幕文件

本 skill 主要关注模式二和模式三。

## 模式二：提取内嵌字幕

### 适用场景

- 视频已包含高质量的源语言字幕
- 视频已包含目标语言字幕
- 想要利用现有字幕节省转录时间
- 字幕时间轴已经非常精确

### 工作流程

```
1. 下载视频（确保包含字幕轨道）
   ↓
2. 分析字幕轨道（列出所有可用字幕）
   ↓
3. 选择字幕轨道（源语言和目标语言）
   ↓
4. 提取字幕（导出为 SRT）
   ↓
5. 处理字幕（清洗和对齐）
   ↓
6. 生成配音（可选）
```

### 快速开始

**步骤 1：准备视频**

确保视频包含内嵌字幕（MKV 或 MP4 格式）：

```python
from core.step1_ytdlp import download_video_ytdlp

# 下载时嵌入所有字幕
download_video_ytdlp(
    url='https://youtube.com/watch?v=...',
    save_path='output',
    embed_subs=True  # 确保嵌入字幕
)
```

**步骤 2：查看可用字幕轨道**

```python
from core.step2_extract_subtitles import get_subtitle_tracks

# 获取视频文件路径
from core.step1_ytdlp import find_video_files
video_path = find_video_files()[0]

# 列出所有字幕轨道
tracks = get_subtitle_tracks(video_path)

print("可用的字幕轨道:")
for track in tracks:
    print(f"- 索引 {track['index']}: {track['language']} ({track['codec']})")
    if 'title' in track:
        print(f"  标题: {track['title']}")
```

示例输出：
```
可用的字幕轨道:
- 索引 0: en (subrip)
  标题: English
- 索引 1: zh (subrip)
  标题: Chinese (Simplified)
- 索引 2: ja (subrip)
  标题: Japanese
```

**步骤 3：提取字幕**

```python
from core.step2_extract_subtitles import extract_subtitles_main

# 交互式提取（会提示选择轨道）
extract_subtitles_main()

# 或直接指定轨道索引
from core.step2_extract_subtitles import extract_subtitle_track

# 提取源语言字幕（如英文，索引 0）
extract_subtitle_track(
    video_path=video_path,
    track_index=0,
    output_path='output/raw_src.srt'
)

# 提取目标语言字幕（如中文，索引 1）
extract_subtitle_track(
    video_path=video_path,
    track_index=1,
    output_path='output/raw_trans.srt'
)
```

**步骤 4：处理提取的字幕**

```python
from core.step3_3_process_extracted_subs import process_extracted_subs_main

# 清洗和对齐字幕
process_extracted_subs_main()
```

这会生成：
- `output/src.srt` - 清洗后的源字幕
- `output/trans.srt` - 清洗后的翻译字幕
- `output/src_trans.srt` - 双语字幕（源在上）
- `output/trans_src.srt` - 双语字幕（译在上）

**步骤 5：生成配音（可选）**

如果需要为提取的字幕生成配音：

```bash
# 继续执行配音步骤 (Step 8-12)
python -c "from core.step8_1_gen_audio_task import gen_audio_task_main; gen_audio_task_main()"
python -c "from core.step8_2_gen_dub_chunks import gen_dub_chunks; gen_dub_chunks()"
python -c "from core.step9_extract_refer_audio import extract_refer_audio_main; extract_refer_audio_main()"
python -c "from core.step10_gen_audio import gen_audio; gen_audio()"
python -c "from core.step11_merge_full_audio import merge_full_audio; merge_full_audio()"
python -c "from core.step12_merge_dub_to_vid import merge_video_audio; merge_video_audio()"
```

### 使用 Streamlit 界面

在 Web 界面中：

1. 上传或下载视频
2. 选择"模式二：提取内嵌字幕"
3. 查看可用字幕轨道列表
4. 选择源语言和目标语言轨道
5. 点击"提取并处理"
6. 等待处理完成

## 模式三：使用外部 SRT 文件

### 适用场景

- 已有人工翻译或校对的字幕
- 从其他来源获得的 SRT 文件
- 想要翻译现有字幕
- 为已有字幕生成配音

### 工作流程

```
1. 准备视频文件
   ↓
2. 上传 SRT 字幕文件
   ↓
3. 解析 SRT 并生成中间文件
   ↓
4. 翻译（可选，如果只有源语言字幕）
   ↓
5. 生成配音（可选）
```

### 快速开始

**步骤 1：准备 SRT 文件**

确保 SRT 文件格式正确（UTF-8 编码）：

```srt
1
00:00:00,000 --> 00:00:05,000
This is the first subtitle.

2
00:00:05,500 --> 00:00:10,000
This is the second subtitle.
```

**步骤 2：准备视频和音频**

```python
from core.step2_whisperX import prepare_audio_and_vocals

# 即使使用外部字幕，也需要提取音频用于配音
prepare_audio_and_vocals()
```

这会生成：
- `output/audio/raw.mp3` - 原始音频
- `output/audio/vocal.mp3` - 人声
- `output/audio/background.mp3` - 背景音乐

**步骤 3：处理 SRT 文件**

```python
from core.step2_prepare_from_srt import prepare_from_srt

# 处理上传的 SRT 文件
prepare_from_srt(user_srt_path='path/to/your/subtitle.srt')
```

这会生成：
- `output/log/srt_chunks.xlsx` - SRT 解析结果
- `output/log/cleaned_chunks.xlsx` - 清洗后的字幕块
- `output/log/sentence_by_mark.txt` - 按标点分割的句子

**步骤 4：翻译字幕（如果需要）**

如果 SRT 只包含源语言：

```python
# Step 4.1: 生成摘要和术语表
from core.step4_1_summarize import get_summary
get_summary()

# Step 4.2: 批量翻译
from core.step4_2_translate_all import translate_all
translate_all()

# Step 5: 字幕分割
from core.step5_splitforsub import split_for_sub_main
split_for_sub_main()

# Step 6: 生成最终时间轴
from core.step6_generate_final_timeline import align_timestamp_main
align_timestamp_main()
```

**步骤 5：生成配音（如果需要）**

```bash
# 执行配音步骤 (Step 8-12)
# 同模式二
```

### 使用 Streamlit 界面

在 Web 界面中：

1. 上传视频文件到 `output/` 目录
2. 选择"模式三：提供视频和源字幕"
3. 上传 SRT 文件
4. 选择处理选项：
   - 仅处理字幕
   - 翻译和配音
5. 点击"开始处理"

## 字幕格式和编码

### SRT 格式规范

标准 SRT 格式：

```srt
序号
开始时间 --> 结束时间
字幕文本（可多行）

序号
开始时间 --> 结束时间
字幕文本
```

时间格式：`HH:MM:SS,mmm` （小时:分钟:秒,毫秒）

### 编码要求

- **必须使用 UTF-8 编码**
- 避免使用 BOM (Byte Order Mark)
- 行尾符：LF (`\n`) 或 CRLF (`\r\n`) 均可

### 验证 SRT 文件

```python
import re

def validate_srt(srt_path):
    """验证 SRT 文件格式"""
    with open(srt_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 检查基本格式
    pattern = r'^\d+\s+\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}\s+.+?(?=\n\n|\Z)'
    matches = re.findall(pattern, content, re.MULTILINE | re.DOTALL)

    print(f"找到 {len(matches)} 个字幕条目")

    if not matches:
        print("⚠️  警告: SRT 格式可能不正确")
        return False

    print("✓ SRT 格式验证通过")
    return True

# 使用
validate_srt('path/to/subtitle.srt')
```

### 转换编码

如果 SRT 文件编码不是 UTF-8：

```python
def convert_to_utf8(input_file, output_file, source_encoding='gbk'):
    """转换 SRT 编码为 UTF-8"""
    try:
        with open(input_file, 'r', encoding=source_encoding) as f:
            content = f.read()

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"✓ 已转换为 UTF-8: {output_file}")
        return True
    except Exception as e:
        print(f"✗ 转换失败: {e}")
        return False

# 使用（如果是 GBK 编码的中文字幕）
convert_to_utf8('subtitle_gbk.srt', 'subtitle_utf8.srt', 'gbk')
```

## 字幕清洗

### 清理非对话内容

提取和处理会自动清理：
- 音乐符号 ♪ ♫ 🎵 🎶
- 音效标注 [笑声] [掌声] [音乐]
- HTML 标签 `<i>` `<b>` 等
- 多余的空白和换行

### 自定义清洗规则

```python
import re

def custom_clean_subtitle(text):
    """自定义字幕清洗"""
    # 移除方括号内容
    text = re.sub(r'\[.*?\]', '', text)

    # 移除圆括号内容（可选）
    # text = re.sub(r'\(.*?\)', '', text)

    # 移除多余空白
    text = re.sub(r'\s+', ' ', text).strip()

    # 移除特殊字符
    text = text.replace('♪', '').replace('♫', '')

    return text

# 应用到 SRT 文件
def clean_srt_file(input_srt, output_srt):
    import pysrt
    subs = pysrt.open(input_srt, encoding='utf-8')

    for sub in subs:
        sub.text = custom_clean_subtitle(sub.text)

    subs.save(output_srt, encoding='utf-8')
    print(f"✓ 清洗完成: {output_srt}")

# 使用
clean_srt_file('raw.srt', 'cleaned.srt')
```

## 字幕对齐

### 时间轴对齐

如果源字幕和翻译字幕的时间轴不一致：

```python
import pysrt

def align_subtitles(src_srt, trans_srt, output_srt):
    """
    将翻译字幕的时间轴对齐到源字幕
    """
    src_subs = pysrt.open(src_srt, encoding='utf-8')
    trans_subs = pysrt.open(trans_srt, encoding='utf-8')

    if len(src_subs) != len(trans_subs):
        print(f"警告: 字幕数量不匹配 ({len(src_subs)} vs {len(trans_subs)})")

    # 对齐翻译字幕到源字幕的时间轴
    for i, src_sub in enumerate(src_subs):
        if i < len(trans_subs):
            trans_subs[i].start = src_sub.start
            trans_subs[i].end = src_sub.end

    trans_subs.save(output_srt, encoding='utf-8')
    print(f"✓ 对齐完成: {output_srt}")

# 使用
align_subtitles('output/src.srt', 'output/trans.srt', 'output/trans_aligned.srt')
```

### 时间偏移

如果字幕整体需要时间偏移：

```python
import pysrt
from pysrt import SubRipTime

def shift_subtitles(input_srt, output_srt, shift_seconds):
    """
    偏移字幕时间轴

    Args:
        shift_seconds: 偏移秒数（正数延后，负数提前）
    """
    subs = pysrt.open(input_srt, encoding='utf-8')

    # 转换为毫秒
    shift_ms = int(shift_seconds * 1000)

    subs.shift(milliseconds=shift_ms)
    subs.save(output_srt, encoding='utf-8')

    print(f"✓ 已偏移 {shift_seconds} 秒: {output_srt}")

# 使用（延后 2 秒）
shift_subtitles('input.srt', 'output.srt', 2.0)

# 提前 1.5 秒
shift_subtitles('input.srt', 'output.srt', -1.5)
```

## 常见问题

### 1. 视频没有内嵌字幕

**检查方法：**
```bash
ffprobe -i video.mp4 2>&1 | grep Subtitle
```

**解决方案：**
- 使用模式一（ASR 转录）
- 或使用模式三（提供外部 SRT）

### 2. 字幕轨道语言识别错误

**问题：** 字幕轨道语言标签不正确

**解决方案：**
- 手动查看字幕内容确认语言
- 使用轨道索引而非语言代码

### 3. SRT 解析失败

**常见原因：**
- 编码不是 UTF-8
- 时间格式错误
- 缺少空行分隔

**解决方案：**
```python
# 尝试不同的编码
for encoding in ['utf-8', 'gbk', 'big5', 'shift-jis']:
    try:
        with open('subtitle.srt', 'r', encoding=encoding) as f:
            content = f.read()
        print(f"✓ 成功使用编码: {encoding}")
        break
    except:
        continue
```

### 4. 字幕数量不匹配

**问题：** 源字幕和翻译字幕数量不同

**解决方案：**
- 检查是否有遗漏或重复
- 使用 `process_extracted_subs_main()` 自动对齐
- 必要时手动编辑 SRT 文件

## 工具和库

### 推荐的 Python 库

```bash
pip install pysrt        # SRT 解析和处理
pip install chardet      # 编码检测
```

### SRT 编辑工具

- **Subtitle Edit** (Windows) - 功能强大的字幕编辑器
- **Aegisub** (跨平台) - 专业字幕编辑工具
- **VS Code** + SRT 扩展 - 轻量级编辑

### 在线工具

- **Subtitle Tools** - 在线 SRT 编辑和转换
- **SRT Translator** - 在线字幕翻译
- **Time Sync** - 在线时间轴调整

## 交互指南

当用户需要提取或处理字幕时：
1. 询问字幕来源（内嵌 vs 外部文件）
2. 检查视频是否包含字幕轨道
3. 列出可用的字幕选项
4. 帮助选择合适的轨道或处理 SRT 文件
5. 执行提取和清洗操作
6. 验证输出质量
7. 提供后续处理建议（翻译、配音等）

使用你的工具来：
- 检查视频字幕轨道
- 提取内嵌字幕
- 解析和验证 SRT 文件
- 清洗和对齐字幕
- 转换编码格式
- 诊断格式问题
