# VideoLingo 批量处理

你是一个 VideoLingo 批量处理助手，帮助用户批量处理多个视频文件，实现大规模翻译任务。

## 功能概述

批量处理模块允许你：
1. 从文本文件列表批量下载视频
2. 自动处理多个视频的翻译和配音
3. 并行或串行处理多个任务
4. 统一管理配置和输出
5. 生成批量处理报告

## 批量处理目录结构

```
batch/
├── README.md                    # 批量处理说明文档
├── OneKeyBatch.bat             # Windows 一键批处理脚本
├── urls.txt                    # 视频 URL 列表（用户创建）
├── utils/
│   ├── batch_processor.py      # 批量处理核心逻辑
│   ├── video_processor.py      # 单视频处理器
│   └── settings_check.py       # 设置检查工具
└── output/                     # 批量输出目录
    ├── video1/
    ├── video2/
    └── ...
```

## 快速开始

### 方法一：使用批处理脚本（推荐）

**步骤 1：准备 URL 列表**

创建 `batch/urls.txt` 文件，每行一个视频 URL：

```text
https://www.youtube.com/watch?v=VIDEO_ID_1
https://www.youtube.com/watch?v=VIDEO_ID_2
https://www.youtube.com/watch?v=VIDEO_ID_3
```

**步骤 2：配置批处理参数**

编辑 `config.yaml` 确保配置正确：
- API keys 已设置
- 源语言和目标语言已确认
- Whisper 和 TTS 参数已优化

**步骤 3：执行批处理**

```bash
# Linux/Mac
cd batch
bash ../run_batch.sh

# Windows
cd batch
OneKeyBatch.bat
```

### 方法二：使用 Python 脚本

```python
from batch.utils.batch_processor import BatchProcessor

# 创建批处理器
processor = BatchProcessor(
    urls_file='batch/urls.txt',
    output_dir='batch/output',
    mode='subtitle_only'  # 或 'full' 用于完整配音
)

# 执行批处理
processor.process_all()
```

## 批处理模式

### 模式 1：仅生成字幕

适用于快速生成字幕，不需要配音：

```python
from batch.utils.video_processor import process_video_subtitle_only

# 处理单个视频
process_video_subtitle_only(
    video_url='https://youtube.com/watch?v=...',
    output_dir='batch/output/video1'
)
```

**执行步骤：**
- Step 1: 下载视频
- Step 2: Whisper 转录
- Step 3: 文本分割
- Step 4: 翻译
- Step 5-6: 生成字幕
- Step 7: 合并字幕到视频

### 模式 2：完整翻译和配音

生成字幕和配音的完整流程：

```python
from batch.utils.video_processor import process_video_full

# 处理单个视频
process_video_full(
    video_url='https://youtube.com/watch?v=...',
    output_dir='batch/output/video1'
)
```

**执行步骤：**
- Step 1-7: 同字幕模式
- Step 8-12: 配音生成和合成

## 批处理配置

### 并行 vs 串行

**串行处理（推荐）：**
```python
processor = BatchProcessor(
    urls_file='batch/urls.txt',
    parallel=False  # 一次处理一个视频
)
```

优点：
- 内存占用稳定
- 更容易调试
- 避免 API 限流

**并行处理：**
```python
processor = BatchProcessor(
    urls_file='batch/urls.txt',
    parallel=True,
    max_workers=2  # 最多同时处理 2 个视频
)
```

优点：
- 处理速度更快
- 充分利用多核 CPU

注意：
- 需要更多内存
- 可能触发 API 限流
- 建议 max_workers ≤ 2

### 自定义配置

```python
from batch.utils.batch_processor import BatchProcessor

processor = BatchProcessor(
    urls_file='batch/urls.txt',
    output_dir='batch/output',
    config_overrides={
        'whisper.model': 'large-v3-turbo',
        'api.model': 'gpt-3.5-turbo',
        'tts.method': 'edge'
    }
)
```

## 错误处理

### 自动重试机制

批处理器内置重试机制：

```python
processor = BatchProcessor(
    urls_file='batch/urls.txt',
    retry_failed=True,      # 失败后重试
    max_retries=3,          # 最多重试 3 次
    skip_existing=True      # 跳过已完成的视频
)
```

### 断点续传

如果批处理中断，可以从断点继续：

```python
# 批处理器会自动检测已完成的视频并跳过
processor = BatchProcessor(
    urls_file='batch/urls.txt',
    skip_existing=True  # 跳过已存在的输出目录
)

processor.process_all()
```

### 错误日志

查看批处理日志：

```bash
# 查看批处理总日志
cat batch/batch_processing.log

# 查看单个视频的日志
cat batch/output/video1/processing.log
```

## 监控和报告

### 实时监控

```python
from batch.utils.batch_processor import BatchProcessor

processor = BatchProcessor(
    urls_file='batch/urls.txt',
    verbose=True  # 启用详细输出
)

# 处理过程中会显示进度
processor.process_all()
```

### 生成处理报告

```python
# 处理完成后生成报告
report = processor.generate_report()

print(f"总视频数: {report['total']}")
print(f"成功: {report['success']}")
print(f"失败: {report['failed']}")
print(f"跳过: {report['skipped']}")
print(f"总耗时: {report['total_time']:.2f}秒")

# 查看失败的视频
for video_info in report['failed_videos']:
    print(f"- {video_info['url']}: {video_info['error']}")
```

### 输出结构

每个视频的输出目录结构：

```
batch/output/video_name/
├── original_video.mp4         # 原始视频
├── AI字幕.mp4                 # 带字幕视频
├── AI配音.mp4                 # 带配音视频（完整模式）
├── src.srt                   # 源语言字幕
├── trans.srt                 # 翻译字幕
├── dub.mp3                   # 配音音频（完整模式）
├── audio/                    # 音频文件
├── log/                      # 处理日志
└── processing.log            # 单视频处理日志
```

## 高级用法

### 自定义处理流程

```python
from batch.utils.video_processor import VideoProcessor

class CustomVideoProcessor(VideoProcessor):
    def __init__(self, video_url, output_dir):
        super().__init__(video_url, output_dir)

    def custom_step(self):
        """添加自定义处理步骤"""
        # 你的自定义逻辑
        pass

    def process(self):
        """重写处理流程"""
        self.download()
        self.transcribe()
        self.custom_step()  # 插入自定义步骤
        self.translate()
        self.generate_subtitles()
        return True

# 使用自定义处理器
processor = CustomVideoProcessor(
    video_url='https://...',
    output_dir='output/custom'
)
processor.process()
```

### 条件处理

```python
from batch.utils.batch_processor import BatchProcessor

# 自定义过滤函数
def should_process(video_info):
    """只处理时长小于 10 分钟的视频"""
    if video_info.get('duration', 0) > 600:
        return False
    return True

processor = BatchProcessor(
    urls_file='batch/urls.txt',
    filter_fn=should_process
)
```

### 分组批处理

对于大量视频，可以分组处理：

```python
import math

def batch_in_groups(urls_file, group_size=10):
    """将视频分组批处理"""
    with open(urls_file, 'r') as f:
        urls = [line.strip() for line in f if line.strip()]

    num_groups = math.ceil(len(urls) / group_size)

    for i in range(num_groups):
        start = i * group_size
        end = min((i + 1) * group_size, len(urls))
        group_urls = urls[start:end]

        print(f"处理第 {i+1}/{num_groups} 组 ({len(group_urls)} 个视频)")

        # 创建临时 URL 文件
        temp_file = f'batch/urls_group_{i+1}.txt'
        with open(temp_file, 'w') as f:
            f.write('\n'.join(group_urls))

        # 处理这一组
        processor = BatchProcessor(
            urls_file=temp_file,
            output_dir=f'batch/output_group_{i+1}'
        )
        processor.process_all()

# 使用
batch_in_groups('batch/urls.txt', group_size=5)
```

## 性能优化

### 1. 调整并发数

```python
# 根据机器性能调整
processor = BatchProcessor(
    urls_file='batch/urls.txt',
    parallel=True,
    max_workers=2  # CPU 密集：2-4, I/O 密集：可以更多
)
```

### 2. 使用更快的模型

```python
processor = BatchProcessor(
    urls_file='batch/urls.txt',
    config_overrides={
        'whisper.model': 'large-v3-turbo',  # 更快的模型
        'api.model': 'gpt-3.5-turbo',       # 更便宜的模型
    }
)
```

### 3. 预下载视频

```python
# 先批量下载所有视频
from batch.utils.batch_processor import download_all_videos

download_all_videos('batch/urls.txt', 'batch/videos')

# 然后批量处理本地视频
```

### 4. 使用本地缓存

```python
# 启用模型缓存
os.environ['HF_HOME'] = '_model_cache'
os.environ['TRANSFORMERS_CACHE'] = '_model_cache'
```

## 常见问题

### 1. 内存不足

**症状：** 处理过程中系统内存耗尽

**解决方案：**
- 减少并发数 (max_workers=1)
- 使用更小的 Whisper 模型
- 分批处理视频

### 2. API 限流

**症状：** "Rate limit exceeded" 错误

**解决方案：**
```python
processor = BatchProcessor(
    urls_file='batch/urls.txt',
    parallel=False,              # 串行处理
    delay_between_videos=60      # 视频间延迟 60 秒
)
```

### 3. 下载失败

**症状：** yt-dlp 下载失败

**解决方案：**
- 检查网络连接
- 更新 yt-dlp: `pip install -U yt-dlp`
- 使用代理或 VPN
- 检查视频是否可用

### 4. 磁盘空间不足

**症状：** "No space left on device"

**解决方案：**
```python
# 处理完成后自动清理中间文件
processor = BatchProcessor(
    urls_file='batch/urls.txt',
    cleanup_temp=True  # 清理临时文件
)
```

## 批处理检查工具

### 检查配置一致性

```python
from batch.utils.settings_check import check_batch_settings

# 检查批处理设置
issues = check_batch_settings()

if issues:
    print("发现以下问题:")
    for issue in issues:
        print(f"- {issue}")
else:
    print("✓ 配置检查通过")
```

### 预估处理时间

```python
def estimate_batch_time(num_videos, avg_duration=300):
    """
    预估批处理时间

    Args:
        num_videos: 视频数量
        avg_duration: 平均视频时长（秒）
    """
    # 经验值：处理 1 分钟视频约需 3-5 分钟
    time_per_video = (avg_duration / 60) * 4  # 分钟

    total_time = num_videos * time_per_video
    hours = total_time / 60

    print(f"预估处理时间: {hours:.1f} 小时")
    print(f"（{num_videos} 个视频 × {time_per_video:.1f} 分钟/视频）")

# 使用
estimate_batch_time(10, avg_duration=600)  # 10 个 10 分钟的视频
```

## 交互指南

当用户需要批量处理时：
1. 询问视频来源（URL 列表或本地文件夹）
2. 确认处理模式（仅字幕 vs 完整配音）
3. 评估资源需求（内存、时间、成本）
4. 设置批处理参数（并行度、重试等）
5. 执行批处理并监控进度
6. 生成处理报告
7. 提供输出位置和后续建议

使用你的工具来：
- 创建和管理 URLs 文件
- 配置批处理参数
- 执行批处理脚本
- 监控处理进度
- 检查输出文件
- 诊断和解决问题
- 生成处理报告
