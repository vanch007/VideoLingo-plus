import time
import os
import json
from typing import Dict, List, Optional
from functools import wraps

# 存储所有步骤的耗时
_DEFAULT_TIMING_FILE = "output/log/step_timings.json"
TIMING_FILE = _DEFAULT_TIMING_FILE


# ========== SRT 时间转换工具函数 ==========
# 这些函数用于统一处理各模块中的时间转换逻辑

def srt_time_to_seconds(time_str: str) -> float:
    """
    将 SRT 时间格式转换为秒数。
    
    支持格式：
    - HH:MM:SS,ms (标准 SRT)
    - HH:MM:SS.ms (常见变体)
    - HH:MM:SS (无毫秒)
    
    Args:
        time_str: 时间字符串
        
    Returns:
        float: 秒数
        
    Examples:
        >>> srt_time_to_seconds("00:01:30,500")
        90.5
        >>> srt_time_to_seconds("01:00:00.000")
        3600.0
    """
    if not isinstance(time_str, str):
        return 0.0
    
    time_str = time_str.strip().replace(',', '.')
    
    try:
        if '.' in time_str:
            main_part, ms_part = time_str.rsplit('.', 1)
            h, m, s = main_part.split(':')
            # 补齐毫秒部分到3位
            ms = int(ms_part.ljust(3, '0')[:3])
            return int(h) * 3600 + int(m) * 60 + int(s) + ms / 1000.0
        else:
            # 无毫秒
            h, m, s = time_str.split(':')
            return float(int(h) * 3600 + int(m) * 60 + int(s))
    except (ValueError, AttributeError):
        return 0.0


def seconds_to_srt_time(seconds: float, use_comma: bool = True) -> str:
    """
    将秒数转换为 SRT 时间格式。
    
    Args:
        seconds: 秒数
        use_comma: 是否使用逗号分隔毫秒（SRT标准为逗号）
        
    Returns:
        str: 格式化的时间字符串 (HH:MM:SS,ms 或 HH:MM:SS.ms)
        
    Examples:
        >>> seconds_to_srt_time(90.5)
        '00:01:30,500'
        >>> seconds_to_srt_time(90.5, use_comma=False)
        '00:01:30.500'
    """
    seconds = max(0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds * 1000) % 1000)
    
    separator = ',' if use_comma else '.'
    return f"{h:02d}:{m:02d}:{s:02d}{separator}{ms:03d}"


def time_to_samples(time_str: str, sample_rate: int) -> int:
    """
    将时间字符串转换为音频采样点位置。
    
    Args:
        time_str: SRT格式时间字符串
        sample_rate: 音频采样率 (如 44100, 48000)
        
    Returns:
        int: 采样点位置
    """
    seconds = srt_time_to_seconds(time_str)
    return int(seconds * sample_rate)


def ensure_timing_file():
    """确保计时文件存在"""
    global TIMING_FILE
    try:
        # Check if we're being called during cleanup
        import traceback
        stack = traceback.extract_stack()
        caller_files = [frame[0] for frame in stack]
        in_cleanup = any('onekeycleanup.py' in file for file in caller_files)

        if in_cleanup:
            print(f"⚠️ ensure_timing_file() called during cleanup process - skipping file creation")
            return

        os.makedirs(os.path.dirname(TIMING_FILE), exist_ok=True)
        if not os.path.exists(TIMING_FILE):
            with open(TIMING_FILE, 'w', encoding='utf-8') as f:
                json.dump({}, f)
            print(f"📊 Created new timing file: {TIMING_FILE}")
        # 测试文件是否可写
        elif os.path.exists(TIMING_FILE):
            with open(TIMING_FILE, 'r+', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    # 文件内容无效，重置为空对象
                    f.seek(0)
                    f.truncate()
                    json.dump({}, f)
                    print(f"📊 Reset invalid timing file: {TIMING_FILE}")
    except (IOError, PermissionError) as e:
        print(f"警告: 无法访问计时文件 {TIMING_FILE}: {str(e)}")
        # 尝试使用临时目录
        TIMING_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "temp_timings.json")
        os.makedirs(os.path.dirname(TIMING_FILE), exist_ok=True)
        with open(TIMING_FILE, 'w', encoding='utf-8') as f:
            json.dump({}, f)
        print(f"已切换到临时文件: {TIMING_FILE}")

def get_all_timings() -> Dict[str, float]:
    """获取所有步骤的耗时"""
    # Check if we're being called during cleanup
    import traceback
    stack = traceback.extract_stack()
    caller_files = [frame[0] for frame in stack]
    in_cleanup = any('onekeycleanup.py' in file for file in caller_files)

    if not in_cleanup:
        ensure_timing_file()

    try:
        if os.path.exists(TIMING_FILE):
            with open(TIMING_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            if in_cleanup:
                print(f"⚠️ get_all_timings() called during cleanup but file {TIMING_FILE} doesn't exist - returning empty dict")
            else:
                print(f"警告: 计时文件 {TIMING_FILE} 不存在")
            return {}
    except (json.JSONDecodeError, FileNotFoundError, IOError, PermissionError) as e:
        print(f"警告: 无法读取计时文件: {str(e)}")
        return {}

# 主要步骤列表，只记录这些步骤的耗时
MAIN_STEPS = [
    "step1_ytdlp",
    "step2_whisperX",
    "step3_1_spacy_split",
    "step3_2_splitbymeaning",
    "step4_1_summarize",
    "step4_2_translate_all",
    "step5_splitforsub",
    "step6_generate_final_timeline",
    "step7_merge_sub_to_vid",
    "step8_1_gen_audio_task",
    "step8_2_gen_dub_chunks",
    "step9_extract_refer_audio",
    "step10_gen_audio",
    "step11_merge_full_audio",
    "step12_merge_dub_to_vid",
    "转录",
    "NLP分句",
    "LLM分句",
    "摘要",
    "翻译",
    "字幕分割",
    "时间轴对齐",
    "字幕合并到视频",
    "生成配音任务",
    "生成配音分块",
    "提取参考音频",
    "生成配音",
    "合并配音",
    "配音合并到视频",
    "整体字幕处理",
    "整体配音处理",
    "下载视频",
    "上传视频",
    "项目开始时间",
    "项目总耗时",
]

def save_timing(step_name: str, elapsed_time: float):
    """保存步骤耗时"""
    global TIMING_FILE

    # Check if we're being called during cleanup
    import traceback
    stack = traceback.extract_stack()
    caller_files = [frame[0] for frame in stack]
    in_cleanup = any('onekeycleanup.py' in file for file in caller_files)

    if in_cleanup:
        print(f"⚠️ save_timing() called during cleanup process - skipping for {step_name}")
        return False

    # 特殊处理项目开始时间，它存储的是时间戳而不是耗时
    if step_name == "项目开始时间":
        # 确保计时文件存在
        ensure_timing_file()
        # 获取当前所有耗时数据
        timings = get_all_timings()
        # 更新当前步骤的耗时
        timings[step_name] = elapsed_time
        # 写入文件
        with open(TIMING_FILE, 'w', encoding='utf-8') as f:
            json.dump(timings, f, indent=2)
        return True

    # 检查是否是主要步骤
    is_main_step = False
    for main_step in MAIN_STEPS:
        if main_step in step_name:
            is_main_step = True
            step_name = main_step  # 使用标准化的步骤名称
            break

    # 如果不是主要步骤，则不记录
    if not is_main_step:
        return False

    # 确保计时文件存在
    ensure_timing_file()

    # Check if the file exists after ensure_timing_file
    # This is to prevent recreating the file if it was just moved during cleanup
    if not os.path.exists(TIMING_FILE):
        print(f"⚠️ Timing file {TIMING_FILE} does not exist after ensure_timing_file() - possible cleanup in progress")
        return False

    try:
        # 获取当前所有耗时数据
        timings = get_all_timings()
        # 更新当前步骤的耗时
        timings[step_name] = elapsed_time

        # 尝试写入文件
        try:
            # Double-check that the file hasn't been moved during cleanup
            if not os.path.exists(os.path.dirname(TIMING_FILE)):
                print(f"⚠️ Directory {os.path.dirname(TIMING_FILE)} does not exist - possible cleanup in progress")
                return False

            with open(TIMING_FILE, 'w', encoding='utf-8') as f:
                json.dump(timings, f, indent=2)
        except (IOError, PermissionError) as e:
            # 如果写入失败，尝试切换到临时文件
            print(f"警告: 无法写入耗时文件 {TIMING_FILE}: {str(e)}")
            TIMING_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "temp_timings.json")
            with open(TIMING_FILE, 'w', encoding='utf-8') as f:
                json.dump(timings, f, indent=2)
            print(f"已切换到临时文件: {TIMING_FILE}")

        # 打印耗时信息
        formatted_time = format_time(elapsed_time)
        print(f"⏱️ {step_name} 完成，耗时: {formatted_time}")
        return True
    except Exception as e:
        print(f"警告: 保存耗时数据时出错: {str(e)}")
        return False

def clear_timings():
    """清除所有耗时记录"""
    global TIMING_FILE
    try:
        if os.path.exists(TIMING_FILE):
            os.remove(TIMING_FILE)
            print(f"耗时统计文件 {TIMING_FILE} 已清除")
        # 重置为默认文件路径
        TIMING_FILE = _DEFAULT_TIMING_FILE
        # 创建新的空文件
        ensure_timing_file()
    except (IOError, PermissionError) as e:
        print(f"警告: 无法清除耗时文件: {str(e)}")

def time_it(step_name: Optional[str] = None):
    """装饰器：计算函数执行时间并保存

    Args:
        step_name (str, optional): 步骤名称。如果不提供，则使用函数名。
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 使用函数名作为步骤名称，如果没有提供
            name = step_name or func.__name__

            # Check if this is a main step that should be timed
            # We don't need to store this result as save_timing will do the check again
            # This is just for debugging and understanding the code flow
            for main_step in MAIN_STEPS:
                if main_step in name or main_step in func.__name__:
                    # Found a main step, will be timed
                    break

            try:
                start_time = time.time()
                result = func(*args, **kwargs)
                elapsed_time = time.time() - start_time

                # 保存耗时（只有主要步骤才会被保存）
                save_timing(name, elapsed_time)
            except Exception as e:
                print(f"警告: 计时装饰器出错: {str(e)}")
                # 重新抛出异常，不影响原函数的执行
                raise
            return result
        return wrapper
    return decorator

def format_time(seconds: float) -> str:
    """将秒数格式化为人类可读的时间格式"""
    if seconds < 60:
        return f"{seconds:.2f} 秒"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        seconds = seconds % 60
        return f"{minutes} 分 {seconds:.2f} 秒"
    else:
        hours = int(seconds // 3600)
        seconds %= 3600
        minutes = int(seconds // 60)
        seconds %= 60
        return f"{hours} 小时 {minutes} 分 {seconds:.2f} 秒"

def get_formatted_timings() -> List[Dict[str, str]]:
    """获取格式化的耗时列表，用于显示"""
    timings = get_all_timings()
    result = []

    # 特殊处理项目开始时间和项目总耗时
    # Note: project_start_time is not used directly in this function but is kept
    # for potential future use and code clarity
    _ = timings.get("项目开始时间", 0)

    for step, time_seconds in timings.items():
        # 如果是项目开始时间，则显示具体的时间
        if step == "项目开始时间":
            import datetime
            time_str = datetime.datetime.fromtimestamp(time_seconds).strftime("%Y-%m-%d %H:%M:%S")
            result.append({
                "步骤": step,
                "耗时": time_str
            })
        else:
            result.append({
                "步骤": step,
                "耗时": format_time(time_seconds)
            })

    # 按特定顺序排序 - 更详细的步骤顺序映射
    step_order = {
        # 项目信息
        "项目总耗时": 1,
        "项目开始时间": 2,

        # 视频处理
        "下载视频": 3,
        "上传视频": 4,

        # 字幕处理阶段
        "整体字幕处理": 5,
        "转录": 10,
        "NLP分句": 11,
        "LLM分句": 12,
        "摘要": 13,
        "翻译": 14,
        "字幕分割": 15,
        "时间轴对齐": 16,
        "字幕合并到视频": 17,

        # 配音处理阶段
        "整体配音处理": 20,
        "生成配音任务": 21,
        "生成配音分块": 22,
        "提取参考音频": 23,
        "生成配音": 24,
        "合并配音": 25,
        "配音合并到视频": 26
    }

    # 先按特定顺序排序，然后按耗时排序
    result.sort(key=lambda x: (step_order.get(x["步骤"], 100)))

    return result
