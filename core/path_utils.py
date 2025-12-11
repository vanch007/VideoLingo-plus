"""
统一的路径处理工具，解决各模块重复的 sys.path.append 问题。

使用方法：
    from core.path_utils import setup_project_path, get_project_root
    setup_project_path()  # 在模块顶部调用一次即可
"""
import os
import sys
from pathlib import Path

# 项目根目录（VideoLingo-plus 目录）
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def setup_project_path():
    """
    将项目根目录添加到 Python 路径（仅在需要时）。
    这是替代各模块中 sys.path.append 的统一方法。
    """
    root_str = str(PROJECT_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def get_project_root() -> Path:
    """获取项目根目录的 Path 对象"""
    return PROJECT_ROOT


def get_output_dir() -> Path:
    """获取输出目录"""
    return PROJECT_ROOT / "output"


def get_audio_dir() -> Path:
    """获取音频输出目录"""
    return get_output_dir() / "audio"


def get_log_dir() -> Path:
    """获取日志输出目录"""
    return get_output_dir() / "log"


def get_refers_dir() -> Path:
    """获取参考音频目录"""
    return get_audio_dir() / "refers"


def get_segs_dir() -> Path:
    """获取音频片段目录"""
    return get_audio_dir() / "segs"


def ensure_dirs():
    """确保所有必要的输出目录存在"""
    dirs = [get_output_dir(), get_audio_dir(), get_log_dir(), get_refers_dir(), get_segs_dir()]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    print(f"项目根目录: {get_project_root()}")
    print(f"输出目录: {get_output_dir()}")
    print(f"音频目录: {get_audio_dir()}")
    print(f"日志目录: {get_log_dir()}")
