import os
import sys
import subprocess
import json
from rich import print as rprint
from rich.panel import Panel
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config_utils import load_key
from core.step1_ytdlp import find_video_files
from core.ask_gpt import ask_gpt

def get_subtitle_tracks():
    """获取视频中的所有字幕轨道信息"""
    video_file = find_video_files()
    cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams', '-select_streams', 's', video_file]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        subtitle_info = json.loads(result.stdout)
        
        tracks = []
        for stream in subtitle_info.get('streams', []):
            track_info = {
                'index': stream.get('index'),
                'codec_name': stream.get('codec_name'),
                'codec_type': stream.get('codec_type'),
                'tags': stream.get('tags', {}),
                'language': stream.get('tags', {}).get('language') if stream.get('tags') else None,
                'title': stream.get('tags', {}).get('title') if stream.get('tags') else None,
                'handler_name': stream.get('tags', {}).get('handler_name') if stream.get('tags') else None
            }
            tracks.append(track_info)
        
        return tracks
    except subprocess.CalledProcessError as e:
        rprint(Panel(f"[bold red]获取字幕轨道信息失败: {str(e)}[/bold red]", title="错误", border_style="red"))
        return []
    except json.JSONDecodeError as e:
        rprint(Panel(f"[bold red]解析字幕轨道信息失败: {str(e)}[/bold red]", title="错误", border_style="red"))
        return []

def display_subtitle_tracks(tracks):
    """显示字幕轨道信息"""
    if not tracks:
        rprint(Panel("[bold yellow]未找到字幕轨道[/bold yellow]", title="提示", border_style="yellow"))
        return
        
    rprint(Panel("[bold green]找到以下字幕轨道:[/bold green]", title="字幕轨道信息", border_style="green"))
    for i, track in enumerate(tracks):
        lang_code = track.get('language', '未知')
        codec = track.get('codec_name', '未知')
        title = track.get('title', '无标题')
        handler_name = track.get('handler_name', '无处理名称')
        stream_index = track.get('index', '未知')
        
        rprint(f"[cyan]{i+1}. 语言代码: {lang_code}, 编码: {codec}, 流索引: {stream_index}, 标题: {title}, 处理名称: {handler_name}[/cyan]")

def extract_subtitles(src_track_index, trans_track_index=None):
    """提取指定的字幕轨道"""
    video_file = find_video_files()
    output_dir = 'output'
    raw_src_path = os.path.join(output_dir, 'raw_src.srt')
    raw_trans_path = os.path.join(output_dir, 'raw_trans.srt')
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 提取源语言字幕
    src_cmd = ['ffmpeg', '-y', '-i', video_file, '-map', f'0:s:{src_track_index}', raw_src_path]
    try:
        subprocess.run(src_cmd, check=True, capture_output=True)
        rprint(Panel(f"[bold green]源语言字幕已提取至: {raw_src_path}[/bold green]", title="成功", border_style="green"))
    except subprocess.CalledProcessError as e:
        rprint(Panel(f"[bold red]提取源语言字幕失败: {str(e)}[/bold red]", title="错误", border_style="red"))
        return False
    
    # 如果指定了目标语言字幕轨道，也提取它
    if trans_track_index is not None and trans_track_index >= 0:
        trans_cmd = ['ffmpeg', '-y', '-i', video_file, '-map', f'0:s:{trans_track_index}', raw_trans_path]
        try:
            subprocess.run(trans_cmd, check=True, capture_output=True)
            rprint(Panel(f"[bold green]目标语言字幕已提取至: {raw_trans_path}[/bold green]", title="成功", border_style="green"))
        except subprocess.CalledProcessError as e:
            rprint(Panel(f"[bold red]提取目标语言字幕失败: {str(e)}[/bold red]", title="错误", border_style="red"))
            return False
    elif trans_track_index is None:
        # 如果没有选择目标语言字幕轨道，创建一个空的目标字幕文件
        with open(raw_trans_path, 'w', encoding='utf-8') as f:
            f.write("")
        rprint(Panel(f"[bold yellow]未选择目标语言字幕轨道，已创建空文件: {raw_trans_path}[/bold yellow]", title="提示", border_style="yellow"))
    
    return True

def select_tracks_with_gpt(tracks, source_language, target_language):
    """完全依赖GPT选择字幕轨道"""
    try:
        # 准备轨道信息供GPT分析
        track_info_list = []
        for i, track in enumerate(tracks):
            track_info = {
                'array_index': i,  # 数组索引
                'stream_index': track.get('index'),  # 流索引
                'language_code': track.get('language', '未知'),
                'codec': track.get('codec_name', '未知'),
                'title': track.get('title', '无标题'),
                'handler_name': track.get('handler_name', '无处理名称')
            }
            track_info_list.append(track_info)
        
        # 构造提示词
        prompt = f"""
你是一个专业的字幕轨道选择助手。请根据用户需要的源语言和目标语言，从提供的字幕轨道列表中选择最合适的轨道。

工作流程：
1. 仔细分析所有字幕轨道的完整信息
2. 根据用户需要的源语言和目标语言，选择最匹配的轨道
3. 返回选择的轨道索引

选择规则：
1. 严格按照用户指定的语言进行选择
2. 如果一种语言有多个轨道，优先选择：
   - 更清晰描述语言区域的轨道（如"Chinese (Simplified)"优于"Chinese"）
   - 非SDH（听力障碍）版本的轨道
3. 如果找不到明确匹配目标语言的轨道，返回null

需要选择的语言：
- 源语言: {source_language}
- 目标语言: {target_language}

字幕轨道列表（严格按照列表中的array_index返回结果）:
{json.dumps(track_info_list, ensure_ascii=False, indent=2)}

请严格按照以下JSON格式返回结果：
{{
    "src_track_array_index": 0,
    "trans_track_array_index": 2
}}

其中：
- src_track_array_index: 源语言字幕轨道的数组索引（从0开始）
- trans_track_array_index: 目标语言字幕轨道的数组索引（从0开始），如果找不到则为null

只返回JSON，不要包含其他文字。
"""
        
        rprint("[blue]正在调用GPT进行字幕轨道选择...[/blue]")
        # 调用GPT
        response = ask_gpt(prompt, response_json=True, log_title='subtitle_track_selection')
        
        if response and 'src_track_array_index' in response:
            src_index = response['src_track_array_index']
            trans_index = response.get('trans_track_array_index')
            rprint(f"[green]GPT选择结果 - 源语言轨道: {src_index+1}, 目标语言轨道: {trans_index+1 if trans_index is not None else '无'}[/green]")
            return src_index, trans_index
        else:
            rprint("[red]GPT未能返回有效的轨道选择[/red]")
            return None, None
            
    except Exception as e:
        rprint(f"[red]使用GPT选择字幕轨道时出错: {str(e)}[/red]")
        return None, None

def extract_subtitles_main():
    """主函数：提取内嵌字幕"""
    # 获取项目配置的语言设置
    source_language = load_key("source_language", load_key("whisper.language"))
    target_language = load_key("target_language")
    
    rprint(Panel(f"[bold blue]当前配置 - 源语言: {source_language}, 目标语言: {target_language}[/bold blue]", 
                title="语言配置", border_style="blue"))
    
    # 获取字幕轨道信息
    tracks = get_subtitle_tracks()
    display_subtitle_tracks(tracks)
    
    if not tracks:
        return False
    
    # 使用GPT选择字幕轨道
    src_track_index, trans_track_index = select_tracks_with_gpt(tracks, source_language, target_language)
    
    if src_track_index is None:
        rprint(Panel("[bold red]GPT未能选择源语言字幕轨道[/bold red]", title="错误", border_style="red"))
        return False
    
    success = extract_subtitles(src_track_index, trans_track_index)
    
    if success:
        rprint(Panel("[bold green]字幕提取完成[/bold green]", title="完成", border_style="green"))
    else:
        rprint(Panel("[bold red]字幕提取失败[/bold red]", title="失败", border_style="red"))
    
    return success

if __name__ == "__main__":
    extract_subtitles_main()