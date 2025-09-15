import os
import sys
import re
import pandas as pd
from rich import print as rprint
from rich.panel import Panel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def parse_srt_file(srt_path):
    """解析SRT字幕文件"""
    if not os.path.exists(srt_path):
        rprint(Panel(f"[bold red]文件不存在: {srt_path}[/bold red]", title="错误", border_style="red"))
        return []
    
    with open(srt_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # 使用正则表达式解析SRT块
    pattern = re.compile(r'(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n(.*?)(?=\n\d+\n|\Z)', re.DOTALL)
    matches = pattern.findall(content)
    
    subtitles = []
    for match in matches:
        number = int(match[0])
        start_time = match[1]
        end_time = match[2]
        text = match[3].replace('\n', ' ').strip()
        
        # 移除字幕中的SDH内容（括号内的内容）
        clean_text = re.sub(r'\(.*?\)', '', text).strip()
        clean_text = re.sub(r'（.*?）', '', clean_text).strip()
        
        subtitles.append({
            'number': number,
            'start_time': start_time,
            'end_time': end_time,
            'text': text,
            'clean_text': clean_text
        })
    
    return subtitles

def remove_sdh_content(subtitles):
    """移除SDH内容（已经在parse_srt_file中处理，但保留此函数以备将来扩展）"""
    for subtitle in subtitles:
        # 移除括号中的内容（已经处理过）
        subtitle['clean_text'] = re.sub(r'\[.*?\]', '', subtitle['clean_text']).strip()
        subtitle['clean_text'] = re.sub(r'【.*?】', '', subtitle['clean_text']).strip()
    
    return subtitles

def align_subtitles_by_time(src_subtitles, trans_subtitles):
    """根据时间轴对齐源语言和目标语言字幕"""
    aligned_subs = []
    
    for src_sub in src_subtitles:
        # 寻找时间最接近的目标语言字幕
        best_match = None
        min_time_diff = float('inf')
        
        src_start = time_to_seconds(src_sub['start_time'])
        src_end = time_to_seconds(src_sub['end_time'])
        src_mid = (src_start + src_end) / 2
        
        for trans_sub in trans_subtitles:
            trans_start = time_to_seconds(trans_sub['start_time'])
            trans_end = time_to_seconds(trans_sub['end_time'])
            trans_mid = (trans_start + trans_end) / 2
            
            # 计算时间差
            time_diff = abs(src_mid - trans_mid)
            
            # 如果时间重叠或时间差最小，则认为是匹配的
            if (max(src_start, trans_start) <= min(src_end, trans_end)) or (time_diff < min_time_diff):
                min_time_diff = time_diff
                best_match = trans_sub
        
        # 添加对齐后的字幕
        aligned_subs.append({
            'number': src_sub['number'],
            'start_time': src_sub['start_time'],
            'end_time': src_sub['end_time'],
            'src_text': src_sub['clean_text'],
            'trans_text': best_match['clean_text'] if best_match else ''
        })
    
    return aligned_subs

def filter_complete_bilingual_subtitles(aligned_subtitles):
    """过滤掉只有一种语言字幕的时间轴，确保每条字幕都有双语内容"""
    filtered_subtitles = []
    
    for subtitle in aligned_subtitles:
        src_text = subtitle['src_text'].strip()
        trans_text = subtitle['trans_text'].strip()
        
        # 只保留两种语言都有内容的字幕
        if src_text and trans_text:
            filtered_subtitles.append(subtitle)
        else:
            # 可选：打印被过滤掉的字幕信息
            if not src_text and trans_text:
                rprint(f"[yellow]警告: 过滤掉只有目标语言的字幕: {trans_text[:30]}...[/yellow]")
            elif src_text and not trans_text:
                rprint(f"[yellow]警告: 过滤掉只有源语言的字幕: {src_text[:30]}...[/yellow]")
            else:
                rprint(f"[yellow]警告: 过滤掉空字幕[/yellow]")
    
    rprint(f"[green]原始字幕数: {len(aligned_subtitles)}, 过滤后字幕数: {len(filtered_subtitles)}[/green]")
    return filtered_subtitles

def time_to_seconds(time_str):
    """将时间字符串转换为秒数"""
    # 时间格式: HH:MM:SS,mmm
    h, m, s = time_str.split(':')
    s, ms = s.split(',')
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000

def save_to_excel(subtitles, output_path):
    """将字幕保存到Excel文件"""
    df = pd.DataFrame(subtitles)
    df.to_excel(output_path, index=False)
    rprint(Panel(f"[bold green]时间轴文件已保存至: {output_path}[/bold green]", title="成功", border_style="green"))

def save_srt_file(subtitles, srt_path, text_field):
    """保存为SRT文件格式"""
    with open(srt_path, 'w', encoding='utf-8') as f:
        idx = 1
        for subtitle in subtitles:
            text = subtitle[text_field]
            # 如果文本为空，跳过该字幕
            if not text.strip():
                continue
                
            f.write(f"{idx}\n")
            f.write(f"{subtitle['start_time']} --> {subtitle['end_time']}\n")
            f.write(f"{text}\n\n")
            idx += 1
    
    rprint(f"[green]SRT文件已保存至: {srt_path}[/green]")

def process_extracted_subs_main():
    """主函数：处理提取的字幕"""
    output_dir = 'output'
    log_dir = os.path.join(output_dir, 'log')
    raw_src_path = os.path.join(output_dir, 'raw_src.srt')
    raw_trans_path = os.path.join(output_dir, 'raw_trans.srt')
    timeline_path = os.path.join(log_dir, 'sub_timeline_from_extracted.xlsx')
    
    # 确保输出目录存在
    os.makedirs(log_dir, exist_ok=True)
    
    # 检查输入文件是否存在
    if not os.path.exists(raw_src_path):
        rprint(Panel(f"[bold red]源语言字幕文件不存在: {raw_src_path}[/bold red]", title="错误", border_style="red"))
        return False
    
    # 解析源语言字幕
    rprint("[cyan]正在解析源语言字幕...[/cyan]")
    src_subtitles = parse_srt_file(raw_src_path)
    if not src_subtitles:
        rprint(Panel("[bold red]未能解析源语言字幕[/bold red]", title="错误", border_style="red"))
        return False
    
    rprint(f"[green]成功解析 {len(src_subtitles)} 条源语言字幕[/green]")
    
    # 解析目标语言字幕（如果存在）
    trans_subtitles = []
    if os.path.exists(raw_trans_path):
        rprint("[cyan]正在解析目标语言字幕...[/cyan]")
        trans_subtitles = parse_srt_file(raw_trans_path)
        if trans_subtitles:
            rprint(f"[green]成功解析 {len(trans_subtitles)} 条目标语言字幕[/green]")
        else:
            rprint("[yellow]未能解析目标语言字幕，将继续仅处理源语言字幕[/yellow]")
    else:
        rprint("[yellow]目标语言字幕文件不存在，将继续仅处理源语言字幕[/yellow]")
    
    # 移除SDH内容
    src_subtitles = remove_sdh_content(src_subtitles)
    if trans_subtitles:
        trans_subtitles = remove_sdh_content(trans_subtitles)
    
    # 对齐字幕
    if trans_subtitles and len(trans_subtitles) > 0:
        rprint("[cyan]正在对齐源语言和目标语言字幕...[/cyan]")
        aligned_subtitles = align_subtitles_by_time(src_subtitles, trans_subtitles)
        
        # 过滤掉只有一种语言字幕的时间轴
        rprint("[cyan]正在过滤不完整的双语字幕...[/cyan]")
        aligned_subtitles = filter_complete_bilingual_subtitles(aligned_subtitles)
    else:
        # 如果没有目标语言字幕，仅使用源语言字幕
        aligned_subtitles = []
        for src_sub in src_subtitles:
            aligned_subtitles.append({
                'number': src_sub['number'],
                'start_time': src_sub['start_time'],
                'end_time': src_sub['end_time'],
                'src_text': src_sub['clean_text'],
                'trans_text': ''
            })
    
    # 保存到Excel文件
    save_to_excel(aligned_subtitles, timeline_path)
    
    # 生成SRT文件供后续步骤使用
    src_srt_path = os.path.join(output_dir, 'src.srt')
    trans_srt_path = os.path.join(output_dir, 'trans.srt')
    src_trans_srt_path = os.path.join(output_dir, 'src_trans.srt')
    trans_src_srt_path = os.path.join(output_dir, 'trans_src.srt')
    
    # 生成 src.srt (源语言字幕)
    save_srt_file(aligned_subtitles, src_srt_path, 'src_text')
    
    # 生成 trans.srt (目标语言字幕)
    save_srt_file(aligned_subtitles, trans_srt_path, 'trans_text')
    
    # 生成 src_trans.srt (源语言在上，目标语言在下)
    with open(src_trans_srt_path, 'w', encoding='utf-8') as f:
        idx = 1
        for subtitle in aligned_subtitles:
            src_text = subtitle['src_text']
            trans_text = subtitle['trans_text']
            
            # 如果两个文本都为空，跳过该字幕
            if not src_text.strip() and not trans_text.strip():
                continue
                
            f.write(f"{idx}\n")
            f.write(f"{subtitle['start_time']} --> {subtitle['end_time']}\n")
            f.write(f"{src_text}\n")
            if trans_text.strip():
                f.write(f"{trans_text}\n")
            f.write("\n")
            idx += 1
    
    rprint(f"[green]SRT文件已保存至: {src_trans_srt_path}[/green]")
    
    # 生成 trans_src.srt (目标语言在上，源语言在下)
    with open(trans_src_srt_path, 'w', encoding='utf-8') as f:
        idx = 1
        for subtitle in aligned_subtitles:
            src_text = subtitle['src_text']
            trans_text = subtitle['trans_text']
            
            # 如果两个文本都为空，跳过该字幕
            if not src_text.strip() and not trans_text.strip():
                continue
                
            f.write(f"{idx}\n")
            f.write(f"{subtitle['start_time']} --> {subtitle['end_time']}\n")
            if trans_text.strip():
                f.write(f"{trans_text}\n")
            f.write(f"{src_text}\n\n")
            idx += 1
    
    rprint(f"[green]SRT文件已保存至: {trans_src_srt_path}[/green]")
    
    # 生成配音任务所需的SRT文件
    src_subs_for_audio_path = os.path.join(output_dir, 'audio', 'src_subs_for_audio.srt')
    trans_subs_for_audio_path = os.path.join(output_dir, 'audio', 'trans_subs_for_audio.srt')
    
    os.makedirs(os.path.join(output_dir, 'audio'), exist_ok=True)
    
    save_srt_file(aligned_subtitles, src_subs_for_audio_path, 'src_text')
    save_srt_file(aligned_subtitles, trans_subs_for_audio_path, 'trans_text')
    
    rprint(Panel("[bold green]字幕处理完成[/bold green]", title="完成", border_style="green"))
    return True

if __name__ == "__main__":
    process_extracted_subs_main()