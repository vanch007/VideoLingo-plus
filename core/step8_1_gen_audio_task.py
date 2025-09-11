import pandas as pd
import datetime
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import re
from core.ask_gpt import ask_gpt
from core.prompts_storage import get_subtitle_trim_prompt, get_expand_short_text_prompt
from rich import print as rprint
from rich.panel import Panel
from rich.console import Console
from core.config_utils import load_key
from core.all_tts_functions.estimate_duration import init_estimator, estimate_duration
import concurrent.futures  # 新增多进程支持

console = Console()
speed_factor = load_key("speed_factor")

TRANS_SUBS_FOR_AUDIO_FILE = 'output/audio/trans_subs_for_audio.srt'
SRC_SUBS_FOR_AUDIO_FILE = 'output/audio/src_subs_for_audio.srt'
SOVITS_TASKS_FILE = 'output/audio/tts_tasks.xlsx'
SRC_SRT = 'output/src.srt'
TRANS_SRT = 'output/trans.srt'
ESTIMATOR = None

def expand_short_text(text, origin_text, prev_text=None, next_text=None):
    """Expand short text with natural language continuation without special symbols"""
    if len(text.strip()) <= 3:  # 处理超短文本（3字符或更少）
        rprint(Panel(f"Text '{text}' is too short ({len(text.strip())} chars), expanding...", title="Processing", border_style="yellow"))
        
        # 提取上下文的核心语义信息
        context_hint = ""
        if prev_text and next_text:
            context_hint = f"Previous theme: {' '.join(prev_text.split()[-2:])}, Next theme: {' '.join(next_text.split()[:2])}"
        elif prev_text:
            context_hint = f"Previous theme: {' '.join(prev_text.split()[-2:])}"
        elif next_text:
            context_hint = f"Next theme: {' '.join(next_text.split()[:2])}"
            
        prompt = get_expand_short_text_prompt(text, origin_text, context_hint)  # 传递上下文语义提示

        def valid_expand(response):
            if 'expanded_text' not in response:
                return {'status': 'error', 'message': 'No expanded_text in response'}
            # 新增验证：禁止括号描述
            if '(' in response['expanded_text'] or ')' in response['expanded_text']:
                return {'status': 'error', 'message': 'Expanded text contains parentheses'}
            return {'status': 'success', 'message': ''}

        try:
            model_name = get_valid_model_name()
            response = ask_gpt(prompt, response_json=True, log_title='expand_short_text', valid_def=valid_expand, model=model_name)
            expanded_text = response['expanded_text']
            
            # 自然语言扩展替代方案
            if len(expanded_text.strip()) <= 3:
                # 使用重复关键词扩展而非符号
                expanded_text = ' '.join([text.strip()] * 2) or text + '扩展'  # 最小化自然扩展
                rprint(Panel(f"Applied natural expansion for text: {text}", title="Fallback", border_style="blue"))
                
            rprint(Panel(f"Original: {text}\nContext: {context_hint}\nExpanded: {expanded_text}\nAnalysis: {response['analysis']}",
                         title="Natural Expansion Result", border_style="green"))
            return expanded_text
        except Exception as e:
            rprint(f"[bold red]🚫 Failed to expand text: {str(e)}[/bold red]")
            # 失败时使用最简自然扩展
            return text + '扩展' if len(text.strip()) <= 2 else text
    return text  # 如果不是超短文本返回原文本

def check_len_then_trim(text, duration):
    global ESTIMATOR
    if ESTIMATOR is None:
        ESTIMATOR = init_estimator()
    estimated_duration = estimate_duration(text, ESTIMATOR) / speed_factor['max']

    console.print(f"Subtitle text: {text}, "
                  f"[bold green]Estimated reading duration: {estimated_duration:.2f} seconds[/bold green]")

    if estimated_duration > duration:
        rprint(Panel(f"Estimated reading duration {estimated_duration:.2f} seconds exceeds given duration {duration:.2f} seconds, shortening...", title="Processing", border_style="yellow"))
        original_text = text
        prompt = get_subtitle_trim_prompt(text, duration)
        def valid_trim(response):
            if 'result' not in response:
                return {'status': 'error', 'message': 'No result in response'}
            return {'status': 'success', 'message': ''}
        try:
            response = ask_gpt(prompt, response_json=True, log_title='subtitle_trim', valid_def=valid_trim)
            shortened_text = response['result']
        except Exception:
            rprint("[bold red]🚫 AI refused to answer due to sensitivity, so manually remove punctuation[/bold red]")
            shortened_text = re.sub(r'[,.!?;:，。！？；：]', ' ', text).strip()
        rprint(Panel(f"Subtitle before shortening: {original_text}\nSubtitle after shortening: {shortened_text}", title="Subtitle Shortening Result", border_style="green"))
        return shortened_text
    else:
        return text

def time_diff_seconds(t1: datetime.time, t2: datetime.time, base_date: datetime.date) -> float:
    """Calculate the difference in seconds between two time objects"""
    dt1 = datetime.datetime.combine(base_date, t1)
    dt2 = datetime.datetime.combine(base_date, t2)
    return (dt2 - dt1).total_seconds()

def parallel_expand_short_text(args):
    """包装函数支持进程池调用"""
    text, origin_text, prev_text, next_text, index = args
    try:
        from core.step8_1_gen_audio_task import expand_short_text
        result = expand_short_text(text, origin_text, prev_text, next_text)  # 保持参数一致性
        return (index, result)
    except Exception as e:
        print(f"Error in expand_short_text at index {index}: {str(e)}")
        return (index, text)

def merge_short_subtitles(df):
    i = 0
    MIN_SUB_DUR = load_key("min_subtitle_duration")
    # 设置最小字幕时长阈值，低于此值的字幕将被合并或延长
    MIN_MERGE_THRESHOLD = 1.5  # 设置为1.5秒，避免生成太短的配音任务

    rprint(Panel(f"Starting subtitle merging process with minimum duration threshold: {MIN_MERGE_THRESHOLD} seconds", title="Processing", border_style="cyan"))

    while i < len(df):
        today = datetime.date.today()
        if df.loc[i, 'duration'] < MIN_MERGE_THRESHOLD:
            # 检查前后字幕的duration
            prev_dur = df.loc[i-1, 'duration'] if i > 0 else float('inf')
            next_dur = df.loc[i+1, 'duration'] if i < len(df)-1 else float('inf')

            # 计算与前后字幕的间隔
            prev_gap = time_diff_seconds(df.loc[i-1, 'end_time'], df.loc[i, 'start_time'], today) if i > 0 else float('inf')
            next_gap = time_diff_seconds(df.loc[i, 'end_time'], df.loc[i+1, 'start_time'], today) if i < len(df)-1 else float('inf')

            # 选择duration较短的字幕合并，优先考虑间隔小的方向
            if i > 0 and prev_gap < 0.5 and (prev_dur < next_dur or next_gap > 0.5):
                # 向前合并
                rprint(f"[bold yellow]Merging subtitle {i} (duration: {df.loc[i, 'duration']:.2f}s) with previous {i-1} (duration: {prev_dur:.2f}s)[/bold yellow]")
                df.loc[i-1, 'text'] += ' ' + df.loc[i, 'text']
                df.loc[i-1, 'origin'] += ' ' + df.loc[i, 'origin']
                df.loc[i-1, 'end_time'] = df.loc[i, 'end_time']
                df.loc[i-1, 'duration'] = time_diff_seconds(df.loc[i-1, 'start_time'], df.loc[i, 'end_time'], today)
                df = df.drop(i).reset_index(drop=True)
                i -= 1  # 因为删除了当前行，需要回退索引
            elif i < len(df)-1 and next_gap < 0.5:
                # 向后合并
                rprint(f"[bold yellow]Merging subtitle {i} (duration: {df.loc[i, 'duration']:.2f}s) with next {i+1} (duration: {next_dur:.2f}s)[/bold yellow]")
                df.loc[i, 'text'] += ' ' + df.loc[i+1, 'text']
                df.loc[i, 'origin'] += ' ' + df.loc[i+1, 'origin']
                df.loc[i, 'end_time'] = df.loc[i+1, 'end_time']
                df.loc[i, 'duration'] = time_diff_seconds(df.loc[i, 'start_time'], df.loc[i+1, 'end_time'], today)
                df = df.drop(i+1).reset_index(drop=True)
            else:
                # 如果无法合并，则延长时长
                if i < len(df)-1:  # 不是最后一个字幕
                    next_start = df.loc[i+1, 'start_time']
                    current_end = df.loc[i, 'end_time']
                    available_gap = time_diff_seconds(current_end, next_start, today)

                    if available_gap >= MIN_MERGE_THRESHOLD - df.loc[i, 'duration']:
                        # 有足够空间延长
                        new_duration = MIN_MERGE_THRESHOLD
                        rprint(f"[bold blue]Extending subtitle {i} duration from {df.loc[i, 'duration']:.2f}s to {new_duration:.2f}s[/bold blue]")
                        df.loc[i, 'end_time'] = (datetime.datetime.combine(today, df.loc[i, 'start_time']) +
                                               datetime.timedelta(seconds=new_duration)).time()
                        df.loc[i, 'duration'] = new_duration
                    else:
                        # 没有足够空间，尽可能延长但保留一些间隔
                        new_duration = df.loc[i, 'duration'] + available_gap * 0.8
                        new_end_time = (datetime.datetime.combine(today, df.loc[i, 'start_time']) +
                                      datetime.timedelta(seconds=new_duration)).time()
                        df.loc[i, 'end_time'] = new_end_time
                        df.loc[i, 'duration'] = time_diff_seconds(df.loc[i, 'start_time'], new_end_time, today)
                        rprint(f"[bold blue]Partially extending subtitle {i} duration from {df.loc[i, 'duration']:.2f}s to {new_duration:.2f}s[/bold blue]")
                else:
                    # 最后一个字幕，固定延长
                    new_duration = MIN_MERGE_THRESHOLD
                    rprint(f"[bold blue]Extending last subtitle {i} duration from {df.loc[i, 'duration']:.2f}s to {new_duration:.2f}s[/bold blue]")
                    df.loc[i, 'end_time'] = (datetime.datetime.combine(today, df.loc[i, 'start_time']) +
                                           datetime.timedelta(seconds=new_duration)).time()
                    df.loc[i, 'duration'] = new_duration
                i += 1
        else:
            i += 1
    
    # 统计处理结果
    short_subtitles = len(df[df['duration'] < MIN_MERGE_THRESHOLD])
    if short_subtitles > 0:
        rprint(Panel(f"Warning: Still have {short_subtitles} subtitles with duration less than {MIN_MERGE_THRESHOLD} seconds after processing.", title="Warning", border_style="yellow"))
    else:
        rprint(Panel(f"Successfully processed all subtitles. All durations are now at least {MIN_MERGE_THRESHOLD} seconds.", title="Success", border_style="green"))
    
    return df

def process_srt():
    """Process srt file, generate audio tasks"""

    with open(TRANS_SUBS_FOR_AUDIO_FILE, 'r', encoding='utf-8') as file:
        content = file.read()

    with open(SRC_SUBS_FOR_AUDIO_FILE, 'r', encoding='utf-8') as src_file:
        src_content = src_file.read()

    subtitles = []
    src_subtitles = {}

    for block in src_content.strip().split('\n\n'):
        lines = [line.strip() for line in block.split('\n') if line.strip()]
        if len(lines) < 3:
            continue

        number = int(lines[0])
        src_text = ' '.join(lines[2:])
        src_subtitles[number] = src_text

    for block in content.strip().split('\n\n'):
        lines = [line.strip() for line in block.split('\n') if line.strip()]
        if len(lines) < 3:
            continue

        try:
            number = int(lines[0])
            start_time, end_time = lines[1].split(' --> ')
            start_time = datetime.datetime.strptime(start_time, '%H:%M:%S,%f').time()
            end_time = datetime.datetime.strptime(end_time, '%H:%M:%S,%f').time()
            duration = time_diff_seconds(start_time, end_time, datetime.date.today())
            text = ' '.join(lines[2:])
            # Remove content within parentheses (including English and Chinese parentheses)
            text = re.sub(r'\([^)]*\)', '', text).strip()
            text = re.sub(r'（[^）]*）', '', text).strip()
            # Remove only '-' character, keep other punctuation
            text = text.replace('-', '')

            # Add the original text from src_subs_for_audio.srt
            origin = src_subtitles.get(number, '')

        except ValueError as e:
            rprint(Panel(f"Unable to parse subtitle block '{block}', error: {str(e)}, skipping this subtitle block.", title="Error", border_style="red"))
            continue

        subtitles.append({'number': number, 'start_time': start_time, 'end_time': end_time, 'duration': duration, 'text': text, 'origin': origin})

    df = pd.DataFrame(subtitles)

    merge_subtitles = load_key("merge_subtitles", True)
    if merge_subtitles:
        df = merge_short_subtitles(df)

    # 并行处理短文本字幕，使用GPT扩展文本
    rprint(Panel("Processing short text subtitles (3 characters or less)...", title="Processing", border_style="cyan"))
    short_text_count = 0

    # 添加一个新字段来保存原始文本，用于后续匹配
    if 'original_text' not in df.columns:
        df['original_text'] = df['text'].copy()

    # 统计需要处理的短文本数量
    for i, row in df.iterrows():
        if len(row['text'].strip()) <= 3:
            short_text_count += 1

    # 并行处理短文本字幕
    if short_text_count > 0:
        rprint(Panel(f"Processing {short_text_count} short text subtitles with parallel processing...", title="Processing", border_style="cyan"))
        
        # 创建任务列表
        tasks = []
        for i, row in df.iterrows():
            if len(row['text'].strip()) <= 3:
                prev_text = df.iloc[i-1]['text'] if i > 0 else None
                next_text = df.iloc[i+1]['text'] if i < len(df)-1 else None
                tasks.append((row['text'], row['origin'], prev_text, next_text, i))

        # 使用进程池并行处理
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = list(executor.map(parallel_expand_short_text, tasks))
            
        # 更新结果回DataFrame
        for index, expanded_text in results:
            df.at[index, 'original_text'] = df.at[index, 'text']
            df.at[index, 'text'] = expanded_text

        rprint(Panel(f"Successfully expanded {short_text_count} short text subtitles in parallel", title="Success", border_style="green"))
    else:
        rprint(Panel("No short text subtitles found.", title="Info", border_style="blue"))

    # Add sub_times column
    df['sub_times'] = df.apply(lambda row: [
        (row['start_time'].hour * 3600 + row['start_time'].minute * 60 + row['start_time'].second + row['start_time'].microsecond / 1_000_000),
        (row['end_time'].hour * 3600 + row['end_time'].minute * 60 + row['end_time'].second + row['end_time'].microsecond / 1_000_000)
    ], axis=1)

    df['start_time'] = df['start_time'].apply(lambda x: x.strftime('%H:%M:%S.%f')[:-3])
    df['end_time'] = df['end_time'].apply(lambda x: x.strftime('%H:%M:%S.%f')[:-3])

    ##! No longer perform secondary trim
    # check and trim subtitle length, for twice to ensure the subtitle length is within the limit, 允许tolerance
    # df['text'] = df.apply(lambda x: check_len_then_trim(x['text'], x['duration']+x['tolerance']), axis=1)

    return df


def gen_audio_task_main():
    if os.path.exists(SOVITS_TASKS_FILE):
        rprint(Panel(f"{SOVITS_TASKS_FILE} already exists, skip.", title="Info", border_style="blue"))
    else:
        df = process_srt()
        console.print(df)
        df.to_excel(SOVITS_TASKS_FILE, index=False)
        rprint(Panel(f"Successfully generated {SOVITS_TASKS_FILE}", title="Success", border_style="green"))


if __name__ == '__main__':
    gen_audio_task_main()
