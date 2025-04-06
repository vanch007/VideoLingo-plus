import sys, os
import pandas as pd
from typing import List, Tuple
import concurrent.futures
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.step3_2_splitbymeaning import split_sentence
from core.ask_gpt import ask_gpt
from core.prompts_storage import get_align_prompt
from core.config_utils import load_key, get_joiner
from rich.panel import Panel
from rich.console import Console
from rich.table import Table

console = Console()

# Constants
INPUT_FILE = "output/log/translation_results.xlsx"
OUTPUT_SPLIT_FILE = "output/log/translation_results_for_subtitles.xlsx"
OUTPUT_REMERGED_FILE = "output/log/translation_results_remerged.xlsx"

# ! You can modify your own weights here
# Chinese and Japanese 2.5 characters, Korean 2 characters, Thai 1.5 characters, full-width symbols 2 characters, other English-based and half-width symbols 1 character
def calc_len(text: str) -> float:
    text = str(text) # force convert
    def char_weight(char):
        code = ord(char)
        if 0x4E00 <= code <= 0x9FFF or 0x3040 <= code <= 0x30FF:  # Chinese and Japanese
            return 1.75
        elif 0xAC00 <= code <= 0xD7A3 or 0x1100 <= code <= 0x11FF:  # Korean
            return 1.5
        elif 0x0E00 <= code <= 0x0E7F:  # Thai
            return 1
        elif 0xFF01 <= code <= 0xFF5E:  # full-width symbols
            return 1.75
        else:  # other characters (e.g. English and half-width symbols)
            return 1

    return sum(char_weight(char) for char in text)

def align_subs(src_sub: str, tr_sub: str, tr_part: str) -> Tuple[List[str], List[str], str]:
    # 注意：这里我们反转了参数的角色，tr_part是已经分割的目标语言部分
    # 我们需要修改get_align_prompt函数调用，将源语言和目标语言的角色反转
    align_prompt = get_align_prompt(tr_sub, src_sub, tr_part, reverse=True)
    
    def valid_align(response_data):
        if 'align' not in response_data:
            return {"status": "error", "message": "Missing required key: `align`"}
        if len(response_data['align']) < 2:
            return {"status": "error", "message": "Align does not contain more than 1 part as expected!"}
        return {"status": "success", "message": "Align completed"}

    parsed = ask_gpt(align_prompt, response_json=True, valid_def=valid_align, log_title='align_subs')
    
    align_data = parsed['align']
    tr_parts = tr_part.split('\n')
    src_parts = [item[f'target_part_{i+1}'].strip() for i, item in enumerate(align_data)]
    
    # 获取源语言的连接符
    whisper_language = load_key("source_language") # 假设有这个配置项，如果没有需要添加
    language = whisper_language if whisper_language != 'auto' else 'en' # 默认英语
    joiner = get_joiner(language)
    src_remerged = joiner.join(src_parts)
    
    table = Table(title="🔗 Aligned parts")
    table.add_column("Language", style="cyan")
    table.add_column("Parts", style="magenta")
    table.add_row("TARGET_LANG", "\n".join(tr_parts))
    table.add_row("SRC_LANG", "\n".join(src_parts))
    table.add_row("REMERGED", src_remerged)
    console.print(table)
    
    return src_parts, tr_parts, src_remerged

def split_align_subs(src_lines: List[str], tr_lines: List[str]) -> Tuple[List[str], List[str], List[str]]:
    subtitle_set = load_key("subtitle")
    MAX_SUB_LENGTH = subtitle_set["max_length"]
    TARGET_SUB_MULTIPLIER = subtitle_set["target_multiplier"]
    remerged_src_lines = src_lines.copy()
    
    to_split = []
    for i, (src, tr) in enumerate(zip(src_lines, tr_lines)):
        src, tr = str(src), str(tr)
        if len(src) > MAX_SUB_LENGTH or calc_len(tr) * TARGET_SUB_MULTIPLIER > MAX_SUB_LENGTH:
            to_split.append(i)
            table = Table(title=f"📏 Line {i} needs to be split")
            table.add_column("Type", style="cyan")
            table.add_column("Content", style="magenta")
            table.add_row("Source Line", src)
            table.add_row("Target Line", tr)
            console.print(table)
    
    def process(i):
        # 首先分割目标语言（翻译）
        split_tr = split_sentence(tr_lines[i], num_parts=2).strip()
        # 然后将源语言与分割后的目标语言对齐
        src_parts, tr_parts, src_remerged = align_subs(src_lines[i], tr_lines[i], split_tr)
        src_lines[i] = src_parts
        tr_lines[i] = tr_parts
        remerged_src_lines[i] = src_remerged
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=load_key("max_workers")) as executor:
        executor.map(process, to_split)
    
    # Flatten `src_lines` and `tr_lines`
    src_lines = [item for sublist in src_lines for item in (sublist if isinstance(sublist, list) else [sublist])]
    tr_lines = [item for sublist in tr_lines for item in (sublist if isinstance(sublist, list) else [sublist])]
    
    return src_lines, tr_lines, remerged_src_lines

def split_for_sub_main():
    console.print("[bold green]🚀 Start splitting subtitles...[/bold green]")
    
    df = pd.read_excel(INPUT_FILE)
    src = df['Source'].tolist()
    trans = df['Translation'].tolist()
    
    subtitle_set = load_key("subtitle")
    MAX_SUB_LENGTH = subtitle_set["max_length"]
    TARGET_SUB_MULTIPLIER = subtitle_set["target_multiplier"]
    
    for attempt in range(3):  # 使用固定的3次重试
        console.print(Panel(f"🔄 Split attempt {attempt + 1}", expand=False))
        split_src, split_trans, remerged_src = split_align_subs(src.copy(), trans)
        
        # 检查是否所有字幕都符合长度要求
        if all(len(src) <= MAX_SUB_LENGTH for src in split_src) and \
           all(calc_len(tr) * TARGET_SUB_MULTIPLIER <= MAX_SUB_LENGTH for tr in split_trans):
            break
        
        # 更新源数据继续下一轮分割
        src = split_src
        trans = split_trans

    pd.DataFrame({'Source': split_src, 'Translation': split_trans}).to_excel(OUTPUT_SPLIT_FILE, index=False)
    pd.DataFrame({'Source': remerged_src, 'Translation': trans}).to_excel(OUTPUT_REMERGED_FILE, index=False)

if __name__ == '__main__':
    split_for_sub_main()
