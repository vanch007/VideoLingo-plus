import os, sys, json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.ask_gpt import ask_gpt
from core.prompts_storage import get_summary_prompt
from core.config_utils import load_key
import pandas as pd
from rich.console import Console

console = Console()

TERMINOLOGY_JSON_PATH = 'output/log/terminology.json'
SENTENCE_TXT_PATH = 'output/log/sentence_splitbymeaning.txt'
CUSTOM_TERMS_PATH = 'custom_terms.xlsx'

def combine_chunks():
    """Combine the text chunks identified by whisper into a single long text"""
    with open(SENTENCE_TXT_PATH, 'r', encoding='utf-8') as file:
        sentences = file.readlines()
    cleaned_sentences = [line.strip() for line in sentences if line.strip()]
    # 用换行符连接，便于 LLM 逐行纠错
    combined_text = '\n'.join(cleaned_sentences)
    # 限制长度
    max_length = load_key('summary_length')
    if len(combined_text) > max_length:
        # 按行截断，而不是字符截断
        lines = combined_text.split('\n')
        result = []
        current_length = 0
        for line in lines:
            if current_length + len(line) + 1 > max_length:
                break
            result.append(line)
            current_length += len(line) + 1
        return '\n'.join(result), len(result)
    return combined_text, len(cleaned_sentences)

def search_things_to_note_in_prompt(sentence):
    """Search for terms to note in the given sentence"""
    with open(TERMINOLOGY_JSON_PATH, 'r', encoding='utf-8') as file:
        things_to_note = json.load(file)
    things_to_note_list = [term['src'] for term in things_to_note['terms'] if term['src'].lower() in sentence.lower()]
    if things_to_note_list:
        prompt = '\n'.join(
            f'{i+1}. "{term["src"]}": "{term["tgt"]}",'
            f' meaning: {term["note"]}'
            for i, term in enumerate(things_to_note['terms'])
            if term['src'] in things_to_note_list
        )
        return prompt
    else:
        return None

def save_corrected_text(corrected_lines):
    """Save corrected text back to the source file"""
    with open(SENTENCE_TXT_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(corrected_lines))
    console.print(f'[green]📝 STT 纠错完成，已覆盖 → {SENTENCE_TXT_PATH}[/green]')

def get_summary():
    src_content, line_count = combine_chunks()
    custom_terms = pd.read_excel(CUSTOM_TERMS_PATH)
    custom_terms_json = {
        "terms": [
            {
                "src": str(row.iloc[0]),
                "tgt": str(row.iloc[1]), 
                "note": str(row.iloc[2])
            }
            for _, row in custom_terms.iterrows()
        ]
    }
    if len(custom_terms) > 0:
        console.print(f"[cyan]📖 Custom Terms Loaded: {len(custom_terms)} terms[/cyan]")
    
    summary_prompt = get_summary_prompt(src_content, custom_terms_json)
    console.print("[cyan]📝 Summarizing, extracting terminology, and correcting STT errors...[/cyan]")
    
    def valid_summary(response_data):
        required_keys = {'src', 'tgt', 'note'}
        if 'terms' not in response_data:
            return {"status": "error", "message": "Missing 'terms' key"}
        if 'corrected_lines' not in response_data:
            return {"status": "error", "message": "Missing 'corrected_lines' key"}
        if not isinstance(response_data['corrected_lines'], list):
            return {"status": "error", "message": "'corrected_lines' must be a list"}
        for term in response_data['terms']:
            if not all(key in term for key in required_keys):
                return {"status": "error", "message": "Invalid term format"}
        return {"status": "success", "message": "Summary completed"}

    summary = ask_gpt(summary_prompt, response_json=True, valid_def=valid_summary, log_title='summary')
    
    # 处理纠错结果
    if 'corrected_lines' in summary:
        corrected_lines = summary['corrected_lines']
        
        # 读取原始完整文件
        with open(SENTENCE_TXT_PATH, 'r', encoding='utf-8') as f:
            original_lines = [line.strip() for line in f.readlines() if line.strip()]
        
        # 如果纠错的行数与发送的行数一致，进行替换
        if len(corrected_lines) == line_count:
            # 只替换前 line_count 行
            new_lines = corrected_lines + original_lines[line_count:]
            
            # 统计纠正了多少行
            corrections = sum(1 for i in range(line_count) if original_lines[i] != corrected_lines[i])
            console.print(f"[cyan]🔧 STT 纠错: 共 {line_count} 行，纠正 {corrections} 行[/cyan]")
            
            # 显示纠正的内容
            if corrections > 0:
                for i in range(min(line_count, len(corrected_lines))):
                    if i < len(original_lines) and original_lines[i] != corrected_lines[i]:
                        console.print(f"[yellow]  {i+1}: {original_lines[i]}[/yellow]")
                        console.print(f"[green]  → {corrected_lines[i]}[/green]")
            
            save_corrected_text(new_lines)
        else:
            console.print(f"[yellow]⚠️ 纠错行数不匹配 (期望 {line_count}, 得到 {len(corrected_lines)}), 跳过纠错[/yellow]")
    
    # 保存术语和主题
    if 'terms' in summary:
        summary['terms'].extend(custom_terms_json['terms'])
    
    # 移除 corrected_lines，只保存 terminology
    save_data = {
        'topic': summary.get('topic', ''),
        'terms': summary.get('terms', [])
    }
    
    with open(TERMINOLOGY_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, ensure_ascii=False, indent=4)

    console.print(f'[green]💾 Summary saved to → {TERMINOLOGY_JSON_PATH}[/green]')

if __name__ == '__main__':
    get_summary()