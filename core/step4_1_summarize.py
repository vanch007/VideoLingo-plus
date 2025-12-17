import os, sys, json
import re
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
CLEANED_CHUNKS_PATH = 'output/log/cleaned_chunks.xlsx'

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

def update_cleaned_chunks(original_lines, corrected_lines):
    """Update cleaned_chunks.xlsx with corrected text while preserving timestamps.
    
    This function maps the corrected sentence-level text back to word-level timestamps.
    It works by:
    1. Joining original lines to get the full original text
    2. Joining corrected lines to get the full corrected text  
    3. Matching characters position by position (allowing for length differences)
    4. Updating the 'text' column in cleaned_chunks.xlsx
    """
    if not os.path.exists(CLEANED_CHUNKS_PATH):
        console.print(f"[yellow]⚠️ {CLEANED_CHUNKS_PATH} not found, skipping cleaned_chunks update[/yellow]")
        return
    
    try:
        df = pd.read_excel(CLEANED_CHUNKS_PATH)
        
        # Remove quotes from text column for comparison
        df['text_clean'] = df['text'].str.strip('"').str.strip()
        
        # Build the original text from cleaned_chunks
        original_from_chunks = ''.join(df['text_clean'].tolist())
        
        # Build original and corrected text from sentence lines (remove punctuation for matching)
        def remove_punct(text):
            return re.sub(r'[^\w]', '', text)
        
        original_text = remove_punct(''.join(original_lines))
        corrected_text_full = ''.join(corrected_lines)
        corrected_text = remove_punct(corrected_text_full)
        
        # If lengths match, do character-by-character replacement
        if len(original_from_chunks) == len(corrected_text):
            new_texts = []
            for i, char in enumerate(corrected_text):
                new_texts.append(f'"{char}"')
            df['text'] = new_texts
            df.drop(columns=['text_clean'], inplace=True)
            df.to_excel(CLEANED_CHUNKS_PATH, index=False)
            console.print(f'[green]📝 已同步更新词级时间戳 → {CLEANED_CHUNKS_PATH}[/green]')
        else:
            # Length mismatch - try fuzzy character mapping
            console.print(f"[yellow]⚠️ 字符数不匹配 (词级: {len(original_from_chunks)}, 纠错后: {len(corrected_text)})[/yellow]")
            
            # Use the shorter length for safe replacement
            min_len = min(len(original_from_chunks), len(corrected_text))
            new_texts = []
            for i in range(len(df)):
                if i < min_len:
                    new_texts.append(f'"{corrected_text[i]}"')
                else:
                    # Keep original for extra characters
                    new_texts.append(df.iloc[i]['text'])
            
            df['text'] = new_texts
            df.drop(columns=['text_clean'], inplace=True)
            df.to_excel(CLEANED_CHUNKS_PATH, index=False)
            console.print(f'[yellow]⚠️ 部分更新词级时间戳 (前 {min_len} 个字符) → {CLEANED_CHUNKS_PATH}[/yellow]')
            
    except Exception as e:
        console.print(f"[red]❌ 更新 cleaned_chunks.xlsx 失败: {e}[/red]")

def save_corrected_text(corrected_lines, original_lines=None):
    """Save corrected text back to the source file and update cleaned_chunks.xlsx"""
    with open(SENTENCE_TXT_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(corrected_lines))
    console.print(f'[green]📝 STT 纠错完成，已覆盖 → {SENTENCE_TXT_PATH}[/green]')
    
    # Also update cleaned_chunks.xlsx if original_lines provided
    if original_lines is not None:
        update_cleaned_chunks(original_lines, corrected_lines)


def get_summary(skip_stt_correction: bool = False):
    """Get summary and optionally correct STT errors.
    
    Args:
        skip_stt_correction: If True, skip the STT correction step. 
                            This should be True for Mode 2 (extracted subs) and Mode 3 (provided SRT),
                            as STT correction is only needed for ASR transcription results (Mode 1).
    """
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
    
    # ============ Step 1: Summarize topic and extract terms ============
    from core.prompts_storage import get_summary_prompt, get_stt_correction_prompt
    
    summary_prompt = get_summary_prompt(src_content, custom_terms_json)
    console.print("[cyan]📝 Step 1: Summarizing topic and extracting terminology...[/cyan]")
    
    def valid_summary(response_data):
        required_keys = {'src', 'tgt', 'note'}
        if 'terms' not in response_data:
            return {"status": "error", "message": "Missing 'terms' key"}
        if 'topic' not in response_data:
            return {"status": "error", "message": "Missing 'topic' key"}
        for term in response_data['terms']:
            if not all(key in term for key in required_keys):
                return {"status": "error", "message": "Invalid term format"}
        return {"status": "success", "message": "Summary completed"}

    summary = ask_gpt(summary_prompt, response_json=True, valid_def=valid_summary, log_title='summary')
    
    topic = summary.get('topic', '')
    terms = summary.get('terms', [])
    
    console.print(f"[green]✅ Topic: {topic[:50]}...[/green]")
    console.print(f"[green]✅ Extracted {len(terms)} terms[/green]")
    
    # ============ Step 2: STT Correction with context ============
    if skip_stt_correction:
        console.print("[yellow]⏭️ Skipping STT correction (not needed for provided subtitles)[/yellow]")
    else:
        console.print("[cyan]🔧 Step 2: Correcting STT errors with context...[/cyan]")
        
        # Combine custom terms with extracted terms for context
        all_terms = terms + custom_terms_json['terms']
        
        correction_prompt = get_stt_correction_prompt(src_content, topic, all_terms)
        
        def valid_correction(response_data):
            if 'corrected_lines' not in response_data:
                return {"status": "error", "message": "Missing 'corrected_lines' key"}
            if not isinstance(response_data['corrected_lines'], list):
                return {"status": "error", "message": "'corrected_lines' must be a list"}
            return {"status": "success", "message": "Correction completed"}

        correction_result = ask_gpt(correction_prompt, response_json=True, valid_def=valid_correction, log_title='stt_correction')
        
        # 处理纠错结果
        if 'corrected_lines' in correction_result:
            corrected_lines = correction_result['corrected_lines']
            
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
                
                save_corrected_text(new_lines, original_lines[:line_count])
            else:
                console.print(f"[yellow]⚠️ 纠错行数不匹配 (期望 {line_count}, 得到 {len(corrected_lines)}), 跳过纠错[/yellow]")
    
    # 保存术语和主题
    if 'terms' in summary:
        summary['terms'].extend(custom_terms_json['terms'])
    
    # 保存 terminology
    save_data = {
        'topic': summary.get('topic', ''),
        'terms': summary.get('terms', [])
    }
    
    with open(TERMINOLOGY_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, ensure_ascii=False, indent=4)

    console.print(f'[green]💾 Summary saved to → {TERMINOLOGY_JSON_PATH}[/green]')

if __name__ == '__main__':
    get_summary()