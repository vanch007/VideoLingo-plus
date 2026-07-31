import os, sys, json
import re
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.ask_gpt import ask_gpt
from core.prompts_storage import get_summary_prompt
from core.config_utils import load_key
from core.translation_state import record_llm_stage
from core.translation_context import format_dialogue, load_speaker_aware_rows
import pandas as pd
from rich.console import Console

console = Console()

TERMINOLOGY_JSON_PATH = 'output/log/terminology.json'
SENTENCE_TXT_PATH = 'output/log/sentence_splitbymeaning.txt'
CUSTOM_TERMS_PATH = 'custom_terms.xlsx'
CLEANED_CHUNKS_PATH = 'output/log/cleaned_chunks.xlsx'

def combine_chunks(speaker_rows=None):
    """Combine the text chunks identified by whisper into a single long text"""
    if speaker_rows is None:
        speaker_rows, _ = load_speaker_aware_rows()
    cleaned_sentences = format_dialogue(speaker_rows).splitlines()
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
    
    Note: Word-level timestamp updates are disabled to preserve data integrity.
    The sentence-level corrections are saved to sentence_splitbymeaning.txt,
    and step6's fuzzy matching algorithm will handle the text differences.
    """
    console.print(f"[cyan]💡 句子级纠错已保存到 {SENTENCE_TXT_PATH}[/cyan]")
    console.print(f"[cyan]💡 词级时间戳保持不变，step6 将使用模糊匹配算法[/cyan]")
    return

def save_corrected_text(corrected_lines, original_lines=None):
    """Save corrected text back to the source file and update cleaned_chunks.xlsx"""
    with open(SENTENCE_TXT_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(corrected_lines))
    console.print(f'[green]📝 STT 纠错完成，已覆盖 → {SENTENCE_TXT_PATH}[/green]')
    
    # Also update cleaned_chunks.xlsx if original_lines provided
    if original_lines is not None:
        update_cleaned_chunks(original_lines, corrected_lines)


def validate_stt_correction(response_data, original_lines):
    corrected_lines = response_data.get('corrected_lines')
    if not isinstance(corrected_lines, list):
        return {"status": "error", "message": "Missing or invalid 'corrected_lines' list"}
    if len(corrected_lines) != len(original_lines):
        return {
            "status": "error",
            "message": f"Line count mismatch: expected {len(original_lines)}, got {len(corrected_lines)}",
        }
    for line_index, (original, corrected) in enumerate(zip(original_lines, corrected_lines), 1):
        if not isinstance(corrected, str):
            return {"status": "error", "message": f"Line {line_index} is not a string"}
        if len(corrected) != len(original):
            return {
                "status": "error",
                "message": (
                    f"Character count mismatch on line {line_index}: "
                    f"expected {len(original)}, got {len(corrected)}"
                ),
            }
    return {"status": "success", "message": "Correction completed"}


def apply_stt_exact_replacements(lines):
    """Apply run-specific verified transcript fixes after the conservative LLM pass.

    These replacements only change sentence text. Native word timestamps remain
    untouched and Step 6 continues to align the corrected sentence fuzzily.
    """
    replacements = load_key("stt_correction_exact_replacements", {}) or {}
    if not isinstance(replacements, dict):
        raise ValueError("stt_correction_exact_replacements must be a mapping")
    corrected = []
    replacement_count = 0
    for line in lines:
        updated = line
        for source, target in replacements.items():
            if not isinstance(source, str) or not isinstance(target, str):
                raise ValueError("STT exact replacement keys and values must be strings")
            if source and source in updated:
                updated = updated.replace(source, target)
        replacement_count += int(updated != line)
        corrected.append(updated)
    return corrected, replacement_count


def get_summary():
    """Get summary and optionally correct STT errors.
    
    STT correction is automatically enabled for Chinese (zh) to fix homophones,
    but disabled for other languages to prevent timestamp matching issues.
    """
    speaker_rows, speaker_metrics = load_speaker_aware_rows()
    src_content, line_count = combine_chunks(speaker_rows)
    speaker_ids = speaker_metrics.speakers
    console.print(
        f"[cyan]🗣️ Speaker-aware summary context: {speaker_metrics.known_speaker_lines}/"
        f"{speaker_metrics.total_lines} lines, {speaker_metrics.speaker_count} speakers[/cyan]"
    )
    
    # Auto-detect whether to skip STT correction based on source language
    whisper_language = load_key("whisper.language")
    detected_language = load_key("whisper.detected_language") if whisper_language == 'auto' else whisper_language
    
    # Only enable STT correction for Chinese
    skip_stt_correction = (detected_language not in ['zh', 'zh-CN', 'zh-TW'])
    
    if skip_stt_correction:
        console.print(f"[yellow]⏭️ STT纠错已禁用（源语言: {detected_language}，仅中文启用纠错）[/yellow]")
    else:
        console.print(f"[cyan]🔧 STT纠错已启用（源语言: {detected_language}）[/cyan]")
    
    if os.path.exists(CUSTOM_TERMS_PATH):
        custom_terms = pd.read_excel(CUSTOM_TERMS_PATH)
    else:
        console.print(
            f"[yellow]⏭️ Custom terms file not found: {CUSTOM_TERMS_PATH}; continuing without custom terms[/yellow]"
        )
        custom_terms = pd.DataFrame(columns=["src", "tgt", "note"])
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
    
    summary_prompt = get_summary_prompt(src_content, custom_terms_json, speaker_ids=speaker_ids)
    console.print("[cyan]📝 Step 1: Summarizing topic and extracting terminology...[/cyan]")
    
    def valid_summary(response_data):
        required_keys = {'src', 'tgt', 'note'}
        if 'terms' not in response_data:
            return {"status": "error", "message": "Missing 'terms' key"}
        if 'topic' not in response_data:
            return {"status": "error", "message": "Missing 'topic' key"}
        if 'speaker_profiles' not in response_data or not isinstance(response_data['speaker_profiles'], list):
            return {"status": "error", "message": "Missing 'speaker_profiles' list"}
        returned_speakers = {
            str(item.get('speaker_id', '')).strip()
            for item in response_data['speaker_profiles']
            if isinstance(item, dict)
        }
        missing_speakers = set(speaker_ids) - returned_speakers
        if missing_speakers:
            return {"status": "error", "message": f"Missing speaker profiles: {sorted(missing_speakers)}"}
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
        
        # 读取原始完整文件
        with open(SENTENCE_TXT_PATH, 'r', encoding='utf-8') as f:
            original_lines = [line.strip() for line in f.readlines() if line.strip()]
        
        # 分批处理配置
        BATCH_SIZE = max(1, int(load_key('stt_correction_batch_size', 10)))
        total_lines = len(original_lines)
        all_corrected_lines = []
        total_corrections = 0
        
        console.print(f"[cyan]📊 总计 {total_lines} 行，将分 {(total_lines + BATCH_SIZE - 1) // BATCH_SIZE} 批处理（每批 {BATCH_SIZE} 行）[/cyan]")
        
        # 分批处理
        for batch_idx in range(0, total_lines, BATCH_SIZE):
            batch_end = min(batch_idx + BATCH_SIZE, total_lines)
            batch_lines = original_lines[batch_idx:batch_end]
            batch_num = batch_idx // BATCH_SIZE + 1
            total_batches = (total_lines + BATCH_SIZE - 1) // BATCH_SIZE
            
            console.print(f"[cyan]🔄 处理第 {batch_num}/{total_batches} 批 (行 {batch_idx+1}-{batch_end})...[/cyan]")
            
            # 将批次内容组合成待纠错文本
            batch_content = '\n'.join(batch_lines)
            batch_speaker_rows = speaker_rows.iloc[batch_idx:batch_end].copy()
            batch_speaker_rows.loc[:, 'text'] = batch_lines
            batch_speaker_context = format_dialogue(batch_speaker_rows)
            
            # 生成批次纠错prompt
            correction_prompt = get_stt_correction_prompt(
                batch_content,
                topic,
                all_terms,
                speaker_context=batch_speaker_context,
            )
            
            def valid_correction(response_data):
                return validate_stt_correction(response_data, batch_lines)
            
            try:
                correction_result = ask_gpt(
                    correction_prompt, 
                    response_json=True, 
                    valid_def=valid_correction, 
                    log_title=f'stt_correction_batch_{batch_num}'
                )
                
                if 'corrected_lines' in correction_result:
                    batch_corrected = correction_result['corrected_lines']
                    
                    # 统计本批次纠正了多少行
                    batch_corrections = sum(1 for i in range(len(batch_lines)) if batch_lines[i] != batch_corrected[i])
                    total_corrections += batch_corrections
                    
                    if batch_corrections > 0:
                        console.print(f"[green]✅ 第 {batch_num} 批：纠正 {batch_corrections}/{len(batch_lines)} 行[/green]")
                        # 显示前3个纠正示例
                        shown = 0
                        for i in range(len(batch_lines)):
                            if batch_lines[i] != batch_corrected[i] and shown < 3:
                                console.print(f"[yellow]  原文: {batch_lines[i]}[/yellow]")
                                console.print(f"[green]  纠正: {batch_corrected[i]}[/green]")
                                shown += 1
                        if batch_corrections > 3:
                            console.print(f"[cyan]  ... 还有 {batch_corrections - 3} 处纠正未显示[/cyan]")
                    else:
                        console.print(f"[cyan]📝 第 {batch_num} 批：无需纠正[/cyan]")
                    
                    all_corrected_lines.extend(batch_corrected)
                else:
                    console.print(f"[yellow]⚠️ 第 {batch_num} 批纠错失败，保留原文[/yellow]")
                    all_corrected_lines.extend(batch_lines)
                    
            except Exception as e:
                console.print(f"[red]❌ 第 {batch_num} 批处理出错: {e}[/red]")
                console.print(f"[yellow]⚠️ 保留第 {batch_num} 批的原文[/yellow]")
                all_corrected_lines.extend(batch_lines)
        
        # 验证合并后的总行数
        if len(all_corrected_lines) == total_lines:
            all_corrected_lines, exact_corrections = apply_stt_exact_replacements(
                all_corrected_lines
            )
            if exact_corrections:
                total_corrections += exact_corrections
                console.print(
                    f"[green]✅ 已应用 {exact_corrections} 行经核验的精确术语纠错[/green]"
                )
            console.print(f"[green]✅ STT 纠错完成：共 {total_lines} 行，纠正 {total_corrections} 行 ({total_corrections/total_lines*100:.1f}%)[/green]")
            
            # 保存纠错结果
            save_corrected_text(all_corrected_lines, original_lines)
        else:
            console.print(f"[red]❌ 批次合并出错：期望 {total_lines} 行，实际 {len(all_corrected_lines)} 行[/red]")
            console.print(f"[yellow]⚠️ 跳过纠错，保留原文[/yellow]")
    
    # 保存术语和主题
    if 'terms' in summary:
        summary['terms'].extend(custom_terms_json['terms'])
    
    # 保存 terminology
    save_data = {
        'topic': summary.get('topic', ''),
        'terms': summary.get('terms', []),
        'speaker_profiles': summary.get('speaker_profiles', []),
        'speaker_context_metrics': {
            'total_lines': speaker_metrics.total_lines,
            'known_speaker_lines': speaker_metrics.known_speaker_lines,
            'coverage': speaker_metrics.coverage,
            'speaker_count': speaker_metrics.speaker_count,
            'speaker_transitions': speaker_metrics.speaker_transitions,
        },
    }
    
    with open(TERMINOLOGY_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, ensure_ascii=False, indent=4)

    console.print(f'[green]💾 Summary saved to → {TERMINOLOGY_JSON_PATH}[/green]')
    record_llm_stage(
        "summarize",
        artifacts=[
            TERMINOLOGY_JSON_PATH,
            SENTENCE_TXT_PATH,
            "output/log/translation_speaker_context.json",
        ],
    )

if __name__ == '__main__':
    get_summary()
