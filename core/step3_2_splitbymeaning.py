import sys,os,math
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import concurrent.futures
from core.ask_gpt import ask_gpt
from core.prompts_storage import get_split_prompt
from difflib import SequenceMatcher
import math
from core.spacy_utils.load_nlp_model import init_nlp
from core.config_utils import load_key, get_joiner
from rich.console import Console
from rich.table import Table

console = Console()

def tokenize_sentence(sentence, nlp):
    # tokenizer counts the number of words in the sentence
    doc = nlp(sentence)
    return [token.text for token in doc]

def find_split_positions(original, modified):
    split_positions = []
    parts = modified.split('[br]')
    start = 0
    whisper_language = load_key("whisper.language")
    language = load_key("whisper.detected_language") if whisper_language == 'auto' else whisper_language
    joiner = get_joiner(language)

    for i in range(len(parts) - 1):
        max_similarity = 0
        best_split = None

        for j in range(start, len(original)):
            original_left = original[start:j]
            modified_left = joiner.join(parts[i].split())

            left_similarity = SequenceMatcher(None, original_left, modified_left).ratio()

            if left_similarity > max_similarity:
                max_similarity = left_similarity
                best_split = j

        if max_similarity < 0.9:
            console.print(f"[yellow]Warning: low similarity found at the best split point: {max_similarity}[/yellow]")
        if best_split is not None:
            split_positions.append(best_split)
            start = best_split
        else:
            console.print(f"[yellow]Warning: Unable to find a suitable split point for the {i+1}th part.[/yellow]")

    return split_positions

def split_sentence(sentence, num_parts, word_limit=18, index=-1, retry_attempt=0, force_split=False):
    """Split a long sentence using GPT and return the result as a string.
    Args:
        force_split: If True, force split even if it results in single punctuation.
    """
    split_prompt = get_split_prompt(sentence, num_parts, word_limit)

    def valid_split(response_data):
        if 'split' not in response_data:
            return {"status": "error", "message": "Missing required key: `split`"}
        if "[br]" not in response_data["split"]:
            return {"status": "error", "message": "Split failed, no [br] found"}

        # Validate split parts
        parts = response_data["split"].split('[br]')
        if not force_split:
            for part in parts:
                part = part.strip()
                # Avoid splitting on single punctuation or single character
                if len(part) == 1 and part in ',.?!，。？！' or len(part) == 1:
                    return {"status": "error", "message": "Split resulted in single punctuation or character"}
                # Avoid splitting in the middle of a phrase
                if part.endswith(('的', '地', '得', '了', '着', '过')):
                    return {"status": "error", "message": "Split in the middle of a phrase"}

        return {"status": "success", "message": "Split completed"}

    response_data = ask_gpt(split_prompt + ' ' * retry_attempt, response_json=True, valid_def=valid_split, log_title='sentence_splitbymeaning')
    best_split = response_data["split"]
    split_points = find_split_positions(sentence, best_split)

    # split the sentence based on the split points
    for i, split_point in enumerate(split_points):
        if i == 0:
            best_split = sentence[:split_point] + '\n' + sentence[split_point:]
        else:
            parts = best_split.split('\n')
            last_part = parts[-1]
            parts[-1] = last_part[:split_point - split_points[i-1]] + '\n' + last_part[split_point - split_points[i-1]:]
            best_split = '\n'.join(parts)

    if index != -1:
        console.print(f'[green]✅ Sentence {index} has been successfully split[/green]')

    table = Table(title="")
    table.add_column("Type", style="cyan")
    table.add_column("Sentence")
    table.add_row("Original", sentence, style="yellow")
    table.add_row("Split", best_split.replace('\n', ' ||'), style="yellow")
    console.print(table)

    return best_split

def parallel_split_sentences(sentences, max_length, max_workers, nlp, retry_attempt=0):
    """Split sentences in parallel using a thread pool."""
    new_sentences = [None] * len(sentences)
    futures = []

    # 添加调试信息，显示句子长度统计
    total_sentences = len(sentences)
    sentences_to_split = 0
    max_token_length = 0
    min_token_length = float('inf') if total_sentences > 0 else 0

    console.print(f"[cyan]开始处理 {total_sentences} 个句子，分割阈值为 {max_length} tokens[/cyan]")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for index, sentence in enumerate(sentences):
            # Use tokenizer to split the sentence
            tokens = tokenize_sentence(sentence, nlp)
            token_length = len(tokens)

            # 更新统计信息
            max_token_length = max(max_token_length, token_length)
            min_token_length = min(min_token_length, token_length)

            # 显示分词结果
            console.print(f"[句子 {index+1}/{total_sentences}] 长度: {token_length} tokens, 内容: {sentence[:30]}{'...' if len(sentence) > 30 else ''}")
            if token_length > 0:
                console.print(f"  分词结果: {tokens[:10]}{'...' if len(tokens) > 10 else ''}")

            num_parts = math.ceil(token_length / max_length)
            if token_length > max_length:
                sentences_to_split += 1
                console.print(f"[yellow]需要分割为 {num_parts} 部分[/yellow]")
                future = executor.submit(split_sentence, sentence, num_parts, max_length, index=index, retry_attempt=retry_attempt)
                futures.append((future, index, num_parts, sentence))
            else:
                new_sentences[index] = [sentence]

        for future, index, num_parts, sentence in futures:
            split_result = future.result()
            if split_result:
                split_lines = split_result.strip().split('\n')
                new_sentences[index] = [line.strip() for line in split_lines]
            else:
                new_sentences[index] = [sentence]

    # 添加统计信息
    result_sentences = [sentence for sublist in new_sentences for sentence in sublist]

    # 显示统计信息
    console.print(f"[cyan]句子统计:[/cyan]")
    console.print(f"  原始句子数量: {total_sentences}")
    console.print(f"  需要分割的句子数量: {sentences_to_split} ({sentences_to_split/total_sentences*100:.1f}%)")
    console.print(f"  分割后的句子数量: {len(result_sentences)}")
    console.print(f"  最短句子长度: {min_token_length} tokens")
    console.print(f"  最长句子长度: {max_token_length} tokens")
    console.print(f"  分割阈值: {max_length} tokens")

    return result_sentences

def split_sentences_by_meaning():
    """The main function to split sentences by meaning."""
    # 检查输出文件是否已存在，如果存在则跳过
    if os.path.exists('output/log/sentence_splitbymeaning.txt'):
        console.print("[yellow]File 'sentence_splitbymeaning.txt' already exists. Skipping split_sentences_by_meaning.[/yellow]")
        return

    # 检查输入文件是否存在
    if not os.path.exists('output/log/sentence_splitbynlp.txt'):
        console.print("[red]Error: Input file 'sentence_splitbynlp.txt' does not exist. Cannot proceed with meaning-based splitting.[/red]")
        return

    # read input sentences
    with open('output/log/sentence_splitbynlp.txt', 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f.readlines()]

    nlp = init_nlp()
    # 🔄 process sentences multiple times to ensure all are split
    # 临时降低 max_split_length 值进行测试
    test_max_length = 10  # 将阈值降低到 10，强制分割更多句子
    console.print(f"[cyan]测试模式: 使用降低的 max_split_length={test_max_length} 而不是配置文件中的 {load_key('max_split_length')}[/cyan]")

    for retry_attempt in range(3):
        sentences = parallel_split_sentences(sentences, max_length=test_max_length, max_workers=load_key("max_workers"), nlp=nlp, retry_attempt=retry_attempt)

    # 比较分割前后的句子
    with open('output/log/sentence_splitbynlp.txt', 'r', encoding='utf-8') as f:
        original_sentences = [line.strip() for line in f.readlines()]

    # 显示分割前后的差异
    console.print(f"[cyan]分割前句子数量: {len(original_sentences)}[/cyan]")
    console.print(f"[cyan]分割后句子数量: {len(sentences)}[/cyan]")

    if len(original_sentences) == len(sentences) and all(a == b for a, b in zip(original_sentences, sentences)):
        console.print("[yellow]警告: 分割前后的句子完全相同，没有进行任何分割[/yellow]")
    else:
        diff_count = sum(1 for a, b in zip(original_sentences, sentences) if a != b)
        console.print(f"[green]有 {diff_count} 个句子被分割或修改[/green]")

    # 💾 save results
    with open('output/log/sentence_splitbymeaning.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(sentences))
    console.print('[green]✅ All sentences have been successfully split![/green]')

if __name__ == '__main__':
    # print(split_sentence('Which makes no sense to the... average guy who always pushes the character creation slider all the way to the right.', 2, 22))
    split_sentences_by_meaning()