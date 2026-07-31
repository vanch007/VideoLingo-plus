import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import os,sys
import json
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from core.spacy_utils.load_nlp_model import init_nlp
from core.config_utils import load_key, get_joiner
from rich import print
from core.providers.speaker_diarization import group_word_rows_by_speaker, validate_cached_speaker_rows

def split_by_mark(nlp):
    whisper_language = load_key("whisper.language")
    language = load_key("whisper.detected_language") if whisper_language == 'auto' else whisper_language # consider force english case
    joiner = get_joiner(language)
    print(f"[blue]🔍 Using {language} language joiner: '{joiner}'[/blue]")
    chunks = pd.read_excel("output/log/cleaned_chunks.xlsx")
    chunks.text = chunks.text.apply(lambda x: x.strip('"').strip(""))
    validate_cached_speaker_rows(chunks.to_dict("records"))
    
    speaker_groups = group_word_rows_by_speaker(chunks.to_dict("records"))
    sentences_by_mark = []
    boundary_rows = []
    punctuation_only = {',', '.', '?', '!', '，', '。', '？', '！'}

    # NLP is run independently inside every speaker turn. No sentence detector
    # can join text across a MOSS speaker boundary after this point.
    for group_index, group in enumerate(speaker_groups):
        input_text = joiner.join(group["words"])
        doc = nlp(input_text)
        assert doc.has_annotation("SENT_START")
        group_sentences = []
        for sentence in (sent.text.strip() for sent in doc.sents):
            if not sentence:
                continue
            if group_sentences and sentence in punctuation_only:
                group_sentences[-1] += sentence
            else:
                group_sentences.append(sentence)
        for sentence in group_sentences:
            sentences_by_mark.append(sentence)
            boundary_rows.append(
                {"line": len(sentences_by_mark), "speaker": group["speaker"], "speaker_group": group_index}
            )

    with open("output/log/sentence_by_mark.txt", "w", encoding="utf-8") as output_file:
        for sentence in sentences_by_mark:
            output_file.write(sentence + "\n")

    with open("output/log/speaker_sentence_boundaries.json", "w", encoding="utf-8") as output_file:
        json.dump(boundary_rows, output_file, ensure_ascii=False, indent=2)
    
    print(
        f"[green]💾 {len(sentences_by_mark)} sentences saved with "
        f"{len(speaker_groups)} hard speaker groups → `sentence_by_mark.txt`[/green]"
    )

if __name__ == "__main__":
    nlp = init_nlp()
    split_by_mark(nlp)
