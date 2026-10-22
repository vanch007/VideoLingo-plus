import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import os,sys
import json
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from core.spacy_utils.load_nlp_model import init_nlp
from core.config_utils import load_key, get_joiner
from rich import print
from core.providers.speaker_diarization import validate_cached_speaker_rows
from core.dialogue_segmentation import BOUNDARY_REPORT, group_dialogue_words

def split_by_mark(nlp):
    whisper_language = load_key("whisper.language")
    language = load_key("whisper.detected_language") if whisper_language == 'auto' else whisper_language # consider force english case
    joiner = get_joiner(language)
    print(f"[blue]🔍 Using {language} language joiner: '{joiner}'[/blue]")
    chunks = pd.read_excel("output/log/cleaned_chunks.xlsx")
    chunks.text = chunks.text.apply(lambda x: x.strip('"').strip(""))
    validate_cached_speaker_rows(chunks.to_dict("records"))
    
    speaker_groups, lexical_report = group_dialogue_words(chunks.to_dict("records"), nlp, joiner)
    BOUNDARY_REPORT.write_text(json.dumps(lexical_report, ensure_ascii=False, indent=2), encoding="utf-8")
    sentences_by_mark = []
    boundary_rows = []
    punctuation_only = {',', '.', '?', '!', '，', '。', '？', '！'}

    # Retain acoustic boundaries except recorded short lexical joins. These
    # joins carry unknown identity; no acoustic label is reassigned.
    for group_index, group in enumerate(speaker_groups):
        input_text = group.get("source_text") or joiner.join(group["words"])
        group_sentences = []
        if group.get("source_text"):
            sentence_texts = [input_text.strip()]
        else:
            doc = nlp(input_text)
            assert doc.has_annotation("SENT_START")
            sentence_texts = [sent.text.strip() for sent in doc.sents]
        for sentence in sentence_texts:
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
        f"{len(speaker_groups)} dialogue groups → `sentence_by_mark.txt`[/green]"
    )

if __name__ == "__main__":
    nlp = init_nlp()
    split_by_mark(nlp)
