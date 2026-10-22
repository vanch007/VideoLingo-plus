import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from spacy_utils.split_by_comma import split_by_comma_main
from spacy_utils.split_by_connector import split_sentences_main
from spacy_utils.split_by_mark import split_by_mark
from spacy_utils.split_long_by_root import split_long_by_root_main
from spacy_utils.load_nlp_model import init_nlp

def split_by_spacy():
    if os.path.exists('output/log/sentence_splitbynlp.txt'):
        print("File 'sentence_splitbynlp.txt' already exists. Skipping split_by_spacy.")
        return
    
    nlp = init_nlp()
    split_by_mark(nlp)
    # The recognizer has already supplied coherent timed dialogue units.
    # Comma splitting separates a promise from its actor/recipient; display
    # wrapping belongs after complete-unit translation instead.
    from core.dialogue_segmentation import BOUNDARY_REPORT
    import json
    from pathlib import Path
    report = json.loads(BOUNDARY_REPORT.read_text(encoding="utf-8"))
    if report.get("policy", "").startswith("recognition_phrases_and_punctuation_preserved"):
        Path('output/log/sentence_splitbynlp.txt').write_text(
            Path('output/log/sentence_by_mark.txt').read_text(encoding="utf-8"), encoding="utf-8")
        return
    split_by_comma_main(nlp)
    split_sentences_main(nlp)
    split_long_by_root_main(nlp)
    return

if __name__ == '__main__':
    split_by_spacy()
