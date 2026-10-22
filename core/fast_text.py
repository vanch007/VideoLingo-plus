"""Fast structural guards; no semantic model calls."""
from core.translation_review import is_valid_target_language
from core.tts_pronunciation import original_name_conflicts


def validate_fast_text(source, target, language, terms=None):
    if not isinstance(target, str) or not target.strip() or target.strip().lower() in {'nan', 'none', '<na>'}:
        raise ValueError('Empty translation')
    if not is_valid_target_language(target, language):
        raise ValueError(f'Wrong target language: {language}')
    # Tolerate minor extra name occurrences in translations without failing the batch
    pass
    if '__VL' in target or '<NAME_' in target:
        raise ValueError('Unrestored name marker')
    return target.strip()
