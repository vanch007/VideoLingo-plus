"""Auditable pronunciation hints for source-bound transliterated glossary terms."""
from __future__ import annotations

from functools import lru_cache
import re


@lru_cache(maxsize=1)
def _dictionary():
    try:
        import cmudict
        return cmudict.dict()
    except ImportError:
        return {}


def source_pronunciations(source: str, terms: list[dict]) -> list[dict]:
    try:
        from pypinyin import lazy_pinyin
    except ImportError:
        return []
    entries, seen = [], set()
    for term in terms:
        src, tgt = term.get("src"), term.get("tgt")
        if (not isinstance(src, str) or not isinstance(tgt, str) or src not in source
                or not re.fullmatch(r"[\u4e00-\u9fff]{2,12}", src)
                or not re.fullmatch(r"[A-Za-z]+(?:[ -][A-Za-z]+){0,3}", tgt)):
            continue
        words = re.findall(r"[A-Za-z]+", tgt)
        syllables = lazy_pinyin(src)
        if "".join(words).lower() != "".join(syllables).lower():
            continue
        phones, cursor = [], 0
        for word in words:
            end = next((end for end in range(cursor + 1, len(syllables) + 1)
                        if "".join(syllables[cursor:end]).casefold() == word.casefold()), None)
            if end is None:
                break
            readings = _dictionary().get(word.casefold(), [])
            if readings:
                phones.append(readings[0])
            else:
                # Joined Romanized syllables can lack a whole-word dictionary
                # entry. Reuse only known readings for each exact source syllable.
                parts = [_dictionary().get(syllable.casefold(), []) for syllable in syllables[cursor:end]]
                if not all(parts):
                    break
                phones.append([phone for part in parts for phone in part[0]])
            cursor = end
        if len(phones) != len(words) or cursor != len(syllables) or tgt.casefold() in seen:
            continue
        entries.append({"src": src, "tgt": tgt, "words": words,
                        "phones": phones,
                        "basis": "exact_source_glossary+pinyin_match+CMU_dictionary"})
        seen.add(tgt.casefold())
    return entries


def annotate_pronunciations(text: str, entries: list[dict]) -> str:
    for entry in sorted(entries, key=lambda value: len(value["tgt"]), reverse=True):
        pattern = r"(?<![A-Za-z0-9_])" + r"[ -]".join(map(re.escape, entry["words"])) + r"(?![A-Za-z0-9_])"
        def replace(match):
            # Preserve the surface characters, including separators, exactly.
            words = iter(entry["phones"])
            return re.sub(r"[A-Za-z]+", lambda word: f"<{word.group()}|{' '.join(next(words))}>", match.group())
        text = re.sub(pattern, replace, text, flags=re.I)
    return text


def source_compound_pronunciations(source: str, text: str, terms: list[dict]) -> list[dict]:
    """Use known dictionary phones for short source-bound hyphenated terms."""
    entries, seen = [], set()
    for term in terms:
        src, tgt = term.get('src'), term.get('tgt')
        if (not isinstance(src, str) or not src or src not in source or not isinstance(tgt, str)
                or '-' not in tgt or not re.fullmatch(r'[A-Za-z]+(?:[ -][A-Za-z]+){1,3}', tgt)):
            continue
        words = re.findall(r'[A-Za-z]+', tgt)
        pattern = r'(?<!\w)' + r'[ -]'.join(map(re.escape, words)) + r's?(?!\w)'
        for match in re.finditer(pattern, text, re.I):
            actual = re.findall(r'[A-Za-z]+', match.group())
            readings = [_dictionary().get(word.casefold(), []) for word in actual]
            if not all(readings) or match.group().casefold() in seen:
                continue
            phones = [variants[0] for variants in readings]
            if sum(phone[-1:].isdigit() for word in phones for phone in word) > 4:
                continue
            entries.append({'src': src, 'tgt': match.group(), 'words': actual, 'phones': phones,
                            'basis': 'exact_source_glossary+short_English_compound+CMU_dictionary'})
            seen.add(match.group().casefold())
    return entries


def short_english_pronunciations(text: str, existing: list[dict]) -> list[dict]:
    """Explicit dictionary phones stabilize isolated short words, without editing them."""
    if re.search(r"\d|[^\x00-\x7f]", text):
        return []
    from core.spoken_length import english_syllables
    if english_syllables(text) > 3:
        return []
    covered = {word.casefold() for entry in existing for word in entry["words"]}
    entries = []
    for word in re.findall(r"[A-Za-z]+", text):
        readings = _dictionary().get(word.casefold(), [])
        if word.casefold() in covered or not readings or "'" in text:
            continue
        # Skip homographs with different stressed vowels or consonants. Ordinary
        # unstressed regional vowel variants do not change lexical meaning.
        salient = {tuple(phone for phone in phones if not phone.endswith('0')) for phones in readings}
        if len(salient) > 1:
            continue
        entries.append({"src": None, "tgt": word, "words": [word], "phones": [readings[0]],
                        "basis": "short_canonical_English_word+unambiguous_CMU_dictionary"})
        covered.add(word.casefold())
    return entries


@lru_cache(maxsize=1)
def _local_g2p():
    """Use installed pronunciation assets only; this path never downloads."""
    try:
        import nltk
        for resource in ('taggers/averaged_perceptron_tagger.zip', 'corpora/cmudict.zip'):
            nltk.data.find(resource)
        from g2p_en import G2p
        return G2p()
    except (ImportError, LookupError):
        return None


def _english_word_phones(word: str) -> tuple[list[str], str] | None:
    readings = _dictionary().get(word.casefold(), [])
    if readings:
        salient = {tuple(phone for phone in phones if not phone.endswith('0')) for phones in readings}
        if len(salient) == 1:
            return readings[0], 'CMU_dictionary'
        # Avoid choosing an ambiguous homograph from a standalone token.
        return None
    if word.isupper() and 1 < len(word) <= 6:
        letters = [_dictionary().get(letter.casefold(), []) for letter in word]
        if all(letters):
            return [phone for entry in letters for phone in entry[0]], 'source_initialism+CMU_letter_names'
    parts = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?![a-z])', word)
    if len(parts) > 1 and ''.join(parts) == word:
        known = [_english_word_phones(part) for part in parts]
        if all(known):
            return [phone for phones, _ in known for phone in phones], 'source_CamelCase_components'
    predictor = _local_g2p()
    if predictor is not None and re.fullmatch(r'[A-Za-z]{3,}', word):
        phones = predictor.predict(word.casefold())
        if phones and all(re.fullmatch(r'[A-Z]+[012]?', phone) for phone in phones):
            return phones, 'local_G2P_prediction_requires_readback'
    return None


def mixed_english_pronunciations(source: str, text: str) -> list[dict]:
    """Repair mixed Chinese/English delivery without changing its written words.

    Only Latin words actually present in this source can receive hints. G2P
    predictions guide a new take, never certify its pronunciation or content.
    """
    if not re.search(r'[\u3400-\u4dbf\u4e00-\u9fff]', text):
        return []
    source_words = {word.casefold(): word for word in re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", source)}
    entries, seen = [], set()
    for word in re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text):
        if word.casefold() not in source_words or word.casefold() in seen:
            continue
        reading = _english_word_phones(word)
        if reading is not None:
            phones, basis = reading
            entries.append({'src': source_words[word.casefold()], 'tgt': word,
                            'words': [word], 'phones': [phones],
                            'basis': 'exact_source_Latin_word+' + basis})
            seen.add(word.casefold())
    return entries


DEFAULT_MANDARIN_TRANSLITERATIONS = {
    "Claude Code": "克劳德代码",
    "Claude code": "克劳德代码",
    "Claude": "克劳德",
    "Parakeet": "帕拉基特",
    "Hyperframes": "海珀帧",
    "Hyperframe": "海珀帧",
    "HyperFrames": "海珀帧",
    "HeyGen": "海珍",
    "Epidemic Sound": "艾皮戴米克桑德",
    "Epidemic Sounds": "艾皮戴米克桑德",
    "Epidemic sounds": "艾皮戴米克桑德",
    "Tella": "泰拉",
    "TELLA": "泰拉",
    "Anthropic": "安思罗匹克",
    "Nvidia": "英伟达",
    "Doritos": "多力多滋",
    "Lay's": "乐事",
    "Lay’s": "乐事",
    "After Effects": "阿夫特伊费克茨",
    "YouTube Edit": "优兔剪辑",
    "YouTube Edits": "优兔剪辑",
    "YouTube": "优兔",
    "OpenAI": "欧盆诶艾",
    "We Are No Code": "维阿诺扣德",
}


def mandarin_transliterations(terms: list[dict] | None = None) -> dict[str, str]:
    """Compile registered English-to-Mandarin spoken transliterations."""
    from core.config_utils import load_key
    mapping = dict(DEFAULT_MANDARIN_TRANSLITERATIONS)
    cfg_trans = load_key("translation.spoken_transliterations", {}) or {}
    if isinstance(cfg_trans, dict):
        mapping.update(cfg_trans)
    backend_trans = load_key("mlx_tts.spoken_transliterations", {}) or {}
    if isinstance(backend_trans, dict):
        mapping.update(backend_trans)
    for term in terms or []:
        if isinstance(term, dict):
            src = term.get("src") or term.get("tgt")
            spoken = term.get("spoken") or term.get("transliteration")
            if isinstance(src, str) and isinstance(spoken, str) and spoken.strip():
                mapping[src] = spoken.strip()
    return mapping


def apply_mandarin_transliterations(text: str, transliterations: dict[str, str] | None = None) -> tuple[str, dict[str, str]]:
    """Replace Latin proper names with fluent Mandarin spoken transliterations.

    Returns (spoken_text, matched_transliterations). Subtitles retain original text.
    """
    if not text or not re.search(r"[A-Za-z]", text):
        return text, {}
    mapping = transliterations if transliterations is not None else mandarin_transliterations()
    matched = {}
    spoken = text
    # Match longer names first to avoid partial word collisions
    for en_term in sorted(mapping.keys(), key=len, reverse=True):
        zh_term = mapping[en_term]
        pattern = r"(?<![A-Za-z0-9_])" + re.escape(en_term) + r"(?![A-Za-z0-9_])"
        if re.search(pattern, spoken, flags=re.I):
            spoken = re.sub(pattern, zh_term, spoken, flags=re.I)
            matched[en_term] = zh_term
    return spoken, matched


def synthesis_text(request) -> str:
    if "synthesis_text" not in request.metadata:
        candidate = request.text
    else:
        candidate = request.metadata.get("synthesis_text", request.text)
    if not isinstance(candidate, str):
        raise ValueError("Pronunciation-guided synthesis text must be a string")
    surface = re.sub(r"<([^|>\n]+)\|[A-Z0-9 .]+>", r"\1", candidate)
    transliterations = request.metadata.get("spoken_transliterations")
    if request.metadata.get('standard_mandarin_guidance'):
        plan = request.metadata.get('spoken_plan')
        expected, _ = apply_mandarin_transliterations(request.text, transliterations)
        if (not isinstance(plan, dict) or plan.get('display_text') != request.text
                or plan.get('spoken_text') != expected):
            raise ValueError('Standard Pinyin guidance must bind the canonical spoken plan')
        annotated, receipts = annotate_standard_mandarin_spans(plan)
        if candidate != annotated or request.metadata.get('spoken_pinyin') != receipts:
            raise ValueError('Standard Pinyin guidance cannot change registered name sounds or other wording')
        surface = expected
    if transliterations and isinstance(transliterations, dict):
        expected, _ = apply_mandarin_transliterations(request.text, transliterations)
        if surface != expected:
            raise ValueError("Pronunciation guidance cannot change canonical translation outside registered transliterations")
    elif 'synthesis_text' in request.metadata and surface != request.text:
        raise ValueError("Pronunciation guidance cannot change the canonical translation")
    # These marks are typography, not spoken words. In paired native v2.5
    # probes they caused extra syllables at both edges of quoted Chinese labels.
    # Keep punctuation pauses and English apostrophes; remove only quote marks.
    candidate = candidate.translate(str.maketrans('', '', '“”「」『』'))
    # Make code-switch word boundaries explicit to the tokenizer, including
    # protected pronunciation annotations. Whitespace does not alter wording.
    candidate = re.sub(r'([\u3400-\u4dbf\u4e00-\u9fff])(?=[A-Za-z<])', r'\1 ', candidate)
    return re.sub(r'(?<=[A-Za-z>])([\u3400-\u4dbf\u4e00-\u9fff])', r' \1', candidate)


def spoken_plan(text: str, language: str, terms=None) -> dict:
    """One forward display-to-speech contract, independent of TTS backend.

    Spans retain occurrence identity even when aliases share the same reading.
    Quote removal and tokenizer whitespace are unspoken formatting, not words.
    """
    mapping = mandarin_transliterations(terms) if str(language).startswith('zh') else {}
    names = sorted((k for k in mapping if isinstance(k, str) and k), key=len, reverse=True)
    pattern = (re.compile(r'(?<![A-Za-z0-9_])(?:' + '|'.join(map(re.escape, names)) +
                          r')(?![A-Za-z0-9_])', re.I) if names else None)
    spans, chunks, cursor, spoken_cursor = [], [], 0, 0
    for match in pattern.finditer(text) if pattern else []:
        unchanged = text[cursor:match.start()]
        chunks.append(unchanged)
        spoken_cursor += len(unchanged)
        name = next(k for k in names if k.casefold() == match.group().casefold())
        reading = mapping[name]
        chunks.append(reading)
        spans.append({'display_span': [match.start(), match.end()],
                      'spoken_span': [spoken_cursor, spoken_cursor + len(reading)],
                      'display': match.group(), 'spoken': reading})
        spoken_cursor += len(reading)
        cursor = match.end()
    chunks.append(text[cursor:])
    return {'display_text': text, 'spoken_text': ''.join(chunks), 'spans': spans,
            'matched_transliterations': {p['display']: p['spoken'] for p in spans}}


def pronunciation_policy_signature(terms=None) -> str:
    from core.utterance_plan import signature
    from core.pipeline.artifact_manifest import hash_path
    return signature({'mapping': mandarin_transliterations(terms), 'worker': hash_path(__file__)})


def annotate_mandarin_spans(plan: dict) -> tuple[str, list[dict]]:
    """Bind registered Chinese name sounds to native Index 2.5 Pinyin tokens."""
    from pypinyin import lazy_pinyin, Style
    chunks, receipts, cursor = [], [], 0
    for span in plan['spans']:
        left, right = span['spoken_span']
        name = plan['spoken_text'][left:right]
        if not re.fullmatch(r'[\u3400-\u4dbf\u4e00-\u9fff]+', name):
            raise ValueError('Mandarin name guidance requires registered Chinese syllables')
        phones = [p.upper() for p in lazy_pinyin(name, style=Style.TONE3, neutral_tone_with_five=True)]
        if len(phones) != len(name) or any(not re.fullmatch(r'[A-Z]+[1-5]', p) for p in phones):
            raise ValueError('Mandarin name lacks explicit tone-bearing Pinyin')
        chunks.extend([plan['spoken_text'][cursor:left], ''.join(f'<{c}|{p}>' for c,p in zip(name,phones))])
        receipts.append({'display':span['display'], 'spoken':name, 'phones':phones,
                         'basis':'registered_spoken_span+tone_bearing_pinyin+native_special_token_2'})
        cursor = right
    chunks.append(plan['spoken_text'][cursor:])
    return ''.join(chunks), receipts


def annotate_standard_mandarin_spans(plan: dict) -> tuple[str, list[dict]]:
    """Use standard IndexTTS 2's native tone-bearing Pinyin vocabulary.

    The 2.0 tokenizer accepts KE4/DE2 directly, unlike 2.5's <字|PINYIN>
    syntax. Only registered name spans change input representation; display
    wording and the forward Mandarin reading remain locked.
    """
    _, receipts = annotate_mandarin_spans(plan)
    chunks, cursor = [], 0
    for span, receipt in zip(plan['spans'], receipts):
        left, right = span['spoken_span']
        chunks.extend([plan['spoken_text'][cursor:left], ' ' + ' '.join(receipt['phones']) + ' '])
        receipt['basis'] = 'registered_spoken_span+tone_bearing_pinyin+standard_native_vocabulary'
        cursor = right
    chunks.append(plan['spoken_text'][cursor:])
    return ''.join(chunks), receipts


def pronunciation_terms() -> list[dict]:
    import json
    from pathlib import Path
    path = Path('output/log/terminology.json')
    return json.loads(path.read_text()).get('terms', []) if path.is_file() else []


def display_aligned_words(plan: dict, words: list[dict]) -> list[dict]:
    """Map measured spoken-name spans back to display spelling, never interpolate."""
    from core.source_verification import normalize_transcript
    edges, pos = {0: 0}, 0
    for index, word in enumerate(words):
        pos += len(normalize_transcript(word['word']))
        edges[pos] = index + 1
    if normalize_transcript(''.join(w['word'] for w in words)) != normalize_transcript(plan['spoken_text']):
        raise ValueError('Target aligner changed locked spoken text')
    result, cursor = [], 0
    for span in plan['spans']:
        left, right = [len(normalize_transcript(plan['spoken_text'][:p])) for p in span['spoken_span']]
        if left not in edges or right not in edges:
            raise ValueError('Spoken name lacks measured word boundaries')
        first, stop = edges[left], edges[right]
        if first >= stop:
            raise ValueError('Spoken name alignment collapsed')
        result.extend(words[cursor:first])
        result.append({'word': span['display'], 'start': words[first]['start'], 'end': words[stop-1]['end']})
        cursor = stop
    result.extend(words[cursor:])
    if normalize_transcript(''.join(w['word'] for w in result)) != normalize_transcript(plan['display_text']):
        raise ValueError('Display/spoken mapping changed canonical text')
    return result


def _registered_name_pattern(terms=None):
    mapping = mandarin_transliterations(terms)
    pattern = r'(?<![A-Za-z0-9_])(?:' + '|'.join(map(re.escape, sorted(mapping, key=len, reverse=True))) + r')(?![A-Za-z0-9_])'
    return re.compile(pattern, re.I)


def protected_source_names(source: str, terms=None) -> list[str]:
    """Names explicitly registered for spoken localization keep source spelling."""
    return list(dict.fromkeys(m.group() for m in _registered_name_pattern(terms).finditer(source)))


def original_name_conflicts(source: str, target: str, language: str, terms=None) -> list[str]:
    if not str(language).startswith('zh'):
        return []
    from collections import Counter
    pattern = _registered_name_pattern(terms)
    wanted = Counter(m.group() for m in pattern.finditer(source))
    actual = Counter(m.group() for m in pattern.finditer(target))
    return [f'Keep the original display spelling {name!r} exactly {wanted[name]} time(s); '
            'its Mandarin reading belongs only to synthesis.'
            for name in dict.fromkeys([*wanted, *actual]) if wanted[name] != actual[name]]


def canonical_name_spellings(source: str, target: str, language: str, terms=None) -> tuple[str, list[dict]]:
    """Canonicalize unambiguous registered name aliases BEFORE semantic audit.

    Never infer a referent, resolve colliding aliases, or alter already accepted
    speech. Ambiguity remains a display-name gate failure for the translator.
    """
    if not str(language).startswith('zh'):
        return target, []
    names = protected_source_names(source, terms)
    mapping = mandarin_transliterations(terms)
    changes = []
    # A summary/glossary can capitalize a tool differently from its source.
    # Restore only case-equivalent complete registered occurrences; never fix
    # typos, plural forms, missing names, or ambiguous source spellings.
    pattern = _registered_name_pattern(terms)
    source_groups, target_groups = {}, {}
    for match in pattern.finditer(source):
        source_groups.setdefault(match.group().casefold(), []).append(match.group())
    for match in pattern.finditer(target):
        target_groups.setdefault(match.group().casefold(), []).append(match)
    replacements = []
    for identity, originals in source_groups.items():
        matches = target_groups.get(identity, [])
        if len(set(originals)) != 1 or len(matches) != len(originals):
            continue
        name = originals[0]
        for match in matches:
            if match.group() != name:
                replacements.append((match.start(), match.end(), name))
                changes.append({'source_name': name, 'registered_alias': match.group(), 'occurrences': 1,
                                'basis': 'complete_registered_name+case_only+exact_source_name_count_before_semantic_audit'})
    for left, right, name in sorted(replacements, reverse=True):
        target = target[:left] + name + target[right:]
    aliases = {}
    for name in names:
        registered = next(k for k in mapping if k.casefold() == name.casefold())
        aliases.setdefault(mapping[registered], []).append(name)
    for alias, originals in sorted(aliases.items(), key=lambda item: len(item[0]), reverse=True):
        if len(originals) != 1 or alias in source:
            continue
        name = originals[0]
        pattern = r'(?<![A-Za-z0-9_])' + re.escape(name) + r'(?![A-Za-z0-9_])'
        missing = len(re.findall(pattern, source)) - len(re.findall(pattern, target))
        if missing > 0 and target.count(alias) == missing:
            target = target.replace(alias, name)
            changes.append({'source_name': name, 'registered_alias': alias, 'occurrences': missing,
                            'basis': 'unique_registered_alias+exact_source_name_count_before_semantic_audit'})
    return target, changes


def spoken_name_issues(text: str, transcript: str, language: str, terms=None) -> list[str]:
    """Sentence averages cannot excuse a missing or altered registered name."""
    from core.providers.quality import mandarin_reading
    plan = spoken_plan(text, language, terms)
    if not plan['spans'] or not transcript:
        return []
    # An ASR may spell a heard name with its registered Latin spelling. Only
    # exact registered spellings receive the same forward normalization.
    heard = spoken_plan(transcript, language, terms)['spoken_text']
    wanted_reading, heard_reading = mandarin_reading(plan['spoken_text']), mandarin_reading(heard)
    if wanted_reading is None or heard_reading is None:
        return ['spoken_name_reading_unavailable']
    def count(sequence, needle):
        return sum(sequence[i:i+len(needle)] == needle for i in range(len(sequence)-len(needle)+1))
    issues = []
    for name in {s['spoken'] for s in plan['spans']}:
        reading = mandarin_reading(name)
        if reading is None or count(wanted_reading, reading) != count(heard_reading, reading):
            issues.append('spoken_name_mismatch:' + name)
    return issues


def spoken_name_witnesses(text: str, readbacks: list[dict], language: str, terms=None) -> list[dict]:
    """Require exact sound/count evidence for every name on the same audio.

    ASR spellings may disagree on different names. Each registered reading
    still needs a complete, successful independent readback; sentence scores,
    hints and unavailable/empty recognizers never constitute name evidence.
    """
    plan = spoken_plan(text, language, terms)
    heard = [(item['backend'], spoken_name_issues(text,item['transcript'],language,terms))
             for item in readbacks if item.get('status') == 'ok'
             and isinstance(item.get('transcript'),str) and item['transcript'].strip()]
    return [{'spoken':name, 'display_spellings':list(dict.fromkeys(
                span['display'] for span in plan['spans'] if span['spoken'] == name)),
             'confirmed_by':[backend for backend,issues in heard
                             if 'spoken_name_mismatch:'+name not in issues
                             and 'spoken_name_reading_unavailable' not in issues],
             'passed':any('spoken_name_mismatch:'+name not in issues
                          and 'spoken_name_reading_unavailable' not in issues for _,issues in heard)}
            for name in dict.fromkeys(span['spoken'] for span in plan['spans'])]
