"""Translation owns duration adaptation; measured TTS precedes acceptance."""
from __future__ import annotations

import json
import math
import re
from functools import lru_cache

from core.ask_gpt import ask_gpt
from core.config_utils import load_key
from core.runtime_context import effective_target_language
from core.duration_candidate_audit import audit_duration_candidate

SOURCE_PROFANITY = re.compile(r"他[妈娘]|妈的|操你|他奶奶|狗日|王八蛋|混蛋|滚蛋|该死|fuck|damn|shit", re.I)
TARGET_PROFANITY = re.compile(r"\b(?:the\s+hell|fuck(?:ing)?|(?:god)?damn(?:ed)?|dammit|shit|bastard)\b", re.I)


def normalize_question_emphasis(source: str, text: str) -> tuple[str, list[str]]:
    """Remove only an unspoken question expletive, never source profanity."""
    if SOURCE_PROFANITY.search(source) or re.search(r"地狱|地獄|hell", source, re.I):
        return text, []
    text, count = re.subn(r"\b(what|who|where|when|why|how)\s+the\s+hell\b", r"\1", text, flags=re.I)
    return text, ["unsupported_question_expletive_removed"] if count else []


def spoken_word_count(text: str) -> int:
    # Short numeric notation on screen is still pronounced in full.
    def cardinal_words(number):
        if number < 100:
            return 1 if number < 21 or number % 10 == 0 else 2
        if number < 1000:
            return 2 + (cardinal_words(number % 100) if number % 100 else 0)
        for scale in (10 ** 12, 10 ** 9, 10 ** 6, 1000):
            if number >= scale:
                return cardinal_words(number // scale) + 1 + (cardinal_words(number % scale) if number % scale else 0)
    def numeric_words(match):
        parts = match.group().replace(",", "").split(".")
        count = cardinal_words(int(parts[0]))
        if len(parts) == 2:
            count += 1 + len(parts[1])
        return " ".join(["number"] * count)
    text = re.sub(r"\d+(?:,\d{3})*(?:\.\d+)?", numeric_words, text)
    return len(re.findall(r"[^\W_]+(?:['’][^\W_]+)*", text, flags=re.UNICODE))


def budget_for(row: dict) -> dict:
    seconds = float(row["duration"])
    rate = float(load_key("translation.speech_budget.words_per_second", 2.7))
    max_speed = float(load_key("translation.speech_budget.max_speed_factor", 1.08))
    if (not math.isfinite(rate) or rate <= 0 or not math.isfinite(seconds) or seconds <= 0
            or not math.isfinite(max_speed) or max_speed < 1):
        raise ValueError("Invalid speech budget")
    from core.spoken_length import source_form_hint
    return {"unit_id": row["line_id"], "target_speech_seconds": round(seconds, 3),
            "max_natural_speech_seconds": round(seconds * max_speed, 3),
            "max_speed_factor": max_speed,
            "target_max_words": max(2, int(math.ceil(seconds * rate))),
            "estimated_words_per_second": rate, "basis": "initial_estimate; measured_TTS_is_authoritative",
            "source_form_hint":source_form_hint(str(row.get("text",row.get("source",""))))
            if effective_target_language("en").startswith("en") else ""}


def constraints(budgets: list[dict]) -> str:
    lines = []
    for i, budget in enumerate(budgets, 1):
        cap = budget.get("max_natural_speech_seconds", budget["target_speech_seconds"] * budget["max_speed_factor"])
        length_hint = (f"尽量压到约 {budget['search_max_chinese_units']} 个读音单元（汉字按字、夹杂英文按音节估算），"
                       "采用紧凑自然的中文表达。" if budget.get('search_max_chinese_units') else
                       f"尽量用不超过 {budget['target_max_words']} 个读出来的词，优先缩短音节和表达结构。")
        lines.append(f"第{i}句：目标 {budget['target_speech_seconds']:.3f} 秒，完整自然配音最多 {cap:.3f} 秒（包含上限）；"
                     f"最多允许 {budget['max_speed_factor']:.3f} 倍轻微加速；"
                     + length_hint)
        if budget.get("search_max_syllables"):
            lines.append(f"第{i}句候选搜索目标：英语读音尽量不超过约 {budget['search_max_syllables']} 个音节。"
                         "这是实测超时声音推导的搜索建议，不是验收阈值；不同用词与停连节奏的时长不同，不能为满足估算而删掉意思。"
                         "使用紧凑自然的同义表达和口语省略，仍完整保留语义；最终以完整声音的实际时长验收。")
        if budget.get("source_form_hint"):
            lines.append(f"第{i}句口语结构要求：{budget['source_form_hint']}")
    return ("\n【配音长度目标】\n" + "\n".join(lines) +
            "\n数字按读出来的词计算，连字符不减少词数。参考上限不是硬性词数限制，不能少译事实；"
            "只有必要事实或专名要求更多词时才超出参考值，完整实际声音才是验收依据。"
            "不用长停顿凑语气，不用空话补时长，最多按给定上限轻微加速。")


@lru_cache(maxsize=1)
def _source_continuation_parser():
    import spacy
    return spacy.load(dict(load_key('spacy_model_map', {})).get('en', 'en_core_web_md'), exclude=['ner'])


def source_continuation_context(source: str, previous: str) -> dict | None:
    """Expose a real dependency crossing a source split, never join assertions.

    The parser only proposes grammar. The original two strings remain the
    evidence; all current facts and caption correspondences still need review.
    """
    if not previous.strip() or re.search(r'[.!?。！？][\"\'’”)]*\s*$', previous):
        return None
    try:
        cut = len(previous) + 1
        doc = _source_continuation_parser()(previous + ' ' + source)
        links = []
        for token in doc:
            if (token.idx < cut or token.head.idx >= cut or token.dep_ not in
                    {'dobj', 'obj', 'pobj', 'advcl', 'ccomp', 'xcomp', 'acl', 'relcl', 'attr', 'appos', 'conj'}):
                continue
            span = [t for t in token.subtree if t.idx >= cut]
            if not span:
                continue
            left, right = min(t.idx for t in span), max(t.idx + len(t.text) for t in span)
            links.append({'current_source_quote': source[left-cut:right-cut],
                          'preceding_governor_quote': token.head.text, 'dependency': token.dep_})
        if not links:
            return None
        return {'current_source_quote': source, 'preceding_source_quote': previous,
                'grammar_hypotheses': links,
                'scope': 'Interpret only the grammatical role and referent of the current fragment; '
                         'preceding independent assertions are not added to this fragment.'}
    except (OSError, ImportError):
        # No grammar witness is not an acceptance. Ordinary strict review
        # remains available, without guessing a relationship between units.
        return None


def duration_context(unit: dict, terminology: dict) -> str:
    """A duration edit needs meaning constraints, not the full scene schema."""
    dialogue = terminology.get('dialogue_units', [])
    # Synthesis IDs are children of original dialogue IDs. Resolve the exact
    # source lineage before looking up roles or neighboring dialogue.
    slices = unit.get('source_slices') or [{
        'line_id': unit.get('parent_line_id', unit['unit_id']), 'source': unit['source']}]
    bound = {}
    for part in slices:
        bound.setdefault(part['line_id'], []).append(part['source'])
    roles = [row for row in dialogue if
             (row.get('line_id') == unit['unit_id'] and row.get('source_text', unit['source']) == unit['source'])
             or (row.get('line_id') in bound and isinstance(row.get('source_text'), str)
                 and all(part in row['source_text'] for part in bound[row['line_id']]))]
    terms = [{k:term.get(k,"") for k in ("src","tgt","note")} for term in terminology.get("terms",[])
             if term.get("src") and str(term["src"]) in unit["source"]]
    characters = {c.get("character_id"): c for c in terminology.get("characters", [])}
    def label(value):
        profile = characters.get(value, {})
        return profile.get("name") or profile.get("label") or "unknown"
    role = {}
    if roles:
        for field in ('character_id', 'addressed_to', 'relays_character', 'action_agent', 'action_patient', 'speech_act'):
            values = {row.get(field) for row in roles}
            role[field] = next(iter(values)) if len(values) == 1 else None
        confidence = [row.get('confidence') for row in roles]
        role['confidence'] = (confidence[0] if all(v == confidence[0] for v in confidence) and confidence[0] is not None
                              else min(confidence) if all(isinstance(v, (int, float)) for v in confidence) else 'unknown')
        role['identity_verified'] = all(row.get('identity_verified') is True for row in roles)
    surrounding=[]
    preceding=[]
    if roles:
        positions = [dialogue.index(row) for row in roles]
        current_ids = {row['line_id'] for row in roles}
        surrounding=[row["source_text"] for i,row in enumerate(dialogue)
                     if row.get('line_id') not in current_ids and min(abs(i-position) for position in positions)<=2
                     and isinstance(row.get("source_text"),str)
                     and row["source_text"].strip()]
        preceding=[row['source_text'] for i,row in enumerate(dialogue)
                   if min(positions)-2 <= i < min(positions)
                   and isinstance(row.get('source_text'),str) and row['source_text'].strip()]
    continuation = None
    if roles:
        left = min(dialogue.index(row) for row in roles)
        if left > 0:
            previous = dialogue[left - 1]
            voice = unit.get('speaker')
            if (voice and voice not in {'unknown', 'UNKNOWN'}
                    and previous.get('acoustic_speaker') == voice
                    and all(row.get('acoustic_speaker') == voice for row in roles)):
                continuation = source_continuation_context(unit['source'], previous.get('source_text', ''))
    hints = {key: label(role.get(field)) for key, field in (
        ("speaker", "character_id"), ("addressee", "addressed_to"),
        ("relayed_speaker", "relays_character"), ("action_agent", "action_agent"),
        ("action_patient", "action_patient"))}
    hints.update(confidence=role.get("confidence", "unknown"),
                 identity_verified=role.get("identity_verified", False), advisory=True)
    result = {"parent_source":unit["parent_source"], "glossary_meanings":terms,
                       "topic": terminology.get('topic', ''),
                       "surrounding_source":surrounding,
                       "preceding_source":preceding,
                       "speech_act":role.get('speech_act') or 'unknown',
                       "context_source_line_ids":[row['line_id'] for row in roles],
                       "dialogue_role_hints": hints,
                       "source_pronouns_are_authoritative":True}
    if continuation:
        result['source_continuation'] = continuation
    return json.dumps(result,ensure_ascii=False)


def candidate_model_contract() -> str | None:
    name = load_key("translation.speech_budget.candidate_provider", "")
    if not name:
        return None
    from core.llm_provider import completion_fingerprint, get_llm_provider_config
    from core.homura_translation import is_homura
    provider = get_llm_provider_config(str(name))
    return completion_fingerprint(provider, response_json=not is_homura(provider))


def with_source_category_terms(context: str, categories: list[dict]) -> str:
    try:
        meanings=json.loads(context) if context else {}
    except ValueError:
        meanings={}
    if not isinstance(meanings,dict):
        meanings={}
    # Automatically proposed category labels guide wording; they are not
    # approved glossary meanings and must not redefine source semantics.
    if categories:
        meanings["source_category_guidance"] = categories
    else:
        # No guidance is the same semantic input as the original context.
        # Do not create a second audit identity solely for an empty field.
        meanings.pop("source_category_guidance", None)
    return json.dumps(meanings,ensure_ascii=False)


def compact_chinese_framing(source: str, current: str) -> list[str]:
    """Propose brief Chinese framing only for explicit source constructions.

    These are search candidates, never semantic verdicts. Keep list items and
    the subject intact; current whole-source, name, display and actual speech
    gates must approve the entire result before it can be used.
    """
    proposals = []
    open_list = re.search(r'\b(?:yada\s*,?\s*yada|etc(?:etera)?\.?|and\s+so\s+(?:on|forth))'
                          r'\s*[.!]?\s*$', source, re.I)
    if (open_list and source.count(',') >= 2
            and len(re.findall(r'[、，,]', current)) >= 2):
        brief = re.sub(r'[、，,\s]*(?:等{2,}|诸如此类)[。.!！]?\s*$', '等。', current)
        if brief != current:
            proposals.append(brief)
    same_subject = re.fullmatch(r'\s*(?:and\s+)?(?:as\s+)?for\s+[^,]+,\s*'
                               r"it(?:['’]s|\s+is)\s+the\s+same\s+thing[.!]?\s*", source, re.I)
    framing = re.fullmatch(r'(?:而?至于|对于)(.+?)[，,]\s*也是一样的(?:操作|做法|事情|情况)[。.!！]?', current)
    if same_subject and framing:
        proposals.append(framing.group(1) + '也一样。')
    return list(dict.fromkeys(proposals))


def retranslate_unit(source: str, current: str, budget: dict, context: str, *, measured_seconds=None,
                     measured_candidates=None) -> str:
    from core.local_translation import validate_translation_rows, clean_dialogue_rendering, spoken_translation_prompt, TranslationNameGuard
    from core.translation_invariants import source_fact_conflicts
    target = effective_target_language("en")
    max_speed = float(budget.get("max_speed_factor", load_key("translation.speech_budget.max_speed_factor", 1.08)))
    max_natural_seconds = budget.get("max_natural_speech_seconds", budget["target_speech_seconds"] * max_speed)
    budget = dict(budget)
    budget["max_natural_speech_seconds"] = max_natural_seconds
    if target.startswith('zh') and measured_seconds and measured_seconds > max_natural_seconds + 1e-6:
        from core.spoken_length import chinese_spoken_units
        from core.tts_pronunciation import spoken_plan, pronunciation_terms
        # The duration witness heard the spoken transliteration, not the
        # shorter Latin spelling displayed in subtitles.
        count = chinese_spoken_units(spoken_plan(current, target, pronunciation_terms())['spoken_text'])
        budget['search_max_chinese_units'] = max(1, min(count - 1,
            math.floor(count * max_natural_seconds / measured_seconds)))
        budget['measured_spoken_units'] = count
    if target.startswith("en"):
        from core.spoken_length import english_syllables, source_form_hint
        budget["source_form_hint"] = source_form_hint(source)
        if measured_seconds and measured_seconds > max_natural_seconds + 1e-6:
            count = english_syllables(current)
            # Material overruns need an actual reduction in pronunciation;
            # small DSP/voice variance may keep the same syllable count.
            limit = math.ceil(count * max_natural_seconds / measured_seconds)
            if measured_seconds / max_natural_seconds > 1.10:
                limit = min(count-1,limit)
            budget["search_max_syllables"] = max(1,limit)
    rejected = {}
    for candidate in measured_candidates or []:
        if (isinstance(candidate.get("text"), str) and candidate.get("natural_seconds", 0.) > max_natural_seconds + 1e-6):
            text = candidate["text"]
            rejected[text] = min(rejected.get(text, math.inf), float(candidate["natural_seconds"]))
    feedback = {"current_draft":current, "natural_seconds":measured_seconds,
                "failed_drafts":[{"text":text,"natural_seconds":seconds} for text,seconds in rejected.items()]}
    delivery_feedback = budget.get('speech_delivery_feedback')
    if delivery_feedback:
        feedback['speech_delivery_feedback'] = delivery_feedback
        feedback['delivery_instruction'] = (
            '同一稿多次完整回读仍不符，请重新从源文翻译成自然易读的同义表达。'
            '回读不是正确译文，不能照抄其错词；保留源文全部动作、列举项、专名、数字、否定与关系。')
    if target.startswith('zh') and measured_seconds and measured_seconds > max_natural_seconds + 1e-6:
        max_chars = budget.get('search_max_chinese_units')
        feedback["speech_compaction_instruction"] = (
            f"上一稿‘{current}’实测已超时（耗时{measured_seconds:.2f}s，限额{max_natural_seconds:.2f}s）。"
            f"严禁原样复读旧稿或仅改标点！整句尽量压到约{max_chars}个实际读音单元，包含专名中文音译，"
            "只压缩其余语言的冗余表达，保留完整事实。此估算不代替完整录音的实测验收。"
        )
    if budget.get("source_semantic_feedback"):
        review=budget['source_semantic_feedback']
        if isinstance(review,dict):
            spans=[{k:d.get(k,'') for k in ('source_quote','candidate_quote')}
                   for d in review.get('defects',[]) if isinstance(d,dict)
                   and isinstance(d.get('source_quote'),str) and d['source_quote'] in source
                   and isinstance(d.get('candidate_quote'),str) and d['candidate_quote'] in current]
            feedback['source_semantic_feedback']={'disputed_spans':spans,
                'instruction':'These are disputed comparison scopes, not approved source meanings. '
                              'Translate the complete original source independently; do not follow a reviewer interpretation '
                              'that changes its negation, referent, domain, role, number, time or purpose.'}
        else:
            # Display partition feedback governs presentation, not new facts.
            feedback['source_semantic_feedback']={'scope':'bilingual_display' if 'bilingual' in str(review).lower() else 'source_review',
                'instruction':'Keep full original meaning and source clause order. Derive every fact from the source.'}
    from core.translation_invariants import source_fact_guidance
    factual_guidance = source_fact_guidance(source_fact_conflicts(source, current, target))
    if factual_guidance:
        feedback['factual_repair_instruction'] = factual_guidance
    names = TranslationNameGuard([source], target)
    prompt = names.prompt(spoken_translation_prompt(names.lines, context, target,
        [{**budget,"max_natural_speech_seconds":max_natural_seconds,"max_speed_factor":max_speed}], feedback=feedback),
        spoken_budget_units=budget.get('search_max_chinese_units'))
    retry_history = []
    no_progress_validation = [False]
    def native_retry(_original, rejected_draft, error):
        retry_history.append({"draft":rejected_draft,"validation_error":error})
        return names.prompt(spoken_translation_prompt(names.lines, context, target,
            [{**budget,"max_natural_speech_seconds":max_natural_seconds,"max_speed_factor":max_speed}],
            feedback={**feedback,"validation_failures":retry_history}),
            spoken_budget_units=budget.get('search_max_chinese_units'))

    def normalized(text):
        text, _ = clean_dialogue_rendering(source, text, target)
        if target.startswith("en"):
            text, _ = normalize_question_emphasis(source, text)
        return text

    def validate(data):
        status = validate_translation_rows(data, [source], None, target)
        if status["status"] != "success":
            return status
        text = normalized(data["1"])
        if delivery_feedback and re.sub(r"[\W_]+", "", text.casefold()) == re.sub(r"[\W_]+", "", normalized(current).casefold()):
            return {'status':'error','message':'This wording repeatedly failed complete speech readback. '
                    'Return a different faithful source translation; punctuation-only changes cannot repair delivery.'}
        if budget.get('measured_spoken_units'):
            from core.spoken_length import chinese_spoken_units
            from core.tts_pronunciation import spoken_plan, pronunciation_terms
            cost = chinese_spoken_units(spoken_plan(text, target, pronunciation_terms())['spoken_text'])
            material_overrun = measured_seconds and measured_seconds > max_natural_seconds * 1.10
            if cost > budget['measured_spoken_units'] or (material_overrun and cost >= budget['measured_spoken_units']):
                no_progress_validation[0] = True
                return {'status':'error','message':'The measured Chinese draft already exceeded its duration. '
                        'This candidate adds spoken syllables or fails to reduce a material overrun; shorten redundant sentence framing while retaining '
                        'every source fact and enumeration item. Do not expand the already overlong delivery.'}
        semantic_feedback = budget.get('source_semantic_feedback')
        display_only_feedback = 'bilingual' in str(semantic_feedback).lower()
        if (semantic_feedback and not display_only_feedback and
                re.sub(r"[\W_]+", "", text.casefold()) == re.sub(r"[\W_]+", "", normalized(current).casefold())
                and len(retry_history) < 1):
            return {"status": "error", "message": "This draft failed source meaning review. Return a corrected source translation rather than the same draft or punctuation-only changes."}
        fact_conflicts = source_fact_conflicts(source, text, target)
        if target.startswith("en"):
            from core.spoken_length import source_form_conflicts
            fact_conflicts += source_form_conflicts(source, text, budget["target_speech_seconds"])
        if fact_conflicts:
            from core.translation_invariants import source_fact_guidance
            return {"status": "error", "message": "Duration translation changed explicit source facts: " + ", ".join(fact_conflicts)
                    + ' ' + source_fact_guidance(fact_conflicts)}
        if (target.startswith("en") and re.search(r"\b(?:ya|yer)\b", text)
                and not re.search(r"\b(?:ya|yer)\b", source, re.I)):
            return {"status": "error", "message": "Use standard spoken pronouns you/your rather than eye-dialect spellings ya/yer; spelling shortcuts do not shorten spoken syllables and can cause TTS misreading."}
        from core.translation_invariants import number_annotations
        if target.startswith("en"):
            from decimal import Decimal
            anchor = budget.get("meaning_anchor",current)
            expected = [Decimal(n["value"]) for n in number_annotations(anchor)]
            actual = [Decimal(n["value"]) for n in number_annotations(text)]
            if expected != actual:
                return {"status":"error","message":"Preserve the translated source's numeric values and occurrence counts"}
        if target.startswith("en") and TARGET_PROFANITY.search(text) and not SOURCE_PROFANITY.search(source):
            return {"status": "error", "message": "Do not add profanity absent from this source unit"}
        if target.startswith("en") and len(TARGET_PROFANITY.findall(text)) < len(SOURCE_PROFANITY.findall(source)):
            return {"status":"error", "message":"Preserve the source's spoken profanity and repeated emphasis"}
        spoken_key = lambda value: re.sub(r"[\W_]+", "", value.casefold(), flags=re.UNICODE)
        if any(spoken_key(text) == spoken_key(normalized(old)) for old in rejected):
            no_progress_validation[0] = True
            return {"status": "error", "message": (
                "This wording has already exceeded the measured duration budget. Translate the complete source "
                "using shorter spoken phrasing; do not recycle rejected drafts or change punctuation alone.")}
        if (measured_seconds is not None and measured_seconds > max_natural_seconds + 1e-6
                and text.casefold().replace("’", "'") == normalized(current).casefold().replace("’", "'")):
            no_progress_validation[0] = True
            return {"status": "error", "message": (
                f"This unchanged draft already took {measured_seconds:.3f}s; the complete take may use at most "
                f"{max_natural_seconds:.3f}s. Return a genuinely more compact complete translation. "
                "Word count is only guidance; retain source facts, use shorter phrasing and omit added explanations.")}
        return status

    def guarded_validate(data):
        no_progress_validation[0] = False
        return names.validate(data, validate)

    def ordered_candidate(*, reset_feedback=False):
        from core.ask_gpt import IncompleteLLMResponseError
        from core.bilingual_alignment import translate_source_partition, prepare_candidate_alignment
        ordered_budget = dict(budget)
        if reset_feedback:
            # A reviewer can issue contradictory repair advice. After the
            # bounded feedback search, start a source-only candidate instead of
            # repeating that advice. All factual, display and speech gates stay
            # unchanged; this clears instructions, never a rejected verdict.
            ordered_budget.pop('source_semantic_feedback', None)
            ordered_budget.pop('meaning_anchor', None)
            # A whole clause's internal word order differs across languages.
            # First recover one complete source translation without the failed
            # draft/reviewer instructions. Mandatory semantics and display
            # checks decide whether it can be paired; only an unpairable or
            # rejected whole clause needs ordered source-span recovery.
            if ((budget.get('source_semantic_feedback') and measured_seconds is None)
                    or (measured_seconds is not None and
                        (budget.get('duration_witness') or {}).get('content_complete') is True)):
                from core.local_translation import source_only_translation_prompt
                source_only_prompt = names.prompt(source_only_translation_prompt(
                    names.lines[0], context, target, ordered_budget), response_json=False,
                    spoken_budget_units=budget.get('search_max_chinese_units'))
                try:
                    text = ask_gpt(source_only_prompt, response_json=False,
                        valid_def=lambda text: guarded_validate({'1': text}),
                        max_retries=2, retry_interval=1, log_title='translate_source_only_recovery')
                    whole = names.restore({'1': text})
                    status = validate(whole)
                    if status['status'] != 'success':
                        raise ValueError(status['message'])
                    candidate = normalized(whole['1']).replace('\n', ' ').strip()
                    verdict = audit_duration_candidate(source, current, candidate, context)
                    if verdict['equivalent']:
                        prepare_candidate_alignment({'unit_id': budget.get('unit_id', budget.get('line_id', 'candidate')),
                                                     'source': source, 'target': candidate}, context)
                        return candidate
                except IncompleteLLMResponseError:
                    raise
                except ValueError:
                    # A failed proposal is never accepted. Fall through to the
                    # existing bounded recovery with every guard intact.
                    pass
        for attempt in range(1 + int(load_key('translation.speech_budget.semantic_edit_retries', 2))):
            candidate = None
            try:
                candidate = normalized(translate_source_partition(source, target, context, ordered_budget))
                status = validate({'1': candidate})
                if status['status'] != 'success':
                    raise ValueError(status['message'])
                verdict = audit_duration_candidate(source, current, candidate, context)
                if not verdict['equivalent']:
                    raise ValueError('Ordered candidate changed source meaning: ' + json.dumps(verdict, ensure_ascii=False))
                # Revalidate the exact partition receipt. Never skip the display
                # gate because an ordered translation was proposed successfully.
                prepare_candidate_alignment({'unit_id': budget.get('unit_id', budget.get('line_id', 'candidate')),
                                             'source': source, 'target': candidate}, context)
                return candidate
            except IncompleteLLMResponseError:
                # A missing review cannot justify rewriting the translation.
                raise
            except ValueError as exc:
                error = str(exc)
                ordered_budget['source_semantic_feedback'] = {'previous_candidate': candidate,
                                                              'rejected_because': error}
        raise ValueError('Ordered translation recovery exhausted: ' + error)

    def aligned_candidate(candidate):
        from core.bilingual_alignment import prepare_candidate_alignment
        try:
            prepare_candidate_alignment({'unit_id': budget.get('unit_id', budget.get('line_id', 'candidate')),
                                         'source': source, 'target': candidate}, context)
        except ValueError:
            # Model-assisted alignment belongs to acceptance, not the JSON
            # validator. Otherwise an unalignable rewrite is retried without
            # ever entering the ordered-source recovery used by primary drafts.
            return ordered_candidate()
        return candidate

    display_feedback = str(budget.get('source_semantic_feedback', ''))
    if 'bilingual' in display_feedback.lower():
        return ordered_candidate()

    if (target.startswith('zh') and (delivery_feedback or
            (measured_seconds and measured_seconds > max_natural_seconds
             and (budget.get('duration_witness') or {}).get('content_complete') is True))):
        from core.bilingual_alignment import prepare_candidate_alignment
        for candidate in compact_chinese_framing(source, current):
            if validate({'1': candidate})['status'] != 'success':
                continue
            verdict = audit_duration_candidate(source, current, candidate, context)
            if verdict.get('equivalent') is not True:
                continue
            try:
                prepare_candidate_alignment({'unit_id': budget.get('unit_id', budget.get('line_id', 'candidate')),
                                             'source': source, 'target': candidate}, context)
            except ValueError:
                continue
            return candidate

    candidate_provider = load_key("translation.speech_budget.candidate_provider", "")
    primary_tries = int(load_key("translation.speech_budget.candidate_retries", 3))
    provider = None
    if candidate_provider:
        from core.llm_provider import get_llm_provider_config
        provider = get_llm_provider_config(str(candidate_provider))
    from core.homura_translation import is_homura, propose_homura_translation, NoSafeHomuraCandidate
    if is_homura(provider):
        # Homura is a source translator with a plain-text integer interface.
        # A direct/legacy caller without a complete take cannot calibrate it.
        if target.startswith(("en", "zh")) and (budget.get("duration_witness") or {}).get("content_complete") is True:
            try:
                from core.duration_candidate_audit import source_category_terms
                categories=budget.get("source_category_terms")
                if categories is None and load_key("translation.speech_budget.source_category_guidance", False):
                    categories=source_category_terms(source,context)
                calibrated_context=with_source_category_terms(context,categories or [])
                candidate = propose_homura_translation(source, current, budget, calibrated_context, provider,
                    validate_text=lambda text: validate({"1": text}),
                    normalize_text=lambda text: normalized(text).replace("\n", " ").strip(),
                    audit_candidate=audit_duration_candidate, target_language=target)
                return aligned_candidate(candidate)
            except NoSafeHomuraCandidate:
                # Keep the stronger primary translator as a bounded source-
                # translation fallback. Never send Homura the generic JSON edit.
                pass
        provider = None
    editor_first = provider is not None and provider.provider_type=="mlx_local"
    semantic_gate = bool(load_key("translation.speech_budget.semantic_candidate_gate",False))
    rounds = int(load_key("translation.speech_budget.semantic_edit_retries",2)) if semantic_gate else 1
    for _ in range(rounds):
        primary_error = None
        used_provider = provider if editor_first else None
        try:
            result = ask_gpt(prompt,response_json=True,valid_def=guarded_validate,max_retries=primary_tries,retry_interval=1,
                            log_title="translate_duration_feedback",validation_retry_prompt=native_retry,
                            **({"provider_override":provider} if editor_first else {}))
        except Exception as exc:
            primary_error=str(exc)
            if ('Validation still failed' in primary_error
                    and 'Duration translation changed explicit source facts' in primary_error):
                # A repeated deterministic fact conflict is not an operational
                # outage. Recover once from the source, under the same checks,
                # instead of recycling a factually rejected feedback draft.
                return ordered_candidate(reset_feedback=True)
            if (no_progress_validation[0] and 'Validation still failed' in primary_error
                    and ('already exceeded the measured duration budget' in primary_error
                         or 'This unchanged draft already took' in primary_error
                         or 'This candidate adds spoken syllables' in primary_error)):
                # The feedback-conditioned translator is anchored to a known
                # overlong script. Use the existing bounded source-only route,
                # keeping banned drafts, semantics, names and display checks.
                # Operational failures and other validation defects never
                # trigger this recovery.
                return ordered_candidate(reset_feedback=True)
            if not provider or editor_first:
                raise
            used_provider=provider
            result=ask_gpt(prompt,response_json=True,valid_def=guarded_validate,max_retries=primary_tries,retry_interval=1,
                          log_title="translate_duration_alternatives",validation_retry_prompt=native_retry,
                          provider_override=provider)
        result = names.restore(result)
        checked=validate(result)
        if checked["status"]!="success":
            raise ValueError(checked["message"])
        final=normalized(result["1"]).replace("\n"," ").strip()
        if used_provider:
            from pathlib import Path
            path=Path("output/log/translation_duration_candidate_provenance.json")
            records=json.loads(path.read_text()) if path.is_file() else {"stage":"before_translation_acceptance","candidates":[]}
            records["candidates"].append({"source":source,"target":final,"provider":used_provider.name,
                "model":used_provider.model,"base_url":used_provider.base_url,"primary_error":primary_error,
                "acceptance":"requires_measured_speech_and_semantic_audit"})
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding="utf-8")
        # The semantic model is invoked outside ask_gpt validation callbacks,
        # avoiding nested cache-lock requests and keeping deterministic guards.
        verdict=audit_duration_candidate(source,budget.get("meaning_anchor",current),final,context) if semantic_gate else {"equivalent":True}
        if verdict["equivalent"]:
            break
        prompt=native_retry(prompt,result,"Semantic source comparison rejected this candidate: "+json.dumps(verdict,ensure_ascii=False))
    else:
        return ordered_candidate(reset_feedback=True)
    if final != result["1"]:
        from core.local_translation import record_fact_normalizations
        record_fact_normalizations([{"source": source, "before": result["1"], "after": final,
                                    "reasons": ["duration_candidate_presentation_and_source_emphasis_guard"]}])
    return aligned_candidate(final)


def guard_translation_context(frame, rows, terminology):
    """Splitting for synthesis cannot import neighbouring profanity or a payer."""
    from core.local_translation import localized_translation_prompt, validate_translation_rows, record_fact_normalizations
    from core.translation_context import speaker_profiles_context
    from core.translation_invariants import possible_reported_payment, normalize_explicit_source_facts, source_fact_conflicts
    target = effective_target_language("en")
    if not target.startswith("en"):
        return frame
    frame = frame.copy()
    changes = []
    curse_source, curse_target = SOURCE_PROFANITY, TARGET_PROFANITY
    for idx, row in rows.iterrows():
        source, parent, text = str(row.text), str(row.parent_source), str(frame.loc[idx, "Translation"])
        guarded, reasons = normalize_question_emphasis(source, text)
        if guarded != text:
            changes.append({"source": source, "before": text, "after": guarded, "reasons": reasons})
            text = guarded
        if possible_reported_payment(parent):
            text, reasons = normalize_explicit_source_facts(parent, text, target)
            if text != frame.loc[idx, "Translation"]:
                changes.append({"source": source, "before": frame.loc[idx, "Translation"], "after": text,
                                "reasons": reasons, "basis": "complete_parent_source_omitted_payer"})
        if curse_target.search(text) and not curse_source.search(source):
            # A fact-only retry must not re-import the disputed words from
            # neighbouring quotations or inferred character style.
            prompt = localized_translation_prompt([source], "本句是强硬责问，但没有说脏话；请保留质问语气。", target)
            prompt += ("\n【硬性要求】本句没有脏话，严禁添加 hell/fuck/damn/shit 等粗口；"
                       "可用短促强硬的语气表达责问。只翻译当前源句，不从邻句借用任何文字。只输出键1的JSON。")
            def validate(value):
                status = validate_translation_rows(value, [source], None, target)
                if status["status"] == "success" and (curse_target.search(value["1"]) or source_fact_conflicts(source, value["1"], target)):
                    return {"status": "error", "message": "Preserve current unit facts without importing profanity"}
                return status
            result = ask_gpt(prompt, response_json=True, valid_def=validate, max_retries=2,
                             retry_interval=1, log_title="translate_unit_context_guard")
            if validate(result)["status"] != "success":
                raise ValueError("Unit-context guard could not preserve source meaning")
            changes.append({"source": source, "before": text, "after": result["1"],
                            "reasons": ["neighbouring_profanity_not_in_current_source"]})
            text = result["1"].replace("\n", " ").strip()
        frame.at[idx, "Translation"] = text
    record_fact_normalizations(changes)
    return frame
