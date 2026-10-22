"""Source translations with Homura's native syllable interface, before TTS."""
from __future__ import annotations

import json
import math
from pathlib import Path

from core.ask_gpt import ask_gpt
from core.config_utils import load_key
from core.spoken_length import english_syllables


class NoSafeHomuraCandidate(ValueError):
    pass


def is_homura(provider) -> bool:
    return provider is not None and "index-homura" in provider.model.lower()


def _spoken_units(text, target_language):
    if str(target_language).startswith('zh'):
        from core.spoken_length import chinese_spoken_units
        from core.tts_pronunciation import spoken_plan, pronunciation_terms
        return chinese_spoken_units(spoken_plan(text, target_language, pronunciation_terms())['spoken_text'])
    if str(target_language).startswith('en'):
        return english_syllables(text)
    raise ValueError('Homura duration calibration is unavailable for this target language')


def syllable_targets(current: str, budget: dict, *, target_language='en') -> list[int]:
    witness = budget.get("duration_witness") or {}
    measured = witness.get("natural_seconds")
    if (witness.get("content_complete") is not True or witness.get("text") != current
            or not witness.get("natural_audio_sha256") or not witness.get("conditioning_signature")
            or not witness.get("readback_signature")
            or not isinstance(measured, (int, float)) or not math.isfinite(measured) or measured <= 0):
        raise ValueError("Homura requires a complete, bound natural-speech duration witness")
    cap = float(budget["max_natural_speech_seconds"])
    if not math.isfinite(cap) or cap <= 0 or measured <= cap + 1e-6:
        raise ValueError("Homura duration search requires a measured complete overrun")
    count = _spoken_units(current, target_language)
    if count <= 0:
        raise ValueError("No supported pronunciation estimate for Homura")
    # Search the longest plausible wording first. Neither this estimate nor
    # Homura's requested integer certifies natural duration or semantic quality.
    upper = max(1, min(count - 1, math.floor(count * cap / measured)))
    tries = max(1, min(3, int(load_key("translation.speech_budget.homura.max_candidates", 3))))
    return [n for n in range(upper, max(0, upper - tries), -1)]


def build_homura_prompt(source: str, syllables: int, context: str, *, source_constraints=(), validation_feedback=(), target_language='en') -> str:
    if isinstance(syllables, bool) or not isinstance(syllables, int) or syllables <= 0:
        raise ValueError("Homura target syllables must be a positive integer")
    try:
        meanings = json.loads(context)
    except (TypeError, ValueError):
        meanings = {}
    if not isinstance(meanings, dict):
        meanings = {}
    pairs = []
    mapped = set()
    for term in [*meanings.get("glossary_meanings", []), *meanings.get("source_category_guidance", []), *source_constraints]:
        src, tgt = term.get("src"), term.get("tgt")
        if isinstance(src, str) and src and src in source and src not in mapped and isinstance(tgt, str) and tgt.strip():
            pairs.append(f"{src} -> {tgt}")
            mapped.add(src)
    from core.translation_language import translation_language_name
    native = f"请将以下文本翻译为{translation_language_name(target_language)}，译文严格控制在 {syllables} 个音节。"
    if pairs:
        native += f"要求：遵守术语含义【{', '.join(pairs)}】。专名译名保持一致，普通长词组可换成更短的等义表达。"
        native += "术语映射规定的是语境含义，不能为缩短音节退回字面词义或泛化类别。优先压缩句框、连接词和冗余表达，不能删掉独立断言。"
    native += "保留原句的动作方向、明确的地点、人称和所有者、否定、数字与单位、脏话和刻意重复。不要根据背景补出原句省略的主体。"
    if str(target_language).startswith('zh'):
        native += "中文列举用短促的自然名词短语，未尽列举的口头填充按完整惯用语表达，可用‘等’；不逐个展开无独立事实意义的填充词。每个实质列举项、独立断言和有意强调的重复仍完整保留。"
    from core.spoken_length import source_form_hint, contextual_compaction_hint
    form = source_form_hint(source) if str(target_language).startswith('en') else ''
    if form:
        native += "口语结构要求：" + form
    compaction = contextual_compaction_hint(source, context) if str(target_language).startswith('en') else ''
    if compaction:
        native += "上下文短答要求：" + compaction
    from core.translation_invariants import possible_reported_payment
    if possible_reported_payment(source):
        native += "当前原句没有明示说话人是付款人；不能添加 I/we give/pay。给你可表达为 you get/receive，保留第三人称所有者。"
    if validation_feedback:
        native += "上次候选未通过原句核对：" + "；".join(validation_feedback[-3:]) + "。请从原句重新翻译并修正上述问题。"
    native += f"只翻译当前原句，不借用邻句的事实或脏话。直接输出翻译结果，不要进行任何解释。\n\n{source}"
    if context:
        native = ("理解参考（仅辅助理解，不翻译这些背景；人物推断不能覆盖原句事实）：\n"
                  + context + "\n\n" + native)
    return native


def _record(record: dict) -> None:
    path = Path("output/log/translation_duration_candidate_provenance.json")
    report = json.loads(path.read_text()) if path.is_file() else {
        "stage": "before_translation_acceptance", "candidates": []}
    report["candidates"].append(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def record_homura_speech_result(unit_id: str, text: str, take: dict) -> None:
    """Link a safe model draft to its actual speech gates, including retakes."""
    path = Path("output/log/translation_duration_candidate_provenance.json")
    if not path.is_file():
        return
    report = json.loads(path.read_text(encoding="utf-8"))
    for record in reversed(report["candidates"]):
        if (record.get("strategy") == "homura_native_source_syllables"
                and record.get("unit_id") == unit_id and record.get("target") == text
                and (record.get("semantic_verdict") or {}).get("equivalent") is True):
            record.setdefault("measured_speech", []).append(dict(take))
            record["status"] = ("speech_pass_before_final_review" if
                                take.get("content_complete") is True and take.get("fits") is True
                                else "speech_rejected")
            path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            return


def propose_homura_translation(source: str, current: str, budget: dict, context: str,
                               provider, *, validate_text, audit_candidate, normalize_text=lambda text: text,
                               target_language='en') -> str:
    """Try finite source-based candidates; never mutate or accept downstream audio."""
    failures = []
    meaning_constraints = {}
    validation_feedback = []
    fidelity_retry_target = None
    from core.local_translation import TranslationNameGuard
    from core.tts_pronunciation import pronunciation_terms
    names = TranslationNameGuard([source], target_language, pronunciation_terms())
    for target in syllable_targets(current, budget, target_language=target_language):
        # A meaning failure needs clearer source constraints at the same length,
        # not progressively tighter pressure to drop the disputed meaning.
        target = fidelity_retry_target or target
        record = {"unit_id": budget.get("unit_id"), "source": source, "provider": provider.name,
                  "model": provider.model, "base_url": provider.base_url,
                  "model_revision": provider.model_revision, "strategy": "homura_native_source_syllables",
                  "target_syllables": target, "duration_witness": budget["duration_witness"],
                  "target_language": target_language,
                  "source_constraint_feedback": list(meaning_constraints.values()),
                  "validation_feedback": list(validation_feedback),
                  "acceptance": "requires_measured_speech_and_semantic_audit"}
        try:
            def validate(value):
                if isinstance(value, str):
                    record["raw_target"] = value
                if (not isinstance(value, str) or not value.strip() or value.lstrip().startswith(("{", "[", "```"))
                        or "<think>" in value):
                    return {"status": "error", "message": "Homura must return only the complete translated text"}
                try:
                    canonical = names.restore({'1': value})['1']
                except ValueError as exc:
                    checked = {'status': 'error', 'message': str(exc)}
                else:
                    checked = validate_text(canonical)
                if checked["status"] != "success":
                    record["validation_error"] = checked["message"]
                return checked
            prompt = build_homura_prompt(names.lines[0], target, context,
                source_constraints=meaning_constraints.values(), validation_feedback=validation_feedback,
                target_language=target_language)
            prompt = names.prompt(prompt, response_json=False,
                                  spoken_budget_units=target if str(target_language).startswith('zh') else None)
            text = ask_gpt(prompt, response_json=False,
                           valid_def=validate, provider_override=provider, max_retries=1,
                           retry_interval=1, log_title="translate_homura_candidates").strip()
            checked = validate(text)
            if checked["status"] != "success":
                raise ValueError(checked["message"])
            record["raw_target"] = text
            text = names.restore({'1': text})['1']
            text = normalize_text(text)
            # This comparison is outside ask_gpt's cache-lock callback.
            verdict = audit_candidate(source, budget.get("meaning_anchor", current), text, context)
            record.update(target=text, estimated_syllables=_spoken_units(text, target_language),
                          semantic_verdict=verdict)
            if not verdict["equivalent"]:
                validation_feedback.append("源文语义核对：" + "；".join(verdict.get("issues", [])))
                for constraint in verdict.get("source_constraints", []):
                    if (isinstance(constraint.get("src"), str) and constraint["src"] in source
                            and isinstance(constraint.get("tgt"), str) and constraint["tgt"].strip()):
                        meaning_constraints[constraint["src"]] = constraint
                if meaning_constraints:
                    fidelity_retry_target = target
                raise ValueError("Source semantic comparison rejected Homura: " + json.dumps(verdict, ensure_ascii=False))
        except Exception as exc:
            error = record.get("validation_error")
            if error:
                validation_feedback.append(error)
                if any(key in error for key in ("source facts", "numeric values", "profanity", "spoken profanity")):
                    fidelity_retry_target = target
            record.update(status="rejected", error=str(exc))
            _record(record)
            failures.append(str(exc))
            continue
        record["status"] = "awaiting_measured_speech"
        _record(record)
        return text
    raise NoSafeHomuraCandidate("No safe Homura candidate within the finite search: " + "; ".join(failures))
