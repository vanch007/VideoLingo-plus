"""Small, deterministic checks on proposed translation repairs."""
from __future__ import annotations

import re
from decimal import Decimal


_SMALL = dict(zip(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split(),
    range(20),
))
_SMALL.update(dict(zip("twenty thirty forty fifty sixty seventy eighty ninety".split(), range(20, 100, 10))))
_SCALES = {"hundred": 100, "thousand": 1000, "million": 1000000, "billion": 1000000000}


def english_number_value(phrase: str) -> Decimal | None:
    """Parse one cardinal phrase, including decimal magnitudes, without an LLM."""
    tokens = phrase.lower().replace(",", "").replace("-", " ").split()
    if "point" in tokens:
        at = tokens.index("point")
        scale = _SCALES.get(tokens[-1], 1)
        fractional = tokens[at + 1:-1] if scale != 1 else tokens[at + 1:]
        whole = english_number_value(" ".join(tokens[:at]))
        if whole is None or not fractional or any(t not in _SMALL or _SMALL[t] >= 10 for t in fractional):
            return None
        digits = "".join(str(_SMALL[t]) for t in fractional)
        return (whole + Decimal("0." + digits)) * scale
    total, group = Decimal(0), Decimal(0)
    seen = False
    previous = ""
    for token in tokens:
        if token == "and":
            continue
        if token in _SMALL:
            if previous in _SMALL and not (_SMALL[previous] >= 20 and _SMALL[token] < 10):
                return None
            group += _SMALL[token]
        elif token == "hundred":
            group = (group or 1) * 100
        elif token in _SCALES:
            total += (group or 1) * _SCALES[token]
            group = Decimal(0)
        elif re.fullmatch(r"\d+(?:\.\d+)?", token):
            group += Decimal(token)
        else:
            return None
        seen = True
        previous = token
    return total + group if seen else None


def number_annotations(text: str) -> list[dict]:
    atom = "(?:" + "|".join(_SMALL.keys() | _SCALES.keys()) + r"|\d[\d,]*(?:\.\d+)?)"
    phrases = re.finditer(r"\b" + atom + r"(?:[ -]+(?:(?:and|point)[ -]+)?" + atom + r")*\b", text, re.I)
    annotations = []
    for match in phrases:
        phrase = match.group()
        if phrase.lower() == "one":
            before, after = text[:match.start()], text[match.end():]
            # Indefinite/relative person pronouns do not state a quantity.
            # Retain actual cardinals such as "one witness" or "one million".
            relative_person = (re.search(r"\bthe\s+$", before, re.I) and
                               re.match(r"\s+(?:who|that|which|I|you|he|she|we|they)\b", after, re.I))
            indefinite_person = re.search(r"\b(?:no|any|every)\s+$", before, re.I)
            if relative_person or indefinite_person:
                continue
        if (value := english_number_value(phrase)) is not None:
            annotations.append({"text": phrase, "value": str(value)})
    return annotations


def payment_amount_keys(text: str) -> set[str]:
    keys = set(re.findall(r"\d+(?:\.\d+)?[万亿]|\d{3,}|[一二两三四五六七八九]+[百千万亿][零〇一二两三四五六七八九十百千万亿]*", text))
    # A final 两 after a scale is the tael unit, not another digit in the amount.
    return {key[:-1] if len(key) > 1 and key.endswith("两") and key[-2] in "十百千万亿" else key for key in keys}


def possible_reported_payment(source: str) -> bool:
    return bool(payment_amount_keys(source) and re.search(r"给(?:你|您|他|她)", source)
                and re.search(r"(?:他|她|他们|她们)的", source) and "我" not in source)


def source_has_tael_unit(source: str) -> bool:
    # 两 in 两千 is a digit; 两 after an amount scale/digit is an explicit
    # unit, even when 银子 is omitted (e.g. 一百两 / 180万两).
    return bool(re.search(r"两银|兩銀|银两|銀兩|[十百千万亿\d][两兩](?![十百千万亿])|\btaels?\b", source, re.I))


def negated_need_conflicts(source: str, translated: str) -> list[str]:
    """An explicit negated need cannot disappear into a neighboring assertion.

    Common compact English replies are allowed. Presence of one is only a
    necessary condition; the source semantic gate still checks its action.
    """
    phrases = [p.strip() for p in re.split(r"[。！？!?；;，,]", source)]
    if not any(re.fullmatch(r"(?:不用|不必|无需|无须)[\u4e00-\u9fff]{1,8}", p) for p in phrases):
        return []
    if re.search(r"\b(?:no|not|never|unnecessary|needless|optional|skip|forget|spare|without)\b|n['’]t\b", translated, re.I):
        return []
    return ["explicit_negated_need_lost"]


def source_fact_guidance(conflicts) -> str:
    """Explain machine findings without prescribing a run-specific script."""
    hints = []
    for conflict in conflicts:
        if str(conflict).startswith('explicit_departure_location_lost:'):
            kind = str(conflict).split(':')[-1]
            location = {'settlement': '城镇（如 leave town / leave the city）',
                        'village': '村庄（如 leave the village）',
                        'border': '国境（如 go abroad / cross the border）'}.get(kind, '地点')
            hints.append(f'原句明确点明离开的{location}，译文必须保留这个地点类别和离开动作，不能只说 go out。')
        elif conflict == 'explicit_negated_need_lost':
            hints.append('原句明确说不用或不必，必须保留无需进行该动作的否定意思；可用 No need 等上下文明确的短答，但不能整段省掉否定。')
        elif str(conflict).startswith('explicit_probability_lost:'):
            hints.append('源文的 probably/possibly/likely 表示不确定性，不是肯定断言。'
                         '完整重译时保留同一命题的可能性，可自然使用可能、大概、多半等表达；'
                         '不能改成肯定、一定、必然，也不能删除原文的否定或其他事实。')
        elif conflict == 'explicit_frequency_strengthened':
            hints.append('源文usually/normally/typically只表示通常或一般情况，不能加强为总是、始终或每次都；'
                         '保留原来的频率以及原句其余事实。')
        elif conflict == 'explicit_frequency_lost':
            hints.append('源文明确给出通常或一般情况下的频率；重译必须保留这一限定，不能为缩短而省略，'
                         '也不能变成总是。用紧凑表达保留其余动作与有意重复。')
        elif str(conflict).startswith('explicit_possession:'):
            hints.append('源文明确给出my/our的所属关系；完整译文须保留我/我们的所有者，不能改为不明所属对象。')
        elif conflict == 'independent_teaching_goals_collapsed':
            hints.append('源文逐项教多个独立技能或目标，译文也须保留各项并列教学目标；'
                         '不能把学习使用工具和学习靠工具赚钱压成仅学习用工具赚钱。')
    return ' '.join(dict.fromkeys(hints))


def source_vocative(source: str):
    """Bind a comma-delimited friendly address, not third-person ownership."""
    match = re.match(r'^\s*(?:all\s*right|alright|okay|ok|hey|hello|hi|well)[,\s]+'
                     r'(my\s+(?:dear\s+)?friends?)\s*,', source, re.I)
    return match.group(1) if match else None


def source_fact_conflicts(source: str, translated: str, target_language: str) -> list[str]:
    """Protect explicit possession and keep an omitted payer unspecified.

    These checks do not infer an identity or correct uncertain ASR text.
    """
    if target_language.lower().startswith('zh'):
        # A subjective probability is not a categorical statement. Chinese
        # has several natural ways to retain it; the whole semantic gate still
        # checks which proposition the qualifier modifies.
        uncertain = re.search(r'\b(?:I|you|he|she|it|we|they|there)\s+'
                              r'(?:probably|possibly|likely)\b', source, re.I)
        conflicts = []
        for word, chinese in [('my',r'我|本人|咱'),('our',r'我们|咱们|我方|本(?:团队|公司|校|部门|项目)')]:
            # Interjections are not possession. Other explicit personal
            # possessives require at least a corresponding target possessor;
            # presence is necessary, not proof that actor/owner relations fit.
            possessive_source = re.sub(r'\bmy\s+(?:god|gosh|goodness|dear|bad)\b','',source,flags=re.I)
            vocative = source_vocative(source)
            if vocative:
                possessive_source = possessive_source.replace(vocative, '', 1)
            if re.search(r'\b'+word+r'\s+[a-z]',possessive_source,re.I) and not re.search(chinese,translated):
                conflicts.append('explicit_possession:'+word)
        if uncertain and not re.search(r'可能|大概|也许|或许|说不定|恐怕|大抵|大约|八成|没准|'
                                      r'想必|多半|估计|应该|似乎|看起来|看样子', translated):
            conflicts.append('explicit_probability_lost:' + uncertain.group())
        if (re.search(r'\b(?:usually|normally|generally|typically)\b',source,re.I)
                and not re.search(r'\balways\b',source,re.I)):
            if re.search(r'总是|始终|每次都|无一例外',translated):
                conflicts.append('explicit_frequency_strengthened')
            elif not re.search(r'通常|一般|往往|经常|常常|常会|惯常|照例|大多|多半|普遍|平时|基本上',translated):
                conflicts.append('explicit_frequency_lost')
        teaching = re.search(r'\b(?:teach(?:es|ing)?|taught)\b',source,re.I)
        parallel_goals = (len(re.findall(r'\bhow\s+to\b',source,re.I))>1
                          or re.search(r'\bto\s+\w.+\band\s+(?:to|how\s+to)\s+\w',source,re.I))
        if teaching and parallel_goals and not re.search(r'[，,、；;]|以及|并|和|与|及|还|又',translated):
            # A single instrumental predicate cannot stand for two explicit
            # independent teaching goals. Coordination is only necessary;
            # the source semantic gate still verifies each actual goal.
            conflicts.append('independent_teaching_goals_collapsed')
        return conflicts
    if not target_language.lower().startswith("en"):
        return []
    conflicts = repair_fact_conflicts(source, "", translated, target_language)
    conflicts += negated_need_conflicts(source, translated)
    for original, english in (("我的", r"\b(?:my|mine)\b"), ("我们的", r"\b(?:our|ours)\b")):
        # "my beloved" can naturally become "the one I love". This is an
        # explicit first-person relationship, not an inferred object owner.
        relationship = (original == "我的" and re.search(r"我的[至挚]爱|我所爱", source)
                        and re.search(r"\b(?:one|person|man|woman)\s+(?:(?:who|that)\s+)?I\s+(?:love|cherish)\b", translated, re.I))
        if original in source and not re.search(english, translated, re.I) and not relationship:
            conflicts.append(f"explicit_possession:{original}")
    if possible_reported_payment(source) and re.search(r"\b(?:I|me|my|we|us|our)\b", translated, re.I):
        conflicts.append("unconfirmed_first_person_payer")
    # Check the subject of a clearly stated payment predicate, not every
    # pronoun in the dialogue. "You pay; I'll fight" has two different agents.
    payer_subjects = re.findall(
        r"(我们|咱们|你们|我|你|您)(?:来|会|要|愿意|愿|肯|负责|一定|就|亲自|不|再){0,3}"
        r"(?:出钱|付钱|掏钱|付款|支付|(?:出|付)[零〇一二两三四五六七八九十百千万亿\d]*[百千万亿])", source)
    payer_persons = {"first" if subject in {"我","我们","咱们"} else "second" for subject in payer_subjects}
    payment_predicate = r"\s+(?:pay(?:ing)?|fund(?:ing)?|financ(?:e|ing)|put\s+up\s+(?:the\s+)?money)\b"
    modal = r"(?:['’]ll|['’]m|['’]re|\s+(?:will|would|must|should|can|could|am|are))?"
    if payer_persons == {"second"} and re.search(r"\b(?:I|we)"+modal+payment_predicate, translated, re.I):
        conflicts.append("explicit_second_person_payer_reversed")
    if payer_persons == {"first"} and re.search(r"\byou"+modal+payment_predicate, translated, re.I):
        conflicts.append("explicit_first_person_payer_reversed")
    if re.search(r"出(?:城|镇|鎮|村|境)",source) and re.search(
            r"\b(?:take\s+over|capture|conquer|occupy|invade|seize|attack)\s+(?:the\s+)?(?:city|town|village|country)\b",translated,re.I):
        conflicts.append("departing_location_changed_to_conquering_location")
    # "Go out" alone can erase a source's explicit settlement/border. Keep
    # the named kind of location as well as the departure action; compact
    # phrases such as "leave town" remain fully supported.
    for pattern, english, location in (
        (r"出(?:城|镇|鎮)", r"\b(?:city|town|settlement)\b", "settlement"),
        (r"出村", r"\bvillage\b", "village"),
        (r"出境", r"\b(?:border|country|abroad)\b", "border"),
    ):
        if re.search(pattern, source) and not re.search(english, translated, re.I):
            conflicts.append(f"explicit_departure_location_lost:{location}")
    return conflicts


def normalize_explicit_source_facts(source: str, translated: str, target_language: str):
    """Apply narrow grammatical corrections; leave ambiguous edits to review."""
    if not target_language.lower().startswith("en"):
        return translated, []
    changed = []
    # The model sometimes imports the following line's silver unit into a
    # number whose unit is deliberately omitted. Remove only the numeric
    # tael construction; other unsupported currencies still fail validation.
    if payment_amount_keys(source) and not source_has_tael_unit(source):
        cardinal = "(?:" + "|".join(_SMALL.keys() | _SCALES.keys()) + r"|\d[\d,.]*)"
        pattern = r"(?P<number>\b" + cardinal + r")\s+(?:silver\s+)?taels?(?:\s+of\s+silver)?\b"
        translated, count = re.subn(pattern, r"\g<number>", translated, flags=re.I)
        if count:
            changed.append("unsupported_numeric_tael_unit_removed")
    # Only one source owner and one target possessive: there is no choice of
    # referent to guess. Multiple owners/possessives require fresh translation.
    if "我的" in source and not re.search(r"(?:你|您|他|她|他们|她们|我们)的", source):
        possessions = list(re.finditer(r"\b(?:my|your|his|their|our)\b", translated, re.I))
        if len(possessions) == 1 and possessions[0].group().lower() != "my":
            match = possessions[0]
            possessive = "My" if match.group()[0].isupper() else "my"
            translated = translated[:match.start()] + possessive + translated[match.end():]
            changed.append("explicit_first_person_possession")
    if possible_reported_payment(source):
        pattern = r"\b(?:I|we)(?P<modal>['’]ll|\s+will|\s+would)\s+(?P<verb>give|pay)\s+you\b"
        def neutral_payer(match):
            prefix = translated[:match.start()].rstrip()
            recipient = "You" if not prefix or prefix.endswith((".", "!", "?")) else "you"
            predicate = " receive" if match["verb"].lower() == "give" else " be paid"
            changed.append("omitted_payer_kept_unspecified")
            return recipient + match["modal"].lower() + predicate
        translated = re.sub(pattern, neutral_payer, translated, flags=re.I)
    return translated, changed


def repair_fact_conflicts(source: str, before: str, after: str, target_language: str) -> list[str]:
    """Block unsupported currency additions; unresolved semantics still need audit."""
    if not target_language.lower().startswith("en"):
        return []
    currencies = {
        "dollar": (r"\$|\bdollars?\b", r"美元|美金|dollars?|\$"),
        "euro": (r"€|\beuros?\b", r"欧元|歐元|euros?|€"),
        "pound": (r"£|\bpounds?\b", r"英镑|英鎊|磅|pounds?|£"),
        "yuan": (r"\byuan\b|\brenminbi\b", r"人民币|人民幣|元|yuan|renminbi"),
        "tael": (r"\btaels?\b", r"两银|兩銀|taels?"),
    }
    conflicts = []
    for name, (english, original) in currencies.items():
        if name == "tael" and source_has_tael_unit(source):
            continue
        if name == "pound" and "£" not in after:
            cardinal = "(?:" + "|".join(_SMALL.keys() | _SCALES.keys()) + r"|\d[\d,.]*)"
            if not re.search(r"\b" + cardinal + r"[ -]+(?:British[ -]+)?pounds?\b", after, re.I):
                continue  # the verb "pound" is not a currency unit
        if re.search(english, after, re.I) and not re.search(original, source, re.I):
            if not re.search(english, before, re.I):
                conflicts.append(f"unsupported_currency:{name}")
    return conflicts
