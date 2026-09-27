import os,sys,json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.config_utils import load_key
from core.runtime_context import effective_target_language

## ================================================================
# @ step4_splitbymeaning.py
def get_split_prompt(sentence, num_parts = 2, word_limit = 20, retry_count = 0):
    language = load_key("whisper.detected_language")

    retry_instruction = ""
    if retry_count > 0:
        retry_instruction = f"""
(Important: This is a retry attempt #{retry_count}. The previous attempts failed because the split was in the middle of phrases or resulted in single characters. Please pay special attention to creating meaningful splits.)"""

    split_prompt = f"""
## Role
You are a professional Netflix subtitle splitter in {language}.

## Task
Split the given subtitle text into {num_parts} parts, each less than {word_limit} words.

1. Maintain sentence meaning coherence according to Netflix subtitle standards
2. Keep parts roughly equal in length (minimum 3 words each)
3. Split at natural points like punctuation marks or conjunctions
4. If provided text is repeated words, simply split at the middle of the repeated words.
5. NEVER split in the middle of grammatical phrases or words
6. NEVER create segments with just a single character or punctuation mark
7. Each segment should be grammatically complete as much as possible{retry_instruction}

## Output in only JSON format
{{
    "analysis": "Brief analysis of the text structure",
    "split": "Complete sentence with [br] tags at split positions"
}}

## Given Text
<split_this_sentence>
{sentence}
</split_this_sentence>
""".strip()
    return split_prompt


## ================================================================
# @ step4_1_summarize.py
def get_summary_prompt(source_content, custom_terms_json=None, speaker_ids=None):
    """Step 1: Summarize topic and extract terminology (without correction)"""
    src_lang = load_key("whisper.detected_language")
    tgt_lang = load_key("target_language")

    # add custom terms note
    terms_note = ""
    if custom_terms_json:
        terms_list = []
        for term in custom_terms_json['terms']:
            terms_list.append(f"- {term['src']}: {term['tgt']} ({term['note']})")
        terms_note = "\n### Existing Terms\nPlease exclude these terms in your extraction:\n" + "\n".join(terms_list)

    summary_prompt = f"""
## Role
You are a video dialogue analyst, translation expert, and terminology consultant.
You specialize in {src_lang} comprehension and professional terminology extraction.

## Task
For the provided {src_lang} video text (transcribed by ASR, may contain errors):
1. Summarize main topic in two sentences
2. Extract professional terms/names with {tgt_lang} translations (excluding existing terms)
3. Provide brief explanation for each term{terms_note}
4. Infer a conservative role profile for every diarized speaker ID: {speaker_ids or []}

## Steps
1. Topic Summary:
   - Quick scan for general understanding
   - Write two sentences: first for main topic, second for key point

2. Term Extraction:
   - Mark proper nouns, character names, recurring dialogue motifs (thematic words repeated in the scene), and culturally specific action idioms.
   - In 'tgt', provide ONLY the clean, concise translation in {tgt_lang} (e.g. 'surprise', 'bandits', 'Master Huang'). NEVER put parenthetical notes or transliterations in 'tgt'. Put all background notes in 'note'.
   - Ensure core recurring motifs and names (e.g. key nouns repeated multiple times) are included with consistent translations.

3. Speaker/Character Analysis:
   - Treat labels such as [S01] as stable speaker identities, never as spoken text
   - Infer role, relationship, status/formality, and speaking style only from dialogue evidence
   - Use "unknown" when evidence is insufficient; never invent names, gender, or biography
   - Explain how each profile should affect pronouns, honorifics, tone, and address terms in translation

## Output in only JSON format
{{
    "topic": "Two-sentence video summary",
    "scenes": [
        {{
            "scene_id": "scene_1",
            "line_range": "L00001-L00012",
            "scene_goal": "goal of this scene or confrontation",
            "conflict_focus": "primary conflict or topic"
        }}
    ],
    "terms": [
        {{
            "src": "{src_lang} term",
            "tgt": "{tgt_lang} translation or original",
            "note": "Brief explanation"
        }},
        ...
    ],
    "speaker_profiles": [
        {{
            "speaker_id": "S01",
            "role": "conservative inferred role or unknown",
            "relationship": "relationship to other speakers or unknown",
            "status_and_formality": "relative status and expected register",
            "speaking_style": "tone, temperament, recurring verbal style",
            "relationships": [
                {{
                    "to_speaker": "S02",
                    "status_dynamic": "relative power / hierarchy",
                    "address_terms": "how this speaker addresses the other",
                    "attitude": "attitude and stance"
                }}
            ],
            "translation_guidance": "pronoun, honorific, address-term and tone guidance",
            "evidence": ["short dialogue evidence"]
        }}
    ],
    "dialogue_units": [
        {{
            "line_id": "L00001",
            "speaker_id": "S01",
            "addressed_to": "S02 or all or unknown",
            "speech_act": "accusation, question, threat, demand, bargain, or reaction",
            "core_proposition": "essential action/predicate/claim in this line"
        }}
    ]
}}

## INPUT
<text>
{source_content}
</text>
""".strip()
    return summary_prompt


def get_stt_correction_prompt(source_content, topic, terms, speaker_context=None):
    """Step 2: Correct STT errors based on summary context"""
    src_lang = load_key("whisper.detected_language")

    # Build context from summary
    terms_context = ""
    if terms:
        terms_list = [f"- {term['src']}: {term.get('note', '')}" for term in terms[:15]]  # Limit to 15 terms
        terms_context = "\n".join(terms_list)

    correction_prompt = f"""
## Role
You are a speech-to-text correction specialist for {src_lang}.
You correct ASR (Automatic Speech Recognition) transcription errors.

## Context (Use this to guide corrections)
### Video Topic
{topic}

### Key Terms in This Video
{terms_context}

### Speaker-Aware Dialogue Context
{speaker_context or "Not available"}

## Task
Correct ASR errors in the provided text lines based on the context above.

## CRITICAL RULES
⚠️ You must ONLY correct characters that sound similar (homophones).
⚠️ DO NOT add or remove any characters.
⚠️ The corrected line MUST have the EXACT SAME character count as the original line.

Only fix these types of errors:
- Homophones (同音字/谐音错误): Characters that sound the same but are written differently
  Examples: "绿" vs "微", "一起" vs "一气", "的" vs "地" vs "得"
- Similar pronunciation errors: Characters with similar sounds misrecognized
  Examples: "专业" vs "专研", "成分" vs "成份"
- Context-based corrections: Use the topic and terms above to identify likely misrecognitions

DO NOT fix:
- Missing or extra characters (字数变化)
- Word boundary errors (不要合并或拆分词语)
- Punctuation changes (do not add/remove punctuation)
- Rephrasing or rewriting sentences

## Steps
1. Read the topic and key terms to understand the video context
2. For each line, check if any characters appear to be homophone errors
3. Use the context to determine the correct character
4. ONLY replace if you are confident; otherwise keep original
5. Verify character count matches before and after

## Output in only JSON format
{{
    "corrected_lines": [
        "Corrected line 1 (SAME character count as input line 1)",
        "Corrected line 2 (SAME character count as input line 2)",
        "... one corrected line for each input line, maintaining the same order, count, AND character length"
    ]
}}

## IMPORTANT
- The "corrected_lines" array MUST have the EXACT SAME number of lines as the input
- Each corrected line MUST have the EXACT SAME character count as the original line
- ONLY replace homophones/similar-sounding characters, NEVER add or delete characters
- Keep the line order unchanged
- If a line has no homophone errors, return it UNCHANGED

## INPUT
<text>
{source_content}
</text>
""".strip()
    return correction_prompt


## ================================================================
# @ step5_translate.py & translate_lines.py
def generate_shared_prompt(previous_content_prompt, after_content_prompt, summary_prompt, things_to_note_prompt, speaker_context_prompt=None):
    return f'''### Context Information
<previous_content>
{previous_content_prompt}
</previous_content>

<subsequent_content>
{after_content_prompt}
</subsequent_content>

### Content Summary
{summary_prompt}

### Current Lines and Speaker IDs
<speaker_context>
{speaker_context_prompt}
</speaker_context>

Speaker IDs are metadata, not dialogue. Use them to preserve each character's role, status,
pronouns, honorifics, address terms, and speaking style. Never output speaker IDs in subtitles.

### Critical Semantic Invariants
1. Preserve explicit actions, numbers (taels, days, amounts), and conditions.
2. Maintain directional relationships: who speaks to whom; never turn a relayer into a promiser.
3. Maintain aggressive repetition and pressure (e.g. repeated demands for an explanation).

### Points to Note
{things_to_note_prompt}'''

def get_prompt_faithfulness(lines, shared_prompt):
    TARGET_LANGUAGE = effective_target_language("auto")
    # Split lines by \n
    line_splits = lines.split('\n')

    # Create JSON return format example
    json_format = {}
    for i, line in enumerate(line_splits, 1):
        json_format[i] = {
            "origin": line,
            "direct": f"<<direct {TARGET_LANGUAGE} translation>>"
        }

    src_language = load_key("whisper.detected_language")
    prompt_faithfulness = f'''
## Role
You are a professional Netflix subtitle translator, fluent in both {src_language} and {TARGET_LANGUAGE}, as well as their respective cultures.
Your expertise lies in accurately understanding the semantics and structure of the original {src_language} text and faithfully translating it into {TARGET_LANGUAGE} while preserving the original meaning.

## Task
We have a segment of original {src_language} subtitles that need to be directly translated into {TARGET_LANGUAGE}. These subtitles come from a specific context and may contain specific themes and terminology.

1. Translate the original {src_language} subtitles into {TARGET_LANGUAGE} line by line
2. Ensure the translation is faithful to the original, accurately conveying the original meaning
3. Consider the context and professional terminology

{shared_prompt}

<translation_principles>
1. Faithful to the original: Accurately convey the content and meaning of the original text, without arbitrarily changing, adding, or omitting content.
2. Accurate terminology: Use professional terms correctly and maintain consistency in terminology.
3. Understand the context: Fully comprehend and reflect the background and contextual relationships of the text.
4. Speaker consistency: Use the supplied speaker identity and profile to resolve pronouns, titles,
   formality, implied subjects, and dialogue intent consistently across batch boundaries.
</translation_principles>

## INPUT
<subtitles>
{lines}
</subtitles>

## Output in only JSON format
{json.dumps(json_format, ensure_ascii=False, indent=4)}

Note: << >> represents placeholders that should not appear in your answer
'''
    return prompt_faithfulness.strip()


def get_prompt_expressiveness(faithfulness_result, lines, shared_prompt):
    TARGET_LANGUAGE = load_key("target_language")

    # Examples for filler words in different languages
    filler_examples = {
        "zh": '''
        - "Okay" → "好的", "没问题", "行", "可以", "明白了", "我知道了" (not just "好")
        - "Right" → "对的", "没错", "是这样的", "确实如此" (not just "对")
        - "Yeah" → "是的", "没错", "对的对的", "确实如此", "就是这样" (not just "对")
        - "Yes" → "是的", "没错", "当然了", "没问题" (not just "是")
        - "Well" → "嗯", "这个嘛", "其实呢", "那个" (context-dependent)
        - "You know" → "你知道的", "你懂的", "你看" (conversational)
        - "I mean" → "我是说", "我的意思是", "就是说" (clarification)''',

        "ja": '''
        - "Okay" → "わかりました", "了解しました", "大丈夫です", "いいですよ" (not just "はい")
        - "Right" → "そうですね", "その通りです", "確かに" (not just "はい")
        - "Yeah" → "そうですね", "その通りです", "そうですよ", "確かに" (not just "はい")
        - "Well" → "えーと", "そうですね", "まあ", "あのー" (context-dependent)
        - "You know" → "ご存知の通り", "わかるでしょう", "ね" (conversational)
        - "I mean" → "つまり", "言いたいのは", "私が言いたいのは" (clarification)''',

        "fr": '''
        - "Okay" → "D'accord", "Très bien", "Entendu", "Pas de problème" (not just "OK")
        - "Right" → "C'est vrai", "Exactement", "Tout à fait", "En effet" (not just "Oui")
        - "Yeah" → "Ouais", "Effectivement", "Tout à fait", "C'est ça" (not just "Oui")
        - "Well" → "Eh bien", "Bon", "Alors", "En fait" (context-dependent)
        - "You know" → "Tu sais", "Vous savez", "Tu vois ce que je veux dire" (conversational)
        - "I mean" → "Je veux dire", "C'est-à-dire", "En d'autres termes" (clarification)''',

        "es": '''
        - "Okay" → "De acuerdo", "Está bien", "Vale", "Perfecto" (not just "OK")
        - "Right" → "Exacto", "Así es", "Correcto", "Efectivamente" (not just "Sí")
        - "Yeah" → "Claro", "Por supuesto", "Desde luego", "Efectivamente" (not just "Sí")
        - "Well" → "Bueno", "Pues", "Verás", "Es que" (context-dependent)
        - "You know" → "Ya sabes", "Como sabrás", "¿Sabes?" (conversational)
        - "I mean" → "Quiero decir", "O sea", "Es decir" (clarification)''',

        "ru": '''
        - "Okay" → "Хорошо", "Ладно", "Понятно", "Договорились" (not just "ОК")
        - "Right" → "Верно", "Правильно", "Именно так", "Действительно" (not just "Да")
        - "Yeah" → "Да конечно", "Точно", "Именно так", "Верно говоришь" (not just "Да")
        - "Well" → "Ну", "Так", "Видите ли", "Дело в том, что" (context-dependent)
        - "You know" → "Знаете", "Понимаете", "Видите ли" (conversational)
        - "I mean" → "Я имею в виду", "То есть", "В смысле" (clarification)''',

        "de": '''
        - "Okay" → "In Ordnung", "Alles klar", "Verstanden", "Geht klar" (not just "OK")
        - "Right" → "Richtig", "Genau", "Stimmt", "Ganz recht" (not just "Ja")
        - "Yeah" → "Jawohl", "Na klar", "Aber sicher", "Selbstverständlich" (not just "Ja")
        - "Well" → "Nun", "Also", "Tja", "Naja" (context-dependent)
        - "You know" → "Weißt du", "Verstehst du", "Nicht wahr" (conversational)
        - "I mean" → "Ich meine", "Das heißt", "Anders gesagt" (clarification)'''
    }

    # Get language code (first 2 characters)
    lang_code = TARGET_LANGUAGE.lower()[:2]

    # Get examples for the target language or use a generic message
    examples = filler_examples.get(lang_code, "Use natural expressions in the target language for filler words like 'Okay', 'Right', etc.")

    json_format = {}
    for key, value in faithfulness_result.items():
        json_format[key] = {
            "origin": value['origin'],
            "direct": value['direct'],
            "reflection": "reflection on the direct translation version",
            "free": f"retranslated result, aiming for fluency and naturalness, conforming to {TARGET_LANGUAGE} expression habits, DO NOT leave empty line here!"
        }

    src_language = load_key("whisper.detected_language")
    prompt_expressiveness = f'''
## Role
You are a professional Netflix subtitle translator and language consultant.
Your expertise lies not only in accurately understanding the original {src_language} but also in optimizing the {TARGET_LANGUAGE} translation to better suit the target language's expression habits and cultural background.

## Task
We already have a direct translation version of the original {src_language} subtitles. Now we need you to reflect on and improve these direct translations to create more natural and fluent {TARGET_LANGUAGE} subtitles.

1. Analyze the direct translation results line by line, pointing out existing issues
2. Provide detailed modification suggestions
3. Perform free translation based on your analysis
4. Do not add comments or explanations in the translation, as the subtitles are for the audience to read
5. Preserve each speaker's established register, status relationship, pronouns, titles, and recurring address terms

{shared_prompt}

<Translation Analysis Steps>
Please use a two-step thinking process to handle the text line by line:

1. Direct Translation Reflection:
   - Evaluate language fluency
   - Check if the language style is consistent with the original text
   - Check the conciseness of the subtitles, point out where the translation is too wordy, the translation should be close to the original text in length

2. {TARGET_LANGUAGE} Free Translation:
   - **DUBBING LENGTH LIMIT**: The translated text must be short enough to speak naturally inside the original subtitle time window. Character count is only a weak proxy; prioritize spoken duration and rhythm.
   - **If translation is too long**: Use shorter synonyms, remove filler words, simplify sentence structure, or rephrase more concisely.
   - **MEANING PRESERVATION OVERRIDES BREVITY**: Never shorten by deleting a core proposition. Preserve every explicit predicate/action, object, negation, quantity, name, and contrast from the source and direct translation.
   - Before returning each `free` line, compare it with both `origin` and `direct`. If an action such as tax, collect, pay, take, give, stand, kneel, or earn is explicit, that action must remain explicit in `free`.
   - Bad shortening example: translating “If you don't tax the poor, who do you tax?” as “If not the poor, then who?” because the action “tax” was dropped. A valid concise version is “If not the poor, who do you tax?”
   - **Avoid Unnecessary Expansion.** Do not add filler words or extra explanations unless absolutely necessary for meaning.
   - Aim for contextual smoothness and naturalness, conforming to {TARGET_LANGUAGE} expression habits
   - Ensure it's easy for {TARGET_LANGUAGE} audience to understand and accept
   - Adapt the language style to match the video's theme
   - Pay special attention to conversational markers and filler words (e.g., "Okay", "right", "well", "you know", "I mean", etc.):
     * Translate these using natural, culturally appropriate expressions in {TARGET_LANGUAGE}
     * Use varied expressions that sound natural to native speakers
     * IMPORTANT: NEVER translate filler words or conversational markers into single-character words (e.g., never translate "Yeah" as just "对" in Chinese)
     * Always use multi-character expressions for filler words to make them more suitable for dubbing and more natural sounding
     * Consider the speaker's tone and context when translating these expressions
     * Examples for common filler words in {TARGET_LANGUAGE}:
       - For "Okay", "Right", "Well", "You know", "I mean", "Yeah", "Yes", "No", etc., use natural expressions that a native speaker would use
       - Avoid direct word-for-word translations that sound unnatural
       - Consider the context, tone, and formality level when choosing appropriate expressions
       {examples}
</Translation Analysis Steps>

## INPUT
<subtitles>
{lines}
</subtitles>

### Output in only JSON format, repeat "origin" and "direct" in the JSON format
{json.dumps(json_format, ensure_ascii=False, indent=4)}
'''
    return prompt_expressiveness.strip()


def get_dubbing_rewrite_prompt(
    original,
    current_lines,
    target_duration,
    available_duration,
    reason,
    source_lines=None,
    direction="shorten",
    word_budget=None,
    speaker=None,
    speaker_profile=None,
    semantic_constraints=None,
):
    TARGET_LANGUAGE = effective_target_language("auto")
    target_code = str(TARGET_LANGUAGE).lower()
    source_lines = source_lines or []
    current_text = "\n".join(f"{i + 1}. {line}" for i, line in enumerate(current_lines))
    source_text = "\n".join(f"{i + 1}. {line}" for i, line in enumerate(source_lines))
    line_count = max(len(current_lines), 1)
    if direction == "expand":
        line_rule = f"Keep the same number of output lines as the current dubbing text: exactly {line_count} line(s)."
        task = f"Rewrite the current {TARGET_LANGUAGE} dubbing text so it fills the target time window more naturally without sounding padded."
        duration_rule = (
            f"The current speech is too short. Make the wording naturally richer so it can be spoken close to "
            f"{target_duration:.2f}s, while staying within the hard window of {available_duration:.2f}s."
        )
        if word_budget:
            duration_rule += (
                f" The current text has about {word_budget['current_words']} words; output should total "
                f"{word_budget['min_words']} to {word_budget['max_words']} words across all lines."
            )
        if target_code.startswith("vi"):
            style_rule = (
                "For Vietnamese, use natural live-commerce phrasing, short connective words, and clear product context; "
                "do not invent new claims, prices, guarantees, or features."
            )
        elif target_code.startswith("en"):
            style_rule = "For English, use plain spoken wording, short common words, and natural contractions."
        else:
            style_rule = "Use natural spoken wording for the target language without adding new claims."
        edit_rule = "Prefer adding only enough natural context to close the timing gap; do not double the script length."
    else:
        line_rule = (
            f"Use no more than {line_count} output line(s). Fewer lines are allowed when the time window is very short."
        )
        task = f"Rewrite the current {TARGET_LANGUAGE} dubbing text so it can be spoken naturally within the target time window."
        duration_rule = f"Make the wording concise enough for natural speech within {target_duration:.2f}s. The available hard window is {available_duration:.2f}s."
        if word_budget:
            duration_rule += (
                f" The current text has about {word_budget['current_words']} words; output must total "
                f"at most {word_budget['max_words']} words across all lines."
            )
        if target_code.startswith("vi"):
            style_rule = "For Vietnamese, use standard tone marks, natural short phrasing, and avoid unnecessary filler words."
        elif target_code.startswith("en"):
            style_rule = "For English, use plain spoken wording, short common words, and natural contractions."
        else:
            style_rule = "Use natural short phrasing in the target language and avoid unnecessary filler words."
        edit_rule = "CRITICAL FACT INTEGRITY: You MUST preserve all participants (who does what to whom), all explicit amounts/quantities, dates/deadlines, and conjoined actions. DO NOT drop participants or core bargains just to shorten the text. Prefer natural contractions, concise vocabulary, and compressing wordiness."

    speaker_info = ""
    if speaker or speaker_profile:
        speaker_info = f"\n## Character & Speaker\nSpeaker: {speaker or 'UNKNOWN'}\nProfile: {speaker_profile or 'Maintain speaker tone'}\n"
    constraints_info = ""
    if semantic_constraints:
        constraints_info = f"\n## Semantic Invariants to Preserve\n{semantic_constraints}\n"

    return f'''
## Role
You are a professional dubbing script editor for high-sync video localization.

## Task
{task}

## Hard Constraints
1. Preserve the original meaning, terminology, named entities, numbers, and speaker intent.
2. {line_rule}
3. {duration_rule}
4. Do not add explanations, comments, brackets, markdown, or extra lines.
5. {style_rule}
6. {edit_rule}

{speaker_info}{constraints_info}
## Reason This Needs Rewriting
{reason}

## Original Source Context
{original}

## Source Lines
{source_text}

## Current Dubbing Lines
{current_text}

## Output JSON Only
{{
  "lines": [
    "rewritten line 1",
    "rewritten line 2"
  ]
}}
'''.strip()


## ================================================================
# @ step6_splitforsub.py
def get_align_prompt(src_sub, tr_sub, src_part):
    TARGET_LANGUAGE = load_key("target_language")
    src_language = load_key("whisper.detected_language")
    src_splits = src_part.split('\n')
    num_parts = len(src_splits)
    src_part = src_part.replace('\n', ' [br] ')
    align_prompt = '''
## Role
You are a Netflix subtitle alignment expert fluent in both {src_language} and {target_language}.

## Task
We have {src_language} and {target_language} original subtitles for a Netflix program, as well as a pre-processed split version of {src_language} subtitles. Your task is to create the best splitting scheme for the {target_language} subtitles based on this information.

1. Analyze the word order and structural correspondence between {src_language} and {target_language} subtitles
2. Split the {target_language} subtitles according to the pre-processed {src_language} split version
3. Never split at single punctuation marks like commas - only split at natural phrase boundaries or longer pauses
4. Never leave empty lines. If it's difficult to split based on meaning, you may appropriately rewrite the sentences that need to be aligned
5. Do not add comments or explanations in the translation, as the subtitles are for the audience to read

## INPUT
<subtitles>
{src_language} Original: "{src_sub}"
{target_language} Original: "{tr_sub}"
Pre-processed {src_language} Subtitles ([br] indicates split points): {src_part}
</subtitles>

## Output in only JSON format
{{
    "analysis": "Brief analysis of word order, structure, and semantic correspondence between {src_language} and {target_language} subtitles",
    "align": [
        {align_parts_json}
    ]
}}
'''

    align_parts_json = ','.join(
        f'''
        {{
            "src_part_{i+1}": "{src_splits[i]}",
            "target_part_{i+1}": "Corresponding aligned {TARGET_LANGUAGE} subtitle part"
        }}''' for i in range(num_parts)
    )

    return align_prompt.format(
        src_language=src_language,
        target_language=TARGET_LANGUAGE,
        src_sub=src_sub,
        tr_sub=tr_sub,
        src_part=src_part,
        align_parts_json=align_parts_json,
    )

## ================================================================
# @ step8_gen_audio_task.py @ step10_gen_audio.py
def get_subtitle_trim_prompt(text, duration):

    rule = '''Consider a. Reducing filler words without modifying meaningful content. b. Omitting unnecessary modifiers or pronouns, for example:
    - "Please explain your thought process" can be shortened to "Please explain thought process"
    - "We need to carefully analyze this complex problem" can be shortened to "We need to analyze this problem"
    - "Let's discuss the various different perspectives on this topic" can be shortened to "Let's discuss different perspectives on this topic"
    - "Can you describe in detail your experience from yesterday" can be shortened to "Can you describe yesterday's experience" '''

    trim_prompt = '''
## Role
You are a professional subtitle editor, editing and optimizing lengthy subtitles that exceed voiceover time before handing them to voice actors.
Your expertise lies in cleverly shortening subtitles slightly while ensuring the original meaning and structure remain unchanged.

## INPUT
<subtitles>
Subtitle: "{text}"
Duration: {duration} seconds
</subtitles>

## Processing Rules
{rule}

## Processing Steps
Please follow these steps and provide the results in the JSON output:
1. Analysis: Briefly analyze the subtitle's structure, key information, and filler words that can be omitted.
2. Trimming: Based on the rules and analysis, optimize the subtitle by making it more concise according to the processing rules.

## Output in only JSON format
{{
    "analysis": "Brief analysis of the subtitle, including structure, key information, and potential processing locations",
    "result": "Optimized and shortened subtitle in the original subtitle language"
}}
'''.strip()
    return trim_prompt.format(
        text=text,
        duration=duration,
        rule=rule
    )

def get_batch_subtitle_trim_prompt(subtitles_list):
    """
    subtitles_list: list of dict, e.g.,
    [
        {"index": 1, "text": "...", "duration": 2.5},
        ...
    ]
    """

    rule = '''Consider a. Reducing filler words without modifying meaningful content. b. Omitting unnecessary modifiers or pronouns, for example:
    - "Please explain your thought process" can be shortened to "Please explain thought process"
    - "We need to carefully analyze this complex problem" can be shortened to "We need to analyze this problem"
    '''

    subtitles_text = json.dumps(subtitles_list, ensure_ascii=False, indent=2)

    trim_prompt = f'''
## Role
You are a professional subtitle editor, editing and optimizing lengthy subtitles that exceed voiceover time before handing them to voice actors.
Your expertise lies in cleverly shortening subtitles slightly while ensuring the original meaning and structure remain unchanged.

## INPUT
<subtitles>
{subtitles_text}
</subtitles>

## Processing Rules
{rule}

## Task
For each item in the input list, optimize the "text" to fit within the "duration" (or make it shorter).
Keep the "index" unchanged.

## Output in only JSON format
{{
    "results": [
        {{
            "index": <original_index>,
            "analysis": "Brief analysis...",
            "shortened_text": "Optimized subtitle"
        }},
        ...
    ]
}}
'''.strip()
    return trim_prompt

## ================================================================
# @ tts_main
def get_correct_text_prompt(text):
    return f'''
## Role
You are a text cleaning expert for TTS (Text-to-Speech) systems.

## Task
Clean the given text by:
1. Keep only basic punctuation (.,?!)
2. Preserve the original meaning

## INPUT
{text}

## Output in only JSON format
{{
    "text": "cleaned text here"
}}
'''.strip()

## ================================================================
# @ step8_1_gen_audio_task.py
def get_expand_short_text_prompt(text, origin_text):
    TARGET_LANGUAGE = load_key("target_language")
    return f'''
## Role
You are a professional subtitle editor specializing in {TARGET_LANGUAGE}, tasked with optimizing very short subtitles for text-to-speech systems.

## Task
Expand the given short subtitle text to make it more suitable for TTS processing while preserving the original meaning.

1. The original text is too short (3 characters or less), which often causes TTS systems to fail
2. Add natural filler words, descriptive elements, or context-appropriate phrases to extend the text
3. Maintain the exact same meaning and tone as the original
4. Keep the expanded text concise and natural - aim for 8-15 characters
5. Do not change the core message or add new information

## INPUT
<subtitle>
Short text: "{text}"
Original source text: "{origin_text}"
</subtitle>

## Output in only JSON format
{{
    "analysis": "Brief analysis of why the text needs expansion",
    "expanded_text": "The expanded text in {TARGET_LANGUAGE} that preserves the original meaning"
}}
'''.strip()
