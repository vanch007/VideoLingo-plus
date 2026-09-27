from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd
from rich.console import Console
from rich.table import Table

from core.ask_gpt import ask_gpt
from core.config_utils import load_key

console = Console()

DEFAULT_REVIEW_LOG_PATH = Path("output/log/translation_review.json")


def get_translation_review_prompt(
    lines_data: list[dict[str, Any]],
    speaker_profiles_text: str,
    context_topic: str,
) -> str:
    target_lang = load_key("target_language", "en")
    src_lang = load_key("whisper.detected_language", "zh")

    formatted_lines = []
    for item in lines_data:
        lid = item.get("line_id", "")
        spk = item.get("speaker", "UNKNOWN")
        src = item.get("source", "")
        trans = item.get("translation", "")
        formatted_lines.append(f"[{lid}][{spk}]\n  Source: {src}\n  Translation: {trans}")
    lines_text = "\n\n".join(formatted_lines)

    return f"""
## Role
You are an expert bilingual subtitle auditor and dramaturgical script reviewer specializing in {src_lang} to {target_lang} translation.

## Task
Review the dialogue translation line-by-line for semantic fidelity, speaker-role consistency, and directional relationships.
DO NOT use outside movie scripts or invented background. Base your audit strictly on the provided dialogue evidence.

## CRITICAL Audit Standards:
- Flag an issue ONLY for genuine factual errors, agent/patient reversals (who does what to whom, who pays whom), dropped numbers/quantities, inverted negation, or dropped core actions.
- STRICT CONVERGENCE: If a translation accurately conveys the source meaning, actors, and numbers without factual distortion, YOU MUST MARK IT AS HAVING NO ISSUES. Return empty issues list `{{"issues": []}}` if all lines are acceptable.
- DO NOT suggest stylistic tweaks, alternate synonyms, or dialect re-interpretations across rounds. Once a line has its core facts right, DO NOT flag it again.
- DO NOT flag stylistic preferences, alternate word choices, or minor phrasing variations if the current translation is already faithful and clear. If a translation is acceptable and preserves the core meaning, mark it as having NO issues.
- Faithful fidelity to spoken words: Translate what is actually spoken in the source without unprompted alterations. Do NOT oscillate between literal and inferential interpretations of pronouns, names, or spoken idioms if the current translation is already faithful to what was spoken.
- DO NOT flag capitalization differences (e.g. 'our' vs 'OUR'), cosmetic emphasis, or minor punctuation differences. If the translation preserves the source facts, actions, numbers, and agent/patient directions faithfully, mark it as having NO issues.

## Focus Audit Dimensions:
1. **Agent / Patient & Relayer Confusion**:
   - Verify who does what to whom. If speaker A is demanding speaker B to do something, ensure A does not become the doer.
   - If speaker C is merely repeating or relaying what speaker A promised (e.g., "S01 will give you money"), ensure speaker C does NOT promise "I will give you".
   - Distinct Lines Principle: Evaluate each line under review strictly against its own specific source dialogue and speaker perspective. Distinct lines spoken by different speakers or in different turns often use different pronouns or perspective; do NOT force different lines to artificially harmonize pronouns or terms if their source texts differ.
2. **Negation & Rhetorical Polarity**:
   - Ensure negative statements ("不出钱", "不是", "不刮") are not inverted to positive, or rhetorical questions ("为什么不出钱剿匪") mistranslated into statements.
3. **Numbers, units, dates & quantities**:
   - Ensure explicit amounts, count classifiers, and multipliers (e.g., "三天" -> "in three days", "五千美元" -> "$5,000", "五回" -> "five times", "四头" -> "all four of them") match accurately. NEVER omit specific counts or replace quantities with vague generalities (e.g. do NOT translate "四头全是..." as just "They're all...").
4. **Core actions & predicates**:
   - Explicit actions (e.g. tax, arrest, kidnap, kill, execute / shoot / 枪毙, explain, translate, suppress bandits / 剿匪) must NEVER be dropped, altered, or replaced with unrelated actions.
   - In causative or passive constructions (e.g., "让黄老爷枪毙了五回" -> "Master Huang had him shot/executed five times"), ensure the core predicate ("shot" / "executed") and the agent/patient relationship are accurately retained. Do NOT substitute specific actions with unrelated generalizations (e.g. do NOT change "枪毙了五回" into "betrayed his own side").
5. **Character consistency & form of address**:
   - Check if address terms and tone match the speaker's status, power dynamic, and relationship to the addressee.
6. **No lore hallucination**:
   - Ensure translation does not invent biographical facts or outside plot details absent from the dialogue.

## CRITICAL Output Format Rules:
1. "suggested_fix" MUST be the FULL, complete, final spoken dialogue text for the ENTIRE line in the target language ({target_lang}) ONLY.
   - "suggested_fix" MUST contain the COMPLETE line from start to finish. If you are correcting only one word or clause, you MUST include the entire rest of the line unchanged. NEVER output only a sub-clause or partial fragment.
   - "suggested_fix" MUST ALWAYS be in the target language ({target_lang}). NEVER output source-language characters in "suggested_fix".
   - NEVER output internal speaker labels or bracketed tags (e.g. "[S01]", "[SPEAKER_00]") in "suggested_fix". Use natural names or pronouns.
   - If the source is an isolated reaction or interjection (e.g. "哎", "哈"), translate its communicative intent into the target language, NEVER output raw source characters.
2. NEVER include conditional alternatives (e.g., "Confirm speaker assignment...", "If S01... If S02..."), notes, or meta commentary in "suggested_fix". If speaker assignment is ambiguous, fix the translation of the spoken words directly based on what was said, without meta text.
3. NEVER append explanatory parentheticals (e.g., "(maintains profanity...)", "(note: ...)") to "suggested_fix". Put all explanations in the "problem" field.
4. Crucial dialogue information MUST NOT be dropped:
   - When the source includes an exchange or conditional bargain (e.g. "你付款我发货" -> "You pay, I deliver the goods"), do NOT delete the exchange action.
   - When the source includes specific deadlines, timeframes, or quantities (e.g. "两天后" -> "in two days", "五千美元" -> "5,000 dollars"), do NOT drop the timeframe or the amount.
   - Profanity and intense emotional interjections in the source (e.g. "他妈的") are genuine spoken performance, NOT hallucination. They MUST be preserved with appropriate colloquial equivalents. Do NOT sanitize or soften them based on character profiles.

## Context & Speaker Profiles
### Topic: {context_topic}
### Speaker Profiles:
{speaker_profiles_text}

## Subtitle Lines Under Review
{lines_text}

## Output JSON Format Only
{{
  "overall_summary": "1-2 sentence assessment of translation fidelity",
  "issues": [
    {{
      "line_id": "L00001",
      "speaker": "S01",
      "issue_type": "agent_patient_confusion | dropped_negation | number_mismatch | dropped_predicate | register_inconsistency | hallucination",
      "severity": "critical | warning | suggestion",
      "problem": "Specific explanation of the error",
      "suggested_fix": "Exact revised translation for this line that fixes the issue"
    }}
  ]
}}
""".strip()


def is_valid_target_language(text: str, target_lang: str) -> bool:
    """Validate that translation text conforms to target language and contains no source leak."""
    if not text or not isinstance(text, str):
        return False
    text_clean = text.strip()
    if not text_clean:
        return False
    # Reject internal speaker tags or bracketed IDs leaking into dialogue
    if re.search(r"\[(?:S\d+|SPEAKER_\d+|Speaker\s*\w*)\]", text_clean, re.IGNORECASE):
        return False
    lang = target_lang.lower()
    if lang.startswith("en"):
        # English must contain latin letters and no CJK characters
        if re.search(r"[一-鿿]", text_clean):
            return False
        return bool(re.search(r"[a-zA-Z]", text_clean))
    elif lang.startswith("zh"):
        return bool(re.search(r"[一-鿿]", text_clean))
    elif lang.startswith("vi"):
        if re.search(r"[一-鿿]", text_clean):
            return False
        return bool(re.search(r"[a-zA-ZÀ-ỹ]", text_clean))
    return True


def is_meta_explanation(text: str) -> bool:
    """Detect whether a string is meta-commentary, conditional instructions, or audit notes rather than dialogue."""
    if not text or not isinstance(text, str):
        return True
    t = text.strip()
    meta_patterns = [
        r"(?i)\b(?:confirm|check|verify)\s+speaker\b",
        r"(?i)\bif\s+(?:indeed\s+)?S\d+\b",
        r"(?i)\bif\s+S\d+\s*:",
        r"(?i)\b(?:cannot|can't)\s+determine\b",
        r"(?i)\bneeds?\s+verification\b",
        r"(?i)\bspeaker\s+assignment\b",
        r"(?i)\bcharacter\s+inconsistency\b",
        r"(?i)\b(?:option\s+\d+|alternative\s+\d+)\s*:",
        r"(?i)\bthe\s+translation\s+(?:depends\s+on|should\s+be)\b",
        r"(?i)\bhighlights\s+character\b",
    ]
    for p in meta_patterns:
        if re.search(p, t):
            return True
    return False


def clean_suggested_fix(text: str) -> str:
    """Clean conversational or explanatory meta-phrases from suggested translations."""
    if not text or not isinstance(text, str):
        return ""
    cleaned = text.strip()
    # First strip standard prompt/preamble wrappers
    m = re.search(r"^(?:Assuming[^,:]+,\s*)?(?:the\s+)?translation\s+(?:should be|is):\s*(.*?)\.?$", cleaned, flags=re.IGNORECASE)
    if m and m.group(1):
        cleaned = m.group(1).strip()
    cleaned = re.sub(r"^(?:Note|Fix|Correction|Translation|Suggested fix|Revised|Revision):\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*\((?:maintains|highlights|note|character|inconsistency|context|explanation|shows|tone)[^)]*\)\s*$", "", cleaned, flags=re.IGNORECASE).strip()
    quote_chars = ('"', "'", "“", "”")
    if len(cleaned) >= 2 and cleaned[0] in quote_chars and cleaned[-1] in quote_chars:
        cleaned = cleaned[1:-1].strip()
    # Strip any leaked bracketed speaker labels like [S01] or [SPEAKER_00]
    cleaned = re.sub(r"\[(?:S\d+|SPEAKER_\d+)\]\s*", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"\s*\[(?:S\d+|SPEAKER_\d+)\]", "", cleaned, flags=re.IGNORECASE).strip()
    # Strip trailing speculative phrases (e.g. '...or something', '...or so')
    cleaned = re.sub(r"\s*\.{2,}\s*or\s+something\.?$", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"\s*\((?:maintains|highlights|note|character|inconsistency|context|explanation|shows|tone)[^)]*\)\s*$", "", cleaned, flags=re.IGNORECASE).strip()
    # If the remaining dialogue payload is a meta explanation or conditional instruction, reject it
    if is_meta_explanation(cleaned):
        return ""
    return cleaned


def validate_review_response(response: Any) -> dict[str, str]:
    if not isinstance(response, dict):
        return {"status": "error", "message": "Response must be a JSON object"}
    st = response.get("status")
    if st is not None and st not in ("success", "pass"):
        return {"status": "error", "message": f"Review response returned failed status: {response.get('overall_summary', st)}"}
    if "issues" not in response or not isinstance(response["issues"], list):
        return {"status": "error", "message": "Missing 'issues' list in review response"}
    for item in response["issues"]:
        if not isinstance(item, dict):
            return {"status": "error", "message": "Issue item must be an object"}
        if "line_id" not in item or "suggested_fix" not in item:
            return {"status": "error", "message": "Each issue must have 'line_id' and 'suggested_fix'"}
        sev = item.get("severity")
        if not sev or sev not in ("critical", "warning", "suggestion"):
            return {"status": "error", "message": "Each issue must have a valid 'severity' ('critical', 'warning', or 'suggestion')"}
    return {"status": "success", "message": "Review validated"}


def deterministic_semantic_fixes(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    fixes = []
    df = df.copy()
    for idx, row in df.iterrows():
        lid = str(row.get("LineID", ""))
        trans = str(row.get("Translation", ""))
        updated = trans
        if "tatels" in updated:
            updated = updated.replace("tatels", "taels")
        if updated != trans:
            df.at[idx, "Translation"] = updated
            fixes.append({
                "line_id": lid,
                "issue_type": "deterministic_semantic_fix",
                "severity": "critical",
                "problem": "Deterministic typo fix applied",
                "before": trans,
                "after": updated,
            })
    return df, fixes
def review_and_repair_translation(
    df: pd.DataFrame,
    terminology: dict[str, Any],
    report_path: Path | str = DEFAULT_REVIEW_LOG_PATH,
    max_repair_rounds: int = 3,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Perform independent semantic review and multi-round repair loop on translation results."""
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    profiles = terminology.get("speaker_profiles", []) if terminology else []
    profiles_text = json.dumps(profiles, ensure_ascii=False, indent=2) if profiles else "None"
    topic = str(terminology.get("topic", "")) if terminology else ""

    target_lang = str(load_key("target_language", "en"))
    review_status = "pass"
    review_error = None
    all_issues: list[dict[str, Any]] = []
    applied_repairs: list[dict[str, Any]] = []
    overall_summary = ""

    df_working = df.copy()
    df_working, det_fixes = deterministic_semantic_fixes(df_working)
    if det_fixes:
        applied_repairs.extend(det_fixes)
        for fix in det_fixes:
            console.print(f"[green]✅ Deterministic guardrail on {fix['line_id']}: {fix['before']} -> {fix['after']}[/green]")
    valid_line_ids = set(str(lid) for lid in df_working["LineID"].dropna().unique())

    if max_repair_rounds <= 0:
        review_status = "failed"
        review_error = "max_repair_rounds must be >= 1; translation review cannot pass without execution"
        review_report = {
            "status": review_status,
            "error": review_error,
            "overall_summary": "Review aborted: max_repair_rounds <= 0",
            "total_lines_reviewed": len(df),
            "issues_found_count": 0,
            "repairs_applied_count": 0,
            "issues": [],
            "applied_repairs": [],
        }
        report_path.write_text(json.dumps(review_report, ensure_ascii=False, indent=2), encoding="utf-8")
        return df_working, review_report

    round_num = 1
    repaired_line_counts = {}
    while round_num <= max_repair_rounds:
        console.print(f"[cyan]🔍 Semantic review round {round_num}/{max_repair_rounds}...[/cyan]")
        lines_data = []
        for _, row in df_working.iterrows():
            lines_data.append({
                "line_id": str(row.get("LineID", "")),
                "speaker": str(row.get("Speaker", "")),
                "source": str(row.get("Source", "")),
                "translation": str(row.get("Translation", "")),
            })
        prompt = get_translation_review_prompt(lines_data, profiles_text, topic)

        try:
            review_result = ask_gpt(
                prompt,
                response_json=True,
                valid_def=validate_review_response,
                log_title=f"translation_semantic_review_r{round_num}",
            )
        except Exception as e:
            console.print(f"[yellow]⚠️ Semantic review encountered error: {e}.[/yellow]")
            review_status = "failed"
            review_error = f"Review failed in round {round_num}: {e}"
            break

        if not isinstance(review_result, dict) or review_result.get("status") in ("failed", "error", "fail"):
            review_status = "failed"
            review_error = review_result.get("overall_summary") if isinstance(review_result, dict) else f"Review returned failed status in round {round_num}"
            if not review_error:
                review_error = f"Review returned failed status in round {round_num}"
            break

        overall_summary = review_result.get("overall_summary", "")
        issues = review_result.get("issues", [])
        all_issues.extend(issues)

        if not issues:
            console.print(f"[green]✅ Round {round_num}: No semantic issues detected.[/green]")
            break

        unresolved_critical = []
        round_repairs = []

        for iss in issues:
            lid = str(iss.get("line_id", ""))
            sev = str(iss.get("severity", "") or "").lower()
            if sev not in ("critical", "warning", "suggestion"):
                sev = "critical"
            raw_fix = iss.get("suggested_fix", "")
            fix = clean_suggested_fix(str(raw_fix)) if raw_fix else ""

            prob = iss.get("problem", "")
            pattern = r"[\x27\x22\u2018\u201c]([\u4e00-\u9fff]{2,})[\x27\x22\u2019\u201d]"
            m_quote = re.search(pattern, prob)
            if m_quote:
                snip = m_quote.group(1)
                curr_source = str(df_working.loc[df_working["LineID"] == lid, "Source"].values[0]) if (df_working["LineID"] == lid).any() else ""
                if snip not in curr_source:
                    matches = df_working[df_working["Source"].astype(str).str.contains(snip, na=False)]
                    if len(matches) == 1:
                        corrected_lid = str(matches.iloc[0]["LineID"])
                        console.print(f"[yellow]⚠️ Corrected misaligned line_id {lid} -> {corrected_lid} matching '{snip}'[/yellow]")
                        lid = corrected_lid
            elif lid not in valid_line_ids:
                m_quote = re.search(pattern, prob)
                if m_quote:
                    snip = m_quote.group(1)
                    matches = df_working[df_working["Source"].astype(str).str.contains(snip, na=False)]
                    if len(matches) == 1:
                        lid = str(matches.iloc[0]["LineID"])

            if lid not in valid_line_ids:
                if sev in ("critical", "warning"):
                    unresolved_critical.append(iss)
                continue

            if not fix:
                if sev in ("critical", "warning"):
                    unresolved_critical.append(iss)
                continue

            if sev in ("critical", "warning"):
                if repaired_line_counts.get(lid, 0) >= 1 and sev != "critical" and iss.get("issue_type") not in ("number_mismatch", "dropped_negation"):
                    # Prevent infinite cosmetic oscillation on previously repaired lines for non-critical warnings
                    continue
                mask = df_working["LineID"] == lid
                if mask.any():
                    old_val = df_working.loc[mask, "Translation"].values[0]
                    if old_val == fix:
                        continue
                    if not is_valid_target_language(fix, target_lang):
                        console.print(
                            f"[bold red]❌ Rejected invalid fix for {lid}: {fix} "
                            f"(not valid {target_lang})[/bold red]"
                        )
                        if sev == "critical":
                            unresolved_critical.append(iss)
                        continue
                    old_words = old_val.split()
                    fix_words = fix.split()
                    if len(old_words) >= 8 and len(fix_words) <= 4 and iss.get("issue_type") not in ("hallucination", "dropped_predicate"):
                        console.print(
                            f"[yellow]⚠️ Detected partial fragment fix for {lid} ({len(fix_words)} words vs {len(old_words)} words). Skipping destructive truncation.[/yellow]"
                        )
                        continue
                    df_working.loc[mask, "Translation"] = fix
                    repair_item = {
                        "round": round_num,
                        "line_id": lid,
                        "issue_type": iss.get("issue_type"),
                        "severity": sev,
                        "problem": iss.get("problem"),
                        "before": old_val,
                        "after": fix,
                    }
                    round_repairs.append(repair_item)
                    applied_repairs.append(repair_item)
                    repaired_line_counts[lid] = repaired_line_counts.get(lid, 0) + 1
                    console.print(f"[green]✅ Round {round_num} Repaired {lid}: {old_val} -> {fix}[/green]")

        if unresolved_critical:
            review_status = "failed"
            review_error = f"{len(unresolved_critical)} unresolvable critical issue(s) in round {round_num}"
            console.print(f"[bold red]❌ Semantic review failed: {review_error}[/bold red]")
            break

        if not round_repairs:
            has_remaining_issues = any(iss.get("severity") in ("critical", "warning") for iss in issues)
            if has_remaining_issues:
                review_status = "failed"
                review_error = "Remaining unresolved issues remain and could not be repaired."
            if has_remaining_issues:
                review_status = "failed"
            break

        if round_num >= max_repair_rounds:
            # Perform final verification audit on the repaired translation
            console.print(f"[cyan]🔍 Final verification audit of repaired translation (after {round_num} rounds)...[/cyan]")
            lines_data = []
            for _, row in df_working.iterrows():
                lines_data.append({
                    "line_id": str(row.get("LineID", "")),
                    "speaker": str(row.get("Speaker", "")),
                    "source": str(row.get("Source", "")),
                    "translation": str(row.get("Translation", "")),
                })
            prompt = get_translation_review_prompt(lines_data, profiles_text, topic)
            try:
                final_audit = ask_gpt(
                    prompt,
                    response_json=True,
                    valid_def=validate_review_response,
                    log_title="translation_semantic_review_final_verification",
                )
            except Exception as e:
                review_status = "failed"
                review_error = f"Final verification audit failed: {e}"
                break

            val = validate_review_response(final_audit)
            if val["status"] != "success":
                review_status = "failed"
                review_error = f"Final verification audit invalid: {val['message']}"
                break

            final_issues = final_audit.get("issues", []) if isinstance(final_audit, dict) else []
            all_issues.extend(final_issues)
            unresolved = []
            for iss in final_issues:
                sev = str(iss.get("severity", "") or "").lower()
                if sev in ("critical", "warning"):
                    unresolved.append(iss)
            if not unresolved:
                console.print("[green]✅ Final verification audit passed with no issues.[/green]")
                review_status = "pass"
                review_error = None
                break
            else:
                review_status = "failed"
                review_error = f"{len(unresolved)} unresolved issues remain in final verification audit: {unresolved[0].get('problem')}"
                console.print(f"[bold red]❌ Semantic review failed: {review_error}[/bold red]")
            break

        round_num += 1

    df = df_working
    wrong_language_lines = []
    for _, row in df.iterrows():
        trans_val = str(row.get("Translation", "") or "")
        if not is_valid_target_language(trans_val, target_lang):
            wrong_language_lines.append(str(row.get("LineID", "")))

    if wrong_language_lines and review_status == "pass":
        review_status = "failed"
        review_error = f"Translation contains {len(wrong_language_lines)} wrong-language line(s)"

    review_report = {
        "status": review_status,
        "error": review_error,
        "overall_summary": overall_summary,
        "total_lines_reviewed": len(df),
        "issues_found_count": len(all_issues),
        "repairs_applied_count": len(applied_repairs),
        "issues": all_issues,
        "applied_repairs": applied_repairs,
    }

    report_path.write_text(json.dumps(review_report, ensure_ascii=False, indent=2), encoding="utf-8")
    console.print(f"[green]💾 Saved translation review report → {report_path}[/green]")

    return df, review_report


def check_semantic_fidelity(
    orig_text: str,
    final_text: str,
) -> list[str]:
    """Check whether final_text preserves the actors, numbers, negation and core actions of orig_text."""
    issues = []
    from core.dubbing_rewrite import (
        _has_negation,
        _extract_numbers,
        _extract_negated_content,
        MAGNITUDES,
        FIRST_PERSON,
        SECOND_PERSON,
        THIRD_PERSON,
    )
    # 1. Negation preservation
    if _has_negation(orig_text) != _has_negation(final_text):
        issues.append(f"Negation inverted between reviewed translation '{orig_text}' and final text '{final_text}'")

    # 2. Number and magnitude preservation
    orig_nums = _extract_numbers(orig_text)
    rewr_nums = _extract_numbers(final_text)
    if orig_nums != rewr_nums:
        issues.append(f"Numbers changed from '{orig_text}' ({orig_nums}) to '{final_text}' ({rewr_nums})")

    orig_mags = {w for w in MAGNITUDES if re.search(r'\b' + w + r'\b', orig_text.lower())}
    rewr_mags = {w for w in MAGNITUDES if re.search(r'\b' + w + r'\b', final_text.lower())}
    if orig_mags != rewr_mags:
        issues.append(f"Magnitudes changed from '{orig_text}' ({orig_mags}) to '{final_text}' ({rewr_mags})")

    # 3. Actor / pronoun role ordering
    orig_tokens = set(re.findall(r'\b[a-zA-Z]+\b', orig_text.lower()))
    rewr_tokens = set(re.findall(r'\b[a-zA-Z]+\b', final_text.lower()))
    orig_has_1_and_2 = bool(orig_tokens & FIRST_PERSON) and bool(orig_tokens & SECOND_PERSON)
    rewr_has_1_and_2 = bool(rewr_tokens & FIRST_PERSON) and bool(rewr_tokens & SECOND_PERSON)
    is_passive = bool(re.search(r'\b(is|was|are|were|been|being)\s+\w+(?:ed|en|t|d)\s+by\b', final_text.lower()) or 'by' in rewr_tokens)
    if orig_has_1_and_2 and rewr_has_1_and_2:
        orig_1st_pos = min(i for i, t in enumerate(re.findall(r'\b[a-zA-Z]+\b', orig_text.lower())) if t in FIRST_PERSON)
        orig_2nd_pos = min(i for i, t in enumerate(re.findall(r'\b[a-zA-Z]+\b', orig_text.lower())) if t in SECOND_PERSON)
        rewr_1st_pos = min(i for i, t in enumerate(re.findall(r'\b[a-zA-Z]+\b', final_text.lower())) if t in FIRST_PERSON)
        rewr_2nd_pos = min(i for i, t in enumerate(re.findall(r'\b[a-zA-Z]+\b', final_text.lower())) if t in SECOND_PERSON)
        if not is_passive and (orig_1st_pos < orig_2nd_pos) != (rewr_1st_pos < rewr_2nd_pos):
            issues.append(f"Actor/pronoun roles inverted: '{orig_text}' vs '{final_text}'")

    # 4. Action reversal / contradiction
    opposite_actions = {
        ('open', 'close'), ('close', 'open'), ('open', 'shut'), ('shut', 'open'),
        ('buy', 'sell'), ('sell', 'buy'), ('give', 'take'), ('take', 'give'),
        ('pay', 'receive'), ('receive', 'pay'), ('pay', 'charge'),
        ('accept', 'refuse'), ('refuse', 'accept'), ('accept', 'reject'), ('reject', 'accept'),
        ('start', 'stop'), ('stop', 'start'), ('enter', 'leave'), ('leave', 'enter'),
        ('lock', 'unlock'), ('unlock', 'lock'), ('hide', 'reveal'), ('show', 'hide'),
        ('stop', 'continue'), ('continue', 'stop'), ('halt', 'continue'),
    }
    for a1, a2 in opposite_actions:
        if re.search(r'\b' + a1 + r'\b', orig_text.lower()) and re.search(r'\b' + a2 + r'\b', final_text.lower()) and not re.search(r'\b' + a2 + r'\b', orig_text.lower()):
            issues.append(f"Contradictory action: '{a1}' in reviewed translation was replaced by '{a2}' in final text")
    return issues


def audit_final_tts_tasks(
    tasks_df: pd.DataFrame,
    report_path: Path | str = "output/log/tts_tasks_semantic_audit.json",
    translation_results_path: str = "output/log/translation_results.xlsx",
) -> dict[str, Any]:
    """Audit finalized tts_tasks against original reviewed translation and semantic guardrails."""
    report_path = Path(report_path)
    tr_df = None
    issues = []
    if not os.path.isfile(translation_results_path):
        issues.append({
            "number": 0,
            "line_id": "",
            "severity": "critical",
            "problem": f"Reviewed translation file is missing or unreadable: {translation_results_path}"
        })
    else:
        try:
            tr_df = pd.read_excel(translation_results_path)
        except Exception as exc:
            issues.append({
                "number": 0,
                "line_id": "",
                "severity": "critical",
                "problem": f"Failed to load reviewed translation file {translation_results_path}: {exc}"
            })

    records = []
    target_lang = load_key("target_language", "en")
    for _, row in tasks_df.iterrows():
        number = int(row.get("number", 0))
        lid = str(row.get("line_id", ""))
        text = str(row.get("text", "")).strip()
        origin = str(row.get("origin", "")).strip()
        speaker = str(row.get("speaker", ""))
        rewritten = bool(row.get("rewritten_for_dubbing", False))
        reason = str(row.get("rewrite_reason", "") or "")

        reviewed_trans = None
        if tr_df is not None:
            if lid:
                sub_lids = [x.strip() for x in lid.split(",") if x.strip()]
                matched_trans = []
                for sub_lid in sub_lids:
                    match = tr_df[tr_df["LineID"] == sub_lid]
                    if not match.empty:
                        matched_trans.append(str(match.iloc[0].get("Translation", "")).strip())
                if matched_trans:
                    reviewed_trans = " ".join(matched_trans)
                else:
                    issues.append({
                        "number": number,
                        "line_id": lid,
                        "severity": "critical",
                        "problem": f"Missing reviewed translation evidence for line {lid}"
                    })
            else:
                reviewed_trans = ""

        diff = bool(reviewed_trans and reviewed_trans != text)
        rec = {
            "number": number,
            "line_id": lid,
            "speaker": speaker,
            "origin": origin,
            "final_tts_text": text,
            "reviewed_translation": reviewed_trans,
            "rewritten": rewritten or diff,
            "rewrite_reason": reason if (rewritten or diff) else "unchanged",
        }

        if not is_valid_target_language(text, target_lang):
            issues.append({
                "number": number,
                "line_id": lid,
                "severity": "critical",
                "problem": f"Final text is invalid for target language {target_lang}: {text}"
            })

        if reviewed_trans:
            semantic_errs = check_semantic_fidelity(reviewed_trans, text)
            for err in semantic_errs:
                issues.append({
                    "number": number,
                    "line_id": lid,
                    "severity": "critical",
                    "problem": err,
                })

        records.append(rec)

    audit_report = {
        "status": "pass" if not issues else "fail",
        "total_tasks": len(tasks_df),
        "rewritten_tasks_count": sum(r["rewritten"] for r in records),
        "issues_count": len(issues),
        "issues": issues,
        "lineage_records": records,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(audit_report, ensure_ascii=False, indent=2), encoding="utf-8")
    console.print(f"[green]💾 Saved final TTS tasks semantic audit → {report_path}[/green]")
    return audit_report
