"""One identity and one clock for source dialogue, translation and synthesis.

Parent dialogue is retained for interpretation. Synthesis units are exact
source slices at punctuation or measured pauses, never a reworded transcript.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

import pandas as pd

from core.config_utils import load_key
from core.source_verification import normalize_transcript

PLAN_PATH = Path("output/log/utterance_plan.json")


def plan_enabled() -> bool:
    if not bool(load_key("translation.utterance_plan", False)) or not Path("output/source.mp4").is_file():
        return False
    state = Path("output/pipeline_state.json")
    return not (state.is_file() and json.loads(state.read_text()).get("subtitle_only", False))


def signature(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    allow_nan=False).encode()).hexdigest()


def plan_active() -> bool:
    """Legacy artifacts retain their route; CLI provenance forbids stale reuse."""
    return plan_enabled() and PLAN_PATH.is_file()


def save_plan(plan: dict, path: Path = PLAN_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(plan, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def load_plan(path: Path = PLAN_PATH) -> dict:
    plan = json.loads(path.read_text(encoding="utf-8"))
    validate_plan(plan)
    return plan


def validate_plan(plan: dict) -> None:
    units = plan.get("units", [])
    if plan.get("version") != 1 or not units:
        raise ValueError("Missing or unsupported utterance plan")
    ids, previous_end = set(), -1.0
    by_parent = {}
    for unit in units:
        uid = unit["unit_id"]
        start, end = float(unit["start"]), float(unit["end"])
        if uid in ids or not unit["source"].strip():
            raise ValueError(f"Duplicate or empty utterance {uid}")
        if not all(math.isfinite(t) for t in (start, end)) or start < 0 or end <= start or start < previous_end - .002:
            raise ValueError(f"Invalid/overlapping utterance clock at {uid}")
        source_start = float(unit.get("source_start", start))
        source_end = float(unit["source_end"])
        if not math.isfinite(source_start + source_end) or not start <= source_start < source_end <= end + .002:
            raise ValueError(f"Source word clock changed at {uid}")
        if source_start > start + .002:
            pause = unit.get("lead_pause_span")
            limit = float(plan.get("policy", {}).get("max_lead_in_seconds", 0.))
            if (source_start - start > limit + .002 or not pause or
                    float(pause[0]) > start + .002 or float(pause[1]) < source_start - .002):
                raise ValueError(f"Dubbing lead lacks measured quiet source evidence at {uid}")
        ids.add(uid)
        previous_end = end
        slices = unit.get("source_slices") or [{"line_id":unit["parent_line_id"], "source":unit["source"]}]
        if "".join(part["source"] for part in slices).replace(" ", "") != unit["source"].replace(" ", ""):
            raise ValueError(f"Utterance lineage changed source text at {uid}")
        for part in slices:
            by_parent.setdefault(part["line_id"], []).append(part["source"])
    parents = {p["line_id"]: p["source"] for p in plan["parents"]}
    if set(by_parent) != set(parents):
        raise ValueError("Utterance parents do not cover the source dialogue")
    for lid, parts in by_parent.items():
        if "".join(parts).replace(" ", "") != parents[lid].replace(" ", ""):
            raise ValueError(f"Utterance slices lost source text or punctuation at {lid}")


def build_plan(parents: pd.DataFrame, words: pd.DataFrame, *, max_seconds=12.,
               min_seconds=.8, gap_extension=.35, lead_extension=.12,
               quiet_spans=None, pause_identity=None, short_turn_gap=.08) -> dict:
    """Use an exact sequential character cursor; stale/fuzzy source fails closed."""
    if (max_seconds <= min_seconds or min_seconds <= 0 or gap_extension < 0
            or not 0 <= lead_extension <= .12 or not 0 <= short_turn_gap <= .45):
        raise ValueError("Invalid utterance segmentation limits")
    from core.providers.speaker_diarization import normalize_speaker
    word_records = [{"text": normalize_transcript(str(r.text).strip().strip('"')),
                     "start": float(r.start), "end": float(r.end),
                     "speaker": normalize_speaker(getattr(r, "speaker", None))} for r in words.itertuples()]
    if any(b['start'] < a['end'] - .002 for a, b in zip(word_records, word_records[1:])):
        raise ValueError("Overlapping source word clock; normalize or realign ASR before utterance planning")
    if not word_records or any(not w["text"] or not math.isfinite(w["start"]) or
                               not math.isfinite(w["end"]) or w["end"] <= w["start"] for w in word_records):
        raise ValueError("Utterance planning requires complete measured word timing")
    full, ranges, cursor = "", [], 0
    for w in word_records:
        ranges.append((cursor, cursor + len(w["text"])))
        full += w["text"]
        cursor += len(w["text"])
    wanted = "".join(normalize_transcript(str(t)) for t in parents.text)
    if full != wanted:
        raise ValueError("Source/word identity differs; repair ASR and realign before translation")
    owner = [i for i, w in enumerate(word_records) for _ in w["text"]]
    units, parent_records, offset = [], [], 0
    for row in parents.to_dict("records"):
        source, lid = str(row["text"]), str(row["line_id"])
        normalized = normalize_transcript(source)
        # Raw offsets include original punctuation/spaces. Every cut must also
        # be a complete aligned-word boundary (no split inside a name/token).
        counts = [len(normalize_transcript(source[:i])) for i in range(len(source) + 1)]
        valid = {b - offset for _, b in ranges if offset < b < offset + len(normalized)}
        cuts = {0: "parent", len(source): "parent"}
        for match in re.finditer(r"[。！？!?；;，,]|\.(?=\s|$)", source):
            at = match.end()
            if counts[at] in valid:
                # English decimal points are not clause boundaries.
                cuts[at] = "sentence" if match.group() in "。！？!?；;." else "clause"
        for at in range(1, len(source)):
            pos = counts[at]
            if pos not in valid or counts[at - 1] == pos:
                continue
            left, right = owner[offset + pos - 1], owner[offset + pos]
            if word_records[right]["start"] - word_records[left]["end"] >= .45:
                cuts.setdefault(at, "measured_pause")

        def clock(left, right):
            a, b = owner[offset + counts[left]], owner[offset + counts[right] - 1]
            return word_records[a]["start"], word_records[b]["end"], a, b

        def voice(left, right):
            _, _, first, last = clock(left, right)
            labels = {word_records[i]["speaker"] for i in range(first, last + 1)}
            return (labels.pop() if len(labels) == 1 else None) if "speaker" in words else normalize_speaker(row.get("speaker"))

        # Keep questions apart unless repeated sub-second phrases belong to
        # this same source turn. Plan that repetition before translation so
        # English can retain the rhythm without fitting each fragment alone.
        # Commas split only units above the synthesis limit.
        primary = [0, *sorted(i for i, kind in cuts.items() if kind == "sentence"), len(source)]
        spans = []
        def partition(left, right):
            start, end, _, _ = clock(left, right)
            if end - start <= max_seconds:
                spans.append((left, right, cuts.get(right, "parent")))
                return
            choices = [at for at in cuts if left < at < right and
                       counts[left] < counts[at] < counts[right]]
            if not choices:
                # Fallback: find any valid word boundary between left and right closest to midpoint
                word_bounds = [at for at in range(left + 1, right) if counts[at] in valid and counts[left] < counts[at] < counts[right]]
                if word_bounds:
                    choices = word_bounds
                else:
                    # If still no boundary, span cannot be subdivided further, accept as is
                    spans.append((left, right, cuts.get(right, "parent")))
                    return
            # Prefer balanced complete clauses, rather than a tiny trailing word.
            at = min(choices, key=lambda p: abs(clock(left, p)[1] - (start + end) / 2))
            partition(left, at)
            partition(at, right)
        for left, right in zip(primary, primary[1:]):
            if normalize_transcript(source[left:right]):
                partition(left, right)
        grouped = []
        previous_phrase, previous_short = None, False
        for left, right, basis in spans:
            start, end, _, _ = clock(left, right)
            phrase = normalize_transcript(source[left:right])
            short = end - start < min_seconds
            if grouped and (short or previous_short):
                old_left, old_right, _ = grouped[-1]
                old_start, old_end, _, _ = clock(old_left, old_right)
                same_voice = voice(old_left, old_right) is not None and voice(old_left, old_right) == voice(left, right)
                if same_voice and start - old_end < .45 and end - old_start <= max_seconds:
                    grouped[-1] = (old_left, right, "rapid_repetition" if phrase == previous_phrase else "short_clause_continuation")
                else:
                    grouped.append((left, right, basis))
            else:
                grouped.append((left, right, basis))
            previous_phrase, previous_short = phrase, short
        spans = grouped
        parent_records.append({"line_id": lid, "source": source})
        for n, (left, right, basis) in enumerate(spans, 1):
            start, end, first, last = clock(left, right)
            speaker = voice(left, right)
            units.append({"unit_id": f"{lid}.{n:02d}", "parent_line_id": lid,
                          "parent_line_ids": [lid],
                          "source_slices": [{"line_id":lid, "source":source[left:right].strip()}],
                          "source": source[left:right].strip(), "parent_source": source,
                          "start": start, "source_start": start, "source_end": end, "end": end,
                          "speaker": speaker, "word_start": first, "word_end": last,
                          "split_basis": basis, "target": "", "status": "source_planned"})
        offset += len(normalized)
    # ASR/subtitle row boundaries do not end an uninterrupted spoken turn.
    # Share a short fragment's budget only with the same observed voice, never
    # across unknown identities, a pause, or the maximum synthesis window.
    grouped_units = []
    for unit in units:
        previous = grouped_units[-1] if grouped_units else None
        short = unit["source_end"] - unit["source_start"] < min_seconds
        previous_short = previous and previous["source_end"] - previous["source_start"] < min_seconds
        if (previous and previous["parent_line_ids"][-1] != unit["parent_line_id"]
                and (short or previous_short) and previous["speaker"] is not None
                and previous["speaker"] == unit["speaker"]
                and -.002 <= unit["source_start"] - previous["source_end"] <= short_turn_gap
                and unit["source_end"] - previous["source_start"] <= max_seconds):
            previous["unit_id"] += "__" + unit["unit_id"]
            previous["source_slices"].extend(unit["source_slices"])
            previous["parent_line_ids"].extend(unit["parent_line_ids"])
            previous["source"] += " " + unit["source"]
            previous["parent_source"] += " " + unit["parent_source"]
            previous.update(source_end=unit["source_end"], end=unit["end"], word_end=unit["word_end"],
                            split_basis="short_same_speaker_continuation")
        else:
            grouped_units.append(unit)
    units = grouped_units
    for i in range(1, len(units)):
        if units[i]["source_start"] < units[i - 1]["source_end"] - 0.001:
            mid = round((units[i - 1]["source_end"] + units[i]["source_start"]) / 2.0, 4)
            if mid > units[i - 1]["source_start"] + 0.01 and mid < units[i]["source_end"] - 0.01:
                units[i - 1]["source_end"] = mid
                units[i - 1]["end"] = mid
                units[i]["source_start"] = mid
                units[i]["start"] = mid
            else:
                units[i]["source_start"] = units[i - 1]["source_end"]
                units[i]["start"] = units[i - 1]["source_end"]
                if units[i]["source_end"] <= units[i]["source_start"]:
                    units[i]["source_end"] = units[i]["source_start"] + 0.05
                    units[i]["end"] = units[i]["source_end"]
    # A subtitle-sized fragment must not dictate an impossible speech slot.
    # Short source phrases can use a bounded, acoustically quiet lead-in. Their
    # original word anchors and the following actor's onset never move.
    for i, unit in enumerate(units):
        unit["lead_extension"] = 0.
        unit["lead_pause_span"] = None
        if unit["source_end"] - unit["source_start"] >= min_seconds:
            continue
        previous = units[i - 1]["source_end"] if i else 0.
        for pause in quiet_spans or []:
            if pause[0] <= unit["source_start"] <= pause[1] + .001:
                lead = min(lead_extension, max(0., unit["source_start"] - max(pause[0], previous + .02)))
                unit["start"] = round(unit["source_start"] - lead, 6)
                unit["lead_extension"] = round(lead, 6)
                unit["lead_pause_span"] = list(pause) if lead else None
                break
    for i, unit in enumerate(units):
        # Only measured silence can be borrowed; the following speaker never
        # shifts. Keep a small guard at the next audible source onset.
        following = units[i + 1]["start"] if i + 1 < len(units) else unit["source_end"]
        extension = min(gap_extension, max(0., following - unit["source_end"] - .02))
        unit["end"] = round(unit["source_end"] + extension, 6)
        unit["duration"] = round(unit["end"] - unit["start"], 6)
        unit["source_duration"] = round(unit["source_end"] - unit["source_start"], 6)
        unit["gap_extension"] = round(extension, 6)
    plan = {"version": 1, "status": "source_planned", "parents": parent_records, "units": units,
            "policy": {"max_unit_seconds": max_seconds, "min_unit_seconds": min_seconds,
                       "segmentation_revision": 6,
                       "max_short_turn_gap_seconds": short_turn_gap,
                       "max_lead_in_seconds": lead_extension,
                       "pause_evidence": pause_identity,
                       "max_gap_extension_seconds": gap_extension,
                       "timing_basis": "measured_source_words; observed_silence_only"}}
    plan["source_signature"] = signature({"parents": parent_records, "words": word_records,
                                         "policy": plan["policy"]})
    validate_plan(plan)
    return plan


def planned_rows(parents: pd.DataFrame, words: pd.DataFrame) -> pd.DataFrame:
    quiet, identity = [], None
    audio = Path("output/audio/vocal.wav")
    lead = float(load_key("translation.speech_budget.max_lead_in_seconds", .12))
    if lead and audio.is_file():
        from core.source_pause_budget import quiet_source_spans
        pauses = quiet_source_spans(audio, Path("output/log/source_pause_budget.json"))
        quiet, identity = pauses["quiet_spans"], pauses["identity"]
    fresh = build_plan(parents, words,
                       max_seconds=float(load_key("translation.speech_budget.max_unit_seconds", 12.)),
                       min_seconds=float(load_key("translation.speech_budget.min_unit_seconds", .8)),
                       gap_extension=float(load_key("translation.speech_budget.max_gap_extension_seconds", .35)),
                       lead_extension=lead, quiet_spans=quiet, pause_identity=identity,
                       short_turn_gap=float(load_key("translation.speech_budget.max_short_turn_gap_seconds", .08)))
    if PLAN_PATH.is_file():
        old = load_plan()
        if old["source_signature"] != fresh["source_signature"]:
            raise ValueError("Utterance source/timing policy changed; use a fresh run")
    else:
        save_plan(fresh)
    return pd.DataFrame([{**u, "line_id": u["unit_id"], "text": u["source"]}
                         for u in fresh["units"]])


def bind_translation(frame: pd.DataFrame) -> None:
    plan = load_plan()
    if list(frame.LineID) != [u["unit_id"] for u in plan["units"]]:
        raise ValueError("Translation/utterance identity mismatch")
    for unit, row in zip(plan["units"], frame.to_dict("records")):
        if row["Source"] != unit["source"] or not str(row["Translation"]).strip():
            raise ValueError(f"Translation changed source identity at {unit['unit_id']}")
        unit.update(target=str(row["Translation"]), status="translated")
        for field in ("Character", "AddressedTo", "RelaysCharacter", "ActionAgent", "ActionPatient", "RoleConfidence"):
            if field in row:
                unit[field] = row[field]
    plan["status"] = "translated"
    save_plan(plan)


def tasks_from_plan(plan: dict) -> pd.DataFrame:
    validate_plan(plan)
    if any(not u.get("target") or u.get("status") not in {"translated", "synthesized"} for u in plan["units"]):
        raise ValueError("Unaccepted translation cannot enter TTS")
    from core.timing_utils import seconds_to_srt_time
    records = []
    for i, u in enumerate(plan["units"]):
        gap = max(0., plan["units"][i + 1]["start"] - u["end"]) if i + 1 < len(plan["units"]) else 0.
        records.append({"number": i + 1, "unit_id": u["unit_id"], "line_id": u["unit_id"],
                        "parent_line_id": u["parent_line_id"], "text": u["target"], "origin": u["source"],
                        "speaker": u.get("speaker"), "character": u.get("Character", "unknown"),
                        "start_time": seconds_to_srt_time(u["start"]), "end_time": seconds_to_srt_time(u["end"]),
                        "source_start": u.get("source_start", u["start"]),
                        "source_end": u["source_end"], "duration": u["duration"],
                        "planned_lead_in": u.get("lead_extension", 0.),
                        "sub_times": [u["start"], u["end"]], "lines": [u["target"]], "src_lines": [u["source"]],
                        "gap": gap, "raw_gap": gap, "tolerance": 0., "tol_dur": u["duration"],
                        "available_duration": u["duration"], "target_duration": u["source_duration"],
                        "est_dur": u.get("estimated_speech_seconds", u["duration"]),
                        "if_too_fast": False, "cut_off": 1, "disable_native_fit": True})
    return pd.DataFrame(records)


def validate_tasks(frame: pd.DataFrame, plan: dict, *, generated=False) -> None:
    from core.dubbing_quality import normalize_lines, parse_list
    if "unit_id" not in frame or list(frame.unit_id) != [u["unit_id"] for u in plan["units"]]:
        raise ValueError("TTS tasks must correspond one-to-one to the utterance plan")
    for row, unit in zip(frame.to_dict("records"), plan["units"]):
        if (normalize_lines(row.get("lines")) != [unit["target"]] or
            normalize_lines(row.get("src_lines")) != [unit["source"]] or
            row.get("text") != unit["target"] or row.get("origin") != unit["source"]):
            raise ValueError(f"TTS/source mapping changed at {unit['unit_id']}")
        if parse_list(row["sub_times"]) != [unit["start"], unit["end"]]:
            raise ValueError(f"TTS clock changed at {unit['unit_id']}")
        if generated:
            times = parse_list(row.get("new_sub_times"))
            if len(times) != 1 or len(times[0]) != 2 or any(abs(float(a) - float(b)) > .002 for a, b in zip(times[0], [unit["start"], unit["end"]])):
                raise ValueError(f"Generated audio clock changed at {unit['unit_id']}")
