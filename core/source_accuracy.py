"""Reference-based source ASR evaluation, independent of translation output."""
from __future__ import annotations

from core.source_verification import normalize_transcript


def edit_distance(reference: str, hypothesis: str) -> int:
    previous = list(range(len(hypothesis) + 1))
    for i, left in enumerate(reference, 1):
        current = [i]
        for j, right in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (left != right)))
        previous = current
    return previous[-1]


def transcript_in_window(result, start, end):
    parts = []
    for segment in result.get("segments", []):
        for item in segment.get("words") or [segment]:
            midpoint = (float(item["start"]) + float(item["end"])) / 2
            if start <= midpoint <= end:
                parts.append(str(item.get("word", item.get("text", ""))))
    return "".join(parts)


def closest_reference_span(reference, context):
    """Align a reference excerpt inside nearby ASR context, excluding neighbors.

    A word at a 200ms window edge must not become a lexical omission. Timing
    accuracy remains a separate measurement; internal insertions still count.
    """
    previous = [(0, i) for i in range(len(context) + 1)]
    for i, left in enumerate(reference, 1):
        current = [(i, 0)]
        for j, right in enumerate(context, 1):
            candidates = [(current[-1][0] + 1, current[-1][1]),
                          (previous[j][0] + 1, previous[j][1]),
                          (previous[j - 1][0] + (left != right), previous[j - 1][1])]
            current.append(min(candidates))
        previous = current
    end = min(range(len(previous)), key=lambda j: (previous[j][0], abs(j - previous[j][1] - len(reference))))
    distance, start = previous[end]
    return distance, context[start:end]


def evaluate_source_checkpoints(result, reference):
    """Measure character errors and explicit source facts in supplied windows.

    The reference is external data, never an ASR-generated answer. Report its
    scope; character accuracy on selected windows is not whole-file accuracy.
    """
    if not reference.get("checkpoints"):
        raise ValueError("Source reference requires at least one explicit checkpoint")
    checks, total_distance, total_chars = [], 0, 0
    for item in reference["checkpoints"]:
        observed = transcript_in_window(result, float(item["start"]), float(item["end"]))
        margin = float(reference.get("context_margin_seconds", 1.0))
        context = normalize_transcript(transcript_in_window(result, float(item["start"]) - margin, float(item["end"]) + margin))
        variants = [item["text"], *item.get("alternatives", [])]
        expected = min((normalize_transcript(v) for v in variants), key=lambda v: closest_reference_span(v, context)[0])
        distance, hypothesis = closest_reference_span(expected, context)
        missing = [fact for fact in item.get("required_facts", []) if normalize_transcript(fact) not in hypothesis]
        checks.append({**item, "observed": observed, "matched_text": hypothesis, "errors": distance, "reference_characters": len(expected),
                       "cer": distance / max(1, len(expected)), "missing_facts": missing,
                       "status": "fail" if missing else ("pass" if distance == 0 else "minor_text_difference")})
        total_distance += distance
        total_chars += len(expected)
    cer = total_distance / max(1, total_chars)
    passed = not any(c["missing_facts"] for c in checks) and cer <= float(reference.get("max_cer", 0.05))
    return {"status": "pass" if passed else "fail", "basis": "external_reference_checkpoints_not_model_self_grade",
            "method": "time-anchored excerpt text alignment with 1s context; not a timestamp quality score",
            "scope": reference["scope"], "reference_characters": total_chars,
            "character_errors": total_distance, "cer": cer, "critical_fact_failures": sum(bool(c["missing_facts"]) for c in checks),
            "checkpoints": checks}
