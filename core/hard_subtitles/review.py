import json
import difflib
from typing import List, Dict, Any, Optional

def compute_similarity(s1: str, s2: str) -> float:
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    return difflib.SequenceMatcher(None, s1, s2).ratio()

def match_ocr_with_asr(
    ocr_events: List[Dict[str, Any]],
    asr_rows: List[Dict[str, Any]],
    time_slack: float = 1.0
) -> Dict[str, Any]:
    """
    Compare OCR events with ASR transcript rows.
    Identifies discrepancies, entity corrections, and alignment hints.
    """
    matches = []
    discrepancies = []
    
    for ocr in ocr_events:
        ocr_text = ocr.get("text", "").strip()
        if not ocr_text:
            continue
            
        o_start = ocr["start"]
        o_end = ocr["end"]
        
        # Find overlapping ASR rows
        overlapping_asr = []
        for asr in asr_rows:
            a_start = float(asr.get("start", asr.get("orig_start", 0.0)))
            a_end = float(asr.get("end", asr.get("orig_end", 0.0)))
            
            # Check overlap or proximity
            if max(o_start, a_start) <= min(o_end, a_end) + time_slack:
                overlapping_asr.append(asr)
                
        asr_text_combined = " ".join([r.get("text", "") for r in overlapping_asr]).strip()
        sim = compute_similarity(ocr_text, asr_text_combined)
        
        match_item = {
            "ocr_id": ocr["id"],
            "ocr_time": [o_start, o_end],
            "ocr_text": ocr_text,
            "ocr_language": ocr.get("language", "unknown"),
            "asr_overlapping_count": len(overlapping_asr),
            "asr_text": asr_text_combined,
            "similarity": round(sim, 3)
        }
        matches.append(match_item)
        
        if len(overlapping_asr) > 0 and sim < 0.70:
            # Significant discrepancy
            discrepancies.append({
                "ocr_id": ocr["id"],
                "ocr_time": [o_start, o_end],
                "ocr_text": ocr_text,
                "asr_text": asr_text_combined,
                "similarity": round(sim, 3),
                "suggestion": "Review term differences between on-screen text and ASR audio"
            })
            
    return {
        "total_ocr_events": len(ocr_events),
        "total_asr_rows": len(asr_rows),
        "matches": matches,
        "discrepancies_count": len(discrepancies),
        "discrepancies": discrepancies
    }

