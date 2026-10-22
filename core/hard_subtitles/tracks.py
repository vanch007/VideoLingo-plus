import os
import re
import cv2
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from core.hard_subtitles.ocr import HardSubtitleOCRClient

def is_japanese(text: str) -> bool:
    # Contains Hiragana or Katakana
    return bool(re.search(r'[぀-ゟ゠-ヿ]', text))

def is_chinese(text: str) -> bool:
    # Contains CJK unified ideographs without Kana
    has_cjk = bool(re.search(r'[一-鿿]', text))
    has_kana = bool(re.search(r'[぀-ゟ゠-ヿ]', text))
    return has_cjk and not has_kana

def extract_track_events(
    video_path: str,
    intervals: List[Dict[str, Any]],
    bbox_px: List[int],
    client: HardSubtitleOCRClient,
    tmp_dir: str = "output/hard_subtitles/tmp"
) -> List[Dict[str, Any]]:
    """
    For each interval, extract representative frame and run OCR to obtain:
    - Text content
    - Language classification (Japanese, Chinese, Bilingual, Unknown)
    - Specific bounding box
    """
    os.makedirs(tmp_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    events = []
    
    for idx, inv in enumerate(intervals):
        rep_t = inv["rep_time"]
        cap.set(cv2.CAP_PROP_POS_MSEC, rep_t * 1000.0)
        ret, frame = cap.read()
        if not ret:
            continue
            
        frame_file = os.path.join(tmp_dir, f"event_{idx:03d}.jpg")
        cv2.imwrite(frame_file, frame)
        
        try:
            res = client.recognize(frame_file, task="boxes")
            matched_regions = []
            combined_text = []
            
            for r in res.get("regions", []):
                label = r.get("label", "").lower()
                bbox = r.get("bbox_2d", []) # [ymin, xmin, ymax, xmax] in 0-999
                content = r.get("content", "").strip()
                if label in ["text", "header"] and bbox and content:
                    ymin, xmin, ymax, xmax = bbox
                    if ymin >= 600 or ymax >= 750:
                        matched_regions.append(r)
                        combined_text.append(content)
                        
            full_str = " ".join(combined_text)
            lang = "unknown"
            if is_japanese(full_str) and is_chinese(full_str):
                lang = "bilingual_ja_zh"
            elif is_japanese(full_str):
                lang = "ja"
            elif is_chinese(full_str):
                lang = "zh"
                
            events.append({
                "id": idx,
                "start": inv["start"],
                "end": inv["end"],
                "rep_time": rep_t,
                "text": full_str,
                "language": lang,
                "regions": matched_regions,
                "bbox_px": bbox_px
            })
        except Exception as e:
            print(f"[Warn] Event {idx} OCR failed: {e}")
        finally:
            if os.path.exists(frame_file):
                os.remove(frame_file)
                
    cap.release()
    return events

