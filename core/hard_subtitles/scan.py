import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from core.hard_subtitles.ocr import HardSubtitleOCRClient

def probe_video_metadata(video_path: str) -> Dict[str, Any]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0.0
    cap.release()
    return {
        "width": width,
        "height": height,
        "fps": fps,
        "total_frames": total_frames,
        "duration": duration,
        "is_portrait": height > width
    }

def detect_initial_subtitle_zone(video_path: str, client: HardSubtitleOCRClient, num_samples: int = 16, tmp_dir: str = "output/hard_subtitles/tmp") -> Dict[str, Any]:
    """
    Uniformly sample num_samples frames across video, run OCR to detect bottom subtitle bounding boxes.
    Returns aggregated subtitle bounding box in pixels and normalized coordinates [ymin, xmin, ymax, xmax].
    """
    meta = probe_video_metadata(video_path)
    duration = meta["duration"]
    width = meta["width"]
    height = meta["height"]
    
    os.makedirs(tmp_dir, exist_ok=True)
    timestamps = np.linspace(duration * 0.05, duration * 0.95, num_samples)
    
    cap = cv2.VideoCapture(video_path)
    fps = meta["fps"]
    
    bottom_boxes = []
    text_samples = []
    
    for i, ts in enumerate(timestamps):
        cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000.0)
        ret, frame = cap.read()
        if not ret:
            continue
        frame_file = os.path.join(tmp_dir, f"probe_{i:02d}.jpg")
        cv2.imwrite(frame_file, frame)
        
        try:
            res = client.recognize(frame_file, task="boxes")
            for r in res.get("regions", []):
                label = r.get("label", "").lower()
                bbox = r.get("bbox_2d", []) # [ymin, xmin, ymax, xmax] in 0-999
                content = r.get("content", "").strip()
                if label in ["text", "header"] and bbox and len(bbox) == 4 and content:
                    ymin, xmin, ymax, xmax = bbox
                    # Filter for bottom half of the screen
                    if ymin >= 600 or ymax >= 750:
                        bottom_boxes.append(bbox)
                        text_samples.append({"time": ts, "text": content, "bbox": bbox})
        except Exception as e:
            print(f"[Warn] Probe at {ts:.2f}s failed: {e}")
        finally:
            if os.path.exists(frame_file):
                os.remove(frame_file)
                
    cap.release()
    
    if not bottom_boxes:
        return {
            "has_hard_subtitles": False,
            "confidence": 0.0,
            "bbox_norm": None,
            "bbox_px": None,
            "metadata": meta,
            "samples": []
        }
    
    boxes_np = np.array(bottom_boxes)
    # ymin: 10th percentile, ymax: 90th percentile with safety padding
    ymin_norm = max(0, int(np.percentile(boxes_np[:, 0], 10)) - 15)
    xmin_norm = max(0, int(np.percentile(boxes_np[:, 1], 5)) - 25)
    ymax_norm = min(999, int(np.percentile(boxes_np[:, 2], 90)) + 20)
    xmax_norm = min(999, int(np.percentile(boxes_np[:, 3], 95)) + 25)
    
    # Map to pixels [x, y, w, h]
    y_px = int(ymin_norm * height / 1000.0)
    x_px = int(xmin_norm * width / 1000.0)
    h_px = int((ymax_norm - ymin_norm) * height / 1000.0)
    w_px = int((xmax_norm - xmin_norm) * width / 1000.0)
    
    # Ensure even dimensions for ffmpeg crop/blur filters
    if x_px % 2 != 0: x_px -= 1
    if y_px % 2 != 0: y_px -= 1
    if w_px % 2 != 0: w_px += 1
    if h_px % 2 != 0: h_px += 1
    
    return {
        "has_hard_subtitles": True,
        "confidence": len(bottom_boxes) / float(num_samples),
        "bbox_norm": [ymin_norm, xmin_norm, ymax_norm, xmax_norm],
        "bbox_px": [x_px, y_px, w_px, h_px],
        "metadata": meta,
        "sample_count": len(bottom_boxes),
        "samples": text_samples
    }

def scan_subtitle_intervals(
    video_path: str,
    bbox_px: List[int],
    start_time: float = 0.0,
    end_time: Optional[float] = None,
    sample_rate: float = 4.0, # 4 checks per second
    diff_thresh: float = 12.0
) -> List[Dict[str, Any]]:
    """
    Scans the subtitle ROI across time using frame differencing to identify
    temporal intervals [start, end] where subtitles appear and stay relatively constant.
    Returns raw event segments with representative timestamps.
    """
    cap = cv2.VideoCapture(video_path)
    meta = probe_video_metadata(video_path)
    duration = meta["duration"] if end_time is None else min(end_time, meta["duration"])
    fps = meta["fps"]
    x, y, w, h = bbox_px
    
    step_sec = 1.0 / sample_rate
    t = start_time
    
    prev_roi_gray = None
    intervals = []
    curr_interval = None
    
    while t < duration:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ret, frame = cap.read()
        if not ret:
            break
            
        roi = frame[y:y+h, x:x+w]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # Apply slight blur to reduce video compression noise
        gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Check if ROI has text-like high frequency content (edges)
        edges = cv2.Canny(gray_blur, 50, 150)
        edge_density = np.sum(edges > 0) / float(w * h)
        has_text = edge_density > 0.012 # threshold for presence of subtitles/strokes
        
        if prev_roi_gray is not None:
            diff = cv2.absdiff(gray_blur, prev_roi_gray)
            mean_diff = np.mean(diff)
        else:
            mean_diff = 100.0
            
        if has_text:
            if curr_interval is None:
                curr_interval = {
                    "start": t,
                    "end": t + step_sec,
                    "rep_time": t,
                    "rep_edge": edge_density
                }
            else:
                if mean_diff > diff_thresh:
                    # Content inside subtitle area changed significantly -> new subtitle line
                    curr_interval["end"] = t
                    if curr_interval["end"] - curr_interval["start"] >= 0.35:
                        intervals.append(curr_interval)
                    curr_interval = {
                        "start": t,
                        "end": t + step_sec,
                        "rep_time": t,
                        "rep_edge": edge_density
                    }
                else:
                    curr_interval["end"] = t + step_sec
                    if edge_density > curr_interval["rep_edge"]:
                        curr_interval["rep_time"] = t
                        curr_interval["rep_edge"] = edge_density
        else:
            if curr_interval is not None:
                curr_interval["end"] = t
                if curr_interval["end"] - curr_interval["start"] >= 0.35:
                    intervals.append(curr_interval)
                curr_interval = None
                
        prev_roi_gray = gray_blur
        t += step_sec
        
    if curr_interval is not None and curr_interval["end"] - curr_interval["start"] >= 0.35:
        intervals.append(curr_interval)
        
    cap.release()
    return intervals

