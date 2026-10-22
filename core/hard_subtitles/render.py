import os
from typing import List, Dict, Any, Optional

def build_mask_plan(
    events: List[Dict[str, Any]],
    default_bbox_px: List[int],
    mode: str = "blur",
    blur_radius: int = 24,
    blur_power: int = 3,
    mosaic_size: int = 16
) -> Dict[str, Any]:
    """
    Builds mask intervals and bounding boxes for video rendering.
    Supports 'blur', 'mosaic', and 'solid' (black bar/color bar).
    """
    intervals = []
    for ev in events:
        intervals.append({
            "start": ev["start"],
            "end": ev["end"],
            "bbox_px": ev.get("bbox_px", default_bbox_px)
        })
        
    return {
        "mode": mode,
        "default_bbox_px": default_bbox_px,
        "blur_radius": blur_radius,
        "blur_power": blur_power,
        "mosaic_size": mosaic_size,
        "intervals": intervals
    }

def generate_ffmpeg_mask_filter(mask_plan: Dict[str, Any]) -> str:
    """
    Generates an FFmpeg complex or video filter expression.
    Uses 'enable' expressions for temporal activation of crop/boxblur/overlay.
    
    Structure:
    [v_in]split[base][to_mask];
    [to_mask]crop=w:h:x:y,boxblur=r:p[masked];
    [base][masked]overlay=x:y:enable='between(t,s1,e1)+between(t,s2,e2)...'[v_out]
    """
    intervals = mask_plan.get("intervals", [])
    if not intervals:
        return ""
        
    bbox = mask_plan["default_bbox_px"]
    x, y, w, h = bbox
    mode = mask_plan.get("mode", "blur")
    
    # Build temporal enable expression
    enable_terms = []
    for inv in intervals:
        s = inv["start"]
        e = inv["end"]
        enable_terms.append(f"between(t,{s:.3f},{e:.3f})")
        
    enable_expr = "+".join(enable_terms) if enable_terms else "0"
    
    if mode == "mosaic":
        ms = mask_plan.get("mosaic_size", 16)
        # downscale then upscale with nearest neighbor
        filter_str = (
            f"split[v_base][v_crop];"
            f"[v_crop]crop={w}:{h}:{x}:{y},scale=iw/{ms}:ih/{ms}:flags=neighbor,scale={w}:{h}:flags=neighbor[v_masked];"
            f"[v_base][v_masked]overlay={x}:{y}:enable='{enable_expr}'"
        )
    elif mode == "solid":
        # solid dark box
        filter_str = (
            f"drawbox=x={x}:y={y}:w={w}:h={h}:color=black@0.9:t=fill:enable='{enable_expr}'"
        )
    else: # default blur
        r = mask_plan.get("blur_radius", 20)
        p = mask_plan.get("blur_power", 3)
        filter_str = (
            f"split[v_base][v_crop];"
            f"[v_crop]crop={w}:{h}:{x}:{y},boxblur={r}:{p}[v_masked];"
            f"[v_base][v_masked]overlay={x}:{y}:enable='{enable_expr}'"
        )
        
    return filter_str

