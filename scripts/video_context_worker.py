"""Visual context of a video with WeMM-Video (MLX), run in the WeMM runtime, not VideoLingo's.

Prints one JSON object with the raw model replies; VideoLingo parses and filters them
(core/video_context.py), so this worker carries no policy. It answers three questions:
  terms      - product/brand/person names and labels written on screen (spelling as shown)
  subtitles  - whether the video already carries burned-in subtitles, their language and position
  characters - who appears (name only when shown or clearly said), gender, appearance; scene summary
Spoken dialogue, speaker attribution and timing are deliberately not asked: the ASR owns them.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

TERMS_PROMPT = (
    "List every product name, brand, model name, person name, place name and UI label that is written on screen "
    "or clearly spoken in this video. Spell each exactly as it is written on screen when it is visible. "
    "Do not list whole sentences. Output JSON only: "
    '{"terms":[{"text":"...","source":"screen|speech|both"}]}'
)
SUBTITLES_PROMPT = (
    "Does this video already contain burned-in (hard-coded) subtitles or captions of the speech? "
    "Output JSON only: "
    '{"burned_in_subtitles":true|false,"language":"...","position":"top|middle|bottom|varies","style":"..."}'
)
CHARACTERS_PROMPT_ZH = (
    "请识别视频中出现的人物。只有在画面文字显示或对白清楚说出时才写名字，否则写 unknown；"
    "给出性别和外形特征，并用两句话概括场景与冲突。不要逐句转写对白，不要判断每句话是谁说的。只输出 JSON："
    '{"summary":"...","characters":[{"name":"...","gender":"male|female|unknown","appearance":"..."}]}'
)
CHARACTERS_PROMPT_EN = (
    "Identify the people who appear in this video. Give a name only when it is written on screen or clearly said, "
    "otherwise 'unknown'; give gender and appearance, and summarize the scene and conflict in two sentences. "
    "Do not transcribe the dialogue and do not attribute lines to speakers. Output JSON only: "
    '{"summary":"...","characters":[{"name":"...","gender":"male|female|unknown","appearance":"..."}]}'
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="mlx-wemm-video checkout")
    parser.add_argument("--model", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--source-language", default="auto")
    parser.add_argument("--max-frames", type=int, default=96)
    parser.add_argument("--av-max-seconds", type=float, default=180.0,
                        help="Longer videos are analyzed from frames only (audio tokens grow with length)")
    parser.add_argument("--total-pixels", type=int, default=1572864)
    parser.add_argument("--max-tokens", type=int, default=1500)
    args = parser.parse_args()

    sys.path.insert(0, args.root)
    from mlx_wemm_video.generate import load_wemm_model, stream_generate

    use_audio = args.duration <= args.av_max_seconds
    fps = min(1.0, args.max_frames / max(args.duration, 1.0))
    started = time.time()
    model, processor = load_wemm_model(args.model)
    load_seconds = time.time() - started
    characters_prompt = CHARACTERS_PROMPT_ZH if args.source_language.lower().startswith(("zh", "yue")) else CHARACTERS_PROMPT_EN

    replies, timings = {}, {}
    for name, prompt in (("terms", TERMS_PROMPT), ("subtitles", SUBTITLES_PROMPT), ("characters", characters_prompt)):
        content = [{"type": "video_audio" if use_audio else "video", "video": args.video, "fps": fps,
                    "min_frames": 4, "max_frames": args.max_frames, "total_pixels": args.total_pixels},
                   {"type": "text", "text": prompt}]
        t = time.time()
        text, last = [], {}
        for chunk in stream_generate(model, processor, [{"role": "user", "content": content}],
                                     max_tokens=args.max_tokens, temperature=0.0, use_audio_in_video=use_audio):
            text.append(chunk["text"])
            last = chunk
        replies[name] = "".join(text)
        timings[name] = {"seconds": round(time.time() - t, 2), "finish": last.get("finish_reason"),
                         "prompt_tokens": last.get("prompt_tokens"), "generated_tokens": last.get("generated_tokens")}

    print(json.dumps({"replies": replies, "timings": timings, "load_seconds": round(load_seconds, 2),
                      "used_audio": use_audio, "fps": fps}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
