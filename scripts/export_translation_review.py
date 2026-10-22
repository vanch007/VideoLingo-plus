"""Export a local, explicitly unaccepted dialogue/translation review page."""
from __future__ import annotations

import argparse
import html
import json
import os
import subprocess
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--preview", action="store_true", help="Render a visibly marked review copy with original audio")
    args = parser.parse_args()
    root = Path(args.workspace).resolve()
    os.chdir(root)
    from core.step6_generate_final_timeline import get_sentence_timestamps

    def read(name):
        path = root / "output/log" / name
        return json.loads(path.read_text()) if path.exists() else {}

    role_report = read("dialogue_role_review.json")
    role_map = {u["line_id"]: u for u in role_report.get("assignments", [])}
    cards = {c["character_id"]: c.get("label", c["character_id"]) for c in role_report.get("characters", [])}
    source_report = read("source_verification.json")
    review = read("translation_review.json")
    reference = read("reference_translation_assessment.json")
    from core.subtitle_quality import subtitle_quality_report
    quality = subtitle_quality_report(root / "output/log")
    candidate = root / "output/log/translation_results.xlsx"
    if not candidate.exists():
        candidate = root / "output/log/translation_candidate.xlsx"
    frame = pd.read_excel(candidate).fillna("")
    words = pd.read_excel(root / "output/log/cleaned_chunks.xlsx")
    timings = get_sentence_timestamps(words, frame)
    rows = []
    for (_, row), (start, end, _) in zip(frame.iterrows(), timings):
        lid = str(row.LineID)
        role = role_map.get(lid, {})
        uncertainties = [g for g in source_report.get("groups", []) if lid in g["line_ids"] and g["status"] != "pass"]
        issues = [i for i in review.get("issues", []) if i.get("line_id") == lid]
        if quality.get("reference_translation_quality") in {"pass", "fail"}:
            issues += [i for i in reference.get("findings", []) if lid in i.get("line_ids", [])]
        rows.append({"id": lid, "start": start, "end": end,
                     "source": str(row.Source), "translation": str(row.Translation),
                     "voice": str(row.Speaker), "character": cards.get(role.get("character_id"), "未确定"),
                     "relay": cards.get(role.get("relays_character"), "未确定"),
                     "actor": cards.get(role.get("action_agent"), "未确定"),
                     "confidence": role.get("confidence", "unknown") if role.get("decision") == "applied_inferred" else "未确定",
                     "proposal_confidence": role.get("confidence", "unknown"), "reason": role.get("reason", ""),
                     "needs_review": bool(uncertainties or issues or role.get("decision") != "applied_inferred"
                                          or role.get("relations_status") == "needs_review"),
                     "role_conflicts": role.get("consistency_conflicts", []),
                     "witness": [g.get("witness", "") for g in uncertainties],
                     "issues": [str(i.get("problem", "")) + ("（审校引用待核实）" if i.get("evidence_status") == "unverified" else "") for i in issues]})
    data = json.dumps(rows, ensure_ascii=False).replace("<", "\\u003c")
    body = []
    for index, row in enumerate(rows):
        esc = lambda key: html.escape(str(row[key]))
        notes = html.escape("；".join(row["issues"]))
        witnesses = "".join(f"<p>另一套语音识别：{html.escape(text)}</p>" for text in row["witness"])
        conflicts = html.escape("；".join(row["role_conflicts"]))
        body.append(f'''<tr data-index="{index}" data-review="{str(row['needs_review']).lower()}">
<td><button onclick="jump({index})">{row['start']:.1f} 秒</button><small>{esc('id')}</small></td>
<td>{esc('character')}<small>声纹分组 {esc('voice')} · {esc('confidence')}</small></td>
<td>{esc('source')}</td><td>{esc('translation')}</td>
<td>转述：{esc('relay')}<br>动作主体：{esc('actor')}<details><summary>判断依据与疑点</summary>{esc('reason')}<p>{notes}</p><p>{conflicts}</p>{witnesses}</details></td></tr>''')
    page = '''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>对白角色与译文复核</title><style>
body{font:15px/1.65 system-ui,sans-serif;background:#f5f6f8;color:#152536;margin:0;padding:28px}main{max-width:1500px;margin:auto}
h1{font-size:25px;margin:0}.note{background:#fff1d5;border-left:4px solid #ad7200;padding:12px 18px;margin:18px 0}
.player{max-width:780px;margin:22px auto;position:relative;background:#111;color:white}video{width:100%;display:block}
#caption{padding:12px 20px;min-height:32px;text-align:center;font-size:20px}#source{font-size:14px;color:#b9c5d5;text-align:center;padding:0 15px 12px}
table{width:100%;border-collapse:collapse;background:white}td,th{text-align:left;padding:13px;vertical-align:top;border-bottom:1px solid #dfe5ed}
th{background:#e9edf3}small{display:block;color:#697586;font-size:12px}tr.active{background:#e8f3ff}button{border:1px solid #b9c7db;border-radius:6px;padding:6px 10px;background:white;cursor:pointer}
details{font-size:12px;color:#647184}label{display:block;margin:15px 0}td:nth-child(3),td:nth-child(4){width:27%}</style>
<main><h1>对白角色与译文复核</h1><p>点击时间播放对应原声，核对说话者、转述对象和译文。候选译文尚未通过质量验收。</p>
<div class="note">此页是待复核候选。角色来自模型推断，识别分歧与审校问题保留在下表；不能以页面生成或模型置信度代替验收。</div>
<div class="player"><video id="video" src="output/source.mp4" controls preload="metadata"></video><div id="caption"></div><div id="source"></div></div>
<label><input id="filter" type="checkbox"> 只看有疑点的行</label><table><thead><tr><th>原声位置</th><th>对白角色</th><th>原文</th><th>候选译文</th><th>关系与依据</th></tr></thead><tbody>'''+"\n".join(body)+'''</tbody></table></main>
<script>const rows='''+data+''';const video=document.getElementById('video');
function jump(i){video.currentTime=rows[i].start;video.play();}
video.addEventListener('timeupdate',()=>{const i=rows.findIndex(r=>video.currentTime>=r.start&&video.currentTime<r.end);document.getElementById('caption').textContent=i<0?'':rows[i].translation;document.getElementById('source').textContent=i<0?'':rows[i].source;document.querySelectorAll('tbody tr').forEach((tr,n)=>tr.classList.toggle('active',n===i));});
document.getElementById('filter').addEventListener('change',e=>document.querySelectorAll('tbody tr').forEach(tr=>tr.hidden=e.target.checked&&tr.dataset.review!=='true'));
</script></html>'''
    (root / "review.html").write_text(page, encoding="utf-8")
    if quality.get("reference_translation_quality") == "fail":
        page = page.replace("此页是待复核候选。", "参考对白核验：不合格（fail）。此版本不适合交付。")
        (root / "review.html").write_text(page, encoding="utf-8")
    elif quality.get("reference_delivery_quality") == "fail":
        page = page.replace("此页是待复核候选。", "本次交付评估：不合格（fail）。参考语义通过不能解除显示、片尾或角色问题。")
        (root / "review.html").write_text(page, encoding="utf-8")
    (root / "review_rows.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.preview:
        from core.subtitle_layout import layout_subtitles
        from core.config_utils import load_key
        from core.step6_generate_final_timeline import convert_to_srt_format
        from core.pipeline.artifact_manifest import hash_path
        from core.step12_merge_dub_to_vid import get_ffmpeg_binary
        timeline = frame.copy()
        timeline["timestamp"] = [(start, end) for start, end, _ in timings]
        timeline["speaker"] = [speaker for _, _, speaker in timings]
        display, layout = layout_subtitles(
            timeline, words,
            chars_per_line=int(load_key("subtitle.max_length", 38)),
            max_lines=int(load_key("subtitle.max_lines", 2)),
            max_cps=float(load_key("subtitle.max_cps", 20)),
            min_duration=float(load_key("subtitle.min_duration", 0.8)),
            max_lead_in=float(load_key("subtitle.display_lead_in", 0.15)),
            max_linger=float(load_key("subtitle.display_linger", 0.65)))
        subtitles = "\n\n".join(f"{i + 1}\n{convert_to_srt_format(*r.timestamp)}\n{r.Translation}" for i, r in display.iterrows()) + "\n"
        (root / "review_preview.srt").write_text(subtitles, encoding="utf-8")
        (root / "review_subtitle_layout.json").write_text(json.dumps(layout, ensure_ascii=False, indent=2), encoding="utf-8")
        subprocess.run([get_ffmpeg_binary(), "-y", "-v", "error", "-i", str(root / "output/source.mp4"),
                        "-vf", "subtitles=review_preview.srt:force_style='FontName=Arial,Fontsize=13,MarginV=12',drawtext=text='DRAFT - NEEDS REVIEW':fontsize=14:fontcolor=yellow:box=1:boxcolor=black@0.65:x=10:y=10",
                        "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-c:a", "copy", "-movflags", "+faststart",
                        str(root / "review_preview.mp4")], check=True)
        (root / "review_preview.manifest.json").write_text(json.dumps({
            "status": quality["status"], "accepted": False, "purpose": "marked review copy; original source audio, no dubbing",
            "source_sha256": hash_path(root / "output/source.mp4"), "translation_sha256": hash_path(candidate),
            "roles_sha256": hash_path(root / "output/log/dialogue_role_review.json"),
            "preview_sha256": hash_path(root / "review_preview.mp4"), "subtitle_layout_status": layout["status"],
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"page": str(root / 'review.html'), "rows": len(rows),
                      "needs_review": sum(r['needs_review'] for r in rows), "quality": quality["status"]}))


if __name__ == "__main__":
    main()
