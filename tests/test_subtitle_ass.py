import subprocess

import numpy as np
import pytest

from core.subtitle_ass import TrackStyle, ass_colour, build_ass, parse_srt


def test_colour_and_srt_parsing():
    assert ass_colour("&H00FFFF") == "&H0000FFFF"
    assert ass_colour("&H33000000") == "&H33000000"
    assert ass_colour("#FFCC00") == "&H0000CCFF"
    assert ass_colour("nonsense") == "&H00FFFFFF"
    cues = parse_srt("1\n00:00:01,500 --> 00:00:03,000\nHello\nthere\n\n2\n00:00:03,000 --> 00:00:04,250\nBye\n")
    assert cues == [(1.5, 3.0, "Hello\nthere"), (3.0, 4.25, "Bye")]


def _events(ass, style):
    return [l for l in ass.splitlines() if l.startswith("Dialogue:") and f",{style}," in l]


def test_source_margin_follows_the_translation_rows_shown_and_text_is_conserved():
    trans = TrackStyle("Arial", 17, margin_v=23)
    src = TrackStyle("Arial", 13, margin_v=23)
    long_zh = "然后在我睡着的时候自动剪辑完成，还使用了 Epidemic Sounds 的音乐和 FFmpeg 处理，真的非常离谱。"
    ass = build_ass(1920, 1080, [(0, 4, "I made this video while I was asleep.")],
                    [(0, 2, "很短。"), (2, 4, long_zh)], src, trans, (22, 42))
    src_events = _events(ass, "Src")
    assert len(src_events) == 2                                   # cut where the translation changes
    margins = [int(e.split(",")[7]) for e in src_events]
    assert margins[1] > margins[0] > trans.margin_v               # more rows below -> source sits higher
    body = "".join(e.split(",,")[-1] for e in _events(ass, "Trans"))
    assert body.replace("\\N", "").replace(" ", "") == ("很短。" + long_zh).replace(" ", "")   # nothing cut
    assert "WrapStyle: 2" in ass and "PlayResY: 288" in ass and "PlayResX: 512" in ass


def test_translation_only_has_no_source_style():
    ass = build_ass(1920, 1080, [], [(0, 1, "你好")], None, TrackStyle("Arial", 17), (22, 42))
    assert "Style: Src" not in ass and not _events(ass, "Src")


def _libass_ffmpeg():
    from core.step12_merge_dub_to_vid import get_ffmpeg_binary
    exe = get_ffmpeg_binary()
    out = subprocess.run([exe, "-hide_banner", "-filters"], capture_output=True, text=True).stdout
    return exe if " ass " in out else None


@pytest.mark.skipif(_libass_ffmpeg() is None, reason="ffmpeg with libass required")
def test_rendered_tracks_never_overlap(tmp_path):
    import cv2
    from core.subtitle_ass import ass_filter
    exe = _libass_ffmpeg()
    trans = TrackStyle("Arial", 17, "&H00FFFF", margin_v=23)
    src = TrackStyle("Arial", 13, "&HFFFFFF", margin_v=23)
    zh = "然后在我睡着的时候自动剪辑完成，还使用了 Epidemic Sounds 的音乐和 FFmpeg 处理，真的非常离谱。"
    en = "I made this entire video using Claude Code and then edited it automatically while I was asleep."
    masks = {}
    for name, s_cues, t_cues in (("trans", [], [(0, 2, zh)]), ("both", [(0, 2, en)], [(0, 2, zh)])):
        ass = tmp_path / f"{name}.ass"
        ass.write_text(build_ass(1920, 1080, s_cues, t_cues, src if s_cues else None, trans, (22, 42)), encoding="utf-8")
        png = tmp_path / f"{name}.png"
        subprocess.run([exe, "-y", "-v", "error", "-f", "lavfi", "-i", "color=c=0x303a4a:s=1920x1080:d=2:r=5",
                        "-vf", ass_filter(str(ass)), "-frames:v", "1", str(png)], check=True)
        img = cv2.imread(str(png)).astype(int)
        masks[name] = np.abs(img - np.array([0x4a, 0x3a, 0x30])).sum(axis=2) > 60
    trans_rows = np.where(masks["trans"].any(axis=1))[0]
    extra = masks["both"] & ~masks["trans"]            # pixels that only the source contributes
    src_rows = np.where(extra.any(axis=1))[0]
    assert src_rows.size and src_rows.max() < trans_rows.min()      # the source ends above the first translation row
    inside = masks["both"][trans_rows.min():trans_rows.max() + 1]
    assert (inside == masks["trans"][trans_rows.min():trans_rows.max() + 1]).all()   # nothing intrudes on the translation
