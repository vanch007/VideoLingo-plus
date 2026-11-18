import re
from core.subtitle_layout import compute_subtitle_burn_style, compute_subtitle_line_limits
from core.fast_subtitles import build_cues


def test_landscape_and_unknown_keep_default_limits():
    assert compute_subtitle_line_limits(None, None) == (22, 42)
    assert compute_subtitle_line_limits(1920, 1080) == (22, 42)
    assert compute_subtitle_line_limits(640, 360) == (22, 42)
    style = compute_subtitle_burn_style(1920, 1080, platform_system="Darwin")
    assert style["is_portrait"] is False
    assert style["font_size"] == 20
    assert style["margin_v"] == 18


def test_portrait_9_16_line_limits_and_burn_style():
    for w, h in [(608, 1080), (360, 640), (1080, 1920)]:
        cjk_cap, latin_cap = compute_subtitle_line_limits(w, h)
        assert cjk_cap == 14
        assert latin_cap == 24
        style = compute_subtitle_burn_style(w, h, platform_system="Darwin")
        assert style["is_portrait"] is True
        assert style["font_size"] == 12
        assert style["margin_v"] == 26
        assert style["margin_lr"] == 24
        assert style["font_name"] == "PingFang SC"


def test_portrait_cues_wrap_short_lines_without_orphan_punctuation():
    rows = [{
        "number": 1,
        "unit_id": "U0001",
        "origin": "Because OmniSend now connects directly to Claude via MCP.",
        "text": "这是因为OmniSend现在能通过MCP直接连接Claude。",
        "final_start": 0.0,
        "final_end": 4.0,
    }]
    cues = build_cues(rows, width=608, height=1080)
    for cue in cues:
        for line in cue["tgt"].splitlines():
            assert len(line) <= 14
            assert not re.match(r"^[，。？！；：、,.?!;:]", line)
        for line in cue["src"].splitlines():
            assert len(line) <= 24
    norm = lambda s: re.sub(r"\s+", "", s)
    assert norm("".join(c["tgt"] for c in cues)) == norm(rows[0]["text"])
    assert norm("".join(c["src"] for c in cues)) == norm(rows[0]["origin"])
