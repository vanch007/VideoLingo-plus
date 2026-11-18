import re
import pytest
from core.subtitle_measure import atoms, break_lines, text_width, char_width, LATIN_EM
from core.fast_subtitles import build_cues

norm = lambda s: re.sub(r"\s+", "", s)


def test_atoms_concatenate_back_and_keep_names_whole():
    text = "我想用 Claude Code 和 Epidemic sounds，还有FFmpeg。"
    parts = atoms(text)
    assert norm("".join(parts)) == norm(text)
    assert any("Claude Code" in p for p in parts)
    assert any("Epidemic sounds" in p for p in parts)
    assert not any(p.strip().startswith(("，", "。")) for p in parts)


def test_latin_words_never_torn_and_rows_balanced():
    text = "I made this entire video using Claude Code and then edited it automatically while I was asleep."
    lines = break_lines(text, 42)
    assert norm("".join(lines)) == norm(text)
    assert all(len(l) <= 42 for l in lines)
    assert all(w in text.split() for l in lines for w in l.split())


def test_wide_letters_are_measured_wider():
    assert char_width("W") > char_width("i")
    assert text_width("WWWW") > text_width("iiii")


def test_unbreakable_token_raises_instead_of_cutting():
    with pytest.raises(ValueError, match="Unbreakable"):
        break_lines("supercalifragilisticexpialidocious", 24)


def test_cues_keep_a_name_in_one_cue_and_conserve_text():
    rows = [dict(number=1, unit_id="u", origin="Music by Epidemic Sounds and edited with FFmpeg tonight.",
                 text="音乐来自 Epidemic Sounds，并且使用 FFmpeg 剪辑完成，今晚发布第二部分内容与更多后续安排。",
                 final_start=0.0, final_end=8.0)]
    for size in ((1920, 1080), (1080, 1920)):
        cues = build_cues(rows, width=size[0], height=size[1])
        assert norm("".join(c["tgt"] for c in cues)) == norm(rows[0]["text"])
        assert norm("".join(c["src"] for c in cues)) == norm(rows[0]["origin"])
        for name in ("Epidemic Sounds", "FFmpeg"):
            assert any(name in c["tgt"].replace("\n", " ") for c in cues)


def test_accented_words_are_never_split():
    from core.subtitle_measure import atoms
    assert atoms("Je suis allé au café près de la gare") == ["Je", " suis", " allé", " au", " café", " près", " de", " la", " gare"]
    rows = break_lines("Je suis allé au café près de la gare avec mon ami José", 12)
    assert all(not r.startswith(("s ", "é")) for r in rows) and " ".join(rows).split() == "Je suis allé au café près de la gare avec mon ami José".split()


def test_lenient_split_keeps_every_character_of_a_url():
    url = "Visit https://www.example.com/very/long/path/to/resource now"
    rows = break_lines(url, 24, lenient=True)
    assert "".join(rows).replace(" ", "") == url.replace(" ", "")
    assert all(text_width(r) <= 24 * LATIN_EM + 1e-6 for r in rows)
    with pytest.raises(ValueError, match="Unbreakable"):
        break_lines(url, 24)


def test_last_row_is_not_an_orphan():
    rows = break_lines("中毕了五回大姨灭亲杀人灭口杀人灭口你就是马匪的", 22)
    assert len(rows) == 2 and min(len(r) for r in rows) >= 10
    assert "".join(rows) == "中毕了五回大姨灭亲杀人灭口杀人灭口你就是马匪的"
