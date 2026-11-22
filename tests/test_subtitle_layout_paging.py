from core.subtitle_layout import balanced_pages, snap_source_cut, split_display_text


def test_long_cue_pages_are_balanced_at_a_sentence_end():
    # Recorded from the hongmenyan run: fixed two-line paging left "name." alone on a page.
    text = "See? You can't even pretend to be clueless. You gave her a Japanese-sounding name."
    pages = balanced_pages(text, 38, 2)
    assert [p.replace("\n", " ") for p in pages] == ["See? You can't even pretend to be clueless.",
                                                     "You gave her a Japanese-sounding name."]
    assert all(len(split_display_text(p.replace("\n", " "), 38)) <= 2 for p in pages)


def test_short_cue_is_one_page_and_no_text_is_lost():
    assert balanced_pages("Hello there.", 38, 2) == ["Hello there."]
    text = " ".join(["word"] * 60)
    pages = balanced_pages(text, 20, 2)
    assert " ".join(p.replace("\n", " ") for p in pages) == text
    assert max(len(p) for p in pages) - min(len(p) for p in pages) <= 12


def test_source_cut_never_leaves_only_punctuation():
    source = "说你不会装糊涂吧，起了个东洋名字。"
    # With balanced translation pages the proportional estimate lands mid-sentence and snaps to the comma.
    pages = balanced_pages("See? You can't even pretend to be clueless. You gave her a Japanese-sounding name.", 38, 2)
    pos = round(len(source) * len(pages[0]) / sum(len(p) for p in pages))
    cut = snap_source_cut(source, pos, 0)
    assert source[:cut] == "说你不会装糊涂吧，" and source[cut:] == "起了个东洋名字。"
    # Even the old worst case (estimate just before "。") keeps words on both sides.
    cut = snap_source_cut(source, len(source) - 1, 0)
    assert source[cut:].strip("。") and source[:cut]
    latin = "You can't even pretend to be clueless and you gave her a name"
    cut = snap_source_cut(latin, 20, 0)
    assert latin[cut - 1] == " " or latin[cut] == " "
