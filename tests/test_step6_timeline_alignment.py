import pandas as pd

from core.step6_generate_final_timeline import get_sentence_timestamps


def _word_df(text: str, start: float = 0.0, step: float = 0.2) -> pd.DataFrame:
    rows = []
    current = start
    for char in text:
        rows.append({"text": char, "start": current, "end": current + step, "speaker": None})
        current += step
    return pd.DataFrame(rows)


def test_timeline_fallback_consumes_local_cursor_for_asr_correction():
    words = _word_df("尺寸高308宽2661到10人口用量的能烤10斤左右东西")
    sentences = pd.DataFrame(
        {
            "Source": [
                "尺寸高308宽266",
                "一到十人口用的",
                "能烤10斤左右东西",
            ]
        }
    )

    timestamps = get_sentence_timestamps(words, sentences)

    assert timestamps[1][0] < 3.0
    assert timestamps[-1][1] <= words["end"].iloc[-1]
    assert timestamps[0][1] <= timestamps[1][0]
    assert timestamps[1][1] <= timestamps[2][0]


def test_timeline_rejects_distant_repeated_match_when_similarity_is_not_strong():
    local_words = "尺寸高308宽2661到10人口用量的能烤10斤左右东西"
    filler = "甲" * 500
    repeated_later = "一到十人口用的"
    words = _word_df(local_words + filler + repeated_later)
    sentences = pd.DataFrame(
        {
            "Source": [
                "尺寸高308宽266",
                "一到十人口用的",
                "能烤10斤左右东西",
            ]
        }
    )

    timestamps = get_sentence_timestamps(words, sentences)

    assert timestamps[1][0] < 8.0
    assert timestamps[2][0] < 12.0
