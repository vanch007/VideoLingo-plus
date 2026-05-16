from pathlib import Path

import pandas as pd

from core.providers.timeline_rescue import build_timeline_rescue_report, write_timeline_rescue_report


def _tasks_df():
    return pd.DataFrame(
        [
            {
                "number": 1,
                "start_time": "00:00:01.000",
                "end_time": "00:00:02.000",
                "duration": 2.0,
                "sub_times": [1.0, 2.0],
                "gap": -0.5,
                "tolerance": -0.5,
                "text": "short row",
                "lines": ["short row"],
                "real_dur": 2.5,
            },
            {
                "number": 2,
                "start_time": "00:00:01.000",
                "end_time": "00:00:02.000",
                "duration": 2.0,
                "sub_times": [1.0, 2.0],
                "gap": 0.0,
                "tolerance": 0.0,
                "text": "duplicate row",
                "lines": ["duplicate row"],
                "real_dur": 2.5,
            },
        ]
    )


def _eval_df():
    return pd.DataFrame(
        [
            {
                "number": 1,
                "status": "warn",
                "reason": "over_duration",
                "available_duration": 1.5,
                "final_audio_dur": 2.5,
                "duration_ratio": 1.7,
            },
            {
                "number": 2,
                "status": "warn",
                "reason": "over_duration",
                "available_duration": 1.0,
                "final_audio_dur": 2.5,
                "duration_ratio": 2.5,
            },
        ]
    )


def test_timeline_rescue_report_detects_duplicate_and_negative_timing():
    report = build_timeline_rescue_report(_tasks_df(), _eval_df())
    assert report["summary"]["issue_counts"]["negative_gap"] == 1
    assert report["summary"]["issue_counts"]["negative_tolerance"] == 1
    assert report["summary"]["issue_counts"]["duplicate_window"] == 2
    assert report["clusters"][0]["row_count"] == 2
    assert report["clusters"][0]["suggested_action"] == "rebuild_monotonic_from_declared_duration"


def test_timeline_rescue_report_can_write_json_and_xlsx(tmp_path: Path):
    report = build_timeline_rescue_report(_tasks_df(), _eval_df())
    paths = write_timeline_rescue_report(
        report,
        json_path=str(tmp_path / "timeline.json"),
        xlsx_path=str(tmp_path / "timeline.xlsx"),
    )
    assert Path(paths["json_path"]).exists()
    assert Path(paths["xlsx_path"]).exists()
