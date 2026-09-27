import tempfile
from pathlib import Path
import pandas as pd
import pytest

from core.step4_1_summarize import detect_source_anomalies, resolve_source_anomalies


@pytest.mark.parametrize(
    "text",
    [
        "请联系Alice明天付款",
        "请用USB接口连接",
        "这是BBC新闻",
        "升级到iPhone 15 Pro",
        "支持OAuth 2.0协议",
    ],
)
def test_preserve_valid_mixed_language_source(text: str):
    with tempfile.TemporaryDirectory() as tmp:
        sentences_path = Path(tmp) / "sentences.txt"
        sentences_path.write_text(text, encoding="utf-8")
        rows = pd.DataFrame([{"text": text, "line_id": "L00001", "speaker": "S01"}])
        anomalies = detect_source_anomalies(str(Path(tmp) / "absent.xlsx"), rows)
        resolved = resolve_source_anomalies(anomalies, rows, str(sentences_path))

        # Text must not be stripped or corrupted
        assert sentences_path.read_text(encoding="utf-8") == text

        # Any script mixing flags a risk requiring acoustic witness, not marked resolved=True
        for item in resolved:
            if item.get("type") == "corrupted_source_script":
                assert item["resolved"] is False
                assert "acoustic_witness_required" in item["action"]
