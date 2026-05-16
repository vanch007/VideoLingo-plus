from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class QualityGateResult:
    passed: bool
    total: int
    ok: int
    warn: int
    fail: int
    reasons: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "total": self.total,
            "ok": self.ok,
            "warn": self.warn,
            "fail": self.fail,
            "reasons": self.reasons,
        }


def summarize_quality_gate(eval_df: pd.DataFrame) -> QualityGateResult:
    if eval_df.empty:
        return QualityGateResult(False, 0, 0, 0, 0, {})
    reasons: dict[str, int] = {}
    for raw in eval_df.get("reason", []):
        for reason in str(raw or "").split(","):
            reason = reason.strip()
            if reason:
                reasons[reason] = reasons.get(reason, 0) + 1
    fail = int((eval_df["status"] == "fail").sum())
    warn = int((eval_df["status"] == "warn").sum())
    ok = int((eval_df["status"] == "ok").sum())
    return QualityGateResult(
        passed=fail == 0 and warn == 0 and ok > 0,
        total=int(len(eval_df)),
        ok=ok,
        warn=warn,
        fail=fail,
        reasons=reasons,
    )
