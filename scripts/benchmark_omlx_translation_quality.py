#!/usr/bin/env python3
"""Benchmark one oMLX model on VideoLingo translation and score its quality."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from benchmark_siliconflow_models import (
    API_BASE,
    ROOT,
    WORKLOAD_LINES,
    benchmark,
    chat_stream,
    load_env_key,
    parse_project_json,
)


OMLX_BASE = "http://127.0.0.1:8000/v1"
DEFAULT_MODEL = "Qwen3.6-35B-A3B-Qwable-Holo3-Qwopus-oQ6-mtp"
JUDGES = ("deepseek-ai/DeepSeek-V3", "zai-org/GLM-4.5-Air")
RUBRIC_MAX = {
    "faithfulness": 40,
    "completeness": 20,
    "vietnamese_naturalness": 20,
    "terminology_consistency": 10,
    "dubbing_conciseness": 10,
}


def judge_prompt(translations: list[str]) -> str:
    source = "\n".join(f"{index}. {line}" for index, line in enumerate(WORKLOAD_LINES.splitlines(), 1))
    candidate = "\n".join(f"{index}. {line}" for index, line in enumerate(translations, 1))
    return f"""You are an independent senior Chinese-to-Vietnamese subtitle quality evaluator.
Evaluate the candidate translation against the Chinese source. Do not reward verbosity.

SOURCE:
{source}

CANDIDATE VIETNAMESE:
{candidate}

Score exactly these dimensions using integer points:
- faithfulness: 0-40 (meaning, relationships, facts)
- completeness: 0-20 (no omissions/additions; all three lines preserved)
- vietnamese_naturalness: 0-20 (native, grammatical, idiomatic Vietnamese)
- terminology_consistency: 0-10 (Rex, Entropy, COO, MCN, TikTok, Indonesia/US and ecommerce terms)
- dubbing_conciseness: 0-10 (subtitle readability and speakable brevity)

Return JSON only with this schema:
{{
  "scores": {{
    "faithfulness": 0,
    "completeness": 0,
    "vietnamese_naturalness": 0,
    "terminology_consistency": 0,
    "dubbing_conciseness": 0
  }},
  "total": 0,
  "line_findings": [
    {{"line": 1, "severity": "ok|minor|major", "finding": "concise evidence"}}
  ],
  "summary": "concise evidence-based verdict"
}}
The total must equal the sum of the five dimensions."""


def validate_judgement(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict) or not isinstance(payload.get("scores"), dict):
        raise ValueError("judge response has no scores object")
    scores: dict[str, int] = {}
    for name, maximum in RUBRIC_MAX.items():
        value = payload["scores"].get(name)
        if not isinstance(value, (int, float)) or value < 0 or value > maximum:
            raise ValueError(f"invalid {name} score: {value}")
        scores[name] = int(value)
    payload["scores"] = scores
    payload["total"] = sum(scores.values())
    return payload


def run_judge(model: str, translations: list[str], key: str) -> dict[str, Any]:
    extra_payload = {"enable_thinking": False} if model.startswith("Qwen/") else None
    response = chat_stream(model, judge_prompt(translations), key, api_base=API_BASE, extra_payload=extra_payload)
    if response["status"] != "pass":
        return {"model": model, "status": "fail", "request": response}
    content = response.pop("content")
    try:
        judgement = validate_judgement(parse_project_json(content))
    except Exception as error:
        return {"model": model, "status": "fail", "error": f"{type(error).__name__}: {error}", "request": response}
    return {"model": model, "status": "pass", "request": response, "judgement": judgement}


def render_markdown(report: dict[str, Any]) -> str:
    result = report["translation_benchmark"]
    lines = [
        "# oMLX Qwen3.6 翻译速度与质量评分",
        "",
        f"- 生成时间（UTC）：{report['generated_at']}",
        f"- 模型：`{report['model']}`",
        f"- 接口：`{report['api_base']}`",
        "- 负载：与 SiliconFlow 测试相同的三行中文→越南语真实字幕、忠实翻译 + 表达优化两阶段。",
        "- 质量评分：DeepSeek-V3 与 GLM-4.5-Air 独立评分；单一样本，仅作初步选型证据。",
        "",
        "## 速度与结构",
        "",
    ]
    if result["status"] != "pass":
        lines.append(f"- 状态：fail（`{result.get('failure_stage', 'unknown')}`）")
        return "\n".join(lines) + "\n"
    faith = result["faithfulness"]
    express = result["expressiveness"]
    lines += [
        "| 阶段 | 首 Token | 总耗时 | completion tokens | 输出 tok/s |",
        "|---|---:|---:|---:|---:|",
        f"| 忠实翻译 | {faith.get('ttft_seconds')}s | {faith.get('total_seconds')}s | {faith.get('completion_tokens')} | {faith.get('estimated_output_tokens_per_second')} |",
        f"| 表达优化 | {express.get('ttft_seconds')}s | {express.get('total_seconds')}s | {express.get('completion_tokens')} | {express.get('estimated_output_tokens_per_second')} |",
        f"| 合计 | — | **{result.get('total_seconds')}s** | — | — |",
        "",
        "## 最终译文",
        "",
    ]
    for index, translation in enumerate(result["final_translations"], 1):
        lines.append(f"{index}. {translation}")
    quality = report.get("quality", {})
    lines += ["", "## 质量评分", ""]
    if quality.get("score") is None:
        lines.append("评分失败：没有足够的有效独立评审结果。")
    else:
        lines += [
            f"- 综合分：**{quality['score']:.1f}/100**",
            f"- 评审范围：{quality['min_total']}–{quality['max_total']} 分",
            f"- 有效评审：{quality['valid_judges']}/{len(report['judges'])}",
            "",
            "| 评审模型 | 忠实度 /40 | 完整性 /20 | 越南语自然度 /20 | 术语 /10 | 配音简洁度 /10 | 总分 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for judge in report["judges"]:
            if judge["status"] != "pass":
                lines.append(f"| {judge['model']} | — | — | — | — | — | fail |")
                continue
            scores = judge["judgement"]["scores"]
            lines.append(
                f"| {judge['model']} | {scores['faithfulness']} | {scores['completeness']} | "
                f"{scores['vietnamese_naturalness']} | {scores['terminology_consistency']} | "
                f"{scores['dubbing_conciseness']} | {judge['judgement']['total']} |"
            )
        lines += ["", "### 评审意见", ""]
        for judge in report["judges"]:
            if judge["status"] == "pass":
                lines.append(f"- **{judge['model']}**：{judge['judgement'].get('summary', '')}")
    return "\n".join(lines) + "\n"


def refresh_quality(report: dict[str, Any]) -> None:
    valid = [judge for judge in report["judges"] if judge["status"] == "pass"]
    totals = [judge["judgement"]["total"] for judge in valid]
    report["quality"] = {
        "score": round(mean(totals), 1) if totals else None,
        "min_total": min(totals) if totals else None,
        "max_total": max(totals) if totals else None,
        "valid_judges": len(valid),
        "confidence": "preliminary_single_sample_two_judges" if len(valid) == 2 else "insufficient",
    }


def save_report(prefix: Path, report: dict[str, Any]) -> None:
    refresh_quality(report)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    Path(f"{prefix}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(f"{prefix}.md").write_text(render_markdown(report), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-prefix", default="docs/reports/omlx-qwen3.6-35b-translation-quality-2026-07-31")
    parser.add_argument("--translation-only", action="store_true", help="Run and checkpoint local translation without judges")
    parser.add_argument("--judge-only", action="store_true", help="Load checkpoint and append missing judges")
    parser.add_argument("--judge-model", action="append", default=[], help="Judge model to run; repeat as needed")
    args = parser.parse_args()
    prefix = ROOT / args.output_prefix
    if args.judge_only:
        report = json.loads(Path(f"{prefix}.json").read_text(encoding="utf-8"))
        result = report["translation_benchmark"]
    else:
        print(json.dumps({"event": "translation_started", "model": args.model}), flush=True)
        result = benchmark(args.model, "1234", api_base=OMLX_BASE)
        print(json.dumps({"event": "translation_finished", "status": result["status"], "failure_stage": result.get("failure_stage")}), flush=True)
        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model": args.model,
            "api_base": OMLX_BASE,
            "workload": {"source_language": "zh", "target_language": "vi", "source_lines": WORKLOAD_LINES.splitlines()},
            "translation_benchmark": result,
            "quality": {},
            "judges": [],
        }
        save_report(prefix, report)
    if result["status"] == "pass" and not args.translation_only:
        key = load_env_key()
        existing = {judge["model"] for judge in report["judges"]}
        for model in (args.judge_model or list(JUDGES)):
            if model in existing:
                continue
            print(json.dumps({"event": "judge_started", "model": model}), flush=True)
            judgement = run_judge(model, result["final_translations"], key)
            report["judges"].append(judgement)
            save_report(prefix, report)
            print(json.dumps({"event": "judge_finished", "model": model, "status": judgement["status"]}), flush=True)
    save_report(prefix, report)
    quality = report["quality"]
    print(json.dumps({"status": result["status"], "quality_score": quality["score"], "json": f"{prefix}.json", "markdown": f"{prefix}.md"}))
    return 0 if result["status"] == "pass" and (args.translation_only or quality["score"] is not None) else 2


if __name__ == "__main__":
    raise SystemExit(main())
