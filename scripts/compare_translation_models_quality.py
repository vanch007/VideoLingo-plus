#!/usr/bin/env python3
"""Build a like-for-like speed, cost, compatibility, and quality comparison."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from benchmark_omlx_translation_quality import RUBRIC_MAX, run_judge
from benchmark_siliconflow_models import ROOT, extract_catalog, load_env_key


CLOUD_REPORT = ROOT / "docs/reports/siliconflow-selected-over-100b-translation-benchmark-2026-07-31.json"
LOCAL_REPORT = ROOT / "docs/reports/omlx-qwen3.6-35b-translation-quality-2026-07-31.json"
OUTPUT_PREFIX = ROOT / "docs/reports/siliconflow-vs-omlx-translation-comparison-2026-07-31"
JUDGES = ("Qwen/Qwen3.5-35B-A3B", "deepseek-ai/DeepSeek-V3.1-Terminus")


def metric_summary(result: dict[str, Any]) -> dict[str, Any]:
    faith = result.get("faithfulness", {})
    express = result.get("expressiveness", {})
    return {
        "status": result["status"],
        "failure_stage": result.get("failure_stage"),
        "total_seconds": result.get("total_seconds"),
        "faithfulness_ttft_seconds": faith.get("ttft_seconds"),
        "expressiveness_ttft_seconds": express.get("ttft_seconds"),
        "faithfulness_tokens_per_second": faith.get("estimated_output_tokens_per_second"),
        "expressiveness_tokens_per_second": express.get("estimated_output_tokens_per_second"),
    }


def average_quality(judges: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [judge for judge in judges if judge["status"] == "pass"]
    if not valid:
        return {"score": None, "valid_judges": 0, "confidence": "insufficient"}
    dimensions = {
        name: round(mean(judge["judgement"]["scores"][name] for judge in valid), 1)
        for name in RUBRIC_MAX
    }
    totals = [judge["judgement"]["total"] for judge in valid]
    return {
        "score": round(mean(totals), 1),
        "min_total": min(totals),
        "max_total": max(totals),
        "valid_judges": len(valid),
        "dimensions": dimensions,
        "confidence": "preliminary_single_sample_two_judges" if len(valid) == 2 else "low_single_judge",
    }


def save(report: dict[str, Any]) -> None:
    Path(f"{OUTPUT_PREFIX}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(f"{OUTPUT_PREFIX}.md").write_text(markdown(report), encoding="utf-8")


def cell(value: Any, suffix: str = "") -> str:
    return "—" if value is None else f"{value}{suffix}"


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# SiliconFlow >100B 与本地 oMLX 翻译模型全面对比",
        "",
        f"- 生成时间（UTC）：{report['generated_at']}",
        "- 工作负载：相同三行中文→越南语真实字幕，忠实翻译 + 表达优化两阶段。",
        "- 质量口径：Qwen3.5-35B-A3B 与 DeepSeek-V3.1-Terminus 独立盲评；候选模型名称未写入评分提示。",
        "- 限制：单一短样本，仅用于初步选型；不构成人工译审或整片质量结论。",
        "",
        "## 速度、兼容性与质量",
        "",
        "| 模型 | 平台 | 参数 | 两阶段总耗时 | 输出 tok/s（两阶段） | 质量 /100 | 结构状态 |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    ordered = sorted(
        report["candidates"],
        key=lambda item: (item["metrics"]["status"] != "pass", item["metrics"]["total_seconds"] or 10**9),
    )
    for item in ordered:
        metrics = item["metrics"]
        quality = item.get("quality", {})
        throughput = f"{cell(metrics['faithfulness_tokens_per_second'])} / {cell(metrics['expressiveness_tokens_per_second'])}"
        status = "通过" if metrics["status"] == "pass" else f"失败：{metrics.get('failure_stage')}"
        lines.append(
            f"| {item['display_name']} | {item['platform']} | {item['parameter_spec']} | "
            f"{cell(metrics['total_seconds'], 's')} | {throughput} | {cell(quality.get('score'))} | {status} |"
        )
    lines += [
        "",
        "## 质量分项",
        "",
        "| 模型 | 忠实度 /40 | 完整性 /20 | 越南语自然度 /20 | 术语 /10 | 配音简洁度 /10 | 有效评审 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in ordered:
        quality = item.get("quality", {})
        dimensions = quality.get("dimensions", {})
        lines.append(
            f"| {item['display_name']} | {cell(dimensions.get('faithfulness'))} | {cell(dimensions.get('completeness'))} | "
            f"{cell(dimensions.get('vietnamese_naturalness'))} | {cell(dimensions.get('terminology_consistency'))} | "
            f"{cell(dimensions.get('dubbing_conciseness'))} | {quality.get('valid_judges', 0)}/2 |"
        )
    lines += [
        "",
        "## 规格、计费与限速",
        "",
        "| 模型 | 上下文 | 输入价 | 输出价 | 抵用券 | L0 RPM | L0 TPM |",
        "|---|---:|---:|---:|---|---:|---:|",
    ]
    for item in ordered:
        rate = item["rate_limit"]
        price = item["pricing"]
        lines.append(
            f"| {item['display_name']} | {item['context']} | {price['input']} | {price['output']} | "
            f"{item['voucher']} | {cell(rate.get('rpm'))} | {cell(rate.get('tpm'))} |"
        )
    lines += ["", "## 评审摘要", ""]
    for item in ordered:
        if item["metrics"]["status"] != "pass":
            lines.append(f"- **{item['display_name']}**：结构门禁未通过，未进入质量评分。")
            continue
        summaries = [judge["judgement"].get("summary", "") for judge in item["judges"] if judge["status"] == "pass"]
        lines.append(f"- **{item['display_name']}**（{cell(item.get('quality', {}).get('score'))}）：{' / '.join(summaries)}")
    return "\n".join(lines) + "\n"


def main() -> int:
    cloud = json.loads(CLOUD_REPORT.read_text(encoding="utf-8"))
    local = json.loads(LOCAL_REPORT.read_text(encoding="utf-8"))
    catalog = extract_catalog()
    candidates: list[dict[str, Any]] = []
    for source in cloud["benchmarks"]:
        record = catalog.get(source["model"], {})
        currency = record.get("currency", "¥")
        unit = record.get("priceUnit", "/ M Tokens")
        candidates.append({
            "model": source["model"],
            "display_name": source["model"].split("/")[-1],
            "platform": "SiliconFlow",
            "parameter_spec": source["parameter_spec"],
            "context": f"{source['context_tokens'] // 1024}K" if source.get("context_tokens") else "未公布",
            "pricing": {"input": f"{currency}{record.get('inputPrice', '—')} {unit}", "output": f"{currency}{record.get('outputPrice', '—')} {unit}"},
            "voucher": "适用",
            "rate_limit": source["l0_rate_limit"],
            "metrics": metric_summary(source["result"]),
            "translations": source["result"].get("final_translations"),
            "judges": [],
            "quality": {"score": None, "valid_judges": 0, "confidence": "not_scored"},
        })
    local_result = local["translation_benchmark"]
    candidates.append({
        "model": local["model"],
        "display_name": "oMLX Qwen3.6-35B-A3B",
        "platform": "本地 oMLX",
        "parameter_spec": "35B（A3B 激活，oQ6）",
        "context": "未在服务元数据公布",
        "pricing": {"input": "本地 ¥0", "output": "本地 ¥0"},
        "voucher": "不需要",
        "rate_limit": {"rpm": None, "tpm": None},
        "metrics": metric_summary(local_result),
        "translations": local_result.get("final_translations"),
        "judges": [],
        "quality": {"score": None, "valid_judges": 0, "confidence": "not_scored"},
    })
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "methodology": {"judges": list(JUDGES), "rubric_max": RUBRIC_MAX, "sample_count": 1},
        "candidates": candidates,
    }
    save(report)
    key = load_env_key()
    for candidate in candidates:
        if candidate["metrics"]["status"] != "pass" or not candidate["translations"]:
            continue
        for judge_model in JUDGES:
            judgement = run_judge(judge_model, candidate["translations"], key)
            candidate["judges"].append(judgement)
            save(report)
        candidate["quality"] = average_quality(candidate["judges"])
        save(report)
        print(json.dumps({"candidate": candidate["model"], "quality": candidate["quality"]["score"]}), flush=True)
    save(report)
    print(json.dumps({"json": f"{OUTPUT_PREFIX}.json", "markdown": f"{OUTPUT_PREFIX}.md"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
