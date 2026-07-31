#!/usr/bin/env python3
"""Benchmark the currently accessible non-Pro SiliconFlow chat models.

Reads the API key from VIDEOLINGO_API_KEY (or the local .env file), never writes
it to disk, and creates a JSON/Markdown evidence pair under docs/reports/.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from json_repair import loads as json_repair_loads

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.prompts_storage import (
    generate_shared_prompt,
    get_prompt_expressiveness,
    get_prompt_faithfulness,
)


API_BASE = "https://api.siliconflow.cn/v1"
CATALOG_URL = "https://www.siliconflow.cn/models"

# A real three-line zh→vi batch from the current project input.  The prompt is
# assembled by the same functions used by core.translate_once.translate_lines.
WORKLOAD_LINES = "\n".join([
    "RexEntropy的COO当国内的MCN还在卷价格抢达人的时候他们直接把战场搬到了海外做到了美国TikTok带货机构的第一名",
    "李嘉琪美万姚万交个朋友这种头部的机构他们在海外其实目前做的都是比较一般的为什么偏偏是他们跑出来了这些我们会聊透国内和海外的达人体系到底差在哪",
    "选中了印尼和美国从选品物流仓储到投放达人带货整条跨境电商的链路怎么跑以及最关键的普通人现在还能抓住哪些TikTok上的赚钱信息差",
])
WORKLOAD_SHARED = generate_shared_prompt(
    "[L00001][S01] 在美国现在拍一条15秒的酒店视频平台就直接给你50美金。",
    "[L00005][S01] 哈喽大家好，欢迎来到 SpicyBrain，我是 Stephanie。",
    "访谈主题：Entropy 如何帮助中国电商品牌通过 TikTok 在美国和印尼扩张。术语：MCN、TikTok、Amazon、GNV 保留或采用越南语行业通行写法。",
    "保持品牌、人名和平台名一致；字幕须自然、简洁，适合配音。",
    '[L00002][S01] 主持人\n[L00003][S01] 主持人\n[L00004][S01] 主持人',
)


def load_env_key() -> str:
    key = os.getenv("VIDEOLINGO_API_KEY", "").strip()
    if key:
        return key
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("VIDEOLINGO_API_KEY="):
                return line.partition("=")[2].strip().strip("\"'")
    raise RuntimeError("VIDEOLINGO_API_KEY is not configured")


def fetch_json(url: str, key: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def extract_catalog() -> dict[str, dict[str, Any]]:
    """Extract the model metadata embedded by SiliconFlow's public model page."""
    with urllib.request.urlopen(CATALOG_URL, timeout=30) as response:
        html = response.read().decode("utf-8")
    fragments = re.findall(r'self\.__next_f\.push\(\[1,"(.*?)"\]\)</script>', html, re.S)
    decoded = "".join(json.loads(f'"{fragment}"') for fragment in fragments)
    marker = '"data":['
    start = decoded.find(marker)
    if start < 0:
        raise RuntimeError("SiliconFlow model page did not contain catalog data")
    start = decoded.find("[", start)
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(decoded)):
        char = decoded[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                records = json.loads(decoded[start : index + 1])
                return {record["modelName"]: record for record in records}
    raise RuntimeError("could not parse SiliconFlow catalog data")


def l0_limit(record: dict[str, Any]) -> dict[str, int | None]:
    l0 = next((item for item in record.get("modelInfo", []) if item.get("Level") == 0), {})
    return {field.lower(): (value if isinstance(value := l0.get(field), int) and value >= 0 else None)
            for field in ("RPM", "TPM", "IPM", "IPD")}


def compact_spec(record: dict[str, Any]) -> str:
    size = record.get("size")
    return f"{size}B" if isinstance(size, (int, float)) and size > 0 else "未公布"


def chat_stream(
    model: str,
    prompt: str,
    key: str,
    *,
    api_base: str | None = None,
    extra_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 1024,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    payload.update(extra_payload or {})
    request = urllib.request.Request(
        f"{api_base or API_BASE}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    started = time.perf_counter()
    first_token_seconds: float | None = None
    usage: dict[str, Any] | None = None
    chunks = 0
    text_parts: list[str] = []
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                item = json.loads(line[6:])
                chunks += 1
                if first_token_seconds is None and item.get("choices"):
                    first_token_seconds = time.perf_counter() - started
                for choice in item.get("choices", []):
                    content = (choice.get("delta") or {}).get("content")
                    if content:
                        text_parts.append(content)
                if item.get("usage"):
                    usage = item["usage"]
        elapsed = time.perf_counter() - started
        completion_tokens = (usage or {}).get("completion_tokens")
        return {
            "status": "pass",
            "ttft_seconds": round(first_token_seconds, 3) if first_token_seconds is not None else None,
            "total_seconds": round(elapsed, 3),
            "completion_tokens": completion_tokens,
            "estimated_output_tokens_per_second": round(completion_tokens / elapsed, 2)
            if isinstance(completion_tokens, int) and elapsed else None,
            "stream_chunks": chunks,
            "content": "".join(text_parts),
        }
    except urllib.error.HTTPError as error:
        return {"status": "fail", "http_status": error.code,
                "error": error.read().decode("utf-8", errors="replace")[:500]}
    except Exception as error:  # Evidence should preserve unavailable models too.
        return {"status": "fail", "error": f"{type(error).__name__}: {error}"}


def valid_stage(payload: Any, field: str) -> bool:
    return (
        isinstance(payload, dict)
        and all(str(index) in payload and isinstance(payload[str(index)], dict)
                and isinstance(payload[str(index)].get(field), str)
                and payload[str(index)][field].strip()
                for index in range(1, 4))
    )


def parse_project_json(content: str) -> Any:
    """Match core.ask_gpt's tolerant JSON parsing for a faithful workflow test."""
    content = content.strip()
    if content.startswith("```"):
        end_marker = content.rfind("```")
        first_newline = content.find("\n")
        if end_marker > 3 and first_newline != -1:
            content = content[first_newline + 1 : end_marker].strip()
    return json_repair_loads(content)


def benchmark(model: str, key: str, *, api_base: str | None = None) -> dict[str, Any]:
    """Run VideoLingo's actual faithful + expressive translation stages."""
    stage1 = chat_stream(model, get_prompt_faithfulness(WORKLOAD_LINES, WORKLOAD_SHARED), key, api_base=api_base)
    if stage1["status"] != "pass":
        return {"status": "fail", "failure_stage": "faithfulness", "faithfulness": stage1}
    try:
        faithfulness = parse_project_json(stage1.pop("content"))
    except Exception:
        return {"status": "fail", "failure_stage": "faithfulness_json", "faithfulness": stage1}
    if not valid_stage(faithfulness, "direct"):
        return {"status": "fail", "failure_stage": "faithfulness_schema", "faithfulness": stage1}
    stage2 = chat_stream(model, get_prompt_expressiveness(faithfulness, WORKLOAD_LINES, WORKLOAD_SHARED), key, api_base=api_base)
    if stage2["status"] != "pass":
        return {"status": "fail", "failure_stage": "expressiveness", "faithfulness": stage1, "expressiveness": stage2}
    try:
        expressiveness = parse_project_json(stage2.pop("content"))
    except Exception:
        return {"status": "fail", "failure_stage": "expressiveness_json", "faithfulness": stage1, "expressiveness": stage2}
    if not valid_stage(expressiveness, "free"):
        return {"status": "fail", "failure_stage": "expressiveness_schema", "faithfulness": stage1, "expressiveness": stage2}
    total_seconds = stage1["total_seconds"] + stage2["total_seconds"]
    return {
        "status": "pass", "total_seconds": round(total_seconds, 3),
        "faithfulness": stage1, "expressiveness": stage2,
        "final_translations": [expressiveness[str(index)]["free"] for index in range(1, 4)],
    }


def markdown(report: dict[str, Any]) -> str:
    scope = report.get("scope", {})
    minimum = float(scope.get("min_parameter_b_exclusive") or 0)
    selected = bool(scope.get("include_model"))
    title = "# SiliconFlow 指定范围内 >100B 聊天模型速度测试" if selected and minimum else "# SiliconFlow 非 `Pro/` 聊天模型速度测试"
    selection = (
        f"仅用户指定白名单，官方规格严格大于 {minimum:g}B、API 当前可访问且目录标为 `text/chat`。"
        if selected and minimum else
        "API 当前可访问、模型 ID 不以 `Pro/` 开头、且官方目录标为 `text/chat`。"
    )
    lines = [
        title,
        "",
        f"- 生成时间（UTC）：{report['generated_at']}",
        "- 平台：SiliconFlow OpenAI-compatible API（`https://api.siliconflow.cn/v1`）",
        "- 样本：项目实际的中文→越南语三行访谈字幕批次，使用 `core.translate_once.translate_lines` 的原始忠实翻译与表达优化提示词；每阶段流式请求统一 `temperature=0`、`max_tokens=1024`，并校验 3 行 JSON 结构。",
        f"- 筛选：{selection} 其他类型保留在 JSON 中并标注为不适用。",
        "- L0：官方目录的当前 L0 限速。`—` 表示该维度未设置；速率策略可调整，应以模型中心实时页面为准。",
        "",
        "| 模型 | 平台 | 规格 | 上下文 | L0 RPM | L0 TPM | 忠实阶段首 Token (s) | 表达阶段首 Token (s) | 两阶段总耗时 (s) | 状态 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in report["benchmarks"]:
        result = item["result"]
        def cell(value: Any) -> str:
            return "—" if value is None else str(value)
        faithfulness = result.get("faithfulness", {})
        expressiveness = result.get("expressiveness", {})
        lines.append("| {model} | SiliconFlow | {spec} | {context} | {rpm} | {tpm} | {stage1_ttft} | {stage2_ttft} | {total} | {status} |".format(
            model=item["model"], spec=item["parameter_spec"],
            context=(f"{item['context_tokens'] // 1024}K" if item["context_tokens"] else "未公布"),
            rpm=cell(item["l0_rate_limit"]["rpm"]), tpm=cell(item["l0_rate_limit"]["tpm"]),
            stage1_ttft=cell(faithfulness.get("ttft_seconds")), stage2_ttft=cell(expressiveness.get("ttft_seconds")),
            total=cell(result.get("total_seconds")),
            status=(result["status"] if result["status"] == "pass" else f"fail: {result.get('failure_stage', 'unknown')}"),
        ))
    failure_counts: dict[str, int] = {}
    for item in report["benchmarks"]:
        result = item["result"]
        if result["status"] == "fail":
            stage = result.get("failure_stage", "unknown")
            failure_counts[stage] = failure_counts.get(stage, 0) + 1
    if failure_counts:
        lines += ["", "## Chat 模型失败摘要", ""]
        for stage, count in sorted(failure_counts.items()):
            description = {
                "faithfulness": "忠实翻译 API 拒绝或请求失败",
                "faithfulness_json": "忠实翻译未返回可解析 JSON",
                "faithfulness_schema": "忠实翻译 JSON 缺少项目必需字段",
                "expressiveness": "表达优化 API 拒绝或请求失败",
                "expressiveness_json": "表达优化未返回可解析 JSON",
                "expressiveness_schema": "表达优化 JSON 缺少项目必需字段",
            }.get(stage, stage)
            lines.append(f"- `{stage}`：{count} 个（{description}）。")
    lines += ["", "## 非聊天模型（未进行 Chat 速度测试）", "",
              "这些模型没有作为 VideoLingo 的 LLM 翻译候选进行请求：", ""]
    for item in report["not_benchmarked"]:
        lines.append(f"- `{item['model']}`：`{item['type']}/{item['sub_type']}`；规格 {item['parameter_spec']}；L0 RPM/TPM {item['l0_rate_limit']['rpm'] or '—'}/{item['l0_rate_limit']['tpm'] or '—'}。")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-prefix", default="docs/reports/siliconflow-non-pro-model-benchmark-2026-07-31")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent model workloads (default: 4)")
    parser.add_argument("--include-model", action="append", default=[], help="Exact model ID to include; repeat as needed")
    parser.add_argument("--min-parameter-b", type=float, default=0, help="Only test catalog models strictly larger than this size in B")
    args = parser.parse_args()
    key = load_env_key()
    accessible = {item["id"] for item in fetch_json(f"{API_BASE}/models", key).get("data", [])}
    catalog = extract_catalog()
    benchmarks, not_benchmarked = [], []
    candidates: list[dict[str, Any]] = []
    for model, record in sorted(catalog.items()):
        if model not in accessible or model.startswith("Pro/"):
            continue
        if args.include_model and model not in args.include_model:
            continue
        if float(record.get("size") or 0) <= args.min_parameter_b:
            continue
        entry = {
            "model": model, "platform": "SiliconFlow", "type": record.get("type", "unknown"),
            "sub_type": record.get("subType", "unknown"), "parameter_spec": compact_spec(record),
            "context_tokens": record.get("contextLen"), "l0_rate_limit": l0_limit(record),
        }
        if record.get("type") == "text" and record.get("subType") == "chat":
            candidates.append(entry)
        else:
            not_benchmarked.append(entry)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(benchmark, item["model"], key): item for item in candidates}
        for future in concurrent.futures.as_completed(futures):
            entry = futures[future]
            try:
                entry["result"] = future.result()
            except Exception as error:
                entry["result"] = {"status": "fail", "failure_stage": "runner", "error": f"{type(error).__name__}: {error}"}
            benchmarks.append(entry)
    benchmarks.sort(key=lambda item: item["model"])
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "api_model_count": len(accessible), "benchmarks": benchmarks, "not_benchmarked": not_benchmarked,
        "scope": {"include_model": args.include_model, "min_parameter_b_exclusive": args.min_parameter_b},
        "source": {"api_models": f"{API_BASE}/models", "catalog": CATALOG_URL},
    }
    prefix = ROOT / args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    prefix.with_suffix(".md").write_text(markdown(report), encoding="utf-8")
    print(json.dumps({"benchmark_count": len(benchmarks), "not_benchmarked_count": len(not_benchmarked),
                      "json": str(prefix.with_suffix('.json')), "markdown": str(prefix.with_suffix('.md'))}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
