"""Run local-only translation regression samples in a fresh, isolated directory.

This is a small integration check with explicit lexical assertions, not a
human evaluation or an independent translation quality benchmark.
"""
import argparse
import json
import os
from pathlib import Path
import re
import sys
import time
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from ruamel.yaml import YAML

from core import config_utils
from core.llm_provider import get_llm_provider_config
from core.translate_once import translate_lines
from core.translation_review import review_and_repair_translation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare-legacy", action="store_true")
    parser.add_argument("--dataset", type=Path, default=ROOT / "tests/fixtures/local_translation.json")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    yaml = YAML()
    base = yaml.load((ROOT / "config.yaml").read_text())
    if base["llm"]["provider"] != "omlx":
        raise ValueError("This check requires the local oMLX provider")
    cases = json.loads(args.dataset.read_text())
    results = []
    for case in cases:
        for strategy in (["localized", "legacy"] if args.compare_legacy else ["localized"]):
            workspace = output / case["id"] / strategy
            workspace.mkdir(parents=True)
            config = yaml.load((ROOT / "config.yaml").read_text())
            config["source_language"] = case["source"]
            config["whisper"]["detected_language"] = case["source"]
            config["target_language"] = case["target"]
            config["translation"]["strategy"] = strategy
            config_path = workspace / "config.yaml"
            with config_path.open("w") as f:
                yaml.dump(config, f)
            config_utils.CONFIG_PATH = str(config_path)
            os.chdir(workspace)
            provider = get_llm_provider_config()
            if provider.name != "omlx" or urlsplit(provider.base_url).hostname not in {"localhost", "127.0.0.1", "::1"}:
                raise ValueError("Refusing non-local inference")
            rows = case["lines"]
            source = "\n".join(line for _, line in rows)
            speakers = "\n".join(f"[L{i:05d}][{speaker}] {line}" for i, (speaker, line) in enumerate(rows, 1))
            started = time.monotonic()
            record = {"id": case["id"], "strategy": strategy, "model": provider.model, "source": case["source"], "target": case["target"]}
            try:
                translated, echoed = translate_lines(source, None, None, json.dumps(case["terms"], ensure_ascii=False), case["topic"], speaker_context_prompt=speakers, glossary=case["terms"])
                translation_seconds = time.monotonic() - started
                raw_df = pd.DataFrame({"LineID": [f"L{i:05d}" for i in range(1, len(rows) + 1)], "Speaker": [r[0] for r in rows], "Source": source.splitlines(), "Translation": translated.splitlines()})
                df, review = review_and_repair_translation(raw_df, {"topic": case["topic"], "terms": case["terms"]}, max_repair_rounds=3)
                checks = []
                for check in case["checks"]:
                    text = str(df.iloc[check["row"]]["Translation"])
                    checks.extend({"row": check["row"], "pattern": pattern, "pass": bool(re.search(pattern, text))} for pattern in check["patterns"])
                df.to_excel(workspace / "translation.xlsx", index=False)
                record.update({"draft": raw_df.to_dict("records"), "rows": df.to_dict("records"), "review": review, "checks": checks, "translation_seconds": round(translation_seconds, 3), "structural_pass": echoed == source and len(df) == len(rows), "status": "pass" if review["status"] == "pass" and all(c["pass"] for c in checks) else "fail"})
            except Exception as exc:
                record.update({"status": "fail", "error": f"{type(exc).__name__}: {exc}"})
            record["total_seconds"] = round(time.monotonic() - started, 3)
            results.append(record)
            (output / "results.json").write_text(json.dumps({"evaluation_scope": "integration_and_lexical_checks; human_quality_review_pending", "results": results}, ensure_ascii=False, indent=2))
            print(f"{case['id']} / {strategy}: {record['status']} ({record['total_seconds']}s)", flush=True)
    return int(any(r["status"] != "pass" for r in results))


if __name__ == "__main__":
    raise SystemExit(main())
