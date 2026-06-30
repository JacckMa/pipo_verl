#!/usr/bin/env python3
"""Convert BFCL-v4 offline function-call data to VERL parquet for tooluse eval.

This intentionally keeps BFCL as evaluation-only data. It includes categories that
can be scored offline with static function-call matching and excludes memory,
web_search, format_sensitivity, and live_relevance by default.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

CATEGORIES_WITH_PA = [
    "simple_python", "simple_java", "simple_javascript",
    "multiple", "parallel", "parallel_multiple",
    "live_simple", "live_multiple", "live_parallel", "live_parallel_multiple",
    "multi_turn_base", "multi_turn_miss_func", "multi_turn_miss_param", "multi_turn_long_context",
]
NO_CALL_CATEGORIES = ["irrelevance", "live_irrelevance"]
EXCLUDED = ["memory", "web_search", "format_sensitivity", "live_relevance"]

SYSTEM = """You are a helpful assistant that selects and calls tools when needed.

Use the available tools to solve the user's request. If a tool call is needed, output it exactly in this format:
<think>brief reasoning</think>
<tool_call>
{"name": "tool_name", "parameters": {"arg": "value"}}
</tool_call>

If no tool is relevant, do not invent a tool call; answer normally in <response>...</response>."""


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def load_multiturn_docs(data_root: Path) -> dict[str, dict]:
    docs: dict[str, dict] = {}
    for path in (data_root / "multi_turn_func_doc").glob("*.json"):
        for obj in read_jsonl(path):
            docs.setdefault(obj.get("name"), obj)
    return docs


def tool_doc_text(functions: list[dict]) -> str:
    lines = ["**Available Tools**"]
    for idx, fn in enumerate(functions, 1):
        lines.append(f"{idx}. Name: {fn.get('name', '')}")
        lines.append(f"Description: {fn.get('description', '')}")
        lines.append("Parameters: " + json.dumps(fn.get("parameters", {}), ensure_ascii=False))
    return "\n".join(lines)


def flatten_question(question: list) -> str:
    parts = []
    for turn_idx, turn in enumerate(question, 1):
        messages = []
        for msg in turn:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            messages.append(f"<{role}> {content} </{role}>")
        prefix = f"**Turn {turn_idx}**\n" if len(question) > 1 else ""
        parts.append(prefix + "\n".join(messages))
    return "\n\n".join(parts)


def functions_for_sample(sample: dict, multiturn_docs: dict[str, dict]) -> list[dict]:
    if "function" in sample:
        return sample["function"]
    functions = []
    seen = set()
    for path in sample.get("path", []):
        name = path.split(".")[-1]
        if name in seen:
            continue
        fn = multiturn_docs.get(name)
        if fn:
            functions.append(fn)
            seen.add(name)
    return functions


def make_prompt(sample: dict, category: str, multiturn_docs: dict[str, dict]) -> str:
    functions = functions_for_sample(sample, multiturn_docs)
    sections = [SYSTEM, tool_doc_text(functions)]
    if category.startswith("multi_turn") and sample.get("initial_config"):
        # Keep this bounded so long-context rows remain usable under max_prompt_length.
        sections.append("**Initial Environment State**\n" + json.dumps(sample["initial_config"], ensure_ascii=False)[:6000])
    sections.append("**User Request**\n" + flatten_question(sample.get("question", [])))
    sections.append("Remember: output <tool_call> JSON only when a tool is needed.")
    return "\n\n".join(sections)


def make_record(sample: dict, ground_truth, category: str, index: int, multiturn_docs: dict[str, dict]) -> dict:
    input_text = flatten_question(sample.get("question", []))
    return {
        "data_source": f"bfcl_v4/{category}",
        "prompt": [{"role": "user", "content": make_prompt(sample, category, multiturn_docs)}],
        "ability": "tooluse",
        # Store as a string to keep parquet schema stable across BFCL categories.
        "reward_model": {"style": "rule", "ground_truth": json.dumps(ground_truth, ensure_ascii=False)},
        # Keep schema exactly aligned with ToolRL/RLLA so HuggingFace
        # datasets.concatenate_datasets can concatenate val files.
        "extra_info": {
            "index": index,
            "input": input_text,
            "instruction": make_prompt(sample, category, multiturn_docs),
            "output": json.dumps(ground_truth, ensure_ascii=False),
            "split": "test",
        },
    }


def build_records(data_root: Path) -> tuple[list[dict], dict[str, dict]]:
    possible_root = data_root / "possible_answer"
    multiturn_docs = load_multiturn_docs(data_root)
    records = []
    stats = {}

    for category in CATEGORIES_WITH_PA:
        question_file = data_root / f"BFCL_v4_{category}.json"
        answer_file = possible_root / f"BFCL_v4_{category}.json"
        if not question_file.exists() or not answer_file.exists():
            stats[category] = {"rows": 0, "missing_questions": None, "skipped": True}
            continue
        questions = {row["id"]: row for row in read_jsonl(question_file)}
        before = len(records)
        missing = 0
        for answer in read_jsonl(answer_file):
            sample = questions.get(answer["id"])
            if sample is None:
                missing += 1
                continue
            records.append(make_record(sample, answer.get("ground_truth", []), category, len(records), multiturn_docs))
        stats[category] = {"rows": len(records) - before, "missing_questions": missing, "skipped": False}

    for category in NO_CALL_CATEGORIES:
        question_file = data_root / f"BFCL_v4_{category}.json"
        if not question_file.exists():
            stats[category] = {"rows": 0, "missing_questions": None, "skipped": True}
            continue
        before = len(records)
        for sample in read_jsonl(question_file):
            records.append(make_record(sample, [], category, len(records), multiturn_docs))
        stats[category] = {"rows": len(records) - before, "missing_questions": 0, "skipped": False}

    return records, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bfcl-data-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--smoke-size", type=int, default=32)
    args = parser.parse_args()

    records, stats = build_records(args.bfcl_data_root)
    if not records:
        raise SystemExit("No BFCL records generated")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    df.to_parquet(args.output_dir / "test.parquet", index=False)
    df.head(args.smoke_size).to_parquet(args.output_dir / "smoke_test.parquet", index=False)
    (args.output_dir / "README.json").write_text(json.dumps({
        "source": str(args.bfcl_data_root),
        "num_rows": len(df),
        "categories": stats,
        "excluded": EXCLUDED,
    }, ensure_ascii=False, indent=2))
    print(json.dumps({"rows": len(df), "output_dir": str(args.output_dir), "stats": stats}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
