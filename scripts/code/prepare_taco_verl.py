#!/usr/bin/env python3
"""Convert raw BAAI/TACO parquet shards into the VERL code-RL format."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = PROJECT_ROOT / "dataset" / "taco_raw"
DEFAULT_OUT_DIR = PROJECT_ROOT / "dataset" / "taco_verl"
DEFAULT_SHARD_SIZE = 1000

PROMPT_PREFIX = (
    "You are a coding expert. You will be given a coding problem, and you need "
    "to write a correct Python program that matches the specification and passes "
    "all tests. The time limit is 1 second. You may start by outlining your "
    "thought process. In the end, please provide the complete code in a code "
    "block enclosed with ``` ```.\n\n"
)


def parse_solutions(value) -> list[str]:
    if isinstance(value, list):
        return value
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = ast.literal_eval(value)
    except Exception:
        return []
    return parsed if isinstance(parsed, list) else []


def parse_cases(value, max_cases: int | None) -> str | None:
    if isinstance(value, str):
        try:
            cases = json.loads(value)
        except Exception:
            return None
    elif isinstance(value, dict):
        cases = dict(value)
    else:
        return None

    inputs = cases.get("inputs")
    outputs = cases.get("outputs")
    if not isinstance(inputs, list) or not isinstance(outputs, list):
        return None
    if len(inputs) == 0 or len(outputs) == 0:
        return None
    n = min(len(inputs), len(outputs))
    if max_cases is not None:
        n = min(n, max_cases)
    cases["inputs"] = inputs[:n]
    cases["outputs"] = outputs[:n]
    return json.dumps(cases, ensure_ascii=False)


def convert_frame(df: pd.DataFrame, split: str, max_rows: int | None, max_cases: int | None) -> pd.DataFrame:
    rows = []
    for raw_index, row in df.iterrows():
        question = str(row.get("question") or "").strip()
        if not question:
            continue

        tags = str(row.get("raw_tags") or row.get("tags") or "").lower()
        if "interactive" in tags or "interactive problem" in question.lower():
            continue

        ground_truth = parse_cases(row.get("input_output"), max_cases)
        if ground_truth is None:
            continue

        solutions = parse_solutions(row.get("solutions"))
        if split == "train" and not solutions:
            # Empty-solution train rows in TACO are often noisy/interactive style tasks.
            continue

        prompt = PROMPT_PREFIX + question
        source = row.get("source")
        rows.append(
            {
                "prompt": [{"role": "user", "content": prompt}],
                "embedding": [],
                "data_source": "taco",
                "ability": "code",
                "reward_model": {"ground_truth": ground_truth},
                "extra_info": {
                    "achievement_prior": 0,
                    "description": question[:2000],
                    "difficulty": row.get("difficulty"),
                    "index": str(raw_index),
                    "name": row.get("name"),
                    "source": source,
                    "split": split,
                    "url": row.get("url"),
                },
            }
        )
        if max_rows is not None and len(rows) >= max_rows:
            break
    return pd.DataFrame(rows)


def read_train(raw_dir: Path) -> pd.DataFrame:
    shards = sorted(raw_dir.glob("train-*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no train shards found under {raw_dir}")
    return pd.concat((pd.read_parquet(path) for path in shards), ignore_index=True)


def write_shards(df: pd.DataFrame, out_dir: Path, split: str, shard_size: int) -> list[Path]:
    shard_dir = out_dir / f"{split}_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    for old in shard_dir.glob("*.parquet"):
        old.unlink()

    paths = []
    for start in range(0, len(df), shard_size):
        shard = df.iloc[start : start + shard_size]
        path = shard_dir / f"{split}-{len(paths):05d}.parquet"
        shard.to_parquet(path, index=False)
        paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--shard-size", type=int, default=DEFAULT_SHARD_SIZE)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_df = convert_frame(read_train(args.raw_dir), "train", args.max_train, args.max_cases)
    test_path = args.raw_dir / "test-00000-of-00001.parquet"
    test_df = convert_frame(pd.read_parquet(test_path), "test", args.max_test, args.max_cases)

    train_out = args.out_dir / "train.parquet"
    test_out = args.out_dir / "test.parquet"
    train_df.to_parquet(train_out, index=False)
    test_df.to_parquet(test_out, index=False)
    print(f"wrote {train_out} rows={len(train_df)}")
    print(f"wrote {test_out} rows={len(test_df)}")
    if args.shard_size > 0:
        train_shards = write_shards(train_df, args.out_dir, "train", args.shard_size)
        print(f"wrote {len(train_shards)} train shards under {args.out_dir / 'train_shards'}")


if __name__ == "__main__":
    main()
