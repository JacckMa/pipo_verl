#!/usr/bin/env python3
"""Prepare SciKnowEval splits for PIPO reward routing.

The reward router expects subject-specific data_source values such as
"sciknoweval_chemistry", so this script copies the split and rewrites that field.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

TASKS = ("biology", "chemistry", "material", "physics")
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src-root",
        type=Path,
        default=PROJECT_ROOT / "dataset" / "sciknoweval_raw",
    )
    parser.add_argument(
        "--dst-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "dataset" / "sciknoweval_sdpo_split",
    )
    parser.add_argument("--tasks", nargs="*", default=list(TASKS), choices=TASKS)
    args = parser.parse_args()

    args.dst_root.mkdir(parents=True, exist_ok=True)
    for task in args.tasks:
        src_dir = args.src_root / task
        dst_dir = args.dst_root / task
        dst_dir.mkdir(parents=True, exist_ok=True)
        for split in ("train", "test"):
            src = src_dir / f"{split}.parquet"
            if not src.exists():
                raise FileNotFoundError(src)
            df = pd.read_parquet(src)
            df = df.copy()
            df["data_source"] = f"sciknoweval_{task}"
            out = dst_dir / f"{split}.parquet"
            df.to_parquet(out, index=False)
            print(f"wrote {out} rows={len(df)} data_source=sciknoweval_{task}")


if __name__ == "__main__":
    main()
