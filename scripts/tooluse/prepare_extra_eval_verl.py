
import json
import os
import random
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = os.environ.get("PROJECT_ROOT", str(PROJECT_ROOT))
DATASETS = os.environ.get("DATASETS_ROOT", str(PROJECT_ROOT / "dataset" / "raw"))


def write_ifeval():
    src = os.path.join(DATASETS, "huggingface.co/datasets/google/IFEval/ifeval_input_data.jsonl")
    rows = []
    with open(src, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            obj = json.loads(line)
            rows.append(
                {
                    "data_source": "ifeval",
                    "prompt": [{"role": "user", "content": obj["prompt"]}],
                    "ability": "InstructionFollowing",
                    "reward_model": {
                        "style": "rule",
                        "ground_truth": json.dumps(
                            {
                                "instruction_id_list": obj["instruction_id_list"],
                                "kwargs": obj["kwargs"],
                            },
                            ensure_ascii=False,
                        ),
                    },
                    "extra_info": {
                        "index": int(idx),
                        "key": int(obj.get("key", idx)),
                        "split": "test",
                        "domain": "",
                        "record_id": "",
                    },
                }
            )
    out_dir = os.path.join(ROOT, "data/ifeval_verl")
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(rows).to_parquet(os.path.join(out_dir, "test.parquet"), index=False)
    pd.DataFrame(rows[:16]).to_parquet(os.path.join(out_dir, "smoke_test.parquet"), index=False)
    print("wrote", out_dir, len(rows))


def write_gpqa():
    src = os.path.join(DATASETS, "huggingface.co/datasets/Idavidrein/gpqa/gpqa_diamond.csv")
    df = pd.read_csv(src)
    rows = []
    template = (
        "{question}\n\n"
        "A. {a}\nB. {b}\nC. {c}\nD. {d}\n\n"
        "Please reason step by step, and put your final answer (only the choice letter) within \\boxed{{}}."
    )
    for idx, ex in df.iterrows():
        choices = [
            str(ex["Incorrect Answer 1"]).strip(),
            str(ex["Incorrect Answer 2"]).strip(),
            str(ex["Incorrect Answer 3"]).strip(),
        ]
        rng = random.Random(idx)
        rng.shuffle(choices)
        gold_index = rng.randint(0, 3)
        choices.insert(gold_index, str(ex["Correct Answer"]).strip())
        gold = "ABCD"[gold_index]
        prompt = template.format(question=str(ex["Question"]).strip(), a=choices[0], b=choices[1], c=choices[2], d=choices[3])
        rows.append(
            {
                "data_source": "gpqa-diamond",
                "prompt": [{"role": "user", "content": prompt}],
                "ability": "General",
                "reward_model": {"style": "rule", "ground_truth": gold},
                "extra_info": {
                    "index": int(idx),
                    "key": int(idx),
                    "split": "test",
                    "domain": str(ex.get("High-level domain", "")),
                    "record_id": str(ex.get("Record ID", "")),
                },
            }
        )
    out_dir = os.path.join(ROOT, "data/gpqa_diamond_verl")
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(rows).to_parquet(os.path.join(out_dir, "test.parquet"), index=False)
    pd.DataFrame(rows[:16]).to_parquet(os.path.join(out_dir, "smoke_test.parquet"), index=False)
    print("wrote", out_dir, len(rows))


write_ifeval()
write_gpqa()
