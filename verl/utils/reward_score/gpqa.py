
import re


def compute_score(solution_str, ground_truth):
    text = "" if solution_str is None else str(solution_str)
    gold = str(ground_truth).strip().upper()
    patterns = [
        r"\\boxed\{\s*([A-D])\s*\}",
        r"(?i)answer\s*[:：]\s*\$?\s*([A-D])\s*\$?",
        r"(?i)final\s+answer\s*(?:is|:)?\s*\$?\s*([A-D])\s*\$?",
        r"(?m)^\s*([A-D])\s*$",
    ]
    for pattern in patterns:
        m = re.search(pattern, text)
        if m:
            return 1.0 if m.group(1).upper() == gold else 0.0
    letters = re.findall(r"\b([A-D])\b", text.upper())
    if letters:
        return 1.0 if letters[-1] == gold else 0.0
    return 0.0
