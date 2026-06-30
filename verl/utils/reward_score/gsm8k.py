# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import math

_SOLUTION_CLIP_CHARS = 300


def _last_boxed_only_string(string: str) -> str | None:
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    left_brace_idx = None
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
            if left_brace_idx is None:
                left_brace_idx = i
        elif string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if left_brace_idx is None or right_brace_idx is None:
        return None

    return string[left_brace_idx + 1 : right_brace_idx].strip()


def _clean_numeric_string(s: str) -> str:
    # remove currency/commas/spaces
    return s.replace(",", "").replace("$", "").strip()


def extract_solution(solution_str, method="flexible"):
    assert method in ["strict", "flexible"]

    # Optimization: only search the tail of long generations
    tail = solution_str[-_SOLUTION_CLIP_CHARS:] if len(solution_str) > _SOLUTION_CLIP_CHARS else solution_str

    # 1) Prefer a boxed final answer if present
    boxed = _last_boxed_only_string(tail)
    if boxed is not None and any(ch.isdigit() for ch in boxed):
        return _clean_numeric_string(boxed)

    # 2) Strict GSM8K pattern: #### <number>
    solutions = re.findall(r"#### (\\-?[0-9\\.\\,]+)", tail)
    if solutions:
        return _clean_numeric_string(solutions[-1])

    # 3) Flexible: take the last numeric token near the end
    if method == "flexible":
        candidates = re.findall(r"(\\-?[0-9]+(?:\\.[0-9]+)?)", tail)
        for token in reversed(candidates):
            token = token.strip()
            if token and token != ".":
                return _clean_numeric_string(token)

    return None


def compute_score(
    solution_str,
    ground_truth,
    method="flexible",
    format_score=0.0,
    score=1.0,
):
    """The scoring function for GSM8k.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual
    Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str, method=method)
    if answer is None:
        return 0.0

    # Exact string match (after simple cleaning)
    # ground_truth may be a list like ['18']; handle gracefully
    if isinstance(ground_truth, (list, tuple)):
        ground_truth = ground_truth[-1] if len(ground_truth) > 0 else ""
    gt_clean = _clean_numeric_string(str(ground_truth))
    if answer == gt_clean:
        return float(score)

    # Try numeric partial credit based on relative error
    def to_float_safe(x: str) -> float | None:
        try:
            return float(x)
        except Exception:
            return None

    pred_val = to_float_safe(answer)
    gt_val = to_float_safe(gt_clean)

    if pred_val is not None and gt_val is not None:
        # Relative error based shaping (continuous)
        denom = max(1.0, abs(gt_val))
        rel_err = abs(pred_val - gt_val) / denom

        # Smooth mapping: r = 1 / (1 + k * rel_err)
        # k controls steepness; choose k=5 so 20% err -> ~0.5, 100% err -> ~0.167
        k = 5.0
        shaped = 1.0 / (1.0 + k * rel_err)
        shaped = max(float(format_score), min(float(score), shaped))
        return shaped

    # Non-numeric mismatch: at least give format credit if format looked okay
    return float(format_score)
