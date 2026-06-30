# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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
# Adapted from https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hendrycks_math/utils.py

import signal
import ast
import re
import numpy as np
from typing import Optional, Tuple
from math_verify import parse as mv_parse, verify as mv_verify
from verl.utils.reward_score.math_dapo import normalize_final_answer

FORMAT_PENALTY = False


def last_boxed_only_string(string: str) -> Optional[str]:
    if not isinstance(string, str):
        string = str(string) if string is not None else ""
    
    # Handle both escaped \\boxed{ and plain \boxed{
    pattern = re.compile(r'\\+boxed\{', re.IGNORECASE)
    matches = list(pattern.finditer(string))
    if not matches:
        return None
    
    # Get the position of the last match
    idx = matches[-1].start()
    i = idx
    right_brace_idx = None
    num_left_braces_open = 0

    # Precisely match nested braces to support complex formulas
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    return string[idx : right_brace_idx + 1] if right_brace_idx is not None else None


def remove_boxed(s: str) -> str:
    """Strictly remove the outer \boxed{} wrapper, handling any number of escaped backslashes."""
    if not s:
        return ""
    if not isinstance(s, str):
        s = str(s)
    
    # Match any leading backslashes + boxed{ at start, and closing } at end
    match = re.match(r'^\\+boxed\{(.*)\}$', s, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


class timeout:
    def __init__(self, seconds=5, error_message="Timeout"):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


def is_correct_strict_box(
    pred: str, gt: str, pause_tokens_index: Optional[list[int]] = None
) -> Tuple[bool, Optional[str]]:
    """Strictly match content extracted from \boxed{}, fixing truncation issues. Normalize pred and gt before comparison."""
    if not isinstance(pred, str):
        pred = str(pred) if pred is not None else ""

    # Fix: if no pause_tokens, take the last 200 characters of the full text to avoid missing matches in long texts
    if pause_tokens_index is not None and len(pause_tokens_index) == 4:
        pred = pred[pause_tokens_index[-1] - 200 :]
    else:
        pred = pred[-200:]  # expanded to 200 chars to avoid missing matches

    boxed_pred = last_boxed_only_string(pred)
    extracted_pred = remove_boxed(boxed_pred) if boxed_pred is not None else None

    if extracted_pred is None:
        return False, None

    # Key fix: normalize both pred and gt before comparison
    # This handles differences in \left/\right, extra spaces, \frac, etc.
    norm_pred = normalize_final_answer(extracted_pred)
    norm_gt = normalize_final_answer(gt)

    return norm_pred == norm_gt, extracted_pred


def extract_gt_answer(raw_gt) -> str:
    """Dedicated fix for GT extraction, compatible with numpy array / list / str / None formats."""
    if raw_gt is None:
        return ""

    # Step 0: support numpy array
    if isinstance(raw_gt, np.ndarray):
        raw_gt = raw_gt.tolist()

    # Step 1: if list, take the first element
    if isinstance(raw_gt, list):
        if len(raw_gt) > 0:
            cleaned_gt = raw_gt[0]
        else:
            return ""
    else:
        # If string, strip outer single/double quotes
        cleaned_gt = str(raw_gt).strip().strip("'\"")

        # Handle string-formatted lists (e.g. "['xxx']" / '["xxx"]')
        if cleaned_gt.startswith("[") and cleaned_gt.endswith("]"):
            try:
                gt_list = ast.literal_eval(cleaned_gt)
                if isinstance(gt_list, list) and len(gt_list) > 0:
                    cleaned_gt = gt_list[0]
            except Exception:
                pass

    # Step 2: try to extract pure answer from \boxed{}
    boxed_gt = last_boxed_only_string(str(cleaned_gt))
    final_gt = remove_boxed(boxed_gt) if boxed_gt else ""

    # Step 3: if no \boxed{}, use the cleaned answer directly (handles pure answer formats like AIME/Minerva)
    if not final_gt:
        final_gt = str(cleaned_gt).strip().strip("'\"")

    return final_gt

def verify(
    solution_str: str, answer: str, pause_tokens_index: Optional[list[int]] = None
) -> Tuple[bool, Optional[str]]:
    """Core verification logic, fixing all matching issues with full None checks."""
    # Initialize default values
    correct = False
    pred = None
    
    try:
        # ------------------------------
        # 1. Force extract GT pure answer to completely solve GT format issues
        # ------------------------------
        original_gt = answer
        final_gt = extract_gt_answer(original_gt)

        # ------------------------------
        # Key debug prints (check these to locate issues)
        # ------------------------------
        # print(f"\n[DEBUG] --------------- 验证步骤 ---------------")
        # print(f"[DEBUG] 原始GT类型: {type(original_gt)}")
        # print(f"[DEBUG] 原始GT长度: {len(str(original_gt)) if original_gt else 0}")
        # print(f"[DEBUG] 最终提取GT答案: '{final_gt}'")
        # print(f"[DEBUG] 模型输出末尾200字符: '{str(solution_str)[-200:] if solution_str else ''}'")

        # If GT extraction fails, return False directly
        if not final_gt:
            # print(f"[DEBUG] 警告：GT答案提取失败！")
            return False, None

        # ------------------------------
        # 2. Extract predicted answer and match
        # ------------------------------
        correct, pred = is_correct_strict_box(solution_str, final_gt, pause_tokens_index)
        # print(f"[DEBUG] 提取预测答案: '{pred}'")
        # print(f"[DEBUG] 严格字符串匹配结果: {correct}")

        if pred is None:
            pred = ""

        # If string matching fails, try mathematical semantic equivalence verification
        if not correct and pred != "" and final_gt != "":
            try:
                with timeout(seconds=5):
                    gold_expr = mv_parse(final_gt)
                    pred_expr = mv_parse(pred)
                    correct = mv_verify(gold_expr, pred_expr)
                    # Ensure math_verify returns a boolean
                    correct = bool(correct)
                    # print(f"[DEBUG] 数学语义验证结果: {correct}")
            except Exception as e:
                # print(f"[DEBUG] 数学验证异常: {type(e).__name__}: {e}")
                correct = False

        # print(f"[DEBUG] 最终是否正确: {correct}")
    except Exception as e:
        # print(f"[DEBUG] 验证流程异常: {type(e).__name__}: {e}")
        correct = False
        pred = None
    
    return correct, pred


def compute_score(
    solution_str: str,
    ground_truth: str,
    extra_info=None,
    pause_tokens_index: Optional[list[int]] = None,
    format_feedback: bool = True,
    correctness_feedback: bool = False,
) -> dict:
    """Compute the final reward score, ensuring all numeric metrics are never None."""
    # 1. Initialize all return values with absolutely safe default values
    result = {
        "score": 0.0,
        "acc": 0.0,
        "pred": "",
        "incorrect_format": 0,
        "truncated": 0,
        "truncated_and_missing_answer": 0,
        "feedback": "",
    }
    
    try:
        if extra_info is None:
            extra_info = {}
        split = extra_info.get("split", "test")
        was_truncated = bool(extra_info.get("truncated", False))

        # 2. Core verification
        correct, pred = verify(solution_str, ground_truth, pause_tokens_index)

        # 3. Fill in results (forced type conversion)
        result["acc"] = 1.0 if correct else 0.0
        result["score"] = result["acc"]
        result["pred"] = pred if pred is not None else ""
        result["incorrect_format"] = 1 if (result["pred"] == "") else 0
        result["truncated"] = 1 if was_truncated else 0
        result["truncated_and_missing_answer"] = 1 if (result["incorrect_format"] and result["truncated"]) else 0

        # 4. Format penalty for training set (disabled by default)
        if FORMAT_PENALTY and split == "train" and result["incorrect_format"] and (not result["truncated"]):
            result["score"] -= 0.5

        # 5. Feedback messages
        if result["incorrect_format"] and not result["truncated"] and format_feedback:
            result["feedback"] = "Your answer had the wrong format. The solution must be given in the format: \\boxed{your_answer}."
        elif result["truncated"] and format_feedback:
            result["feedback"] = "Your response was truncated because it exceeded the maximum length."
        elif not correct and correctness_feedback:
            result["feedback"] = f"Your answer is incorrect. The correct answer is {ground_truth}."
    except Exception as e:
        pass  # Keep default values unchanged on exception
    
    # 6. Final safety check: iterate all fields to ensure numeric types are absolutely not None
    for key in result:
        if key in ["score", "acc"]:
            result[key] = float(result[key]) if result[key] is not None else 0.0
        elif key in ["incorrect_format", "truncated", "truncated_and_missing_answer"]:
            result[key] = int(result[key]) if result[key] is not None else 0
        elif result[key] is None:
            result[key] = "" if key == "feedback" else 0
    
    # print(f"[DEBUG] compute_score 最终返回: {result}")
    return result
