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
"""
Reward scoring for Multiple Choice Questions (MCQ).
Extracts answer from XML tags and compares with ground truth.
"""

import re


def extract_answer_from_xml(solution_str: str) -> str:
    """
    Extract answer from <answer>...</answer> XML tags.
    
    Args:
        solution_str: Model output containing XML tags
        
    Returns:
        Extracted answer string (e.g., "A", "B", "C", "D") or empty string if not found
    """
    # Try to extract from <answer> tags
    match = re.search(r'<answer>\s*([A-D])\s*</answer>', solution_str, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).upper()
    
    # Fallback: try to find standalone answer pattern at the end
    match = re.search(r'\b([A-D])\s*$', solution_str.strip(), re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    return ""


def is_correct_format(solution_str: str) -> bool:
    """
    Check if the solution follows the expected XML format.
    
    Args:
        solution_str: Model output to validate
        
    Returns:
        True if format is correct, False otherwise
    """
    # Check for both <reasoning> and <answer> tags
    has_reasoning = bool(re.search(r'<reasoning>.*?</reasoning>', solution_str, re.DOTALL | re.IGNORECASE))
    has_answer = bool(re.search(r'<answer>.*?</answer>', solution_str, re.DOTALL | re.IGNORECASE))
    
    return has_reasoning and has_answer


def compute_score(solution_str: str, ground_truth: str) -> dict:
    """
    Compute reward score for MCQ by exact match.
    
    Args:
        solution_str: Model's generated solution
        ground_truth: Correct answer (A, B, C, or D)
        
    Returns:
        Dictionary containing:
            - score: 1.0 if correct, 0.0 otherwise
            - acc: Same as score (for compatibility)
            - pred: Predicted answer
            - incorrect_format: 1 if format is wrong, 0 otherwise
            - feedback: Empty string (no detailed feedback for MCQ)
    """
    # Extract predicted answer
    predicted_answer = extract_answer_from_xml(solution_str)
    
    # Check format
    format_correct = is_correct_format(solution_str)
    
    # Compute reward: 1.0 if correct, 0.0 otherwise
    reward = 1.0 if (predicted_answer == ground_truth.upper() and format_correct) else 0.0
    
    return {
        "score": reward,
        "acc": reward,
        "pred": predicted_answer,
        "incorrect_format": 0 if format_correct else 1,
        "feedback": "",  # MCQ doesn't provide detailed feedback
    }


if __name__ == "__main__":
    # Test cases
    test_cases = [
        {
            "solution": "<reasoning>This is my reasoning...</reasoning>\n<answer>A</answer>",
            "ground_truth": "A",
            "expected_score": 1.0,
        },
        {
            "solution": "<reasoning>Wrong reasoning...</reasoning>\n<answer>B</answer>",
            "ground_truth": "A",
            "expected_score": 0.0,
        },
        {
            "solution": "The answer is A",  # Wrong format
            "ground_truth": "A",
            "expected_score": 0.0,
        },
        {
            "solution": "<reasoning>Correct reasoning</reasoning>\n<answer> C </answer>",  # With spaces
            "ground_truth": "C",
            "expected_score": 1.0,
        },
    ]
    
    print("Running MCQ reward scoring tests...")
    for i, test in enumerate(test_cases):
        result = compute_score(test["solution"], test["ground_truth"])
        passed = result["score"] == test["expected_score"]
        status = "✓" if passed else "✗"
        print(f"Test {i+1}: {status} (score={result['score']}, expected={test['expected_score']})")
        if not passed:
            print(f"  Solution: {test['solution'][:50]}...")
            print(f"  Result: {result}")
