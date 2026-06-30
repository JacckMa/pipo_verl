# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
import json
import logging
import os
import traceback


def _sample_test_cases(test_cases, max_cases=0, mode="hardest"):
    """Return a deterministic subset of test cases while preserving input/output alignment."""
    try:
        max_cases = int(max_cases or 0)
    except (TypeError, ValueError):
        max_cases = 0
    if max_cases <= 0:
        return test_cases

    inputs = test_cases.get("inputs") or []
    outputs = test_cases.get("outputs") or []
    num_cases = min(len(inputs), len(outputs))
    if num_cases <= max_cases:
        return test_cases

    if mode == "first":
        selected = list(range(max_cases))
    else:
        # Cheap hard-case heuristic: longer stdin often means heavier tests.
        selected = sorted(range(num_cases), key=lambda i: len(str(inputs[i])), reverse=True)[:max_cases]
        selected = sorted(selected)

    sampled = dict(test_cases)
    for key in ("inputs", "outputs", "assert_case"):
        values = test_cases.get(key)
        if isinstance(values, list) and len(values) >= num_cases:
            sampled[key] = [values[i] for i in selected]
    sampled["_sampled_case_indices"] = selected
    return sampled


from .utils import check_correctness

"""
Verify code correctness using the Sandbox Fusion (https://github.com/bytedance/SandboxFusion).
You can either deploy the sandbox_fusion service yourself or use the
FaaS service provided by public cloud, eg: volcengine.com.
"""
logger = logging.getLogger(__name__)


def _is_read_timeout_error(value):
    text = str(value).lower()
    return "read timed out" in text or "readtimeout" in text or "read timeout" in text


def _metadata_has_read_timeout(metadata_list):
    return any(
        isinstance(item, dict)
        and (
            _is_read_timeout_error(item.get("api_request_error"))
            or _is_read_timeout_error(item.get("error"))
            or _is_read_timeout_error(item)
        )
        for item in metadata_list or []
    )


def compute_score(
    sandbox_fusion_url,
    concurrent_semaphore,
    memory_limit_mb,
    completion,
    test_cases,
    continuous=False,
    timeout=10,
    case_sample_size=0,
    case_sample_mode="hardest",
):
    """
    Computes the code score using the remote sandbox API.

    Args:
        sandbox_fusion_url: The URL of the sandbox_fusion service, eg: "https://<your service endpoint>/run_code"

        completion: The completion string containing the code.
        test_cases: JSON string or dictionary containing "inputs" and "outputs".
        continuous: Whether to compute a continuous score (based on the first N test cases).
        timeout: Timeout for each test case.

    Returns:
        A tuple (score, metadata_list).
        score: Float score (0.0 to 1.0).
        metadata_list: List containing execution metadata for each test case.
    """
    solution = completion
    if "```python" in completion:
        solution = completion.split("```python")[-1].split("```")[0]
    elif "```" in completion:
        # Handle cases like ```\ncode\n```
        parts = completion.split("```")
        if len(parts) >= 2:
            solution = parts[1]
            # Remove potential language specifier like 'python\n'
            if "\n" in solution:
                first_line, rest = solution.split("\n", 1)
                if first_line.strip().isalpha():  # Simple check for language name
                    solution = rest
    else:
        return 0.0, [{"error": "Invalid completion (missing code block)"}]

    try:
        if not isinstance(test_cases, dict):
            try:
                test_cases = json.loads(test_cases)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse test_cases JSON: {e}")
                return 0.0, [{"error": "Invalid test_cases JSON format"}]

        if test_cases is not None and "assert_case" in test_cases and isinstance(test_cases.get("assert_case"), list):
            assert_cases = test_cases.get("assert_case")
            test_cases.setdefault("inputs", ["" for _ in assert_cases])
            test_cases.setdefault("outputs", [None for _ in assert_cases])
        elif not test_cases or "inputs" not in test_cases or "outputs" not in test_cases:
            logger.error("Invalid test_cases structure.")
            return 0.0, [{"error": "Invalid test_cases structure (missing inputs/outputs)"}]

        test_cases = _sample_test_cases(test_cases, case_sample_size, case_sample_mode)

        # Check selected/all test cases
        # Note: The return value of check_correctness might need adaptation here
        # Assume check_correctness returns (results_list, metadata_list)
        # results_list contains True, False, or error codes (-1, -2, -3, etc.)
        res_list, metadata_list = check_correctness(
            sandbox_fusion_url=sandbox_fusion_url,
            in_outs=test_cases,
            generation=solution,
            timeout=timeout,
            concurrent_semaphore=concurrent_semaphore,
            memory_limit_mb=memory_limit_mb,
        )

        # A single sandbox HTTP read timeout is a bad completion, not a trainer-level failure.
        # Keep fail-fast available for debugging via HACPO_FAIL_ON_SANDBOX_ERROR=1.
        fail_on_sandbox_error = os.environ.get("HACPO_FAIL_ON_SANDBOX_ERROR", "0") != "0"
        infra_error_statuses = {"api_error", "sandbox_error", "unknown_api_state", "internal_error"}
        bad_metadata = [m for m in metadata_list if isinstance(m, dict) and m.get("status") in infra_error_statuses]
        if _metadata_has_read_timeout(metadata_list):
            logger.warning("Sandbox read timeout while scoring code; assigning reward 0 for this completion.")
            return 0.0, metadata_list
        if fail_on_sandbox_error and bad_metadata:
            first_error = bad_metadata[0].get("api_request_error") or bad_metadata[0].get("error") or bad_metadata[0]
            raise RuntimeError(f"Sandbox infrastructure error while scoring code: {first_error}")

        # Calculate score
        if not res_list:  # If there are no results (e.g., invalid input)
            return 0.0, metadata_list

        if continuous:
            # Calculate pass rate for the first N (e.g., 10) test cases
            num_to_consider = min(len(res_list), 10)
            if num_to_consider == 0:
                score = 0.0
            else:
                passed_count = sum(1 for r in res_list[:num_to_consider] if r is True)
                score = passed_count / num_to_consider
            # Return all metadata, even if score is based on the first N
            final_metadata = metadata_list
        else:
            # Calculate pass rate for all test cases
            passed_count = sum(1 for r in res_list if r is True)
            total_cases = len(res_list)
            score = passed_count / total_cases if total_cases > 0 else 0.0
            final_metadata = metadata_list

    except Exception as e:
        final_metadata = metadata_list if "metadata_list" in locals() else [{"error": f"Unhandled exception: {e}"}]
        if _is_read_timeout_error(e) or _metadata_has_read_timeout(final_metadata):
            logger.warning("Sandbox read timeout escaped compute_score; assigning reward 0 for this completion: %s", e)
            score = 0.0
        elif os.environ.get("HACPO_FAIL_ON_SANDBOX_ERROR", "0") != "0":
            logger.error(f"Error during compute_score: {e}")
            traceback.print_exc()
            raise
        else:
            logger.error(f"Error during compute_score: {e}")
            traceback.print_exc()
            score = 0.0
        # Try to return partial metadata if available, otherwise return error info

    # Ensure float and list are returned
    return float(score), final_metadata if isinstance(final_metadata, list) else [final_metadata]
