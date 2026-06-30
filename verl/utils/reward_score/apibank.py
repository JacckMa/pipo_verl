# Copyright 2026
"""Reward for API-Bank-RLVR style tool-call tasks."""

from __future__ import annotations

import ast
import json
import re
from typing import Any


def _loads_maybe(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        try:
            return ast.literal_eval(text)
        except Exception:
            return None


def _normalize(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip().lower()
    if isinstance(value, list):
        return [_normalize(v) for v in value]
    if isinstance(value, dict):
        return {str(k).strip().lower(): _normalize(v) for k, v in value.items() if v is not None}
    return value


def _iter_json_objects(text: str):
    decoder = json.JSONDecoder()
    for start, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[start:])
        except Exception:
            continue
        if isinstance(obj, dict):
            yield obj


def _extract_tool_call(solution: str) -> dict[str, Any] | None:
    if not isinstance(solution, str):
        return None
    candidates: list[str] = []
    blocks = re.findall(r"<tool_call>(.*?)</tool_call>", solution, flags=re.I | re.S)
    candidates.extend(blocks)
    fenced = re.findall(r"```(?:json|plaintext)?\s*(.*?)```", solution, flags=re.I | re.S)
    candidates.extend(fenced)
    candidates.append(solution)

    for chunk in candidates:
        parsed = _loads_maybe(chunk)
        if isinstance(parsed, dict) and "name" in parsed:
            return parsed
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict) and "name" in item:
                    return item
        for obj in _iter_json_objects(chunk):
            if "name" in obj:
                return obj
    return None


def compute_score(solution_str: str, ground_truth: Any) -> float:
    expected = _loads_maybe(ground_truth)
    if not isinstance(expected, dict):
        return 0.0
    pred = _extract_tool_call(solution_str)
    if not isinstance(pred, dict):
        return 0.0

    expected_name = str(expected.get("name", "")).strip().lower()
    pred_name = str(pred.get("name", "")).strip().lower()
    if not expected_name or pred_name != expected_name:
        return 0.0

    expected_params = expected.get("parameters") or {}
    pred_params = pred.get("parameters") or {}
    if not isinstance(expected_params, dict) or not isinstance(pred_params, dict):
        return 0.5

    required = {k: v for k, v in expected_params.items() if v is not None}
    pred_non_null = {k: v for k, v in pred_params.items() if v is not None}
    if not required:
        return 1.0 if not pred_non_null else 0.8

    correct = 0
    for key, exp_value in required.items():
        if key in pred_params and _normalize(pred_params[key]) == _normalize(exp_value):
            correct += 1
    param_score = correct / max(1, len(required))

    # Penalize hallucinated non-null parameters that are not required by the gold call.
    extra_non_null = [k for k, v in pred_non_null.items() if k not in required]
    extra_penalty = min(0.25, 0.05 * len(extra_non_null))
    score = 0.5 + 0.5 * param_score - extra_penalty
    return float(max(0.0, min(1.0, score)))
