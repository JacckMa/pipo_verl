# Copyright 2026
"""Offline reward for static tool/function calling tasks.

Supports the tool-call formats used by ToolRL/RLLA, API-Bank, BFCL, and the
older pipo tooluse data. The scorer intentionally stays offline: it compares
function names and arguments against gold calls instead of executing tools.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any


def _loads_maybe(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
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
            # Plain strings such as MSFT or black are valid argument values, not
            # parse failures that should collapse to None.
            return value


def _normalize_name(name: Any) -> str:
    return str(name or "").strip().lower()


def _normalize_value(value: Any) -> Any:
    parsed = _loads_maybe(value)
    if parsed is not value:
        return _normalize_value(parsed)
    if isinstance(value, str):
        return " ".join(value.strip().lower().split())
    if isinstance(value, dict):
        return {str(k).strip().lower(): _normalize_value(v) for k, v in value.items() if v is not None}
    if isinstance(value, (list, tuple)):
        return [_normalize_value(v) for v in value]
    return value


def _as_params(value: Any) -> dict[str, Any]:
    value = _loads_maybe(value)
    if isinstance(value, dict):
        return value
    return {}


def _make_call(name: Any, params: Any = None) -> dict[str, Any] | None:
    name = str(name or "").strip()
    if not name:
        return None
    return {"name": name, "parameters": _as_params(params)}


def _iter_json_objects(text: str):
    decoder = json.JSONDecoder()
    for start, ch in enumerate(text):
        if ch not in "[{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[start:])
        except Exception:
            continue
        yield obj


def _parse_python_call(text: str) -> dict[str, Any] | None:
    """Parse BFCL-style call strings such as cd(folder='document')."""
    if not isinstance(text, str):
        return None
    text = text.strip().strip("`")
    if not text or "(" not in text or ")" not in text:
        return None
    try:
        expr = ast.parse(text, mode="eval").body
    except Exception:
        return None
    if not isinstance(expr, ast.Call):
        return None

    def name_of(node: ast.AST) -> str | None:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            base = name_of(node.value)
            return f"{base}.{node.attr}" if base else node.attr
        return None

    name = name_of(expr.func)
    if not name:
        return None
    params: dict[str, Any] = {}
    for idx, arg in enumerate(expr.args):
        try:
            params[f"arg{idx}"] = ast.literal_eval(arg)
        except Exception:
            params[f"arg{idx}"] = ast.unparse(arg) if hasattr(ast, "unparse") else str(arg)
    for kw in expr.keywords:
        if kw.arg is None:
            continue
        try:
            params[kw.arg] = ast.literal_eval(kw.value)
        except Exception:
            params[kw.arg] = ast.unparse(kw.value) if hasattr(ast, "unparse") else str(kw.value)
    return _make_call(name, params)


def _dict_to_calls(obj: Any) -> list[dict[str, Any]]:
    if not isinstance(obj, dict):
        return []

    # OpenAI / ToolRL / APIBank style.
    if "name" in obj:
        params = obj.get("parameters", obj.get("arguments", {}))
        call = _make_call(obj.get("name"), params)
        return [call] if call else []

    # pipo tooluse style.
    if "Action" in obj:
        call = _make_call(obj.get("Action"), obj.get("Action_Input", {}))
        return [call] if call else []

    # BFCL possible_answer style: {"func_name": {"arg": [accepted_values]}}.
    calls: list[dict[str, Any]] = []
    for name, params in obj.items():
        if isinstance(params, dict):
            call = _make_call(name, params)
            if call:
                calls.append(call)
    return calls


def _objects_to_calls(obj: Any) -> list[dict[str, Any]]:
    obj = _loads_maybe(obj)
    if obj is None:
        return []
    if isinstance(obj, dict):
        return _dict_to_calls(obj)
    if isinstance(obj, (list, tuple)):
        calls: list[dict[str, Any]] = []
        # BFCL multi-turn possible_answer is list[turn][call_string]. Flatten for
        # offline sequence-level scoring.
        for item in obj:
            if isinstance(item, str):
                call = _parse_python_call(item)
                if call:
                    calls.append(call)
            else:
                calls.extend(_objects_to_calls(item))
        return calls
    if isinstance(obj, str):
        call = _parse_python_call(obj)
        return [call] if call else []
    return []


def extract_calls(text_or_obj: Any) -> list[dict[str, Any]]:
    """Extract tool calls from XML tags, JSON snippets, or BFCL call strings."""
    direct = _objects_to_calls(text_or_obj)
    if direct:
        return direct
    if not isinstance(text_or_obj, str):
        return []

    text = text_or_obj.strip()
    blocks = re.findall(r"<tool_call>(.*?)</tool_call>", text, flags=re.I | re.S)
    fenced = re.findall(r"```(?:json|plaintext|python)?\s*(.*?)```", text, flags=re.I | re.S)
    chunks: list[str] = blocks + fenced
    if not chunks:
        chunks = [text]

    calls: list[dict[str, Any]] = []
    for chunk in chunks:
        parsed = _loads_maybe(chunk)
        parsed_calls = _objects_to_calls(parsed)
        if parsed_calls:
            calls.extend(parsed_calls)
            continue
        for obj in _iter_json_objects(chunk):
            calls.extend(_objects_to_calls(obj))
        for line in chunk.splitlines():
            call = _parse_python_call(line)
            if call:
                calls.append(call)
    return calls


def _value_matches(pred: Any, expected: Any) -> bool:
    pred_n = _normalize_value(pred)
    expected_n = _normalize_value(expected)

    # BFCL stores acceptable scalar values as lists. Treat list values as an OR
    # set unless the prediction is also a list and exactly matches the list.
    if isinstance(expected_n, list):
        if isinstance(pred_n, list) and pred_n == expected_n:
            return True
        return any(_value_matches(pred_n, option) for option in expected_n)
    if isinstance(expected_n, dict):
        if not isinstance(pred_n, dict):
            return False
        for key, exp_val in expected_n.items():
            if key not in pred_n or not _value_matches(pred_n[key], exp_val):
                return False
        return True
    return pred_n == expected_n


def _call_score(pred: dict[str, Any], expected: dict[str, Any]) -> float:
    if _normalize_name(pred.get("name")) != _normalize_name(expected.get("name")):
        return 0.0

    exp_params = _as_params(expected.get("parameters"))
    pred_params = _as_params(pred.get("parameters"))
    exp_params_n = {str(k).strip().lower(): v for k, v in exp_params.items() if v is not None}
    pred_params_n = {str(k).strip().lower(): v for k, v in pred_params.items() if v is not None}

    if not exp_params_n:
        return 1.0 if not pred_params_n else 0.85

    correct = 0
    for key, exp_val in exp_params_n.items():
        if key in pred_params_n and _value_matches(pred_params_n[key], exp_val):
            correct += 1
    param_score = correct / max(1, len(exp_params_n))

    extra = [k for k in pred_params_n if k not in exp_params_n]
    extra_penalty = min(0.25, 0.05 * len(extra))
    return max(0.0, min(1.0, 0.4 + 0.6 * param_score - extra_penalty))


def compute_score(solution_str: str, ground_truth: Any) -> float:
    expected_calls = extract_calls(ground_truth)
    pred_calls = extract_calls(solution_str)

    # Response-only ToolRL examples should not hallucinate a tool call.
    if not expected_calls:
        return 1.0 if not pred_calls else 0.0
    if not pred_calls:
        return 0.0

    used: set[int] = set()
    total = 0.0
    for expected in expected_calls:
        best_idx = -1
        best_score = 0.0
        for idx, pred in enumerate(pred_calls):
            if idx in used:
                continue
            score = _call_score(pred, expected)
            if score > best_score:
                best_idx = idx
                best_score = score
        if best_idx >= 0:
            used.add(best_idx)
        total += best_score

    score = total / max(1, len(expected_calls))
    extra_calls = max(0, len(pred_calls) - len(expected_calls))
    score -= min(0.2, 0.03 * extra_calls)
    return float(max(0.0, min(1.0, score)))
