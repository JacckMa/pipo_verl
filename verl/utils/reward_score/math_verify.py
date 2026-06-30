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

try:
    from latex2sympy2_extended import NormalizationConfig
    from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify
    from math_verify.errors import TimeoutException
except ImportError:
    print("To use Math-Verify, please install it first by running `pip install math-verify`.")


def _unwrap_ground_truth(ground_truth):
    if hasattr(ground_truth, "tolist") and not isinstance(ground_truth, str):
        ground_truth = ground_truth.tolist()
    if isinstance(ground_truth, (list, tuple)):
        ground_truth = ground_truth[0] if ground_truth else ""
    return str(ground_truth)


def _ground_truth_variants(ground_truth):
    gt = _unwrap_ground_truth(ground_truth).strip()
    variants = [gt]

    converted = gt
    for before, after in {
        "np.arcsin": r"\arcsin",
        "np.arccos": r"\arccos",
        "np.arctan": r"\arctan",
        "arcsin": r"\arcsin",
        "arccos": r"\arccos",
        "arctan": r"\arctan",
    }.items():
        converted = converted.replace(before, after)
    if converted != gt:
        variants.append(converted)

    sci = re.fullmatch(r"([+-]?\d+(?:\.\d+)?)[eE]([+-]?\d+)", gt)
    if sci:
        mant, exp = sci.groups()
        variants.append(rf"{mant}\times 10^{{{int(exp)}}}")
        variants.append(rf"{mant} \times 10^{{{int(exp)}}}")

    unique = []
    for item in variants:
        if item not in unique:
            unique.append(item)
    return unique


def _parse_gold(ground_truth):
    parsed = []
    for variant in _ground_truth_variants(ground_truth):
        for candidate in (rf"\boxed{{{variant}}}", rf"${variant}$", variant):
            try:
                value = parse(
                    candidate,
                    extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()],
                    extraction_mode="first_match",
                )
            except Exception:
                continue
            if value:
                parsed.append(value)
    return parsed


def _parse_prediction(model_output):
    try:
        return parse(
            model_output or "",
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        equations=True,
                        boxed="all",
                        units=True,
                    ),
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                ),
                ExprExtractionConfig(),
            ],
            extraction_mode="first_match",
        )
    except Exception:
        return []


def compute_score(model_output: str, ground_truth, timeout_score: float = 0) -> float:
    ret_score = 0.0
    try:
        pred = _parse_prediction(model_output)
        if not pred:
            return ret_score
        for gold in _parse_gold(ground_truth):
            if verify(gold, pred):
                return 1.0
    except TimeoutException:
        ret_score = timeout_score
    except Exception:
        pass
    return ret_score
