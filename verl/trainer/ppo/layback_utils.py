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

"""Utilities for SDPO with PIPO feedback.

The cache stores token inputs rather than logits, so the current student and
self-teacher are forwarded again on the previous response.
"""

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from verl import DataProto


SDPO_LAYBACK_BATCH_KEYS = (
    "input_ids",
    "attention_mask",
    "position_ids",
    "responses",
    "response_mask",
    "teacher_input_ids",
    "teacher_attention_mask",
    "teacher_position_ids",
    "self_distillation_mask",
)


@dataclass
class SDPODistillLaybackCache:
    """One-step cache used by SDPO+PIPO."""

    tensors: Optional[dict[str, torch.Tensor]] = None
    step_count: Optional[int] = None

    def has_cache(self) -> bool:
        return self.tensors is not None

    def update(self, batch: DataProto, step_count: int) -> None:
        missing = [key for key in SDPO_LAYBACK_BATCH_KEYS if key not in batch.batch.keys()]
        if missing:
            raise KeyError(f"Cannot cache SDPO layback batch; missing keys: {missing}")

        self.tensors = {
            key: batch.batch[key].detach().cpu()
            for key in SDPO_LAYBACK_BATCH_KEYS
        }
        self.step_count = step_count

    def build_batch(self) -> DataProto:
        if self.tensors is None:
            raise RuntimeError("SDPO PIPO cache is empty.")
        return DataProto.from_dict(
            tensors={key: value.clone() for key, value in self.tensors.items()},
            meta_info={},
        )


class LaybackRewardHistory:
    """Rolling window for recent batch-level rewards."""

    def __init__(self, window_size: int = 8):
        self.mean_rewards = deque(maxlen=window_size)

    def update(self, mean_reward: float) -> None:
        self.mean_rewards.append(float(mean_reward))

    def previous_rewards(self) -> list[float]:
        return list(self.mean_rewards)

    def __len__(self) -> int:
        return len(self.mean_rewards)


def compute_linear_layback_coefficient(
    current_mean: float,
    historical_mean_rewards: list[float],
    loss_scale_neg: float = 0.0,
    epsilon: float = 1e-6,
) -> tuple[float, dict[str, float]]:
    """Compute the PIPO coefficient from standardized reward progress."""

    if len(historical_mean_rewards) == 0:
        history_mean = float(current_mean)
        history_std = 1.0
        xi_t = 0.0
    else:
        history_mean = float(np.mean(historical_mean_rewards))
        history_std = float(np.std(historical_mean_rewards))
        if history_std < epsilon:
            history_std = 1.0
        xi_t = float((current_mean - history_mean) / history_std)

    coefficient = float(xi_t if xi_t >= 0 else xi_t * float(loss_scale_neg))

    metrics = {
        "layback/mu_t": float(current_mean),
        "layback/rho_t_minus_1": history_mean,
        "layback/sigma_t_minus_1": history_std,
        "layback/xi_t": xi_t,
        "layback/coefficient": coefficient,
    }
    return coefficient, metrics
