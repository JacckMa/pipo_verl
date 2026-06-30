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

"""Utilities for PIPO feedback updates."""

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


@dataclass
class LaybackBatchCache:
    """Cached data from the previous policy update."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: Optional[torch.Tensor]
    response_mask: torch.Tensor
    old_log_probs: torch.Tensor
    ref_log_probs: Optional[torch.Tensor]
    advantages: torch.Tensor
    step_count: int
    loss_mode: str = "vanilla"


class LaybackHistoryManager:
    """Tracks recent mean rewards for computing the PIPO coefficient."""

    def __init__(self, window_size: int = 8):
        self.window_size = window_size
        self.mean_rewards = deque(maxlen=window_size)
        self._current_mu: Optional[float] = None

    def store_step_data(self, mean_reward: float) -> None:
        self._current_mu = mean_reward
        self.mean_rewards.append(mean_reward)

    def get_current_mu(self) -> Optional[float]:
        return self._current_mu

    def get_previous_mean_rewards(self) -> list:
        """Return the reward history before the current step.

        `store_step_data` is called as soon as the current batch reward is known,
        so the rolling window contains the current reward at the tail.
        """
        historical = list(self.mean_rewards)
        if self._current_mu is not None and historical:
            historical = historical[:-1]
        return historical

    def __len__(self) -> int:
        return len(self.mean_rewards)


def _convert_to_3d_grouped_format(
    tensor_2d: torch.Tensor,
    uid: np.ndarray,
) -> tuple[torch.Tensor, np.ndarray, int, int]:
    unique_uids, uid_indices = np.unique(uid, return_inverse=True)
    num_prompts = len(unique_uids)

    _, counts = np.unique(uid_indices, return_counts=True)
    n = counts[0]
    if not np.all(counts == n):
        raise ValueError(
            f"All questions must have the same number of responses for layback grouping. "
            f"Got counts: {counts}."
        )

    batch_size, seq_len = tensor_2d.shape
    sort_indices = np.argsort(uid_indices)
    tensor_sorted = tensor_2d[sort_indices]
    tensor_3d = tensor_sorted.view(num_prompts, n, seq_len)
    return tensor_3d, sort_indices, num_prompts, n


def _convert_from_3d_to_2d(
    tensor_3d: torch.Tensor,
    sort_indices: np.ndarray,
) -> torch.Tensor:
    num_prompts, n, seq_len = tensor_3d.shape
    tensor_2d_sorted = tensor_3d.view(num_prompts * n, seq_len)
    unsort_indices = np.argsort(sort_indices)
    return tensor_2d_sorted[unsort_indices]


def compute_group_scaled_advantages(
    response_mask: torch.Tensor,
    advantages: torch.Tensor,
    uid: np.ndarray,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, dict]:
    """Compute G * A / sum(abs(A)) within each prompt group for GRPO-style layback."""
    with torch.no_grad():
        advantages_3d, sort_indices, _, n = _convert_to_3d_grouped_format(advantages, uid)
        response_mask_3d, _, _, _ = _convert_to_3d_grouped_format(response_mask, uid)

        seq_advantages = (advantages_3d * response_mask_3d).sum(dim=-1) / response_mask_3d.sum(dim=-1).clamp(min=1)
        group_abs_sum = torch.abs(seq_advantages).sum(dim=1).clamp(min=epsilon)
        scaled_seq_advantages = seq_advantages / group_abs_sum.unsqueeze(1) * n

        scaled_advantages_3d = scaled_seq_advantages.unsqueeze(-1) * response_mask_3d
        scaled_advantages = _convert_from_3d_to_2d(scaled_advantages_3d, sort_indices)

        metrics = {
            "layback_cache/group_scaled_adv_abs_mean": scaled_seq_advantages.abs().mean().item(),
            "layback_cache/group_scaled_adv_mean": scaled_seq_advantages.mean().item(),
            "layback_cache/group_scaled_adv_std": scaled_seq_advantages.std().item(),
        }
    return scaled_advantages, metrics


def compute_layback_coefficients(
    current_mu: float,
    historical_mean_rewards: list,
    loss_scale_neg: float = 0.0,
    epsilon: float = 1e-8,
) -> tuple[float, dict]:
    """Compute the PIPO coefficient from standardized reward progress."""
    historical = historical_mean_rewards
    if len(historical) == 0:
        rho = current_mu
        sigma = 1.0
        xi_t = 0.0
    else:
        rho = float(np.mean(historical))
        sigma = float(np.std(historical))
        if sigma < epsilon:
            sigma = 1.0
        xi_t = (current_mu - rho) / sigma

    if xi_t >= 0:
        coefficient = float(xi_t)
    else:
        coefficient = float(xi_t * loss_scale_neg)

    metrics = {
        "layback/mu_t": current_mu,
        "layback/rho_t_minus_1": rho,
        "layback/sigma_t_minus_1": sigma,
        "layback/xi_t": xi_t,
        "layback/coefficient": coefficient,
    }

    return coefficient, metrics


def apply_layback_update(
    actor_rollout_wg,
    layback_cache: LaybackBatchCache,
    coefficient: float,
    ref_log_probs: Optional[torch.Tensor] = None,
    config=None,
    clip_ratio_high: Optional[float] = None,
    loss_mode: Optional[str] = None,
) -> tuple[dict, dict]:
    """Replay the previous batch with advantages modulated by the PIPO coefficient."""
    response_mask = layback_cache.response_mask
    old_log_probs = layback_cache.old_log_probs
    scaled_advantages = layback_cache.advantages * coefficient * response_mask
    loss_mode = loss_mode or getattr(layback_cache, "loss_mode", "vanilla")

    response_length = layback_cache.response_mask.size(1)
    layback_batch = {
        "input_ids": layback_cache.input_ids,
        "attention_mask": layback_cache.attention_mask,
        "response_mask": layback_cache.response_mask,
        "old_log_probs": layback_cache.old_log_probs,
        "advantages": scaled_advantages,
        "responses": layback_cache.input_ids[:, -response_length:],
    }
    if layback_cache.position_ids is not None:
        layback_batch["position_ids"] = layback_cache.position_ids
    if ref_log_probs is not None:
        layback_batch["ref_log_prob"] = ref_log_probs

    from verl import DataProto

    num_tokens = torch.sum(layback_cache.attention_mask, dim=-1).tolist()
    layback_data = DataProto.from_dict(
        tensors=layback_batch,
        meta_info={
            "micro_batch_size": config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
            "temperature": config.actor_rollout_ref.rollout.temperature,
            "use_dynamic_bsz": config.actor_rollout_ref.rollout.get("log_prob_use_dynamic_bsz", False),
            "multi_turn": config.actor_rollout_ref.rollout.multi_turn.enable,
            "loss_mode": loss_mode,
            "layback_enabled": True,
            "global_token_num": num_tokens,
            "layback_clip_ratio_high": clip_ratio_high,
        },
    )

    actor_output = actor_rollout_wg.update_actor(layback_data)
    actor_metrics_raw = actor_output.meta_info["metrics"]
    from verl.utils.metric import reduce_metrics
    actor_metrics = reduce_metrics(actor_metrics_raw)

    layback_metrics = {
        "layback/policy_loss": actor_metrics.get("actor/pg_loss", 0.0),
        "layback/kl_loss": actor_metrics.get("actor/kl_loss", 0.0),
        "layback/ppo_kl": actor_metrics.get("actor/ppo_kl", 0.0),
        "layback/pg_clipfrac": actor_metrics.get("actor/pg_clipfrac", 0.0),
        "layback/pg_clipfrac_lower": actor_metrics.get("actor/pg_clipfrac_lower", 0.0),
        "layback/coefficient": coefficient,
        "layback/grad_norm": actor_metrics.get("actor/grad_norm", 0.0),
        "layback/clip_ratio_high": clip_ratio_high if clip_ratio_high is not None else 0.0,
    }

    return layback_metrics, None
