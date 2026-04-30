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

from dataclasses import dataclass, field
from typing import Any, Optional

from verl.base_config import BaseConfig

__all__ = ["AlgoConfig", "FilterGroupsConfig", "KLControlConfig", "LaybackConfig", "BAPOConfig", "ExplorabilityFilterConfig"]


@dataclass
class KLControlConfig(BaseConfig):
    """Configuration for KL control.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        type (str): Type of KL control. Can be "fixed" or "adaptive".
        kl_coef (float): Initial coefficient for KL penalty.
        horizon (int): Horizon value for adaptive controller.
        target_kl (float): Target KL divergence for adaptive controller.
    """

    type: str = "fixed"
    kl_coef: float = 0.001
    horizon: int = 10000
    target_kl: float = 0.1


@dataclass
class FilterGroupsConfig(BaseConfig):
    """Configuration for filter groups (used in DAPO and Entropy).

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        enable (bool): Whether to enable filter groups.
        metric (Optional[str]): Metric to use for filtering: "acc", "score", "seq_reward", "seq_final_reward", etc.
        max_num_gen_batches (int): Non-positive values mean no upper limit.
    """

    enable: bool = False
    metric: Optional[str] = None
    max_num_gen_batches: int = 0


@dataclass
class LaybackConfig(BaseConfig):
    """Configuration for PIPO (Policy Improvement Policy Optimization) mechanism.

    PIPO adaptively reuses current step data with progress-aware reweighting
    to improve training stability and performance without additional sampling cost.

    The layback coefficient C_{t,i} = A_prog * (mu_t - rho_{t-1}), where:
    - mu_t: current step mean reward (no EMA smoothing)
    - rho_{t-1}: mean of historical rewards over the window
    - A_prog: progress advantage (group-normalized advantage, scaled by group size G)
    - variance normalization: (mu_t - rho_{t-1}) / sigma_{t-1}
    When progress_scale < 0, the layback loss is scaled by `loss_scale_neg`.

    Args:
        enable (bool): Whether to enable PIPO mechanism.
        history_window_size (int): Window size K for computing historical statistics.
        layback_every_n_steps (int): Perform PIPO update every N steps. Default 1.
        min_steps_before_layback (int): Minimum historical steps required before starting PIPO.
        loss_scale_neg (float): Loss scale applied when progress < 0. Default 0.0.
        enable_variance_normalization (bool): Whether to normalize progress by historical variance. Default True.
        layback_threshold (float): Skip layback when |xi_t| <= threshold. Default 0.0 (disabled).
    """

    enable: bool = False
    history_window_size: int = 8
    layback_every_n_steps: int = 1
    min_steps_before_layback: int = 8
    loss_scale_neg: float = 0.0
    enable_variance_normalization: bool = True
    layback_threshold: float = 0.0
    ppo_gate_min: float = -2.0
    ppo_gate_max: float = 2.0


@dataclass
class BAPOConfig(BaseConfig):
    """Configuration for BAPO (Balanced Policy Optimization).

    Dynamically searches for optimal clipping bounds (c_low, c_high) per batch
    such that the proportion of positive-advantage contribution reaches rho_0.

    Paper: https://arxiv.org/abs/2510.18927

    Args:
        enable (bool): Whether to enable BAPO adaptive clipping.
        rho_0 (float): Target ratio of positive-advantage loss contribution. Default 0.4 (paper value).
        a_minus (float): Initial lower clipping bound c_low. Default 0.6.
        b_minus (float): Hard upper bound for c_low search. Default 0.9 (paper value).
        a_plus (float): Initial upper clipping bound c_high. Default 1.2.
        b_plus (float): Hard upper bound for c_high search. Default 3.0 (paper value).
        delta_1 (float): Step size for expanding c_high. Default 0.05.
        delta_2 (float): Step size for expanding c_low. Default 0.02 (asymmetric, paper value).
    """

    enable: bool = False
    rho_0: float = 0.4
    a_minus: float = 0.6
    b_minus: float = 0.9
    a_plus: float = 1.2
    b_plus: float = 3.0
    delta_1: float = 0.05
    delta_2: float = 0.02


@dataclass
class ExplorabilityFilterConfig(BaseConfig):
    """Configuration for Explorability-based sample filtering (DEPO).

    Dynamically selects training samples based on their historical explorability score,
    which measures how much the model benefits from continued training on each sample.

    Core mechanism:
    1. Track historical training dynamics (reward, entropy) for each sample
    2. Compute explorability_score = |pos_adv| * H_pos + |neg_adv| * H_neg
    3. Select top-k samples for training based on threshold
    4. Support replay: ensure all samples are trained at least once

    Args:
        enable (bool): Whether to enable ExplorabilityFilter.
        history_k (int): Window size for computing explorability score. Default 5.
        init_sampling_ratio (float): Initial sampling ratio (1.0 = all samples). Default 1.0.
        min_threshold (float): Minimum threshold for sampling ratio (floor). Default 0.4.
        pruning_decay_weight (float): Decay weight per epoch. threshold = init_ratio - decay * epoch.
        neg_pos_ratio (float): Ratio threshold for filtering negative sample entropy. Default 0.5.
        data_id_key (str): Key in non_tensor_batch to use as data identifier. Default "uid".
        max_history_per_sample (int): Maximum history entries per sample. Default 50.
    """

    enable: bool = False
    history_k: int = 5
    init_sampling_ratio: float = 1.0
    min_threshold: float = 0.4
    pruning_decay_weight: float = 0.01
    neg_pos_ratio: float = 0.5
    data_id_key: str = "uid"
    max_history_per_sample: int = 50


@dataclass
class AlgoConfig(BaseConfig):
    """Configuration for the algorithm.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        gamma (float): Discount factor for future rewards.
        lam (float): Trade-off between bias and variance in the GAE estimator.
        adv_estimator (str): Advantage estimator type: "gae", "grpo", "reinforce_plus_plus", etc.
        norm_adv_by_std_in_grpo (bool): Whether to normalize advantages by std (specific to GRPO).
        use_kl_in_reward (bool): Whether to enable in-reward KL penalty.
        kl_penalty (str): How to estimate KL divergence: "kl", "abs", "mse", "low_var_kl", or "full".
        kl_ctrl (KLControlConfig): KL control configuration.
        use_pf_ppo (bool): Whether to enable preference feedback PPO.
        pf_ppo (dict[str, Any]): Preference feedback PPO settings.
        filter_groups (Optional[FilterGroupsConfig]): Filter groups configuration, used in DAPO and Entropy
        layback (Optional[LaybackConfig]): Layback configuration for adaptive historical data reuse.
        explorability_filter (Optional[ExplorabilityFilterConfig]): ExplorabilityFilter configuration for DEPO-style online filtering.
    """

    gamma: float = 1.0
    lam: float = 1.0
    adv_estimator: str = "gae"
    norm_adv_by_std_in_grpo: bool = True
    use_kl_in_reward: bool = False
    kl_penalty: str = "kl"
    kl_ctrl: KLControlConfig = field(default_factory=KLControlConfig)
    use_pf_ppo: bool = False
    pf_ppo: dict[str, Any] = field(default_factory=dict)
    filter_groups: Optional[FilterGroupsConfig] = None
    layback: Optional[LaybackConfig] = None
    bapo: Optional[BAPOConfig] = None
    explorability_filter: Optional[ExplorabilityFilterConfig] = None
