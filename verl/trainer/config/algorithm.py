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

__all__ = ["AlgoConfig", "FilterGroupsConfig", "KLControlConfig", "LaybackConfig", "AdaptiveTemperatureConfig"]


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
class AdaptiveTemperatureConfig(BaseConfig):
    """Configuration for adaptive temperature based on training progress.
    
    Dynamically adjusts rollout sampling temperature using the formula:
    T = base_temperature * exp(-sensitivity * (progress_scale - progress_scale_center))
    
    Args:
        enable (bool): Whether to enable adaptive temperature.
        base_temperature (float): Base temperature when progress_scale equals progress_scale_center.
        min_temperature (float): Minimum temperature clamp to prevent greedy collapse.
        max_temperature (float): Maximum temperature clamp to prevent random sampling.
        progress_scale_center (float): The "normal progress" threshold. Below this, temperature increases.
        sensitivity (float): How aggressively temperature responds to progress changes.
    """
    
    enable: bool = False
    base_temperature: float = 0.7
    min_temperature: float = 0.1
    max_temperature: float = 2.5
    progress_scale_center: float = 0.05
    sensitivity: float = 10.0


@dataclass
class LaybackConfig(BaseConfig):
    """Configuration for PIPO (Policy Improvement Policy Optimization) mechanism.
    
    PIPO adaptively reuses current step data with progress-aware reweighting
    to improve training stability and performance without additional sampling cost.
    
    Args:
        enable (bool): Whether to enable PIPO mechanism.
        history_window_size (int): Window size K for computing historical statistics.
        layback_every_n_steps (int): Perform PIPO update every N steps. Default 1.
        min_steps_before_layback (int): Minimum historical steps required before starting PIPO.
        ema_alpha (float): EMA smoothing coefficient for mu_t. Range (0, 1]. Default 0.3.
        enable_progress_scale_bounds (bool): Whether to enforce bounds on progress scale.
        progress_scale_min (float): Lower bound for progress scale (ReLU truncation).
        progress_scale_max (float): Upper bound for progress scale.
        loss_scale (float): Scale factor for PIPO loss. Default 1.0.
        loss_scale_neg (float): Scale factor when progress < 0. Default 0.
        enable_group_size_scaling (bool): Whether to multiply A_prog by group size G.
        enable_variance_normalization (bool): Whether to normalize progress scale by historical variance.
    """
    
    enable: bool = False
    history_window_size: int = 8
    layback_every_n_steps: int = 1
    min_steps_before_layback: int = 8
    ema_alpha: float = 1.0
    enable_progress_scale_bounds: bool = True
    progress_scale_min: float = 0
    progress_scale_max: float = 0.5
    loss_scale: float = 0.5
    loss_scale_neg: float = 0.0
    
    enable_group_size_scaling: bool = True
    enable_variance_normalization: bool = True


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
