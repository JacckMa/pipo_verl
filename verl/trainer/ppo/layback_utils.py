from collections import deque
from typing import Optional
import torch
import numpy as np


class LaybackHistoryManager:
    def __init__(self, window_size: int = 10, ema_alpha: float = 0.3):
        self.window_size = window_size
        self.ema_alpha = ema_alpha
        self.mean_rewards = deque(maxlen=window_size)
        self.mu_t_smooth: float | None = None
        self.v_ema: float | None = None
    
    def smooth_mu_t(self, mu_t: float) -> float:
        if self.mu_t_smooth is None:
            self.mu_t_smooth = mu_t
        else:
            self.mu_t_smooth = self.ema_alpha * mu_t + (1 - self.ema_alpha) * self.mu_t_smooth
        
        if self.ema_alpha != 1.0:
            if self.v_ema is None:
                self.v_ema = mu_t ** 2
            else:
                self.v_ema = self.ema_alpha * (mu_t ** 2) + (1 - self.ema_alpha) * self.v_ema
        
        return self.mu_t_smooth
    
    def get_mu_t_smooth(self) -> float | None:
        return self.mu_t_smooth
    
    def compute_variance(self, epsilon: float = 1e-8) -> float | None:
        if self.window_size == 1:
            return 1.0
        
        if self.ema_alpha != 1.0:
            if self.mu_t_smooth is None or self.v_ema is None:
                return None
            variance = self.v_ema - (self.mu_t_smooth ** 2)
            sigma = float(np.sqrt(max(variance, 0.0) + epsilon))
            return sigma
        
        historical = list(self.mean_rewards)
        if len(historical) < 2:
            return None
        
        sigma = float(np.std(historical))
        if sigma < epsilon:
            sigma = 1.0
        
        return sigma
    
    def store_step_data(self, mean_reward: float) -> None:
        self.mean_rewards.append(mean_reward)
    
    def get_mean_rewards(self) -> list:
        return list(self.mean_rewards)
    
    def __len__(self) -> int:
        return len(self.mean_rewards)
    
    def is_ready(self, min_steps: int = 1) -> bool:
        return len(self.mean_rewards) >= min_steps


def _compute_progress_scale(
    current_mu: float,
    historical_mean_rewards: list[float],
    window_size: int = 10,
    ema_alpha: float = 1.0,
    mu_ema: float | None = None,
    v_ema: float | None = None,
    epsilon: float = 1e-8,
) -> tuple[float, float, float]:
    if len(historical_mean_rewards) == 0:
        rho_t_minus_1 = current_mu
    else:
        rho_t_minus_1 = np.mean(historical_mean_rewards)

    progress_scale = current_mu - rho_t_minus_1
    progress_scale_clipped = np.clip(progress_scale, -5.0, 5.0)
    
    if window_size == 1:
        sigma_t_minus_1 = 1.0
    elif ema_alpha != 1.0 and mu_ema is not None and v_ema is not None:
        variance = v_ema - (mu_ema ** 2)
        sigma_t_minus_1 = float(np.sqrt(max(variance, 0.0) + epsilon))
    elif len(historical_mean_rewards) >= 2:
        sigma_t_minus_1 = float(np.std(historical_mean_rewards))
        if sigma_t_minus_1 < epsilon:
            sigma_t_minus_1 = 1.0
    else:
        sigma_t_minus_1 = 1.0
    
    return progress_scale_clipped, rho_t_minus_1, sigma_t_minus_1


def _convert_to_3d_grouped_format(
    tensor_2d: torch.Tensor,
    uid: np.ndarray,
) -> tuple[torch.Tensor, np.ndarray, int, int]:
    unique_uids, uid_indices = np.unique(uid, return_inverse=True)
    num_prompts = len(unique_uids)
    
    unique_indices, counts = np.unique(uid_indices, return_counts=True)
    n = counts[0]
    
    if not np.all(counts == n):
        raise ValueError(
            f"All questions must have the same number of responses. "
            f"Got counts: {counts}. Please ensure all groups have size {n}."
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
    batch_size = num_prompts * n
    
    tensor_2d_sorted = tensor_3d.view(batch_size, seq_len)
    
    unsort_indices = np.argsort(sort_indices)
    tensor_2d = tensor_2d_sorted[unsort_indices]
    
    return tensor_2d


def compute_layback_coefficients(
    response_mask: torch.Tensor,
    current_mu: float,
    historical_mean_rewards: list[float],
    advantages: torch.Tensor,
    uid: np.ndarray,
    epsilon: float = 1e-6,
    config: Optional['LaybackConfig'] = None,
    history_manager: Optional['LaybackHistoryManager'] = None,
) -> tuple[torch.Tensor, np.ndarray, dict]:
    """
    Compute Layback coefficients for the previous batch.
    """
    from verl.trainer.config.algorithm import LaybackConfig
    
    if config is None:
        config = LaybackConfig()
    
    enable_group_size_scaling = getattr(config, 'enable_group_size_scaling', True)
    enable_variance_normalization = getattr(config, 'enable_variance_normalization', True)
    window_size = getattr(config, 'history_window_size', 8)
    ema_alpha = getattr(config, 'ema_alpha', 1.0)
    
    with torch.no_grad():
        mu_ema = None
        v_ema = None
        if history_manager is not None:
            mu_ema = history_manager.get_mu_t_smooth()
            v_ema = history_manager.v_ema
        
        progress_scale, rho_t_minus_1, sigma_t_minus_1 = _compute_progress_scale(
            current_mu=current_mu,
            historical_mean_rewards=historical_mean_rewards,
            window_size=window_size,
            ema_alpha=ema_alpha,
            mu_ema=mu_ema,
            v_ema=v_ema,
            epsilon=epsilon,
        )
        
        if not np.isfinite(progress_scale):
            import warnings
            warnings.warn(f"Invalid progress_scale: {progress_scale}. Skipping feature application.")
            progress_scale = 0.0
        advantages_3d, sort_indices, num_prompts, n = _convert_to_3d_grouped_format(
            tensor_2d=advantages,
            uid=uid,
        )
        response_mask_3d, _, _, _ = _convert_to_3d_grouped_format(
            tensor_2d=response_mask,
            uid=uid,
        )
        
        seq_advantages = (advantages_3d * response_mask_3d).sum(dim=-1) / response_mask_3d.sum(dim=-1).clamp(min=1)
        
        group_abs_sum = torch.abs(seq_advantages).sum(dim=1) + epsilon
        
        A_prog_seq = seq_advantages / group_abs_sum.unsqueeze(1)
        
        if enable_group_size_scaling:
            A_prog_seq = A_prog_seq * n
        
        normalized_progress_scale = progress_scale
        if enable_variance_normalization and sigma_t_minus_1 > epsilon:
            normalized_progress_scale = progress_scale / sigma_t_minus_1
        
        layback_coefficients_seq = A_prog_seq * normalized_progress_scale
        
        layback_coefficients_3d = layback_coefficients_seq.unsqueeze(-1) * response_mask_3d
        layback_coefficients_3d = torch.clamp(layback_coefficients_3d, min=-10.0, max=10.0)

        layback_coefficients = _convert_from_3d_to_2d(
            tensor_3d=layback_coefficients_3d,
            sort_indices=sort_indices,
        )
        
        metrics = {
            'mu_t': current_mu,
            'mu_his': rho_t_minus_1,
            'sigma_his': sigma_t_minus_1,
            'progress_scale': progress_scale,
            'xi_t': normalized_progress_scale,
            'A_prog_mean': A_prog_seq.mean().item(),
            'A_prog_abs_mean': A_prog_seq.abs().mean().item(),
            'A_prog_std': A_prog_seq.std().item(),
        }
    
    return layback_coefficients, sort_indices, metrics
