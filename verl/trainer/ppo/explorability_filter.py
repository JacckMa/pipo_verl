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
Explorability-based sample filtering with IGM (Imputed Global Mean) support.

Implementation of the ExplorabilityFilter module from DEPO.
Reference: https://arxiv.org/abs/2509.01321
"""

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import numpy as np


@dataclass
class SampleHistoryEntry:
    """Single history entry for a sample."""
    correctness: bool
    pos_entropy: float
    neg_entropy: float
    reward: float


class ExplorabilityFilter:
    """Explorability-based sample filtering (inspired by DEPO).
    
    Core mechanism:
    1. Track historical training dynamics (reward, correctness, entropy) for each sample
    2. Compute explorability_score = |pos_adv| * H_pos + |neg_adv| * H_neg
    3. Select top-k samples for training based on decaying threshold
    
    The filtering is applied BEFORE generation, so only selected samples participate
    in the current training step.
    """

    def __init__(
        self,
        history_k: int = 5,
        init_sampling_ratio: float = 1.0,
        min_threshold: float = 0.4,
        pruning_decay_weight: float = 0.01,
        neg_pos_ratio: float = 0.5,
        data_id_key: str = "uid",
        max_history_per_sample: int = 50,
    ):
        """
        Args:
            history_k: Window size for computing explorability score.
            init_sampling_ratio: Initial sampling ratio (1.0 = all samples).
            min_threshold: Minimum threshold for sampling ratio (floor).
            pruning_decay_weight: Decay weight per epoch.
            neg_pos_ratio: Ratio threshold for filtering negative sample entropy.
            data_id_key: Key in non_tensor_batch to use as data identifier.
            max_history_per_sample: Maximum history entries per sample.
        """
        self.history_k = history_k
        self.init_sampling_ratio = init_sampling_ratio
        self.min_threshold = min_threshold
        self.pruning_decay_weight = pruning_decay_weight
        self.neg_pos_ratio = neg_pos_ratio
        self.data_id_key = data_id_key
        self.max_history_per_sample = max_history_per_sample

        # History storage: {data_id: deque of SampleHistoryEntry}
        self.history: Dict[str, deque] = {}
        
        # Current epoch for threshold computation
        self.current_epoch: int = 0

    def _get_threshold(self, epoch: int) -> float:
        """Compute current threshold with decay.
        
        threshold = init_ratio - decay * epoch
        Clamped to [min_threshold, 1.0]
        """
        threshold = self.init_sampling_ratio - self.pruning_decay_weight * epoch
        return max(self.min_threshold, min(1.0, threshold))

    def _get_or_create_history(self, data_id: str) -> deque:
        """Get or create history deque for a data_id."""
        if data_id not in self.history:
            self.history[data_id] = deque(maxlen=self.max_history_per_sample)
        return self.history[data_id]

    def update_sample(
        self,
        data_id: str,
        correctness: bool,
        pos_entropy: float = 0.0,
        neg_entropy: float = 0.0,
        reward: float = 0.0,
    ) -> None:
        """Update sample history with new training result.
        
        Args:
            data_id: Unique identifier for the sample.
            correctness: Whether the current response was correct (True/False).
            pos_entropy: Entropy of positive (correct) responses (if any).
            neg_entropy: Entropy of negative (incorrect) responses (if any).
            reward: Reward value for this sample.
        """
        hist = self._get_or_create_history(data_id)
        entry = SampleHistoryEntry(
            correctness=correctness,
            pos_entropy=pos_entropy,
            neg_entropy=neg_entropy,
            reward=reward,
        )
        hist.append(entry)

    def get_historical_mean_reward(self, data_id: str, window: int = 3) -> Optional[float]:
        """Get historical mean reward for a sample.
        
        Args:
            data_id: Unique identifier for the sample.
            window: Number of recent entries to average.
            
        Returns:
            Mean reward over the window, or None if no history.
        """
        if data_id not in self.history or len(self.history[data_id]) == 0:
            return None
        
        hist = self.history[data_id]
        recent = list(hist)[-window:]
        return np.mean([entry.reward for entry in recent])

    def compute_explorability_score(self, data_id: str) -> Tuple[float, int]:
        """Compute explorability score for a sample.
        
        The score is computed over the history window as:
        exploratory_score = mean_over_window(epoch_exp_score)
        
        where:
        - epoch_exp_score = |pos_adv| * H_pos + |neg_adv| * H_neg
        - pos_adv = (1 - acc_mean) / (acc_std + eps)
        - neg_adv = (0 - acc_mean) / (acc_std + eps)
        
        Args:
            data_id: Unique identifier for the sample.
            
        Returns:
            Tuple of (avg_explorability_score, num_updates).
            Returns (0.0, 0) if no history.
        """
        if data_id not in self.history:
            return 0.0, 0
        
        hist = self.history[data_id]
        if len(hist) == 0:
            return 0.0, 0
        
        history_window = min(self.history_k, len(hist))
        window_entries = list(hist)[-history_window:]
        
        sample_exploratory_scores = []
        eps = 1e-8
        
        for entry in window_entries:
            pos_ent = entry.pos_entropy
            neg_ent = entry.neg_entropy
            
            acc_mean = 1.0 if entry.correctness else 0.0
            acc_std = 0.0
            
            pos_adv = (1.0 - acc_mean) / (acc_std + eps)
            neg_adv = (0.0 - acc_mean) / (acc_std + eps)
            
            exploratory_score = abs(pos_adv * pos_ent) + abs(neg_adv * neg_ent)
            sample_exploratory_scores.append(exploratory_score)
        
        if len(sample_exploratory_scores) == 0:
            return 0.0, len(hist)
        
        avg_score = np.mean(sample_exploratory_scores)
        return avg_score, len(hist)

    def get_selected_indices(
        self,
        batch_size: int,
        data_ids: List[str],
        epoch: int,
        total_gpus: int = 1,
    ) -> Tuple[List[int], List[int], Dict[str, Any]]:
        """Select samples for training based on explorability scores.
        
        Args:
            batch_size: Total number of samples in the batch.
            data_ids: List of data_ids for each sample in the batch.
            epoch: Current epoch number (for threshold decay).
            total_gpus: Number of GPUs (for padding alignment).
            
        Returns:
            Tuple of:
            - selected_indices: Indices of selected samples
            - dropped_indices: Indices of dropped samples  
            - stats: Dictionary of statistics for logging
        """
        self.current_epoch = epoch
        threshold = self._get_threshold(epoch)
        
        # Compute explorability scores for all samples
        exploratory_scores = []
        no_history_ids = []  # Samples without history should always be selected
        
        for i, data_id in enumerate(data_ids):
            score, num_updates = self.compute_explorability_score(data_id)
            
            if num_updates == 0:
                # No history = should always be selected
                no_history_ids.append(i)
            
            exploratory_scores.append((i, score, num_updates))
        
        # Sort by explorability score (descending), then by number of updates (ascending)
        # This ensures high explorability samples are prioritized
        exploratory_scores.sort(key=lambda x: (-x[1], x[2]))
        
        # Calculate how many samples to select
        num_to_select = max(
            len(no_history_ids),  # At least all no-history samples
            int(batch_size * threshold)
        )
        num_to_select = min(num_to_select, batch_size)
        
        # Get selected indices (top-k by explorability)
        selected_indices = [idx for idx, _, _ in exploratory_scores[:num_to_select]]
        
        # Ensure no-history samples are included
        for idx in no_history_ids:
            if idx not in selected_indices:
                if len(selected_indices) < num_to_select:
                    selected_indices.append(idx)
        
        remainder = len(selected_indices) % total_gpus
        if remainder != 0:
            num_to_pad = total_gpus - remainder
            # Get candidates not in selected_indices
            selected_set = set(selected_indices)
            remaining_candidates = [item for item in exploratory_scores if item[0] not in selected_set]
            
            if remaining_candidates:
                # Sort by exploration_time (ascending), then by explorability_score (descending)
                # This ensures samples with the fewest exploration times are chosen first
                remaining_candidates.sort(key=lambda x: (x[2], -x[1]))
                
                # Add the first num_to_pad candidates
                indices_to_add = [item[0] for item in remaining_candidates[:num_to_pad]]
                selected_indices.extend(indices_to_add)
        
        # Get dropped indices
        selected_set = set(selected_indices)
        dropped_indices = [i for i in range(batch_size) if i not in selected_set]
        
        # Statistics
        stats = {
            'ef/threshold': threshold,
            'ef/batch_size': batch_size,
            'ef/num_selected': len(selected_indices),
            'ef/num_dropped': len(dropped_indices),
            'ef/num_no_history': len(no_history_ids),
            'ef/selection_ratio': len(selected_indices) / batch_size if batch_size > 0 else 0,
        }
        
        # Add top scores for logging (as individual scalar metrics)
        if len(exploratory_scores) > 0:
            top_scores = [s for _, s, _ in exploratory_scores[:min(3, len(exploratory_scores))]]
            for idx, score in enumerate(top_scores):
                stats[f'ef/top_score_{idx}'] = score
        
        return selected_indices, dropped_indices, stats

    def update_batch(
        self,
        data_ids: List[str],
        correctnesss: List[bool],
        rewards: List[float],
        pos_entropys: Optional[List[float]] = None,
        neg_entropys: Optional[List[float]] = None,
    ) -> None:
        """Update history for a batch of samples.
        
        Args:
            data_ids: List of data_ids.
            correctnesss: List of correctness values.
            rewards: List of reward values.
            pos_entropys: Optional list of positive sample entropies.
            neg_entropys: Optional list of negative sample entropies.
        """
        for i, data_id in enumerate(data_ids):
            correctness = correctnesss[i] if i < len(correctnesss) else False
            reward = rewards[i] if i < len(rewards) else 0.0
            pos_ent = pos_entropys[i] if pos_entropys and i < len(pos_entropys) else 0.0
            neg_ent = neg_entropys[i] if neg_entropys and i < len(neg_entropys) else 0.0
            
            self.update_sample(
                data_id=data_id,
                correctness=correctness,
                pos_entropy=pos_ent,
                neg_entropy=neg_ent,
                reward=reward,
            )

    def get_avg_explorability(self, data_ids: List[str]) -> float:
        """Get average explorability score across a list of samples.
        
        Args:
            data_ids: List of data_ids.
            
        Returns:
            Average explorability score.
        """
        if len(data_ids) == 0:
            return 0.0
        
        scores = [self.compute_explorability_score(did)[0] for did in data_ids]
        return np.mean(scores) if scores else 0.0

    def reset(self) -> None:
        """Reset all history (useful for new training runs)."""
        self.history.clear()
        self.current_epoch = 0

    def __len__(self) -> int:
        """Return number of samples with history."""
        return len(self.history)

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the filter state."""
        if len(self.history) == 0:
            return {
                'ef/num_samples_tracked': 0,
                'ef/avg_history_length': 0.0,
                'ef/avg_explorability': 0.0,
            }
        
        history_lengths = [len(h) for h in self.history.values()]
        all_scores = [self.compute_explorability_score(did)[0] for did in self.history.keys()]
        
        return {
            'ef/num_samples_tracked': len(self.history),
            'ef/avg_history_length': np.mean(history_lengths) if history_lengths else 0.0,
            'ef/avg_explorability': np.mean(all_scores) if all_scores else 0.0,
        }


def create_explorability_filter(config) -> Optional[ExplorabilityFilter]:
    """Factory function to create ExplorabilityFilter from config.
    
    Args:
        config: Algorithm config containing explorability_filter settings.
        
    Returns:
        ExplorabilityFilter instance or None if not enabled.
    """
    if not hasattr(config, 'explorability_filter') or config.explorability_filter is None:
        return None
    
    ef_config = config.explorability_filter
    if not ef_config.enable:
        return None
    
    return ExplorabilityFilter(
        history_k=ef_config.history_k,
        init_sampling_ratio=ef_config.init_sampling_ratio,
        min_threshold=ef_config.min_threshold,
        pruning_decay_weight=ef_config.pruning_decay_weight,
        neg_pos_ratio=ef_config.neg_pos_ratio,
        data_id_key=ef_config.data_id_key,
        max_history_per_sample=ef_config.max_history_per_sample,
    )
