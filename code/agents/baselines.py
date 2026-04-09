"""
Baseline agents for comparison with GAT-SAC.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class NoDefenseAgent:
    """Always produces zero throttle (no intervention)."""

    def act(self, state: np.ndarray, adj: np.ndarray) -> np.ndarray:
        n = state.shape[0]
        return np.zeros(n, dtype=np.float32)


class HeuristicThrottleAgent:
    """
    If any node's utilization exceeds threshold, throttle it by a fixed amount.
    This is the packaged threshold-based graceful-drain heuristic used in the finalized submission.
    """

    def __init__(self, util_threshold: float = 0.90, throttle_amount: float = 0.3):
        self.util_threshold = float(util_threshold)
        self.throttle_amount = float(throttle_amount)

    def act(self, state: np.ndarray, adj: np.ndarray) -> np.ndarray:
        n = state.shape[0]
        action = np.zeros(n, dtype=np.float32)
        # state[:, 1] = utilization_norm (already /3.0 in env)
        util = state[:, 1] * 3.0  # denormalize
        operational = state[:, 0]
        for i in range(n):
            if operational[i] > 0.5 and util[i] > self.util_threshold:
                action[i] = self.throttle_amount
        return action


class HeuristicHardPruneAgent:
    """
    Archived hard-prune ablation retained for legacy experiments.
    It is not part of the finalized submission comparator set.
    """

    def __init__(self, util_threshold: float = 0.95):
        self.util_threshold = float(util_threshold)

    def act(self, state: np.ndarray, adj: np.ndarray) -> np.ndarray:
        n = state.shape[0]
        action = np.zeros(n, dtype=np.float32)
        util = state[:, 1] * 3.0
        operational = state[:, 0]

        util_masked = util.copy()
        util_masked[operational < 0.5] = -1.0
        worst = int(np.argmax(util_masked))
        if util_masked[worst] > self.util_threshold:
            action[worst] = 1.0  # full shutdown
        return action


class BackupAgent:
    """
    A dummy agent for backup scenarios. 
    In our environment, 'Backup' is handled by setting redundancy_factor > 1.0
    in the environment constructor. The agent itself does nothing (No Defense).
    """

    def __init__(self, n: int, extra_cap_ratio: float = 0.25):
        self.n = n
        self.extra_cap_ratio = extra_cap_ratio

    def act(self, state: np.ndarray, adj: np.ndarray) -> np.ndarray:
        return np.zeros(self.n, dtype=np.float32)
