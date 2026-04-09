"""
Replay Buffer for SAC with continuous actions.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


class ReplayBuffer:
    """
    Stores transitions (state, action, reward, next_state, done, cost).

    state:   (n_nodes, feat_dim)  – node-level features
    action:  (n_nodes,)           – continuous throttle per node
    cost:    float                – constraint violation signal for Lagrangian
    """

    def __init__(self, capacity: int, n_nodes: int, feat_dim: int = 4):
        self.capacity = int(capacity)
        self.n = n_nodes
        self.feat_dim = feat_dim

        self.ptr = 0
        self.size = 0

        self.states = np.zeros((capacity, n_nodes, feat_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, n_nodes), dtype=np.float32)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_states = np.zeros((capacity, n_nodes, feat_dim), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        self.costs = np.zeros((capacity,), dtype=np.float32)

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        cost: float = 0.0,
    ) -> None:
        i = self.ptr
        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_states[i] = next_state
        self.dones[i] = 1.0 if done else 0.0
        self.costs[i] = cost

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator
               ) -> Tuple[np.ndarray, ...]:
        idx = rng.integers(0, self.size, size=batch_size)
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx],
            self.costs[idx],
        )
