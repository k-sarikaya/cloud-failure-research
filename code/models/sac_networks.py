"""
SAC Actor – Gaussian Policy for Continuous Throttle Actions
============================================================
Outputs per-node throttle values ∈ [0, 1] via a squashed Gaussian:
  throttle_i = sigmoid(μ_i + σ_i * ε),  ε ~ N(0, 1)

The actor takes [node_embed_i || graph_embed] as input for each node
and outputs (μ, log_σ) which define the Gaussian distribution.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
EPSILON = 1e-6


class SACActorPerNode(nn.Module):
    """
    Per-node Gaussian policy.

    Input:  (B, N, embed_dim)  – concatenation of [node_embed || graph_embed]
    Output: (B, N)             – throttle values in [0, 1]
    """

    def __init__(self, embed_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden_dim, 1)
        self.log_sigma_head = nn.Linear(hidden_dim, 1)

    def forward(self, node_graph_embed: torch.Tensor,
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            node_graph_embed: (B, N, embed_dim)

        Returns:
            mu:        (B, N)
            log_sigma: (B, N)
        """
        h = self.net(node_graph_embed)
        mu = self.mu_head(h).squeeze(-1)           # (B, N)
        log_sigma = self.log_sigma_head(h).squeeze(-1)
        log_sigma = torch.clamp(log_sigma, LOG_SIG_MIN, LOG_SIG_MAX)
        return mu, log_sigma

    def sample(self, node_graph_embed: torch.Tensor,
               ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample throttle actions and compute log probabilities.

        Returns:
            action:   (B, N) throttle values in [0, 1] via sigmoid squashing
            log_prob: (B,)   sum of log probs over nodes
        """
        mu, log_sigma = self.forward(node_graph_embed)
        sigma = log_sigma.exp()
        dist = Normal(mu, sigma)

        # Reparameterization trick
        z = dist.rsample()

        # Squash to [0, 1] via sigmoid
        action = torch.sigmoid(z)

        # Log probability with sigmoid correction
        # log π(a|s) = log Normal(z) - log|da/dz|
        # da/dz = sigmoid(z) * (1 - sigmoid(z))
        log_prob = dist.log_prob(z)
        log_prob -= torch.log(action * (1 - action) + EPSILON)
        log_prob = log_prob.sum(dim=-1)  # sum over nodes → (B,)

        return action, log_prob

    def deterministic(self, node_graph_embed: torch.Tensor) -> torch.Tensor:
        """Deterministic action (for evaluation): sigmoid(mu)."""
        mu, _ = self.forward(node_graph_embed)
        return torch.sigmoid(mu)


class SACCritic(nn.Module):
    """
    Twin Q-network for SAC.
    Takes (state_embed, action) and outputs Q-values.

    Input:
        graph_embed: (B, hidden_dim)  – graph-level embedding
        action:      (B, N)           – throttle actions
    Output:
        q1, q2:      (B,)            – twin Q-values
    """

    def __init__(self, graph_dim: int, n_nodes: int, hidden_dim: int = 256):
        super().__init__()
        input_dim = graph_dim + n_nodes

        self.q1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.q2 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, graph_embed: torch.Tensor, action: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            graph_embed: (B, graph_dim)
            action:      (B, N)

        Returns:
            q1, q2: (B,)
        """
        x = torch.cat([graph_embed, action], dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)
