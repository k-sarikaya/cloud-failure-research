"""
GAT-SAC Agent – Full training logic with Lagrangian safety constraints
=======================================================================
Combines:
  - GAT Encoder (graph-aware node embeddings)
  - SAC Actor (per-node Gaussian throttle policy)
  - SAC Twin Critic (Q1, Q2)
  - Automatic entropy temperature α tuning
  - Lagrangian multiplier λ for safety constraints
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.gat_encoder import GATEncoder
from models.sac_networks import SACActorPerNode, SACCritic
from training.replay_buffer import ReplayBuffer


class GATSACAgent:
    """
    Graph Attention Soft Actor-Critic with Lagrangian Safety.

    The agent:
      1) Encodes the graph state via GAT → node_embeds, graph_embed
      2) Actor produces per-node throttle ∈ [0, 1]
      3) Critic evaluates (graph_embed, actions) → Q1, Q2
      4) Entropy temperature α is auto-tuned
      5) Lagrangian multiplier λ penalizes excessive capacity loss
    """

    def __init__(
        self,
        *,
        n_nodes: int,
        node_feat_dim: int = 4,
        gat_hidden: int = 64,
        gat_heads: int = 4,
        gat_dropout: float = 0.1,
        sac_hidden: int = 256,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        lr_alpha: float = 3e-4,
        lr_lambda: float = 1e-3,
        discount: float = 0.99,
        tau: float = 0.005,
        init_alpha: float = 0.2,
        capacity_loss_limit: float = 0.15,
        dropped_load_limit: float = 0.10,
        buffer_size: int = 500_000,
        batch_size: int = 256,
        device: str = "cpu",
    ):
        self.n_nodes = n_nodes
        self.discount = discount
        self.tau = tau
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.capacity_loss_limit = capacity_loss_limit
        self.dropped_load_limit = dropped_load_limit

        # --- Networks ---
        self.gat = GATEncoder(
            node_feat_dim=node_feat_dim,
            hidden_dim=gat_hidden,
            num_heads=gat_heads,
            dropout=gat_dropout,
        ).to(self.device)

        embed_dim = gat_hidden * 2  # [node_embed || graph_embed]
        self.actor = SACActorPerNode(
            embed_dim=embed_dim, hidden_dim=sac_hidden
        ).to(self.device)

        self.critic = SACCritic(
            graph_dim=gat_hidden, n_nodes=n_nodes, hidden_dim=sac_hidden
        ).to(self.device)

        self.critic_target = SACCritic(
            graph_dim=gat_hidden, n_nodes=n_nodes, hidden_dim=sac_hidden
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target.eval()

        # --- Entropy temperature α (auto-tuned) ---
        self.log_alpha = torch.tensor(
            [np.log(init_alpha)], requires_grad=True, device=self.device,
            dtype=torch.float32,
        )
        self.target_entropy = -float(n_nodes)  # heuristic: -dim(action)

        # --- Lagrangian multiplier λ (for safety) ---
        self.log_lambda = torch.tensor(
            [0.0], requires_grad=True, device=self.device, dtype=torch.float32
        )

        # --- Optimizers ---
        self.gat_actor_params = list(self.gat.parameters()) + list(self.actor.parameters())
        self.optim_actor = torch.optim.Adam(self.gat_actor_params, lr=lr_actor)
        self.optim_critic = torch.optim.Adam(
            list(self.gat.parameters()) + list(self.critic.parameters()), lr=lr_critic
        )
        self.optim_alpha = torch.optim.Adam([self.log_alpha], lr=lr_alpha)
        self.optim_lambda = torch.optim.Adam([self.log_lambda], lr=lr_lambda)

        # --- Replay Buffer ---
        self.replay = ReplayBuffer(buffer_size, n_nodes, node_feat_dim)

        self.total_steps = 0

        # --- Adjacency cache (avoid repeated numpy→tensor conversion) ---
        self._cached_adj: Optional[torch.Tensor] = None
        self._cached_adj_batch: Optional[torch.Tensor] = None

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    @property
    def lagrange_lambda(self) -> torch.Tensor:
        return self.log_lambda.exp()

    # ---------- Mode switching ----------

    def eval_mode(self) -> None:
        """Switch to eval mode: disable dropout for faster/stable inference."""
        self.gat.eval()
        self.actor.eval()
        self.critic.eval()

    def train_mode(self) -> None:
        """Switch back to training mode."""
        self.gat.train()
        self.actor.train()
        self.critic.train()

    def cache_adjacency(self, adj: np.ndarray) -> None:
        """Pre-cache adjacency tensor to avoid per-step conversion."""
        self._cached_adj = torch.tensor(adj, device=self.device, dtype=torch.float32)

    # ---------- Inference ----------

    def _encode(self, state: torch.Tensor, adj: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode state through GAT, returning concatenated embed for actor."""
        node_embeds, graph_embed = self.gat(state, adj)  # (B,N,H), (B,H)
        g_exp = graph_embed.unsqueeze(1).expand_as(node_embeds)  # (B,N,H)
        combined = torch.cat([node_embeds, g_exp], dim=-1)  # (B,N,2H)
        return combined, graph_embed

    def act(self, state: np.ndarray, adj: np.ndarray,
            deterministic: bool = False) -> np.ndarray:
        """
        Select throttle actions for given state.

        Args:
            state: (N, feat_dim) node features
            adj:   (N, N) adjacency matrix
            deterministic: if True, use mean action (no sampling)

        Returns:
            action: (N,) throttle values in [0, 1]
        """
        with torch.no_grad():
            s = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(0)
            # Use cached adjacency if available
            if self._cached_adj is not None:
                a_mat = self._cached_adj.unsqueeze(0)
            else:
                a_mat = torch.tensor(adj, device=self.device, dtype=torch.float32).unsqueeze(0)

            combined, _ = self._encode(s, a_mat)

            if deterministic:
                action = self.actor.deterministic(combined)
            else:
                action, _ = self.actor.sample(combined)

            return action.squeeze(0).cpu().numpy()

    # ---------- Training ----------

    def update(self, adj: np.ndarray, rng: np.random.Generator
               ) -> Optional[dict]:
        """
        Perform one SAC + Lagrangian update step.

        Returns dict of losses or None if buffer insufficient.
        """
        if self.replay.size < self.batch_size:
            return None

        s, a, r, ns, d, c = self.replay.sample(self.batch_size, rng)

        s_t = torch.tensor(s, device=self.device)
        a_t = torch.tensor(a, device=self.device)
        r_t = torch.tensor(r, device=self.device)
        ns_t = torch.tensor(ns, device=self.device)
        d_t = torch.tensor(d, device=self.device)
        c_t = torch.tensor(c, device=self.device)

        adj_t = torch.tensor(adj, device=self.device).float()
        adj_batch = adj_t.unsqueeze(0).expand(self.batch_size, -1, -1)

        # 1) Encode current and next states
        combined_s, graph_s = self._encode(s_t, adj_batch)
        with torch.no_grad():
            combined_ns, graph_ns = self._encode(ns_t, adj_batch)

        # 2) Critic loss (TD target with clipped double-Q)
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(combined_ns)
            q1_next, q2_next = self.critic_target(graph_ns, next_action)
            q_next = torch.min(q1_next, q2_next) - self.alpha.detach() * next_log_prob
            target_q = r_t + (1.0 - d_t) * self.discount * q_next

        q1, q2 = self.critic(graph_s.detach(), a_t)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.optim_critic.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.optim_critic.step()

        # 3) Actor loss (maximize Q - α * entropy + Lagrangian penalty)
        new_action, log_prob = self.actor.sample(combined_s.detach())
        q1_new, q2_new = self.critic(graph_s.detach(), new_action)
        q_pi = torch.min(q1_new, q2_new)

        # Lagrangian: penalize excessive throttling (proxy for capacity loss)
        throttle_cost = new_action.sum(dim=-1)  # (B,) total throttle intensity
        lagrangian_penalty = self.lagrange_lambda.detach() * (
            throttle_cost - self.capacity_loss_limit * self.n_nodes
        )

        actor_loss = (self.alpha.detach() * log_prob - q_pi + lagrangian_penalty).mean()

        self.optim_actor.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.gat_actor_params, 10.0)
        self.optim_actor.step()

        # 4) Temperature α update
        alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
        self.optim_alpha.zero_grad()
        alpha_loss.backward()
        self.optim_alpha.step()

        # 5) Lagrangian λ update (dual ascent)
        lambda_loss = -self.log_lambda * (
            c_t.mean() - self.capacity_loss_limit
        )
        self.optim_lambda.zero_grad()
        lambda_loss.backward()
        self.optim_lambda.step()

        # 6) Soft target update
        self._soft_update()

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha": self.alpha.item(),
            "lambda": self.lagrange_lambda.item(),
            "mean_throttle": new_action.mean().item(),
        }

    def _soft_update(self) -> None:
        for tp, sp in zip(self.critic_target.parameters(),
                          self.critic.parameters()):
            tp.data.copy_(self.tau * sp.data + (1.0 - self.tau) * tp.data)

    # ---------- Save / Load ----------

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "gat": self.gat.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "log_alpha": self.log_alpha.data,
            "log_lambda": self.log_lambda.data,
        }, path)

    def load(self, path: str, *, inference_only: bool = False) -> None:
        """
        Load model weights from disk.

        When ``inference_only`` is enabled we only restore the GAT encoder,
        actor, and scalar temperatures. This supports fixed-policy transfer
        experiments across different graph sizes, where the critic shape no
        longer matches because it concatenates the full action vector.
        """
        ckpt = torch.load(path, map_location=self.device)
        self.gat.load_state_dict(ckpt["gat"])
        self.actor.load_state_dict(ckpt["actor"])

        if not inference_only:
            self.critic.load_state_dict(ckpt["critic"])
            self.critic_target.load_state_dict(ckpt["critic_target"])

        self.log_alpha.data = ckpt["log_alpha"]
        self.log_lambda.data = ckpt["log_lambda"]
