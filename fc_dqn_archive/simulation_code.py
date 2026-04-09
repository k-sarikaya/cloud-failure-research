"""
Fully reproducible DQN vs baseline comparison for cascading-failure mitigation.

Paper context:
  "Dynamic Risk Mitigation in Edge Computing: A Queuing-Theoretic Framework
   for Cascading Failure Prevention"

Core mismatch we resolve:
  - Physical environment has N=100 nodes.
  - Trained DQN expects a 150-dim state.

Active Decision Window (paper mechanism):
  1) Compute utilization for each node: U_i = L_i / C_i (failed nodes -> 0).
  2) Take top Na=50 nodes by U_i.
  3) Build state vector (150-dim):
       [o_1..o_50, U_1..U_50, C_1/C_total .. C_50/C_total]

Paired benchmarking:
  - We pre-generate 20 scenarios (topology + capacities + loads + per-step random uniforms).
  - Every method is evaluated on the *exact same* scenarios and per-step uniforms.

Dependencies:
  Python 3.10+, NumPy, Pandas, PyTorch.
"""

from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_global_seeds(seed: int) -> None:
    """Best-effort determinism."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------------
# Barabasi-Albert topology (no networkx dependency)
# -----------------------------


import networkx as nx

def generate_ba_adj(n: int, m: int, rng: np.random.Generator) -> List[List[int]]:
    G = nx.barabasi_albert_graph(n, m, seed=int(rng.integers(0, 100000)))
    return [list(G.neighbors(i)) for i in range(n)]

def generate_ws_adj(n: int, k: int=4, p: float=0.2, rng=None) -> List[List[int]]:
    seed = int(rng.integers(0, 100000)) if rng else None
    G = nx.connected_watts_strogatz_graph(n, k, p, tries=100, seed=seed)
    return [list(G.neighbors(i)) for i in range(n)]

def generate_er_adj(n: int, p: float=0.06, rng=None) -> List[List[int]]:
    seed = int(rng.integers(0, 100000)) if rng else None
    G = nx.erdos_renyi_graph(n, p, seed=seed)
    while not nx.is_connected(G):
        G = nx.erdos_renyi_graph(n, p, seed=int(rng.integers(0, 100000)))
    return [list(G.neighbors(i)) for i in range(n)]


# -----------------------------
# Paired scenario: initial state + per-step uniforms (paired randomness)
# -----------------------------


@dataclass(frozen=True)
class Scenario:
    adj: List[List[int]]  # adjacency list (length N)
    capacities: np.ndarray  # shape (N,), float32
    loads: np.ndarray  # shape (N,), float32
    fail_uniforms: np.ndarray  # shape (max_steps, N), float32


def make_scenario(
    *,
    n_nodes: int,
    topology_args: dict,
    max_steps: int,
    capacity_range: Tuple[float, float],
    init_util_range: Tuple[float, float],
    rng: np.random.Generator,
) -> Scenario:
    topo_type = topology_args.get("type", "ba")
    if topo_type == "ba":
        adj = generate_ba_adj(n_nodes, topology_args.get("ba_m", 3), rng)
    elif topo_type == "ws":
        adj = generate_ws_adj(n_nodes, k=4, p=0.2, rng=rng)
    elif topo_type == "er":
        adj = generate_er_adj(n_nodes, p=0.06, rng=rng)
    else:
        adj = generate_ba_adj(n_nodes, 3, rng)

    c_low, c_high = capacity_range
    capacities = rng.uniform(c_low, c_high, size=n_nodes).astype(np.float32)

    u_low, u_high = init_util_range
    init_util = rng.uniform(u_low, u_high, size=n_nodes).astype(np.float32)
    loads = (init_util * capacities).astype(np.float32)

    # Fixed U(0,1) draws used for node failure sampling at each step (paired eval).
    fail_uniforms = rng.random(size=(max_steps, n_nodes), dtype=np.float32)
    return Scenario(adj=adj, capacities=capacities, loads=loads, fail_uniforms=fail_uniforms)


def generate_paired_scenarios(
    *,
    seed: int,
    num_scenarios: int,
    n_nodes: int,
    topo_args: dict,
    max_steps: int,
    capacity_range: Tuple[float, float],
    init_util_range: Tuple[float, float],
) -> List[Scenario]:
    rng = np.random.default_rng(seed)
    return [
        make_scenario(
            n_nodes=n_nodes,
            topology_args=topo_args,
            max_steps=max_steps,
            capacity_range=capacity_range,
            init_util_range=init_util_range,
            rng=rng,
        )
        for _ in range(int(num_scenarios))
    ]


# -----------------------------
# Environment
# -----------------------------


class EdgeNetworkEnv:
    """
    Cascading-failure simulation environment.

    Per-step dynamics (dt = 1 minute):
      1) Optionally apply a pruning action (controlled shutdown).
      2) Sample natural failures using M/M/1-inspired exponential hazard:
           lambda_i(L_i) = lambda_base * exp(alpha * (L_i / C_i))
           P(fail in dt) = 1 - exp(-lambda_i * dt_hours)
         Randomness is *paired* across methods via scenario.fail_uniforms[t, i].
      3) For each node that fails:
           - set o_i=0, C_i=0
           - redistribute gamma * L_i to operational neighbors
         Redistribution weights are proportional to each neighbor's remaining empty capacity:
           w_j proportional to max(C_j - L_j, 0).
    """

    def __init__(
        self,
        *,
        scenario: Scenario,
        dt_hours: float,
        max_hours: float,
        lambda_base: float,
        alpha: float,
        gamma_redistribute: float,
        gamma_prune: float,
        demand_threshold_frac: float,
        redundancy_factor: float = 1.0,
        prune_util_min: float = 2.0,
    ) -> None:
        self.adj = scenario.adj
        self.n = len(self.adj)

        self.dt_hours = float(dt_hours)
        self.max_hours = float(max_hours)
        self.max_steps = int(round(self.max_hours / self.dt_hours))

        self.lambda_base = float(lambda_base)
        self.alpha = float(alpha)
        self.gamma_redistribute = float(gamma_redistribute)
        self.gamma_prune = float(gamma_prune)
        self.demand_threshold_frac = float(demand_threshold_frac)
        self.redundancy_factor = float(redundancy_factor)
        self.prune_util_min = float(prune_util_min)

        self._cap0 = (scenario.capacities * self.redundancy_factor).astype(np.float32)
        self._load0 = scenario.loads.astype(np.float32)
        self._fail_uniforms = scenario.fail_uniforms.astype(np.float32)
        if self._fail_uniforms.shape != (self.max_steps, self.n):
            raise ValueError(
                f"scenario.fail_uniforms must have shape ({self.max_steps}, {self.n}), "
                f"got {self._fail_uniforms.shape}"
            )

        # Mutable state (initialized in reset()).
        self.t = 0
        self.operational: np.ndarray
        self.capacities: np.ndarray
        self.loads: np.ndarray
        self.initial_total_capacity = 0.0
        self.demand_threshold = 0.0
        self.cum_pruned_capacity = 0.0
        self.cum_natural_failed_capacity = 0.0   # capacity lost to natural failures
        self.natural_failure_count = 0            # total nodes that died naturally
        self.done = False

        self.reset()

    def reset(self) -> None:
        self.t = 0
        self.operational = np.ones(self.n, dtype=np.bool_)
        self.capacities = self._cap0.copy()
        self.loads = self._load0.copy()

        self.initial_total_capacity = float(self.capacities.sum())
        self.base_nominal_capacity = self.initial_total_capacity / self.redundancy_factor
        self.demand_threshold = self.demand_threshold_frac * self.base_nominal_capacity
        self.cum_pruned_capacity = 0.0
        self.cum_natural_failed_capacity = 0.0
        self.natural_failure_count = 0
        self.prune_cooldown = 0
        self.done = False

    def utilization(self) -> np.ndarray:
        """U_i = L_i/C_i for operational nodes; 0 for failed nodes."""
        util = np.zeros(self.n, dtype=np.float32)
        mask = self.operational & (self.capacities > 0)
        util[mask] = self.loads[mask] / self.capacities[mask]
        return util

    def get_active_subgraph_state(self, na: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
          state: float32, shape (3*na,)
          window_nodes: int64, shape (na,), global node IDs for each window slot
          action_mask: bool, shape (na+1,), True for valid actions

        Action convention for the DQN:
          - action 0..(na-1): prune the corresponding window slot
          - action na: no-op
        """
        if not (1 <= na <= self.n):
            raise ValueError("na must satisfy 1 <= na <= n_nodes")

        util = self.utilization()
        node_ids = np.arange(self.n, dtype=np.int64)

        # Deterministic ordering: primary = -util (descending), secondary = node_id (ascending).
        order = np.lexsort((node_ids, -util))
        window_nodes = order[:na].astype(np.int64, copy=False)

        o_feat = self.operational[window_nodes].astype(np.float32)
        u_feat = util[window_nodes].astype(np.float32)
        
        # State Normalization (MinMaxScaler): Clip abnormal spikes to 3.0 and normalize to [0, 1]
        u_feat = np.clip(u_feat, 0.0, 3.0) / 3.0
        
        cap_rel = (self.capacities[window_nodes] / max(float(self.initial_total_capacity), 1e-9)).astype(np.float32)
        state = np.concatenate([o_feat, u_feat, cap_rel], axis=0).astype(np.float32)

        action_mask = np.zeros((na + 1,), dtype=np.bool_)
        for k, nid in enumerate(window_nodes):
            action_mask[k] = bool(self.operational[int(nid)])
            
        # Action Masking: Enforce 5-step cooldown where pruning is physically impossible
        if getattr(self, "prune_cooldown", 0) > 0:
            action_mask[:na] = False

        # Action Masking: Heuristic Safety Mask (Step 5)
        prune_min = getattr(self, "prune_util_min", 0.0)
        if prune_min > 0.0:
            for k, nid in enumerate(window_nodes):
                if action_mask[k] and (util[nid] < prune_min):
                    action_mask[k] = False

        action_mask[na] = True  # no-op always valid

        return state, window_nodes, action_mask

    def step(self, prune_node_id: Optional[int]) -> Tuple[float, bool, dict]:
        """
        One dt step. Returns (reward, done, info).
        """
        if self.done:
            raise RuntimeError("step() called after termination")

        total_load_before = float(self.loads.sum())
        dropped_load = 0.0

        # Cooldown countdown
        if getattr(self, "prune_cooldown", 0) > 0:
            self.prune_cooldown -= 1

        # 1) Apply proactive pruning first.
        pruned_capacity = 0.0
        if prune_node_id is not None:
            nid = int(prune_node_id)
            if 0 <= nid < self.n and self.operational[nid] and getattr(self, "prune_cooldown", 0) == 0:
                pruned_capacity = float(self.capacities[nid])
                self.cum_pruned_capacity += pruned_capacity
                dropped_load += self._shutdown_and_redistribute(nid, gamma=self.gamma_prune)
                self.prune_cooldown = 5  # Start 5-step cooldown penalty

        # 2) Sample natural failures (paired randomness).
        failed_nodes = self._sample_failures()
        for nid in failed_nodes:
            self.cum_natural_failed_capacity += float(self._cap0[nid])   # original capacity
            self.natural_failure_count += 1
            dropped_load += self._shutdown_and_redistribute(nid, gamma=self.gamma_redistribute)

        num_failures = int(len(failed_nodes))

        # 3) Termination check.
        self.t += 1
        remaining_capacity = float(self.capacities.sum())
        terminated_by_capacity = remaining_capacity < self.demand_threshold
        terminated_by_time = self.t >= self.max_steps
        self.done = terminated_by_capacity or terminated_by_time

        # Reward formulation (Reviewer Proof Action Plan):
        reward = 1.0  # survival bonus

        if total_load_before > 0:
            task_loss_ratio = dropped_load / total_load_before
            reward -= 0.1 * task_loss_ratio

        if num_failures == 0:
            reward += 10.0
        else:
            reward -= 5.0 * float(num_failures)

        if self.done and terminated_by_capacity:
            reward -= 100.0

        info = {
            "t": self.t,
            "hours": self.t * self.dt_hours,
            "num_failures": num_failures,
            "pruned_capacity": pruned_capacity,
            "proactive_cap_loss_ratio": self.cum_pruned_capacity / max(self.initial_total_capacity, 1e-9),
            "natural_cap_loss_ratio": self.cum_natural_failed_capacity / max(self.initial_total_capacity, 1e-9),
            "total_cap_loss_ratio": (self.cum_pruned_capacity + self.cum_natural_failed_capacity) / max(self.initial_total_capacity, 1e-9),
            "natural_failure_count": self.natural_failure_count,
            "remaining_capacity": remaining_capacity,
            "terminated_by_capacity": terminated_by_capacity,
            "terminated_by_time": terminated_by_time,
        }
        return float(reward), self.done, info

    def _sample_failures(self) -> List[int]:
        util = self.utilization()
        lam = self.lambda_base * np.exp(self.alpha * util)
        p_fail = 1.0 - np.exp(-lam * self.dt_hours)
        u = self._fail_uniforms[self.t]

        will_fail = (u < p_fail) & self.operational
        return np.where(will_fail)[0].astype(int).tolist()

    def _shutdown_and_redistribute(self, node_id: int, *, gamma: float) -> float:
        node_id = int(node_id)
        if not self.operational[node_id]:
            return 0.0

        load = float(self.loads[node_id])
        dropped = load * (1.0 - gamma)

        # Shutdown: remove node from capacity pool and clear its load.
        self.operational[node_id] = False
        self.capacities[node_id] = 0.0
        self.loads[node_id] = 0.0

        # Redistribute gamma*load to operational neighbors.
        neigh = self.adj[node_id]
        alive = [j for j in neigh if self.operational[j]]
        if not alive:
            return load

        moved = gamma * load

        caps = self.capacities[alive].astype(np.float32)
        loads = self.loads[alive].astype(np.float32)
        empty = np.maximum(caps - loads, 0.0)
        total_empty = float(empty.sum())
        if total_empty <= 1e-12:
            weights = np.ones((len(alive),), dtype=np.float32) / float(len(alive))
        else:
            weights = empty / total_empty

        for j, w in zip(alive, weights.tolist()):
            self.loads[int(j)] += moved * float(w)

        return dropped


# -----------------------------
# Baseline agents
# -----------------------------


class NoDefenseAgent:
    """Always no-op (no proactive pruning)."""

    def act(self, _env: EdgeNetworkEnv) -> Optional[int]:
        return None


class HeuristicPruningAgent:
    """
    If any node is extremely over-utilized (U_i > threshold), prune the single most utilized node.
    """

    def __init__(self, util_threshold: float = 0.95) -> None:
        self.util_threshold = float(util_threshold)

    def act(self, env: EdgeNetworkEnv) -> Optional[int]:
        util = env.utilization()
        if not np.any(env.operational):
            return None

        util_masked = util.copy()
        util_masked[~env.operational] = -1.0
        worst = int(np.argmax(util_masked))
        if util_masked[worst] > self.util_threshold:
            return worst
        return None


# -----------------------------
# DQN (PyTorch)
# -----------------------------


class QNet(nn.Module):
    """
    DQN function approximator:
      150 -> 256 -> 256 -> 51
    """

    def __init__(self, state_dim: int, action_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GNNQNet(nn.Module):
    """
    GraphSAGE-style Q-network (pure PyTorch, no torch_geometric required).

    Architecture:
        2 × GraphSAGE message-passing layers → per-node embeddings
        Global mean pool for graph-level context
        Q-head: [node_embed || global_embed] → Q-value per action

    Node features (dim=3): [operational, utilization_norm, cap_rel]
    """

    def __init__(self, node_feat_dim: int, hidden_dim: int, na: int) -> None:
        super().__init__()
        self.na = na
        # GraphSAGE layers: concat(self, neigh_mean) => hidden
        self.sage1 = nn.Linear(node_feat_dim * 2, hidden_dim)
        self.sage2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.global_fc = nn.Linear(hidden_dim, hidden_dim)
        # Q-head: for each window node and the no-op action
        self.q_head = nn.Linear(hidden_dim * 2, 1)
        self.relu = nn.ReLU()

    def _sage(self, x: torch.Tensor, adj_norm: torch.Tensor, layer: nn.Linear) -> torch.Tensor:
        neigh = torch.mm(adj_norm, x)
        return self.relu(layer(torch.cat([x, neigh], dim=-1)))

    def forward(
        self,
        node_feats: torch.Tensor,  # (n, node_feat_dim)
        adj_norm: torch.Tensor,    # (n, n) row-normalised
        window_idx: torch.Tensor,  # (na,) indices of candidate nodes
    ) -> torch.Tensor:             # returns (na+1,) Q-values
        h = self._sage(node_feats, adj_norm, self.sage1)  # (n, hidden)
        h = self._sage(h, adj_norm, self.sage2)           # (n, hidden)

        g = self.relu(self.global_fc(h.mean(dim=0)))      # (hidden,)
        h_win = h[window_idx]                             # (na, hidden)
        g_exp = g.unsqueeze(0).expand(self.na, -1)        # (na, hidden)

        q_prune = self.q_head(torch.cat([h_win, g_exp], dim=-1)).squeeze(-1)  # (na,)
        q_noop  = self.q_head(torch.cat([g, g], dim=-1).unsqueeze(0)).squeeze()
        if q_noop.dim() > 0:
            q_noop = q_noop.mean()
        return torch.cat([q_prune, q_noop.unsqueeze(0)], dim=0)  # (na+1,)

    @staticmethod
    def build_adj_norm(adj_list: List[List[int]], n: int, device: torch.device) -> torch.Tensor:
        """Row-normalised adjacency with self-loops from adjacency list."""
        A = torch.zeros(n, n, device=device)
        for i, nbrs in enumerate(adj_list):
            A[i, i] = 1.0
            for j in nbrs:
                A[i, j] = 1.0
        deg = A.sum(dim=1, keepdim=True).clamp(min=1.0)
        return A / deg


# GNN Constants
GNN_NODE_FEAT_DIM = 3   # [operational, util_norm, cap_rel]
GNN_HIDDEN_DIM    = 64


class ReplayBuffer:

    """
    Replay buffer storing (state, action, reward, next_state, done) plus masks.

    Masks are required here because not every action is valid at every step
    (you cannot prune an already failed node, and padding slots can exist).
    """

    def __init__(self, capacity: int, state_dim: int, action_dim: int) -> None:
        self.capacity = int(capacity)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)

        self.ptr = 0
        self.size = 0

        self.s = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.a = np.zeros((self.capacity,), dtype=np.int64)
        self.r = np.zeros((self.capacity,), dtype=np.float32)
        self.ns = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.d = np.zeros((self.capacity,), dtype=np.float32)
        self.mask = np.zeros((self.capacity, self.action_dim), dtype=np.bool_)
        self.nmask = np.zeros((self.capacity, self.action_dim), dtype=np.bool_)

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        mask: np.ndarray,
        next_mask: np.ndarray,
    ) -> None:
        i = self.ptr
        self.s[i] = state
        self.a[i] = int(action)
        self.r[i] = float(reward)
        self.ns[i] = next_state
        self.d[i] = 1.0 if done else 0.0
        self.mask[i] = mask
        self.nmask[i] = next_mask

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator) -> Tuple[np.ndarray, ...]:
        idx = rng.integers(0, self.size, size=int(batch_size))
        return (
            self.s[idx],
            self.a[idx],
            self.r[idx],
            self.ns[idx],
            self.d[idx],
            self.mask[idx],
            self.nmask[idx],
        )


class DQNAgent:
    def __init__(
        self,
        *,
        state_dim: int,
        action_dim: int,
        lr: float,
        discount: float,
        buffer_size: int,
        batch_size: int,
        target_update: int,
        device: str,
        use_gnn: bool = True,                         # NEW: flag to switch networks
        adj_list: Optional[List[List[int]]] = None,   # full graph adj, needed for GNN
        n_nodes: int = 100,                           # total graph nodes
        na: int = 50,                                 # active window size
    ) -> None:
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.discount = float(discount)
        self.batch_size = int(batch_size)
        self.target_update = int(target_update)
        self.device = torch.device(device)
        self.na = na
        self.n_nodes = n_nodes
        self.use_gnn = use_gnn and (adj_list is not None)

        if self.use_gnn:
            self.q  = GNNQNet(GNN_NODE_FEAT_DIM, GNN_HIDDEN_DIM, na).to(self.device)
            self.tq = GNNQNet(GNN_NODE_FEAT_DIM, GNN_HIDDEN_DIM, na).to(self.device)
            # Pre-build full-graph adj_norm (stays on device for fast matmul)
            self.adj_norm = GNNQNet.build_adj_norm(adj_list, n_nodes, self.device)
        else:
            self.q  = QNet(self.state_dim, self.action_dim).to(self.device)
            self.tq = QNet(self.state_dim, self.action_dim).to(self.device)
            self.adj_norm = None

        self.tq.load_state_dict(self.q.state_dict())
        self.tq.eval()

        self.optim = torch.optim.Adam(self.q.parameters(), lr=float(lr))
        self.replay = ReplayBuffer(buffer_size, self.state_dim, self.action_dim)

        self.total_steps = 0

    # ── helpers for GNN forward pass ─────────────────────────────────────────
    def _gnn_forward(
        self,
        net: GNNQNet,
        state_vec: torch.Tensor,   # (B, 3*na) or (3*na,)
    ) -> torch.Tensor:             # (B, na+1) or (na+1,)
        """
        Vectorized batch GNN forward.
        State layout: [op_0..op_{na-1} | util_0..util_{na-1} | cap_0..cap_{na-1}]
        Reshapes to (B, na, 3) node features and runs GraphSAGE using bmm.
        """
        batched = (state_vec.dim() == 2)
        if not batched:
            state_vec = state_vec.unsqueeze(0)

        B   = state_vec.shape[0]
        na  = self.na
        n   = self.n_nodes

        # ── Unpack flat state → per-node features ──────────────────────────
        op_f = state_vec[:, :na]           # (B, na)
        ut_f = state_vec[:, na:2*na]       # (B, na)
        cp_f = state_vec[:, 2*na:3*na]     # (B, na)
        # Build full-graph node feat tensor; only window nodes (0..na-1) are
        # populated from state; remaining nodes get zero features.
        nf_full = torch.zeros(B, n, GNN_NODE_FEAT_DIM, device=self.device)
        nf_full[:, :na, 0] = op_f
        nf_full[:, :na, 1] = ut_f
        nf_full[:, :na, 2] = cp_f          # (B, n, 3)

        # adj_norm: (n, n) → expand to (B, n, n) for bmm
        A = self.adj_norm.unsqueeze(0).expand(B, -1, -1)   # (B, n, n)

        # ── Two GraphSAGE layers (vectorized) ────────────────────────────
        h = nf_full  # (B, n, 3)
        for layer in (net.sage1, net.sage2):
            neigh = torch.bmm(A, h)                        # (B, n, d)
            cat   = torch.cat([h, neigh], dim=-1)          # (B, n, 2d)
            h     = net.relu(layer(cat))                   # (B, n, hidden)

        # ── Global embedding ─────────────────────────────────────────────
        g = h.mean(dim=1)                                  # (B, hidden)
        g = net.relu(net.global_fc(g))                     # (B, hidden)

        # ── Per-window-node Q scores ─────────────────────────────────────
        win_idx = torch.arange(na, device=self.device)
        h_win   = h[:, win_idx, :]                         # (B, na, hidden)
        g_exp   = g.unsqueeze(1).expand(-1, na, -1)        # (B, na, hidden)
        cat_win = torch.cat([h_win, g_exp], dim=-1)        # (B, na, hidden*2)
        q_prune = net.q_head(cat_win).squeeze(-1)          # (B, na)

        # ── No-op Q score ────────────────────────────────────────────────
        g2 = torch.cat([g, g], dim=-1)                     # (B, hidden*2)
        q_noop = net.q_head(g2)                            # (B, 1)

        q_vals = torch.cat([q_prune, q_noop], dim=-1)      # (B, na+1)
        return q_vals if batched else q_vals.squeeze(0)

    def act(self, state: np.ndarray, mask: np.ndarray, epsilon: float, rng: np.random.Generator) -> int:
        """
        Epsilon-greedy action selection with masking.

        Action convention:
          - 0..(na-1): prune the corresponding Active Window slot
          - na: no-op
        """
        valid = np.where(mask)[0]
        if valid.size == 0:
            return self.action_dim - 1  # no-op fallback

        if rng.random() < float(epsilon):
            # Safe Exploration: 95% chance to pick no-op, 5% chance to prune randomly
            if rng.random() < 0.95 and (self.action_dim - 1) in valid:
                return self.action_dim - 1
            else:
                return int(rng.choice(valid))

        with torch.no_grad():
            s = torch.tensor(state, device=self.device).float()
            if self.use_gnn:
                qvals = self._gnn_forward(self.q, s).cpu().numpy()
            else:
                qvals = self.q(s.unsqueeze(0)).squeeze(0).cpu().numpy()
        qvals[~mask] = -1e9
        return int(np.argmax(qvals))

    def update(self, rng: np.random.Generator) -> Optional[float]:
        if self.replay.size < self.batch_size:
            return None

        s, a, r, ns, d, _mask, nmask = self.replay.sample(self.batch_size, rng)

        s_t      = torch.tensor(s,     device=self.device).float()
        a_t      = torch.tensor(a,     device=self.device).long()
        r_t      = torch.tensor(r,     device=self.device)
        ns_t     = torch.tensor(ns,    device=self.device).float()
        d_t      = torch.tensor(d,     device=self.device)
        nmask_t  = torch.tensor(nmask, device=self.device).bool()

        if self.use_gnn:
            q_all    = self._gnn_forward(self.q,  s_t)   # (B, na+1)
            q_sa     = q_all.gather(1, a_t.view(-1, 1)).squeeze(1)
            with torch.no_grad():
                q_next_online = self._gnn_forward(self.q,  ns_t)
                q_next_online = q_next_online.masked_fill(~nmask_t, -1e9)
                a_next        = q_next_online.argmax(dim=1, keepdim=True)
                q_next_target = self._gnn_forward(self.tq, ns_t)
                q_next        = q_next_target.gather(1, a_next).squeeze(1)
        else:
            q_sa = self.q(s_t).gather(1, a_t.view(-1, 1)).squeeze(1)
            with torch.no_grad():
                q_next_online = self.q(ns_t)
                q_next_online = q_next_online.masked_fill(~nmask_t, -1e9)
                a_next        = q_next_online.argmax(dim=1, keepdim=True)
                q_next_target = self.tq(ns_t)
                q_next        = q_next_target.gather(1, a_next).squeeze(1)

        target = r_t + (1.0 - d_t) * self.discount * q_next
        loss   = F.smooth_l1_loss(q_sa, target)
        self.optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), max_norm=10.0)
        self.optim.step()

        return float(loss.item())

    def maybe_sync_target(self) -> None:
        if self.total_steps > 0 and (self.total_steps % self.target_update == 0):
            self.tq.load_state_dict(self.q.state_dict())

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({"state_dict": self.q.state_dict()}, path)

    def load(self, path: str) -> None:
        payload = torch.load(path, map_location=self.device)
        self.q.load_state_dict(payload["state_dict"])
        self.tq.load_state_dict(payload["state_dict"])
        self.tq.eval()


def epsilon_by_step(step: int, *, eps_start: float, eps_end: float, decay_steps: int) -> float:
    """
    Linear epsilon decay over a fixed number of environment steps.

    Requested schedule: 1.0 -> 0.01 over 50,000 steps (then hold at 0.01).
    """
    if step >= decay_steps:
        return float(eps_end)
    frac = step / float(max(decay_steps, 1))
    return float(eps_start + frac * (eps_end - eps_start))


def train_dqn(
    *,
    agent: DQNAgent,
    n_nodes: int,
    ba_m: int,
    na: int,
    episodes: int,
    seed: int,
    # Environment params
    dt_hours: float,
    max_hours: float,
    lambda_base: float,
    alpha: float,
    gamma_redistribute: float,
    gamma_prune: float,
    demand_threshold_frac: float,
    prune_util_min: float,
    capacity_range: Tuple[float, float],
    init_util_range: Tuple[float, float],
    # DQN exploration
    eps_start: float,
    eps_end: float,
    eps_decay_steps: int,
) -> None:
    """
    Train the DQN for 'episodes' episodes.

    For reproducibility, we generate each episode's scenario from rng(seed).
    """
    rng = np.random.default_rng(seed)
    max_steps = int(round(max_hours / dt_hours))

    # === EXPERT DEMONSTRATION (Pre-fill Buffer with No-Op bounds) ===
    print("[pre-fill] Collecting No-Op (Expert) demonstrations...")
    num_prefill = 100
    for _ in range(num_prefill):
        p_scenario = make_scenario(
            n_nodes=n_nodes, topology_args={"type": "ba", "ba_m": ba_m}, max_steps=max_steps,
            capacity_range=capacity_range, init_util_range=init_util_range, rng=rng,
        )
        p_env = EdgeNetworkEnv(
            scenario=p_scenario, dt_hours=dt_hours, max_hours=max_hours,
            lambda_base=lambda_base, alpha=alpha, gamma_redistribute=gamma_redistribute,
            gamma_prune=gamma_prune, demand_threshold_frac=demand_threshold_frac, redundancy_factor=1.0,
            prune_util_min=prune_util_min,
        )
        p_env.reset()
        p_done = False
        p_state, _, p_mask = p_env.get_active_subgraph_state(na)
        while not p_done:
            p_action = na  # Action index for No-Op
            p_reward, p_done, _ = p_env.step(None)
            p_next_state, _, p_next_mask = p_env.get_active_subgraph_state(na)
            agent.replay.add(p_state, p_action, p_reward, p_next_state, p_done, p_mask, p_next_mask)
            p_state, p_mask = p_next_state, p_next_mask
            
    print(f"[pre-fill] Added {agent.replay.size} No-Op steps to Replay Buffer.")
    # ================================================================

    for ep in range(int(episodes)):
        scenario = make_scenario(
            n_nodes=n_nodes,
            topology_args={"type": "ba", "ba_m": ba_m},
            max_steps=max_steps,
            capacity_range=capacity_range,
            init_util_range=init_util_range,
            rng=rng,
        )
        env = EdgeNetworkEnv(
            scenario=scenario,
            dt_hours=dt_hours,
            max_hours=max_hours,
            lambda_base=lambda_base,
            alpha=alpha,
            gamma_redistribute=gamma_redistribute,
            gamma_prune=gamma_prune,
            demand_threshold_frac=demand_threshold_frac,
            redundancy_factor=1.0,
            prune_util_min=prune_util_min,
        )

        env.reset()
        done = False
        ep_steps = 0
        ep_return = 0.0
        losses: List[float] = []

        state, window_nodes, mask = env.get_active_subgraph_state(na)

        while not done:
            eps = epsilon_by_step(
                agent.total_steps, eps_start=eps_start, eps_end=eps_end, decay_steps=eps_decay_steps
            )
            action = agent.act(state, mask, eps, rng)

            # Map action -> global node id (or no-op).
            prune_nid = None
            if action != na:
                prune_nid = int(window_nodes[int(action)])

            reward, done, _info = env.step(prune_nid)
            next_state, next_window_nodes, next_mask = env.get_active_subgraph_state(na)

            agent.replay.add(state, action, reward, next_state, done, mask, next_mask)

            agent.total_steps += 1
            agent.maybe_sync_target()
            loss = agent.update(rng)
            if loss is not None:
                losses.append(loss)

            state, window_nodes, mask = next_state, next_window_nodes, next_mask
            ep_return += float(reward)
            ep_steps += 1

        if (ep + 1) % 25 == 0 or ep == 0:
            mean_loss = float(np.mean(losses)) if losses else float("nan")
            print(
                f"[train] ep={ep+1:4d}/{episodes} steps={ep_steps:4d} "
                f"return={ep_return:8.1f} mean_loss={mean_loss:8.5f} total_steps={agent.total_steps}"
            )


def run_episode(
    *,
    env: EdgeNetworkEnv,
    method: str,
    na: int,
    dqn: Optional[DQNAgent],
    heuristic: Optional[HeuristicPruningAgent],
) -> Tuple[float, float, float, float, int]:
    """
    Run a single episode and return:
      (sst_hours, proactive_loss_ratio, natural_loss_ratio, total_loss_ratio, n_natural_failures)
    """
    env.reset()
    done = False

    if method == "DQN Strategic":
        if dqn is None:
            raise ValueError("dqn is required for DQN Strategic")
        rng = np.random.default_rng(0)  # epsilon=0 in eval, so deterministic regardless
        state, window_nodes, mask = env.get_active_subgraph_state(na)
        while not done:
            action = dqn.act(state, mask, epsilon=0.0, rng=rng)
            prune_nid = None
            if action != na:
                prune_nid = int(window_nodes[int(action)])
            _r, done, _info = env.step(prune_nid)
            state, window_nodes, mask = env.get_active_subgraph_state(na)

    elif method == "Heuristic":
        if heuristic is None:
            raise ValueError("heuristic is required for Heuristic")
        while not done:
            prune_nid = heuristic.act(env)
            _r, done, _info = env.step(prune_nid)

    elif method in ("No Defense", "+25% Backup", "+50% Backup"):
        while not done:
            _r, done, _info = env.step(None)

    else:
        raise ValueError(f"Unknown method: {method}")

    sst_hours = env.t * env.dt_hours
    c_tot = max(env.initial_total_capacity, 1e-9)
    proactive_loss = env.cum_pruned_capacity / c_tot
    natural_loss   = env.cum_natural_failed_capacity / c_tot
    total_loss     = proactive_loss + natural_loss
    n_failed       = env.natural_failure_count
    return float(sst_hours), float(proactive_loss), float(natural_loss), float(total_loss), int(n_failed)


def run_all_experiments(
    *,
    scenarios: List[Scenario],
    na: int,
    dt_hours: float,
    max_hours: float,
    lambda_base: float,
    alpha: float,
    gamma_redistribute: float,
    gamma_prune: float,
    demand_threshold_frac: float,
    prune_util_min: float,
    dqn: DQNAgent,
    heuristic_threshold: float,
) -> pd.DataFrame:
    """
    Paired benchmark: every method is evaluated on the exact same scenarios.
    """
    methods = ["No Defense", "+25% Backup", "+50% Backup", "Heuristic", "DQN Strategic"]
    redundancy = {
        "No Defense": 1.0,
        "+25% Backup": 1.25,
        "+50% Backup": 1.50,
        "Heuristic": 1.0,
        "DQN Strategic": 1.0,
    }

    heuristic = HeuristicPruningAgent(util_threshold=heuristic_threshold)

    rows = []
    for method in methods:
        ssts:           List[float] = []
        pro_losses:     List[float] = []
        nat_losses:     List[float] = []
        tot_losses:     List[float] = []
        n_faileds:      List[int]   = []

        for scenario in scenarios:
            env = EdgeNetworkEnv(
                scenario=scenario,
                dt_hours=dt_hours,
                max_hours=max_hours,
                lambda_base=lambda_base,
                alpha=alpha,
                gamma_redistribute=gamma_redistribute,
                gamma_prune=gamma_prune,
                demand_threshold_frac=demand_threshold_frac,
                redundancy_factor=redundancy[method],
                prune_util_min=prune_util_min,
            )
            sst, pro, nat, tot, n_fail = run_episode(
                env=env,
                method=method,
                na=na,
                dqn=dqn if method == "DQN Strategic" else None,
                heuristic=heuristic if method == "Heuristic" else None,
            )
            ssts.append(sst)
            pro_losses.append(pro)
            nat_losses.append(nat)
            tot_losses.append(tot)
            n_faileds.append(n_fail)

        rows.append(
            {
                "Method": method,
                "Mean SST (h)": float(np.mean(ssts)),
                "SST Std (h)": float(np.std(ssts, ddof=1)) if len(ssts) > 1 else 0.0,
                "Proactive Loss (%)": 100.0 * float(np.mean(pro_losses)),
                "Natural Loss (%)": 100.0 * float(np.mean(nat_losses)),
                "Total Cap Loss (%)": 100.0 * float(np.mean(tot_losses)),
                "Avg Natural Failures": float(np.mean(n_faileds)),
            }
        )

    df = pd.DataFrame(rows)
    baseline = float(df.loc[df["Method"] == "No Defense", "Mean SST (h)"].iloc[0])
    df["Improvement vs No Defense (%)"] = 100.0 * (df["Mean SST (h)"] - baseline) / max(baseline, 1e-9)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42, help="Master seed for reproducibility.")
    ap.add_argument(
        "--weights",
        type=str,
        default=os.path.join("Version4_paper_repro", "dqn_weights.pt"),
        help="Path to save/load DQN weights.",
    )
    ap.add_argument("--force-train", action="store_true", help="Retrain even if weights exist.")

    # Environment defaults (paper spec)
    ap.add_argument("--n-nodes", type=int, default=100)
    ap.add_argument("--ba-m", type=int, default=3)
    ap.add_argument("--na", type=int, default=50)
    ap.add_argument("--dt-min", type=float, default=1.0)
    ap.add_argument("--max-hours", type=float, default=8.0)
    ap.add_argument("--lambda-base", type=float, default=0.02)
    ap.add_argument("--alpha", type=float, default=0.9)
    ap.add_argument("--gamma", type=float, default=0.85, help="Failure redistribution friction.")
    ap.add_argument(
        "--gamma-prune",
        type=float,
        default=0.5,
        help="Controlled redistribution factor for pruning (1.0 = no loss).",
    )
    ap.add_argument("--demand-threshold-frac", type=float, default=0.60)
    ap.add_argument("--prune-util-min", type=float, default=2.0)
    ap.add_argument("--capacity-low", type=float, default=50.0)
    ap.add_argument("--capacity-high", type=float, default=500.0)
    ap.add_argument("--init-util-low", type=float, default=0.40)
    ap.add_argument("--init-util-high", type=float, default=0.70)

    # DQN training defaults (paper spec)
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--eps-start", type=float, default=1.0)
    ap.add_argument("--eps-end", type=float, default=0.01)
    ap.add_argument("--eps-decay-steps", type=int, default=50_000)
    ap.add_argument("--buffer-size", type=int, default=500_000)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--target-update", type=int, default=200)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--discount", type=float, default=0.99)
    ap.add_argument("--device", type=str, default="cpu")

    # Evaluation defaults (paper spec)
    ap.add_argument("--eval-scenarios", type=int, default=20)
    ap.add_argument("--heuristic-threshold", type=float, default=0.95)
    ap.add_argument("--use-gnn", action="store_true", default=True, help="Enable GNN architecture (otherwise use FC).")
    ap.add_argument("--no-gnn", action="store_false", dest="use_gnn", help="Disable GNN architecture (use FC).")
    ap.add_argument("--topo-type", type=str, default="ba", choices=["ba", "ws", "er"], help="Topology type for evaluation.")

    args = ap.parse_args()

    set_global_seeds(args.seed)

    dt_hours = float(args.dt_min) / 60.0
    max_steps = int(round(float(args.max_hours) / dt_hours))

    print(
        "[config] "
        f"N={args.n_nodes} BA(m)={args.ba_m} Na={args.na} dt={args.dt_min}min max_hours={args.max_hours} "
        f"lambda_base={args.lambda_base} alpha={args.alpha} gamma={args.gamma} demand_thresh={args.demand_threshold_frac}"
    )

    topo_args = {"type": args.topo_type, "ba_m": args.ba_m}
    # 1) Pre-generate paired evaluation scenarios.
    scenarios = generate_paired_scenarios(
        seed=args.seed,
        num_scenarios=args.eval_scenarios,
        n_nodes=args.n_nodes,
        topo_args=topo_args,
        max_steps=max_steps,
        capacity_range=(args.capacity_low, args.capacity_high),
        init_util_range=(args.init_util_low, args.init_util_high),
    )

    # 2) Train (or load) the DQN on fresh randomized episodes.
    state_dim = 3 * int(args.na)
    action_dim = int(args.na) + 1  # 0..49 prune slots, 50 no-op
    # Use the first eval scenario's topology for GNN adj_norm construction.
    # All scenarios share the same n_nodes; adj_list is topology-specific but
    # representative for building the graph message-passing structure.
    _rep_adj = scenarios[0].adj if args.use_gnn else None
    dqn = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=args.lr,
        discount=args.discount,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        target_update=args.target_update,
        device=args.device,
        use_gnn=args.use_gnn,
        adj_list=_rep_adj,
        n_nodes=int(args.n_nodes),
        na=int(args.na),
    )

    if (not args.force_train) and os.path.exists(args.weights):
        print(f"[dqn] Loading weights: {args.weights}")
        dqn.load(args.weights)
    else:
        print(f"[dqn] Training for {args.episodes} episodes ...")
        train_dqn(
            agent=dqn,
            n_nodes=args.n_nodes,
            ba_m=args.ba_m,
            na=args.na,
            episodes=args.episodes,
            seed=args.seed + 12345,
            dt_hours=dt_hours,
            max_hours=args.max_hours,
            lambda_base=args.lambda_base,
            alpha=args.alpha,
            gamma_redistribute=args.gamma,
            gamma_prune=args.gamma_prune,
            demand_threshold_frac=args.demand_threshold_frac,
            prune_util_min=args.prune_util_min,
            capacity_range=(args.capacity_low, args.capacity_high),
            init_util_range=(args.init_util_low, args.init_util_high),
            eps_start=args.eps_start,
            eps_end=args.eps_end,
            eps_decay_steps=args.eps_decay_steps,
        )
        dqn.save(args.weights)
        print(f"[dqn] Saved weights: {args.weights}")

    # 3) Paired evaluation on the 20 fixed scenarios.
    df = run_all_experiments(
        scenarios=scenarios,
        na=args.na,
        dt_hours=dt_hours,
        max_hours=args.max_hours,
        lambda_base=args.lambda_base,
        alpha=args.alpha,
        gamma_redistribute=args.gamma,
        gamma_prune=args.gamma_prune,
        demand_threshold_frac=args.demand_threshold_frac,
        prune_util_min=args.prune_util_min,
        dqn=dqn,
        heuristic_threshold=args.heuristic_threshold,
    )

    pd.set_option("display.max_columns", None)
    print()
    print(df.to_string(index=False, justify="left", float_format=lambda x: f"{x:0.3f}"))


if __name__ == "__main__":
    main()
