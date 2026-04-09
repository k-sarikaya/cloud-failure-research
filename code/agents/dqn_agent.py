"""
Original DQN (Binary Pruning) Agent from v1 Manuscript.
Adapted to run in EdgeEnvV2 (Continuous Action) via a Binary-to-Continuous Wrapper.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

# GNN Constants from original repro script
GNN_NODE_FEAT_DIM = 3   # [operational, util_norm, cap_rel]
GNN_HIDDEN_DIM    = 64

class QNet(nn.Module):
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
    def __init__(self, node_feat_dim: int, hidden_dim: int, na: int) -> None:
        super().__init__()
        self.na = na
        self.sage1 = nn.Linear(node_feat_dim * 2, hidden_dim)
        self.sage2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.global_fc = nn.Linear(hidden_dim, hidden_dim)
        self.q_head = nn.Linear(hidden_dim * 2, 1)
        self.relu = nn.ReLU()

    def build_adj_norm(adj_list: List[List[int]], n: int, device: torch.device) -> torch.Tensor:
        A = torch.zeros(n, n, device=device)
        for i, nbrs in enumerate(adj_list):
            A[i, i] = 1.0
            for j in nbrs:
                A[i, j] = 1.0
        deg = A.sum(dim=1, keepdim=True).clamp(min=1.0)
        return A / deg

    def forward(self, node_feats: torch.Tensor, adj_norm: torch.Tensor, window_idx: torch.Tensor) -> torch.Tensor:
        # Simplified for inference
        neigh1 = torch.mm(adj_norm, node_feats)
        h1 = self.relu(self.sage1(torch.cat([node_feats, neigh1], dim=-1)))
        
        neigh2 = torch.mm(adj_norm, h1)
        h2 = self.relu(self.sage2(torch.cat([h1, neigh2], dim=-1)))
        
        g = self.relu(self.global_fc(h2.mean(dim=0)))
        h_win = h2[window_idx]
        g_exp = g.unsqueeze(0).expand(self.na, -1)
        
        q_prune = self.q_head(torch.cat([h_win, g_exp], dim=-1)).squeeze(-1)
        q_noop = self.q_head(torch.cat([g, g], dim=-1).unsqueeze(0)).squeeze()
        if q_noop.dim() > 0: q_noop = q_noop.mean()
        return torch.cat([q_prune, q_noop.unsqueeze(0)], dim=0)

class DQNAgent:
    def __init__(self, state_dim: int, action_dim: int, device: str = "cpu", na: int = 50):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device(device)
        self.na = na
        self.q = QNet(state_dim, action_dim).to(self.device)

    def select_action(self, state: np.ndarray, mask: np.ndarray, epsilon: float = 0.0) -> int:
        if np.random.random() < epsilon:
            valid = np.where(mask)[0]
            return int(np.random.choice(valid))
        
        with torch.no_grad():
            s = torch.tensor(state, device=self.device).float()
            qvals = self.q(s.unsqueeze(0)).squeeze(0).cpu().numpy()
        
        qvals[~mask] = -1e9
        return int(np.argmax(qvals))

class DQNBinaryWrapper:
    """
    Adapts the original 50-node active window DQN to the 100-node EdgeEnvV2.
    Converts binary 'prune node i' action to a continuous [0, 1] throttle vector.
    """
    def __init__(self, agent: DQNAgent, n_total: int = 100, na: int = 50):
        self.agent = agent
        self.n_total = n_total
        self.na = na

    def act(self, state_v2: np.ndarray, adj: np.ndarray) -> np.ndarray:
        # 1) Get utilization to find the active window (top na nodes)
        # state_v2: (n, feat_dim). feat_dim=4: [op, util, cap, risk]
        # util is state_v2[:, 1]
        util = state_v2[:, 1]
        op = state_v2[:, 0]
        
        # Select top Na operational nodes by utilization
        op_indices = np.where(op > 0.5)[0]
        if len(op_indices) == 0:
            return np.zeros(self.n_total, dtype=np.float32)
            
        sorted_by_util = op_indices[np.argsort(util[op_indices])][::-1]
        window_nodes = sorted_by_util[:self.na]
        
        # 2) Construct 150-dim state for DQN
        # DQN state: [op_0..op_na, util_0..util_na, cap_0..cap_na]
        window_op = op[window_nodes]
        window_util = util[window_nodes]
        window_cap = state_v2[window_nodes, 2]
        
        # Padding if window_nodes < na
        if len(window_nodes) < self.na:
            pad_len = self.na - len(window_nodes)
            window_op = np.concatenate([window_op, np.zeros(pad_len)])
            window_util = np.concatenate([window_util, np.zeros(pad_len)])
            window_cap = np.concatenate([window_cap, np.zeros(pad_len)])
            
        dqn_state = np.concatenate([window_op, window_util, window_cap])
        
        # 3) Build mask for DQN (only real operational nodes in window can be pruned)
        mask = np.zeros(self.na + 1, dtype=bool)
        mask[:len(window_nodes)] = (window_op[:len(window_nodes)] > 0.5)
        mask[self.na] = True # No-op is always valid
        
        # 4) Get DQN action
        action_idx = self.agent.select_action(dqn_state, mask, epsilon=0.0)
        
        # 5) Map back to full throttle vector
        throttle_vec = np.zeros(self.n_total, dtype=np.float32)
        if action_idx < len(window_nodes):
            target_node = window_nodes[action_idx]
            throttle_vec[target_node] = 1.0 # Binary pruning = 100% throttle
            
        return throttle_vec
