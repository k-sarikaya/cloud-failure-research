"""
GAT Encoder – 2-layer Graph Attention Network (pure PyTorch, no PyG required)
==============================================================================
Converts per-node features into topology-aware node embeddings using
multi-head attention over the adjacency structure.

Key advantage over GraphSAGE:
  - Attention weights are LEARNED, not uniform
  - Critical neighbors (high utilization) get higher attention
  - Enables the agent to "see" which neighbors are about to fail
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    """Single Graph Attention Layer with multi-head attention."""

    def __init__(self, in_dim: int, out_dim: int, num_heads: int = 4,
                 dropout: float = 0.1, concat: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.concat = concat

        # Linear transform for each head
        self.W = nn.Linear(in_dim, out_dim * num_heads, bias=False)
        # Attention coefficients: a^T [Wh_i || Wh_j]
        self.a_src = nn.Parameter(torch.zeros(num_heads, out_dim))
        self.a_dst = nn.Parameter(torch.zeros(num_heads, out_dim))
        nn.init.xavier_uniform_(self.a_src.unsqueeze(0))
        nn.init.xavier_uniform_(self.a_dst.unsqueeze(0))

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, in_dim) node features
            adj: (B, N, N) adjacency matrix (1 = connected, 0 = not)
        Returns:
            (B, N, out_dim * num_heads) if concat, else (B, N, out_dim)
        """
        B, N, _ = x.shape
        H = self.num_heads
        D = self.out_dim

        # Linear projection: (B, N, H*D)
        h = self.W(x).view(B, N, H, D)  # (B, N, H, D)

        # Attention scores
        # e_ij = LeakyReLU(a_src^T h_i + a_dst^T h_j)
        attn_src = (h * self.a_src).sum(dim=-1)  # (B, N, H)
        attn_dst = (h * self.a_dst).sum(dim=-1)  # (B, N, H)

        # Broadcast: attn_src[i] + attn_dst[j] for all (i, j) pairs
        attn = attn_src.unsqueeze(2) + attn_dst.unsqueeze(1)  # (B, N, N, H)
        attn = self.leaky_relu(attn)

        # Mask: set non-adjacent pairs to -inf
        mask = (adj.unsqueeze(-1) == 0)  # (B, N, N, 1)
        attn = attn.masked_fill(mask, float('-inf'))

        # Softmax over neighbors
        attn = F.softmax(attn, dim=2)
        attn = torch.nan_to_num(attn, nan=0.0)  # handle isolated nodes
        attn = self.dropout(attn)

        # Aggregate: weighted sum of neighbor features
        # attn: (B, N, N, H), h: (B, N, H, D)
        # out[i] = sum_j attn[i,j] * h[j]
        out = torch.einsum('bnmh,bmhd->bnhd', attn, h)  # (B, N, H, D)

        if self.concat:
            out = out.reshape(B, N, H * D)
        else:
            out = out.mean(dim=2)  # average heads

        return out


class GATEncoder(nn.Module):
    """
    2-layer GAT encoder producing per-node and graph-level embeddings.

    Architecture:
        Input: (B, N, node_feat_dim)
        → GATLayer 1 (concat heads) → ELU → Dropout
        → GATLayer 2 (average heads) → ELU
        → node_embeddings: (B, N, hidden_dim)
        → graph_embedding: mean_pool → Linear → ELU → (B, hidden_dim)
    """

    def __init__(self, node_feat_dim: int = 4, hidden_dim: int = 64,
                 num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Layer 1: concat heads → output dim = hidden_dim * num_heads
        self.gat1 = GATLayer(node_feat_dim, hidden_dim, num_heads,
                             dropout=dropout, concat=True)
        self.elu1 = nn.ELU()
        self.drop1 = nn.Dropout(dropout)

        # Layer 2: average heads → output dim = hidden_dim
        self.gat2 = GATLayer(hidden_dim * num_heads, hidden_dim, num_heads,
                             dropout=dropout, concat=False)
        self.elu2 = nn.ELU()

        # Graph-level embedding
        self.graph_fc = nn.Linear(hidden_dim, hidden_dim)
        self.graph_elu = nn.ELU()

    def forward(self, x: torch.Tensor, adj: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, N, node_feat_dim) node features
            adj: (B, N, N) adjacency matrix

        Returns:
            node_embeds: (B, N, hidden_dim)
            graph_embed: (B, hidden_dim)
        """
        h = self.drop1(self.elu1(self.gat1(x, adj)))
        h = self.elu2(self.gat2(h, adj))

        # Graph embedding = mean pool → FC → ELU
        g = h.mean(dim=1)  # (B, hidden_dim)
        g = self.graph_elu(self.graph_fc(g))

        return h, g
