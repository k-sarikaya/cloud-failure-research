"""
Training script for Binary DQN on EdgeEnvV2.
Used as an ablation baseline to compare discrete vs continuous actions.
"""

import sys
import os
import torch
import numpy as np
import argparse
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import EnvConfig
from env.edge_env_v2 import EdgeEnvV2, Scenario, make_scenario
from agents.dqn_agent import DQNAgent, DQNBinaryWrapper, GNN_NODE_FEAT_DIM

class BinaryReplayBuffer:
    def __init__(self, capacity, state_dim):
        self.capacity = capacity
        self.s = np.zeros((capacity, *state_dim), dtype=np.float32)
        self.a = np.zeros(capacity, dtype=np.int64)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.ns = np.zeros((capacity, *state_dim), dtype=np.float32)
        self.d = np.zeros(capacity, dtype=np.float32)
        self.m = np.zeros((capacity, 51), dtype=bool) # (na+1)
        self.nm = np.zeros((capacity, 51), dtype=bool)
        self.ptr, self.size = 0, 0

    def add(self, s, a, r, ns, d, m, nm):
        self.s[self.ptr] = s
        self.a[self.ptr] = a
        self.r[self.ptr] = r
        self.ns[self.ptr] = ns
        self.d[self.ptr] = d
        self.m[self.ptr] = m
        self.nm[self.ptr] = nm
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, batch_size)
        return (torch.FloatTensor(self.s[idx]), 
                torch.LongTensor(self.a[idx]), 
                torch.FloatTensor(self.r[idx]), 
                torch.FloatTensor(self.ns[idx]), 
                torch.FloatTensor(self.d[idx]),
                torch.BoolTensor(self.m[idx]),
                torch.BoolTensor(self.nm[idx]))

def train_dqn(episodes=300):
    cfg = EnvConfig()
    rng = np.random.default_rng(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # DQN params (matching v1)
    na = 50
    state_dim = (cfg.n_nodes, 4)
    dqn_state_dim = 150 # na * 3 (op, util, cap)
    action_dim = na + 1
    
    agent = DQNAgent(state_dim=dqn_state_dim, action_dim=action_dim, device=device, na=na)
    buffer = BinaryReplayBuffer(50000, (dqn_state_dim,))
    optimizer = torch.optim.Adam(agent.q.parameters(), lr=1e-4)
    
    import time
    print(f"[{time.strftime('%H:%M:%S')}] Starting DQN Training (Binary Pruning) for {episodes} eps...")
    
    best_sst = 0.0
    for ep in range(1, episodes + 1):
        # Generate scenario for this episode
        ep_rng = np.random.default_rng(ep + 1000)
        scenario = make_scenario(
            n_nodes=cfg.n_nodes,
            ba_m=cfg.ba_m,
            rng=ep_rng
        )
        env = EdgeEnvV2(
            scenario=scenario,
            dt_hours=cfg.dt_hours,
            max_hours=cfg.max_hours,
            lambda_base=cfg.lambda_base,
            alpha=cfg.alpha,
            demand_threshold_frac=cfg.demand_threshold_frac,
            drop_fraction=cfg.drop_fraction
        )
        state = env.reset()
        adj = env.get_adjacency_matrix()
        
        # Helper to get DQN-specific state and mask
        def get_dqn_context(s_v2):
            util = s_v2[:, 1]
            op = s_v2[:, 0]
            op_idx = np.where(op > 0.5)[0]
            if len(op_idx) == 0: return np.zeros(150), np.array([False]*50 + [True])
            
            # Select top na
            sorted_idx = op_idx[np.argsort(util[op_idx])][::-1]
            win_nodes = sorted_idx[:na]
            
            # Construct 150-dim
            w_op = np.pad(op[win_nodes], (0, na - len(win_nodes)))
            w_ut = np.pad(util[win_nodes], (0, na - len(win_nodes)))
            w_cp = np.pad(s_v2[win_nodes, 2], (0, na - len(win_nodes)))
            s_dqn = np.concatenate([w_op, w_ut, w_cp])
            
            # Mask
            mask = np.zeros(na+1, dtype=bool)
            mask[:len(win_nodes)] = (w_op[:len(win_nodes)] > 0.5)
            mask[na] = True
            return s_dqn, mask, win_nodes

        ep_reward = 0
        eps = max(0.01, 1.0 - ep/200)
        
        s_dqn, mask, win_nodes = get_dqn_context(state)
        
        for t in range(cfg.max_steps):
            action_idx = agent.select_action(s_dqn, mask, epsilon=eps)
            
            # Convert to throttle vector
            throttle = np.zeros(cfg.n_nodes, dtype=np.float32)
            if action_idx < len(win_nodes):
                throttle[win_nodes[action_idx]] = 1.0 # Hard prune
                
            next_state, reward, done, info = env.step(throttle)
            ns_dqn, n_mask, _ = get_dqn_context(next_state)
            
            buffer.add(s_dqn, action_idx, reward, ns_dqn, done, mask, n_mask)
            
            state = next_state
            s_dqn, mask = ns_dqn, n_mask
            ep_reward += reward
            
            # Update DQN
            if buffer.size > 1000:
                bs, ba, br, bns, bd, bm, bnm = buffer.sample(64)
                bs, ba, br, bns, bd, bm, bnm = bs.to(device), ba.to(device), br.to(device), bns.to(device), bd.to(device), bm.to(device), bnm.to(device)
                
                q_vals = agent.q(bs)
                q_sa = q_vals.gather(1, ba.unsqueeze(1)).squeeze(1)
                
                with torch.no_grad():
                    # Double DQN
                    next_q_online = agent.q(bns)
                    next_q_online.masked_fill_(~bnm, -1e9)
                    next_a = next_q_online.argmax(dim=1)
                    # We use same Q for target for simplicity in v1 alignment
                    next_q = agent.q(bns).gather(1, next_a.unsqueeze(1)).squeeze(1)
                    target = br + (1.0 - bd) * 0.99 * next_q
                
                loss = torch.nn.functional.mse_loss(q_sa, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            if done: break
            
        sst = info['hours']
        if sst > best_sst:
            best_sst = sst
            torch.save(agent.q.state_dict(), "weights/dqn_binary.pt")
            
        if ep % 25 == 0:
            print(f"Ep {ep:3d} | SST: {sst:5.2f}h | Best: {best_sst:5.2f}h | Reward: {ep_reward:7.2f}")

    print(f"Training complete. Best SST: {best_sst:.2f}h Weights: weights/dqn_binary.pt")

if __name__ == "__main__":
    os.makedirs("weights", exist_ok=True)
    train_dqn(episodes=300)
