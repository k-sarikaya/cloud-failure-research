"""
EdgeEnvV2 - Continuous-Action Cascading Failure Environment
===========================================================

Key difference from v1 (EdgeNetworkEnv):
  - Action space is continuous: throttle[i] in [0, 1] for each node.
    0.0 = no intervention, 1.0 = full shutdown (hard prune),
    0.3 = shed 30% of the node's current load.
  - The action controls how much load is shed from the node.
  - drop_fraction controls what fraction of the shed load is dropped
    instead of redistributed to operational neighbors.
  - This enables graceful degradation by reducing the centred hazard
    lambda_i = lambda_base * exp(alpha * (rho_i - 1)) without forcing
    immediate hard shutdown.

Per-step dynamics (dt = 1 minute):
  1) Apply throttle actions: for each node i with throttle[i] > threshold,
     shed throttle[i] * L_i load from the node.
  2) Split the shed load into redistributed and dropped portions according
     to drop_fraction.
  3) Sample natural failures using paired randomness.
  4) For each naturally failed node, redistribute the surviving share of
     its load and drop the remainder according to drop_fraction.
  5) Check termination (remaining capacity < demand threshold).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import networkx as nx
import numpy as np


# ---------------------------------------------------------------------------
# Topology generators
# ---------------------------------------------------------------------------

def generate_ba_adj(n: int, m: int, rng: np.random.Generator) -> List[List[int]]:
    G = nx.barabasi_albert_graph(n, m, seed=int(rng.integers(0, 100_000)))
    return [list(G.neighbors(i)) for i in range(n)]


def generate_5g_mec_adj(n: int, rng: np.random.Generator) -> List[List[int]]:
    """Generate a simple hierarchical 5G MEC-inspired topology.

    Structure (same spirit as legacy generator):
      - Core layer: fully connected mesh (data centers)
      - Edge layer: each connects to 2-3 core nodes
      - Access layer: each connects to 1-2 edge nodes

    This generator is deterministic given `rng`.
    """
    if n < 10:
        # Too small for a 3-layer hierarchy; fall back to ER.
        return generate_er_adj(n, rng=rng)

    n_core = max(3, int(round(n * 0.05)))
    n_edge = max(5, int(round(n * 0.15)))
    if n_core + n_edge >= n:
        n_edge = max(1, n - n_core - 1)
    n_access = n - n_core - n_edge
    assert n_access >= 1

    adj: List[List[int]] = [[] for _ in range(n)]

    def add_edge(i: int, j: int) -> None:
        if j not in adj[i]:
            adj[i].append(j)
        if i not in adj[j]:
            adj[j].append(i)

    # Core mesh.
    for i in range(n_core):
        for j in range(i + 1, n_core):
            add_edge(i, j)

    edge_start = n_core
    edge_nodes = np.arange(edge_start, edge_start + n_edge)

    # Edge-to-core connections.
    for i in edge_nodes:
        k = int(rng.integers(2, 4))
        targets = rng.choice(n_core, size=min(k, n_core), replace=False)
        for t in targets:
            add_edge(int(i), int(t))

    # Access-to-edge connections.
    access_start = n_core + n_edge
    for i in range(access_start, n):
        k = int(rng.integers(1, 3))
        targets = rng.choice(edge_nodes, size=min(k, n_edge), replace=False)
        for t in targets:
            add_edge(int(i), int(t))

    # Stabilize ordering for reproducibility of adjacency list serialization.
    for i in range(n):
        adj[i].sort()
    return adj


def generate_ws_adj(n: int, k: int = 4, p: float = 0.2,
                    rng: Optional[np.random.Generator] = None) -> List[List[int]]:
    seed = int(rng.integers(0, 100_000)) if rng else None
    G = nx.connected_watts_strogatz_graph(n, k, p, tries=100, seed=seed)
    return [list(G.neighbors(i)) for i in range(n)]


def generate_er_adj(n: int, p: float = 0.06,
                    rng: Optional[np.random.Generator] = None) -> List[List[int]]:
    seed = int(rng.integers(0, 100_000)) if rng else None
    G = nx.erdos_renyi_graph(n, p, seed=seed)
    while not nx.is_connected(G):
        G = nx.erdos_renyi_graph(n, p, seed=int(rng.integers(0, 100_000)))
    return [list(G.neighbors(i)) for i in range(n)]


# ---------------------------------------------------------------------------
# Scenario (paired randomness for fair benchmarking)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Scenario:
    adj: List[List[int]]       # adjacency list (length N)
    capacities: np.ndarray     # shape (N,), float32
    loads: np.ndarray          # shape (N,), float32
    fail_uniforms: np.ndarray  # shape (max_steps, N), float32


def make_scenario(
    n_nodes: int = 100,
    topo_type: str = "ba",
    ba_m: int = 3,
    max_steps: int = 480,
    capacity_range: Tuple[float, float] = (50, 500),
    init_util_range: Tuple[float, float] = (0.75, 0.90),
    lambda_base: float = 0.12,
    alpha: float = 1.2,
    rng: Optional[np.random.Generator] = None,
) -> Scenario:
    topo_key = (topo_type or "ba").strip().lower().replace("-", "_")
    if topo_key == "ba":
        adj = generate_ba_adj(n_nodes, ba_m, rng)
    elif topo_key == "ws":
        adj = generate_ws_adj(n_nodes, rng=rng)
    elif topo_key == "er":
        adj = generate_er_adj(n_nodes, rng=rng)
    elif topo_key in {"5g_mec", "mec"}:
        adj = generate_5g_mec_adj(n_nodes, rng)
    else:
        adj = generate_ba_adj(n_nodes, ba_m, rng)

    c_lo, c_hi = capacity_range
    capacities = rng.uniform(c_lo, c_hi, size=n_nodes).astype(np.float32)

    u_lo, u_hi = init_util_range
    init_util = rng.uniform(u_lo, u_hi, size=n_nodes).astype(np.float32)
    loads = (init_util * capacities).astype(np.float32)

    fail_uniforms = rng.random(size=(max_steps, n_nodes), dtype=np.float32)
    return Scenario(adj=adj, capacities=capacities, loads=loads,
                    fail_uniforms=fail_uniforms)


def generate_paired_scenarios(
    *,
    seed: int,
    num_scenarios: int,
    n_nodes: int,
    topo_type: str = "ba",
    ba_m: int = 3,
    max_steps: int,
    capacity_range: Tuple[float, float],
    init_util_range: Tuple[float, float],
) -> List[Scenario]:
    rng = np.random.default_rng(seed)
    return [
        make_scenario(
            n_nodes=n_nodes, topo_type=topo_type, ba_m=ba_m,
            max_steps=max_steps, capacity_range=capacity_range,
            init_util_range=init_util_range, rng=rng,
        )
        for _ in range(num_scenarios)
    ]

def load_scenario_bank_npz(path: str | Path) -> List[Scenario]:
    """Load a deterministic scenario bank exported by scripts/export_scenario_bank.py.

    The bank stores graph edges and per-scenario arrays (capacities, loads, paired RNG uniforms)
    so evaluation can be replayed without re-sampling.
    """

    p = Path(path)
    data = np.load(p)

    edges = data["edges"].astype(np.int32, copy=False)
    edge_offsets = data["edge_offsets"].astype(np.int64, copy=False)
    capacities = data["capacities"].astype(np.float32, copy=False)
    loads = data["loads"].astype(np.float32, copy=False)
    fail_uniforms = data["fail_uniforms"].astype(np.float32, copy=False)

    n_scenarios = int(capacities.shape[0])
    scenarios: List[Scenario] = []
    for i in range(n_scenarios):
        n_nodes = int(capacities.shape[1])
        adj: List[List[int]] = [[] for _ in range(n_nodes)]
        start = int(edge_offsets[i])
        end = int(edge_offsets[i + 1])
        for u, v in edges[start:end]:
            uu = int(u)
            vv = int(v)
            adj[uu].append(vv)
            adj[vv].append(uu)

        scenarios.append(
            Scenario(
                adj=adj,
                capacities=capacities[i],
                loads=loads[i],
                fail_uniforms=fail_uniforms[i],
            )
        )

    return scenarios



# ---------------------------------------------------------------------------
# EdgeEnvV2: Continuous-Action Environment
# ---------------------------------------------------------------------------

class EdgeEnvV2:
    """
    Cascading-failure simulation with CONTINUOUS throttle actions.

    Action: np.ndarray of shape (n_nodes,), each element in [0, 1].
      - 0.0 = no intervention on this node
      - 0.3 = shed 30% of this node's current load to neighbors
      - 1.0 = full shutdown (equivalent to old hard prune)

    State: np.ndarray of shape (n_nodes, 4):
      - [:, 0] = operational (0 or 1)
      - [:, 1] = utilization L_i/C_i (normalized, clipped to [0, 3])
      - [:, 2] = relative capacity C_i / C_total
      - [:, 3] = neighbor risk: mean utilization of neighbors
    """

    def __init__(
        self,
        *,
        scenario: Scenario,
        dt_hours: float,
        max_hours: float,
        lambda_base: float = 0.12,
        alpha: float = 1.2,
        demand_threshold_frac: float = 0.60,
        throttle_min: float = 0.05,
        throttle_cost_coeff: float = 0.02,
        redundancy_factor: float = 1.0,
        drop_fraction: float = 0.5,
    ) -> None:
        self.adj = scenario.adj
        self.n = len(self.adj)

        self.dt_hours = float(dt_hours)
        self.max_hours = float(max_hours)
        self.max_steps = int(round(self.max_hours / self.dt_hours))

        self.lambda_base = float(lambda_base)
        self.alpha = float(alpha)
        self.demand_threshold_frac = float(demand_threshold_frac)
        self.throttle_min = float(throttle_min)
        self.throttle_cost_coeff = float(throttle_cost_coeff)
        self.drop_fraction = float(drop_fraction)
        self.redundancy_factor = float(redundancy_factor)

        self._cap0 = (scenario.capacities * self.redundancy_factor).astype(np.float32)
        self._load0 = scenario.loads.astype(np.float32)
        self._fail_uniforms = scenario.fail_uniforms

        # Mutable state
        self.t: int = 0
        self.operational: np.ndarray = np.ones(self.n, dtype=np.bool_)
        self.capacities: np.ndarray = self._cap0.copy()
        self.loads: np.ndarray = self._load0.copy()
        self.initial_total_capacity: float = 0.0
        self.base_nominal_capacity: float = 0.0
        self.demand_threshold: float = 0.0
        self.cum_throttled_load: float = 0.0
        self.cum_dropped_load: float = 0.0  # Total load explicitly shedded
        self.cum_proactive_cap_loss: float = 0.0  # Sum of C_i * throttle_i
        self.cum_natural_failed_capacity: float = 0.0
        self.cum_total_load: float = 0.0  # Cumulative total load across all steps
        self.natural_failure_count: int = 0
        self.done: bool = False

        self.reset()

    def set_scenario(self, scenario: Scenario) -> None:
        """Hot-swap the scenario without re-creating the environment."""
        self.adj = scenario.adj
        self.n = len(self.adj)
        self._cap0 = (scenario.capacities * self.redundancy_factor).astype(np.float32)
        self._load0 = scenario.loads.astype(np.float32)
        self._fail_uniforms = scenario.fail_uniforms
        self.reset()

    def reset(self) -> np.ndarray:
        """Reset environment and return initial state."""
        self.t = 0
        self.operational = np.ones(self.n, dtype=np.bool_)
        self.capacities = self._cap0.copy()
        self.loads = self._load0.copy()
        self.initial_total_capacity = float(self.capacities.sum())
        self.base_nominal_capacity = self.initial_total_capacity / self.redundancy_factor
        self.demand_threshold = self.demand_threshold_frac * self.base_nominal_capacity
        self.cum_throttled_load = 0.0
        self.cum_dropped_load = 0.0
        self.cum_proactive_cap_loss = 0.0
        self.cum_natural_failed_capacity = 0.0
        self.cum_total_load = 0.0
        self.natural_failure_count = 0
        self.done = False
        return self.get_state()

    # -- Observation ----------------------------------------------------------

    def utilization(self) -> np.ndarray:
        """U_i = L_i / C_i for operational nodes; 0 for failed."""
        util = np.zeros(self.n, dtype=np.float32)
        mask = self.operational & (self.capacities > 0)
        util[mask] = self.loads[mask] / self.capacities[mask]
        return util

    def get_state(self) -> np.ndarray:
        """
        Returns node-level state matrix of shape (n_nodes, 4).
        Features: [operational, utilization_norm, cap_ratio, neighbor_risk]
        """
        util = self.utilization()
        util_norm = np.clip(util, 0.0, 3.0) / 3.0

        op = self.operational.astype(np.float32)
        cap_ratio = self.capacities / max(self.initial_total_capacity, 1e-9)

        # Neighbor risk: mean utilization of neighbors
        neigh_risk = np.zeros(self.n, dtype=np.float32)
        for i in range(self.n):
            if self.operational[i] and len(self.adj[i]) > 0:
                nbr_utils = [util[j] for j in self.adj[i] if self.operational[j]]
                if nbr_utils:
                    neigh_risk[i] = float(np.mean(nbr_utils))

        state = np.stack([op, util_norm, cap_ratio, neigh_risk], axis=-1)
        return state.astype(np.float32)

    def get_adjacency_matrix(self) -> np.ndarray:
        """Return (n, n) adjacency matrix with self-loops, row-normalized."""
        A = np.zeros((self.n, self.n), dtype=np.float32)
        for i, nbrs in enumerate(self.adj):
            A[i, i] = 1.0
            for j in nbrs:
                A[i, j] = 1.0
        deg = A.sum(axis=1, keepdims=True)
        deg = np.maximum(deg, 1.0)
        return A / deg

    # -- Step -----------------------------------------------------------------

    def step(self, throttle: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Execute one time step with continuous throttle actions.

        Args:
            throttle: np.ndarray of shape (n_nodes,), each in [0, 1].

        Returns:
            next_state, reward, done, info
        """
        if self.done:
            raise RuntimeError("step() called after termination")

        throttle = np.clip(throttle, 0.0, 1.0).astype(np.float32)
        total_load_before = float(self.loads.sum())
        self.cum_total_load += total_load_before

        # 1) Apply throttle actions (Graceful Degradation)
        total_throttled = 0.0
        for i in range(self.n):
            if not self.operational[i]:
                continue
            t_i = float(throttle[i])
            if t_i < self.throttle_min:
                continue

            if t_i > 0.95:
                # Full shutdown (hard prune equivalent)
                cap_i = float(self.capacities[i])
                total_throttled += float(self.loads[i])
                self.cum_proactive_cap_loss += cap_i
                self._shutdown_and_redistribute(i)
            else:
                # Graceful load shedding
                t_i = float(throttle[i])
                shed_amount = float(self.loads[i]) * t_i
                self.loads[i] -= shed_amount
                total_throttled += shed_amount
                self.cum_throttled_load += shed_amount

                # 50% of shed load is redistributed, 50% is dropped
                redist_amount = shed_amount * (1.0 - self.drop_fraction)
                dropped_amount = shed_amount * self.drop_fraction
                
                self.cum_dropped_load += dropped_amount
                if redist_amount > 0:
                    self._redistribute_load(i, redist_amount)

        # 2) Sample natural failures (paired randomness)
        failed_nodes = self._sample_failures()
        for nid in failed_nodes:
            self.cum_natural_failed_capacity += float(self._cap0[nid])
            self.natural_failure_count += 1
            self._shutdown_and_redistribute(nid)

        num_failures = len(failed_nodes)

        # 3) Termination check
        self.t += 1
        remaining_capacity = float(self.capacities[self.operational].sum())
        terminated_by_capacity = remaining_capacity < self.demand_threshold
        terminated_by_time = self.t >= self.max_steps
        self.done = terminated_by_capacity or terminated_by_time

        # 4) Reward computation
        reward = self._compute_reward(
            num_failures=num_failures,
            total_throttled=total_throttled,
            total_load_before=total_load_before,
            terminated_by_capacity=terminated_by_capacity,
        )

        info = {
            "t": self.t,
            "hours": self.t * self.dt_hours,
            "num_failures": num_failures,
            "total_throttled": total_throttled,
            "throttled_load_ratio": self.cum_throttled_load / max(self.cum_total_load, 1e-9),
            "dropped_load_ratio": self.cum_dropped_load / max(self.cum_total_load, 1e-9),
            "proactive_cap_loss_ratio": self.cum_proactive_cap_loss / max(self.initial_total_capacity, 1e-9),
            "natural_cap_loss_ratio": self.cum_natural_failed_capacity / max(self.initial_total_capacity, 1e-9),
            "total_cap_loss_ratio": (self.cum_proactive_cap_loss + self.cum_natural_failed_capacity) / max(self.initial_total_capacity, 1e-9),
            "natural_failure_count": self.natural_failure_count,
            "remaining_capacity": remaining_capacity,
            "remaining_capacity_ratio": remaining_capacity / max(self.initial_total_capacity, 1e-9),
            "terminated_by_capacity": terminated_by_capacity,
            "terminated_by_time": terminated_by_time,
            "n_operational": int(self.operational.sum()),
        }

        return self.get_state(), float(reward), self.done, info

    # -- Internals ------------------------------------------------------------

    def _compute_reward(
        self,
        *,
        num_failures: int,
        total_throttled: float,
        total_load_before: float,
        terminated_by_capacity: bool,
    ) -> float:
        """
        Reward = survival bonus - throttle cost - failure penalty - terminal penalty.
        """
        # Survival bonus (every step alive is good)
        r = 1.0

        # Throttle cost (small penalty for interventions to avoid over-throttling)
        if total_load_before > 0:
            throttle_ratio = total_throttled / total_load_before
            r -= self.throttle_cost_coeff * throttle_ratio

        # Natural failure penalty (big: natural failures are BAD)
        if num_failures == 0:
            r += 5.0  # bonus for zero failures this step
        else:
            r -= 10.0 * float(num_failures)

        # Terminal penalty
        if self.done and terminated_by_capacity:
            r -= 100.0

        return r

    def _get_failure_probs(self) -> np.ndarray:
        """
        Calculates the probability of failure for each operational node
        within the current time step (dt_hours).
        Failure rate follows the centred hazard lambda_base * exp(alpha * (rho - 1)),
        where lambda_base is anchored at unit utilisation (rho = 1).
        """
        util = self.utilization()
        rates = self.lambda_base * np.exp(self.alpha * (util - 1.0))
        
        # Convert instantaneous rate to probability over dt_hours
        p_fail = 1.0 - np.exp(-rates * self.dt_hours)
        
        # Cap probability at 1.0
        return np.minimum(p_fail, 1.0)

    def _sample_failures(self) -> List[int]:
        """Sample natural failures using paired randomness."""
        p_fail = self._get_failure_probs()

        if self.t < self._fail_uniforms.shape[0]:
            u = self._fail_uniforms[self.t]
        else:
            # Fallback to random if scenario fail_uniforms are exhausted
            u = np.random.random(self.n).astype(np.float32)

        will_fail = (u < p_fail) & self.operational
        return np.where(will_fail)[0].tolist()

    def _shutdown_and_redistribute(self, node_id: int) -> None:
        """Fully shut down a node and redistribute its load with drop_fraction."""
        if not self.operational[node_id]:
            return

        load = float(self.loads[node_id])
        self.operational[node_id] = False
        self.capacities[node_id] = 0.0
        self.loads[node_id] = 0.0

        # Redistribute (1 - drop_fraction) * load to neighbors
        redist_amount = load * (1.0 - self.drop_fraction)
        dropped_amount = load * self.drop_fraction
        
        self.cum_dropped_load += dropped_amount
        if redist_amount > 0:
            self._redistribute_load(node_id, redist_amount)

    def _redistribute_load(self, source_id: int, amount: float) -> None:
        """Redistribute `amount` of load from source to its operational neighbors."""
        if amount <= 0:
            return

        alive_nbrs = [j for j in self.adj[source_id] if self.operational[j]]
        if not alive_nbrs:
            return  # load is lost (dropped)

        # Weight by remaining empty capacity
        caps = self.capacities[alive_nbrs].astype(np.float32)
        loads_nbr = self.loads[alive_nbrs].astype(np.float32)
        empty = np.maximum(caps - loads_nbr, 0.0)
        total_empty = float(empty.sum())

        if total_empty <= 1e-12:
            weights = np.ones(len(alive_nbrs), dtype=np.float32) / len(alive_nbrs)
        else:
            weights = empty / total_empty

        for j, w in zip(alive_nbrs, weights.tolist()):
            self.loads[j] += amount * w
