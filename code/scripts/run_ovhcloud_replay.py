from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

CODE_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = CODE_ROOT.parent
sys.path.insert(0, str(CODE_ROOT))

from config import Config
from training.train_sac import build_agent, make_env
from env.edge_env_v2 import Scenario
from agents.baselines import HeuristicThrottleAgent


def make_campus_scenario(seed: int) -> tuple[Scenario, list[list[int]]]:
    rng = np.random.default_rng(seed)
    sizes = [20, 30, 25, 25]  # SBG1..SBG4
    n = sum(sizes)

    groups: list[list[int]] = []
    start = 0
    for size in sizes:
        groups.append(list(range(start, start + size)))
        start += size

    adj = [[] for _ in range(n)]

    def add_edge(i: int, j: int) -> None:
        if j not in adj[i]:
            adj[i].append(j)
        if i not in adj[j]:
            adj[j].append(i)

    # Dense intra-building connectivity.
    for group in groups:
        for i in group:
            k = min(6, len(group) - 1)
            nbrs = rng.choice([x for x in group if x != i], size=k, replace=False)
            for j in nbrs:
                add_edge(i, int(j))

    # SBG2 is the main hub to SBG1 and SBG3; SBG3 links onward to SBG4.
    for i in groups[1]:
        for j in rng.choice(groups[0], size=2, replace=False):
            add_edge(i, int(j))
        for j in rng.choice(groups[2], size=2, replace=False):
            add_edge(i, int(j))
    for i in groups[2]:
        for j in rng.choice(groups[3], size=2, replace=False):
            add_edge(i, int(j))

    capacities = np.zeros(n, dtype=np.float32)
    cap_ranges = [(70.0, 110.0), (80.0, 130.0), (85.0, 125.0), (85.0, 125.0)]
    for group, (lo, hi) in zip(groups, cap_ranges):
        capacities[group] = rng.uniform(lo, hi, size=len(group)).astype(np.float32)

    util = np.zeros(n, dtype=np.float32)
    util_ranges = [(0.70, 0.80), (0.76, 0.86), (0.68, 0.78), (0.66, 0.76)]
    for group, (lo, hi) in zip(groups, util_ranges):
        util[group] = rng.uniform(lo, hi, size=len(group)).astype(np.float32)
    loads = (capacities * util).astype(np.float32)

    fail_uniforms = rng.random(size=(1440, n), dtype=np.float32)
    return Scenario(adj=adj, capacities=capacities, loads=loads, fail_uniforms=fail_uniforms), groups


def apply_exogenous_fire(env, sbg2_nodes: list[int], t_min: int) -> None:
    # Stylized fire spread: one additional SBG2 node fails every 3 minutes,
    # so the 30-node SBG2 cluster is exhausted over the first 90 minutes.
    if t_min >= 90:
        return
    idx = min(len(sbg2_nodes) - 1, t_min // 3)
    for node_id in sbg2_nodes[: idx + 1]:
        if env.operational[node_id]:
            env.cum_natural_failed_capacity += float(env._cap0[node_id])
            env.natural_failure_count += 1
            env._shutdown_and_redistribute(int(node_id))


def rollout(policy_fn, env, adj, groups, detection_delay_min: int, horizon_min: int) -> tuple[float, float]:
    state = env.reset()
    cap_trace = []
    for _ in range(horizon_min):
        apply_exogenous_fire(env, groups[1], env.t)
        state = env.get_state()
        cap_trace.append(float(env.capacities[env.operational].sum()) / float(env.initial_total_capacity))

        if env.t < detection_delay_min:
            action = np.zeros(env.n, dtype=np.float32)
        else:
            action = policy_fn(state, adj)

        if env.done:
            break

        state, _, done, _info = env.step(action)
        if done:
            cap_trace.append(float(env.capacities[env.operational].sum()) / float(env.initial_total_capacity))
            last = cap_trace[-1]
            cap_trace.extend([last] * max(0, horizon_min - len(cap_trace)))
            break

    if not cap_trace:
        cap_trace = [0.0] * horizon_min
    elif len(cap_trace) < horizon_min:
        cap_trace.extend([cap_trace[-1]] * (horizon_min - len(cap_trace)))

    trace = np.asarray(cap_trace, dtype=float)
    remaining_capacity_6h = float(trace[-1])
    aupc_6h = float(trace.mean() * (horizon_min / 60.0))
    return remaining_capacity_6h, aupc_6h


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=str(CODE_ROOT / "weights" / "gat_sac_v8_retrained_seed42.pt"))
    parser.add_argument("--start-seed", type=int, default=100)
    parser.add_argument("--num-seeds", type=int, default=10)
    parser.add_argument("--detection-delay-min", type=int, default=7)
    parser.add_argument("--horizon-min", type=int, default=360)
    parser.add_argument("--out-csv", type=str, default=str(PACKAGE_ROOT / "artifacts" / "ovhcloud_replay_results.csv"))
    parser.add_argument("--out-json", type=str, default=str(PACKAGE_ROOT / "artifacts" / "ovhcloud_replay_summary.json"))
    args = parser.parse_args()

    torch.set_num_threads(min(8, os.cpu_count() or 8))

    cfg = Config()
    cfg.train.device = "cpu"
    cfg.env.n_nodes = 100
    cfg.env.max_hours = 24.0
    cfg.env.alpha = 1.2
    cfg.env.lambda_base = 0.12
    cfg.env.drop_fraction = 0.5
    # Keep the replay alive long enough to measure the first 6 hours.
    cfg.env.demand_threshold_frac = 0.10

    agent = build_agent(cfg)
    agent.load(args.weights)
    agent.eval_mode()

    rows = []
    for offset in range(args.num_seeds):
        seed = args.start_seed + offset
        scenario, groups = make_campus_scenario(seed=seed)

        methods = {
            "No Defence": lambda s, a: np.zeros(cfg.env.n_nodes, dtype=np.float32),
            "Heuristic Throttle (30%)": HeuristicThrottleAgent(util_threshold=0.90, throttle_amount=0.30).act,
            "GAT-SAC (Ours)": None,
        }

        for method_name, policy_fn in methods.items():
            env = make_env(scenario, cfg, redundancy=1.0)
            adj = env.get_adjacency_matrix()
            if policy_fn is None:
                agent.cache_adjacency(adj)
                policy_fn = lambda s, a: agent.act(s, a, deterministic=True)

            remaining_capacity_6h, aupc_6h = rollout(
                policy_fn,
                env,
                adj,
                groups,
                detection_delay_min=args.detection_delay_min,
                horizon_min=args.horizon_min,
            )
            rows.append(
                {
                    "Seed": seed,
                    "Method": method_name,
                    "Remaining Capacity at 6h (%)": 100.0 * remaining_capacity_6h,
                    "Unavailable Capacity at 6h (%)": 100.0 * (1.0 - remaining_capacity_6h),
                    "AUPC over first 6h": aupc_6h,
                }
            )

    out_csv = Path(args.out_csv)
    out_json = Path(args.out_json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    summary = {}
    grouped = df.groupby("Method")
    for method_name, group in grouped:
        summary[method_name] = {
            "remaining_capacity_6h_mean_pct": float(group["Remaining Capacity at 6h (%)"].mean()),
            "remaining_capacity_6h_std_pct": float(group["Remaining Capacity at 6h (%)"].std(ddof=1)),
            "aupc_6h_mean": float(group["AUPC over first 6h"].mean()),
            "aupc_6h_std": float(group["AUPC over first 6h"].std(ddof=1)),
        }

    no_def = summary["No Defence"]
    gat = summary["GAT-SAC (Ours)"]
    compare = {
        "remaining_capacity_6h_abs_gain_pct_points": gat["remaining_capacity_6h_mean_pct"] - no_def["remaining_capacity_6h_mean_pct"],
        "remaining_capacity_6h_rel_gain_pct": 100.0 * (gat["remaining_capacity_6h_mean_pct"] - no_def["remaining_capacity_6h_mean_pct"]) / max(no_def["remaining_capacity_6h_mean_pct"], 1e-9),
        "aupc_6h_abs_gain": gat["aupc_6h_mean"] - no_def["aupc_6h_mean"],
        "aupc_6h_rel_gain_pct": 100.0 * (gat["aupc_6h_mean"] - no_def["aupc_6h_mean"]) / max(no_def["aupc_6h_mean"], 1e-9),
    }

    payload = {
        "observed_context_used_for_mapping": {
            "incident": "OVHcloud Strasbourg fire",
            "incident_date": "2021-03-10",
            "reported_peak_service_disruption": "3.6 million websites",
            "reported_detection_delay_min": 7,
            "reported_recovery_window": "services resumed by 2021-03-22",
        },
        "replay_assumptions": {
            "graph_mapping": "4-cluster, 100-node campus graph representing SBG1-SBG4",
            "cluster_sizes": {"SBG1": 20, "SBG2": 30, "SBG3": 25, "SBG4": 25},
            "exogenous_shock": "progressive SBG2 loss over first 90 minutes",
            "detection_delay_min": args.detection_delay_min,
            "evaluation_horizon_min": args.horizon_min,
            "weights": args.weights,
        },
        "summary": summary,
        "comparison_gat_vs_no_defence": compare,
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[save] {out_csv}")
    print(f"[save] {out_json}")


if __name__ == "__main__":
    main()
