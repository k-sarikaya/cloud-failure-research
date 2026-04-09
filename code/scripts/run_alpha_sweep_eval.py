from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Ensure we can import from the packaged code tree.
CODE_ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT))

from config import Config  # noqa: E402
from env.edge_env_v2 import EdgeEnvV2, Scenario, load_scenario_bank_npz  # noqa: E402
from agents.baselines import HeuristicThrottleAgent, NoDefenseAgent  # noqa: E402
from agents.sac_agent import GATSACAgent  # noqa: E402


ARTIFACTS = PKG_ROOT / "artifacts"
WEIGHTS_DIR = CODE_ROOT / "weights"
ARTIFACTS.mkdir(exist_ok=True)


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_agent(cfg: Config) -> GATSACAgent:
    return GATSACAgent(
        n_nodes=cfg.env.n_nodes,
        node_feat_dim=cfg.gat.node_feat_dim,
        gat_hidden=cfg.gat.hidden_dim,
        gat_heads=cfg.gat.num_heads,
        gat_dropout=cfg.gat.dropout,
        sac_hidden=cfg.sac.hidden_dim,
        lr_actor=cfg.sac.lr_actor,
        lr_critic=cfg.sac.lr_critic,
        lr_alpha=cfg.sac.lr_alpha,
        lr_lambda=cfg.sac.lr_lambda,
        discount=cfg.sac.discount,
        tau=cfg.sac.tau,
        init_alpha=cfg.sac.init_alpha,
        capacity_loss_limit=cfg.sac.capacity_loss_limit,
        dropped_load_limit=cfg.sac.dropped_load_limit,
        buffer_size=cfg.sac.buffer_size,
        batch_size=cfg.sac.batch_size,
        device=cfg.train.device,
    )


def make_env(scenario: Scenario, cfg: Config, *, alpha: float) -> EdgeEnvV2:
    env = EdgeEnvV2(
        scenario=scenario,
        dt_hours=cfg.env.dt_hours,
        max_hours=cfg.env.max_hours,
        lambda_base=cfg.env.lambda_base,
        alpha=float(alpha),
        demand_threshold_frac=cfg.env.demand_threshold_frac,
        throttle_min=cfg.env.throttle_min,
        throttle_cost_coeff=cfg.env.throttle_cost_coeff,
        redundancy_factor=1.0,
        drop_fraction=cfg.env.drop_fraction,
    )

    # Ensure replay stays deterministic by never falling back to fresh randomness.
    max_steps = int(round(cfg.env.max_hours / cfg.env.dt_hours))
    if getattr(env, "_fail_uniforms", None) is None:
        raise RuntimeError("EdgeEnvV2 missing paired fail_uniforms; cannot guarantee replay")
    if max_steps > int(env._fail_uniforms.shape[0]):
        raise ValueError(
            f"Scenario bank fail_uniforms too short for this cfg: max_steps={max_steps} "
            f"> fail_uniforms_steps={int(env._fail_uniforms.shape[0])}. "
            "Reduce max_hours or regenerate the scenario bank with a larger max_steps."
        )

    return env


@torch.no_grad()
def eval_episode(env: EdgeEnvV2, policy, adj: np.ndarray) -> dict:
    state = env.reset()
    done = False
    while not done:
        action = policy(state, adj)
        state, _, done, info = env.step(action)

    return {
        "SST (h)": info["hours"],
        "Natural Failures": info["natural_failure_count"],
        "Hard Capacity Removal (%)": 100.0 * info["proactive_cap_loss_ratio"],
        "Natural Loss (%)": 100.0 * info["natural_cap_loss_ratio"],
        "Total Loss (%)": 100.0 * info["total_cap_loss_ratio"],
        "Final Capacity (%)": 100.0 * info["remaining_capacity_ratio"],
        "Throttled Load (%)": 100.0 * info["throttled_load_ratio"],
        "Dropped Load (%)": 100.0 * info["dropped_load_ratio"],
        "Operational Nodes": info["n_operational"],
    }


def parse_alphas(raw: str) -> list[float]:
    out: list[float] = []
    for part in (raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    if not out:
        raise ValueError("--alphas must not be empty")
    return out


def summarize(results: pd.DataFrame, out_csv: Path) -> pd.DataFrame:
    metric_cols = [
        "SST (h)",
        "Hard Capacity Removal (%)",
        "Natural Loss (%)",
        "Total Loss (%)",
        "Final Capacity (%)",
        "Throttled Load (%)",
        "Dropped Load (%)",
    ]
    summary = (
        results.groupby(["alpha", "Method"], as_index=False)
        .agg({
            "SST (h)": ["mean", "std"],
            "Hard Capacity Removal (%)": "mean",
            "Natural Loss (%)": "mean",
            "Total Loss (%)": "mean",
            "Final Capacity (%)": "mean",
            "Throttled Load (%)": "mean",
            "Dropped Load (%)": "mean",
        })
    )

    # Flatten columns
    summary.columns = [
        (a if b == "" else f"{a}_{b}")
        for a, b in [(c[0], c[1]) if isinstance(c, tuple) else (c, "") for c in summary.columns]
    ]

    # Convenience deltas vs. No Defence at each alpha
    nodef = summary[summary["Method"] == "No Defence"][
        ["alpha", "SST (h)_mean"]
    ].rename(columns={"SST (h)_mean": "nodefence_mean_h"})

    merged = summary.merge(nodef, on="alpha", how="left")
    merged["delta_vs_nodefence_mean_h"] = merged["SST (h)_mean"] - merged["nodefence_mean_h"]
    merged["ratio_vs_nodefence_mean"] = merged["SST (h)_mean"] / merged["nodefence_mean_h"].replace(0.0, np.nan)

    merged = merged.round(3)
    merged.to_csv(out_csv, index=False)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=str(WEIGHTS_DIR / "gat_sac_v8_retrained_seed42.pt"))
    parser.add_argument(
        "--scenario-bank",
        type=str,
        default=str(ARTIFACTS / "scenario_bank_ba_n100_k40_seed42.npz"),
        help="NPZ exported by scripts/export_scenario_bank.py",
    )
    parser.add_argument("--eval-scenarios", type=int, default=40)
    parser.add_argument("--alphas", type=str, default="0.5,0.7,0.9,1.1,1.3,1.5")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out-tag", type=str, default="alpha_sweep_ba_n100_k40_seed42")
    parser.add_argument("--seed", type=int, default=42, help="Only affects torch/random seeding; scenarios come from the bank")
    args = parser.parse_args()

    alphas = parse_alphas(args.alphas)
    scenario_bank = Path(args.scenario_bank)
    weights_path = Path(args.weights)
    if not scenario_bank.exists():
        raise FileNotFoundError(f"Missing scenario bank: {scenario_bank}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing weights: {weights_path}")
    if int(args.eval_scenarios) <= 0:
        raise ValueError("--eval-scenarios must be positive")

    # Freeze evaluation environment config (canonical BA-100) and only sweep cfg.env.alpha.
    cfg = Config()
    cfg.train.device = str(args.device)
    cfg.train.seed = int(args.seed)
    cfg.env.topo_type = "ba"
    cfg.env.n_nodes = 100

    torch.set_num_threads(min(8, os.cpu_count() or 8))
    set_seeds(cfg.train.seed)

    scenarios = load_scenario_bank_npz(scenario_bank)
    n_req = int(args.eval_scenarios)
    if len(scenarios) < n_req:
        raise ValueError(f"Scenario bank size mismatch: expected at least {n_req}, got {len(scenarios)}")
    if len(scenarios) > n_req:
        # Allow quick smoke tests by taking the first N scenarios from the frozen bank.
        scenarios = scenarios[:n_req]
    if scenarios and len(scenarios[0].adj) != int(cfg.env.n_nodes):
        raise ValueError(f"Scenario bank node count mismatch: expected N={cfg.env.n_nodes}, got N={len(scenarios[0].adj)}")

    agent = build_agent(cfg)
    agent.load(str(weights_path), inference_only=True)
    agent.eval_mode()

    no_def = NoDefenseAgent()
    heur = HeuristicThrottleAgent(util_threshold=0.90, throttle_amount=0.30)

    methods = {
        "No Defence": no_def,
        "Heuristic Pruning (Graceful Drain)": heur,
        "GAT-SAC (Ours)": None,
    }

    out_tag = str(args.out_tag).strip()
    out_results = ARTIFACTS / f"{out_tag}_results.csv"
    out_summary = ARTIFACTS / f"{out_tag}_summary.csv"
    out_meta = ARTIFACTS / f"{out_tag}_meta.json"

    rows: list[dict] = []
    t_all = time.time()

    for a in alphas:
        t_alpha = time.time()
        for method_name, baseline in methods.items():
            for idx, scenario in enumerate(scenarios, start=1):
                env = make_env(scenario, cfg, alpha=a)
                adj = env.get_adjacency_matrix()

                if baseline is None:
                    agent.cache_adjacency(adj)
                    policy = lambda s, mat: agent.act(s, mat, deterministic=True)
                else:
                    policy = baseline.act

                result = eval_episode(env, policy, adj)
                result["alpha"] = float(a)
                result["Method"] = method_name
                result["Scenario"] = int(idx)
                rows.append(result)

        elapsed = time.time() - t_alpha
        print(f"[alpha] {a:.3f} done in {elapsed:.1f}s")

    results = pd.DataFrame(rows)
    results.to_csv(out_results, index=False)
    summary = summarize(results, out_summary)

    meta = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "weights": str(weights_path),
        "scenario_bank": str(scenario_bank),
        "eval_scenarios": int(args.eval_scenarios),
        "alphas": alphas,
        "methods": list(methods.keys()),
        "config_env": asdict(cfg.env),
        "config_train": asdict(cfg.train),
        "wall_time_s": round(time.time() - t_all, 3),
    }
    out_meta.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

    print(f"[save] raw    -> {out_results}")
    print(f"[save] summary -> {out_summary}")
    print(f"[save] meta   -> {out_meta}")
    if not summary.empty:
        print(summary.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
