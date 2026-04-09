from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import time
from pathlib import Path
from typing import Callable, List

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from env.edge_env_v2 import EdgeEnvV2, Scenario, generate_paired_scenarios, load_scenario_bank_npz, make_scenario
from agents.sac_agent import GATSACAgent
from agents.baselines import HeuristicThrottleAgent, NoDefenseAgent
from agents.dqn_agent import DQNAgent, DQNBinaryWrapper


CODE_ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = PKG_ROOT / "artifacts"
WEIGHTS = CODE_ROOT / "weights"
ARTIFACTS.mkdir(exist_ok=True)
WEIGHTS.mkdir(exist_ok=True)


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_env(scenario: Scenario, cfg: Config, redundancy: float = 1.0) -> EdgeEnvV2:
    return EdgeEnvV2(
        scenario=scenario,
        dt_hours=cfg.env.dt_hours,
        max_hours=cfg.env.max_hours,
        lambda_base=cfg.env.lambda_base,
        alpha=cfg.env.alpha,
        demand_threshold_frac=cfg.env.demand_threshold_frac,
        throttle_min=cfg.env.throttle_min,
        throttle_cost_coeff=cfg.env.throttle_cost_coeff,
        redundancy_factor=redundancy,
        drop_fraction=cfg.env.drop_fraction,
    )


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


def init_training_log(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "episode",
            "steps",
            "return",
            "sst_hours",
            "natural_failures",
            "remaining_capacity_ratio",
            "proactive_cap_loss_ratio",
            "dropped_load_ratio",
            "critic_loss",
            "actor_loss",
            "alpha",
            "lagrange_lambda",
            "mean_throttle",
        ])


def append_training_log(path: Path, row: list[object]) -> None:
    with path.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)


def get_last_logged_episode(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        df = pd.read_csv(path)
    except Exception:
        return 0
    if "episode" not in df.columns or df.empty:
        return 0
    try:
        return int(df["episode"].max())
    except Exception:
        return 0

@torch.no_grad()
def eval_episode(env: EdgeEnvV2, policy: Callable[[np.ndarray, np.ndarray], np.ndarray], adj: np.ndarray) -> dict:
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


def build_fc_dqn_policy(cfg: Config) -> callable:
    """Load the packaged FC-DQN weights and expose an EdgeEnvV2-compatible policy."""
    na = 50
    state_dim = na * 3
    action_dim = na + 1
    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim, device=cfg.train.device, na=na)
    weights_path = WEIGHTS / 'dqn_binary.pt'
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing FC-DQN weights: {weights_path}")
    agent.q.load_state_dict(torch.load(weights_path, map_location=cfg.train.device))
    wrapper = DQNBinaryWrapper(agent, n_total=cfg.env.n_nodes, na=na)
    return wrapper.act


def paired_eval(
    cfg: Config,
    agent: GATSACAgent,
    num_scenarios: int,
    out_csv: Path,
    scenario_bank: Path | None = None,
) -> pd.DataFrame:
    if scenario_bank is not None:
        scenarios = load_scenario_bank_npz(scenario_bank)
        if len(scenarios) != int(num_scenarios):
            raise ValueError(f"Scenario bank size mismatch: expected {num_scenarios}, got {len(scenarios)}")

        # Sanity check: bank node count must match the configured environment.
        if scenarios and len(scenarios[0].adj) != int(cfg.env.n_nodes):
            raise ValueError(
                f"Scenario bank node count mismatch: expected N={cfg.env.n_nodes}, got N={len(scenarios[0].adj)}"
            )
    else:
        scenarios = generate_paired_scenarios(
            seed=cfg.train.seed,
            num_scenarios=num_scenarios,
            n_nodes=cfg.env.n_nodes,
            topo_type=cfg.env.topo_type,
            ba_m=cfg.env.ba_m,
            max_steps=cfg.env.max_steps,
            capacity_range=cfg.env.capacity_range,
            init_util_range=cfg.env.init_util_range,
        )

    fc_dqn_policy = build_fc_dqn_policy(cfg)

    methods = {
        "No Defence": (NoDefenseAgent(), 1.0),
        "Heuristic Pruning (Graceful Drain)": (HeuristicThrottleAgent(util_threshold=0.90, throttle_amount=0.30), 1.0),
        "+25% Backup": (NoDefenseAgent(), 1.25),
        "+50% Backup": (NoDefenseAgent(), 1.50),
        "FC-DQN": (fc_dqn_policy, 1.0),
        "GAT-SAC (Ours)": (None, 1.0),
    }

    agent.eval_mode()
    rows = []
    for method_name, (baseline, redundancy) in methods.items():
        for idx, scenario in enumerate(scenarios, start=1):
            env = make_env(scenario, cfg, redundancy=redundancy)
            adj = env.get_adjacency_matrix()
            if baseline is None:
                agent.cache_adjacency(adj)
                result = eval_episode(env, lambda s, a: agent.act(s, a, deterministic=True), adj)
            else:
                policy = baseline.act if hasattr(baseline, "act") else baseline
                result = eval_episode(env, policy, adj)
            result["Method"] = method_name
            result["Scenario"] = idx
            rows.append(result)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return df


def train(
    agent: GATSACAgent,
    cfg: Config,
    *,
    weights_path: Path,
    log_path: Path,
    resume: bool,
) -> None:
    rng = np.random.default_rng(cfg.train.seed + 12345)
    start_episode = 1
    if resume and log_path.exists():
        start_episode = get_last_logged_episode(log_path) + 1
    else:
        init_training_log(log_path)

    print(f"[train] start_episode={start_episode}")

    for episode in range(start_episode, cfg.train.episodes + 1):
        scenario = make_scenario(
            n_nodes=cfg.env.n_nodes,
            topo_type=cfg.env.topo_type,
            ba_m=cfg.env.ba_m,
            max_steps=cfg.env.max_steps,
            capacity_range=cfg.env.capacity_range,
            init_util_range=cfg.env.init_util_range,
            rng=rng,
        )
        env = make_env(scenario, cfg)
        adj = env.get_adjacency_matrix()
        state = env.reset()
        done = False
        episode_return = 0.0
        steps = 0
        updates = []

        while not done:
            action = agent.act(state, adj, deterministic=False)
            next_state, reward, done, info = env.step(action)
            cost = max(
                info.get("proactive_cap_loss_ratio", 0.0) - cfg.sac.capacity_loss_limit,
                info.get("dropped_load_ratio", 0.0) - cfg.sac.dropped_load_limit,
                0.0,
            )
            agent.replay.add(state, action, reward, next_state, done, cost)
            if agent.total_steps % cfg.train.update_every == 0:
                loss_info = agent.update(adj, rng)
                if loss_info is not None:
                    updates.append(loss_info)
            agent.total_steps += 1
            state = next_state
            episode_return += reward
            steps += 1

        critic_loss = float(np.mean([x["critic_loss"] for x in updates])) if updates else float("nan")
        actor_loss = float(np.mean([x["actor_loss"] for x in updates])) if updates else float("nan")
        mean_throttle = float(np.mean([x["mean_throttle"] for x in updates])) if updates else 0.0
        append_training_log(log_path, [
            episode,
            steps,
            round(episode_return, 6),
            round(info["hours"], 6),
            info["natural_failure_count"],
            round(info["remaining_capacity_ratio"], 6),
            round(info["proactive_cap_loss_ratio"], 6),
            round(info["dropped_load_ratio"], 6),
            round(critic_loss, 6) if critic_loss == critic_loss else "nan",
            round(actor_loss, 6) if actor_loss == actor_loss else "nan",
            round(agent.alpha.item(), 6),
            round(agent.lagrange_lambda.item(), 6),
            round(mean_throttle, 6),
        ])

        if episode == 1 or episode == start_episode or episode % cfg.train.log_interval == 0:
            print(
                f"[train] ep={episode:4d}/{cfg.train.episodes} steps={steps:4d} "
                f"return={episode_return:8.2f} sst={info['hours']:.2f}h "
                f"critic={critic_loss:.4f} actor={actor_loss:.4f} throttle={mean_throttle:.4f} "
                f"alpha={agent.alpha.item():.4f} lambda={agent.lagrange_lambda.item():.4f}"
            )

        if episode % cfg.train.save_interval == 0:
            ckpt = weights_path.with_name(f"{weights_path.stem}_ep{episode}{weights_path.suffix}")
            agent.save(str(ckpt))

    agent.save(str(weights_path))


def summarize(df: pd.DataFrame, out_csv: Path) -> pd.DataFrame:
    if df is None or df.empty:
        # Allow training-only runs (or eval_scenarios=0) without crashing.
        out_csv.write_text("empty\n", encoding="utf-8")
        return pd.DataFrame()
    summary = df.groupby("Method").agg({
        "SST (h)": ["mean", "std"],
        "Hard Capacity Removal (%)": "mean",
        "Natural Loss (%)": "mean",
        "Total Loss (%)": "mean",
        "Final Capacity (%)": "mean",
        "Throttled Load (%)": "mean",
        "Dropped Load (%)": "mean",
    }).round(3)
    summary.to_csv(out_csv)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    # Canonical weights used for the verified 40-scenario BA-100 benchmark in this submission package.
    parser.add_argument("--weights", type=str, default=str(WEIGHTS / "gat_sac_v8_retrained_seed42.pt"))
    parser.add_argument(
        "--inference-only-load",
        action="store_true",
        help="Load only GAT+actor weights for fixed-policy transfer across graph sizes during evaluation.",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--eval-scenarios", type=int, default=40)
    parser.add_argument("--scenario-bank", type=str, default=None, help="NPZ exported by scripts/export_scenario_bank.py")
    parser.add_argument("--topo-type", type=str, default=None)
    parser.add_argument("--n-nodes", type=int, default=None)
    parser.add_argument("--ba-m", type=int, default=None)
    args = parser.parse_args()

    if args.inference_only_load and not args.eval_only:
        raise ValueError("--inference-only-load is only supported together with --eval-only")

    cfg = Config()
    cfg.train.seed = args.seed
    cfg.train.episodes = args.episodes
    cfg.train.device = args.device
    if args.topo_type is not None:
        cfg.env.topo_type = str(args.topo_type)
    if args.n_nodes is not None:
        cfg.env.n_nodes = int(args.n_nodes)
    if args.ba_m is not None:
        cfg.env.ba_m = int(args.ba_m)

    torch.set_num_threads(min(8, os.cpu_count() or 8))
    set_seeds(cfg.train.seed)

    print("=" * 72)
    print("GAT-SAC for Cascading Failure Prevention")
    print("=" * 72)
    print(
        f"[config] topo={cfg.env.topo_type} N={cfg.env.n_nodes} BA(m)={cfg.env.ba_m} max_hours={cfg.env.max_hours} "
        f"lambda_base={cfg.env.lambda_base} alpha={cfg.env.alpha} phi={cfg.env.drop_fraction}"
    )

    agent = build_agent(cfg)
    weights_path = Path(args.weights)

    topo_key = str(cfg.env.topo_type).strip().lower().replace('-', '_')
    tag = f"{topo_key}_n{cfg.env.n_nodes}"
    canonical_tag = "ba_n100"

    # Safety: avoid clobbering the submission's canonical artifacts when running
    # exploratory topology/scalability experiments.
    if tag == canonical_tag:
        log_path = ARTIFACTS / "training_log.csv"
        eval_csv = ARTIFACTS / "paired_eval_results.csv"
        summary_csv = ARTIFACTS / "paired_eval_summary.csv"
    else:
        log_path = ARTIFACTS / f"training_log_{tag}.csv"
        eval_csv = ARTIFACTS / f"paired_eval_results_{tag}.csv"
        summary_csv = ARTIFACTS / f"paired_eval_summary_{tag}.csv"

    if args.eval_only:
        print(f"[load] loading weights from {weights_path}")
        agent.load(str(weights_path), inference_only=args.inference_only_load)
    else:
        if args.resume:
            # Load weights if present; otherwise start from scratch.
            if weights_path.exists():
                print(f"[resume] loading weights from {weights_path}")
                agent.load(str(weights_path))
            else:
                print(f"[resume] weights not found, starting from scratch: {weights_path}")
        else:
            # Safety: avoid accidentally overwriting an existing training log.
            if log_path.exists():
                ts = int(time.time())
                backup = log_path.with_name(f"training_log_{ts}.csv")
                log_path.replace(backup)
                print(f"[warn] existing training log moved to {backup}")
        print(f"[train] training for {cfg.train.episodes} episodes")
        t0 = time.time()
        train(agent, cfg, weights_path=weights_path, log_path=log_path, resume=args.resume)
        print(f"[train] done in {time.time() - t0:.1f}s")

    if args.skip_eval:
        print("[eval] skipped (requested)")
        return

    if args.eval_scenarios <= 0:
        print("[eval] skipped (eval_scenarios <= 0)")
        return

    print("[eval] running paired evaluation")
    scenario_bank = Path(args.scenario_bank) if args.scenario_bank else None
    df = paired_eval(cfg, agent, args.eval_scenarios, eval_csv, scenario_bank=scenario_bank)
    summary = summarize(df, summary_csv)
    if not summary.empty:
        print(summary.to_string())
    print(f"[save] raw eval -> {eval_csv}")
    print(f"[save] summary  -> {summary_csv}")


if __name__ == "__main__":
    main()
