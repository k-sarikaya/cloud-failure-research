from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Allow running as a script from arbitrary working directories.
CODE_ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT))

from config import Config
from training.train_sac import build_agent, paired_eval, set_seeds, summarize, train


def parse_csv_list(value: str) -> list[str]:
    return [v.strip() for v in (value or '').split(',') if v.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(v.strip()) for v in (value or '').split(',') if v.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', type=str, default='cpu')
    ap.add_argument('--topos', type=str, default='ba,ws,er,5g_mec')
    ap.add_argument('--node-counts', type=str, default='100,150,200')
    ap.add_argument('--episodes', type=int, default=500)
    ap.add_argument('--train-max-hours', type=float, default=None, help='Optional shorter horizon during training only (evaluation keeps cfg default)')
    ap.add_argument('--eval-scenarios', type=int, default=20)
    ap.add_argument('--train', action='store_true', help='Train per (topo,n) before eval')
    ap.add_argument('--resume', action='store_true', help='Resume training if weights exist')
    ap.add_argument('--out', type=str, default=None)
    ap.add_argument('--shared-weights', type=str, default=None, help='Optional single weights file reused for every (topo,n) as a fixed trained policy')
    args = ap.parse_args()

    if args.shared_weights and args.train:
        raise ValueError('Refusing to --train when --shared-weights is set (avoid clobbering canonical weights).')

    topos = parse_csv_list(args.topos)
    node_counts = parse_int_list(args.node_counts)

    art = PKG_ROOT / 'artifacts'
    art.mkdir(exist_ok=True)
    weights_dir = CODE_ROOT / 'weights'
    weights_dir.mkdir(exist_ok=True)

    out_csv = Path(args.out) if args.out else art / 'topology_scalability_results.csv'
    rows: list[dict] = []

    for topo in topos:
        for n_nodes in node_counts:
            cfg = Config()
            cfg.train.seed = int(args.seed)
            cfg.train.device = str(args.device)
            cfg.train.episodes = int(args.episodes)
            cfg.env.topo_type = str(topo)
            cfg.env.n_nodes = int(n_nodes)

            set_seeds(cfg.train.seed)
            agent = build_agent(cfg)
            log_path = art / f"training_log_{topo}_n{n_nodes}_seed{cfg.train.seed}.csv"

            if args.shared_weights:
                weights_path = Path(args.shared_weights)
            else:
                weights_path = weights_dir / f"gat_sac_{topo}_n{n_nodes}_seed{cfg.train.seed}.pt"

            if args.train:
                if args.resume and weights_path.exists():
                    agent.load(str(weights_path))
                eval_max_hours = float(cfg.env.max_hours)
                if args.train_max_hours is not None:
                    cfg.env.max_hours = float(args.train_max_hours)
                train(agent, cfg, weights_path=weights_path, log_path=log_path, resume=bool(args.resume))
                cfg.env.max_hours = eval_max_hours
            else:
                if not weights_path.exists():
                    raise FileNotFoundError(
                        f"Missing weights for topo={topo} n={n_nodes}: {weights_path}. "
                        "Run with --train to produce them."
                    )
                agent.load(str(weights_path), inference_only=bool(args.shared_weights))

            eval_csv = art / f"paired_eval_results_{topo}_n{n_nodes}.csv"
            summary_csv = art / f"paired_eval_summary_{topo}_n{n_nodes}.csv"

            df = paired_eval(cfg, agent, int(args.eval_scenarios), eval_csv)
            summ = summarize(df, summary_csv)
            if summ is None or summ.empty:
                continue

            # Flatten multiindex columns from summarize().
            if isinstance(summ.columns, pd.MultiIndex):
                summ.columns = ['_'.join([c for c in col if c]) for col in summ.columns.to_flat_index()]

            for method, row in summ.iterrows():
                rows.append({
                    'topo_type': topo,
                    'n_nodes': int(n_nodes),
                    'seed': int(cfg.train.seed),
                    'episodes': int(cfg.train.episodes),
                    'eval_scenarios': int(args.eval_scenarios),
                    'weights': str(weights_path),
                    'policy_mode': 'fixed_trained_policy' if args.shared_weights else 'matched_checkpoint',
                    'method': str(method),
                    'sst_mean_h': float(row.get('SST (h)_mean', row.get('SST (h)', float('nan')))),
                    'sst_std_h': float(row.get('SST (h)_std', float('nan'))),
                })

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[ok] wrote {out_csv}")


if __name__ == '__main__':
    main()
