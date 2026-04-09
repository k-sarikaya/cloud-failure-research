from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import Config
from training.train_sac import build_agent, paired_eval, set_seeds

WEIGHTS = ROOT / 'weights'
ART = ROOT / 'artifacts'

CHECKPOINTS = [
    ('ep100', WEIGHTS / 'gat_sac_v8_retrained_seed42_ep100.pt'),
    ('ep200', WEIGHTS / 'gat_sac_v8_retrained_seed42_ep200.pt'),
    ('ep300', WEIGHTS / 'gat_sac_v8_retrained_seed42_ep300.pt'),
    ('ep400', WEIGHTS / 'gat_sac_v8_retrained_seed42_ep400.pt'),
    ('ep500', WEIGHTS / 'gat_sac_v8_retrained_seed42_ep500.pt'),
    ('final', WEIGHTS / 'gat_sac_v8_retrained_seed42.pt'),
]


def main():
    cfg = Config()
    cfg.train.seed = 42
    cfg.train.device = 'cpu'
    set_seeds(cfg.train.seed)
    rows = []
    for label, ckpt in CHECKPOINTS:
        agent = build_agent(cfg)
        agent.load(str(ckpt))
        out_csv = ART / f'checkpoint_eval_{label}.csv'
        df = paired_eval(cfg, agent, 20, out_csv)
        gat = df[df['Method'] == 'GAT-SAC (Ours)']
        rows.append({
            'checkpoint': label,
            'mean_sst_h': float(gat['SST (h)'].mean()),
            'std_sst_h': float(gat['SST (h)'].std(ddof=1)),
            'mean_throttled_pct': float(gat['Throttled Load (%)'].mean()),
            'mean_dropped_pct': float(gat['Dropped Load (%)'].mean()),
        })
        print(label, rows[-1])
    pd.DataFrame(rows).to_csv(ART / 'checkpoint_progression.csv', index=False)


if __name__ == '__main__':
    main()
