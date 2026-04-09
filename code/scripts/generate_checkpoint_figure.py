from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts'
FIG = ROOT / 'figures'
FIG.mkdir(exist_ok=True)


def main():
    df = pd.read_csv(ART / 'checkpoint_progression.csv')
    order_map = {'ep100':100,'ep200':200,'ep300':300,'ep400':400,'ep500':500,'final':550}
    df['x'] = df['checkpoint'].map(order_map)
    plt.figure(figsize=(8,4.8))
    plt.plot(df['x'], df['mean_sst_h'], marker='o')
    plt.fill_between(df['x'], df['mean_sst_h'] - df['std_sst_h'], df['mean_sst_h'] + df['std_sst_h'], alpha=0.2)
    plt.xticks(df['x'], df['checkpoint'])
    plt.ylabel('Mean SST (h)')
    plt.xlabel('Checkpoint')
    plt.title('Clean paired evaluation across checkpoint lineage')
    plt.tight_layout()
    plt.savefig(FIG / 'fig_checkpoint_progression.png', dpi=300)
    plt.savefig(FIG / 'fig_checkpoint_progression.pdf')


if __name__ == '__main__':
    main()
