from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / 'artifacts'
FIG = ROOT / 'main_text' / 'figures'
FIG.mkdir(parents=True, exist_ok=True)


sns.set_theme(style='whitegrid')
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11})


METHOD_RENAME = {
    'FC-DQN (Legacy Baseline)': 'FC-DQN',
    'FC-DQN ': 'FC-DQN',
    'DQN Baseline (Discrete)': 'FC-DQN',
}

ORDER = [
    'No Defence',
    'Heuristic Pruning (Graceful Drain)',
    '+25% Backup',
    '+50% Backup',
    'FC-DQN',
    'GAT-SAC (Ours)',
]

DISPLAY_LABELS = {
    'No Defence': 'No\nDefence',
    'Heuristic Pruning (Graceful Drain)': 'Heuristic\nDrain',
    '+25% Backup': '+25%\nBackup',
    '+50% Backup': '+50%\nBackup',
    'FC-DQN': 'FC-DQN',
    'GAT-SAC (Ours)': 'GAT-SAC\n(ours)',
}

PALETTE = {
    'No Defence': '#9AA1A6',
    'Heuristic Pruning (Graceful Drain)': '#6FA8DC',
    '+25% Backup': '#B6D7A8',
    '+50% Backup': '#93C47D',
    'FC-DQN': '#F6B26B',
    'GAT-SAC (Ours)': '#E06666',
}

SENSITIVITY_SERIES = [
    ('No Defence', '#7f7f7f'),
    ('GAT-SAC (Ours)', '#1f77b4'),
]


def normalize_methods(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out['Method'] = out['Method'].astype(str).str.strip().replace(METHOD_RENAME)
    return out


def save_with_aliases(fig, stem: str, aliases: list[str] | None = None) -> None:
    aliases = aliases or []
    targets = [stem, *aliases]
    for name in targets:
        fig.savefig(FIG / f'{name}.png', dpi=300)
        fig.savefig(FIG / f'{name}.pdf')


def plot_sst_distribution(df: pd.DataFrame) -> None:
    df = normalize_methods(df)
    fig, ax = plt.subplots(figsize=(10.6, 6.1))

    sns.violinplot(
        data=df,
        x='Method',
        y='SST (h)',
        order=ORDER,
        inner=None,
        cut=0,
        bw_adjust=0.8,
        linewidth=1.0,
        saturation=1.0,
        ax=ax,
    )

    violin_collections = list(ax.collections)
    for coll, method in zip(violin_collections, ORDER):
        coll.set_facecolor(PALETTE[method])
        coll.set_alpha(0.40)
        coll.set_edgecolor('#444444')
        coll.set_linewidth(0.9)

    sns.boxplot(
        data=df,
        x='Method',
        y='SST (h)',
        order=ORDER,
        width=0.16,
        showcaps=True,
        showfliers=False,
        boxprops=dict(facecolor='white', edgecolor='#222222', alpha=0.90, linewidth=1.1),
        whiskerprops=dict(color='#222222', linewidth=1.0),
        capprops=dict(color='#222222', linewidth=1.0),
        medianprops=dict(color='#111111', linewidth=1.8),
        zorder=3,
        ax=ax,
    )

    sns.stripplot(
        data=df,
        x='Method',
        y='SST (h)',
        order=ORDER,
        color='#111111',
        alpha=0.38,
        size=3.0,
        jitter=0.16,
        zorder=2,
        ax=ax,
    )

    ax.set_xticks(range(len(ORDER)), [DISPLAY_LABELS[m] for m in ORDER])
    ax.set_xlabel('')
    ax.set_ylabel('System Survival Time (h)')
    ax.set_title('Paired benchmark: SST distribution across defence strategies', fontweight='bold')
    ax.grid(True, axis='y', alpha=0.25)
    ax.set_axisbelow(True)
    ax.set_ylim(0.0, max(20.0, float(df['SST (h)'].max()) * 1.05))
    sns.despine(ax=ax, left=False, bottom=False)

    plt.tight_layout()
    save_with_aliases(fig, 'fig_sst_violin', aliases=['fig5_sst_violin'])
    plt.close(fig)


def plot_loss_decomposition(df: pd.DataFrame) -> None:
    df = normalize_methods(df)
    summary = (
        df.groupby('Method')[['Throttled Load (%)', 'Dropped Load (%)', 'Natural Loss (%)']]
        .mean()
        .reindex(ORDER)
        .fillna(0.0)
    )
    fig, ax = plt.subplots(figsize=(10, 5.8))
    summary[['Throttled Load (%)', 'Dropped Load (%)', 'Natural Loss (%)']].plot(
        kind='bar',
        stacked=True,
        ax=ax,
        color=['#4C78A8', '#F58518', '#E45756'],
    )
    ax.set_ylabel('Mean percentage')
    ax.set_title('Intervention and loss decomposition')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha='right')
    ax.grid(True, axis='y', alpha=0.25)
    fig.tight_layout()
    save_with_aliases(fig, 'fig_loss_decomposition')
    plt.close(fig)


def plot_paired_gain(df: pd.DataFrame) -> None:
    df = normalize_methods(df)
    gat = df[df['Method'] == 'GAT-SAC (Ours)'].sort_values('Scenario')
    base = df[df['Method'] == 'No Defence'].sort_values('Scenario')
    delta = gat['SST (h)'].to_numpy() - base['SST (h)'].to_numpy()
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.axhline(0.0, color='black', linewidth=1)
    ax.plot(gat['Scenario'], delta, marker='o')
    ax.set_xlabel('Scenario')
    ax.set_ylabel('GAT-SAC - No Defence (h)')
    ax.set_title('Paired scenario gain relative to No Defence')
    fig.tight_layout()
    save_with_aliases(fig, 'fig_paired_gain_vs_nodefence')
    plt.close(fig)


def plot_phi_sensitivity(df: pd.DataFrame) -> None:
    df = normalize_methods(df)
    grouped = (
        df.groupby(['Method', 'Drop Fraction'])['SST (h)']
        .mean()
        .reset_index()
        .sort_values(['Method', 'Drop Fraction'])
    )
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    for method, color in SENSITIVITY_SERIES:
        subset = grouped[grouped['Method'] == method]
        if subset.empty:
            continue
        ax.plot(
            subset['Drop Fraction'],
            subset['SST (h)'],
            marker='o',
            linewidth=2.4,
            label=method,
            color=color,
        )
    ax.set_xlabel('Drop fraction $\\phi$')
    ax.set_ylabel('Mean SST (h)')
    ax.set_title('Fixed-policy sensitivity to drop-fraction physics')
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    save_with_aliases(fig, 'fig8_sensitivity')
    plt.close(fig)


def main() -> None:
    paired = pd.read_csv(ART / 'paired_eval_results.csv')
    plot_sst_distribution(paired)
    plot_loss_decomposition(paired)
    plot_paired_gain(paired)

    sensitivity_path = ART / 'sensitivity_fresh.csv'
    if sensitivity_path.exists():
        sensitivity = pd.read_csv(sensitivity_path)
        plot_phi_sensitivity(sensitivity)


if __name__ == '__main__':
    main()
