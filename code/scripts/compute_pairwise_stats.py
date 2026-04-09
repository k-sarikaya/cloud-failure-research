from pathlib import Path
import json
import pandas as pd
from scipy.stats import wilcoxon

CODE_ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = Path(__file__).resolve().parents[2]
ART = PKG_ROOT / 'artifacts'


def cohens_d_paired(x, y):
    diff = x - y
    sd = diff.std(ddof=1)
    return 0.0 if sd == 0 else float(diff.mean() / sd)


def holm_adjust(p_values):
    indexed = sorted(enumerate(float(p) for p in p_values), key=lambda x: x[1])
    m = len(indexed)
    adjusted = [0.0] * m
    running = 0.0
    for rank, (orig_idx, p_val) in enumerate(indexed, start=1):
        candidate = min((m - rank + 1) * p_val, 1.0)
        running = max(running, candidate)
        adjusted[orig_idx] = running
    return adjusted


def main():
    df = pd.read_csv(ART / 'paired_eval_results.csv')
    piv = df.pivot(index='Scenario', columns='Method', values='SST (h)')
    base = 'GAT-SAC (Ours)'
    rows = []
    for method in [c for c in piv.columns if c != base]:
        stat, p = wilcoxon(piv[base], piv[method])
        rows.append({
            'base_method': base,
            'compare_method': method,
            'mean_delta_h': float((piv[base] - piv[method]).mean()),
            'wilcoxon_stat': float(stat),
            'p_value': float(p),
            'paired_cohens_d': cohens_d_paired(piv[base].to_numpy(), piv[method].to_numpy()),
        })
    out = pd.DataFrame(rows).sort_values('p_value').reset_index(drop=True)
    out['holm_adjusted_p_value'] = holm_adjust(out['p_value'].tolist())
    out.to_csv(ART / 'pairwise_stats.csv', index=False)
    (ART / 'pairwise_stats.json').write_text(out.to_json(orient='records', indent=2), encoding='utf-8')
    print(out.to_string(index=False))


if __name__ == '__main__':
    main()
