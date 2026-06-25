import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IN = PROJECT_ROOT / 'docs' / 'inventory_validation_results.csv'
OUT = PROJECT_ROOT / 'models' / 'inventory_validation_summary.csv'


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_IN
    if not path.exists():
        print(f'No results file at {path}. Export the CSV from the review app first.')
        return
    df = pd.read_csv(path)
    df = df[df['is_real_glacial_lake'].isin(['Y', 'N'])].copy()
    if not len(df):
        print('No reviewed rows (is_real_glacial_lake empty).')
        return
    df['commission'] = (df['is_real_glacial_lake'] == 'N').astype(int)

    def block(sub, label):
        n = len(sub); k = int(sub['commission'].sum())
        lo, hi = wilson(k, n)
        print(f'{label:28s} n={n:4d}  commission={k:3d}  '
              f'rate={100*k/n:5.1f}%  95% CI [{100*lo:.1f}, {100*hi:.1f}]')
        return {'group': label, 'n': n, 'commission': k,
                'rate_pct': round(100 * k / n, 1),
                'ci_lo': round(100 * lo, 1), 'ci_hi': round(100 * hi, 1)}

    rows = []
    print('=== COMMISSION (manual validation vs satellite imagery) ===')
    rows.append(block(df, 'Overall'))
    rows.append(block(df[df.in_watchlist_top2pct == 1], 'Watch-list (top-2%)'))
    rows.append(block(df[df.in_watchlist_top2pct == 0], 'Non-watch-list'))
    print('--- by region ---')
    for area, sub in df.groupby('area_name'):
        rows.append(block(sub, area))
    print('--- by size ---')
    df['size_bin'] = pd.cut(df['area_ha'], [0, 1, 5, 20, 1e9],
                            labels=['<1ha', '1-5ha', '5-20ha', '>20ha'])
    for sb, sub in df.groupby('size_bin', observed=True):
        rows.append(block(sub, f'size {sb}'))

    if 'feature_type' in df.columns:
        print('--- commission feature types ---')
        ft = df[df.commission == 1]['feature_type'].value_counts()
        print(ft.to_string() if len(ft) else '(none)')

    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f'\nwrote {OUT}')
    wl = df[df.in_watchlist_top2pct == 1]
    if len(wl):
        k = int(wl['commission'].sum())
        print(f'\n>>> For the manuscript: of {len(wl)} reviewed watch-list lakes, '
              f'{k} ({100*k/len(wl):.1f}%) were commission; '
              f'{len(wl)-k} confirmed genuine glacial/proglacial lakes.')


if __name__ == '__main__':
    main()
