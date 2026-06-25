import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT = PROJECT_ROOT / 'models'
RANDOM_STATE = 42


def load_train_module():
    spec = importlib.util.spec_from_file_location('t', PROJECT_ROOT / 'scripts' / '08_train_model.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def oof_scores(m):
    lakes, X, y, groups, fc = m.load_data()
    oof = np.full(len(y), np.nan)
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    for tr, te in skf.split(X, y, groups):
        p = Pipeline([('i', SimpleImputer(strategy='median')),
                      ('c', BalancedRandomForestClassifier(n_estimators=500, max_features='sqrt',
                                                           random_state=RANDOM_STATE, n_jobs=-1))])
        p.fit(X[tr], y[tr])
        oof[te] = p.predict_proba(X[te])[:, 1]
    lakes = lakes.copy()
    lakes['oof'] = oof
    lakes['y'] = y
    return lakes


def main():
    m = load_train_module()
    lakes = oof_scores(m)
    n = len(lakes)
    pos = lakes[lakes.y == 1]

    d = lakes['dist_glacier_m']
    e = lakes['elev_mean']
    dp, ep = pos['dist_glacier_m'], pos['elev_mean']

    d_thr = float(dp.quantile(0.95))
    e_lo = float(ep.quantile(0.05))
    shape_thr = float(lakes['elongation'].quantile(0.99)) if 'elongation' in lakes else np.inf

    off_ice = d > d_thr
    off_elev = e < e_lo
    off_shape = lakes['elongation'] > shape_thr if 'elongation' in lakes else pd.Series(False, index=lakes.index)
    off_context = off_ice | off_elev

    print('=== glacial-context envelope (from 32 GLOF-source lakes) ===')
    print(f'  dist_glacier_m: pos median {dp.median():.0f}, pos p95 {d_thr:.0f}, pop median {d.median():.0f}')
    print(f'  elev_mean: pos median {ep.median():.0f}, pos p05 {e_lo:.0f}, pop median {e.median():.0f}')
    print(f'  elongation p99 threshold: {shape_thr:.2f}')
    print()
    print('=== full inventory (n=%d) ===' % n)
    print(f'  off-ice (>{d_thr:.0f} m from glacier): {int(off_ice.sum())} ({100*off_ice.mean():.1f}%)')
    print(f'  off-elev (<{e_lo:.0f} m): {int(off_elev.sum())} ({100*off_elev.mean():.1f}%)')
    print(f'  off-context (either): {int(off_context.sum())} ({100*off_context.mean():.1f}%)')
    print(f'  extreme-elongation flag: {int(off_shape.sum())} ({100*off_shape.mean():.1f}%)')
    print(f'  positives off-context (should be ~0): {int(off_context[lakes.y==1].sum())} of {len(pos)}')
    print()
    print(f'  median OOF score in-context : {lakes.loc[~off_context, "oof"].median():.3f}')
    print(f'  median OOF score off-context: {lakes.loc[off_context, "oof"].median():.3f}')
    print()

    order = lakes['oof'].fillna(0).sort_values(ascending=False)
    rows = []
    for pct in (0.5, 1.0, 2.0, 5.0):
        k = max(1, int(round(n * pct / 100)))
        idx = order.index[:k]
        sub = lakes.loc[idx]
        oc = off_context.loc[idx]
        rows.append({'top_pct': pct, 'n_lakes': k,
                     'off_context': int(oc.sum()),
                     'off_context_pct': round(100 * oc.mean(), 1),
                     'pos_in_set': int(sub.y.sum())})
        print(f'=== top {pct}% watch-list (n={k}) ===')
        print(f'  off-context (likely commission / non-glacial): {int(oc.sum())} ({100*oc.mean():.1f}%)')
        print(f'  in plausible glacial context: {k-int(oc.sum())} ({100*(1-oc.mean()):.1f}%)')

    out = pd.DataFrame(rows)
    out.to_csv(OUT / 'inventory_screen.csv', index=False)
    print('\nwrote', OUT / 'inventory_screen.csv')


if __name__ == '__main__':
    main()
