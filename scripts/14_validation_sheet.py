import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT = PROJECT_ROOT / 'docs' / 'inventory_validation_sheet.csv'
RANDOM_STATE = 42
PER_AREA = 40
TOP_PCT = 2.0


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
    parts = lakes['lake_key'].str.rsplit('_', n=2, expand=True)
    lakes['lon'] = pd.to_numeric(parts[1], errors='coerce') / 1000.0
    lakes['lat'] = pd.to_numeric(parts[2], errors='coerce') / 1000.0

    n = len(lakes)
    k = max(1, int(round(n * TOP_PCT / 100)))
    thr = lakes['oof'].sort_values(ascending=False).iloc[k - 1]
    lakes['in_watchlist_top2pct'] = (lakes['oof'] >= thr).astype(int)

    rng = np.random.RandomState(RANDOM_STATE)
    sample_idx = []
    for area, grp in lakes.groupby('area_name'):
        take = min(PER_AREA, len(grp))
        sample_idx.extend(rng.choice(grp.index.values, size=take, replace=False))
    wl = lakes[lakes.in_watchlist_top2pct == 1].index.values
    pos = lakes[lakes.y == 1].index.values
    keep = sorted(set(sample_idx) | set(wl) | set(pos))
    s = lakes.loc[keep].copy()

    s['google_earth_url'] = ('https://earth.google.com/web/@' + s['lat'].round(5).astype(str)
                             + ',' + s['lon'].round(5).astype(str) + ',1000a,3000d,35y,0h,0t,0r')
    s['gmaps_url'] = ('https://www.google.com/maps/@' + s['lat'].round(5).astype(str)
                      + ',' + s['lon'].round(5).astype(str) + ',2000m/data=!3m1!1e3')

    out = s[['lake_key', 'area_name', 'lat', 'lon', 'area_m2', 'dist_glacier_m',
             'elev_mean', 'oof', 'in_watchlist_top2pct', 'y',
             'google_earth_url', 'gmaps_url']].copy()
    out = out.rename(columns={'oof': 'model_score', 'y': 'known_glof_source'})
    out['is_real_glacial_lake_YN'] = ''
    out['feature_type'] = ''
    out['confidence_1to3'] = ''
    out['notes'] = ''
    out = out.sort_values(['area_name', 'in_watchlist_top2pct', 'model_score'],
                          ascending=[True, False, False])
    out.to_csv(OUT, index=False)
    print(f'wrote {OUT}')
    print(f'  total lakes to review: {len(out)}')
    print(f'  watch-list lakes included: {int(out.in_watchlist_top2pct.sum())}')
    print(f'  known GLOF-source included: {int(out.known_glof_source.sum())}')
    print('  per-area counts:')
    print(out.groupby('area_name').size().to_string())


if __name__ == '__main__':
    main()
