import sys
import json
import warnings
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
warnings.filterwarnings('ignore')

OUT_DIR = PROJECT_ROOT / 'models'
OUT_DIR.mkdir(exist_ok=True)
RANDOM_STATE = 42
N_SPLITS = 5


def _load_train_module():
    spec = importlib.util.spec_from_file_location(
        'train08', str(PROJECT_ROOT / 'scripts' / '08_train_model.py'))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def oof_predict(model_ctor, X, y, groups):
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    skf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof = np.full(len(y), np.nan)
    for tr, te in skf.split(X, y, groups):
        pipe = Pipeline([('imp', SimpleImputer(strategy='median')), ('clf', model_ctor())])
        pipe.fit(X[tr], y[tr])
        oof[te] = pipe.predict_proba(X[te])[:, 1]
    return oof


def watch_list_table(y, proba):
    order = np.argsort(-proba)
    y_sorted = y[order]
    n = len(y)
    total_pos = int(y.sum())
    base = total_pos / n
    rows = []
    for pct in [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]:
        k = max(1, int(round(n * pct / 100)))
        caught = int(y_sorted[:k].sum())
        recall = caught / total_pos if total_pos else 0.0
        prec = caught / k
        lift = (prec / base) if base else 0.0
        rows.append({'top_pct': pct, 'n_lakes': k, 'pos_caught': caught,
                     'recall': round(recall, 3), 'lift': round(lift, 2)})
    return rows


def threshold_table(y, proba):
    total_pos = int(y.sum())
    n = len(y)
    rows = []
    for t in [0.20, 0.25, 0.30, 0.40, 0.50, 0.60]:
        flagged = proba >= t
        nf = int(flagged.sum())
        caught = int(y[flagged].sum())
        rows.append({'threshold': t, 'pct_flagged': round(100 * nf / n, 1),
                     'recall': round(caught / total_pos, 3) if total_pos else 0.0,
                     'pos_caught': caught})
    return rows


def youden_threshold(y, proba):
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y, proba)
    j = tpr - fpr
    return float(thr[np.argmax(j)])


def mann_whitney(df, feature_cols, y):
    pos = df[y == 1]
    neg = df[y == 0]
    out = []
    for c in feature_cols:
        a = pos[c].dropna().values
        b = neg[c].dropna().values
        if len(a) < 3 or len(b) < 3:
            continue
        u, p = stats.mannwhitneyu(a, b, alternative='two-sided')
        rb = 1 - 2 * u / (len(a) * len(b))
        out.append({'feature': c, 'median_glof': float(np.median(a)),
                    'median_nonglof': float(np.median(b)),
                    'U': float(u), 'p_value': float(p),
                    'rank_biserial': round(float(rb), 3), 'n_pos': len(a)})
    out.sort(key=lambda r: r['p_value'])
    m = len(out)
    for r in out:
        r['p_bonferroni'] = min(1.0, r['p_value'] * m)
        r['significant'] = r['p_bonferroni'] < 0.05
    return out


def calibration(y, proba):
    from sklearn.metrics import brier_score_loss
    from sklearn.model_selection import StratifiedKFold
    from sklearn.isotonic import IsotonicRegression
    base = y.mean()
    brier = brier_score_loss(y, proba)
    brier_ref = brier_score_loss(y, np.full_like(proba, base))
    bss = 1 - brier / brier_ref if brier_ref else 0.0
    cal = np.full(len(y), np.nan)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    for tr, te in skf.split(proba.reshape(-1, 1), y):
        ir = IsotonicRegression(out_of_bounds='clip')
        ir.fit(proba[tr], y[tr])
        cal[te] = ir.predict(proba[te])
    cal = np.nan_to_num(cal, nan=base)
    brier_cal = brier_score_loss(y, cal)
    bss_cal = 1 - brier_cal / brier_ref if brier_ref else 0.0
    return {'brier': round(float(brier), 5), 'brier_ref': round(float(brier_ref), 5),
            'brier_skill_score': round(float(bss), 4), 'base_rate': round(float(base), 5),
            'brier_calibrated': round(float(brier_cal), 5),
            'brier_skill_score_calibrated': round(float(bss_cal), 4)}


def ks_test(y, proba):
    d, p = stats.ks_2samp(proba[y == 1], proba[y == 0])
    return {'ks_D': round(float(d), 4), 'p_value': float(p),
            'median_score_glof': round(float(np.median(proba[y == 1])), 3),
            'median_score_nonglof': round(float(np.median(proba[y == 0])), 3)}


def morans_i(lon, lat, values, k_dist_km=55.0, max_n=2500, seed=RANDOM_STATE):
    rng = np.random.RandomState(seed)
    n = len(values)
    if n > max_n:
        idx = rng.choice(n, max_n, replace=False)
        lon, lat, values = lon[idx], lat[idx], values[idx]
        n = max_n
    x = np.radians(np.c_[lat, lon])
    R = 6371.0
    lat1 = x[:, 0][:, None]; lat2 = x[:, 0][None, :]
    dlon = x[:, 1][:, None] - x[:, 1][None, :]
    dlat = lat1 - lat2
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    dist = 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    W = (dist <= k_dist_km).astype(float)
    np.fill_diagonal(W, 0.0)
    z = values - values.mean()
    s0 = W.sum()
    if s0 == 0:
        return {'morans_I': None}
    num = n * (z[:, None] * z[None, :] * W).sum()
    den = s0 * (z ** 2).sum()
    I = num / den if den else 0.0
    return {'morans_I': round(float(I), 4), 'n_used': int(n), 'dist_km': k_dist_km}


def main():
    print("=== Paper stats (new model) ===")
    m = _load_train_module()
    lakes, X, y, groups, feature_cols = m.load_data()
    print(f"  lakes={len(lakes)}  pos={int(y.sum())}  features={len(feature_cols)}")

    def ctor():
        from imblearn.ensemble import BalancedRandomForestClassifier
        return BalancedRandomForestClassifier(
            n_estimators=500, max_features='sqrt', random_state=RANDOM_STATE, n_jobs=-1)

    print("  computing OOF probabilities (BalancedRF, operational)...")
    oof = oof_predict(ctor, X, y, groups)

    t_youden = youden_threshold(y, oof)
    results = {
        'n_lakes': int(len(lakes)), 'n_pos': int(y.sum()), 'n_neg': int((y == 0).sum()),
        'youden_threshold': round(t_youden, 4),
        'watch_list': watch_list_table(y, oof),
        'threshold_sensitivity': threshold_table(y, oof),
        'calibration': calibration(y, oof),
        'ks_test': ks_test(y, oof),
        'mann_whitney': mann_whitney(lakes.assign(**{c: X[:, i] for i, c in enumerate(feature_cols)}),
                                     feature_cols, y),
    }

    if 'lake_key' in lakes.columns:
        parts = lakes['lake_key'].str.rsplit('_', n=2, expand=True)
        lon = pd.to_numeric(parts[1], errors='coerce').values / 1000.0
        lat = pd.to_numeric(parts[2], errors='coerce').values / 1000.0
        ok = np.isfinite(lon) & np.isfinite(lat)
        results['morans_I'] = morans_i(lon[ok], lat[ok], oof[ok])

    flagged = oof >= t_youden
    results['pct_high_risk_overall'] = round(100 * flagged.mean(), 1)
    if 'area_name' in lakes.columns:
        per = lakes.assign(_f=flagged).groupby('area_name')['_f'].agg(['mean', 'count'])
        results['per_area_high_risk'] = {a: {'pct_high_risk': round(100 * r['mean'], 1),
                                             'n_lakes': int(r['count'])}
                                         for a, r in per.iterrows()}

    with open(OUT_DIR / 'paper_stats.json', 'w') as f:
        json.dump(results, f, indent=2)
    pd.DataFrame(results['watch_list']).to_csv(OUT_DIR / 'watch_list.csv', index=False)
    pd.DataFrame(results['threshold_sensitivity']).to_csv(OUT_DIR / 'threshold_sensitivity.csv', index=False)
    pd.DataFrame(results['mann_whitney']).to_csv(OUT_DIR / 'mann_whitney.csv', index=False)

    print(f"\n  Youden t={t_youden:.3f}  high-risk overall={results['pct_high_risk_overall']}%")
    print("  Watch-list:")
    for r in results['watch_list']:
        print(f"    top {r['top_pct']:4}% : {r['n_lakes']:5} lakes, recall {r['recall']}, lift {r['lift']}x")
    print(f"  Calibration BSS={results['calibration']['brier_skill_score']}  KS D={results['ks_test']['ks_D']} p={results['ks_test']['p_value']:.4f}")
    print(f"  Moran's I={results.get('morans_I', {}).get('morans_I')}")
    print("  Top Mann-Whitney features:")
    for r in results['mann_whitney'][:6]:
        print(f"    {r['feature']:16} GLOF {r['median_glof']:.1f} vs {r['median_nonglof']:.1f}  p={r['p_value']:.4f} {'*' if r['significant'] else ''}")
    print("\n  Saved: paper_stats.json + watch_list.csv + threshold_sensitivity.csv + mann_whitney.csv")


if __name__ == '__main__':
    main()
