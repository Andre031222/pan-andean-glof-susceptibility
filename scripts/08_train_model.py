import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
warnings.filterwarnings('ignore')

TRAINING_FILE = PROJECT_ROOT / 'data' / 'processed' / 'labeled' / 'training_data.csv'
MODELS_DIR = PROJECT_ROOT / 'models'
FIGURES_DIR = PROJECT_ROOT / 'figures' / 'publication'

EXCLUDE_COLS = {
    'lake_id', 'area_name', 'year', 'scene_date', 'source', 'satellite',
    'geometry', 'glof', 'matched_area', '_country',
}
TARGET_COL = 'glof'
MIN_LAKE_AREA_M2 = 50000.0
RANDOM_STATE = 42
N_SPLITS = 5
N_BOOTSTRAP = 2000
N_PERMUTATION = 5000


def load_data():
    import geopandas as gpd
    df = pd.read_csv(TRAINING_FILE)
    print(f"  Loaded {len(df)} lake-year rows")

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found")

    if 'area_m2' in df.columns:
        before = len(df)
        keep = (df['area_m2'] >= MIN_LAKE_AREA_M2) | (df[TARGET_COL] == 1)
        df = df[keep].copy()
        print(f"  min-area filter (>= {MIN_LAKE_AREA_M2:.0f} m2, positives kept): {before} -> {len(df)}")

    gpkg = PROJECT_ROOT / 'data' / 'processed' / 'features' / 'lake_features.gpkg'
    g = gpd.read_file(gpkg, columns=['lake_id', 'geometry']).to_crs(4326)
    cent = g.geometry.centroid
    g['_cx'] = (cent.x * 1000).round(0)
    g['_cy'] = (cent.y * 1000).round(0)
    df = df.merge(g[['lake_id', '_cx', '_cy']], on='lake_id', how='left')
    df['lake_key'] = (df['area_name'].astype(str) + '_' +
                      df['_cx'].astype('Int64').astype(str) + '_' +
                      df['_cy'].astype('Int64').astype(str))

    feature_cols = [
        c for c in df.columns
        if c not in EXCLUDE_COLS and c not in ('_cx', '_cy', 'lake_key')
        and df[c].dtype in (float, int, np.float64, np.int64)
    ]
    print(f"  Features ({len(feature_cols)}): {feature_cols}")

    agg = {c: 'mean' for c in feature_cols}
    agg[TARGET_COL] = 'max'
    agg['area_name'] = 'first'
    lakes = df.groupby('lake_key', as_index=False).agg(agg)
    print(f"  Aggregated to {len(lakes)} unique lakes  (pos={int(lakes[TARGET_COL].sum())}, "
          f"neg={int((lakes[TARGET_COL] == 0).sum())})")

    X = lakes[feature_cols].values.astype(np.float64)
    y = lakes[TARGET_COL].values.astype(int)
    groups = lakes['lake_key'].values

    return lakes, X, y, groups, feature_cols


def build_models() -> dict:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer

    models = {}

    try:
        from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier
        models['BalancedRF'] = BalancedRandomForestClassifier(
            n_estimators=500, max_features='sqrt', random_state=RANDOM_STATE, n_jobs=-1)
        models['EasyEnsemble'] = EasyEnsembleClassifier(
            n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    except ImportError:
        print("  [WARN] imbalanced-learn not available")

    models['RandomForest'] = RandomForestClassifier(
        n_estimators=500, max_features='sqrt', class_weight='balanced',
        random_state=RANDOM_STATE, n_jobs=-1)

    try:
        from xgboost import XGBClassifier
        models['XGBoost'] = XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            scale_pos_weight=10, random_state=RANDOM_STATE,
            eval_metric='logloss', verbosity=0)
    except ImportError:
        print("  [WARN] xgboost not available")

    try:
        from lightgbm import LGBMClassifier
        models['LightGBM'] = LGBMClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            class_weight='balanced', random_state=RANDOM_STATE, verbose=-1)
    except ImportError:
        print("  [WARN] lightgbm not available")

    models['LogisticRegression'] = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(C=0.1, class_weight='balanced',
                                   max_iter=1000, random_state=RANDOM_STATE)),
    ])

    return models


def youden_threshold(fpr: np.ndarray, tpr: np.ndarray, thresholds: np.ndarray) -> float:
    return float(thresholds[np.argmax(tpr - fpr)])


def run_cv(model, X: np.ndarray, y: np.ndarray, groups: np.ndarray, feature_cols: list) -> dict:
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.metrics import (roc_auc_score, average_precision_score,
                                  matthews_corrcoef, recall_score, f1_score, roc_curve)
    from sklearn.impute import SimpleImputer

    try:
        from imblearn.combine import SMOTETomek
        use_smote = True
    except ImportError:
        use_smote = False

    skf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    imputer = SimpleImputer(strategy='median')

    oof_proba = np.zeros(len(y))
    fold_aucs = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y, groups)):
        X_tr = imputer.fit_transform(X[train_idx])
        X_val = imputer.transform(X[val_idx])
        y_tr = y[train_idx].copy()
        y_val = y[val_idx]

        if use_smote and not hasattr(model, 'steps'):
            model_name = type(model).__name__
            if 'Balanced' not in model_name and 'Easy' not in model_name:
                try:
                    smt = SMOTETomek(random_state=RANDOM_STATE)
                    X_tr, y_tr = smt.fit_resample(X_tr, y_tr)
                except Exception:
                    pass

        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_val)[:, 1]
        oof_proba[val_idx] = proba

        if y_val.sum() > 0:
            fold_aucs.append(roc_auc_score(y_val, proba))

    auc_roc = roc_auc_score(y, oof_proba) if y.sum() > 0 else np.nan
    auc_pr = average_precision_score(y, oof_proba) if y.sum() > 0 else np.nan

    fpr, tpr, thresholds = roc_curve(y, oof_proba)
    threshold = youden_threshold(fpr, tpr, thresholds)
    y_pred = (oof_proba >= threshold).astype(int)

    recall = recall_score(y, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y, y_pred)
    f1 = f1_score(y, y_pred, zero_division=0)

    n_top = max(1, int(0.1 * len(oof_proba)))
    top_idx = np.argsort(oof_proba)[::-1][:n_top]
    lift = float(y[top_idx].mean() / y.mean()) if y.mean() > 0 else np.nan

    return {
        'auc_roc': auc_roc,
        'auc_pr': auc_pr,
        'recall': recall,
        'mcc': mcc,
        'f1': f1,
        'lift_10pct': lift,
        'threshold': threshold,
        'oof_proba': oof_proba,
        'fold_aucs': fold_aucs,
    }


def bootstrap_ci(y_true: np.ndarray, y_score: np.ndarray, n_reps: int = N_BOOTSTRAP) -> dict:
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(RANDOM_STATE)
    aucs = []
    for _ in range(n_reps):
        idx = rng.integers(0, len(y_true), size=len(y_true))
        if y_true[idx].sum() == 0:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_score[idx]))
    aucs = np.array(aucs)
    return {
        'mean': float(np.mean(aucs)),
        'ci_lower': float(np.percentile(aucs, 2.5)),
        'ci_upper': float(np.percentile(aucs, 97.5)),
    }


def permutation_test(y_true: np.ndarray, y_score: np.ndarray, n_reps: int = N_PERMUTATION) -> float:
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(RANDOM_STATE)
    obs_auc = roc_auc_score(y_true, y_score)
    null_aucs = [roc_auc_score(rng.permutation(y_true), y_score) for _ in range(n_reps)]
    return float(np.mean(np.array(null_aucs) >= obs_auc))


def jackknife_glof(model, X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> list:
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='median')
    glof_events = np.unique(groups[y == 1])
    results = []

    for event in glof_events:
        event_mask = (groups == event) & (y == 1)
        train_mask = ~event_mask

        if event_mask.sum() == 0:
            continue

        X_tr = imputer.fit_transform(X[train_mask])
        X_te = imputer.transform(X[event_mask])
        y_tr = y[train_mask]

        try:
            model.fit(X_tr, y_tr)
            proba = model.predict_proba(X_te)[:, 1]
            results.append({'event': str(event), 'n_test': int(event_mask.sum()),
                            'mean_proba': float(np.mean(proba))})
        except Exception as e:
            results.append({'event': str(event), 'n_test': int(event_mask.sum()),
                            'error': str(e)})

    return results


def loco_validation(model, df: pd.DataFrame, X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import roc_auc_score, average_precision_score
    imputer = SimpleImputer(strategy='median')

    def area_to_country(area: str) -> str:
        if 'chile' in area or 'patagonia' in area:
            return 'Chile'
        if 'bolivia' in area or 'apolobamba' in area:
            return 'Bolivia'
        if 'ecuador' in area:
            return 'Ecuador'
        return 'Peru'

    df = df.copy()
    df['_country'] = df['area_name'].apply(area_to_country)
    results = []

    for country in df['_country'].unique():
        test_mask = (df['_country'] == country).values
        train_mask = ~test_mask

        if y[test_mask].sum() == 0:
            print(f"    [skip LOCO] {country} — no positives in test set")
            continue

        X_tr = imputer.fit_transform(X[train_mask])
        X_te = imputer.transform(X[test_mask])
        y_tr = y[train_mask]
        y_te = y[test_mask]

        try:
            model.fit(X_tr, y_tr)
            proba = model.predict_proba(X_te)[:, 1]
            results.append({
                'country': country,
                'n_test': int(test_mask.sum()),
                'n_pos': int(y_te.sum()),
                'auc_roc': roc_auc_score(y_te, proba) if y_te.sum() > 0 else np.nan,
                'auc_pr': average_precision_score(y_te, proba) if y_te.sum() > 0 else np.nan,
            })
        except Exception as e:
            results.append({'country': country, 'error': str(e)})

    return pd.DataFrame(results)


def delong_test(y_true: np.ndarray, proba_a: np.ndarray, proba_b: np.ndarray) -> dict:
    from scipy.stats import norm
    from sklearn.metrics import roc_auc_score

    auc_a = roc_auc_score(y_true, proba_a)
    auc_b = roc_auc_score(y_true, proba_b)
    diff = auc_a - auc_b
    n1 = int(y_true.sum())
    n0 = int((1 - y_true).sum())
    se = np.sqrt(auc_a * (1 - auc_a) / n1 + auc_b * (1 - auc_b) / n0 + 1e-10)
    z = diff / se
    p_val = 2 * (1 - norm.cdf(abs(z)))

    return {
        'auc_a': float(auc_a),
        'auc_b': float(auc_b),
        'diff': float(diff),
        'z': float(z),
        'p_value': float(p_val),
    }


def compute_shap(model, X: np.ndarray, feature_cols: list, out_dir: Path):
    try:
        import shap
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [WARN] shap not installed")
        return

    clf = model.steps[-1][1] if hasattr(model, 'steps') else model
    X_shap = model[:-1].transform(X) if hasattr(model, 'steps') and len(model.steps) > 1 else X

    if X_shap.shape[0] > 2000:
        idx = np.random.RandomState(RANDOM_STATE).choice(X_shap.shape[0], 2000, replace=False)
        X_shap = X_shap[idx]

    try:
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_shap)
        if isinstance(shap_values, list):
            sv = shap_values[1]
        elif getattr(shap_values, 'ndim', 2) == 3:
            sv = shap_values[:, :, 1]
        else:
            sv = shap_values

        fig, ax = plt.subplots(figsize=(10, 7))
        shap.summary_plot(sv, X_shap, feature_names=feature_cols, show=False)
        plt.tight_layout()
        fig.savefig(out_dir / 'fig05_shap_beeswarm.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

        mean_abs = np.abs(sv).mean(0)
        feat_imp = pd.DataFrame({'feature': feature_cols, 'importance': mean_abs})
        feat_imp = feat_imp.sort_values('importance', ascending=False).head(20)
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh(feat_imp['feature'].values[::-1], feat_imp['importance'].values[::-1])
        ax.set_xlabel('Mean |SHAP value|')
        ax.set_title('SHAP Feature Importance (BRF)')
        plt.tight_layout()
        fig.savefig(out_dir / 'fig05_shap_bar.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  SHAP saved to {out_dir}")
    except Exception as e:
        print(f"  [WARN] SHAP: {e}")


def main():
    print("=== Model Training Pipeline ===")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if not TRAINING_FILE.exists():
        print(f"[ERROR] {TRAINING_FILE} not found. Run 07_match_glofs.py first.")
        sys.exit(1)

    print("\n1. Loading training data...")
    df, X, y, groups, feature_cols = load_data()
    print(f"   n={len(y)}  n+={y.sum()}  n-={(1-y).sum()}")

    if y.sum() < 5:
        print("[ERROR] Too few positives (< 5). Cannot train.")
        sys.exit(1)

    print("\n2. Building models...")
    models = build_models()
    print(f"   Models: {list(models.keys())}")

    print("\n3. Cross-validation (StratifiedGroupKFold, groups=lake_id)...")
    cv_results = {}
    for name, model in models.items():
        print(f"   {name}...", end=' ', flush=True)
        try:
            res = run_cv(model, X, y, groups, feature_cols)
            cv_results[name] = res
            print(f"AUC={res['auc_roc']:.3f}  recall={res['recall']:.3f}  lift={res['lift_10pct']:.1f}x")
        except Exception as e:
            print(f"FAILED: {e}")

    if not cv_results:
        print("[ERROR] All models failed.")
        sys.exit(1)

    best_name = max(cv_results, key=lambda k: cv_results[k]['auc_roc'])
    best_model = models[best_name]
    best_proba = cv_results[best_name]['oof_proba']
    print(f"\n   Best model: {best_name}  AUC={cv_results[best_name]['auc_roc']:.3f}")

    print("\n4. Bootstrap CI (2000 reps)...")
    ci = bootstrap_ci(y, best_proba, N_BOOTSTRAP)
    print(f"   AUC={ci['mean']:.3f}  95%CI [{ci['ci_lower']:.3f}, {ci['ci_upper']:.3f}]")

    print(f"\n5. Permutation test ({N_PERMUTATION} reps)...")
    p_val = permutation_test(y, best_proba, N_PERMUTATION)
    print(f"   p={p_val:.4f}")

    print("\n6. Jackknife (leave-one-GLOF-out)...")
    jk = jackknife_glof(best_model, X, y, groups)
    mean_proba_glof = np.mean([r['mean_proba'] for r in jk if 'mean_proba' in r])
    print(f"   {len(jk)} GLOF events  mean proba = {mean_proba_glof:.3f}")

    print("\n7. LOCO validation (Leave-One-Country-Out)...")
    loco_df = loco_validation(best_model, df, X, y)
    print(loco_df.to_string(index=False))

    print("\n8. DeLong test (BRF vs EasyEnsemble)...")
    delong_result = {}
    if 'BalancedRF' in cv_results and 'EasyEnsemble' in cv_results:
        delong_result = delong_test(
            y,
            cv_results['BalancedRF']['oof_proba'],
            cv_results['EasyEnsemble']['oof_proba'],
        )
        print(f"   AUC_BRF={delong_result['auc_a']:.3f}  AUC_EE={delong_result['auc_b']:.3f}  "
              f"z={delong_result['z']:.2f}  p={delong_result['p_value']:.4f}")

    print("\n9. SHAP analysis...")
    from sklearn.impute import SimpleImputer
    X_imp = SimpleImputer(strategy='median').fit_transform(X)
    best_model.fit(X_imp, y)
    compute_shap(best_model, X_imp, feature_cols, FIGURES_DIR)

    print("\n10. Saving results...")
    comparison_rows = []
    for name, res in cv_results.items():
        row = {
            'model': name,
            'auc_roc': round(res['auc_roc'], 4),
            'auc_pr': round(res['auc_pr'], 4),
            'recall': round(res['recall'], 4),
            'mcc': round(res['mcc'], 4),
            'f1': round(res['f1'], 4),
            'lift_10pct': round(res['lift_10pct'], 2),
            'threshold': round(res['threshold'], 3),
            'fold_auc_mean': round(np.mean(res['fold_aucs']), 4),
            'fold_auc_std': round(np.std(res['fold_aucs']), 4),
        }
        if name == best_name:
            row.update({
                'ci_lower': round(ci['ci_lower'], 4),
                'ci_upper': round(ci['ci_upper'], 4),
                'p_permutation': round(p_val, 4),
            })
        comparison_rows.append(row)

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(MODELS_DIR / 'model_comparison.csv', index=False)
    print(f"  Saved: models/model_comparison.csv")

    robustness = {
        'best_model': best_name,
        'bootstrap_ci': ci,
        'permutation_p': p_val,
        'jackknife': jk,
        'loco': loco_df.to_dict(orient='records'),
        'delong': delong_result,
    }
    with open(MODELS_DIR / 'robustness_results.json', 'w') as f:
        json.dump(robustness, f, indent=2, default=str)
    print(f"  Saved: models/robustness_results.json")

    import joblib
    joblib.dump(best_model, MODELS_DIR / 'best_model.joblib')
    print(f"  Saved: models/best_model.joblib ({best_name})")

    loco_df.to_csv(MODELS_DIR / 'loco_validation.csv', index=False)
    print(f"  Saved: models/loco_validation.csv")

    print(f"\n[done] Best={best_name}  AUC={cv_results[best_name]['auc_roc']:.3f}  "
          f"95%CI=[{ci['ci_lower']:.3f},{ci['ci_upper']:.3f}]  p={p_val:.4f}")


if __name__ == '__main__':
    main()
