import sys
import json
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_curve, precision_recall_curve, auc

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / 'models'
FIG_DIR = PROJECT_ROOT / 'figures' / 'publication'
FIG_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42
N_SPLITS = 5

plt.rcParams.update({
    'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
    'figure.dpi': 120, 'savefig.dpi': 300, 'axes.grid': True,
    'grid.alpha': 0.25, 'axes.axisbelow': True,
})
C_GLOF = '#c1272d'
C_NON = '#3b6ea5'


def load_train_module():
    spec = importlib.util.spec_from_file_location('train08', PROJECT_ROOT / 'scripts' / '08_train_model.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def brf_ctor():
    from imblearn.ensemble import BalancedRandomForestClassifier
    return BalancedRandomForestClassifier(
        n_estimators=500, max_features='sqrt', random_state=RANDOM_STATE, n_jobs=-1)


def rf_ctor():
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(
        n_estimators=500, max_features='sqrt', class_weight='balanced',
        random_state=RANDOM_STATE, n_jobs=-1)


def oof_predict(ctor, X, y, groups):
    skf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof = np.full(len(y), np.nan)
    for tr, te in skf.split(X, y, groups):
        pipe = Pipeline([('imp', SimpleImputer(strategy='median')), ('clf', ctor())])
        pipe.fit(X[tr], y[tr])
        oof[te] = pipe.predict_proba(X[te])[:, 1]
    return oof


def fig_distribution(y, oof, stats, path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    bins = np.linspace(0, 1, 41)
    ax[0].hist(oof[y == 0], bins=bins, color=C_NON, alpha=0.7, label='Non-source lakes', density=True)
    ax[0].hist(oof[y == 1], bins=bins, color=C_GLOF, alpha=0.8, label='GLOF-source lakes', density=True)
    t = stats['youden_threshold']
    ax[0].axvline(t, color='k', ls='--', lw=1.2, label=f'Youden t={t:.2f}')
    ax[0].set_xlabel('Susceptibility score')
    ax[0].set_ylabel('Density')
    ax[0].set_title('(a) Score distribution by class')
    ax[0].legend(fontsize=8, frameon=False)
    wl = pd.DataFrame(stats['watch_list'])
    ax[1].plot(wl['top_pct'], wl['lift'], 'o-', color=C_GLOF, lw=1.8)
    for _, r in wl.iterrows():
        ax[1].annotate(f"{r['lift']:.0f}x", (r['top_pct'], r['lift']), fontsize=7,
                       xytext=(3, 4), textcoords='offset points')
    ax[1].set_xscale('log')
    ax[1].set_xlabel('Watch-list size (% of inventory)')
    ax[1].set_ylabel('Enrichment (lift vs random)')
    ax[1].set_title('(b) Watch-list enrichment')
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


def fig_performance(y, oof, comp, path):
    fig = plt.figure(figsize=(12, 4.2))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1.15])
    fpr, tpr, _ = roc_curve(y, oof)
    ax0 = fig.add_subplot(gs[0])
    ax0.plot(fpr, tpr, color=C_GLOF, lw=2, label=f"AUC = {auc(fpr, tpr):.3f}")
    ax0.plot([0, 1], [0, 1], 'k--', lw=0.8, alpha=0.6)
    ax0.set_xlabel('False positive rate'); ax0.set_ylabel('True positive rate')
    ax0.set_title('(a) ROC (RandomForest, OOF)'); ax0.legend(fontsize=9, frameon=False)
    prec, rec, _ = precision_recall_curve(y, oof)
    ax1 = fig.add_subplot(gs[1])
    ax1.plot(rec, prec, color=C_NON, lw=2, label=f"AUC-PR = {auc(rec, prec):.3f}")
    ax1.axhline(y.mean(), color='k', ls='--', lw=0.8, alpha=0.6, label=f'Base rate = {y.mean():.4f}')
    ax1.set_xlabel('Recall'); ax1.set_ylabel('Precision')
    ax1.set_title('(b) Precision-Recall'); ax1.legend(fontsize=8, frameon=False)
    ax2 = fig.add_subplot(gs[2])
    comp = comp.sort_values('auc_roc')
    ypos = np.arange(len(comp))
    xerr = None
    if comp['ci_lower'].notna().any():
        xerr = np.zeros((2, len(comp)))
    bars = ax2.barh(ypos, comp['auc_roc'], color='#6c8ebf', alpha=0.85)
    best = comp['auc_roc'].idxmax()
    for i, (_, r) in enumerate(comp.iterrows()):
        if r.name == best:
            bars[i].set_color(C_GLOF)
        ax2.text(r['auc_roc'] + 0.005, i, f"{r['auc_roc']:.3f}", va='center', fontsize=8)
    ax2.set_yticks(ypos); ax2.set_yticklabels(comp['model'], fontsize=8)
    ax2.set_xlim(0.6, 0.9); ax2.set_xlabel('AUC-ROC (5-fold CV)')
    ax2.set_title('(c) Model comparison')
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


def fig_robustness(rob, loco, path):
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    boot = rob['bootstrap_ci']
    lo, hi, pt = boot['ci_lower'], boot['ci_upper'], boot['mean']
    sigma = (hi - lo) / (2 * 1.96)
    xs = np.linspace(pt - 4 * sigma, pt + 4 * sigma, 400)
    dens = np.exp(-0.5 * ((xs - pt) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    ax[0].plot(xs, dens, color=C_NON, lw=1.8)
    ax[0].fill_between(xs, dens, where=(xs >= lo) & (xs <= hi), color=C_GLOF, alpha=0.15,
                       label=f'95% CI [{lo:.3f}, {hi:.3f}]')
    ax[0].axvline(pt, color=C_GLOF, lw=2, label=f'AUC = {pt:.3f}')
    ax[0].axvline(0.5, color='k', ls='--', lw=0.8, alpha=0.6)
    ax[0].set_xlabel('Bootstrap AUC (2000 reps)'); ax[0].set_ylabel('Density')
    ax[0].set_title('(a) Bootstrap stability'); ax[0].legend(fontsize=8, frameon=False)
    ld = pd.DataFrame(loco)
    ld = ld.sort_values('auc_roc')
    ax[1].barh(np.arange(len(ld)), ld['auc_roc'], color='#6c8ebf', alpha=0.85)
    for i, (_, r) in enumerate(ld.iterrows()):
        ax[1].text(r['auc_roc'] + 0.01, i, f"{r['auc_roc']:.2f} (n+={int(r['n_pos'])})", va='center', fontsize=8)
    ax[1].axvline(0.5, color='k', ls='--', lw=0.8, alpha=0.6)
    ax[1].set_yticks(np.arange(len(ld))); ax[1].set_yticklabels(ld['country'], fontsize=9)
    ax[1].set_xlim(0, 1.05); ax[1].set_xlabel('AUC-ROC (held-out country)')
    ax[1].set_title('(b) Leave-one-country-out')
    jk = np.array([r['mean_proba'] for r in rob.get('jackknife', []) if r.get('mean_proba') is not None])
    if jk.size:
        ax[2].hist(jk, bins=20, color=C_GLOF, alpha=0.8)
        ax[2].axvline(np.mean(jk), color='k', lw=1.5, label=f'mean = {np.mean(jk):.2f}')
    ax[2].set_xlabel('Predicted probability (held-out GLOF)')
    ax[2].set_ylabel('Count')
    ax[2].set_title('(c) Jackknife (leave-one-GLOF-out)')
    ax[2].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


def main():
    print('=== Figures (new model) ===')
    m = load_train_module()
    lakes, X, y, groups, feature_cols = m.load_data()
    print(f'  lakes={len(lakes)} pos={int(y.sum())}')
    stats = json.load(open(MODELS_DIR / 'paper_stats.json'))
    comp = pd.read_csv(MODELS_DIR / 'model_comparison.csv')
    rob = json.load(open(MODELS_DIR / 'robustness_results.json'))
    loco = pd.read_csv(MODELS_DIR / 'loco_validation.csv').to_dict('records')

    print('  computing OOF (BalancedRF operational + RandomForest best)...')
    oof = oof_predict(brf_ctor, X, y, groups)
    oof_rf = oof_predict(rf_ctor, X, y, groups)

    fig_distribution(y, oof, stats, FIG_DIR / 'fig3_susceptibility_distribution.png')
    print('  fig3 done')
    fig_performance(y, oof_rf, comp, FIG_DIR / 'fig4_model_performance.png')
    print('  fig4 done')
    fig_robustness(rob, loco, FIG_DIR / 'fig6_robustness_analysis.png')
    print('  fig6 done')
    print('  (fig5 SHAP generated by 08_train_model.py)')


if __name__ == '__main__':
    main()
