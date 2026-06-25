"""
GLOF Andes Project — Módulo de Visualización para Publicación
=============================================================
Figuras de alta calidad para artículo científico (Nature Geoscience / Science Advances).

Paleta accesible (daltonismo): Tol's high-contrast + perceptually-uniform colormaps.
DPI: 150 en pantalla, 300 al guardar (TIFF/PDF).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
from pathlib import Path
from typing import Dict, List, Optional, Any

try:
    import geopandas as gpd
    import contextily as ctx
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False

try:
    import folium
    from folium.plugins import MarkerCluster
    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False

# =============================================================================
# PALETA CROMÁTICA (accesible, perceptualmente uniforme)
# =============================================================================
COLORS = {
    'blue':    '#0077BB',
    'orange':  '#EE7733',
    'cyan':    '#33BBEE',
    'magenta': '#EE3377',
    'red':     '#CC3311',
    'teal':    '#009988',
    'grey':    '#BBBBBB',
    'black':   '#000000',
    'yellow':  '#DDAA33',
    'indigo':  '#332288',
}

# Colormap GLOF: azul (seguro) → amarillo → rojo (peligroso)
GLOF_CMAP = LinearSegmentedColormap.from_list(
    'glof_risk',
    ['#2166AC', '#74ADD1', '#FEE090', '#F46D43', '#A50026'],
    N=256,
)

# Colormap para terreno (más contrastado que 'terrain' estándar)
TERRAIN_CMAP = LinearSegmentedColormap.from_list(
    'andean_terrain',
    ['#1a6b3b', '#5aac44', '#c8d45a', '#e8c46a', '#c89a50', '#a07040', '#7a5030', '#ffffff'],
    N=512,
)


def set_publication_style():
    """Configura matplotlib para figuras de publicación científica."""
    plt.rcParams.update({
        'figure.dpi':           150,
        'savefig.dpi':          300,
        'font.family':          'sans-serif',
        'font.sans-serif':      ['Helvetica Neue', 'Arial', 'DejaVu Sans'],
        'font.size':            10,
        'axes.titlesize':       11,
        'axes.labelsize':       10,
        'xtick.labelsize':      9,
        'ytick.labelsize':      9,
        'legend.fontsize':      9,
        'legend.framealpha':    0.9,
        'figure.figsize':       (8, 6),
        'axes.grid':            True,
        'grid.alpha':           0.2,
        'grid.linewidth':       0.5,
        'axes.spines.top':      False,
        'axes.spines.right':    False,
        'axes.linewidth':       0.8,
        'lines.linewidth':      1.5,
        'patch.linewidth':      0.8,
    })


# =============================================================================
# FIG 1 — ROC + Precision-Recall (panel doble compacto)
# =============================================================================

def plot_roc_curves(
    results: Dict[str, Any],
    y_test: np.ndarray,
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Curvas ROC y Precision-Recall para comparación de modelos.

    Panel izquierdo: ROC (AUC). Panel derecho: PR-AUC.
    """
    from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = [COLORS['blue'], COLORS['orange'], COLORS['teal'], COLORS['magenta'], COLORS['red']]

    for i, (name, res) in enumerate(results.items()):
        if 'y_prob' not in res:
            continue
        c      = colors[i % len(colors)]
        y_prob = res['y_prob']

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc     = auc(fpr, tpr)
        axes[0].plot(fpr, tpr, color=c, lw=2,
                     label=f'{name}  AUC={roc_auc:.3f}', zorder=3)

        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        pr_auc       = average_precision_score(y_test, y_prob)
        axes[1].plot(rec, prec, color=c, lw=2,
                     label=f'{name}  AP={pr_auc:.3f}', zorder=3)

    axes[0].fill_between([0, 1], [0, 1], alpha=0.06, color='grey')
    axes[0].plot([0, 1], [0, 1], 'k--', lw=0.8, label='Azar')
    axes[0].set(xlim=[0, 1], ylim=[0, 1.02],
                xlabel='Tasa de Falsos Positivos',
                ylabel='Tasa de Verdaderos Positivos')
    axes[0].set_title('(a) Curva ROC', fontweight='bold')
    axes[0].legend(loc='lower right', fontsize=8)

    baseline = y_test.mean()
    axes[1].axhline(baseline, color='grey', ls='--', lw=0.8,
                    label=f'Línea base ({baseline:.2f})')
    axes[1].set(xlim=[0, 1], ylim=[0, 1.02],
                xlabel='Recall', ylabel='Precision')
    axes[1].set_title('(b) Curva Precision-Recall', fontweight='bold')
    axes[1].legend(loc='upper right', fontsize=8)

    for ax in axes:
        ax.grid(True, alpha=0.2)
        ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Guardado: {output_path}")

    return fig


# =============================================================================
# FIG 2 — Model comparison: heatmap anotado + dot plot
# =============================================================================

def plot_model_comparison(
    comparison_df: pd.DataFrame,
    metrics: List[str] = None,
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Heatmap anotado de métricas + dot plot de ROC-AUC con barras de error.
    """
    if metrics is None:
        metrics = ['Test ROC-AUC', 'Test PR-AUC', 'Test F1', 'Test Recall', 'Test Precision']

    available = [m for m in metrics if m in comparison_df.columns]
    models    = comparison_df['Model'].tolist()

    fig = plt.figure(figsize=(14, 5))
    gs  = GridSpec(1, 2, figure=fig, width_ratios=[1.3, 1], wspace=0.35)

    ax1 = fig.add_subplot(gs[0])
    y   = np.arange(len(models))
    c   = [COLORS['blue'], COLORS['orange'], COLORS['teal'], COLORS['magenta']]

    if 'Test ROC-AUC' in comparison_df.columns:
        ax1.barh(y, comparison_df['Test ROC-AUC'],
                 color=[c[i % len(c)] for i in range(len(models))],
                 edgecolor='white', height=0.55, alpha=0.88)

        if 'CV ROC-AUC' in comparison_df.columns and 'CV Std' in comparison_df.columns:
            ax1.errorbar(comparison_df['CV ROC-AUC'], y,
                         xerr=comparison_df['CV Std'] * 1.96,
                         fmt='D', color='black', markersize=5,
                         capsize=4, zorder=5, label='CV ± 95% CI')

        ax1.axvline(0.5, color='grey', ls='--', lw=0.8, alpha=0.6)
        ax1.set_xlim(0.4, 1.02)
        ax1.set_yticks(y)
        ax1.set_yticklabels(models)
        ax1.set_xlabel('ROC-AUC')
        ax1.set_title('(a) ROC-AUC con IC 95%', fontweight='bold')
        ax1.legend(fontsize=8)

    ax2 = fig.add_subplot(gs[1])
    if available and len(models) > 0:
        data   = comparison_df[available].values.astype(float)
        normed = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0) + 1e-9)

        im = ax2.imshow(normed, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')

        for i in range(len(models)):
            for j in range(len(available)):
                val  = data[i, j]
                tcol = 'white' if normed[i, j] < 0.3 or normed[i, j] > 0.8 else 'black'
                ax2.text(j, i, f'{val:.3f}', ha='center', va='center',
                         fontsize=8.5, color=tcol, fontweight='bold')

        ax2.set_xticks(range(len(available)))
        ax2.set_xticklabels([m.replace('Test ', '') for m in available],
                             rotation=35, ha='right', fontsize=8.5)
        ax2.set_yticks(range(len(models)))
        ax2.set_yticklabels(models, fontsize=9)
        ax2.set_title('(b) Métricas — Heatmap', fontweight='bold')
        plt.colorbar(im, ax=ax2, fraction=0.04, pad=0.02, label='Normalizado')

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Guardado: {output_path}")

    return fig


# =============================================================================
# FIG 3 — SHAP summary: beeswarm + importancia media
# =============================================================================

def plot_shap_summary(
    shap_values: np.ndarray,
    features_df,
    feature_names: List[str],
    top_n: int = 15,
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """
    SHAP beeswarm (izq) + bar chart de importancia media (der).

    Eje cromático: valor bajo (azul) → valor alto (rojo).
    """
    abs_mean   = np.abs(shap_values).mean(axis=0)
    idx_sorted = np.argsort(abs_mean)[::-1][:top_n]
    sel_names  = [feature_names[i] for i in idx_sorted]
    sel_shap   = shap_values[:, idx_sorted]
    sel_vals   = features_df.iloc[:, idx_sorted].values if features_df is not None else None

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    cmap = plt.cm.RdBu_r

    ax = axes[0]
    for j, (name, col_shap) in enumerate(zip(reversed(sel_names),
                                              reversed(sel_shap.T))):
        y_jitter = j + np.random.uniform(-0.3, 0.3, size=len(col_shap))

        if sel_vals is not None:
            col_val  = sel_vals[:, top_n - j - 1].astype(float)
            norm_val = (col_val - col_val.min()) / (col_val.ptp() + 1e-9)
            colors   = cmap(norm_val)
        else:
            colors = COLORS['blue']

        ax.scatter(col_shap, y_jitter, c=colors, s=8, alpha=0.55, linewidths=0)

    ax.axvline(0, color='black', lw=0.8, ls='--')
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(list(reversed(sel_names)), fontsize=8.5)
    ax.set_xlabel('Valor SHAP  (impacto en predicción de GLOF)')
    ax.set_title('(a) SHAP Beeswarm', fontweight='bold')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(0, 1))
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label('Valor de característica\n(bajo → alto)', fontsize=8)
    cb.set_ticks([0, 0.5, 1])
    cb.set_ticklabels(['Bajo', 'Medio', 'Alto'])

    ax2   = axes[1]
    y_pos = np.arange(top_n)
    vals  = abs_mean[idx_sorted]
    cb2   = [COLORS['blue'] if v >= np.median(vals) else COLORS['cyan'] for v in vals]

    ax2.barh(y_pos, vals, color=cb2, edgecolor='white', height=0.65)
    for i, v in enumerate(vals):
        ax2.text(v + vals.max() * 0.01, i, f'{v:.4f}', va='center', fontsize=7.5)

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(sel_names, fontsize=8.5)
    ax2.set_xlabel('|SHAP| medio')
    ax2.set_title('(b) Importancia media SHAP', fontweight='bold')
    ax2.invert_yaxis()

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Guardado: {output_path}")

    return fig


# =============================================================================
# FIG 4 — Evolución temporal de lagunas (ribbon chart)
# =============================================================================

def plot_temporal_lake_evolution(
    area_time_df: pd.DataFrame,
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Ribbon chart de área total de lagunas por cordillera (2017–2025) + tasa dA/dt.
    """
    years  = [c for c in area_time_df.columns if isinstance(c, int)]
    areas  = area_time_df.index.tolist()
    acolors = list(COLORS.values())[:len(areas)]

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    ax = axes[0]
    ax.set_facecolor('#f8f9fa')
    for i, area in enumerate(areas):
        vals = [area_time_df.loc[area, yr] for yr in years if yr in area_time_df.columns]
        ax.fill_between(years[:len(vals)], 0, vals,
                        color=acolors[i % len(acolors)], alpha=0.42,
                        label=area.replace('_', ' ').title())
        ax.plot(years[:len(vals)], vals, color=acolors[i % len(acolors)], lw=1.5)

    ax.set_ylabel('Área total de lagunas (km²)')
    ax.set_title('(a) Evolución del área de lagunas glaciares por cordillera', fontweight='bold')
    ax.legend(loc='upper left', ncol=2, fontsize=8)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

    ax2 = axes[1]
    ax2.set_facecolor('#f8f9fa')
    ax2.axhline(0, color='black', lw=0.8, ls='--')

    total = area_time_df.sum(axis=0)
    total_arr = [total[yr] for yr in years if yr in total.index]
    if len(total_arr) > 1:
        delta      = np.diff(total_arr)
        mid_years  = [(years[i] + years[i + 1]) / 2 for i in range(len(years) - 1)]
        bar_colors = [COLORS['red'] if d > 0 else COLORS['blue'] for d in delta]
        ax2.bar(mid_years, delta, width=0.7, color=bar_colors, edgecolor='white', alpha=0.8)
        ax2.fill_between(mid_years, 0, delta, alpha=0.15, color='red')

    ax2.set_xlabel('Año')
    ax2.set_ylabel('ΔÁrea (km²/año)')
    ax2.set_title('(b) Tasa de cambio anual — dA/dt', fontweight='bold')
    ax2.set_xticks(years)
    ax2.set_xticklabels(years, rotation=45)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Guardado: {output_path}")

    return fig


# =============================================================================
# FIG 5 — Panel de hipótesis H1, H2, H3
# =============================================================================

def plot_hypothesis_panel(
    lakes_gdf,
    importance_df: Optional[pd.DataFrame] = None,
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Validación de las 3 hipótesis del artículo en un panel de 3 columnas.

    H1: ratio área/profundidad (violin + scatter)
    H2: tasa de crecimiento vs elevación (scatter bicolor)
    H3: distancia crítica glaciar-laguna (distribuciones solapadas)
    """
    from scipy import stats

    fig = plt.figure(figsize=(15, 5))
    gs  = GridSpec(1, 3, figure=fig, wspace=0.38)

    # ── H1: Violin + jitter ──
    ax1 = fig.add_subplot(gs[0])
    if 'area_depth_ratio' in lakes_gdf.columns and 'had_glof' in lakes_gdf.columns:
        no_glof = lakes_gdf[lakes_gdf['had_glof'] == 0]['area_depth_ratio'].dropna().values
        glof    = lakes_gdf[lakes_gdf['had_glof'] == 1]['area_depth_ratio'].dropna().values

        parts = ax1.violinplot([no_glof, glof], positions=[0, 1],
                               showmedians=True, showextrema=True)
        for body, col in zip(parts['bodies'], [COLORS['blue'], COLORS['red']]):
            body.set_facecolor(col)
            body.set_alpha(0.55)

        for j, (data, col) in enumerate([(no_glof, COLORS['blue']), (glof, COLORS['red'])]):
            jitter = np.random.uniform(-0.12, 0.12, size=len(data))
            ax1.scatter(j + jitter, data, s=6, color=col, alpha=0.3, zorder=2)

        if len(glof) > 1 and len(no_glof) > 1:
            _, pval = stats.mannwhitneyu(glof, no_glof, alternative='two-sided')
            sig = '***' if pval < 0.001 else ('**' if pval < 0.01 else ('*' if pval < 0.05 else 'n.s.'))
            y_max = max(no_glof.max(), glof.max()) * 1.08
            ax1.annotate('', xy=(1, y_max), xytext=(0, y_max),
                         arrowprops=dict(arrowstyle='-', color='black', lw=1.2))
            ax1.text(0.5, y_max * 1.04, sig, ha='center', fontsize=11)
            ax1.text(0.5, y_max * 0.93, f'p = {pval:.4f}', ha='center', fontsize=7.5)

        ax1.set_xticks([0, 1])
        ax1.set_xticklabels(['Sin GLOF', 'Con GLOF'], fontsize=9)
        ax1.set_ylabel('Ratio Área / Profundidad')
        ax1.set_title('H1: Umbral Área/Profundidad', fontweight='bold')

    # ── H2: Scatter growth_rate vs elevation ──
    ax2 = fig.add_subplot(gs[1])
    if 'growth_rate' in lakes_gdf.columns and 'had_glof' in lakes_gdf.columns:
        for label, col, marker in [(0, COLORS['blue'], 'o'), (1, COLORS['red'], '^')]:
            subset = lakes_gdf[lakes_gdf['had_glof'] == label]
            if 'elev_mean' in subset.columns:
                ax2.scatter(subset['growth_rate'], subset['elev_mean'],
                            c=col, s=20, alpha=0.55, marker=marker,
                            label='Sin GLOF' if label == 0 else 'Con GLOF', zorder=3)
        ax2.axvline(0, color='grey', ls=':', lw=0.8)
        ax2.set_xlabel('dA/dt (km²/año)')
        ax2.set_ylabel('Elevación (m s.n.m.)')
        ax2.set_title('H2: Tasa de Crecimiento', fontweight='bold')
        ax2.legend(fontsize=8, markerscale=1.5)

    # ── H3: Distancia glaciar-laguna ──
    ax3 = fig.add_subplot(gs[2])
    if 'dist_glacier_m' in lakes_gdf.columns and 'had_glof' in lakes_gdf.columns:
        no_glof_d = lakes_gdf[lakes_gdf['had_glof'] == 0]['dist_glacier_m'].dropna() / 1000
        glof_d    = lakes_gdf[lakes_gdf['had_glof'] == 1]['dist_glacier_m'].dropna() / 1000

        bins = np.linspace(0, 10, 25)
        ax3.hist(no_glof_d, bins=bins, color=COLORS['blue'], alpha=0.5,
                 density=True, label='Sin GLOF')
        ax3.hist(glof_d, bins=bins, color=COLORS['red'], alpha=0.65,
                 density=True, label='Con GLOF')
        ax3.axvspan(0.1, 0.5, alpha=0.12, color=COLORS['orange'],
                    label='Zona crítica\n(100–500 m)')
        ax3.set_xlabel('Distancia a glaciar (km)')
        ax3.set_ylabel('Densidad')
        ax3.set_title('H3: Distancia Crítica', fontweight='bold')
        ax3.legend(fontsize=8)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Guardado: {output_path}")

    return fig


# =============================================================================
# FIG 6 — Susceptibilidad espacial (hexbin + scatter)
# =============================================================================

def plot_spatial_susceptibility(
    lakes_gdf,
    susceptibility_col: str = 'glof_prob',
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Mapa de susceptibilidad GLOF: densidad espacial (hexbin) + probabilidad (scatter).
    """
    if not HAS_GEOPANDAS or lakes_gdf is None:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, 'Requiere datos GeoPandas', ha='center', va='center')
        return fig

    l_wgs = lakes_gdf.to_crs('EPSG:4326')
    x = l_wgs.geometry.centroid.x.values
    y = l_wgs.geometry.centroid.y.values

    fig, axes = plt.subplots(1, 2, figsize=(15, 8))

    ax = axes[0]
    ax.set_facecolor('#0d1b2a')
    hb = ax.hexbin(x, y, gridsize=30, cmap='hot_r', mincnt=1, alpha=0.9)
    plt.colorbar(hb, ax=ax, label='Lagunas por celda')
    ax.set_xlabel('Longitud')
    ax.set_ylabel('Latitud')
    ax.set_title('(a) Densidad de lagunas glaciares', fontweight='bold')

    ax2 = axes[1]
    ax2.set_facecolor('#0d1b2a')

    if susceptibility_col in l_wgs.columns:
        probs = l_wgs[susceptibility_col].fillna(0).values
        sizes = 15 + probs * 120
        sc = ax2.scatter(x, y, c=probs, s=sizes, cmap=GLOF_CMAP,
                         vmin=0, vmax=1, alpha=0.75, edgecolors='none')
        cb = plt.colorbar(sc, ax=ax2, fraction=0.03, pad=0.02)
        cb.set_label('Probabilidad GLOF')
    else:
        ax2.scatter(x, y, c=COLORS['cyan'], s=15, alpha=0.6)

    if 'had_glof' in l_wgs.columns:
        glof_pts = l_wgs[l_wgs['had_glof'] == 1]
        ax2.scatter(
            glof_pts.geometry.centroid.x, glof_pts.geometry.centroid.y,
            s=120, c=COLORS['yellow'], marker='*',
            edgecolors='white', linewidths=0.5, zorder=5, label='GLOF histórico',
        )
        ax2.legend(fontsize=8, facecolor='#1a2a3a', labelcolor='white')

    ax2.set_xlabel('Longitud')
    ax2.set_ylabel('Latitud')
    ax2.set_title('(b) Susceptibilidad GLOF — Modelo Ensamble', fontweight='bold')

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Guardado: {output_path}")

    return fig


# =============================================================================
# FIG 7 — Características de lagunas (ridge KDE plots)
# =============================================================================

def plot_lake_characteristics(
    lakes_gdf,
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Distribuciones morfométricas con KDE solapado para GLOF vs no-GLOF.
    """
    from scipy.stats import gaussian_kde

    fig = plt.figure(figsize=(14, 9))
    gs  = GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]

    columns_labels = [
        ('area_m2',        'Área de laguna (km²)',      1e-6,  '#0077BB'),
        ('elev_mean',      'Elevación (m s.n.m.)',       1.0,   '#009988'),
        ('depth_est_m',    'Profundidad estimada (m)',   1.0,   '#EE7733'),
        ('dist_glacier_m', 'Distancia a glaciar (km)',   1e-3,  '#CC3311'),
    ]
    glof_col = 'had_glof' if 'had_glof' in lakes_gdf.columns else None

    for ax, (col, label, scale, base_col) in zip(axes, columns_labels):
        if col not in lakes_gdf.columns:
            ax.text(0.5, 0.5, f'{col}\n(sin datos)', ha='center', va='center',
                    transform=ax.transAxes, fontsize=9, color='grey')
            ax.set_title(label, fontsize=9)
            continue

        data_all = (lakes_gdf[col].dropna() * scale).values
        ax.hist(data_all, bins=40, color=base_col, alpha=0.30,
                density=True, edgecolor='none')

        if len(data_all) > 5:
            kde = gaussian_kde(data_all, bw_method=0.2)
            x0  = np.linspace(data_all.min(), np.percentile(data_all, 98), 300)
            ax.plot(x0, kde(x0), color=base_col, lw=2)

        if glof_col:
            for glof_val, ls in [(0, '--'), (1, '-')]:
                subset = lakes_gdf[lakes_gdf[glof_col] == glof_val][col].dropna() * scale
                if len(subset) > 3:
                    kde2 = gaussian_kde(subset.values, bw_method=0.25)
                    ax.plot(x0, kde2(x0),
                            color=COLORS['red'] if glof_val else base_col,
                            lw=1.5, ls=ls, alpha=0.85,
                            label='GLOF' if glof_val else 'Sin GLOF')

        p25, p75 = np.percentile(data_all, [25, 75])
        ax.axvspan(p25, p75, alpha=0.08, color='black')
        ax.set_xlabel(label, fontsize=9)
        ax.set_ylabel('Densidad', fontsize=9)
        ax.set_title(label, fontweight='bold', fontsize=10)
        if glof_col:
            ax.legend(fontsize=8)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Guardado: {output_path}")

    return fig


# =============================================================================
# Importancia de características (horizontal bar)
# =============================================================================

def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 15,
    title: str = 'Feature Importance',
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """Gráfico horizontal de importancia de características."""
    fig, ax = plt.subplots(figsize=(10, 7))

    top  = importance_df.head(top_n)
    y    = np.arange(len(top))
    col  = 'importance' if 'importance' in top.columns else top.columns[1]
    vals = top[col].values
    bar_colors = [COLORS['blue'] if v >= np.median(vals) else COLORS['cyan'] for v in vals]

    ax.barh(y, vals, color=bar_colors, edgecolor='white', height=0.65)
    for i, v in enumerate(vals):
        ax.text(v + vals.max() * 0.01, i, f'{v:.4f}', va='center', fontsize=7.5)

    ax.set_yticks(y)
    ax.set_yticklabels(top['feature'])
    ax.invert_yaxis()
    ax.set_xlabel(col.replace('_', ' ').title())
    ax.set_title(title, fontweight='bold')

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Guardado: {output_path}")

    return fig


# =============================================================================
# Mapa interactivo Folium
# =============================================================================

def create_interactive_map(
    lakes_gdf,
    glof_gdf=None,
    output_path: Optional[Path] = None,
) -> Any:
    """Mapa interactivo Folium con probabilidad de GLOF por laguna."""
    if not HAS_FOLIUM:
        print("  [WARN] Folium no disponible.")
        return None

    l_wgs  = lakes_gdf.to_crs('EPSG:4326')
    center = [l_wgs.geometry.centroid.y.mean(), l_wgs.geometry.centroid.x.mean()]
    m      = folium.Map(location=center, zoom_start=8, tiles='CartoDB dark_matter')
    cluster = MarkerCluster(name='Lagunas glaciares').add_to(m)

    for idx, row in l_wgs.iterrows():
        prob  = row.get('glof_prob', 0)
        color = '#CC3311' if prob > 0.6 else ('#EE7733' if prob > 0.3 else '#33BBEE')
        html  = (
            f"<b>ID:</b> {idx}<br>"
            f"<b>Área:</b> {row.get('area_m2', 0)/1e6:.4f} km²<br>"
            f"<b>Elevación:</b> {row.get('elev_mean', 'N/A')} m<br>"
            f"<b>Prob. GLOF:</b> {prob:.3f}"
        )
        folium.CircleMarker(
            location=[row.geometry.centroid.y, row.geometry.centroid.x],
            radius=5, color=color, fill=True, fill_opacity=0.7,
            popup=folium.Popup(html, max_width=200),
        ).add_to(cluster)

    if output_path:
        m.save(str(output_path))
        print(f"  Guardado: {output_path}")

    return m


# =============================================================================
# Umbral análisis — distribución por clase + boxplot
# =============================================================================

def plot_threshold_analysis(
    feature_values: np.ndarray,
    y_true: np.ndarray,
    feature_name: str,
    threshold: Optional[float] = None,
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """Distribución de una característica por clase, con umbral marcado."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    no_glof = feature_values[y_true == 0]
    glof    = feature_values[y_true == 1]

    axes[0].hist(no_glof, bins=30, alpha=0.6, label='Sin GLOF',
                 color=COLORS['blue'], density=True)
    axes[0].hist(glof, bins=15, alpha=0.6, label='Con GLOF',
                 color=COLORS['red'], density=True)
    if threshold is not None:
        axes[0].axvline(threshold, color='black', ls='--', lw=2,
                        label=f'Umbral = {threshold:.1f}')
    axes[0].set_xlabel(feature_name)
    axes[0].set_ylabel('Densidad')
    axes[0].set_title(f'Distribución de {feature_name}', fontweight='bold')
    axes[0].legend()

    axes[1].boxplot([no_glof, glof], labels=['Sin GLOF', 'Con GLOF'])
    if threshold is not None:
        axes[1].axhline(threshold, color='black', ls='--', lw=2,
                        label=f'Umbral = {threshold:.1f}')
        axes[1].legend()
    axes[1].set_ylabel(feature_name)
    axes[1].set_title(f'{feature_name} por clase', fontweight='bold')

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Guardado: {output_path}")

    return fig


def plot_threshold_summary(
    lakes_gdf,
    importance_df: Optional[pd.DataFrame] = None,
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """Panel resumen de umbrales: boxplot + importancia + riesgo por elevación + tabla."""
    from scipy import stats

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    if 'area_depth_ratio' in lakes_gdf.columns and 'had_glof' in lakes_gdf.columns:
        no_glof = lakes_gdf[lakes_gdf['had_glof'] == 0]['area_depth_ratio'].dropna()
        glof    = lakes_gdf[lakes_gdf['had_glof'] == 1]['area_depth_ratio'].dropna()
        ax.boxplot([no_glof, glof], labels=['Sin GLOF', 'Con GLOF'])
        ax.set_ylabel('Ratio Área/Profundidad')
        ax.set_title('(a) H1: Ratio Área/Profundidad', fontweight='bold')
        if len(glof) > 0 and len(no_glof) > 0:
            _, pval = stats.mannwhitneyu(glof, no_glof, alternative='two-sided')
            ax.text(0.5, 0.96, f'p = {pval:.4f}', transform=ax.transAxes,
                    ha='center', va='top', fontsize=9)

    ax = axes[0, 1]
    if importance_df is not None:
        top10 = importance_df.head(10)
        y_pos = np.arange(len(top10))
        col   = 'importance' if 'importance' in top10.columns else top10.columns[1]
        ax.barh(y_pos, top10[col], color=COLORS['blue'], edgecolor='white')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top10['feature'], fontsize=8)
        ax.invert_yaxis()
        ax.set_title('(b) Importancia de Características', fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax.transAxes)

    ax = axes[1, 0]
    if 'elev_mean' in lakes_gdf.columns and 'had_glof' in lakes_gdf.columns:
        df  = lakes_gdf.copy()
        df['elev_bin'] = pd.cut(df['elev_mean'], bins=10)
        risk = df.groupby('elev_bin')['had_glof'].agg(['mean', 'count'])
        risk = risk[risk['count'] >= 5]
        if len(risk) > 0:
            centers = [i.mid for i in risk.index]
            ax.plot(centers, risk['mean'], 'bo-', lw=2)
            ax.fill_between(centers, 0, risk['mean'], alpha=0.25)
            ax.set_xlabel('Elevación (m s.n.m.)')
            ax.set_ylabel('Riesgo')
            ax.set_title('(c) H3: Riesgo por Zona de Elevación', fontweight='bold')

    ax = axes[1, 1]
    ax.axis('off')
    if 'had_glof' in lakes_gdf.columns:
        n_glof  = int(lakes_gdf['had_glof'].sum())
        n_total = len(lakes_gdf)
        data = [
            ['Total lagunas',    f'{n_total}'],
            ['Lagunas con GLOF', f'{n_glof}'],
            ['Tasa GLOF',        f'{n_glof / n_total * 100:.1f}%'],
        ]
        table = ax.table(cellText=data, colLabels=['Métrica', 'Valor'], loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        ax.set_title('(d) Estadísticas Resumen', fontweight='bold')

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Guardado: {output_path}")

    return fig
