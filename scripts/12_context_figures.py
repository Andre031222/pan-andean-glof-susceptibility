import sys
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = PROJECT_ROOT / 'figures' / 'publication'
FEAT_GPKG = PROJECT_ROOT / 'data' / 'processed' / 'features' / 'lake_features.gpkg'
GLOF_GPKG = PROJECT_ROOT / 'data' / 'processed' / 'labeled' / 'historical_glofs.gpkg'
RANDOM_STATE = 42

plt.rcParams.update({'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
                     'savefig.dpi': 300, 'axes.grid': True, 'grid.alpha': 0.25,
                     'axes.axisbelow': True})

LABEL = {
    'cordillera_raura': 'Raura', 'cordillera_huayhuash': 'Huayhuash',
    'cordillera_central': 'Central', 'carabaya': 'Carabaya',
    'cordillera_urubamba': 'Urubamba', 'apolobamba': 'Apolobamba',
    'cordillera_vilcanota': 'Vilcanota', 'cordillera_blanca': 'Blanca',
    'ecuador_antisana': 'Antisana', 'bolivia_cordillera_real': 'C. Real',
    'cordillera_huanzo': 'Huanzo', 'chile_andes_centrales': 'Andes Centrales',
    'patagonia_norte': 'Patagonia N', 'patagonia_sur': 'Patagonia S',
    'bolivia_norte': 'Bolivia N',
}


def load_train_module():
    spec = importlib.util.spec_from_file_location('t', PROJECT_ROOT / 'scripts' / '08_train_model.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def oof_scores(m):
    lakes, X, y, groups, fc = m.load_data()
    from imblearn.ensemble import BalancedRandomForestClassifier
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


def _natural_earth():
    url = ('https://raw.githubusercontent.com/nvkelso/natural-earth-vector/'
           'master/geojson/ne_110m_admin_0_countries.geojson')
    try:
        return gpd.read_file(url)
    except Exception as e:
        print(f'  [warn] no basemap ({e})')
        return None


def fig1_map(lakes, glofs):
    from matplotlib.patches import Rectangle
    cx = lakes['lake_key'].str.rsplit('_', n=2, expand=True)
    lon = pd.to_numeric(cx[1], errors='coerce') / 1000.0
    lat = pd.to_numeric(cx[2], errors='coerce') / 1000.0
    area = pd.to_numeric(lakes.get('area_m2', pd.Series(5e4, index=lakes.index)),
                         errors='coerce').fillna(5e4)
    sizes = 4 + 6 * (np.log10(area.clip(lower=1e4)) - 4)
    world = _natural_earth()
    ext = (-81, -65, -53, 3)

    fig, ax = plt.subplots(figsize=(7.5, 9))
    if world is not None:
        world.plot(ax=ax, facecolor='#eef1f4', edgecolor='#aab2bd', linewidth=0.5, zorder=0)
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])

    order = lakes['oof'].fillna(0).argsort()
    sc = ax.scatter(lon.iloc[order], lat.iloc[order], c=lakes['oof'].iloc[order],
                    cmap='RdYlBu_r', s=sizes.iloc[order], alpha=0.85, vmin=0, vmax=1,
                    linewidths=0, zorder=3)

    cen = lakes.copy()
    cen['_lon'] = lon.values; cen['_lat'] = lat.values
    cc = cen.groupby('area_name')[['_lon', '_lat']].mean()
    ax.scatter(cc['_lon'], cc['_lat'], marker='o', facecolors='none', edgecolors='#2b2b2b',
               s=55, linewidths=0.9, zorder=4, label='Study-area centroid')

    if glofs is not None and len(glofs):
        gl = glofs.to_crs(4326)
        ax.scatter(gl.geometry.x, gl.geometry.y, marker='*', s=70, c='black',
                   edgecolors='white', linewidths=0.4, label='Historical GLOF', zorder=5)
    ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
    ax.legend(loc='lower right', fontsize=8, frameon=True)
    cb = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label('Susceptibility score')

    if world is not None:
        iax = fig.add_axes([0.135, 0.135, 0.23, 0.23])
        world.plot(ax=iax, facecolor='#eef1f4', edgecolor='#aab2bd', linewidth=0.3)
        iax.add_patch(Rectangle((ext[0], ext[2]), ext[1] - ext[0], ext[3] - ext[2],
                                fill=False, edgecolor='red', linewidth=1.1, zorder=5))
        iax.set_xlim(-82, -33); iax.set_ylim(-56, 14)
        iax.set_xticks([]); iax.set_yticks([])
        for sp in iax.spines.values():
            sp.set_linewidth(0.6)

    fig.savefig(FIG_DIR / 'fig1_susceptibility_map.jpg', bbox_inches='tight', dpi=300)
    plt.close(fig)


def fig2_context(lakes, glofs):
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 2, hspace=0.42, wspace=0.28)
    counts = lakes.groupby('area_name').size().sort_values(ascending=True)
    pos = lakes[lakes.y == 1].groupby('area_name').size()
    names = [LABEL.get(a, a) for a in counts.index]

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.barh(names, counts.values, color='#3b6ea5', alpha=0.85)
    ax0.set_xlabel('Modelled lakes ($\\geq$0.05 km$^2$)')
    ax0.set_title('(a) Lakes per region')

    ax1 = fig.add_subplot(gs[0, 1])
    rate = (pos.reindex(counts.index).fillna(0) / counts * 100).values
    ax1.barh(names, rate, color='#c1272d', alpha=0.85)
    ax1.set_xlabel('GLOF-source lakes (%)')
    ax1.set_title('(b) GLOF-source rate per region')

    ax2 = fig.add_subplot(gs[1, 0])
    if glofs is not None and 'trigger' in glofs.columns:
        tc = glofs['trigger'].value_counts()
        ax2.barh([t.replace('_', ' ') for t in tc.index[::-1]], tc.values[::-1],
                 color='#6c8ebf', alpha=0.85)
    ax2.set_xlabel('Count'); ax2.set_title('(c) Trigger mechanisms (71 events)')

    ax3 = fig.add_subplot(gs[1, 1])
    if glofs is not None and 'year' in glofs.columns:
        yr = pd.to_numeric(glofs['year'], errors='coerce').dropna()
        vol = pd.to_numeric(glofs.get('volume_released_m3', pd.Series(index=glofs.index)),
                            errors='coerce').reindex(yr.index)
        sz = 20 + 30 * np.log10(vol.fillna(vol.median() if vol.notna().any() else 1e6).clip(lower=1e4))
        ax3.scatter(yr, np.random.RandomState(0).uniform(0, 1, len(yr)), s=sz,
                    c='#c1272d', alpha=0.5, edgecolors='none')
        ax3.set_yticks([])
    ax3.set_xlabel('Year'); ax3.set_title('(d) GLOF timeline 1932–2023')
    fig.savefig(FIG_DIR / 'fig2_inventory_glof_context.png', bbox_inches='tight')
    plt.close(fig)


def main():
    print('=== context figures (fig1 map, fig2 context) ===')
    m = load_train_module()
    lakes = oof_scores(m)
    glofs = gpd.read_file(GLOF_GPKG) if GLOF_GPKG.exists() else None
    fig1_map(lakes, glofs)
    print('  fig1 done')
    fig2_context(lakes, glofs)
    print('  fig2 done')


if __name__ == '__main__':
    main()
