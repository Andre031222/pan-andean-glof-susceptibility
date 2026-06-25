import sys
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from shapely.geometry import Point

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from config_expanded_study_areas import EXPANDED_STUDY_AREAS

IN_GPKG = PROJECT_ROOT / 'data' / 'processed' / 'features' / 'lake_features.gpkg'
OUT_GPKG = PROJECT_ROOT / 'data' / 'processed' / 'features' / 'lake_features_full.gpkg'
OUT_CSV = PROJECT_ROOT / 'data' / 'processed' / 'features' / 'lake_features_full.csv'
DEM_DIR = PROJECT_ROOT / 'data' / 'interim' / 'dem'
ANDES_DEPTH_CORRECTION = 1.15


def add_geometric(gdf):
    g = gdf.geometry
    for c in ['convexity', 'equivalent_diameter_m', 'shore_dev_index']:
        gdf[c] = np.nan
    for area in gdf['area_name'].unique():
        m = gdf['area_name'] == area
        sub = gdf[m].to_crs(gdf[m].estimate_utm_crs())
        conv, eqd, sdi = [], [], []
        for geom in sub.geometry:
            try:
                a = geom.area
                p = geom.length
                ch = geom.convex_hull.area
                conv.append(a / ch if ch > 0 else 1.0)
                eqd.append(2.0 * np.sqrt(a / np.pi))
                cp = 2.0 * np.pi * np.sqrt(a / np.pi)
                sdi.append(p / cp if cp > 0 else 1.0)
            except Exception:
                conv.append(np.nan); eqd.append(np.nan); sdi.append(np.nan)
        gdf.loc[m, 'convexity'] = conv
        gdf.loc[m, 'equivalent_diameter_m'] = eqd
        gdf.loc[m, 'shore_dev_index'] = sdi
    return gdf


def _sample_array(arr, transform, xs, ys, nodata):
    inv = ~transform
    cols, rows = inv * (np.asarray(xs), np.asarray(ys))
    rows = np.floor(rows).astype(int)
    cols = np.floor(cols).astype(int)
    h, w = arr.shape
    ok = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
    out = np.full(rows.shape, np.nan)
    valid = ok.copy()
    if ok.any():
        vals = arr[rows[ok], cols[ok]].astype(float)
        bad = (vals == nodata) | ~np.isfinite(vals)
        vals[bad] = np.nan
        out[ok] = vals
        valid[ok] = ~bad
    return out, valid


def add_dam(gdf):
    for c in ['dam_elev', 'freeboard', 'dam_height']:
        gdf[c] = np.nan
    for area in gdf['area_name'].unique():
        dem_path = DEM_DIR / f'{area}_dem_utm.tif'
        if not dem_path.exists():
            continue
        m = gdf['area_name'] == area
        epsg = EXPANDED_STUDY_AREAS.get(area, {}).get('epsg', 32718)
        sub = gdf[m].to_crs(f'EPSG:{epsg}')
        with rasterio.open(dem_path) as src:
            arr = src.read(1)
            transform = src.transform
            nodata = src.nodata if src.nodata is not None else -9999
        for idx, row in sub.iterrows():
            geom = row.geometry
            try:
                ext = geom.exterior
                n_pts = max(60, int(ext.length / 30))
                pts = [ext.interpolate(i / n_pts, normalized=True) for i in range(n_pts)]
                xs = np.array([p.x for p in pts])
                ys = np.array([p.y for p in pts])
                elevs, valid = _sample_array(arr, transform, xs, ys, nodata)
                if not valid.any():
                    continue
                vi = np.where(valid)[0]
                dam_i = vi[np.argmin(elevs[vi])]
                dam_elev = float(elevs[dam_i])
                dam_pt = pts[dam_i]
                lake_elev = row.get('elev_mean', np.nan)
                if not np.isfinite(lake_elev):
                    lake_elev = float(np.nanmean(elevs[vi]))
                angs = np.linspace(0, 2 * np.pi, 8, endpoint=False)
                tx = dam_pt.x + 100.0 * np.cos(angs)
                ty = dam_pt.y + 100.0 * np.sin(angs)
                outside = np.array([not geom.contains(Point(x, y)) for x, y in zip(tx, ty)])
                dv, dvalid = _sample_array(arr, transform, tx, ty, nodata)
                mask_ds = outside & dvalid & (dv < dam_elev)
                gdf.loc[idx, 'dam_elev'] = dam_elev
                gdf.loc[idx, 'freeboard'] = float(lake_elev) - dam_elev
                gdf.loc[idx, 'dam_height'] = dam_elev - float(np.min(dv[mask_ds])) if mask_ds.any() else np.nan
            except Exception:
                pass
    return gdf


def add_depth_volume(gdf):
    a = gdf['area_m2'].values.astype(float)
    methods = {
        'depth_cook_quincey_2015': 0.104 * (a ** 0.420),
        'depth_huggel_2002': 0.104 * (a ** 0.420),
        'depth_yao_2012': 0.148 * (a ** 0.410),
        'depth_oconnor_2001': 0.133 * (a ** 0.424),
    }
    for k, v in methods.items():
        gdf[k] = v
    stack = np.vstack(list(methods.values()))
    ens_mean = np.mean(stack, axis=0)
    ens_std = np.std(stack, axis=0)
    gdf['depth_ensemble_mean'] = ens_mean
    gdf['depth_ensemble_std'] = ens_std
    gdf['depth_m'] = ens_mean * ANDES_DEPTH_CORRECTION
    gdf['depth_std_m'] = ens_std * ANDES_DEPTH_CORRECTION
    d_max = gdf['depth_m'].values / 0.6
    gdf['volume_m3'] = 0.45 * a * d_max
    gdf['area_depth_ratio'] = a / np.where(gdf['depth_m'].values == 0, np.nan, gdf['depth_m'].values)
    return gdf


def add_risk(gdf):
    if 'volume_m3' in gdf and 'dam_height' in gdf:
        gdf['potential_energy'] = gdf['volume_m3'] * 9.81 * gdf['dam_height']
    gdf['size_class'] = pd.cut(gdf['area_m2'], bins=[0, 10000, 1000000, np.inf],
                               labels=['small', 'medium', 'large'])

    def norm(s):
        mn, mx = s.min(), s.max()
        return (s - mn) / (mx - mn) if mx > mn else s * 0.0
    score = pd.Series(0.0, index=gdf.index)
    grow = 'growth_rate_m2_yr' if 'growth_rate_m2_yr' in gdf else 'area_trend'
    for col, w in [('area_depth_ratio', 0.35), ('slope_mean', 0.35), (grow, 0.30)]:
        if col in gdf:
            score += w * norm(gdf[col].fillna(0))
    gdf['risk_score'] = score
    return gdf


def main():
    print('=== Full 56-feature extraction (NB13 formulas on new inventory) ===')
    gdf = gpd.read_file(IN_GPKG)
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    print(f'  lakes: {len(gdf)}  base features: {len(gdf.columns)}')
    if 'area_trend' in gdf.columns and 'growth_rate_m2_yr' not in gdf.columns:
        gdf['growth_rate_m2_yr'] = gdf['area_trend']
    gdf = add_geometric(gdf)
    print('  + geometric (convexity, eq_diam, shore_dev)')
    gdf = add_dam(gdf)
    print('  + dam (dam_elev, freeboard, dam_height)')
    gdf = add_depth_volume(gdf)
    print('  + depth ensemble + volume + area_depth_ratio')
    gdf = add_risk(gdf)
    print('  + risk indicators')
    gdf.to_file(OUT_GPKG, driver='GPKG')
    cols = [c for c in gdf.columns if c != 'geometry']
    gdf[cols].to_csv(OUT_CSV, index=False)
    nfeat = sum(gdf[c].dtype.kind in 'fi' for c in cols)
    print(f'  saved {len(gdf)} lakes x {nfeat} numeric features -> {OUT_CSV.name}')
    print(f'  volume p50={gdf["volume_m3"].median():.0f} m3  depth_m p50={gdf["depth_m"].median():.1f} m')


if __name__ == '__main__':
    main()
