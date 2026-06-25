import sys
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.features import geometry_mask
from rasterio.windows import Window, transform as window_transform
from scipy.stats import linregress

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config_expanded_study_areas import EXPANDED_STUDY_AREAS

TERRAIN_LAYERS = ['slope', 'aspect', 'tri', 'twi', 'curvature', 'roughness', 'vrm']
RGI_PATH = PROJECT_ROOT / 'data' / 'raw' / 'rgi7_andes.gpkg'
GLACIER_MAX_DIST_M = 5000.0


def zonal_stats(geom, raster_path: Path, stats=('mean', 'std', 'min', 'max')) -> dict:
    if not raster_path.exists():
        return {s: np.nan for s in stats}
    try:
        with rasterio.open(raster_path) as src:
            data, _ = rio_mask(src, [geom.__geo_interface__], crop=True, nodata=np.nan)
            vals = data[0].flatten()
            vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            return {s: np.nan for s in stats}
        result = {}
        if 'mean' in stats:
            result['mean'] = float(np.mean(vals))
        if 'std' in stats:
            result['std'] = float(np.std(vals))
        if 'min' in stats:
            result['min'] = float(np.min(vals))
        if 'max' in stats:
            result['max'] = float(np.max(vals))
        return result
    except Exception:
        return {s: np.nan for s in stats}


def compute_morphometrics(geom) -> dict:
    area = geom.area
    perim = geom.length
    compactness = (4 * np.pi * area / perim ** 2) if perim > 0 else np.nan
    try:
        coords = list(geom.minimum_rotated_rectangle.exterior.coords)
        sides = [
            np.sqrt((coords[i][0] - coords[i-1][0])**2 + (coords[i][1] - coords[i-1][1])**2)
            for i in range(1, 5)
        ]
        major, minor = max(sides), min(sides)
        elongation = major / minor if minor > 0 else np.nan
    except Exception:
        elongation = np.nan
    fractal_dim = (2 * np.log(perim) / np.log(area)) if (area > 1 and perim > 1) else np.nan
    return {
        'area_m2': area,
        'perimeter_m': perim,
        'compactness': compactness,
        'elongation': elongation,
        'fractal_dim': fractal_dim,
    }


def compute_dist_glacier(gdf: gpd.GeoDataFrame) -> pd.Series:
    if not RGI_PATH.exists():
        warnings.warn("RGI file not found — dist_glacier_m = NaN")
        return pd.Series(np.nan, index=gdf.index)
    glaciers_all = gpd.read_file(RGI_PATH)
    out = pd.Series(np.nan, index=gdf.index)
    for area_name in gdf['area_name'].unique():
        mask = gdf['area_name'] == area_name
        sub = gdf[mask]
        utm = sub.estimate_utm_crs()
        pts = sub.to_crs(utm).copy()
        pts['geometry'] = pts.geometry.centroid
        gl = glaciers_all.to_crs(utm)[['geometry']]
        gl = gl[gl.geometry.notna() & ~gl.geometry.is_empty].copy()
        gl['geometry'] = gl.geometry.make_valid()
        gl = gl[gl.geometry.is_valid & ~gl.geometry.is_empty]
        joined = gpd.sjoin_nearest(pts[['geometry']], gl, distance_col='_dgl')
        joined = joined[~joined.index.duplicated(keep='first')]
        out.loc[mask] = joined['_dgl'].reindex(pts.index).values
    return out


def load_all_lakes(source: str) -> gpd.GeoDataFrame:
    frames = []
    if source in ('s2', 'all'):
        d = PROJECT_ROOT / 'data' / 'processed' / 'lakes_s2'
        if d.exists():
            files = sorted(d.glob('*.gpkg'))
            print(f"  Loading {len(files)} S2 lake files...")
            for f in files:
                try:
                    gdf = gpd.read_file(f)
                    if not gdf.empty:
                        frames.append(gdf.to_crs('EPSG:4326'))
                except Exception as e:
                    print(f"    [WARN] {f.name}: {e}")

    if source in ('landsat', 'all'):
        d = PROJECT_ROOT / 'data' / 'processed' / 'lakes_landsat'
        if d.exists():
            files = sorted(d.glob('*.gpkg'))
            print(f"  Loading {len(files)} Landsat lake files...")
            for f in files:
                try:
                    gdf = gpd.read_file(f)
                    if not gdf.empty:
                        frames.append(gdf.to_crs('EPSG:4326'))
                except Exception as e:
                    print(f"    [WARN] {f.name}: {e}")

    if not frames:
        return gpd.GeoDataFrame()

    combined = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True))
    print(f"  Total lake-year observations: {len(combined)}")
    return combined


def extract_terrain_features(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf.copy()
    for col in ['elev_mean', 'elev_min', 'elev_max', 'elev_std',
                 'slope_mean', 'slope_std', 'aspect_mean',
                 'tri_mean', 'twi_mean', 'curvature_mean', 'roughness_mean', 'vrm_mean']:
        if col not in gdf.columns:
            gdf[col] = np.nan

    for area_name in gdf['area_name'].unique():
        mask = gdf['area_name'] == area_name
        dem_path = PROJECT_ROOT / 'data' / 'interim' / 'dem' / f"{area_name}_dem_utm.tif"
        terrain_dir = PROJECT_ROOT / 'data' / 'interim' / 'terrain' / area_name

        if not dem_path.exists():
            print(f"    [WARN] No DEM for {area_name}")
            continue

        row_index = gdf.index[(gdf['area_name'] == area_name).values]
        print(f"  Terrain features: {area_name} ({len(row_index)} polygons)")

        with rasterio.open(dem_path) as src:
            utm_crs = src.crs
            transform = src.transform
            H, W = src.height, src.width
            dem = src.read(1).astype(np.float32)
            nod = src.nodata
        if nod is not None:
            dem[dem == nod] = np.nan

        arrays = {'elev': dem}
        for layer in ('slope', 'aspect', 'tri', 'twi', 'curvature', 'roughness', 'vrm'):
            lp = terrain_dir / f"{layer}.tif"
            if not lp.exists():
                arrays[layer] = None
                continue
            with rasterio.open(lp) as src:
                a = src.read(1).astype(np.float32)
                ln = src.nodata
            if ln is not None:
                a[a == ln] = np.nan
            arrays[layer] = a

        sub = gdf.loc[row_index].to_crs(utm_crs)
        inv = ~transform
        res = {c: np.full(len(sub), np.nan) for c in (
            'elev_mean', 'elev_min', 'elev_max', 'elev_std', 'slope_mean', 'slope_std',
            'aspect_mean', 'tri_mean', 'twi_mean', 'curvature_mean', 'roughness_mean', 'vrm_mean')}

        for i, geom in enumerate(sub.geometry.values):
            if geom is None or geom.is_empty:
                continue
            minx, miny, maxx, maxy = geom.bounds
            c0, r0 = inv * (minx, maxy)
            c1, r1 = inv * (maxx, miny)
            cmin = max(0, int(np.floor(min(c0, c1)))); cmax = min(W, int(np.ceil(max(c0, c1))))
            rmin = max(0, int(np.floor(min(r0, r1)))); rmax = min(H, int(np.ceil(max(r0, r1))))
            if cmax <= cmin or rmax <= rmin:
                continue
            wt = window_transform(Window(cmin, rmin, cmax - cmin, rmax - rmin), transform)
            pm = geometry_mask([geom], out_shape=(rmax - rmin, cmax - cmin),
                               transform=wt, invert=True, all_touched=True)
            if not pm.any():
                continue
            d = dem[rmin:rmax, cmin:cmax][pm]; d = d[~np.isnan(d)]
            if d.size:
                res['elev_mean'][i] = d.mean(); res['elev_min'][i] = d.min()
                res['elev_max'][i] = d.max(); res['elev_std'][i] = d.std()
            for layer in ('slope', 'aspect', 'tri', 'twi', 'curvature', 'roughness', 'vrm'):
                a = arrays[layer]
                if a is None:
                    continue
                v = a[rmin:rmax, cmin:cmax][pm]; v = v[~np.isnan(v)]
                if v.size:
                    res[f'{layer}_mean'][i] = v.mean()
                    if layer == 'slope':
                        res['slope_std'][i] = v.std()

        for c, arr in res.items():
            gdf.loc[row_index, c] = arr

    return gdf


def extract_morphometrics(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf.copy()
    cols = ['area_m2', 'perimeter_m', 'compactness', 'elongation', 'fractal_dim']
    for c in cols:
        gdf[c] = np.nan
    for area_name in gdf['area_name'].unique():
        mask = gdf['area_name'] == area_name
        sub = gdf[mask]
        sub_utm = sub.to_crs(sub.estimate_utm_crs())
        recs = pd.DataFrame(list(sub_utm.geometry.apply(compute_morphometrics)),
                            index=sub_utm.index)
        for c in cols:
            gdf.loc[mask, c] = recs[c].values
    return gdf


def compute_temporal_features(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf.copy()
    gdf['area_trend'] = np.nan
    gdf['area_cv'] = np.nan
    gdf['presence_rate'] = np.nan

    centroids = gdf.geometry.centroid
    gdf['_cx'] = (centroids.x / 100).round(0) * 100
    gdf['_cy'] = (centroids.y / 100).round(0) * 100

    area_year_counts = gdf.groupby('area_name')['year'].agg(lambda x: len(x.unique())).to_dict()

    for group_key, group in gdf.groupby(['area_name', '_cx', '_cy']):
        area_name = group_key[0]
        years_present = sorted(group['year'].unique())
        total_years = area_year_counts.get(area_name, len(years_present))
        presence_rate = len(years_present) / total_years if total_years > 0 else np.nan

        sorted_group = group.sort_values('year')
        areas = sorted_group['area_m2'].values
        year_vals = sorted_group['year'].values

        if len(areas) >= 2:
            try:
                slope, *_ = linregress(year_vals, areas)
                area_trend = slope
            except Exception:
                area_trend = np.nan
            area_cv = np.std(areas) / np.mean(areas) if np.mean(areas) > 0 else np.nan
        else:
            area_trend = np.nan
            area_cv = np.nan

        gdf.loc[group.index, 'area_trend'] = area_trend
        gdf.loc[group.index, 'area_cv'] = area_cv
        gdf.loc[group.index, 'presence_rate'] = presence_rate

    gdf = gdf.drop(columns=['_cx', '_cy'])
    return gdf


def main():
    parser = argparse.ArgumentParser(
        description='Extract terrain and morphometric features for all lake-year observations\n'
                    'Usage: python scripts/06_extract_features.py\n'
                    '       python scripts/06_extract_features.py --source s2\n'
                    '       python scripts/06_extract_features.py --source landsat'
    )
    parser.add_argument('--source', choices=['s2', 'landsat', 'all'], default='all')
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / 'data' / 'processed' / 'features'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / 'lake_features.csv'
    out_gpkg = out_dir / 'lake_features.gpkg'

    print(f"=== Feature Extraction (source={args.source}) ===")

    print("\n1. Loading lake inventories...")
    gdf = load_all_lakes(args.source)
    if gdf.empty:
        print("[ERROR] No lakes found. Run 04/05 first.")
        sys.exit(1)

    print("\n2. Morphometric features...")
    gdf = extract_morphometrics(gdf)

    print("\n3. Terrain features (zonal statistics)...")
    gdf = extract_terrain_features(gdf)

    print("\n4. Distance to glaciers...")
    gdf['dist_glacier_m'] = compute_dist_glacier(gdf)

    print("\n5. Temporal features...")
    gdf = compute_temporal_features(gdf)

    print("\n6. Saving...")
    gdf.to_file(out_gpkg, driver='GPKG')
    print(f"  Saved: {out_gpkg} ({len(gdf)} rows)")

    csv_cols = [c for c in gdf.columns if c != 'geometry']
    gdf[csv_cols].to_csv(out_csv, index=False)
    print(f"  Saved: {out_csv}")

    feat_cols = [c for c in gdf.columns if gdf[c].dtype in (float, np.float64)]
    missing = gdf[feat_cols].isna().mean() * 100
    high_miss = missing[missing > 5]
    if not high_miss.empty:
        print(f"\nColumns with >5% NaN:")
        for col, pct in high_miss.items():
            print(f"  {col:30s}: {100-pct:.1f}% complete")

    print(f"\n[done] {len(gdf)} lake-year observations, {len(feat_cols)} feature columns")


if __name__ == '__main__':
    main()
