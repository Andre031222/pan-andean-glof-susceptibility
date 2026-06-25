import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box, Point

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config_expanded_study_areas import EXPANDED_STUDY_AREAS

GLOF_FILE = PROJECT_ROOT / 'data' / 'processed' / 'labeled' / 'historical_glofs.gpkg'
_FEAT_DIR = PROJECT_ROOT / 'data' / 'processed' / 'features'
_FULL_GPKG = _FEAT_DIR / 'lake_features_full.gpkg'
FEATURES_GPKG = _FULL_GPKG if _FULL_GPKG.exists() else _FEAT_DIR / 'lake_features.gpkg'
FEATURES_CSV = (_FEAT_DIR / 'lake_features_full.csv'
                if (_FEAT_DIR / 'lake_features_full.csv').exists()
                else _FEAT_DIR / 'lake_features.csv')

YEAR_TOL_S2 = 1
YEAR_TOL_LS = 2

AREA_BBOX_GEOMS = {name: box(*cfg['bbox']) for name, cfg in EXPANDED_STUDY_AREAS.items()}
AREA_EPSG = {name: cfg['epsg'] for name, cfg in EXPANDED_STUDY_AREAS.items()}


def assign_area_to_glofs(glofs_4326: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    glofs = glofs_4326.copy()
    glofs['matched_area'] = None

    for idx, row in glofs.iterrows():
        if row.geometry is None:
            continue
        pt = gpd.GeoSeries([row.geometry], crs=glofs.crs).to_crs('EPSG:4326').iloc[0]
        for area_name, bbox_geom in AREA_BBOX_GEOMS.items():
            if bbox_geom.contains(pt):
                glofs.at[idx, 'matched_area'] = area_name
                break

    unmatched = glofs['matched_area'].isna().sum()
    if unmatched > 0:
        print(f"  [WARN] {unmatched} GLOF events outside all study area bboxes")
    return glofs


def get_glof_year(row) -> int:
    if 'year' in row.index and row['year'] is not None:
        try:
            return int(row['year'])
        except (ValueError, TypeError):
            pass
    for col in ['date', 'event_date', 'Date']:
        if col in row.index and row[col] is not None:
            try:
                return int(str(row[col])[:4])
            except (ValueError, TypeError):
                pass
    return None


def match_one_glof(glof_row, lakes_gdf: gpd.GeoDataFrame, buffer_s2: float, buffer_ls: float) -> dict:
    year = get_glof_year(glof_row)
    if year is None:
        return {'status': 'NO_YEAR', 'year': None, 'matched_lake_idx': None, 'distance_m': None}

    area_name = glof_row.get('matched_area', None)
    is_post2017 = year >= 2017
    buffer_m = buffer_s2 if is_post2017 else buffer_ls
    source_filter = 'sentinel2' if is_post2017 else 'landsat'
    year_tol = YEAR_TOL_S2 if is_post2017 else YEAR_TOL_LS

    cands = lakes_gdf.copy()
    if area_name:
        cands = cands[cands['area_name'] == area_name]
    cands = cands[cands['source'] == source_filter]
    cands = cands[np.abs(cands['year'] - year) <= year_tol]

    if cands.empty:
        return {
            'status': 'NO_CANDIDATES',
            'year': year,
            'area': area_name,
            'source': source_filter,
            'matched_lake_idx': None,
            'distance_m': None,
        }

    glof_pt_4326 = gpd.GeoSeries([glof_row.geometry], crs='EPSG:4326')

    if area_name and area_name in AREA_EPSG:
        target_crs = f"EPSG:{AREA_EPSG[area_name]}"
    else:
        target_crs = cands.crs

    glof_utm = glof_pt_4326.to_crs(target_crs).iloc[0]
    cands_utm = cands.to_crs(target_crs)

    distances = cands_utm.geometry.centroid.apply(lambda c: glof_utm.distance(c))

    min_idx = distances.idxmin()
    min_dist = float(distances[min_idx])

    if min_dist <= buffer_m:
        return {
            'status': 'MATCHED',
            'year': year,
            'area': area_name,
            'source': source_filter,
            'matched_lake_idx': min_idx,
            'distance_m': round(min_dist, 1),
        }
    else:
        return {
            'status': f'TOO_FAR ({min_dist:.0f}m > {buffer_m:.0f}m)',
            'year': year,
            'area': area_name,
            'source': source_filter,
            'matched_lake_idx': None,
            'distance_m': round(min_dist, 1),
        }


def main():
    parser = argparse.ArgumentParser(
        description='Match GLOF events to lake inventory and produce labeled training data\n'
                    'Usage: python scripts/07_match_glofs.py\n'
                    '       python scripts/07_match_glofs.py --buffer 5000 --pre2017-buffer 10000'
    )
    parser.add_argument('--buffer', type=float, default=5000,
                        help='Post-2017 match buffer in metres (default: 5000)')
    parser.add_argument('--pre2017-buffer', type=float, default=10000,
                        help='Pre-2017 match buffer in metres (default: 10000)')
    args = parser.parse_args()

    print("=== GLOF Event Matching ===")

    if not GLOF_FILE.exists():
        print(f"[ERROR] GLOF file not found: {GLOF_FILE}")
        sys.exit(1)

    if not FEATURES_GPKG.exists() and not FEATURES_CSV.exists():
        print("[ERROR] Features file not found. Run 06_extract_features.py first.")
        sys.exit(1)

    print(f"\n1. Loading GLOF events from {GLOF_FILE.name}...")
    glofs = gpd.read_file(GLOF_FILE)
    if glofs.crs is None:
        glofs = glofs.set_crs('EPSG:4326')
    glofs_4326 = glofs.to_crs('EPSG:4326')
    print(f"   {len(glofs_4326)} events  |  columns: {list(glofs_4326.columns)}")

    print("\n2. Assigning study areas to GLOF events...")
    glofs_4326 = assign_area_to_glofs(glofs_4326)
    print(f"   Area distribution:\n{glofs_4326['matched_area'].value_counts().to_string()}")

    print(f"\n3. Loading lake features...")
    if FEATURES_GPKG.exists():
        lakes_gdf = gpd.read_file(FEATURES_GPKG)
        print(f"   {len(lakes_gdf)} lake-year rows (from GPKG with geometry)")
    else:
        print(f"   [WARN] GPKG not found, loading from CSV + lake files...")
        df = pd.read_csv(FEATURES_CSV)
        frames = []
        for src_dir in ['lakes_s2', 'lakes_landsat']:
            d = PROJECT_ROOT / 'data' / 'processed' / src_dir
            if d.exists():
                for f in sorted(d.glob('*.gpkg')):
                    try:
                        g = gpd.read_file(f)
                        if not g.empty and 'lake_id' in g.columns:
                            frames.append(g[['lake_id', 'geometry', 'crs']])
                    except Exception:
                        pass
        if frames:
            geom_df = pd.concat(frames, ignore_index=True)
            geom_map = geom_df.drop_duplicates('lake_id').set_index('lake_id')['geometry']
            df['geometry'] = df['lake_id'].map(geom_map)
        first_area = df['area_name'].dropna().iloc[0] if not df.empty else None
        fallback_crs = f"EPSG:{AREA_EPSG[first_area]}" if first_area in AREA_EPSG else 'EPSG:32719'
        lakes_gdf = gpd.GeoDataFrame(df, crs=fallback_crs)
        print(f"   {len(lakes_gdf)} lake-year rows")

    lakes_gdf['glof'] = 0

    print(f"\n4. Matching GLOF events...")
    print(f"   post-2017: {args.buffer/1000:.0f}km buffer  |  pre-2017: {args.pre2017_buffer/1000:.0f}km buffer")

    match_log = []
    matched_count = 0

    for glof_idx, glof_row in glofs_4326.iterrows():
        result = match_one_glof(glof_row, lakes_gdf, args.buffer, args.pre2017_buffer)

        matched_lake_idx = result.pop('matched_lake_idx', None)
        lake_id = None

        if result['status'] == 'MATCHED' and matched_lake_idx is not None:
            lakes_gdf.at[matched_lake_idx, 'glof'] = 1
            lake_id = (lakes_gdf.at[matched_lake_idx, 'lake_id']
                       if 'lake_id' in lakes_gdf.columns else str(matched_lake_idx))
            matched_count += 1

        match_log.append({
            'glof_idx': glof_idx,
            'year': result.get('year'),
            'area': result.get('area'),
            'source': result.get('source'),
            'matched_lake_id': lake_id,
            'distance_m': result.get('distance_m'),
            'status': result['status'],
        })

    n_total = len(glofs_4326)
    n_positive = int((lakes_gdf['glof'] == 1).sum())
    n_negative = int((lakes_gdf['glof'] == 0).sum())

    print(f"\n   MATCHED  : {matched_count}/{n_total} GLOF events")
    print(f"   UNMATCHED: {n_total - matched_count}")

    matched_df = pd.DataFrame(match_log)
    if not matched_df.empty:
        matched_only = matched_df[matched_df['status'] == 'MATCHED']
        if not matched_only.empty:
            print(f"\n   Matches by area:")
            print(matched_only.groupby('area').size().to_string())
            print(f"\n   Matches by source:")
            print(matched_only.groupby('source').size().to_string())

    print(f"\n5. Training data: n+={n_positive}, n-={n_negative}, ratio={n_negative/n_positive:.0f}:1"
          if n_positive > 0 else f"\n5. WARNING: 0 positives matched")

    print("\n6. Saving outputs...")
    out_dir = PROJECT_ROOT / 'data' / 'processed' / 'labeled'
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_cols = [c for c in lakes_gdf.columns if c != 'geometry']
    training_csv = out_dir / 'training_data.csv'
    lakes_gdf[csv_cols].to_csv(training_csv, index=False)
    print(f"  Saved: {training_csv}")

    log_csv = out_dir / 'glof_match_log.csv'
    matched_df.to_csv(log_csv, index=False)
    print(f"  Saved: {log_csv}")

    print(f"\n[done] n+={n_positive}, n-={n_negative}")


if __name__ == '__main__':
    main()
