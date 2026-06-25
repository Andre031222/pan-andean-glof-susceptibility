import sys
import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape
from skimage.morphology import remove_small_objects, binary_closing, binary_opening, disk

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config_expanded_study_areas import EXPANDED_STUDY_AREAS

MIN_AREA_M2 = 9000.0
MAX_AREA_M2 = 50000000.0
MNDWI_THRESH = 0.0
GLACIER_BUFFER_M = 5000
RGI_PATH = PROJECT_ROOT / 'data' / 'raw' / 'rgi7_andes.gpkg'


def load_mndwi(year_dir: Path):
    mndwi_file = year_dir / 'mndwi.tif'
    if not mndwi_file.exists():
        return None
    with rasterio.open(mndwi_file) as src:
        return src.read(1).astype(np.float32), src.profile.copy(), src.transform, src.crs


def build_glacier_buffer(crs):
    if not RGI_PATH.exists():
        warnings.warn(f"RGI file not found: {RGI_PATH}")
        return None
    glaciers = gpd.read_file(RGI_PATH).to_crs(crs)
    geom = glaciers.geometry[glaciers.geometry.notna() & ~glaciers.geometry.is_empty]
    geom = geom.make_valid()
    geom = geom[geom.is_valid & ~geom.is_empty]
    buffered = geom.buffer(GLACIER_BUFFER_M)
    buffered = buffered[buffered.is_valid & ~buffered.is_empty]
    if buffered.empty:
        return None
    return buffered.union_all()


def vectorize_lakes(mndwi: np.ndarray, transform, crs,
                    area_name: str, year: int, satellite: str,
                    scene_date: str, glacier_buffer) -> gpd.GeoDataFrame:
    res_m = abs(transform.a)
    min_px = max(5, int(MIN_AREA_M2 / (res_m ** 2)))

    water = mndwi > MNDWI_THRESH
    water = remove_small_objects(water, min_size=min_px)
    water = binary_closing(water, disk(2))
    water = binary_opening(water, disk(2))
    water = remove_small_objects(water, min_size=min_px)

    if not water.any():
        return gpd.GeoDataFrame()

    mask_u8 = water.astype(np.uint8)
    polys = [shape(g) for g, v in shapes(mask_u8, transform=transform) if v == 1]

    if not polys:
        return gpd.GeoDataFrame()

    gdf = gpd.GeoDataFrame({'geometry': polys}, crs=crs)
    gdf['area_m2'] = gdf.geometry.area
    gdf = gdf[(gdf['area_m2'] >= MIN_AREA_M2) & (gdf['area_m2'] <= MAX_AREA_M2)].copy()

    if gdf.empty:
        return gdf

    if glacier_buffer is not None:
        within = gdf.geometry.centroid.within(glacier_buffer)
        gdf = gdf[within].copy()

    if gdf.empty:
        return gdf

    gdf['area_name'] = area_name
    gdf['year'] = year
    gdf['scene_date'] = scene_date
    gdf['source'] = 'landsat'
    gdf['satellite'] = satellite
    gdf['lake_id'] = [f"{area_name}_{year}_{i:04d}" for i in range(len(gdf))]

    return gdf.reset_index(drop=True)


def detect_one_area(area_name: str) -> bool:
    if area_name not in EXPANDED_STUDY_AREAS:
        print(f"[ERROR] Unknown area: {area_name}")
        return False

    cfg = EXPANDED_STUDY_AREAS[area_name]
    out_dir = PROJECT_ROOT / 'data' / 'processed' / 'lakes_landsat'
    out_dir.mkdir(parents=True, exist_ok=True)

    landsat_dir = PROJECT_ROOT / 'data' / 'raw' / 'landsat' / area_name
    if not landsat_dir.exists():
        print(f"[skip] {area_name} — no Landsat data")
        return True

    years = sorted([int(d.name) for d in landsat_dir.iterdir()
                    if d.is_dir() and d.name.isdigit()])
    if not years:
        print(f"[skip] {area_name} — no year directories")
        return True

    print(f"\n{'='*60}")
    print(f"Lake detection (Landsat): {area_name}  {len(years)} years")

    glacier_buffer = None
    total_lakes = 0

    for year in years:
        out_file = out_dir / f"{area_name}_{year}.gpkg"
        if out_file.exists():
            try:
                existing = gpd.read_file(out_file)
                print(f"  [skip] {year} — {len(existing)} lakes in {out_file.name}")
                total_lakes += len(existing)
                continue
            except Exception:
                out_file.unlink()

        year_dir = landsat_dir / str(year)
        result = load_mndwi(year_dir)
        if result is None:
            print(f"  [skip] {year} — no mndwi.tif")
            continue

        mndwi, profile, transform, crs = result

        if glacier_buffer is None:
            glacier_buffer = build_glacier_buffer(crs)

        satellite = 'unknown'
        scene_date = f"{year}-01-01"
        meta_file = year_dir / 'metadata.json'
        if meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text())
                satellite = meta.get('platform', 'unknown')
                scene_date = meta.get('date', scene_date)
            except Exception:
                pass

        print(f"  Processing {year} ({satellite})...", end=' ', flush=True)

        gdf = vectorize_lakes(mndwi, transform, crs, area_name, year,
                               satellite, scene_date, glacier_buffer)

        if gdf.empty:
            print("0 lakes")
            gpd.GeoDataFrame(columns=['geometry', 'lake_id', 'area_name', 'year',
                                      'area_m2', 'source', 'satellite']).to_file(
                out_file, driver='GPKG')
        else:
            gdf.to_file(out_file, driver='GPKG')
            print(f"{len(gdf)} lakes → {out_file.name}")
            total_lakes += len(gdf)

    print(f"[done] {area_name}: {total_lakes} lakes total")
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Detect glacial lakes from Landsat (MNDWI > 0.0, >= 9000 m2, within 5km glacier)\n'
                    'Usage: python scripts/05_detect_lakes_landsat.py cordillera_blanca\n'
                    '       python scripts/05_detect_lakes_landsat.py --all'
    )
    parser.add_argument('area_name', nargs='?', help='Single area name')
    parser.add_argument('--all', action='store_true', help='All areas')
    args = parser.parse_args()

    if not args.all and not args.area_name:
        parser.print_help()
        sys.exit(1)

    if args.all:
        areas = sorted(EXPANDED_STUDY_AREAS.keys())
        print(f"Lake detection (Landsat) for ALL {len(areas)} areas")
        failed = []
        for name in areas:
            try:
                ok = detect_one_area(name)
                if not ok:
                    failed.append(name)
            except Exception as e:
                print(f"[ERROR] {name}: {e}")
                failed.append(name)
        print(f"\n=== SUMMARY ===")
        print(f"Completed: {len(areas) - len(failed)}/{len(areas)}")
        if failed:
            print(f"Failed   : {failed}")
    else:
        ok = detect_one_area(args.area_name)
        sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
