import sys
import argparse
import warnings
from pathlib import Path

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import shapes, rasterize
from rasterio.warp import reproject, Resampling
from shapely.geometry import shape
from skimage.morphology import remove_small_objects, binary_closing, binary_opening, disk

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config_expanded_study_areas import EXPANDED_STUDY_AREAS

MIN_AREA_M2 = 1000.0
MAX_AREA_M2 = 50000000.0
MNDWI_THRESH = 0.1
NDWI_THRESH = 0.2
S2_SCALE = 10000.0


def read_s2_bands(year_dir: Path) -> dict:
    band_map = {}
    for f in year_dir.glob('*.tif'):
        for b in ['B02', 'B03', 'B04', 'B08', 'B11', 'B12']:
            if f'_{b}.tif' in f.name or f.name.endswith(f'_{b}'):
                band_map[b] = f
    return band_map


def read_band_array(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(1).astype(np.float32)


def read_band_resampled(path: Path, out_shape) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(1, out_shape=out_shape, resampling=Resampling.bilinear).astype(np.float32)


def compute_water_indices(bands: dict):
    with rasterio.open(bands['B03']) as src:
        green = src.read(1).astype(np.float32) / S2_SCALE
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs
    out_shape = green.shape

    mndwi = np.full_like(green, np.nan)
    ndwi = np.full_like(green, np.nan)

    if 'B11' in bands:
        swir = read_band_resampled(bands['B11'], out_shape) / S2_SCALE
        denom = green + swir
        mndwi = np.where(denom != 0, (green - swir) / denom, 0.0).astype(np.float32)

    if 'B08' in bands:
        nir = read_band_resampled(bands['B08'], out_shape) / S2_SCALE
        denom2 = green + nir
        ndwi = np.where(denom2 != 0, (green - nir) / denom2, 0.0).astype(np.float32)

    return mndwi, ndwi, profile, transform, crs


def apply_elevation_filter(water_mask: np.ndarray, transform, crs, dem_path: Path,
                           elev_min: float, elev_max: float) -> np.ndarray:
    if not dem_path.exists():
        warnings.warn(f"DEM not found: {dem_path} — skipping elevation filter")
        return water_mask

    with rasterio.open(dem_path) as dem_src:
        dem_reproj = np.full(water_mask.shape, np.nan, dtype=np.float32)
        reproject(
            source=rasterio.band(dem_src, 1),
            destination=dem_reproj,
            src_transform=dem_src.transform,
            src_crs=dem_src.crs,
            dst_transform=transform,
            dst_crs=crs,
            resampling=Resampling.bilinear,
            dst_nodata=np.nan,
        )

    elev_ok = (dem_reproj >= elev_min) & (dem_reproj <= elev_max)
    return water_mask & elev_ok


def vectorize_and_filter(water_mask: np.ndarray, transform, crs,
                         dem_path: Path, area_name: str, year: int,
                         scene_date: str) -> gpd.GeoDataFrame:
    res_m = abs(transform.a)
    min_px = max(10, int(MIN_AREA_M2 / (res_m ** 2)))

    water_clean = remove_small_objects(water_mask, min_size=min_px)
    water_clean = binary_closing(water_clean, disk(2))
    water_clean = binary_opening(water_clean, disk(2))
    water_clean = remove_small_objects(water_clean, min_size=min_px)

    if not water_clean.any():
        return gpd.GeoDataFrame()

    mask_u8 = water_clean.astype(np.uint8)
    polys = [shape(g) for g, v in shapes(mask_u8, transform=transform) if v == 1]

    if not polys:
        return gpd.GeoDataFrame()

    gdf = gpd.GeoDataFrame({'geometry': polys}, crs=crs)
    gdf['area_m2'] = gdf.geometry.area
    gdf = gdf[(gdf['area_m2'] >= MIN_AREA_M2) & (gdf['area_m2'] <= MAX_AREA_M2)].reset_index(drop=True)

    if gdf.empty:
        return gdf

    elev_stats = []
    if dem_path.exists():
        with rasterio.open(dem_path) as dem_src:
            dem_reproj = np.full(water_mask.shape, np.nan, dtype=np.float32)
            reproject(
                source=rasterio.band(dem_src, 1),
                destination=dem_reproj,
                src_transform=dem_src.transform,
                src_crs=dem_src.crs,
                dst_transform=transform,
                dst_crs=crs,
                resampling=Resampling.bilinear,
                dst_nodata=np.nan,
            )
        for geom in gdf.geometry:
            burned = rasterize(
                [(geom.__geo_interface__, 1)],
                out_shape=water_mask.shape,
                transform=transform,
                fill=0,
                dtype='uint8',
            )
            vals = dem_reproj[burned == 1]
            vals = vals[~np.isnan(vals)]
            if len(vals) > 0:
                elev_stats.append({
                    'elev_mean': float(np.mean(vals)),
                    'elev_min': float(np.min(vals)),
                    'elev_max': float(np.max(vals)),
                    'elev_std': float(np.std(vals)),
                })
            else:
                elev_stats.append({'elev_mean': np.nan, 'elev_min': np.nan,
                                   'elev_max': np.nan, 'elev_std': np.nan})
        elev_df = gpd.GeoDataFrame(elev_stats)
        gdf = gdf.join(elev_df)

    gdf['area_name'] = area_name
    gdf['year'] = year
    gdf['scene_date'] = scene_date
    gdf['source'] = 'sentinel2'
    gdf['lake_id'] = [f"{area_name}_{year}_{i:04d}" for i in range(len(gdf))]

    return gdf


def extract_scene_date(year_dir: Path) -> str:
    for f in year_dir.glob('*.tif'):
        for part in f.stem.split('_'):
            if len(part) == 8 and part.isdigit():
                return f"{part[:4]}-{part[4:6]}-{part[6:]}"
    return 'unknown'


def detect_one_year(area_name: str, cfg: dict, year: int, dem_path: Path):
    year_dir = PROJECT_ROOT / 'data' / 'raw' / 'sentinel2' / area_name / str(year)
    if not year_dir.exists():
        return None

    bands = read_s2_bands(year_dir)
    if 'B03' not in bands:
        print(f"    [skip] {year}: B03 missing")
        return None

    scene_date = extract_scene_date(year_dir)

    try:
        mndwi, ndwi, profile, transform, crs = compute_water_indices(bands)
    except Exception as e:
        print(f"    [ERROR] {year} compute_indices: {e}")
        return None

    water = (mndwi > MNDWI_THRESH)
    if not np.all(np.isnan(ndwi)):
        water = water | (ndwi > NDWI_THRESH)

    water = apply_elevation_filter(
        water, transform, crs, dem_path,
        cfg['elev_min_m'], cfg['elev_max_m'],
    )

    return vectorize_and_filter(water, transform, crs, dem_path, area_name, year, scene_date)


def detect_one_area(area_name: str) -> bool:
    if area_name not in EXPANDED_STUDY_AREAS:
        print(f"[ERROR] Unknown area: {area_name}")
        return False

    cfg = EXPANDED_STUDY_AREAS[area_name]
    out_dir = PROJECT_ROOT / 'data' / 'processed' / 'lakes_s2'
    out_dir.mkdir(parents=True, exist_ok=True)

    s2_dir = PROJECT_ROOT / 'data' / 'raw' / 'sentinel2' / area_name
    if not s2_dir.exists():
        print(f"[skip] {area_name} — no S2 data")
        return True

    dem_path = PROJECT_ROOT / 'data' / 'interim' / 'dem' / f"{area_name}_dem_utm.tif"

    years = sorted([int(d.name) for d in s2_dir.iterdir() if d.is_dir() and d.name.isdigit()])
    if not years:
        print(f"[skip] {area_name} — no year directories")
        return True

    print(f"\n{'='*60}")
    print(f"Lake detection (S2): {area_name}  {len(years)} years  elev:[{cfg['elev_min_m']},{cfg['elev_max_m']}]m")

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

        print(f"  Processing {year}...", end=' ', flush=True)
        gdf = detect_one_year(area_name, cfg, year, dem_path)

        if gdf is None or gdf.empty:
            print("0 lakes")
            gpd.GeoDataFrame(columns=['geometry', 'lake_id', 'area_name', 'year',
                                      'area_m2', 'source', 'scene_date']).to_file(
                out_file, driver='GPKG')
        else:
            gdf.to_file(out_file, driver='GPKG')
            print(f"{len(gdf)} lakes → {out_file.name}")
            total_lakes += len(gdf)

    print(f"[done] {area_name}: {total_lakes} lakes total")
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Detect glacial lakes from Sentinel-2 (MNDWI > 0.1, >= 1000 m2)\n'
                    'Usage: python scripts/04_detect_lakes_s2.py cordillera_blanca\n'
                    '       python scripts/04_detect_lakes_s2.py --all'
    )
    parser.add_argument('area_name', nargs='?', help='Single area name')
    parser.add_argument('--all', action='store_true', help='All areas')
    args = parser.parse_args()

    if not args.all and not args.area_name:
        parser.print_help()
        sys.exit(1)

    if args.all:
        areas = sorted(EXPANDED_STUDY_AREAS.keys())
        print(f"Lake detection (S2) for ALL {len(areas)} areas")
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
