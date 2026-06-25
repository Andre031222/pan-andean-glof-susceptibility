import sys
import argparse
import calendar
import json
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds as warp_bounds

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config_expanded_study_areas import EXPANDED_STUDY_AREAS

LANDSAT_YEARS = list(range(2000, 2017))
MAX_CLOUD = 30

BAND_MAP = {
    'landsat-5': ('green', 'swir16'),
    'landsat-7': ('green', 'swir16'),
    'landsat-8': ('green', 'swir16'),
    'landsat-9': ('green', 'swir16'),
}

LS_SCALE = 0.0000275
LS_OFFSET = -0.2


def get_pc_client():
    import planetary_computer
    import pystac_client
    return pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )


def download_band_window(item, band: str, bbox_wgs84: tuple, output_path: Path, max_retries: int = 3):
    import planetary_computer

    if band not in item.assets:
        return None

    url = planetary_computer.sign(item.assets[band].href)
    west, south, east, north = bbox_wgs84

    for attempt in range(1, max_retries + 1):
        try:
            with rasterio.open(url) as src:
                l, b, r, t = warp_bounds("EPSG:4326", src.crs, west, south, east, north)
                win = from_bounds(l, b, r, t, transform=src.transform)
                if win.width < 0.5 or win.height < 0.5:
                    return None
                data = src.read(1, window=win)
                if data.size == 0:
                    return None
                profile = src.profile | {
                    'compress': 'lzw',
                    'width': data.shape[1],
                    'height': data.shape[0],
                    'transform': rasterio.windows.transform(win, src.transform),
                    'count': 1,
                }
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(data, 1)
            return output_path
        except Exception as exc:
            if output_path.exists():
                output_path.unlink()
            if attempt < max_retries:
                time.sleep(attempt * 3)
            else:
                print(f"    [ERROR] {band}: {exc}")
                return None
    return None


def compute_mndwi_and_save(green_path: Path, swir_path: Path, out_path: Path):
    with rasterio.open(green_path) as gs:
        green = gs.read(1).astype(np.float32)
        profile = gs.profile.copy()
    with rasterio.open(swir_path) as ss:
        swir = ss.read(1).astype(np.float32)

    green = green * LS_SCALE + LS_OFFSET
    swir = swir * LS_SCALE + LS_OFFSET

    denom = green + swir
    mndwi = np.where(denom != 0, (green - swir) / denom, 0.0).astype(np.float32)

    profile.update(dtype='float32', count=1, compress='lzw', nodata=-9999.0)
    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(mndwi, 1)
    return out_path


def download_one_area_year(catalog, area_name: str, cfg: dict, year: int, out_dir: Path) -> bool:
    bbox = cfg['bbox']
    months = cfg['dry_season_months']
    start_m, end_m = min(months), max(months)
    last_day = calendar.monthrange(year, end_m)[1]
    date_range = f"{year}-{start_m:02d}-01/{year}-{end_m:02d}-{last_day:02d}"

    mndwi_out = out_dir / 'mndwi.tif'
    if mndwi_out.exists() and mndwi_out.stat().st_size > 5000:
        print(f"  [skip] {area_name}/{year} — mndwi.tif exists")
        return True

    try:
        items = list(catalog.search(
            collections=['landsat-c2-l2'],
            bbox=bbox,
            datetime=date_range,
            query={
                'eo:cloud_cover': {'lt': MAX_CLOUD},
                'platform': {'in': list(BAND_MAP.keys())},
            },
        ).items())
    except Exception as e:
        print(f"  [ERROR] {area_name}/{year} search: {e}")
        return False

    if not items:
        print(f"  [none] {area_name}/{year} — no items under {MAX_CLOUD}% cloud")
        return False

    best = min(items, key=lambda x: (
        x.properties.get('eo:cloud_cover', 100),
        -x.datetime.timestamp(),
    ))
    platform = best.properties.get('platform', 'unknown')
    cloud = best.properties.get('eo:cloud_cover', 0.0)
    scene_date = best.datetime.strftime('%Y-%m-%d')

    green_band, swir_band = BAND_MAP.get(platform, ('green', 'swir16'))

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  [dl] {area_name}/{year}  {scene_date}  {platform}  cloud:{cloud:.1f}%", end=' ... ', flush=True)

    green_path = out_dir / 'green.tif'
    swir_path = out_dir / 'swir.tif'

    gp = download_band_window(best, green_band, bbox, green_path)
    sp = download_band_window(best, swir_band, bbox, swir_path)

    if gp and sp:
        compute_mndwi_and_save(green_path, swir_path, mndwi_out)
        size_kb = mndwi_out.stat().st_size / 1024
        print(f"ok ({size_kb:.0f} KB)")
        meta = {
            'scene_id': best.id,
            'platform': platform,
            'cloud_cover': cloud,
            'date': scene_date,
            'green_band': green_band,
            'swir_band': swir_band,
            'bbox': bbox,
        }
        (out_dir / 'metadata.json').write_text(json.dumps(meta, indent=2))
        return True
    else:
        print("failed")
        return False


def download_one(area_name: str, years: list) -> bool:
    if area_name not in EXPANDED_STUDY_AREAS:
        print(f"[ERROR] Unknown area: {area_name}")
        return False

    cfg = EXPANDED_STUDY_AREAS[area_name]
    print(f"\n{'='*60}")
    print(f"Landsat download: {area_name}  years {years[0]}-{years[-1]}")
    print(f"  bbox   : {cfg['bbox']}")
    print(f"  months : {cfg['dry_season_months']}")

    try:
        catalog = get_pc_client()
    except Exception as e:
        print(f"[ERROR] Cannot connect to Planetary Computer: {e}")
        return False

    success = 0
    for year in years:
        out_dir = PROJECT_ROOT / 'data' / 'raw' / 'landsat' / area_name / str(year)
        ok = download_one_area_year(catalog, area_name, cfg, year, out_dir)
        if ok:
            success += 1

    print(f"[done] {area_name}: {success}/{len(years)} years")
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Download Landsat C2 L2 (2000-2016)\n'
                    'Usage: python scripts/02_download_landsat.py cordillera_blanca\n'
                    '       python scripts/02_download_landsat.py --all\n'
                    '       python scripts/02_download_landsat.py --all --years 2000-2010'
    )
    parser.add_argument('area_name', nargs='?', help='Single area name')
    parser.add_argument('--all', action='store_true', help='All areas')
    parser.add_argument('--years', default='2000-2016', help='Year range e.g. 2000-2016')
    args = parser.parse_args()

    s, e = args.years.split('-')
    years = list(range(int(s), int(e) + 1))

    if not args.all and not args.area_name:
        parser.print_help()
        sys.exit(1)

    if args.all:
        areas = sorted(EXPANDED_STUDY_AREAS.keys())
        print(f"Downloading Landsat for ALL {len(areas)} areas, years {years[0]}-{years[-1]}")
        failed = []
        for name in areas:
            ok = download_one(name, years)
            if not ok:
                failed.append(name)
        print(f"\n=== SUMMARY ===")
        print(f"Completed: {len(areas) - len(failed)}/{len(areas)}")
        if failed:
            print(f"Failed   : {failed}")
    else:
        ok = download_one(args.area_name, years)
        sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
