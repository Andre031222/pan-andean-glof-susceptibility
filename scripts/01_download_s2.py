import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config_expanded_study_areas import EXPANDED_STUDY_AREAS
from src.download import download_study_area_data

S2_YEARS = list(range(2017, 2026))
S2_BANDS = ['B02', 'B03', 'B04', 'B08', 'B11', 'B12']


def parse_year_range(s: str) -> list:
    parts = s.split('-')
    if len(parts) == 2:
        return list(range(int(parts[0]), int(parts[1]) + 1))
    return [int(p) for p in parts]


def download_one(area_name: str, years: list) -> bool:
    if area_name not in EXPANDED_STUDY_AREAS:
        print(f"[ERROR] Unknown area: {area_name}")
        print(f"  Available: {sorted(EXPANDED_STUDY_AREAS.keys())}")
        return False

    cfg = EXPANDED_STUDY_AREAS[area_name]
    sentinel_dir = PROJECT_ROOT / 'data' / 'raw' / 'sentinel2' / area_name

    years_needed = []
    for y in years:
        year_dir = sentinel_dir / str(y)
        band_files = list(year_dir.glob('*.tif')) if year_dir.exists() else []
        if len(band_files) >= len(S2_BANDS):
            print(f"  [skip] {area_name}/{y} — {len(band_files)} files exist")
        else:
            years_needed.append(y)

    if not years_needed:
        print(f"[ok] {area_name} — all years downloaded")
        return True

    print(f"\n{'='*60}")
    print(f"S2 download: {area_name}")
    print(f"  bbox   : {cfg['bbox']}")
    print(f"  months : {cfg['dry_season_months']}")
    print(f"  cloud  : {cfg['max_cloud_cover']}%")
    print(f"  years  : {years_needed}")

    try:
        results = download_study_area_data(
            area_name=area_name,
            years=years_needed,
            months=cfg['dry_season_months'],
            sentinel_bands=S2_BANDS,
            max_cloud_cover=cfg['max_cloud_cover'],
            visualize=False,
            save_metadata=True,
        )
        n = len(results.get('sentinel_years', []))
        print(f"[done] {area_name}: {n}/{len(years_needed)} years downloaded")
        return True
    except Exception as e:
        print(f"[ERROR] {area_name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Download Sentinel-2 L2A for GLOF study areas\n'
                    'Usage: python scripts/01_download_s2.py cordillera_blanca\n'
                    '       python scripts/01_download_s2.py --all\n'
                    '       python scripts/01_download_s2.py --all --years 2020-2023'
    )
    parser.add_argument('area_name', nargs='?', help='Single area name')
    parser.add_argument('--all', action='store_true', help='Download all areas')
    parser.add_argument('--years', default=None, help='Year range e.g. 2020-2023')
    args = parser.parse_args()

    years = parse_year_range(args.years) if args.years else S2_YEARS

    if not args.all and not args.area_name:
        parser.print_help()
        sys.exit(1)

    if args.all:
        areas = sorted(EXPANDED_STUDY_AREAS.keys())
        print(f"Downloading S2 for ALL {len(areas)} areas, years {years[0]}-{years[-1]}")
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
