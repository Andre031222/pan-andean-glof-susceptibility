import os
import shutil
from pathlib import Path

import pandas as pd

from main import Session, Lake, engine, Base

ROOT = Path(__file__).resolve().parent.parent.parent
SHEET = ROOT / 'docs' / 'inventory_validation_sheet.csv'
SRC_THUMBS = ROOT / 'docs' / 'validation_app' / 'thumbs'
DST_THUMBS = Path(os.environ.get('GLOF_THUMBS', Path(__file__).resolve().parent / 'thumbs'))


def num(v):
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def main():
    Base.metadata.create_all(engine)
    df = pd.read_csv(SHEET)
    df = df[df['lat'].notna() & df['lon'].notna()].reset_index(drop=True)
    DST_THUMBS.mkdir(parents=True, exist_ok=True)

    with Session() as s:
        s.query(Lake).delete()
        for i, r in df.iterrows():
            s.add(Lake(
                lake_key=r['lake_key'], idx=int(i), area_name=r['area_name'],
                lat=num(r['lat']), lon=num(r['lon']), area_ha=num(r['area_m2']) / 1e4 if num(r['area_m2']) else None,
                model_score=num(r['model_score']), dist_glacier_m=num(r['dist_glacier_m']),
                elev_mean=num(r['elev_mean']),
                in_watchlist=bool(int(r.get('in_watchlist_top2pct', 0) or 0)),
                known_glof=bool(int(r.get('known_glof_source', 0) or 0)),
                thumb=f'{i}.jpg'))
            src = SRC_THUMBS / f"{r.get('idx', i)}.jpg"
            if src.exists():
                shutil.copy(src, DST_THUMBS / f'{i}.jpg')
        s.commit()
    print(f'seeded {len(df)} lakes; thumbs -> {DST_THUMBS}')


if __name__ == '__main__':
    main()
