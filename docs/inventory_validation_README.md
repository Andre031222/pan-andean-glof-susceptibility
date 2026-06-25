# Inventory validation sheet — how to fill it

File: `docs/inventory_validation_sheet.csv` (959 lakes)

Goal: an independent omission/commission check of the automated glacial-lake
inventory, to satisfy the reviewer's critical point. Prioritise the
`in_watchlist_top2pct == 1` rows first (these are the operational output).

## For each row
1. Open `google_earth_url` (or `gmaps_url` for satellite view) in a browser.
2. Look at the lake at the given `lat`,`lon`.
3. Fill the empty columns:

| Column | What to enter |
|---|---|
| `is_real_glacial_lake_YN` | `Y` if it is a genuine glacial/proglacial lake; `N` if it is not |
| `feature_type` | one of: `glacial`, `proglacial`, `moraine_dammed`, `bedrock`, `reservoir`, `river`, `wetland`, `cloud_shadow`, `snow_ice`, `other` |
| `confidence_1to3` | `1` low, `2` medium, `3` high |
| `notes` | free text (e.g. "clearly a dam wall", "seasonal pond") |

## What we compute afterwards
- **Commission rate** = fraction marked `N` (false lakes) overall and within the watch-list.
- **By size/region** = commission broken down by `area_m2` bins and `area_name`.
- Compare against the automated physiographic screen (`models/inventory_screen.csv`).

## Minimum for the paper
If full review is too long, do at least:
- ALL `in_watchlist_top2pct == 1` rows (409), plus
- ~30 random rows per main cordillera (Blanca, Huayhuash, Raura, Vilcanota,
  Patagonia N/S, Chile) — enough to report commission per region.
