# Authoritative numbers — DEM-corrected model (2026-06, post-bugfix)

Source: models/paper_stats.json, robustness_results.json, model_comparison.csv,
inventory_screen.csv. Pipeline: 15 areas, corrected NASADEM (windowing bug fixed
in 03_process_dem.py), lake-level aggregation. Terrain features re-extracted with
fixed DEMs (validated vs SRTM: 0% elevation corruption, was 34%).

## Dataset (unchanged)
- Unique lakes (>=0.05 km2): **20,140** | positives **32** (34 lake-years, 38 of 71 events matched) | 1:629

## Headline performance — OLD -> NEW
- Best model RandomForest AUC-ROC: **0.781 -> 0.826** (bootstrap 0.828, 95% CI [0.729, 0.916])
- AUC-PR (RF): 0.441 -> **0.434**
- Permutation p < 0.001 (unchanged)
- Operational BalancedRF AUC: **0.709 -> 0.794**

## Model comparison (AUC-ROC) OLD -> NEW
- RandomForest 0.781 -> **0.826**
- XGBoost     0.767 -> **0.823**
- LightGBM    0.771 -> **0.822**
- EasyEnsemble 0.707 -> **0.800**
- BalancedRF  0.709 -> **0.794**
- LogReg      0.720 -> **0.774**

## Watch-list (BalancedRF OOF) OLD -> NEW
- top 0.5%: 101 lakes, recall 0.438->**0.375**, lift 87x->**74.8x**, caught 14->**12**
- top 1.0%: 201, recall 0.438->**0.406**, lift 44x->**40.7x**
- top 2.0%: 403, recall 0.438->**0.406** (44%->**41%**, 13/32), lift 21.9x->**20.3x**
- top 5.0%: 1007, recall 0.469->**0.50**, lift 9.4x->**10.0x**
- top 10%: recall 0.469->**0.531** | top 20%: **0.688**
- Youden threshold 0.582->**0.392**; high-risk overall 2.9%->**17.0%**

## Calibration / separation OLD -> NEW
- BSS (calibrated): 0.36 -> **0.24**
- KS D: 0.44 -> **0.52** (p 3.7e-6 -> 1.6e-8); median score glof 0.526, nonglof 0.242
- Moran's I (residual): 0.07 -> **0.19** (n=2500, 55 km) -- report honestly as modest; LOCO is stricter test

## LOCO OLD -> NEW
- Peru   (n+=18): 0.75 -> **0.80** (AUC-PR 0.51)
- Chile  (n+=9):  0.77 -> **0.72** (incl. Patagonia)
- Ecuador(n+=4):  0.66 -> **0.61**
- Bolivia(n+=1):  1.00 (unstable, unchanged)
- Jackknife mean held-out proba: ~0.37 (unchanged)
- DeLong BalancedRF vs EasyEnsemble: z=-0.07, p=0.94 (ns)

## Inventory screen (commission) OLD -> NEW
- full inventory off-context: 16.1% (3238) -> **35.1% (7063 of 20140)**
- top-2% watch-list off-context: 14.4% -> **5.0%**
- top-0.5%: 7.9% -> **2.0%** | top-5%: 16.2% -> **11.2%**
- median OOF in-context vs off-context: 0.264/0.290 -> **0.276/0.188** (now in-context HIGHER; bug had it inverted)
- NEW STORY: baseline 35% off-context but watch-list only 2-5% -> prioritization concentrates ~7-17x on plausible glacial lakes

## Mann-Whitney narrative (mostly UNCHANGED -- morphometric, not DEM-dependent)
- area_m2: GLOF 64,533 vs 312,300 (p<0.001) [matches current ms]
- volume_m3: 704,527 vs 6,568,686 | equiv_diameter 286 vs 630
- depth_m: 14.4 vs 27.9 | depth_ensemble_mean 12.5 vs 24.2
- compactness: 0.370 vs 0.244 (GLOF more compact) [matches ms 0.37 vs 0.24]
- freeboard: 36.2 vs 155.3 (was stated 34 vs 144 -> update to 36 vs 155)
- dam_elev: 4380 vs 1817 | elev_std 23 vs 81 | shore_dev 1.65 vs 2.03

## Per-area high-risk % (NEW, at Youden 0.392) -- update study_areas table
raura 46.0 | central 71.0 | huayhuash 46.6 | carabaya 37.5 | vilcanota 36.8 |
blanca 35.2 | urubamba 32.7 | real 27.0 | apolobamba 21.0 | chile 16.3 |
ecuador 48.5 | huanzo 50.0 | bolivia_norte 50.0 | patagonia_norte 6.5 | patagonia_sur 5.4
