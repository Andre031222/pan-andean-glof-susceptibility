# Continental-Scale Glacial Lake Outburst Flood Susceptibility Across the Andes: An Explainable Machine-Learning Framework for Disaster Risk Reduction

**Status:** Prepared for submission to *Natural Hazards* (Springer) — June 2026
**Target journal:** Natural Hazards (Springer), Q1 — https://link.springer.com/journal/11069
**Manuscript:** `natural_hazards_paper/manuscript_nathaz.tex` (compiled, Springer `sn-jnl` template)
**Institution:** Faculty of Statistical and Computer Engineering, Universidad Nacional del Altiplano (UNAP), Puno, Peru

---

## Overview

This repository contains the full research pipeline, trained models, figures, and LaTeX manuscript for a continental-scale glacial lake outburst flood (GLOF) susceptibility assessment across the Andes. The study compiles a multitemporal glacial lake inventory from Sentinel-2 imagery (2017–2025) across eleven cordilleras in four countries (12,700 lake-year observations from 3,608 distinct lakes; 17 matched historical GLOF events; class ratio 1:746), and trains machine-learning models to identify lakes with elevated GLOF susceptibility under a **strictly leakage-free** evaluation (all imputation and scaling performed within each cross-validation fold).

Under out-of-fold (OOF) cross-validation, **EasyEnsemble** achieved the highest discrimination (ROC-AUC = 0.722; 95% CI [0.590, 0.830]), while **Balanced Random Forest** — adopted as the operational, interpretable model — delivered the best prioritization (Lift = 5.7×, Recall = 71%, ROC-AUC = 0.702; 95% CI [0.536, 0.841]). Leave-One-Country-Out validation provided preliminary evidence of transferability (held-out Peru ROC-AUC = 0.734).

This is the first continental-scale, machine-learning-based GLOF susceptibility model for the Andes validated across national boundaries with multitemporal satellite data.

---

## Reproducibility (canonical leak-free pipeline)

Run from the project root, in order. All preprocessing is fold-internal (no data leakage).

```bash
python scripts/recompute_terrain.py        # zonal slope/TRI/ruggedness per lake from DEM rasters
python scripts/run_full_analysis.py        # OOF training (6 models), CI, permutation, DeLong, KS,
                                           # jackknife, isotonic calibration, prioritization, LOCO
python scripts/improve_models.py           # tuning + stacking comparison (negative-result check)
python scripts/augmentation_final.py       # labelling-strategy (temporal vs jitter) experiment + Fig 8
python scripts/recompute_secondary.py      # Moran's I + prospective top-100 validation
python scripts/inventory_and_impact_stats.py  # inventory persistence + operational-impact numbers
for f in scripts/generate_fig0*.py; do python "$f"; done   # regenerate analytical Figures
python scripts/generate_fig09_iconic.py    # Sentinel-2 imagery of iconic GLOF lakes
```

Key outputs land in `models/` (`robustness_results.json`, `oof_predictions.json`,
`model_comparison.csv`, …) and `figures/publication/`. Headline results live in
`models/robustness_results.json`. Superseded/experimental scripts are kept in
`scripts/_deprecated/`.

---

## Abstract

Glacial lake outburst floods (GLOFs) are among the most destructive cryospheric hazards in the Andes, yet disaster-risk managers lack a scalable way to decide which lakes most urgently require inspection. We present the first continental-scale GLOF susceptibility model for the Andes, trained on 12,700 glacial lake-year observations from multitemporal Sentinel-2 imagery (2017–2025) across eleven cordilleras in four countries (Peru, Bolivia, Ecuador, Chile). Of 71 compiled historical GLOFs (1932–2023), 17 spatially matched the satellite inventory — a severe 1:746 class imbalance treated as a data-scarcity regime via balanced ensemble learning. Under strict lake-grouped out-of-fold cross-validation with leakage-free within-fold preprocessing, EasyEnsemble reached the highest discrimination (ROC-AUC = 0.722; 95% CI 0.590–0.830), while Balanced Random Forest — the operational, interpretable model — gave the best prioritization (Lift = 5.7×; Recall = 71%). Leave-One-Country-Out validation provided preliminary evidence of transferability (held-out Peru ROC-AUC = 0.734). SHAP analysis identified glacier proximity, elevation, and terrain steepness as leading predictors. A compact top-2% watch-list of 254 lakes — a 50-fold cut in survey burden — recovers 12% of confirmed GLOF-source lakes at 5.9× random precision (35% within the top 5%), giving resource-constrained Andean authorities an actionable, transferable disaster-risk-reduction tool.

**Keywords:** GLOF susceptibility; disaster risk reduction; Balanced Random Forest; Sentinel-2; SHAP interpretability; Andes; class imbalance; geographic cross-validation

---

## Key Results

### Model Performance (OOF Cross-Validation, 56 features, leak-free)

| Classifier | ROC-AUC | PR-AUC | Lift | MCC |
|---|---|---|---|---|
| **Easy Ensemble** | **0.722** | 0.0040 | 3.0× | 0.029 |
| **Balanced Random Forest** † | 0.702 | 0.0076 | **5.7×** | 0.034 |
| XGBoost | 0.610 | 0.0028 | 2.1× | 0.023 |
| Random Forest | 0.609 | 0.0030 | 2.3× | 0.017 |
| LightGBM | 0.587 | 0.0028 | 2.1× | 0.021 |
| Logistic Regression | 0.465 | 0.0110 | 8.2× | 0.035 |

† Operational/interpretable model (best recall-weighted Lift; tree-ensemble amenable to TreeSHAP).
Bootstrap 95% CI: BalancedRF [0.536, 0.841]; EasyEnsemble [0.590, 0.830]. Permutation test p = 0.0022.

### Geographic Validation (LOCO-CV)

| Country withheld | n_test | n⁺ | ROC-AUC | Lift | FPR@0.510 |
|---|---|---|---|---|---|
| Ecuador | 272 | 1* | 0.993 | 90.7× | 38.7% |
| Bolivia | 117 | 1* | 0.888 | 8.4× | 11.2% |
| Peru | 10,744 | 15 | 0.734 | 3.4× | 13.8% |
| Chile | 1,567 | 0† | null-class | — | 21.0% |

*Single positive test label: ROC-AUC is a directional indicator only. †Null-class domain.

### Statistical Validation

| Test | Result | Interpretation |
|---|---|---|
| Permutation test (5,000 shuffles) | p = 0.0022 | Rejects null of chance performance |
| DeLong test (BRF vs LightGBM) | z = 1.77, p = 0.077 | No significant difference (CIs overlap) |
| Moran's I (OOF residuals) | I = 0.30, p = 0.002 | Significant spatial autocorrelation → justifies LOCO |
| KS test (score distributions) | D = 0.421, p ≈ 3.0e-3 | Significant GLOF vs non-GLOF separation |
| Mann-Whitney U (7 features) | p < 0.05 (Bonferroni) | Dominant features confirmed |
| Jackknife (leave-one-GLOF-out) | AUC 0.684–0.741, max drop 0.038 | No single label dominates |

### Score Calibration

Post-hoc isotonic regression on OOF scores:
- Brier Skill Score: BSS = −144.8 (raw) → BSS = +0.005 (calibrated)
- Ranking preserved (KS D = 0.421 unchanged)

### Susceptibility Map and Operational Triage

- Youden-optimal threshold: t = 0.510
- High-susceptibility lake-years: 3,626 of 12,700 (28.6%)
- **Top-2% watch-list: 254 lakes (220 in Peru) → recovers 12% of GLOF-source lakes at 5.9× lift; 35% within top 5% (50× reduction in survey burden)**
- Highest high-risk fractions: Raura 47.0%, Blanca 40.5%, Antisana 32.0%, Huayhuash 32.0%

### Inventory Reliability (multitemporal persistence)

- 3,608 distinct lakes; 69.5% re-detected in ≥2 years, 43.5% in ≥3 years
- Persistence highest in the GLOF-active cordilleras (Blanca 98%, Central 97%, Raura 93% re-detected ≥2 yr)

### Leading SHAP Predictors

1. Glacier distance (m) — proximity to ice source; non-linear risk peak at 2–15 km (median 512 m for GLOF-source vs 2,357 m for non-GLOF lakes)
2. Elevation — gravitational/energy context
3. Terrain steepness (slope, TRI) — slope-driven triggering
4. Lake volume / depth — ~55× larger volume for GLOF-source lakes

---

## Repository Structure

```
03.-ML_BasedPanAndean/
|
|-- natural_hazards_paper/
|   |-- manuscript_nathaz.tex          Main manuscript (Springer sn-jnl, ~41 pages)
|   |-- manuscript_nathaz.pdf          Compiled PDF
|   |-- cover_letter.tex / .pdf        Submission cover letter to Natural Hazards
|   `-- references_glof.bib            BibTeX database (118 entries)
|
|-- figures/
|   |-- publication/                   Publication-quality figures (600 DPI PNG + PDF)
|   |   |-- fig1_susceptibility_map.*          Pan-Andean susceptibility map
|   |   |-- fig2_inventory_glof_context.*      Lake inventory and GLOF context
|   |   |-- fig3_susceptibility_distribution.* Susceptibility score distributions
|   |   |-- fig4_model_performance.*           ROC, PR, LOCO, Top-K curves
|   |   |-- fig5_shap_risk_drivers.*           SHAP beeswarm and dependence plots
|   |   |-- fig6_robustness_analysis.*         Bootstrap CI, permutation, lift/recall
|   |   |-- fig7_supplementary_validation.*    KS test, calibration, top-100 map
|   |   |-- fig8_augmentation_comparison.*     Labelling-strategy experiment
|   |   |-- fig9_iconic_lakes.*                Sentinel-2 imagery of iconic GLOF lakes
|   |   `-- areas/                             11 individual cordillera zoom maps (Appendix A)
|   `-- panels/                        Intermediate sub-panels (generated by figure scripts)
|
|-- models/
|   |-- robustness_results.json        Headline results: AUCs, CIs, Youden t, prioritization, LOCO
|   |-- oof_predictions.json           y_true, y_score_brf, y_score_lgbm (12,700 rows)
|   |-- model_comparison.csv           OOF metrics for all 6 classifiers
|   |-- glof_verification_worksheet.csv 71-event match audit
|   `-- *.joblib                       Trained models (BalancedRF, EasyEnsemble, etc.)
|
|-- scripts/
|   |-- run_full_analysis.py           Canonical leak-free pipeline (single source of truth)
|   |-- recompute_terrain.py           Zonal slope/TRI from DEM rasters
|   |-- recompute_secondary.py         Moran's I + prospective top-100
|   |-- improve_models.py              Tuning + stacking (negative-result check)
|   |-- augmentation_final.py          Temporal vs jitter augmentation + Fig 8
|   |-- inventory_and_impact_stats.py  Inventory persistence + operational-impact numbers
|   |-- glof_match_audit.py            71-event match audit
|   |-- generate_fig0*.py              Analytical figure generators
|   |-- generate_fig09_iconic.py       Sentinel-2 iconic-lake figure
|   `-- _deprecated/                   Superseded scripts
|
|-- data/
|   `-- processed/                     Feature matrices, labeled datasets (training_data.csv)
|
|-- src/                               Shared download / validation / visualization utilities
|-- requirements.txt / requirements_win.txt   Dependency lists
`-- README.md                         This file
```

---

## Environment Setup

The project targets Python 3.11. Install dependencies (Windows users: use `requirements_win.txt`, which omits cupy):

```bash
pip install -r requirements_win.txt
```

### Key Dependencies

| Package | Purpose |
|---|---|
| scikit-learn | Machine learning pipeline |
| imbalanced-learn | BalancedRandomForest, EasyEnsemble, SMOTETomek |
| shap | TreeExplainer interpretability |
| geopandas / rasterio / rasterstats | Spatial operations, zonal statistics |
| lightgbm / xgboost | Gradient-boosting classifiers |
| contextily | Satellite/basemap tiles |
| matplotlib / pandas / numpy | Figures and data processing |

A `matplotlibrc` in the project root sets the unified figure style.

---

## Compiling the Manuscript (MiKTeX / TeX Live)

From `natural_hazards_paper/`:

```bash
latexmk -pdf -interaction=nonstopmode manuscript_nathaz.tex
pdflatex -interaction=nonstopmode cover_letter.tex
```

The manuscript compiles cleanly (0 errors, 0 undefined references/citations, 0 overfull boxes).

---

## Study Areas

| Cordillera | Country | Latitude | Lake-years | n_GLOF | High-risk (%) |
|---|---|---|---|---|---|
| Blanca | Peru | 8–10°S | 3,083 | 9 | 40.5 |
| Vilcanota | Peru | 13–15°S | 2,770 | 0 | 21.0 |
| Central | Peru | 11–12°S | 2,239 | 0 | 19.7 |
| Raura | Peru | 10–11°S | 1,361 | 2 | 47.0 |
| Andes Centrales | Chile | 33–35°S | 1,559 | 0 | 19.6 |
| Huanzo | Peru | 15°S | 496 | 0 | 19.4 |
| Urubamba | Peru | 13°S | 430 | 0 | 22.6 |
| Huayhuash | Peru | 10°S | 278 | 3 | 32.0 |
| Antisana | Ecuador | 0.3°N–S | 272 | 1 | 32.0 |
| Real | Bolivia | 15–17°S | 108 | 1 | 18.5 |
| Carabaya | Peru | 14–15°S | 87 | 1 | 8.0 |
| **Total** ‡ | | | **12,700** | **17** | **28.6** |

‡ Total includes 17 lake-year observations from Apolobamba (n=9) and Patagonia Sur (n=8), added in the extended pipeline with zero matched GLOFs. GLOF inventory: 71 compiled historical events (1932–2023), 17 matched within 5,000 m of a lake-year observation (class ratio 1:746).

---

## Feature Engineering

56 features in five groups (a composite risk score was excluded for near-perfect multicollinearity, R² = 0.999):

| Group | Count | Examples |
|---|---|---|
| Morphometric | 9 | area, perimeter, compactness, elongation, equivalent diameter |
| Topographic | 13 | elevation (mean/min/max/std), slope, TRI, freeboard, dam height |
| Hydrological/Depth | 12 | depth from 4 empirical models (Cook, Huggel, Yao, O'Connor), ensemble mean/std, volume |
| Temporal | 12 | annual area 2017–2025, growth rate, total change, relative growth |
| Engineered | 10 | log-transforms, slope × log(area) interaction, glacier-absence indicator |

Missing values were imputed with column-wise medians computed **within each training fold** (no leakage). Terrain derivatives were recomputed from DEM rasters (an earlier version had ~50% missing slope/TRI).

---

## Machine Learning Pipeline

- **Imbalance strategy:** BalancedRandomForest (per-tree bootstrap) and EasyEnsemble; SMOTETomek applied only within training folds
- **Validation:** Stratified 5-fold out-of-fold (OOF) cross-validation, fully leakage-free
- **Geographic validation:** Leave-One-Country-Out (LOCO-CV)
- **Interpretability:** SHAP TreeExplainer (on BalancedRF; EasyEnsemble is not TreeSHAP-compatible)
- **Calibration:** Post-hoc isotonic regression on OOF scores
- **Robustness:** bootstrap CI, permutation, DeLong, KS, Mann-Whitney U, Moran's I, jackknife

---

## Limitations

1. The matched GLOF inventory contains 17 positive labels (of 71 compiled events). The remaining 54 are unmatched because they either pre-date the 2017 Sentinel-2 baseline or carry coordinates corresponding to downstream impact zones rather than source-lake centroids — a systematic characteristic of historical GLOF catalogs. A 71-event audit (supplementary worksheet) shows only ~3 are plausibly recoverable; dedicated field-coordinate validation could grow the set to n⁺ ≈ 20.
2. Sentinel-2 at 10 m does not resolve sub-pixel ice-cored moraine-dam features.
3. The model captures **structural susceptibility**, not real-time triggering (no seismicity/precipitation inputs); it is a prior risk layer, not a dynamic forecast.
4. The temporal scope (2017–2025) excludes pre-Sentinel-2 lakes.
5. The inventory was not validated against an independent reference; multitemporal persistence (>93% in GLOF-active cordilleras) supports reliability, but a stratified independent accuracy assessment (≥50 lakes/cordillera) is recommended as future work.

---

## Authors and Affiliations

| Author | Role | ORCID |
|---|---|---|
| Dina Maribel Yana-Yucra | Data processing, spatial validation | 0009-0003-6218-2735 |
| Richar Andre Vilca-Solorzano (corresponding) | Study design, ML pipeline, manuscript | 0009-0003-2385-5263 |
| Milton Vladimir Mamani-Calisaya | Statistical analysis, interpretation | 0000-0002-0676-0989 |
| Fred Torres-Cruz | Supervision, manuscript revision | 0000-0003-0834-6834 |

Faculty of Statistical and Computer Engineering, Universidad Nacional del Altiplano (UNAP), Puno, Peru.

---

## Citation

Yana-Yucra, D.M., Vilca-Solorzano, R.A., Mamani-Calisaya, M.V., and Torres-Cruz, F. (2026).
Continental-scale glacial lake outburst flood susceptibility across the Andes: an explainable
machine-learning framework for disaster risk reduction. Submitted to *Natural Hazards* (Springer).
DOI: [assigned upon acceptance]

---

## Data and Code Availability

Code, trained models, and figure-generation scripts:
https://github.com/Andre031222/pan-andean-glof-susceptibility (License: MIT)

- Sentinel-2 imagery: ESA Copernicus (free, open access)
- NASADEM DEM: NASA EARTHDATA (free, open access)
- RGI v7.0 glacier outlines: GLIMS/NSIDC (free, open access)
- GLOF inventory: GloFLOD (Lütschg et al., 2023), Emmer et al. (2022), INAIGEM (2018)

---

## License

Code: MIT License. Data derived from ESA Copernicus is subject to Copernicus open-access terms; GLOF inventory data derive from published open-access databases (see citations).

---

*Last updated: June 2026*
