# Actualizacion de resultados — pipeline corregido (2026-06-22)

Este documento mapea los numeros VIEJOS del manuscrito a los NUEVOS tras corregir
10+ bugs del pipeline, agregar a lago unico, e incluir patagonia_norte (Cachet 2).

## Headline (numeros que YA tengo del nuevo modelo)

| Claim | VIEJO (manuscrito actual) | NUEVO (verificado) | Donde aparece (lineas .tex) |
|-------|---------------------------|--------------------|------------------------------|
| Regiones / cordilleras | eleven (11) | **15** (anadida patagonia_norte) | 60,148,241,305,309,393,697... |
| Paises | four (Peru,Bolivia,Ecuador,Chile) | four (igual; mas cobertura Chile) | 61,102,149,242 |
| Unidad de modelado | 12,700 lake-YEAR obs | **20,140 unique lakes** (lago fisico) | 59,305,316,390,696... |
| GLOFs documentados | 71 | 71 (igual) | 61,306,373 |
| GLOFs emparejados (eventos) | 17 | **38/71** | 62,373... |
| Positivos n+ (lagos) | 17 | **34** (32 tras agregacion+filtro) | n+=17 en ~30 lugares |
| Class imbalance | 1:746 | ~1:630 (20108/32) | 63,596,775 |
| Mejor discriminacion | EasyEnsemble 0.722 [0.590,0.830] | **RandomForest 0.787 [0.682,0.884]** | 66,748,785,1505 |
| Bootstrap AUC | — | 0.789 (2000 reps) | 1067 |
| Permutacion | p=0.0022 | **p<0.0001** | 663,871,1068 |
| LOCO Peru | 0.734 (n+=15) | **0.774 (n+=18)** | 70,812,856,889,1288 |
| LOCO Ecuador | 0.993 (n+=1) | 0.692 (n+=4) | 816,856,1293 |
| LOCO Bolivia | 0.888 (n+=1) | 1.000 (n+=1, no fiable) | 817,857,1293 |
| LOCO Chile | null (n+=0) | **0.626 (n+=9)** ahora SI evaluable | 826,859 |

## Modelos (tabla tab:model_comparison, lineas 785-791) — NUEVOS

| Modelo | AUC nuevo | recall | lift |
|--------|-----------|--------|------|
| RandomForest | 0.787 | 0.625 | 5.9x |
| LightGBM | 0.757 | 0.500 | 5.0x |
| BalancedRF | 0.747 | 0.469 | 5.0x |
| XGBoost | 0.741 | 0.500 | 5.0x |
| LogisticRegression | 0.713 | 0.594 | 4.7x |
| EasyEnsemble | 0.703 | 0.594 | 4.1x |

## Reencuadre narrativo sugerido (mas fuerte para Q1)
- De "AUC alto en pocas areas" -> "primera evaluacion pan-andina de 15 regiones"
- n DUPLICADO (17->34), incluido Cachet 2 (el GLOF de represa-de-hielo mas estudiado)
- Metodologia mas rigurosa: lago fisico unico (sin pseudo-replicacion), CV agrupado limpio,
  filtro de ruido (ponds <0.05 km2)
- AUC 0.787 SUPERA el rango previo, sobre un dataset mas grande y dificil

## PENDIENTE: stats derivadas que hay que RECALCULAR (no editar a mano)
Estas dependen de analisis cuyos scripts aparecen borrados (compute_calibration.py,
compute_statistical_tests.py, compute_jackknife.py, augmentation_*.py). Hay que
re-generarlas del nuevo modelo antes de tocar esas secciones del manuscrito:
- Watch-list top-2%/5% (254/635 lagos, recall 12%/35%) — tab:robustness (esta en models/robustness_results.json del nuevo run)
- Tabla de sensibilidad de umbral (tab:threshold_sens)
- Medianas SHAP / Mann-Whitney (glacier 512 vs 2357 m, volumen, etc.)
- Calibracion (BSS, Brier), Moran's I, KS test, DeLong
- % alto-riesgo por area, recall por modelo (12/17 -> recalcular con n=34)
- 3,608 distinct lakes -> nuevo conteo de lagos unicos
- Augmentation analysis

## Outputs del nuevo modelo ya disponibles
- models/model_comparison.csv (AUCs nuevos)
- models/robustness_results.json (watch-list, bootstrap, permutacion, LOCO, jackknife)
- models/loco_validation.csv
- models/best_model.joblib (RandomForest)
- figures/publication/fig05_shap_*.png (SHAP nuevo)

## Stats derivadas REGENERADAS (script 09_paper_stats.py, modelo BalancedRF OOF)

### Watch-list (tab:robustness) — NUEVO, mucho mejor
| top % | n lagos | recall | lift |
|-------|---------|--------|------|
| 0.5% | 101 | 0.281 | 56.1x |
| 1.0% | 201 | 0.375 | 37.6x |
| 2.0% | 403 | 0.406 | 20.3x |
| 5.0% | 1007 | 0.469 | 9.4x |
| 10% | 2014 | 0.500 | 5.0x |
| 20% | 4028 | 0.594 | 3.0x |
(viejo top-2%: recall 12%, lift 5.9x -> NUEVO 40.6%, 20.3x)

### Sensibilidad de umbral (tab:threshold_sens) — NUEVO
| t | % flagged | recall |
|---|-----------|--------|
| 0.20 | 71.9% | 0.844 |
| 0.30 | 44.7% | 0.719 |
| 0.40 | 22.8% | 0.625 |
| 0.50 | 9.6% | 0.500 |
| 0.60 (Youden) | 3.4% | 0.469 |

### KS test: D=0.436, p<1e-5, median score GLOF 0.505 vs non 0.278
### Moran's I = 0.111 (55km) — menor autocorrelacion que antes (0.30)
### Calibracion: Brier 0.109, BSS -67.7 SIN calibrar (PENDIENTE: aplicar isotonica como el manuscrito)

### Mann-Whitney (nuevo feature set, 21 features) — top significativos:
| feature | median GLOF | median non-GLOF | p |
|---------|-------------|-----------------|---|
| perimeter_m | 1450 | 4080 | <1e-4 |
| area_m2 | 64533 | 312300 | <1e-4 |
| elev_std | 18.3 | 71.5 | <1e-4 |
| compactness | 0.4 | 0.2 | <1e-4 |
| elongation | 1.6 | 2.1 | <1e-4 |
| area_trend | -24841 | -85376 | <1e-3 |
NOTA: GLOF lakes salen MAS PEQUENOS y MAS COMPACTOS. Distinto a la narrativa vieja
(glacier proximity + volumen). El nuevo pipeline NO tiene volumen/profundidad/Cook/batimetria
(feature set 56 -> 21). Esto cambia la INTERPRETACION del paper, requiere revision cientifica.

### % alto-riesgo por area (Youden t=0.60): raura 21.8, central 21.5, huayhuash 15.9,
### carabaya 15.6, apolobamba 11.7, ..., chile 1.6, patagonia_norte 0.9

## DIVERGENCIAS CIENTIFICAS a resolver con el usuario (no son solo numeros)
1. Feature set 56 -> 21: el manuscrito discute volumen/profundidad/Cook/hidrologicas que ya no existen
2. Top predictores cambiaron: antes glacier proximity+volumen; ahora morfometria (area/perimetro)
3. GLOF lakes ahora salen mas PEQUENOS (contra-intuitivo) -> interpretar/justificar
4. Calibracion: aplicar isotonica para BSS reportable
