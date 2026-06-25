# Estado del Proyecto — GLOF Andes

## Resumen ejecutivo

Paper de machine learning para prediccion de susceptibilidad a inundaciones por desborde glaciar (GLOFs) en los Andes.

**Revista actual:** Natural Hazards (Springer), ISSN 0921-030X  
**Manuscrito:** `natural_hazards_paper/manuscript_nathaz.tex`  
**Estado actual:** LISTO PARA ENVIO — pendiente de subir al portal de Springer  
**Email correspondencia:** 75521963@est.unap.edu.pe

**Historia de envíos:**
- ~~IJACSA (MS-17-6-0561, enviado mayo 2026)~~ → **RECHAZADO**
- Natural Hazards (Springer) → **PRÓXIMO ENVÍO**

---

## Autores (orden definitivo)

| Orden | Nombre | Email | ORCID |
|---|---|---|---|
| 1 | Dina Maribel Yana-Yucra | maribelbel201314@gmail.com | 0009-0003-6218-2735 |
| 2 | Richar Andre Vilca-Solorzano (Corresponding) | 75521963@est.unap.edu.pe | 0009-0003-2385-5263 |
| 3 | Milton Vladimir Mamani-Calisaya | mmamanic@unap.edu.pe | 0000-0002-0676-0989 |
| 4 | Fred Torres-Cruz | ftorres@unap.edu.pe | 0000-0003-0834-6834 |

Afiliacion todos: Universidad Nacional del Altiplano (UNAP), Puno, Peru  
Facultad: Faculty of Statistical and Computer Engineering

---

## Titulo del paper

"Continental-scale glacial lake outburst flood susceptibility across the Andes: an explainable machine-learning framework for disaster risk reduction"

---

## Objetivo y contribucion principal

Primer modelo de ML a escala continental que prediga que lagos glaciares andinos son fuente potencial de GLOFs, cubriendo Peru, Bolivia, Ecuador y Chile con validacion geografica rigurosa (LOCO-CV leakage-free).

### Contribuciones especificas

1. Inventario multitemporal armonizado de 12,703 detecciones de lagos glaciares (2017-2025) en once cordilleras de cuatro paises, derivado de imagenes Sentinel-2 L2A
2. Modelo de susceptibilidad con Balanced Random Forest bajo validacion lake-grouped OOF estricta y Leave-One-Country-Out (LOCO) sin leakage
3. Interpretacion fisica via SHAP: proximidad al glaciar como driver dominante, pico de riesgo no lineal a 2-15 km
4. Seis procedimientos estadisticos complementarios que confirman robustez con solo 17 etiquetas positivas
5. Pipeline reproducible y open-source disponible en GitHub

---

## Resultados clave (metricas correctas — post-fix CV leakage)

| Metrica | Valor |
|---|---|
| ROC-AUC (EasyEnsemble — mejor discriminacion) | 0.722 |
| IC 95% bootstrap (EasyEnsemble) | [0.590, 0.830] |
| ROC-AUC (BalancedRF — modelo operacional) | 0.702 |
| IC 95% bootstrap (BalancedRF) | [0.536, 0.841] |
| Lift BRF (Youden threshold) | 5.7x |
| Recall BRF | 71% (12/17 GLOFs) |
| n+ positivos | 17 de 12,700 observaciones |
| Ratio clases | 1:746 |
| Threshold Youden | t = 0.510 |
| Top-2% watch-list | 254 lagos — recupera 12% de GLOFs a 5.9x precision |
| Top-5% watch-list | 635 lagos — recupera 35% de GLOFs |
| LOCO Peru (principal) | AUC = 0.734 (n+=15, unico estadisticamente estable) |
| LOCO Ecuador (indicativo) | AUC = 0.993* (n+=1, caso de estudio) |
| LOCO Bolivia (indicativo) | AUC = 0.888* (n+=1, caso de estudio) |
| Jackknife (17 estimaciones) | [0.684, 0.741], mean=0.702, max dev=0.038 |
| Mann-Whitney dist_glacier | p = 0.037 (Bonferroni-corrected) |
| Mediana dist glaciar GLOF-source | 512 m |
| Mediana dist glaciar no-GLOF | 2,357 m |
| Permutation test | p = 0.0022 |
| KS test (score distributions) | D=0.421, p=3.0e-3 |
| Moran's I (residual autocorr.) | I=0.30, p=0.002 |

---

## Archivos del manuscrito

| Archivo | Descripcion | Estado |
|---|---|---|
| `natural_hazards_paper/manuscript_nathaz.tex` | LaTeX principal — Natural Hazards (Springer) | **LISTO** |
| `natural_hazards_paper/manuscript_nathaz.pdf` | PDF compilado | **LISTO** |
| `natural_hazards_paper/cover_letter.tex` | Carta de presentacion | **LISTO** |
| `natural_hazards_paper/cover_letter.pdf` | PDF carta | **LISTO** |
| `natural_hazards_paper/references_glof.bib` | Bibliografia | ~70 entradas |

---

## Figuras del manuscrito

| Figura | Archivo | Descripcion |
|---|---|---|
| Fig. 1 | fig1_susceptibility_map.jpg | Mapa pan-andino de susceptibilidad |
| Fig. 2 | fig2_inventory_glof_context.png | Inventario GLOF y contexto |
| Fig. 3 | fig3_susceptibility_distribution.png | Distribucion de susceptibilidad |
| Fig. 4 | fig4_model_performance.png | Performance del modelo (ROC, PR, LOCO, top-k) |
| Fig. 5 | fig5_shap_risk_drivers.png | SHAP — drivers de riesgo |
| Fig. 6 | fig6_robustness_analysis.png | Analisis de robustez |
| Fig. 7 | fig7_supplementary_validation.png | Validacion suplementaria (KS, calibracion, top-100) |
| Fig. 8 | fig8_augmentation_comparison.png | Validacion estrategia etiquetado (augmentacion) |
| Fig. 9 | fig9_iconic_lakes.jpg | Lagos iconicos (Palcacocha, Lake 513, Rasac) |
| Apendice | areas/fig_area_*.jpg | 11 mapas de areas de estudio |

---

## Revista destino: Natural Hazards (Springer)

| Campo | Informacion |
|---|---|
| Publisher | Springer Nature |
| ISSN | 0921-030X (print), 1573-0840 (online) |
| Template | sn-jnl.cls (Springer Nature, en natural_hazards_paper/) |
| Scope | Natural hazards: floods, GLOFs, landslides, risk assessment |
| Cover letter | Dirigida al Editor-in-Chief de Natural Hazards |
| Portal | https://www.springer.com/journal/11069/submission-guidelines |

---

## Pasos para enviar a Natural Hazards (Springer)

1. Compilar PDF final: `cd natural_hazards_paper && pdflatex manuscript_nathaz && bibtex manuscript_nathaz && pdflatex manuscript_nathaz && pdflatex manuscript_nathaz`
2. Verificar PDF generado (todos los refs, figuras, tablas)
3. Subir al portal de Springer (Editorial Manager o similar)
4. Adjuntar: manuscript_nathaz.tex, referencias .bib, figuras (PNG/JPG), cover letter
5. Confirmar que GitHub URL en manuscrito es el correcto (ver nota abajo)

---

## PENDIENTE — Verificar antes de enviar

- [ ] **GitHub URL**: El manuscrito usa `https://github.com/Andre031222/pan-andean-glof-susceptibility` pero el README menciona `github.com/Andre031222/GLOFs-Pan-Andina-Mediante-Machine-Learning`. Confirmar URL correcta y actualizar en el .tex si es necesario.
- [ ] Recompilar PDF tras correccion de bug LR AUC (0.608→0.465, ya corregido en .tex)
- [ ] Verificar que todos los archivos de figuras esten en `figures/publication/` y `figures/publication/areas/`

---

## Artefactos canonicos del modelo

- `data/processed/labeled/training_data.csv` — dataset correcto (n+=17, BRF AUC=0.702)
- `models/best_model.joblib` — modelo BRF (800 arboles, min_samples_leaf=2)
- `models/best_model_calibrated.joblib` — modelo calibrado (isotonic regression)
- Los archivos `training_data_v2.csv`, `labeled_lakes_v2.gpkg` y `match_log_v2.csv` fueron eliminados
  (experimento de reetiquetado con buffer graduado, descartado por idxmin() corrupto en las etiquetas)

---

## GitHub

Repositorio: github.com/Andre031222/pan-andean-glof-susceptibility  
(verificar si es este o `GLOFs-Pan-Andina-Mediante-Machine-Learning`)  
Licencia: MIT  
Pipeline completo, modelos y notebooks disponibles publicamente
