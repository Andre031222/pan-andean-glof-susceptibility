# Susceptibilidad a desbordamientos de lagos glaciares (GLOF) a escala continental en los Andes: un marco de aprendizaje automático explicable para la reducción del riesgo de desastres

**Estado:** Preparado para envío a *Natural Hazards* (Springer) — junio 2026
**Revista objetivo:** Natural Hazards (Springer), Q1 — https://link.springer.com/journal/11069
**Manuscrito:** `natural_hazards_paper/manuscript_nathaz.tex` (compilado, plantilla Springer `sn-jnl`)
**Institución:** Facultad de Ingeniería Estadística e Informática, Universidad Nacional del Altiplano (UNAP), Puno, Perú

> Versión en inglés (completa y canónica): ver [README.md](README.md).

---

## Resumen

Este repositorio contiene el pipeline de investigación, los modelos entrenados, las figuras y el manuscrito LaTeX de una evaluación de susceptibilidad a GLOF a escala continental en los Andes. El estudio compila un inventario multitemporal de lagos glaciares a partir de imágenes Sentinel-2 (2017–2025) en once cordilleras de cuatro países (12.700 observaciones lago-año de 3.608 lagos distintos; 17 eventos GLOF históricos emparejados; razón de clases 1:746), y entrena modelos de aprendizaje automático bajo una evaluación **estrictamente libre de fuga de datos** (toda la imputación y el escalado se realizan dentro de cada pliegue de validación cruzada).

Bajo validación cruzada out-of-fold (OOF) agrupada por lago, **EasyEnsemble** logró la mayor discriminación (ROC-AUC = 0,722; IC 95% [0,590, 0,830]), mientras que **Balanced Random Forest** — adoptado como modelo operativo e interpretable — dio la mejor priorización (Lift = 5,7×; Recall = 71%; ROC-AUC = 0,702; IC 95% [0,536, 0,841]). La validación Leave-One-Country-Out aportó evidencia preliminar de transferibilidad (Perú retenido ROC-AUC = 0,734).

Es el primer modelo de susceptibilidad a GLOF basado en aprendizaje automático a escala continental para los Andes, validado a través de fronteras nacionales con datos satelitales multitemporales.

---

## Resultados clave

- **Discriminación:** EasyEnsemble ROC-AUC 0,722; Balanced Random Forest 0,702 (operativo/interpretable, Lift 5,7×, Recall 71%).
- **Validación geográfica (LOCO):** Perú retenido 0,734; Ecuador 0,993 y Bolivia 0,888 (una sola etiqueta positiva, solo direccional); Chile clase nula.
- **Significancia:** permutación p = 0,0022; DeLong z = 1,77, p = 0,077; Moran's I = 0,30, p = 0,002 (autocorrelación espacial → justifica LOCO); KS D = 0,421, p ≈ 3,0e-3; jackknife AUC 0,684–0,741.
- **SHAP:** distancia al glaciar como factor dominante (mediana 512 m vs 2.357 m), con pico de riesgo no lineal a 2–15 km; elevación, pendiente y volumen/profundidad como secundarios.
- **Triaje operativo:** umbral de Youden t = 0,510; watch-list del top-2% = 254 lagos (220 en Perú) que recupera el 12% de los lagos fuente de GLOF a 5,9× la precisión aleatoria (35% dentro del top 5%) (reducción de 50× en el esfuerzo de inspección).
- **Fiabilidad del inventario:** 69,5% de los lagos re-detectados en ≥2 años, 43,5% en ≥3 (98% en Blanca, 97% en Central, 93% en Raura).

---

## Reproducibilidad (pipeline canónico sin fuga de datos)

Ejecutar desde la raíz del proyecto, en orden:

```bash
python scripts/recompute_terrain.py        # pendiente/TRI zonal por lago desde rásteres DEM
python scripts/run_full_analysis.py        # entrenamiento OOF (6 modelos), IC, permutación, DeLong, KS,
                                           # jackknife, calibración isotónica, priorización, LOCO
python scripts/improve_models.py           # tuning + stacking (verificación de resultado negativo)
python scripts/augmentation_final.py       # experimento de etiquetado (temporal vs jitter) + Fig 8
python scripts/recompute_secondary.py      # Moran's I + validación prospectiva top-100
python scripts/inventory_and_impact_stats.py  # persistencia del inventario + impacto operativo
for f in scripts/generate_fig0*.py; do python "$f"; done   # regenerar figuras analíticas
python scripts/generate_fig09_iconic.py    # imágenes Sentinel-2 de lagos GLOF emblemáticos
```

Resultados principales en `models/robustness_results.json`. Estructura completa del repositorio, tablas de resultados y detalles del pipeline: ver [README.md](README.md).

---

## Compilación del manuscrito

Desde `natural_hazards_paper/`:

```bash
latexmk -pdf -interaction=nonstopmode manuscript_nathaz.tex
pdflatex -interaction=nonstopmode cover_letter.tex
```

Compila limpio (0 errores, 0 referencias/citas indefinidas, 0 cajas desbordadas).

---

## Autores

| Autor | Rol | ORCID |
|---|---|---|
| Dina Maribel Yana-Yucra | Procesamiento de datos, validación espacial | 0009-0003-6218-2735 |
| Richar Andre Vilca-Solorzano (autor de correspondencia) | Diseño del estudio, pipeline ML, manuscrito | 0009-0003-2385-5263 |
| Milton Vladimir Mamani-Calisaya | Análisis estadístico, interpretación | 0000-0002-0676-0989 |
| Fred Torres-Cruz | Supervisión, revisión del manuscrito | 0000-0003-0834-6834 |

Facultad de Ingeniería Estadística e Informática, Universidad Nacional del Altiplano (UNAP), Puno, Perú.

---

## Disponibilidad de datos y código

Código, modelos y scripts: https://github.com/Andre031222/pan-andean-glof-susceptibility (Licencia: MIT)

Sentinel-2 (ESA Copernicus), NASADEM (NASA EARTHDATA) y RGI v7.0 (GLIMS/NSIDC) son de acceso libre; el inventario de GLOF deriva de bases publicadas (GloFLOD, Emmer et al. 2022, INAIGEM 2018).

---

*Última actualización: junio 2026*
