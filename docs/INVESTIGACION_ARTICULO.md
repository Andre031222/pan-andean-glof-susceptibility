# Plan de Investigación y Artículo Científico
## Proyecto GLOF Andes — Documento Maestro

**Versión:** 2.1 — Actualizado a IJACSA (mayo 2026)
**Autores:** Vilca Solorzano et al. — UNAP Puno, Perú
**Revista objetivo:** International Journal of Advanced Computer Science and Applications (IJACSA) — Scopus — SAI Organization

---

## 1. Tipo de Investigación

**Categoría:** Investigación cuantitativa aplicada con validación empírica
**Enfoque:** Teledetección + Machine Learning + Interpretabilidad física
**Alcance:** Pan-Andino (4 países, 10 cordilleras, 0.3°N–33.5°S)
**Tipo de estudio:** Transversal retrospectivo con componente temporal (serie 2017–2025)
**Paradigma:** Hipotético-deductivo (3 hipótesis testadas con evidencia estadística)

Este estudio desarrolla el **primer modelo de susceptibilidad a GLOFs (Glacial Lake Outburst Floods) validado trans-nacional para los Andes**, combinando inventario multitemporal de lagos glaciares derivado de imágenes satelitales Sentinel-2, 57 características morfométricas/topográficas/hidrológicas/temporales, aprendizaje automático con balance de clases, y validación geográfica cruzada Leave-One-Country-Out (LOCO-CV).

---

## 2. Título del Artículo

### Título oficial (en el manuscrito actual)
> *Spatially Explicit Glacial Lake Outburst Flood Susceptibility Assessment in the Tropical Andes Using Multitemporal Sentinel-2 Imagery and Interpretable Machine Learning*

### Análisis del título actual
- [OK] Describe el método (Sentinel-2 + ML interpretable)
- [OK] Menciona el ámbito geográfico
- [!] "Tropical Andes" es incorrecto: el estudio incluye Andes subtropicales (Chile, 33–35°S)
- [!] "Spatially Explicit" es redundante en un estudio geoespacial
- [!] No menciona la validación geográfica cruzada (contribución clave)

### Título recomendado (versión mejorada)
> **Pan-Andean Glacial Lake Outburst Flood Susceptibility Mapping from Multitemporal Sentinel-2 Imagery and Interpretable Machine Learning with Geographic Cross-Validation**

**Ventajas del nuevo título:**
- "Pan-Andean" cubre correctamente los 4 países (Ecuador a Chile)
- "Multitemporal Sentinel-2" es específico y citable
- "Geographic Cross-Validation" diferencia el estudio de trabajos anteriores
- 18 palabras (dentro del rango NHESS)

### Título alternativo (más conciso)
> **A Machine Learning Framework for Pan-Andean GLOF Susceptibility: Multitemporal Sentinel-2 Inventory and Leave-One-Country-Out Validation**

### Running title (máx. 60 caracteres — NHESS)
> Pan-Andean GLOF Susceptibility Using Machine Learning

---

## 3. Objetivo General

Desarrollar y validar geográficamente el primer modelo de susceptibilidad a inundaciones por vaciamiento de lagos glaciares (GLOFs) a escala pan-Andina, a partir de un inventario multitemporal de lagos derivado de imágenes Sentinel-2 (2017–2025) y aprendizaje automático interpretable, para identificar los controles físicos dominantes y proveer una herramienta de priorización operativa para la gestión de riesgo glaciar bajo el contexto de retiro glaciar acelerado por cambio climático.

---

## 4. Objetivos Específicos

1. **Construir un inventario multitemporal de lagos glaciares pan-Andino** a partir de composites anuales Sentinel-2 L2A (2017–2025) aplicando los índices NDWI/MNDWI con filtrado morfológico, cubriendo 10 cordilleras en 4 países (Perú, Bolivia, Ecuador, Chile) con vinculación al Randolph Glacier Inventory v7.0.

2. **Extraer 58 características** morfométricas, topográficas, hidrológicas y temporales por observación lago-año, incluyendo cuatro estimaciones empíricas de profundidad (Cook, Huggel, Yao, O'Connor) y su incertidumbre de ensemble, integrando datos NASADEM a 30 m.

3. **Compilar y georreferenciar un inventario de 52 eventos GLOF históricos** (1932–2021) sintetizando las bases GloFLOD, Emmer (2022), INAIGEM y literatura regional, y asignar etiquetas de susceptibilidad positiva mediante buffer espacial de 5 km.

4. **Entrenar y comparar seis clasificadores de machine learning** (Balanced Random Forest, Easy Ensemble, Random Forest, XGBoost, LightGBM, Regresión Logística) bajo validación out-of-fold (OOF) estrictamente estratificada, apropiada para el régimen de desbalance extremo de clases (1:787).

5. **Validar la transferabilidad geográfica del modelo** mediante Leave-One-Country-Out cross-validation (LOCO-CV) para los tres países con etiquetas GLOF (Perú, Bolivia, Ecuador), evaluando la capacidad predictiva del modelo sin datos de entrenamiento locales.

6. **Interpretar físicamente los controladores de susceptibilidad** mediante SHAP (SHapley Additive exPlanations) con TreeExplainer, identificando relaciones no lineales entre features y susceptibilidad consistentes con la teoría de inestabilidad de diques de morrena.

---

## 5. Hipótesis de Investigación

| # | Hipótesis | Estado | Resultado |
|---|-----------|--------|-----------|
| **H1** | Los lagos con razón área/profundidad por encima de un umbral crítico tienen mayor probabilidad GLOF que los que están por debajo | **Validada** | Umbral identificado: feature `depth_ens_std` (3er más importante SHAP) |
| **H2** | La geometría del lago (proporciones) es mejor predictor de susceptibilidad que el tamaño absoluto | **Validada** | Features geométricas > `area` en importancia SHAP |
| **H3** | Existe una "zona crítica" de distancia glaciar donde el riesgo GLOF se maximiza, reflejando el balance entre alcance de avalanchas de hielo y capacidad de buffering | **Validada** | Pico no lineal: 2–15 km del frente glaciar (consistente con teoría de morrenas) |

---

## 6. Preguntas de Investigación

1. ¿Qué características físicas de los lagos glaciares andinos son los predictores dominantes de susceptibilidad a GLOFs?
2. ¿Puede un modelo entrenado en los Andes Peruanos generalizar a Bolivia, Ecuador y Chile sin datos de entrenamiento locales?
3. ¿Cuáles cordilleras andinas concentran la mayor fracción de lagos de alto riesgo bajo el escenario de retiro glaciar actual?
4. ¿Qué algoritmo de ML es más apropiado para el régimen de desbalance extremo (1:787) inherente a los datasets de susceptibilidad a GLOFs?

---

## 7. Palabras Clave

### Palabras clave primarias (para indexación)
1. `glacial lake outburst floods` (GLOFs)
2. `GLOF susceptibility`
3. `Andes`
4. `Sentinel-2`
5. `machine learning`
6. `Balanced Random Forest`
7. `SHAP`
8. `Leave-One-Country-Out validation`
9. `glacial lake inventory`
10. `cryospheric hazards`

### Palabras clave en español (para indexación en español)
1. inundaciones por vaciamiento de lagos glaciares
2. susceptibilidad a GLOFs
3. inventario de lagos glaciares
4. teledetección multitemporal
5. aprendizaje automático interpretable
6. validación geográfica cruzada
7. Andes tropicales y subtropicales
8. peligros criosféricos

---

## 8. Fuentes de Datos

| Dato | Fuente | Acceso | Resolución |
|------|--------|--------|-----------|
| Imágenes Sentinel-2 L2A | Microsoft Planetary Computer STAC API | Libre | 10 m / 20 m |
| DEM NASADEM | NASA EARTHDATA / Planetary Computer | Libre | 30 m (~1 arcsec) |
| Inventario glaciar | Randolph Glacier Inventory v7.0 (RGI2023) | Libre | Vectorial |
| Eventos GLOF globales | GloFLOD database (Lützow et al., 2023) | Libre | Punto geocodificado |
| Eventos GLOF Andes tropicales | Emmer et al. (2022) | Publicado | Punto geocodificado |
| Eventos GLOF Perú | INAIGEM (2018) | Institucional | Punto geocodificado |
| Límites administrativos | Natural Earth / GADM | Libre | 1:10M |
| Imágenes de referencia (satélite) | CartoDB / OpenStreetMap | CC BY / ODbL | Mapa base |

**Período de datos:** 2017–2025 (9 años, estación seca Jun–Sep hemisferio sur; Dic–Mar Ecuador)
**Cobertura espacial:** 0.3°N – 33.5°S, Andes occidentales y orientales
**Tamaño mínimo de lago:** 1,000 m² (0.1 ha)
**Buffer glaciar para inclusión:** 5 km del RGI v7.0

---

## 9. Metodología — Flujo del Pipeline

```
FASE 1: Descarga de datos
   Sentinel-2 L2A (10 cordilleras × 9 años) + NASADEM + RGI7.0
   → NB01–10: descarga por área de estudio (Planetary Computer STAC API)

FASE 2: Procesamiento
   DEM → slope, TRI, freeboard, flow accumulation
   → NB11: dem_processing.ipynb

   Detección de lagos: MNDWI threshold > 0.0 + morfología + filtros
   → NB12: lake_detection.ipynb → 12,588 detecciones lago-año

   Extracción de 58 features por lago-año
   → NB13: feature_extraction.ipynb

   Inventario GLOF + asignación de etiquetas (buffer 5 km)
   → NB14: historical_glofs.ipynb → 16 positivos, 1:787 imbalance

FASE 3: Modelado
   6 clasificadores × OOF 5-fold estratificado
   → NB15: model_training.ipynb → BalancedRF ganador (ROC-AUC=0.821)

   Análisis de umbrales (H1/H2/H3)
   → NB16: threshold_analysis.ipynb

   Interpretabilidad SHAP (TreeExplainer)
   → NB17: shap_interpretation.ipynb → distancia glaciar: driver dominante

   Visualizaciones publication-quality (4 figuras)
   → NB18: visualization.ipynb → PDF 300 DPI

FASE 4: Manuscrito
   → ijacsa_paper/manuscript_ijacsa.tex (COMPLETO, compilado)
```

---

## 10. Contribuciones Científicas Principales

1. **Primero en escala pan-Andina:** Ningún estudio previo modela la susceptibilidad GLOF cubriendo simultáneamente Ecuador, Perú, Bolivia y Chile con validación cruzada nacional.

2. **Validación geográfica rigurosa:** El LOCO-CV (Leave-One-Country-Out) demuestra que el modelo extrae señales de susceptibilidad transferibles, no artefactos específicos de país (Lift >3.7× en todos los países retenidos).

3. **Manejo de desbalance extremo:** La comparación de 6 clasificadores con OOF estrictamente estratificado para razón 1:787 establece un benchmark metodológico para futuros estudios de susceptibilidad con datos escasos.

4. **Interpretación física con SHAP:** La identificación del pico de riesgo no lineal a 2–15 km del frente glaciar vincula el output del ML con mecanismos físicos documentados de inestabilidad de morrenas (Clague & Evans 2000, Westoby et al. 2014).

5. **Pipeline reproducible y open-source:** Código, modelos entrenados y pipeline de generación de figuras disponibles libremente.

---

## 11. Estructura del Artículo (NHESS)

| Sección | Contenido | Estado |
|---------|-----------|--------|
| **Abstract** | Dataset 12,588 lagos, método, ROC-AUC 0.821, LOCO, distancia glaciar, mapa | Completo |
| **1. Introduction** | Context cambio climático, GLOFs Andes, gap regional, 3 contribuciones | Completo |
| **2. Related Work** | GLOF susceptibility, ML para peligros, RS de lagos glaciares | Completo |
| **3. Study Area** | 10 cordilleras, 4 países, rango latitudinal 0.3°N–33.5°S | Completo |
| **4. Data and Methods** | Inventario, 57 features, pipeline ML, OOF, LOCO, SHAP, tests | Completo |
| **5. Results** | Inventario, métricas, LOCO, Mann-Whitney, SHAP, mapa | Completo |
| **6. Discussion** | Desbalance, interpretación física, transferabilidad, limitaciones | Completo |
| **7. Conclusions** | 6 hallazgos principales, herramienta operativa | Completo |
| **Code/Data Availability** | GitHub + Copernicus OA Hub | Completo |
| **Author Contributions** | RAVS: concepción + análisis + escritura | Completo |
| **References** | ~70 referencias en ijacsa_paper/references_glof.bib | Completo |
| **Table S1** (supplementary) | Inventario 52 eventos GLOF | Pendiente |

---

## 12. Resultados Clave (Resumen Ejecutivo)

### Dataset
- **12,588** observaciones lago-año en 10 cordilleras
- Lagos detectados: rango de área 1,000 m² – >1 km²
- Expansión generalizada de área de lagos 2017–2025

### Modelo
- **BalancedRF** supera a 5 competidores → ROC-AUC = **0.821** (95% CI: [0.737, 0.901]), Lift = **8.5×**
- Desbalance extremo (1:787) manejado con bootstrap balanceado interno
- OOF garantiza que los 16 positivos contribuyen a la evaluación
- Test permutación: p < 0.0002

### Validación geográfica
- Ecuador: ROC-AUC = **0.993**, Lift = **68.2×** (n⁺=1; indicativo)
- Bolivia: ROC-AUC = **0.740**, Lift = **3.7×** (n⁺=1; indicativo)
- Perú: ROC-AUC = **0.707**, Lift = **4.2×** (n⁺=14; 2 positivos en entrenamiento)

### Mapa de susceptibilidad
- **40.0%** de lagos → alto riesgo (5,030 de 12,588) con umbral Youden t=0.304
- Cordilleras de mayor riesgo: Raura (55.7%), Huayhuash (54.0%), Blanca (51.5%)
- Cordilleras de menor riesgo: Cordillera Central (28.9%), Andes Centrales Chile (29.6%)

### Controles físicos (SHAP)
- **Distancia al glaciar** = driver dominante → pico no lineal a 2–15 km
- **Perímetro** (complejidad morfológica) = 2do más importante
- **Incertidumbre de profundidad** = proxy de morfología inusual / riesgo estructural

---

## 13. Limitaciones del Estudio

1. Solo 16 etiquetas positivas confirmadas (GLOF emparejados) → desbalance 1:787 extremo
2. Resolución Sentinel-2 a 10 m no resuelve features sub-pixel de diques con núcleo de hielo
3. El modelo no incorpora disparadores dinámicos (sismicidad, precipitaciones extremas) → es susceptibilidad estructural, no probabilidad de evento
4. Alcance temporal 2017–2025 excluye lagos formados y vaciados antes de la era Sentinel-2
5. El inventario GLOF histórico (52 eventos) está incompleto, especialmente en cordilleras menos estudiadas

---

## 14. Trabajo Futuro

- Integrar topografía del lecho glaciar (GlabTop/Farinotti2019) para simular lagos futuros bajo escenarios CMIP6
- Incorporar triggers dinámicos: índice de precipitación extrema, catálogo sísmico
- Extender a Patagonia y Himalaya para testear transferabilidad global
- Desarrollar sistema de monitoreo operativo con Sentinel-2 new acquisitions
- Completar Supplementary Table S1 con los 52 eventos GLOF georreferenciados

---

## 15. Estado del Proyecto y Próximos Pasos

### Completado
- Pipeline completo (NB00–NB18, 19 notebooks)
- Dataset: 12,588 lagos, 58 features, 16 etiquetas positivas
- Modelos entrenados y guardados (6 clasificadores)
- LOCO-CV ejecutado (3 países)
- SHAP analysis completo
- 4 figuras publication-quality (PDF 300 DPI)
- Manuscrito LaTeX completo y compilado (PDF existe)
- Bibliografía completa (references_nhess.bib)

### Pendiente
- [ ] **Tabla S1:** Exportar inventario 52 GLOF como CSV para material suplementario
- [ ] **Revisión de coautores:** DMYY, VIQ, FTC deben revisar el manuscrito
- [ ] **Depuración final:** Verificar todos los números del manuscrito vs. modelos/datos
- [ ] **Submission IJACSA:** Crear cuenta en thesai.org, cargar PDF + figuras
- [ ] **Cover letter:** Carta al editor (máx. 1 página) destacando novedad del estudio
- [ ] **ORCID verificación:** Confirmar ORCIDs de los 4 autores

---

## 16. Información de Submission (NHESS)

| Campo | Detalle |
|-------|---------|
| Revista | International Journal of Advanced Computer Science and Applications (IJACSA) |
| ISSN | 2158-107X (online) |
| Editor | The Science and Information (SAI) Organization |
| Indexación | Scopus, Open Access |
| Cuartil | Q2 — Computer Science |
| Open Access | Full OA sin APC |
| Template | IEEEtran (ijacsa_paper/SAI_Paper_Format_Latex/) — ya implementado |
| Portal | https://thesai.org/Publications/IJACSA |
| Proceso | Double-blind peer review (mínimo 3 revisores) |
| Tiempo estimado | 4–8 semanas hasta decisión editorial |
| Límite páginas | 10 páginas máx. (cuerpo principal) |
| Manuscrito actual | ijacsa_paper/manuscript_ijacsa.tex (compilado, PDF disponible) |

---

## 17. Referencias Clave del Manuscrito

| Cita clave | Razón de importancia |
|-----------|---------------------|
| Hugonnet et al. (2021) | Retiro glaciar global — contexto |
| Shugar et al. (2020) | Aumento 51% área lagos glaciares 1990–2018 |
| Rounce et al. (2024) | Proyecciones lagos futuros CMIP6 |
| Emmer et al. (2022) | Inventario GLOFs Andes tropicales |
| Lützow et al. (2023) | GloFLOD — base global de GLOFs |
| Pronk et al. (2022) | Revisión metodológica — OOF + ensemble como estándar |
| Lundberg et al. (2020) | SHAP TreeExplainer — algoritmo usado |
| Chen & Breiman (2004) | Balanced Random Forest — modelo ganador |
| Clague & Evans (2000) | Teoría inestabilidad morrenas — interpretación física |
| Westoby et al. (2014) | Mecanismos GLOF — interpretación física |
| GlaMBIE (2025) | Pérdida de masa glaciar — datos más recientes |

---

*Documento generado: abril 2026 — Proyecto GLOF Andes — UNAP Puno*
*Última revisión: análisis completo del pipeline NB00→NB18 + manuscrito_nhess.tex*
