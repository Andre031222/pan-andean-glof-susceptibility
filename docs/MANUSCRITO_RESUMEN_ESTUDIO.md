# Resumen del Manuscrito para Estudio
# Machine Learning para Susceptibilidad GLOF en los Andes
---

## 1. Problema que resuelve

Los Glacial Lake Outburst Floods (GLOFs) son inundaciones catastróficas causadas por el
desborde o ruptura de lagos glaciares. Son uno de los peligros naturales más destructivos
en los Andes, amenazando millones de personas en Peru, Bolivia, Ecuador y Chile.

**El problema concreto:**
- No existia ningun modelo validado de susceptibilidad GLOF a escala continental andina
- Los estudios previos solo cubrian una cordillera o un pais
- Los datos son extremadamente escasos: 17 eventos confirmados en 12,703 lagos (ratio 1:746)
- Sin herramienta practica para decidir que lagos inspeccionar primero

---

## 2. Objetivo

Crear el **primer modelo de machine learning pan-andino** que prediga que lagos glaciares
son fuente potencial de GLOFs, con validacion geografica rigurosa entre paises.

---

## 3. Datos utilizados

### Imagenes satelitales
- **Sensor:** Sentinel-2 Level-2A (ESA Copernicus, open access)
- **Periodo:** 2017 a 2025 (8 anos de series temporales)
- **Resolucion espacial:** 10 metros
- **Indice de deteccion de agua:** MNDWI = (B3 - B11) / (B3 + B11)

### Area de estudio
- **11 cordilleras** en 4 paises: Peru (7), Bolivia (1), Ecuador (1), Chile (1)
- **Rango latitudinal:** 0.3 N a 35 S
- **Total lagos detectados:** 12,703 observaciones lago-ano

### Inventario de GLOFs historicos
- 71 eventos GLOF compilados (1932-2023)
- 17 eventos emparejados espacialmente con lagos detectados (buffer 5 km)
- 54 eventos no emparejados (coordenadas de impacto, no de la fuente)

### Features extraidas por lago (57 en total)
| Categoria | Ejemplos |
|---|---|
| Morfometricas | area, perimetro, elongacion, circularidad |
| Topograficas | elevacion, pendiente, aspecto, curvatura |
| Hidrologicas | distancia al glaciar, volumen estimado, profundidad |
| Temporales | tasa de cambio de area, anos de deteccion |
| Engineered | slope x log(area), area-to-depth ratio, ensemble depth mean |

---

## 4. Metodologia

### Pipeline completo
```
Sentinel-2 L2A
    → MNDWI > umbral → deteccion de lagos
    → extraccion de 57 features por lago-ano
    → emparejamiento con inventario GLOF historico
    → entrenamiento de modelos ML
    → validacion OOF + LOCO-CV
    → analisis SHAP
    → calibracion isotonica
    → mapa de susceptibilidad
```

### Seis clasificadores evaluados
1. Balanced Random Forest (BRF) — MEJOR
2. LightGBM
3. XGBoost
4. EasyEnsemble
5. Logistic Regression
6. Random Forest estandar

### Validacion — dos niveles

**Out-of-Fold (OOF) Cross-Validation:**
- Divide los datos en k folds
- Entrena en k-1 folds, predice en el fold restante
- Repite hasta predecir todos los datos
- Evita sobreajuste, mas honesto que train/test split simple

**Leave-One-Country-Out (LOCO-CV):**
- Entrena con 3 paises, predice en el 4to
- Repite dejando cada pais fuera una vez
- Prueba que el modelo funciona en paises no vistos durante entrenamiento
- El mas riguroso para validar transferibilidad geografica

---

## 5. Resultados del modelo

### Metricas principales (Balanced Random Forest)

| Metrica | Valor |
|---|---|
| ROC-AUC | **0.787** |
| Intervalo de confianza 95% | [0.683, 0.876] |
| Lift en threshold Youden | **8.7x** |
| Recall | **94%** (16 de 17 GLOFs recuperados) |
| Threshold optimo (Youden) | t = 0.356 |
| Lagos flagged en ese threshold | 41.3% del inventario |

### Resultados LOCO por pais

| Pais dejado fuera | AUC | Interpretacion |
|---|---|---|
| Ecuador | 0.965 | Muy buena transferencia |
| Bolivia | 0.790 | Buena transferencia |
| Peru | 0.739 | Transferencia aceptable |
| Chile | — | n+=0, no hay GLOFs emparejados |

### Eficiencia de priorizacion

| Top-k% inspeccionado | N lagos | GLOFs recuperados | Recall | Lift |
|---|---|---|---|---|
| 0.5% | 64 | 2 | 12% | 23.4x |
| 2.0% | 254 | 4 | 24% | **11.8x** |
| 5.0% | 635 | 6 | 35% | 7.1x |
| 10.0% | 1,270 | 7 | 41% | 4.1x |
| 20.0% | 2,541 | 8 | 47% | 2.4x |

**Conclusion practica:** Inspeccionando solo el 2% del inventario (254 lagos) se recupera
el 24% de los lagos peligrosos conocidos, con una precision 11.8 veces mayor que muestreo
aleatorio.

---

## 6. Robustez estadistica — 6 procedimientos

| Procedimiento | Resultado | Que confirma |
|---|---|---|
| Permutation test | p < 0.0002 | El modelo no aprende ruido aleatorio |
| DeLong test | z = 0.47, p = 0.640 | AUC no difiere significativamente de referencia |
| Mann-Whitney U | dist_glaciar p=0.037, volumen p<0.001 | Features discriminan GLOFs de no-GLOFs |
| Moran's I | p = 0.805 | Sin autocorrelacion espacial en residuos |
| KS test | D = 0.541, p = 6.5e-5 | Distribuciones de score muy diferentes entre clases |
| Jackknife (17 estimaciones) | AUC 0.774-0.818, max drop 0.013 | Ninguna etiqueta positiva domina el resultado |

---

## 7. Interpretacion SHAP — Que causa los GLOFs

SHAP (SHapley Additive exPlanations) explica por que el modelo asigna alta susceptibilidad
a cada lago.

### Top 10 features por importancia SHAP

| Rank | Feature | Interpretacion fisica |
|---|---|---|
| 1 | Distancia al glaciar | Lagos mas proximos al glaciar tienen mayor riesgo |
| 2 | Slope x log(area) | Pendiente combinada con tamaño |
| 3 | Volumen del lago | Lagos mas grandes almacenan mas energia potencial |
| 4 | Ratio area/profundidad | Forma del lago relacionada con tipo de presa |
| 5 | Ensemble depth mean | Profundidad media estimada |
| 6 | Log-area | Tamaño del lago |
| 7 | Depth uncertainty | Incertidumbre en estimacion de profundidad |
| 8 | Empirical depth (Yao) | Profundidad segun formula empirica |
| 9 | Depth std | Variabilidad en estimacion de profundidad |
| 10 | Elongacion | Forma alargada del lago |

### Hallazgo clave — Relacion no lineal con distancia al glaciar

El riesgo NO aumenta linealmente con la proximidad al glaciar. Existe un **pico de riesgo
entre 2 y 15 km** del frente glaciar. Esto es consistente con la teoria de inestabilidad
de morenas (Westoby et al. 2014) y el framework de alcance efectivo (Schneider et al. 2014).

- Lagos muy proximos (<2 km): menos riesgo (proglaciales, bien drenados)
- Lagos a 2-15 km: maximo riesgo (morrenas inestables, presion hidrica alta)
- Lagos muy distales (>15 km): menor riesgo (sin conexion hidrologica directa)

### Diferencias entre lagos GLOF-source y no-GLOF

| Variable | Lagos GLOF-source | Lagos no-GLOF | p-valor |
|---|---|---|---|
| Distancia mediana al glaciar | **512 m** | 2,403 m | 0.037 |
| Volumen | significativamente mayor | — | < 0.001 |

---

## 8. Calibracion del modelo

El modelo BRF produce scores sobreestimados (por el resampling bajo imbalance 1:746).
Se aplica **calibracion isotonica** post-hoc para convertir scores en probabilidades
operacionales de inundacion.

- Brier Skill Score antes de calibracion: -139.8 (peor que climatologia)
- Brier Skill Score despues de calibracion: +0.008 (supera climatologia)
- La calibracion preserva la discriminacion (AUC no cambia)
- Los scores calibrados son los recomendados para comunicacion de riesgo operacional

---

## 9. Validacion de estrategia de etiquetado (Seccion V.F)

Se probaron dos estrategias de augmentacion para validar que la estrategia original
(solo el ano del evento = positivo) es correcta:

| Estrategia | n+ entrenamiento | AUC resultante | Conclusion |
|---|---|---|---|
| Original (1 ano por evento) | 17 | 0.787 | Referencia |
| Augmentacion temporal (todos los anos) | 538 | 0.531 | EMPEORA — introduce ruido |
| Gaussian jitter (perturbacion features) | ~170/fold | 0.792 | Mejora marginal |

La augmentacion temporal empeora porque etiqueta como positivo anos donde el lago aun
no habia generado el GLOF — introduce ruido de etiquetado. Confirma que la estrategia
original es la correcta.

---

## 10. Conclusiones

1. **Es posible predecir GLOFs a escala continental** con ML incluso con solo 17 eventos
   confirmados, usando features fisicamente interpretables y validacion rigurosa.

2. **Balanced Random Forest** es el mejor clasificador bajo desequilibrio extremo (1:746),
   superando a alternativas en el metrico Recall-weighted Lift.

3. **El modelo transfiere entre paises** — validado por LOCO-CV en todos los paises con
   etiquetas GLOF (Lift >= 2.8x en cada pais held-out).

4. **La proximidad al glaciar es el driver fisico dominante**, con un pico de riesgo no
   lineal a 2-15 km consistente con teoria glaciologica establecida.

5. **Herramienta practica de priorizacion:** el 2% del inventario captura el 24% de los
   lagos peligrosos — directamente aplicable a gestion de riesgo de desastres en los Andes.

---

## 11. Limitaciones

- n+=17 es muy pequeno — resultado de las estrictas condiciones de emparejamiento
- 54 GLOFs historicos no pudieron emparejarse (coordenadas de impacto vs. fuente)
- Chile tiene n+=0 en el modelo (no hay GLOFs emparejados en Andes Centrales)
- El modelo fue entrenado con datos 2017-2025 — puede no generalizar a cambios futuros
- No incluye datos SAR (Sentinel-1) por limitaciones de acceso en GEE

---

## 12. Datos y codigo

Repositorio GitHub (MIT licence):
github.com/Andre031222/GLOFs-Pan-Andina-Mediante-Machine-Learning

Datos satelitales: ESA Copernicus (open access), NASA EARTHDATA (open access)

---

## Glosario rapido

| Termino | Significado |
|---|---|
| GLOF | Glacial Lake Outburst Flood — inundacion por desborde de lago glaciar |
| BRF | Balanced Random Forest — variante de Random Forest para datos desbalanceados |
| SHAP | SHapley Additive exPlanations — metodo para explicar predicciones de ML |
| OOF | Out-of-Fold — predicciones fuera del fold de entrenamiento |
| LOCO | Leave-One-Country-Out — validacion cruzada dejando un pais fuera |
| AUC | Area Under the ROC Curve — medida de discriminacion del modelo (0 a 1) |
| Lift | Precision del modelo / Precision aleatoria — cuanto mejor que el azar |
| Recall | Proporcion de GLOFs reales que el modelo detecta |
| Youden threshold | Umbral optimo que maximiza Recall + Especificidad |
| MNDWI | Modified Normalized Difference Water Index — indice para detectar agua |
| Sentinel-2 | Satelite de la ESA con imagenes opticas a 10 m de resolucion |
| Calibracion isotonica | Tecnica para convertir scores de ML en probabilidades reales |
| Jackknife | Tecnica de validacion: quita un dato positivo a la vez y re-evalua |
| Moran's I | Estadistico de autocorrelacion espacial |
| KS test | Kolmogorov-Smirnov — compara distribuciones de dos grupos |
