# Plan "siguiente nivel" — extender el dataset para subir n⁺ honestamente

**Objetivo:** elevar el conjunto de positivos de n⁺=17 hacia ~30-40 con datos reales
(no inflando), para fortalecer AUC, priorización y la defensa ante revisores.
La fuga ya está corregida (CV agrupado por lago); este plan es la mejora **legítima**.

## Por qué este es el único camino real

La auditoría de los 71 eventos GLOF mostró que dentro del inventario Sentinel-2
(2017–2025) solo 3 eventos no-matcheados son recuperables. La razón: **36 eventos
son pre-2017** y su lago fuente no existe en el inventario actual. Para
emparejarlos hay que ver el lago **tal como existía cuando ocurrió el evento** →
se necesita imaginería histórica (Landsat 1985–2016). Ese es el gran palanca.

---

## Fases y división de trabajo

| Fase | Qué | Quién ejecuta | Entregable |
|---|---|---|---|
| **0. Decisión** | ¿enviar ahora la versión honesta o esperar a la extensión? | Andre + asesor | luz verde |
| **1. Inventario Landsat 1985–2016** | descargar + detectar lagos (MNDWI) por cordillera | **Andre** (máquina con internet/disco) corriendo el script que dejo | GeoPackages anuales por área |
| **2. Merge** | unir inventario Landsat + Sentinel-2 (1985–2025) | script (corre local) | inventario extendido |
| **3. Catálogos GLOF ampliados** | sumar eventos documentados con coordenadas | Andre descarga; yo armo el ingestor | catálogo ampliado |
| **4. Re-matching** | re-emparejar TODOS los eventos contra el inventario extendido | script | nuevo n⁺ (esperado ~30-40) |
| **5. Re-extracción de features + re-entrenamiento** | pipeline completo con CV agrupado | `run_full_analysis.py` (ya listo) | nuevos resultados |
| **6. Re-sync manuscrito + figuras** | actualizar todo | yo | v2 del paper |

**Tiempo realista:** 2–4 semanas (la Fase 1 es la más pesada: descarga + cómputo).

---

## Fase 1 — Inventario Landsat (detalle técnico)

- **Fuente:** Microsoft Planetary Computer, colección `landsat-c2-l2`
  (Landsat 5 TM 1984–2012, Landsat 7 ETM+ 1999–, Landsat 8 OLI 2013–).
- **Bandas (nombres comunes PC, válidos para todos los sensores):**
  `green`, `swir16` → MNDWI = (green − swir16) / (green + swir16).
- **Composición:** mediana de temporada seca (jun–ago en Hemisferio Sur;
  dic–mar en Ecuador), nubes < 20%, recorte al bbox de cada área.
- **Detección:** MNDWI > 0.0 → vectorizar → filtrar área > 9.000 m²
  (3 px Landsat de 30 m; nota: Landsat no resuelve lagos pequeños como Sentinel-2)
  → dentro de 5 km de glaciar (RGI 7.0) → calcular morfometría (mismo esquema).
- **Salida:** GeoPackage por (área, año) con las mismas columnas que el inventario
  Sentinel-2, para que el merge sea directo.
- **Limitación honesta:** Landsat 30 m → mayor omisión de lagos pequeños y bordes
  más gruesos. Se declara en el manuscrito como compromiso por la cobertura temporal.
- **Script:** `scripts/download_landsat_inventory.py` (lo dejo listo; corre local).

## Fase 3 — Catálogos GLOF a minar (verificar coordenadas, NO inventar)

- Veh et al. — base de datos global de GLOF (con fechas/coordenadas).
- Carrivick & Tweed (2016) — compilación global de impactos GLOF.
- Perú: inventarios ANA / INAIGEM de lagunas y eventos.
- Bolivia, Ecuador, Chile: estudios regionales (Cordillera Real, Antisana, Andes
  Centrales).
- Cada evento nuevo entra solo si tiene **coordenada de lago fuente verificable**.

## Riesgos / límites

- Con n⁺ aún <40, el AUC honesto difícilmente pasa de ~0.75–0.80.
- Landsat sube cobertura temporal pero baja resolución espacial (trade-off).
- Es esfuerzo de semanas, mayormente de descarga/validación en tu máquina.

## Recomendación

Camino A (enviar la versión honesta actual) sigue siendo válido y rápido.
Este plan (Camino B) es la mejora de fondo para una **v2 / segundo paper**.
Decidir en Fase 0 con el asesor.
