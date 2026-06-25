"""
config_expanded_study_areas.py
-------------------------------
Central configuration for all glacial lake study areas in the
pan-Andean GLOF susceptibility project.

Structure per area
------------------
  bbox            : [lon_min, lat_min, lon_max, lat_max]  WGS-84
  epsg            : UTM zone EPSG code (e.g. 32718 = UTM-18S)
  elev_min_m      : lower elevation bound for lake detection
  elev_max_m      : upper elevation bound for lake detection
  dry_season_months : list of months for least-cloudy Sentinel-2 scenes
  max_cloud_cover : maximum acceptable cloud fraction (%)
  lakes_estimated : rough expected lake count
  glof_events_documented : known historical GLOF events in this area
  status          : COMPLETED | DOWNLOADING | PENDING
  priority        : CRITICAL | HIGH | MEDIUM | LOW
  notes           : free-text context

Sentinel-2 download calls:
    from config_expanded_study_areas import EXPANDED_STUDY_AREAS
    area = EXPANDED_STUDY_AREAS['cordillera_blanca']
    results = download_study_area_data(
        area_name='cordillera_blanca',
        months=area['dry_season_months'],
        ...
    )
"""

EXPANDED_STUDY_AREAS = {

    # ------------------------------------------------------------------
    # ORIGINAL 10 AREAS  (2017-2025, all COMPLETED)
    # ------------------------------------------------------------------

    'cordillera_blanca': {
        'description': 'Cordillera Blanca, Ancash, Peru — largest tropical glacier concentration',
        'bbox': [-77.8, -9.8, -77.1, -8.8],
        'epsg': 32718,
        'elev_min_m': 3000,
        'elev_max_m': 6500,
        'dry_season_months': [6, 7, 8],
        'max_cloud_cover': 15,
        'lakes_estimated': 3083,
        'glof_events_documented': 10,
        'status': 'COMPLETED',
        'priority': 'CRITICAL',
        'notes': '9 of 16 matched GLOF labels; highest GLOF density in dataset.',
    },

    'cordillera_vilcanota': {
        'description': 'Cordillera Vilcanota, Cusco, Peru',
        'bbox': [-71.2, -14.0, -70.4, -13.2],
        'epsg': 32719,
        'elev_min_m': 3000,
        'elev_max_m': 6100,
        'dry_season_months': [6, 7, 8],
        'max_cloud_cover': 15,
        'lakes_estimated': 2770,
        'glof_events_documented': 0,
        'status': 'COMPLETED',
        'priority': 'HIGH',
        'notes': '5 inventory events but 0 matched (coordinates correspond to downstream zones).',
    },

    'cordillera_central': {
        'description': 'Cordillera Central, Junin-Lima, Peru',
        'bbox': [-76.2, -11.5, -75.8, -11.0],
        'epsg': 32718,
        'elev_min_m': 3000,
        'elev_max_m': 5100,
        'dry_season_months': [6, 7, 8],
        'max_cloud_cover': 15,
        'lakes_estimated': 2239,
        'glof_events_documented': 0,
        'status': 'COMPLETED',
        'priority': 'HIGH',
        'notes': '4 inventory events, 0 matched. Lakes well-detected; event coords unreliable.',
    },

    'chile_andes_centrales': {
        'description': 'Andes Centrales, Santiago Metropolitan Region, Chile',
        'bbox': [-70.0, -33.5, -69.5, -33.0],
        'epsg': 32719,
        'elev_min_m': 2000,
        'elev_max_m': 6600,
        'dry_season_months': [12, 1, 2],
        'max_cloud_cover': 20,
        'lakes_estimated': 1589,
        'glof_events_documented': 0,
        'status': 'COMPLETED',
        'priority': 'MEDIUM',
        'notes': 'Null-class domain (0 matched GLOF labels). FPR@0.304=29.6%. '
                 'Mediterranean seasonality differs from tropical Andes.',
    },

    'cordillera_raura': {
        'description': 'Cordillera Raura, Lima-Huanuco, Peru',
        'bbox': [-76.85, -10.5, -76.65, -10.25],
        'epsg': 32718,
        'elev_min_m': 3700,
        'elev_max_m': 5700,
        'dry_season_months': [6, 7, 8],
        'max_cloud_cover': 15,
        'lakes_estimated': 1361,
        'glof_events_documented': 2,
        'status': 'COMPLETED',
        'priority': 'HIGH',
        'notes': '2 matched positives; highest high-risk fraction (55.7%).',
    },

    'cordillera_urubamba': {
        'description': 'Cordillera Urubamba, Cusco, Peru',
        'bbox': [-72.2, -13.3, -71.8, -13.0],
        'epsg': 32719,
        'elev_min_m': 2300,
        'elev_max_m': 5800,
        'dry_season_months': [6, 7, 8],
        'max_cloud_cover': 15,
        'lakes_estimated': 430,
        'glof_events_documented': 0,
        'status': 'COMPLETED',
        'priority': 'MEDIUM',
        'notes': '3 inventory events, 0 matched (17-40 km away from detected lakes).',
    },

    'cordillera_huanzo': {
        'description': 'Cordillera Huanzo, Apurimac-Arequipa, Peru',
        'bbox': [-73.1, -15.3, -72.7, -14.9],
        'epsg': 32719,
        'elev_min_m': 1800,
        'elev_max_m': 5300,
        'dry_season_months': [6, 7, 8],
        'max_cloud_cover': 15,
        'lakes_estimated': 385,
        'glof_events_documented': 0,
        'status': 'COMPLETED',
        'priority': 'MEDIUM',
        'notes': '4 inventory events, 0 matched (31-57 km away). Remote area.',
    },

    'cordillera_huayhuash': {
        'description': 'Cordillera Huayhuash, Ancash-Lima-Huanuco, Peru',
        'bbox': [-76.95, -10.35, -76.75, -10.05],
        'epsg': 32718,
        'elev_min_m': 3500,
        'elev_max_m': 6500,
        'dry_season_months': [6, 7, 8],
        'max_cloud_cover': 15,
        'lakes_estimated': 278,
        'glof_events_documented': 4,
        'status': 'COMPLETED',
        'priority': 'CRITICAL',
        'notes': '3 matched positives from v1 inventory; adding 2023 Lago Rasac event (Emmer2025).',
    },

    'ecuador_antisana': {
        'description': 'Antisana Ecological Reserve, Napo-Pichincha, Ecuador',
        'bbox': [-78.3, -0.6, -78.0, -0.3],
        'epsg': 32717,
        'elev_min_m': 2100,
        'elev_max_m': 5800,
        'dry_season_months': [12, 1, 2],
        'max_cloud_cover': 40,
        'lakes_estimated': 273,
        'glof_events_documented': 1,
        'status': 'COMPLETED',
        'priority': 'HIGH',
        'notes': 'Highest LOCO AUC (0.993); 1 matched positive. Higher cloud cover accepted.',
    },

    'bolivia_cordillera_real': {
        'description': 'Cordillera Real, La Paz, Bolivia',
        'bbox': [-68.35, -16.65, -67.7, -15.75],
        'epsg': 32719,
        'elev_min_m': 3500,
        'elev_max_m': 6400,
        'dry_season_months': [6, 7, 8],
        'max_cloud_cover': 15,
        'lakes_estimated': 180,
        'glof_events_documented': 1,
        'status': 'COMPLETED',
        'priority': 'HIGH',
        'notes': '7 inventory events, 1 matched (Tuni-Condoriri). Gap bridged by new Apolobamba area.',
    },

    # ------------------------------------------------------------------
    # NEW AREA 1 — Patagonia Norte / NPI (Chile, 44.5-48.5°S)
    # BBOX CORRECTED 2026-05-06: narrowed from [-74.0,-48.5,-71.0,-39.5]
    # to [-74.5,-48.5,-71.0,-44.5] to target the actual NPI glacial zone.
    # Previous bbox was too large; download captured lakes only at ~-40°S
    # (northern edge) — far from the GLOF events at -46 to -47°S.
    # NPI GLOF events: Huemules (-46.417,-72.650), Exploradores (-46.550,-73.220),
    # San Quintin (-46.850,-74.000) — all within the corrected bbox.
    # ------------------------------------------------------------------
    'patagonia_norte': {
        'description': 'NPI Cachet2/Colonia GLOF cluster, Aysen, Chile (~47.0-47.9S)',
        'bbox': [-73.5, -47.9, -72.5, -47.0],
        'epsg': 32718,
        'elev_min_m': 0,
        'elev_max_m': 3500,
        'dry_season_months': [12, 1, 2],
        'max_cloud_cover': 50,
        'lakes_estimated': 400,
        'glof_events_documented': 3,
        'status': 'PENDING',
        'priority': 'HIGH',
        'notes': (
            'Northern Patagonian Icefield (NPI). Moraine-dammed and ice-dammed lakes. '
            'GLOFs documented by IribarrenAnacona et al. (2014, 2015). '
            'Glaciers: San Rafael, San Quintin, Exploradores, Steffen, Hudson. '
            'BBOX CORRECTED 2026-05-06: narrowed to -44.5 to -48.5 S (NPI glacial zone). '
            'Previous download got data only at lat~-40S (outside glacial area). '
            'max_cloud_cover raised to 50 (was 30) for maritime Patagonia. '
            'Run NB20 again with updated config, then NB11-NB12 for this area only.'
        ),
    },

    # ------------------------------------------------------------------
    # NEW AREA 2 — Patagonia Sur / SPI (Chile/Argentina, 46.5-54.5°S)
    # SEASON CORRECTED 2026-05-06: months changed from [12,1,2,3,4,5] (DJF)
    # to [5,6,7,8,9,10] (austral autumn/winter = JJA).
    # RATIONALE: Lago Cachet 2 is an ice-dammed lake that FILLS during austral
    # winter (May-November) and DRAINS in November-December. A DJF composite
    # coincides with post-drainage empty lake state → 0 lakes detected.
    # JJA composite (June-August) captures the lake during peak filling.
    # ------------------------------------------------------------------
    'patagonia_sur': {
        'description': 'S2 tile T18GXN — Cachet 2 sector, lat -47.05 to -47.5°S, lon -72.5 to -73.4°W',
        'bbox': [-73.4, -47.5, -72.5, -47.05],
        'epsg': 32718,
        'elev_min_m': 0,
        'elev_max_m': 3200,
        'dry_season_months': [5, 6, 7, 8, 9, 10],
        'max_cloud_cover': 70,
        'lakes_estimated': 40,
        'glof_events_documented': 5,
        'status': 'PENDING',
        'priority': 'CRITICAL',
        'notes': (
            'BBOX CORRECTED 2026-06-21 (v3): verified via Planetary Computer STAC.\n'
            'Correct S2 tile for Cachet 2 is T18GXN (lon -73.686 to -72.192, lat -47.933 to -46.920). '
            'T18GXP (adjacent tile, cloud=1.7%) has southern boundary -47.034 and does NOT reach '
            'Cachet 2 at -47.183. Bbox must be entirely south of -47.034 to avoid T18GXP being '
            'selected. New bbox [-73.4,-47.5,-72.5,-47.05] ensures only T18GXN tiles are found.\n'
            'SEASON: months=[5,6,7,8,9,10] (JJA austral winter). '
            'Cachet 2 fills May-Nov, drains Nov-Dec.\n'
            'TARGET: Lago Cachet 2 (-47.183,-73.017) — 4 S2-era GLOFs (2017-2020) → expected +4 n+.\n'
            'Pipeline: delete data/raw/sentinel2/patagonia_sur/, '
            'data/raw/dem/patagonia_sur/*.tif, data/interim/terrain/patagonia_sur/*.tif, '
            'data/interim/dem/patagonia_sur_dem_utm.tif — ALL DONE 2026-06-21.\n'
            'RUN: python scripts/download_patagonia_sur_jja.py  →  NB11  →  '
            'python scripts/run_lake_detection.py patagonia_sur  →  NB13  →  NB14\n'
            'Pipeline: DELETE data/raw/sentinel2/patagonia_sur/ and '
            'data/processed/lakes/patagonia_sur_*.gpkg, then run NB21 → NB12 → NB13 → NB14.'
        ),
    },

    # ------------------------------------------------------------------
    # NEW AREA 3b — Bolivia Norte / Cordillera Real Norte (Bolivia, 14.5-16°S)
    # ADDED 2026-05-06: covers northern Cordillera Real sectors not included
    # in bolivia_cordillera_real (bbox [-68.35,-16.65,-67.7,-15.75]).
    # Events Khara Kkota, Glaciar Khara, Keara (2013), Illampu, Zongo_1,
    # Khara_Khota are 8-34 km from detected lakes in the current bbox —
    # they likely lie in the Zongo Valley and northern La Paz sectors.
    # This new area extends coverage to [-68.4,-15.75,-67.5,-14.3] (north).
    # ------------------------------------------------------------------
    'bolivia_norte': {
        'description': 'Northern Cordillera Real, La Paz, Bolivia — Zongo Valley and northern sectors',
        'bbox': [-68.4, -15.75, -67.5, -14.3],
        'epsg': 32719,
        'elev_min_m': 3500,
        'elev_max_m': 6500,
        'dry_season_months': [6, 7, 8],
        'max_cloud_cover': 15,
        'lakes_estimated': 120,
        'glof_events_documented': 5,
        'status': 'PENDING',
        'priority': 'HIGH',
        'notes': (
            'Northern extension of Bolivia Cordillera Real. Covers Zongo Valley '
            '(Glaciar Zongo GLOF 2009), Laguna Khara Khota, Glaciar Illampu, '
            'and Keara sectors. Events are 8-34km from lakes in the southern bbox. '
            'Khara_Khota (2016) nearest lake = 8,488m → should match with 10km buffer '
            'if lake is detected in this new area or corrected coordinate is used. '
            'Zongo_1 (2009) nearest lake = 13,360m → possible with 14km buffer. '
            'Run new download notebook for this area, then NB11-NB12, then NB14 '
            'with graduated buffer (pre-2017: 10km, post-2017: 5km).'
        ),
    },

    # ------------------------------------------------------------------
    # NEW AREA 3 — Apolobamba (Peru/Bolivia border, 14-15°S)
    # ------------------------------------------------------------------
    'apolobamba': {
        'description': 'Cordillera Apolobamba, Puno (Peru) / La Paz (Bolivia), ~13.8-15.2°S',
        'bbox': [-69.6, -15.3, -68.3, -13.8],
        'epsg': 32719,
        'elev_min_m': 3800,
        'elev_max_m': 6100,
        'dry_season_months': [6, 7, 8],
        'max_cloud_cover': 15,
        'lakes_estimated': 350,
        'glof_events_documented': 3,
        'status': 'PENDING',
        'priority': 'HIGH',
        'notes': (
            'Fills geographic gap between Bolivia Cordillera Real and Peru Vilcanota. '
            'Transboundary area Peru-Bolivia. Several proglacial lakes documented '
            'in the INAIGEM inventory and regional literature. '
            'Similar climate to Cordillera Real: strong dry season Jun-Aug. '
            'Run NB22 first, then follow standard pipeline.'
        ),
    },

    # ------------------------------------------------------------------
    # NEW AREA 4 — Carabaya (Peru, east of Vilcanota, 13.5-15°S)
    # ------------------------------------------------------------------
    'carabaya': {
        'description': 'Cordillera Carabaya, Puno, Peru — eastern extension of Vilcanota system',
        'bbox': [-70.5, -15.0, -69.5, -13.5],
        'epsg': 32719,
        'elev_min_m': 3500,
        'elev_max_m': 5800,
        'dry_season_months': [6, 7, 8],
        'max_cloud_cover': 15,
        'lakes_estimated': 500,
        'glof_events_documented': 2,
        'status': 'PENDING',
        'priority': 'MEDIUM',
        'notes': (
            'Eastern cordillera of Puno region; glaciers drain toward Amazon basin. '
            'Less studied than Vilcanota but shares similar glacier characteristics. '
            'Partial coverage in existing INAIGEM records. '
            'Connects Vilcanota with Apolobamba spatially.'
        ),
    },
}

# ------------------------------------------------------------------
# Convenience helpers
# ------------------------------------------------------------------

def get_area(name: str) -> dict:
    """Return area config dict; raises KeyError if name not found."""
    if name not in EXPANDED_STUDY_AREAS:
        available = sorted(EXPANDED_STUDY_AREAS.keys())
        raise KeyError(
            f"Area '{name}' not found. Available: {available}"
        )
    return EXPANDED_STUDY_AREAS[name]


def list_areas(status: str = None) -> list:
    """Return list of area names, optionally filtered by status."""
    areas = EXPANDED_STUDY_AREAS.items()
    if status:
        areas = [(k, v) for k, v in areas if v['status'] == status]
    return [k for k, _ in areas]


def pending_areas() -> list:
    return list_areas('PENDING')


def completed_areas() -> list:
    return list_areas('COMPLETED')


if __name__ == '__main__':
    print(f"Total study areas : {len(EXPANDED_STUDY_AREAS)}")
    print(f"Completed         : {len(completed_areas())}")
    print(f"Pending download  : {len(pending_areas())}")
    print()
    for name, cfg in EXPANDED_STUDY_AREAS.items():
        lakes = cfg['lakes_estimated']
        glofs = cfg['glof_events_documented']
        status = cfg['status']
        print(f"  {status:12s}  {name:30s}  ~{lakes:5d} lakes  {glofs:2d} GLOFs documented")
