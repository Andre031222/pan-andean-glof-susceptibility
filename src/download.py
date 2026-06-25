# =============================================================================
# GLOF Andes Project - Utilidades de Descarga de Datos
# =============================================================================
"""
Funciones para descargar datos satelitales desde Microsoft Planetary Computer.

Corrección clave (Feb 2026): Las bandas de Sentinel-2 están almacenadas en
proyección UTM. Todos los bounding boxes (WGS-84) son reproyectados al CRS
nativo del raster antes de calcular las ventanas de píxeles, evitando el
crash '0x0 dataset'.

Uso típico (desde notebooks 01-10):
    from src.download import download_study_area_data
    results = download_study_area_data('cordillera_huayhuash')
"""

import os
import time
import calendar
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import warnings

import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds as warp_bounds

# ---------------------------------------------------------------------------
# Configuración de logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.WARNING,  # Solo warnings y errores en producción
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Resiliencia de red GDAL (configurar antes de cualquier apertura de raster)
# ---------------------------------------------------------------------------
os.environ.setdefault('GDAL_HTTP_MAX_RETRY', '5')
os.environ.setdefault('GDAL_HTTP_RETRY_DELAY', '2')
os.environ.setdefault('GDAL_HTTP_TIMEOUT', '60')
os.environ.setdefault('GDAL_DISABLE_READDIR_ON_OPEN', 'EMPTY_DIR')
os.environ.setdefault('CPL_VSIL_CURL_ALLOWED_EXTENSIONS', '.tif,.tiff,.vrt')


# ---------------------------------------------------------------------------
# Conexión a Planetary Computer
# ---------------------------------------------------------------------------

def get_planetary_computer_client():
    """
    Obtiene un cliente STAC autenticado de Microsoft Planetary Computer.

    Returns
    -------
    pystac_client.Client
        Cliente STAC listo para búsquedas.

    Raises
    ------
    ImportError
        Si los paquetes planetary-computer o pystac-client no están instalados.
    ConnectionError
        Si no se puede conectar al servidor STAC.
    """
    try:
        import planetary_computer
        import pystac_client
    except ImportError as e:
        raise ImportError(
            f"Paquete requerido no encontrado: {e}. "
            "Instalar con: pip install planetary-computer pystac-client"
        ) from e

    try:
        client = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=planetary_computer.sign_inplace,
        )
        return client
    except Exception as e:
        raise ConnectionError(
            f"No se pudo conectar a Planetary Computer: {e}"
        ) from e


# ---------------------------------------------------------------------------
# Descarga de una banda individual (COG windowed read)
# ---------------------------------------------------------------------------

def download_band_windowed(
    item,
    band: str,
    bbox_wgs84: Tuple[float, float, float, float],
    output_path: Path,
    max_retries: int = 3,
) -> Optional[Path]:
    """
    Descarga una banda de Sentinel-2 (o elevación DEM) recortada a *bbox_wgs84*
    mediante lectura por ventana de un COG (Cloud-Optimized GeoTIFF).

    El bbox se reproyecta de EPSG:4326 al CRS nativo del raster antes de
    calcular la ventana de píxeles. Esto evita el error '0x0 dataset' que
    ocurre cuando se pasan coordenadas lon/lat a un raster en proyección UTM.

    Parameters
    ----------
    item : pystac.Item
        Item STAC que contiene el asset a descargar.
    band : str
        Nombre del asset (e.g., 'B02', 'B08', 'elevation').
    bbox_wgs84 : tuple
        Bounding box en WGS-84: (west, south, east, north).
    output_path : Path
        Ruta de salida para el archivo .tif.
    max_retries : int
        Número máximo de reintentos ante errores de red.

    Returns
    -------
    Path or None
        Ruta al archivo descargado, o None si falló.
    """
    import planetary_computer

    if band not in item.assets:
        print(f"  [WARN] {band}: asset no encontrado en la escena")
        return None

    url = planetary_computer.sign(item.assets[band].href)
    west, south, east, north = bbox_wgs84

    for attempt in range(1, max_retries + 1):
        try:
            with rasterio.open(url) as src:
                # CORRECCIÓN CLAVE: reproyectar bbox WGS-84 al CRS nativo del raster
                # Sentinel-2 usa UTM (metros), no WGS-84 (grados)
                l, b, r, t = warp_bounds("EPSG:4326", src.crs, west, south, east, north)
                win = from_bounds(l, b, r, t, transform=src.transform)

                # Validar que la ventana tenga píxeles reales (evitar 0xN o Nx0)
                # Usamos un margen de seguridad de 0.5 píxeles antes de redondear
                if win.width < 0.5 or win.height < 0.5:
                    print(f"\n  [SKIP] {band}: Traslape insignificante con el tile (ventana: {win.width:.1f}x{win.height:.1f} px)")
                    return None

                data = src.read(1, window=win)
                
                # Segunda validación: el array de datos no debe estar vacío
                if data.size == 0 or data.shape[0] == 0 or data.shape[1] == 0:
                    print(f"\n  [SKIP] {band}: Datos vacíos leídos del tile")
                    return None

                profile = src.profile | {
                    'compress':  'lzw',
                    'width':     data.shape[1],
                    'height':    data.shape[0],
                    'transform': rasterio.windows.transform(win, src.transform),
                    'count':     1,
                }

            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(data, 1)

            return output_path

        except Exception as exc:
            # Limpiar archivo corrupto si existe
            if output_path.exists():
                output_path.unlink()

            if attempt < max_retries:
                wait = attempt * 3
                print(
                    f"\n    [Retry {attempt}/{max_retries}] "
                    f"{type(exc).__name__}. Esperando {wait}s...",
                    end=" ", flush=True,
                )
                time.sleep(wait)
            else:
                print(f"\n  [ERROR] {band} falló tras {max_retries} intentos: {exc}")
                print(f"  [SKIP] {band}")
                return None

    return None  # Nunca se llega aquí, pero satisface el type checker


# ---------------------------------------------------------------------------
# Pipeline completo de descarga de un área de estudio
# ---------------------------------------------------------------------------

def download_study_area_data(
    area_name: str,
    years: Optional[List[int]] = None,
    months: List[int] = None,
    sentinel_bands: List[str] = None,
    max_cloud_cover: float = 15.0,
    visualize: bool = True,
    save_metadata: bool = True,
) -> Dict:
    """
    Pipeline completo de descarga para un área de estudio.

    Descarga DEM (NASADEM 30m) y Sentinel-2 L2A multitemporal recortados
    al bounding box del área. Llamado por los notebooks de descarga 01-10.

    Parameters
    ----------
    area_name : str
        Clave en EXPANDED_STUDY_AREAS (e.g., 'cordillera_huayhuash').
    years : list[int], optional
        Años a descargar. None = todos los disponibles (2017-2025).
    months : list[int], optional
        Meses de temporada seca. Default: [6, 7, 8] (junio-agosto).
    sentinel_bands : list[str], optional
        Bandas a descargar. Default: 6 bandas para NDWI/MNDWI/NDVI/RGB.
    max_cloud_cover : float
        Cobertura máxima de nubes en %. Default: 15%.
    visualize : bool
        Crear visualización rápida del DEM descargado.
    save_metadata : bool
        Guardar archivo JSON con metadatos de descarga.

    Returns
    -------
    dict
        Diccionario con resultados: rutas de archivos, metadatos, éxito/fallo.

    Notes
    -----
    Sentinel-2 L2A en Planetary Computer está disponible de forma consistente
    desde 2017. El año 2026 se excluye porque la temporada seca (jun-ago)
    aún no ha ocurrido al momento de escribir este código.

    Ejemplo
    -------
    >>> results = download_study_area_data('cordillera_huayhuash')
    >>> print(f"Éxito: {results['success']}")
    >>> print(f"Años descargados: {results['sentinel_years']}")
    """
    import geopandas as gpd
    from shapely.geometry import box
    from rasterio.merge import merge
    import matplotlib.pyplot as plt
    import numpy as np

    from config_expanded_study_areas import EXPANDED_STUDY_AREAS

    # -- Valores por defecto ------------------------------------------------
    if months is None:
        months = [6, 7, 8]  # Temporada seca (invierno austral)

    if sentinel_bands is None:
        sentinel_bands = ['B02', 'B03', 'B04', 'B08', 'B11', 'B12']

    if years is None:
        years = list(range(2017, 2026))  # 2017-2025 inclusive (9 años)
        print(f"Usando TODOS los años disponibles: 2017-2025 ({len(years)} años)")

    # -- Validar configuración ----------------------------------------------
    if area_name not in EXPANDED_STUDY_AREAS:
        available = list(EXPANDED_STUDY_AREAS.keys())
        raise ValueError(
            f"Área '{area_name}' no encontrada en la configuración.\n"
            f"Áreas disponibles: {available}"
        )

    config     = EXPANDED_STUDY_AREAS[area_name]
    bbox_wgs84 = config['bbox']
    epsg_code  = config['epsg']

    # -- Cabecera -----------------------------------------------------------
    print("=" * 80)
    print(f"  GLOF Project >> {config['description'].upper()}")
    print(f"  Bbox: {bbox_wgs84}  |  EPSG:{epsg_code}")
    print(f"  ~{config['lakes_estimated']} lagunas  |  {config['glof_events_documented']} GLOFs documentados")
    print(f"  Serie temporal: {min(years)}–{max(years)} ({len(years)} años)  |  Bandas: {' '.join(sentinel_bands)}")
    print("=" * 80)

    # -- Directorios --------------------------------------------------------
    project_root = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
    data_dir     = project_root / 'data' / 'raw'
    dem_dir      = data_dir / 'dem' / area_name
    sentinel_dir = data_dir / 'sentinel2' / area_name
    dem_dir.mkdir(parents=True, exist_ok=True)
    sentinel_dir.mkdir(parents=True, exist_ok=True)

    # -- Conectar -----------------------------------------------------------
    catalog = get_planetary_computer_client()

    # -- Estructura de resultados -------------------------------------------
    results = {
        'success':        False,
        'area_name':      area_name,
        'bbox':           bbox_wgs84,
        'epsg':           epsg_code,
        'dem_tiles':      0,
        'dem_files':      [],
        'sentinel_years': [],
        'sentinel_files': {},
        'total_size_gb':  0.0,
        'metadata_path':  None,
        'timestamp':      datetime.now().isoformat(),
    }

    dem_files = []  # Inicializar aquí para que esté disponible en la sección de visualización

    # ===================================================================
    # PASO 1/2: DEM (NASADEM 30m)
    # ===================================================================
    print("\n[DEM] NASADEM 30m")
    try:
        items_dem = list(catalog.search(
            collections=['nasadem'],
            bbox=bbox_wgs84,
        ).items())

        if not items_dem:
            print("  [WARN] No se encontraron tiles DEM para esta área.")
        else:
            for i, item in enumerate(items_dem, 1):
                output_file = dem_dir / f"{item.id}_elevation.tif"
                if output_file.exists() and output_file.stat().st_size > 10_000:
                    size_mb = output_file.stat().st_size / 1_048_576
                    print(f"  [ok] {item.id}  ({size_mb:.1f} MB, existente)")
                    dem_files.append(output_file)
                    continue
                print(f"  [dl] {item.id} [{i}/{len(items_dem)}]...", end=" ", flush=True)
                path = download_band_windowed(item, 'elevation', bbox_wgs84, output_file)
                if path:
                    size_mb = path.stat().st_size / 1_048_576
                    print(f"ok ({size_mb:.1f} MB)")
                    dem_files.append(path)

        results['dem_tiles'] = len(dem_files)
        results['dem_files'] = [str(f) for f in dem_files]
        print(f"  Total: {len(dem_files)} tile(s)")

    except Exception as e:
        print(f"  [ERROR] {e}")
        logger.exception("Error en descarga DEM")

    # ===================================================================
    # PASO 2/2: Sentinel-2 L2A (multitemporal)
    # ===================================================================
    print(f"\n[Sentinel-2 L2A] Temporada seca: meses {months}  |  nubosidad < {max_cloud_cover}%")

    sentinel_downloads = {}
    total_size_bytes   = sum(f.stat().st_size for f in dem_files if f.exists())

    for year in years:
        start_month = min(months)
        end_month   = max(months)
        last_day    = calendar.monthrange(year, end_month)[1]
        date_range  = f"{year}-{start_month:02d}-01/{year}-{end_month:02d}-{last_day:02d}"

        try:
            items = list(catalog.search(
                collections=['sentinel-2-l2a'],
                bbox=bbox_wgs84,
                datetime=date_range,
                query={'eo:cloud_cover': {'lt': max_cloud_cover}},
            ).items())

            if not items:
                print(f"  {year}  —  sin escenas claras (nubosidad > {max_cloud_cover}%)")
                continue

            # Escena con menor nubosidad; desempate: fecha más reciente
            best_item   = min(items, key=lambda x: (
                x.properties.get('eo:cloud_cover', 100),
                -x.datetime.timestamp(),
            ))
            cloud_cover = best_item.properties.get('eo:cloud_cover', 0.0)
            scene_date  = best_item.datetime.strftime('%Y-%m-%d')

            year_dir = sentinel_dir / str(year)
            year_dir.mkdir(exist_ok=True)

            downloaded_bands = {}
            year_size_bytes  = 0
            new_downloads    = 0

            for band in sentinel_bands:
                output_file = year_dir / f"{best_item.id}_{band}.tif"
                if output_file.exists() and output_file.stat().st_size > 10_000:
                    downloaded_bands[band] = str(output_file)
                    year_size_bytes += output_file.stat().st_size
                    continue
                path = download_band_windowed(
                    item=best_item, band=band,
                    bbox_wgs84=bbox_wgs84, output_path=output_file,
                )
                if path:
                    downloaded_bands[band] = str(path)
                    year_size_bytes += path.stat().st_size
                    new_downloads += 1

            total_size_bytes += year_size_bytes
            status   = "[dl]" if new_downloads else "[ok]"
            bands_ok = ' '.join(downloaded_bands.keys())
            size_mb  = year_size_bytes / 1_048_576
            new_str  = f"  ({new_downloads} nuevas)" if new_downloads else "  (existentes)"
            print(f"  {status} {year}  {scene_date}  cloud:{cloud_cover:.1f}%  [{bands_ok}]  {size_mb:.1f} MB{new_str}")

            if downloaded_bands:
                sentinel_downloads[year] = {
                    'scene_id':    best_item.id,
                    'cloud_cover': cloud_cover,
                    'date':        scene_date,
                    'bands':       downloaded_bands,
                }
                results['sentinel_years'].append(year)

        except Exception as e:
            print(f"  {year}  [ERROR] {e}")
            logger.exception(f"Error Sentinel-2 año {year}")
            continue

    results['sentinel_files'] = sentinel_downloads
    results['total_size_gb']  = total_size_bytes / 1_073_741_824

    # ===================================================================
    # RESUMEN
    # ===================================================================
    print("\n" + "=" * 80)
    print(f"  Tiles DEM       {results['dem_tiles']}")
    print(f"  Años Sentinel-2 {len(results['sentinel_years'])} / {len(years)}  {results['sentinel_years']}")
    print(f"  Tamaño total    {results['total_size_gb']:.2f} GB")

    # ===================================================================
    # VISUALIZACIÓN DEM
    # ===================================================================
    if visualize and dem_files:
        try:
            import matplotlib.pyplot as plt

            if len(dem_files) > 1:
                src_files = [rasterio.open(f) for f in dem_files]
                mosaic, transform = merge(src_files)
                for s in src_files:
                    s.close()
                mosaic = mosaic[0] if mosaic.ndim == 3 else mosaic
            else:
                with rasterio.open(dem_files[0]) as src:
                    mosaic    = src.read(1)
                    transform = src.transform

            fig, axes = plt.subplots(1, 2, figsize=(16, 6))

            im1 = axes[0].imshow(mosaic, cmap='terrain', aspect='auto')
            axes[0].set_title(f'DEM - {config["description"]}', fontweight='bold')
            plt.colorbar(im1, ax=axes[0], label='Elevación (m)', shrink=0.8)

            bbox_geom = box(*bbox_wgs84)
            gdf = gpd.GeoDataFrame({'geometry': [bbox_geom]}, crs='EPSG:4326')
            gdf.plot(ax=axes[1], facecolor='none', edgecolor='red', linewidth=2)
            axes[1].set_title(f'Área de Estudio\nEPSG:{epsg_code}', fontweight='bold')
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            fig_dir  = project_root / 'figures' / 'downloads'
            fig_dir.mkdir(parents=True, exist_ok=True)
            fig_path = fig_dir / f"download_{area_name}.png"
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')
            plt.show()
            plt.close(fig)
            print(f"  Visualización guardada: {fig_path}")

        except Exception as e:
            print(f"  [WARN] No se pudo crear visualización: {e}")

    # ===================================================================
    # FINALIZAR: marcar éxito ANTES de guardar metadatos
    # ===================================================================
    results['success'] = True

    if save_metadata:
        metadata_file = data_dir / f"metadata_{area_name}.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        results['metadata_path'] = str(metadata_file)
        print(f"  Metadatos: {metadata_file}")

    print("=" * 80)
    print("DESCARGA COMPLETADA EXITOSAMENTE")
    print("=" * 80)
    return results
