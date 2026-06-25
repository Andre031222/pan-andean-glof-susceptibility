# =============================================================================
# GLOF Andes Project - Source Package
# =============================================================================
"""
Módulos fuente para el análisis de susceptibilidad GLOF Pan-Andino.

Módulos disponibles:
    - download:        Descarga de datos satelitales (Sentinel-2, DEM)
                       vía Microsoft Planetary Computer STAC API.
    - data_validator:  Validación de archivos raster, CRS y cobertura.
    - gpu_utils:       Utilidades GPU con CuPy/CUDA para procesamiento
                       acelerado de DEM y ML.
    - visualization:   Funciones de visualización para figuras de publicación.
"""

__version__ = '3.0.0'
__author__ = 'Richar Andre Vilca Solorzano'
__institution__ = 'Universidad Nacional del Altiplano, Puno, Peru'
