"""
Data Validation Module for GLOF Andes Project
==============================================

Provides centralized validation functions for:
- Downloaded raster files (DEM, Sentinel-2)
- CRS matching and reprojection
- File integrity checks
- Metadata validation

Author: GLOF Andes Project Team
Date: 2026-01-11
Version: 1.0
"""

import os
from pathlib import Path
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.warp import transform_bounds
import geopandas as gpd
from typing import Optional, Tuple, List, Dict, Union
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RasterValidationError(Exception):
    """Exception raised when raster validation fails."""
    pass


class CRSValidationError(Exception):
    """Exception raised when CRS validation fails."""
    pass


def validate_raster_file(
    file_path: Union[str, Path],
    min_width: int = 10,
    min_height: int = 10,
    expected_bands: Optional[int] = None,
    expected_crs: Optional[str] = None,
    check_nodata: bool = True,
    remove_if_invalid: bool = False
) -> bool:
    """
    Validate a raster file for integrity and basic properties.

    Parameters
    ----------
    file_path : str or Path
        Path to raster file to validate
    min_width : int, default=10
        Minimum acceptable width in pixels
    min_height : int, default=10
        Minimum acceptable height in pixels
    expected_bands : int, optional
        Expected number of bands (None to skip check)
    expected_crs : str, optional
        Expected CRS (e.g., 'EPSG:4326') (None to skip check)
    check_nodata : bool, default=True
        Check if nodata value is properly set
    remove_if_invalid : bool, default=False
        If True, delete file if validation fails

    Returns
    -------
    bool
        True if file is valid, False otherwise

    Raises
    ------
    RasterValidationError
        If file is invalid and remove_if_invalid=False

    Examples
    --------
    >>> validate_raster_file('dem.tif', expected_crs='EPSG:32718')
    True

    >>> validate_raster_file('corrupted.tif', remove_if_invalid=True)
    [ERROR] File corrupted.tif is invalid, removing...
    False
    """
    file_path = Path(file_path)

    if not file_path.exists():
        error_msg = f"File does not exist: {file_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    try:
        with rasterio.open(file_path) as src:
            # Check dimensions
            if src.width < min_width or src.height < min_height:
                raise RasterValidationError(
                    f"Invalid dimensions: {src.width}x{src.height} "
                    f"(minimum: {min_width}x{min_height})"
                )

            # Check band count
            if expected_bands is not None and src.count != expected_bands:
                raise RasterValidationError(
                    f"Invalid band count: {src.count} (expected: {expected_bands})"
                )

            # Check CRS
            if expected_crs is not None:
                if src.crs is None:
                    raise RasterValidationError("No CRS defined")
                if str(src.crs) != expected_crs:
                    raise RasterValidationError(
                        f"CRS mismatch: {src.crs} (expected: {expected_crs})"
                    )

            # Check nodata value
            if check_nodata and src.nodata is None:
                logger.warning(f"No nodata value defined for {file_path.name}")

            # Try to read a small window to ensure file is not corrupted
            try:
                window = ((0, min(10, src.height)), (0, min(10, src.width)))
                test_data = src.read(1, window=window)

                # Check if data is all zeros (suspicious)
                if np.all(test_data == 0):
                    logger.warning(
                        f"All values are zero in test window for {file_path.name}"
                    )

            except Exception as e:
                raise RasterValidationError(f"Cannot read data: {str(e)}")

        logger.info(f"[OK] {file_path.name} validated successfully")
        return True

    except Exception as e:
        error_msg = f"Validation failed for {file_path.name}: {str(e)}"
        logger.error(error_msg)

        if remove_if_invalid:
            logger.warning(f"Removing invalid file: {file_path}")
            try:
                file_path.unlink()
            except Exception as unlink_error:
                logger.error(f"Failed to remove file: {unlink_error}")

        return False


def validate_crs_match(
    file1: Union[str, Path],
    file2: Union[str, Path],
    allow_same_projection: bool = True
) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Check if two raster files have matching CRS.

    Parameters
    ----------
    file1, file2 : str or Path
        Paths to raster files to compare
    allow_same_projection : bool, default=True
        If True, allow different CRS with same projection
        (e.g., EPSG:32718 and EPSG:32719 are both UTM)

    Returns
    -------
    tuple
        (match: bool, crs1: str, crs2: str)

    Examples
    --------
    >>> match, crs1, crs2 = validate_crs_match('dem.tif', 'lakes.gpkg')
    >>> if not match:
    >>>     print(f"CRS mismatch: {crs1} != {crs2}")
    """
    try:
        with rasterio.open(file1) as src1:
            crs1 = src1.crs

        # Check if file2 is raster or vector
        if str(file2).endswith(('.gpkg', '.geojson', '.shp')):
            gdf = gpd.read_file(file2)
            crs2 = gdf.crs
        else:
            with rasterio.open(file2) as src2:
                crs2 = src2.crs

        if crs1 == crs2:
            return True, str(crs1), str(crs2)

        if allow_same_projection:
            # Check if both are in same projection system
            if crs1 is not None and crs2 is not None:
                # Same projection type (e.g., both UTM)
                if (crs1.to_dict().get('proj') == crs2.to_dict().get('proj') and
                    crs1.to_dict().get('datum') == crs2.to_dict().get('datum')):
                    logger.info(f"Different CRS but same projection: {crs1} vs {crs2}")
                    return True, str(crs1), str(crs2)

        return False, str(crs1) if crs1 else None, str(crs2) if crs2 else None

    except Exception as e:
        logger.error(f"Error checking CRS: {str(e)}")
        return False, None, None


def validate_bbox_overlap(
    file1: Union[str, Path],
    file2: Union[str, Path],
    min_overlap_pct: float = 50.0
) -> Tuple[bool, float]:
    """
    Check if two raster files have sufficient bounding box overlap.

    Parameters
    ----------
    file1, file2 : str or Path
        Paths to files to compare
    min_overlap_pct : float, default=50.0
        Minimum required overlap percentage

    Returns
    -------
    tuple
        (has_overlap: bool, overlap_pct: float)

    Examples
    --------
    >>> has_overlap, pct = validate_bbox_overlap('dem.tif', 'lakes.gpkg')
    >>> print(f"Overlap: {pct:.1f}%")
    """
    try:
        with rasterio.open(file1) as src1:
            bounds1 = src1.bounds
            crs1 = src1.crs

        # Get bounds from file2
        if str(file2).endswith(('.gpkg', '.geojson', '.shp')):
            gdf = gpd.read_file(file2)
            # Reproject to crs1 if needed
            if gdf.crs != crs1:
                gdf = gdf.to_crs(crs1)
            bounds2 = gdf.total_bounds  # (minx, miny, maxx, maxy)
        else:
            with rasterio.open(file2) as src2:
                if src2.crs != crs1:
                    # Transform bounds to crs1
                    bounds2 = transform_bounds(src2.crs, crs1, *src2.bounds)
                else:
                    bounds2 = src2.bounds

        # Calculate intersection
        minx = max(bounds1.left, bounds2[0])
        miny = max(bounds1.bottom, bounds2[1])
        maxx = min(bounds1.right, bounds2[2])
        maxy = min(bounds1.top, bounds2[3])

        if minx >= maxx or miny >= maxy:
            return False, 0.0

        # Calculate areas
        area1 = (bounds1.right - bounds1.left) * (bounds1.top - bounds1.bottom)
        area_intersection = (maxx - minx) * (maxy - miny)

        overlap_pct = (area_intersection / area1) * 100

        has_overlap = overlap_pct >= min_overlap_pct

        if not has_overlap:
            logger.warning(
                f"Insufficient overlap: {overlap_pct:.1f}% "
                f"(minimum: {min_overlap_pct}%)"
            )

        return has_overlap, overlap_pct

    except Exception as e:
        logger.error(f"Error checking overlap: {str(e)}")
        return False, 0.0


def validate_temporal_consistency(
    year_dirs: List[Path],
    expected_bands: List[str],
    study_area: str
) -> Dict[int, Dict[str, bool]]:
    """
    Validate that all required bands exist for each year.

    Parameters
    ----------
    year_dirs : list of Path
        List of year directories (e.g., [year_2019, year_2021, year_2023])
    expected_bands : list of str
        Expected band names (e.g., ['B03', 'B08', 'B11'])
    study_area : str
        Study area name for logging

    Returns
    -------
    dict
        Nested dict: {year: {band: exists}}

    Examples
    --------
    >>> year_dirs = [Path('data/sentinel2/year_2019'), ...]
    >>> result = validate_temporal_consistency(year_dirs, ['B03', 'B08'], 'blanca')
    >>> for year, bands in result.items():
    >>>     print(f"{year}: {sum(bands.values())}/{len(bands)} bands OK")
    """
    results = {}

    for year_dir in year_dirs:
        if not year_dir.exists():
            logger.warning(f"Year directory does not exist: {year_dir}")
            continue

        # Extract year from directory name
        year_str = year_dir.name.split('_')[-1]
        try:
            year = int(year_str)
        except ValueError:
            logger.error(f"Cannot extract year from: {year_dir.name}")
            continue

        band_status = {}
        for band in expected_bands:
            # Look for files with band name
            band_files = list(year_dir.glob(f'*{band}*.tif'))

            if not band_files:
                band_status[band] = False
                logger.warning(
                    f"[{study_area}] Year {year}: Band {band} NOT FOUND"
                )
            else:
                # Validate the first matching file
                try:
                    is_valid = validate_raster_file(
                        band_files[0],
                        min_width=100,
                        min_height=100,
                        expected_bands=1,
                        remove_if_invalid=False
                    )
                    band_status[band] = is_valid
                except Exception as e:
                    logger.error(f"Validation failed for {band_files[0]}: {e}")
                    band_status[band] = False

        results[year] = band_status

        # Summary for this year
        n_valid = sum(band_status.values())
        n_total = len(band_status)
        if n_valid == n_total:
            logger.info(f"[{study_area}] Year {year}: All {n_total} bands OK")
        else:
            logger.warning(
                f"[{study_area}] Year {year}: Only {n_valid}/{n_total} bands OK"
            )

    return results


def clean_and_redownload_check(
    output_dir: Path,
    force_redownload: bool = False,
    file_pattern: str = '*.tif'
) -> bool:
    """
    Check if data should be re-downloaded.

    If force_redownload=True, removes existing files.
    Otherwise, checks if files exist and are valid.

    Parameters
    ----------
    output_dir : Path
        Directory to check/clean
    force_redownload : bool, default=False
        If True, remove all existing files
    file_pattern : str, default='*.tif'
        Pattern for files to check

    Returns
    -------
    bool
        True if download is needed, False if existing data is OK

    Examples
    --------
    >>> should_download = clean_and_redownload_check(
    ...     Path('data/raw/dem/blanca'),
    ...     force_redownload=True
    ... )
    >>> if should_download:
    >>>     print("Downloading fresh data...")
    """
    if not output_dir.exists():
        logger.info(f"Directory does not exist, will create: {output_dir}")
        return True

    existing_files = list(output_dir.glob(file_pattern))

    if not existing_files:
        logger.info(f"No existing files found in {output_dir}")
        return True

    if force_redownload:
        logger.warning(
            f"FORCE REDOWNLOAD enabled - removing {len(existing_files)} files"
        )
        for file in existing_files:
            try:
                file.unlink()
                logger.info(f"Removed: {file.name}")
            except Exception as e:
                logger.error(f"Failed to remove {file.name}: {e}")
        return True

    # Check if existing files are valid
    logger.info(f"Found {len(existing_files)} existing files, validating...")

    invalid_count = 0
    for file in existing_files:
        try:
            is_valid = validate_raster_file(
                file,
                remove_if_invalid=True  # Auto-remove corrupted files
            )
            if not is_valid:
                invalid_count += 1
        except Exception as e:
            logger.error(f"Error validating {file.name}: {e}")
            invalid_count += 1

    if invalid_count > 0:
        logger.warning(
            f"{invalid_count}/{len(existing_files)} files were invalid and removed"
        )
        return True  # Need to re-download

    logger.info(f"All {len(existing_files)} existing files are valid - skipping download")
    return False


def create_download_report(
    output_dir: Path,
    metadata: Dict,
    report_name: str = 'download_report.txt'
) -> None:
    """
    Create a text report summarizing download results.

    Parameters
    ----------
    output_dir : Path
        Directory where report will be saved
    metadata : dict
        Download metadata (files, sizes, timestamps, etc.)
    report_name : str, default='download_report.txt'
        Name of report file

    Examples
    --------
    >>> metadata = {
    ...     'study_area': 'cordillera_blanca',
    ...     'download_date': '2026-01-11',
    ...     'files_downloaded': 10,
    ...     'total_size_mb': 1234.5
    ... }
    >>> create_download_report(Path('data/raw/dem/blanca'), metadata)
    """
    report_path = output_dir / report_name

    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("DOWNLOAD REPORT\n")
        f.write("=" * 80 + "\n\n")

        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write(f"Report created: {output_dir}\n")
        f.write("=" * 80 + "\n")

    logger.info(f"Download report saved: {report_path}")


# Example usage
if __name__ == "__main__":
    # Test validation functions
    print("Data Validator Module")
    print("=" * 60)
    print("Available functions:")
    print("  - validate_raster_file()")
    print("  - validate_crs_match()")
    print("  - validate_bbox_overlap()")
    print("  - validate_temporal_consistency()")
    print("  - clean_and_redownload_check()")
    print("  - create_download_report()")
    print("=" * 60)
