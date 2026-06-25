# =============================================================================
# GPU Utilities for GLOF Andes Project
# Optimized for NVIDIA RTX 3050 (4-8GB VRAM)
# =============================================================================
"""
GPU acceleration utilities for the GLOF Andes Project.

This module provides:
- GPU detection and configuration
- CuPy-based array operations (GPU-accelerated NumPy)
- Numba CUDA JIT compilation
- GPU-accelerated image processing
- XGBoost/LightGBM GPU configuration

Usage:
    from src.gpu_utils import GPUConfig, gpu_array, get_xgboost_gpu_params

    # Check GPU availability
    config = GPUConfig()
    print(config)

    # Use GPU arrays
    if config.has_gpu:
        arr_gpu = gpu_array(my_numpy_array)
"""

import os
import sys
import warnings
from typing import Optional, Dict, Any, Union
import numpy as np

# =============================================================================
# GPU DETECTION AND CONFIGURATION
# =============================================================================

class GPUConfig:
    """
    GPU configuration and detection class.

    Attributes:
        has_gpu (bool): Whether CUDA GPU is available
        device_name (str): GPU device name
        memory_total (int): Total GPU memory in bytes
        memory_free (int): Free GPU memory in bytes
        cuda_version (str): CUDA version string
        cupy_available (bool): Whether CuPy is installed
        torch_available (bool): Whether PyTorch CUDA is available
        numba_available (bool): Whether Numba CUDA is available
    """

    def __init__(self):
        """Initialize GPU configuration by detecting available hardware."""
        self.has_gpu = False
        self.device_name = "CPU"
        self.device_count = 0
        self.memory_total = 0
        self.memory_free = 0
        self.cuda_version = None
        self.cupy_available = False
        self.torch_available = False
        self.numba_available = False

        self._detect_gpu()

    def _detect_gpu(self):
        """Detect GPU and available libraries."""
        # Try nvidia-ml-py (preferred) then pynvml as fallback
        nvml = None
        try:
            from pynvml import nvml as _nvml
            nvml = _nvml
        except Exception:
            pass
        if nvml is None:
            try:
                import pynvml as nvml  # legacy fallback
            except Exception:
                pass

        if nvml is not None:
            try:
                nvml.nvmlInit()
                self.device_count = nvml.nvmlDeviceGetCount()

                if self.device_count > 0:
                    self.has_gpu = True
                    handle = nvml.nvmlDeviceGetHandleByIndex(0)
                    self.device_name = nvml.nvmlDeviceGetName(handle)
                    if isinstance(self.device_name, bytes):
                        self.device_name = self.device_name.decode('utf-8')

                    mem_info = nvml.nvmlDeviceGetMemoryInfo(handle)
                    self.memory_total = mem_info.total
                    self.memory_free = mem_info.free

                    driver_version = nvml.nvmlSystemGetDriverVersion()
                    if isinstance(driver_version, bytes):
                        driver_version = driver_version.decode('utf-8')
                    self.cuda_version = driver_version

                nvml.nvmlShutdown()
            except Exception:
                pass

        # Check CuPy
        try:
            import cupy as cp
            self.cupy_available = True
            if not self.has_gpu:
                self.has_gpu = True
            # Always try to fill in device info from CuPy if pynvml missed it
            if self.device_name in ("CPU", ""):
                try:
                    props = cp.cuda.runtime.getDeviceProperties(0)
                    raw = props.get('name', b'')
                    self.device_name = raw.decode('utf-8') if isinstance(raw, bytes) else str(raw)
                    if not self.memory_total:
                        self.memory_total = props.get('totalGlobalMem', 0)
                        self.memory_free = self.memory_total  # best-effort fallback
                    if not self.device_count:
                        self.device_count = cp.cuda.runtime.getDeviceCount()
                except Exception:
                    pass
        except ImportError:
            pass
        except Exception:
            pass

        # Check PyTorch CUDA
        try:
            import torch
            if torch.cuda.is_available():
                self.torch_available = True
                if not self.has_gpu:
                    self.has_gpu = True
                    self.device_name = torch.cuda.get_device_name(0)
                    self.memory_total = torch.cuda.get_device_properties(0).total_memory
        except ImportError:
            pass

        # Check Numba CUDA
        try:
            from numba import cuda
            if cuda.is_available():
                self.numba_available = True
        except ImportError:
            pass

    def get_memory_gb(self) -> tuple:
        """Get GPU memory in GB (total, free)."""
        return (
            self.memory_total / (1024**3) if self.memory_total else 0,
            self.memory_free / (1024**3) if self.memory_free else 0
        )

    def __str__(self) -> str:
        """String representation of GPU config."""
        mem_total, mem_free = self.get_memory_gb()

        lines = [
            "=" * 60,
            "GPU CONFIGURATION",
            "=" * 60,
            f"GPU Available: {self.has_gpu}",
            f"Device: {self.device_name}",
            f"Device Count: {self.device_count}",
        ]

        if self.has_gpu:
            lines.extend([
                f"Memory Total: {mem_total:.1f} GB",
                f"Memory Free: {mem_free:.1f} GB",
                f"CUDA Version: {self.cuda_version}",
                "",
                "Library Support:",
                f"  - CuPy:         {'yes' if self.cupy_available else 'no'}",
                f"  - PyTorch CUDA: {'yes' if self.torch_available else 'no'}",
                f"  - Numba CUDA:   {'yes' if self.numba_available else 'no'}",
            ])

        lines.append("=" * 60)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"GPUConfig(has_gpu={self.has_gpu}, device='{self.device_name}')"


# Global GPU config instance
_gpu_config = None

def get_gpu_config() -> GPUConfig:
    """Get or create the global GPU configuration."""
    global _gpu_config
    if _gpu_config is None:
        _gpu_config = GPUConfig()
    return _gpu_config


# =============================================================================
# CUPY ARRAY UTILITIES (GPU-ACCELERATED NUMPY)
# =============================================================================

def gpu_array(arr: np.ndarray, dtype=None) -> Union[np.ndarray, 'cp.ndarray']:
    """
    Convert NumPy array to CuPy array if GPU is available.

    Parameters:
    -----------
    arr : np.ndarray
        Input NumPy array
    dtype : optional
        Data type for the array

    Returns:
    --------
    CuPy array if GPU available, otherwise NumPy array
    """
    config = get_gpu_config()

    if config.cupy_available:
        import cupy as cp
        if dtype:
            return cp.asarray(arr, dtype=dtype)
        return cp.asarray(arr)

    if dtype:
        return np.asarray(arr, dtype=dtype)
    return arr


def cpu_array(arr) -> np.ndarray:
    """
    Convert CuPy array back to NumPy array.

    Parameters:
    -----------
    arr : np.ndarray or cp.ndarray
        Input array (GPU or CPU)

    Returns:
    --------
    np.ndarray : CPU NumPy array
    """
    config = get_gpu_config()

    if config.cupy_available:
        import cupy as cp
        if isinstance(arr, cp.ndarray):
            return cp.asnumpy(arr)

    return np.asarray(arr)


def get_array_module(arr):
    """
    Get the array module (numpy or cupy) for the given array.

    Parameters:
    -----------
    arr : array-like
        Input array

    Returns:
    --------
    module : numpy or cupy module
    """
    config = get_gpu_config()

    if config.cupy_available:
        import cupy as cp
        return cp.get_array_module(arr)

    return np


# =============================================================================
# GPU-ACCELERATED IMAGE PROCESSING
# =============================================================================

def gpu_ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """
    Calculate NDWI using GPU acceleration.

    NDWI = (Green - NIR) / (Green + NIR)

    Parameters:
    -----------
    green : np.ndarray
        Green band array
    nir : np.ndarray
        NIR band array

    Returns:
    --------
    np.ndarray : NDWI values
    """
    config = get_gpu_config()

    if config.cupy_available:
        import cupy as cp
        green_gpu = cp.asarray(green, dtype=cp.float32)
        nir_gpu = cp.asarray(nir, dtype=cp.float32)
        denom = green_gpu + nir_gpu
        denom = cp.where(denom == 0, cp.float32(1e-10), denom)
        ndwi = (green_gpu - nir_gpu) / denom
        ndwi = cp.where(cp.isfinite(ndwi), ndwi, cp.float32(0))
        return cp.asnumpy(ndwi)

    # CPU fallback
    with np.errstate(divide='ignore', invalid='ignore'):
        ndwi = (green - nir) / (green + nir)
        ndwi = np.where(np.isfinite(ndwi), ndwi, 0)
    return ndwi


def gpu_mndwi(green: np.ndarray, swir: np.ndarray) -> np.ndarray:
    """
    Calculate MNDWI using GPU acceleration.

    MNDWI = (Green - SWIR) / (Green + SWIR)

    Parameters:
    -----------
    green : np.ndarray
        Green band array
    swir : np.ndarray
        SWIR band array

    Returns:
    --------
    np.ndarray : MNDWI values
    """
    config = get_gpu_config()

    if config.cupy_available:
        import cupy as cp
        green_gpu = cp.asarray(green, dtype=cp.float32)
        swir_gpu = cp.asarray(swir, dtype=cp.float32)
        denom = green_gpu + swir_gpu
        denom = cp.where(denom == 0, cp.float32(1e-10), denom)
        mndwi = (green_gpu - swir_gpu) / denom
        mndwi = cp.where(cp.isfinite(mndwi), mndwi, cp.float32(0))
        return cp.asnumpy(mndwi)

    # CPU fallback
    with np.errstate(divide='ignore', invalid='ignore'):
        mndwi = (green - swir) / (green + swir)
        mndwi = np.where(np.isfinite(mndwi), mndwi, 0)
    return mndwi


def gpu_morphology_clean(mask: np.ndarray, min_size: int = 100) -> np.ndarray:
    """
    Clean binary mask using GPU-accelerated morphological operations.

    Parameters:
    -----------
    mask : np.ndarray
        Binary mask
    min_size : int
        Minimum object size in pixels

    Returns:
    --------
    np.ndarray : Cleaned mask
    """
    config = get_gpu_config()

    if config.cupy_available:
        import cupy as cp
        from cupyx.scipy import ndimage as cp_ndimage

        mask_gpu = cp.asarray(mask.astype(bool))

        # Morphological closing
        struct = cp.ones((3, 3), dtype=bool)
        cleaned = cp_ndimage.binary_closing(mask_gpu, structure=struct)

        # Fill holes
        cleaned = cp_ndimage.binary_fill_holes(cleaned)

        # Note: remove_small_objects is complex on GPU, fall back to CPU for this
        result = cp.asnumpy(cleaned)

        # Remove small objects on CPU
        from skimage import morphology
        result = morphology.remove_small_objects(result, min_size=min_size)

        return result.astype(np.uint8)

    # CPU fallback
    from scipy import ndimage
    from skimage import morphology

    cleaned = ndimage.binary_closing(mask, structure=np.ones((3, 3)))
    cleaned = ndimage.binary_fill_holes(cleaned)
    cleaned = morphology.remove_small_objects(cleaned.astype(bool), min_size=min_size)

    return cleaned.astype(np.uint8)


# =============================================================================
# GPU-ACCELERATED MACHINE LEARNING CONFIGURATION
# =============================================================================

def get_xgboost_gpu_params(base_params: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Get XGBoost parameters configured for GPU training.

    Parameters:
    -----------
    base_params : dict, optional
        Base parameters to extend

    Returns:
    --------
    dict : XGBoost parameters with GPU configuration
    """
    config = get_gpu_config()

    params = base_params.copy() if base_params else {}

    if config.has_gpu:
        params.update({
            'tree_method': 'gpu_hist',  # GPU histogram-based algorithm
            'device': 'cuda',            # Use CUDA
            'predictor': 'gpu_predictor',
            # Memory optimization for RTX 3050 (4-8GB)
            'max_bin': 256,              # Reduce memory usage
            'gpu_id': 0,
        })
        print(f"XGBoost configured for GPU: {config.device_name}")
    else:
        params.update({
            'tree_method': 'hist',       # CPU histogram method (fast)
        })
        print("XGBoost configured for CPU")

    return params


def get_lightgbm_gpu_params(base_params: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Get LightGBM parameters configured for GPU training.

    Parameters:
    -----------
    base_params : dict, optional
        Base parameters to extend

    Returns:
    --------
    dict : LightGBM parameters with GPU configuration
    """
    config = get_gpu_config()

    params = base_params.copy() if base_params else {}

    if config.has_gpu:
        params.update({
            'device': 'gpu',
            'gpu_platform_id': 0,
            'gpu_device_id': 0,
            # Memory optimization
            'gpu_use_dp': False,  # Use single precision (less memory)
            'max_bin': 255,
        })
        print(f"LightGBM configured for GPU: {config.device_name}")
    else:
        params.update({
            'device': 'cpu',
        })
        print("LightGBM configured for CPU")

    return params


def get_pytorch_device() -> 'torch.device':
    """
    Get the best available PyTorch device.

    Returns:
    --------
    torch.device : CUDA device if available, otherwise CPU
    """
    config = get_gpu_config()

    if config.torch_available:
        import torch
        device = torch.device('cuda:0')
        print(f"PyTorch using GPU: {config.device_name}")
        return device

    import torch
    print("PyTorch using CPU")
    return torch.device('cpu')


# =============================================================================
# NUMBA CUDA JIT UTILITIES
# =============================================================================

def cuda_available() -> bool:
    """Check if Numba CUDA is available."""
    return get_gpu_config().numba_available


if cuda_available():
    from numba import cuda

    @cuda.jit
    def _cuda_ndwi_kernel(green, nir, result):
        """CUDA kernel for NDWI calculation."""
        i, j = cuda.grid(2)
        if i < green.shape[0] and j < green.shape[1]:
            g = green[i, j]
            n = nir[i, j]
            denom = g + n
            if denom != 0:
                result[i, j] = (g - n) / denom
            else:
                result[i, j] = 0.0

    def cuda_ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
        """
        Calculate NDWI using Numba CUDA kernel.

        This is faster than CuPy for large arrays.
        """
        green_device = cuda.to_device(green.astype(np.float32))
        nir_device = cuda.to_device(nir.astype(np.float32))
        result_device = cuda.device_array(green.shape, dtype=np.float32)

        # Configure blocks and threads
        threadsperblock = (16, 16)
        blockspergrid_x = (green.shape[0] + threadsperblock[0] - 1) // threadsperblock[0]
        blockspergrid_y = (green.shape[1] + threadsperblock[1] - 1) // threadsperblock[1]
        blockspergrid = (blockspergrid_x, blockspergrid_y)

        _cuda_ndwi_kernel[blockspergrid, threadsperblock](
            green_device, nir_device, result_device
        )

        return result_device.copy_to_host()


# =============================================================================
# MEMORY MANAGEMENT
# =============================================================================

def clear_gpu_memory():
    """Clear GPU memory cache."""
    config = get_gpu_config()

    if config.cupy_available:
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        print("CuPy memory cleared")

    if config.torch_available:
        import torch
        torch.cuda.empty_cache()
        print("PyTorch CUDA cache cleared")


def get_gpu_memory_usage() -> Dict[str, float]:
    """
    Get current GPU memory usage.

    Returns:
    --------
    dict : Memory usage in GB (used, free, total)
    """
    config = get_gpu_config()

    if not config.has_gpu:
        return {'used': 0, 'free': 0, 'total': 0}

    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        pynvml.nvmlShutdown()

        return {
            'used': mem_info.used / (1024**3),
            'free': mem_info.free / (1024**3),
            'total': mem_info.total / (1024**3)
        }
    except:
        return {'used': 0, 'free': 0, 'total': config.memory_total / (1024**3)}


# =============================================================================
# MAIN - Test GPU configuration
# =============================================================================

if __name__ == "__main__":
    # Print GPU configuration
    config = GPUConfig()
    print(config)

    # Test GPU array operations
    if config.cupy_available:
        print("\nTesting CuPy operations...")
        import cupy as cp

        # Create test array
        arr_cpu = np.random.rand(1000, 1000).astype(np.float32)
        arr_gpu = gpu_array(arr_cpu)

        # Benchmark
        import time

        # CPU
        start = time.time()
        result_cpu = np.sqrt(arr_cpu) * np.sin(arr_cpu)
        cpu_time = time.time() - start

        # GPU
        start = time.time()
        result_gpu = cp.sqrt(arr_gpu) * cp.sin(arr_gpu)
        cp.cuda.Stream.null.synchronize()
        gpu_time = time.time() - start

        print(f"CPU time: {cpu_time*1000:.2f} ms")
        print(f"GPU time: {gpu_time*1000:.2f} ms")
        print(f"Speedup: {cpu_time/gpu_time:.1f}x")

        clear_gpu_memory()
