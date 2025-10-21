from __future__ import annotations
from typing import Tuple, Dict, Any, BinaryIO
import numpy as np
import rasterio

def read_geotiff(path_or_buffer) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Lit un GeoTIFF en conservant le CRS et le transform.
    Retourne (array, meta), array en (C, H, W).
    """
    with rasterio.open(path_or_buffer) as src:
        arr = src.read()
        meta = src.meta.copy()
    return arr, meta

def write_geotiff_like(dest: BinaryIO | str, reference_meta: Dict[str, Any], array: np.ndarray) -> None:
    """Écrit un GeoTIFF à partir d'un meta de référence (CRS, transform).
    `array` peut être (H,W) ou (C,H,W); dtype déduit.
    """
    meta = reference_meta.copy()
    if array.ndim == 2:
        meta.update(count=1, height=array.shape[0], width=array.shape[1], dtype=str(array.dtype))
        with rasterio.open(dest, 'w', **meta) as dst:
            dst.write(array, 1)
    elif array.ndim == 3:
        meta.update(count=array.shape[0], height=array.shape[1], width=array.shape[2], dtype=str(array.dtype))
        with rasterio.open(dest, 'w', **meta) as dst:
            dst.write(array)
    else:
        raise ValueError("array doit être 2D ou 3D")
