from __future__ import annotations
from typing import Tuple, Dict, Any, BinaryIO, Union, Optional
import numpy as np
import rasterio
from rasterio.errors import RasterioIOError
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def read_geotiff(
    path_or_buffer: Union[str, Path, BinaryIO],
    bands: Optional[list] = None,
    window: Optional[rasterio.windows.Window] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Lit un GeoTIFF avec gestion d'erreurs robuste.
    
    Args:
        path_or_buffer: Chemin ou buffer
        bands: Liste des bandes à lire (None = toutes)
        window: Fenêtre spatiale à lire
    
    Returns:
        (array, meta) - Array en (C, H, W) et métadonnées
    
    Raises:
        FileNotFoundError: Si le fichier n'existe pas
        RasterioIOError: Si erreur de lecture
    """
    try:
        with rasterio.open(path_or_buffer) as src:
            # Validation
            if bands is not None:
                if not all(1 <= b <= src.count for b in bands):
                    raise ValueError(f"Bandes invalides: {bands}. Disponibles: 1-{src.count}")
                arr = src.read(bands, window=window)
            else:
                arr = src.read(window=window)
            
            meta = src.meta.copy()
            
            # Log des informations
            logger.info(f"GeoTIFF lu: {arr.shape}, CRS: {meta.get('crs', 'N/A')}")
            
            return arr, meta
    
    except FileNotFoundError:
        logger.error(f"Fichier introuvable: {path_or_buffer}")
        raise
    except RasterioIOError as e:
        logger.error(f"Erreur de lecture GeoTIFF: {e}")
        raise
    except Exception as e:
        logger.error(f"Erreur inattendue: {e}")
        raise


def write_geotiff_like(
    dest: Union[BinaryIO, str, Path],
    reference_meta: Dict[str, Any],
    array: np.ndarray,
    compress: str = 'lzw',
    tiled: bool = True,
    blockxsize: int = 256,
    blockysize: int = 256,
    nodata: Optional[float] = None
) -> None:
    """
    Écrit un GeoTIFF avec options de compression et tuilage.
    
    Args:
        dest: Destination (fichier ou buffer)
        reference_meta: Métadonnées de référence
        array: Array (H,W) ou (C,H,W)
        compress: Compression ('lzw', 'deflate', 'jpeg', 'none')
        tiled: Activer le tuilage interne
        blockxsize: Taille de bloc X
        blockysize: Taille de bloc Y
        nodata: Valeur nodata
    
    Raises:
        ValueError: Si dimensions invalides
    """
    try:
        meta = reference_meta.copy()
        
        # Gestion des dimensions
        if array.ndim == 2:
            array = array[None, ...]  # Ajouter dimension canal
        
        if array.ndim != 3:
            raise ValueError(f"Array doit être 2D ou 3D, trouvé: {array.ndim}D")
        
        C, H, W = array.shape
        
        # Mise à jour des métadonnées
        meta.update({
            'count': C,
            'height': H,
            'width': W,
            'dtype': str(array.dtype),
            'compress': compress
        })
        
        if tiled:
            meta.update({
                'tiled': True,
                'blockxsize': blockxsize,
                'blockysize': blockysize
            })
        
        if nodata is not None:
            meta['nodata'] = nodata
        
        # Écriture
        with rasterio.open(dest, 'w', **meta) as dst:
            dst.write(array)
        
        logger.info(f"GeoTIFF écrit: {array.shape}, compression: {compress}")
    
    except Exception as e:
        logger.error(f"Erreur d'écriture GeoTIFF: {e}")
        raise


def create_cog(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    compress: str = 'lzw',
    resampling: str = 'bilinear'
) -> None:
    """
    Crée un Cloud Optimized GeoTIFF (COG).
    
    Args:
        input_path: Chemin du GeoTIFF source
        output_path: Chemin du COG de sortie
        compress: Méthode de compression
        resampling: Méthode de rééchantillonnage pour les overviews
    """
    try:
        from rasterio.shutil import copy
        from rasterio.enums import Resampling
        
        resampling_map = {
            'nearest': Resampling.nearest,
            'bilinear': Resampling.bilinear,
            'cubic': Resampling.cubic,
            'average': Resampling.average
        }
        
        with rasterio.open(input_path) as src:
            profile = src.profile.copy()
            profile.update({
                'tiled': True,
                'compress': compress,
                'blockxsize': 512,
                'blockysize': 512
            })
            
            # Création des overviews
            factors = [2, 4, 8, 16]
            src.build_overviews(factors, resampling_map[resampling])
            
            # Copie avec COG layout
            copy(
                src,
                output_path,
                copy_src_overviews=True,
                **profile
            )
        
        logger.info(f"COG créé: {output_path}")
    
    except Exception as e:
        logger.error(f"Erreur création COG: {e}")
        raise


def get_raster_stats(path: Union[str, Path]) -> Dict[str, Any]:
    """Calcule les statistiques d'un raster."""
    try:
        with rasterio.open(path) as src:
            arr = src.read(masked=True)
            
            stats = {
                'shape': arr.shape,
                'dtype': str(arr.dtype),
                'crs': str(src.crs) if src.crs else None,
                'bounds': src.bounds,
                'resolution': src.res,
                'count': src.count,
                'nodata': src.nodata
            }
            
            # Statistiques par bande
            band_stats = []
            for i in range(src.count):
                band = arr[i]
                band_stats.append({
                    'band': i + 1,
                    'min': float(band.min()),
                    'max': float(band.max()),
                    'mean': float(band.mean()),
                    'std': float(band.std())
                })
            
            stats['bands'] = band_stats
            
            return stats
    
    except Exception as e:
        logger.error(f"Erreur calcul statistiques: {e}")
        raise