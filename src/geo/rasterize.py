from __future__ import annotations
from typing import Dict, Tuple, Optional
import zlib
import numpy as np
import geopandas as gpd
from rasterio import features, Affine
from shapely.geometry import shape
import logging

logger = logging.getLogger(__name__)


def vector_to_mask(
    gdf: gpd.GeoDataFrame,
    out_shape: Tuple[int, int],
    transform: Affine,
    class_map: Dict[str, int],
    property_key: str = 'type',
    all_touched: bool = False,
    default_value: int = 1
) -> np.ndarray:
    """
    Rasterise des géométries vectorielles en masque.
    
    Args:
        gdf: GeoDataFrame avec géométries
        out_shape: (H, W) du raster de sortie
        transform: Transform affine du raster
        class_map: Mapping attribut -> ID de classe
        property_key: Nom de la colonne d'attributs
        all_touched: Rasteriser tous les pixels touchés
        default_value: Valeur par défaut si propriété absente
    
    Returns:
        Masque (H,W) uint8
    
    Raises:
        ValueError: Si GeoDataFrame vide
    """
    if len(gdf) == 0:
        logger.warning("GeoDataFrame vide, retour masque de zéros")
        return np.zeros(out_shape, dtype='uint8')
    
    # Préparation des shapes
    shapes = []
    
    for idx, row in gdf.iterrows():
        geom = row.geometry
        
        if geom is None or geom.is_empty:
            continue
        
        # Récupération de la classe
        if property_key in row:
            prop_value = str(row[property_key]).lower()
            label = class_map.get(prop_value, default_value)
        else:
            label = default_value
        
        shapes.append((geom, label))
    
    if not shapes:
        logger.warning("Aucune géométrie valide trouvée")
        return np.zeros(out_shape, dtype='uint8')
    
    # Rasterisation
    mask = features.rasterize(
        shapes=shapes,
        out_shape=out_shape,
        transform=transform,
        fill=0,
        dtype='uint8',
        all_touched=all_touched
    )
    
    # Statistiques
    unique, counts = np.unique(mask, return_counts=True)
    logger.info(f"Rasterisation: {len(shapes)} géométries -> {len(unique)} classes")
    for val, count in zip(unique, counts):
        if val > 0:
            logger.info(f"  Classe {val}: {count} pixels ({count/mask.size*100:.2f}%)")
    
    return mask


def rasterize_with_buffer(
    gdf: gpd.GeoDataFrame,
    out_shape: Tuple[int, int],
    transform: Affine,
    buffer_size: float = 0.0,
    **kwargs
) -> np.ndarray:
    """
    Rasterise avec un buffer optionnel autour des géométries.
    
    Args:
        gdf: GeoDataFrame
        out_shape: (H, W)
        transform: Transform affine
        buffer_size: Taille du buffer (en unités du CRS)
        **kwargs: Arguments additionnels pour vector_to_mask
    
    Returns:
        Masque (H,W) uint8
    """
    if buffer_size > 0:
        logger.info(f"Application d'un buffer de {buffer_size}")
        gdf = gdf.copy()
        gdf.geometry = gdf.geometry.buffer(buffer_size)
    
    return vector_to_mask(gdf, out_shape, transform, **kwargs)


def multiclass_rasterize(
    gdf: gpd.GeoDataFrame,
    out_shape: Tuple[int, int],
    transform: Affine,
    class_column: str = 'class',
    priority: Optional[Dict[str, int]] = None
) -> np.ndarray:
    """
    Rasterisation multi-classes avec gestion des priorités.
    
    Args:
        gdf: GeoDataFrame
        out_shape: (H, W)
        transform: Transform
        class_column: Nom de la colonne de classe
        priority: Dict définissant la priorité de chaque classe (plus élevé = prioritaire)
    
    Returns:
        Masque (H,W) uint8
    """
    mask = np.zeros(out_shape, dtype='uint8')
    
    # Grouper par classe
    if class_column not in gdf.columns:
        raise ValueError(f"Colonne '{class_column}' introuvable")
    
    classes = gdf[class_column].unique()
    
    # Trier par priorité si fournie
    if priority:
        classes = sorted(
            classes,
            key=lambda c: priority.get(str(c).lower(), 0)
        )
    
    # Rasteriser chaque classe
    for cls in classes:
        subset = gdf[gdf[class_column] == cls]
        shapes = [(geom, 1) for geom in subset.geometry if geom is not None]
        
        if shapes:
            temp_mask = features.rasterize(
                shapes=shapes,
                out_shape=out_shape,
                transform=transform,
                fill=0,
                dtype='uint8'
            )
            
            # Attribuer l'ID de classe (hash déterministe: zlib.crc32, stable
            # entre processus, contrairement à hash() qui est aléatoire par
            # défaut d'un run Python à l'autre - voir PYTHONHASHSEED).
            class_id = zlib.crc32(str(cls).encode('utf-8')) % 255 + 1
            mask[temp_mask > 0] = class_id
    
    logger.info(f"Rasterisation multi-classes: {len(classes)} classes")
    
    return mask