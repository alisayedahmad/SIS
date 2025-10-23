from __future__ import annotations
from typing import List, Dict, Any, Optional
import numpy as np
import rasterio
from rasterio import features, Affine
import geopandas as gpd
from shapely.geometry import shape, mapping, Polygon
from shapely.ops import unary_union
import logging

logger = logging.getLogger(__name__)


def mask_to_polygons(
    mask: np.ndarray,
    transform: Affine,
    crs: Any,
    min_area: float = 10.0,
    simplify_tolerance: float = 1.0,
    merge_adjacent: bool = False
) -> gpd.GeoDataFrame:
    """
    Vectorise un masque en polygones avec options avancées.
    
    Args:
        mask: Masque binaire (H,W)
        transform: Transform affine
        crs: Système de coordonnées
        min_area: Aire minimale des polygones (unités CRS)
        simplify_tolerance: Tolérance de simplification (0 = pas de simplification)
        merge_adjacent: Fusionner les polygones adjacents
    
    Returns:
        GeoDataFrame avec polygones
    """
    if mask.ndim == 3:
        mask = mask[0]
    
    # Extraction des polygones
    polygons = []
    
    for geom, val in features.shapes(
        mask.astype('uint8'),
        transform=transform,
        connectivity=8  # 8-connectivité
    ):
        if int(val) == 0:
            continue
        
        poly = shape(geom)
        
        # Filtrer par aire
        if poly.area < min_area:
            continue
        
        # Simplification
        if simplify_tolerance > 0:
            poly = poly.simplify(simplify_tolerance, preserve_topology=True)
        
        # Validation
        if poly.is_valid and not poly.is_empty:
            polygons.append(poly)
    
    logger.info(f"Vectorisation: {len(polygons)} polygones extraits")
    
    # Fusion des polygones adjacents
    if merge_adjacent and len(polygons) > 0:
        logger.info("Fusion des polygones adjacents...")
        merged = unary_union(polygons)
        
        if merged.geom_type == 'Polygon':
            polygons = [merged]
        elif merged.geom_type == 'MultiPolygon':
            polygons = list(merged.geoms)
        
        logger.info(f"Après fusion: {len(polygons)} polygones")
    
    # Création du GeoDataFrame
    if len(polygons) == 0:
        logger.warning("Aucun polygone valide trouvé")
        return gpd.GeoDataFrame(columns=['geometry'], crs=crs)
    
    gdf = gpd.GeoDataFrame(geometry=polygons, crs=crs)
    
    # Attributs géométriques
    gdf['area'] = gdf.geometry.area
    gdf['perimeter'] = gdf.geometry.length
    gdf['compactness'] = (4 * np.pi * gdf['area']) / (gdf['perimeter'] ** 2)
    
    logger.info(f"GeoDataFrame créé: {len(gdf)} features")
    
    return gdf


def mask_to_linestrings(
    mask: np.ndarray,
    transform: Affine,
    crs: Any,
    min_length: float = 5.0
) -> gpd.GeoDataFrame:
    """
    Vectorise un masque de routes en LineStrings.
    
    Args:
        mask: Masque binaire (H,W)
        transform: Transform affine
        crs: CRS
        min_length: Longueur minimale des segments
    
    Returns:
        GeoDataFrame avec LineStrings
    """
    from skimage.morphology import skeletonize
    
    # Squelettisation
    skeleton = skeletonize(mask.astype(bool))
    
    # Extraction des lignes
    lines = []
    
    for geom, val in features.shapes(
        skeleton.astype('uint8'),
        transform=transform
    ):
        if int(val) == 0:
            continue
        
        poly = shape(geom)
        
        # Convertir en LineString
        if poly.geom_type == 'Polygon':
            line = poly.exterior
        else:
            continue
        
        # Filtrer par longueur
        if line.length < min_length:
            continue
        
        lines.append(line)
    
    logger.info(f"Vectorisation routes: {len(lines)} LineStrings")
    
    if len(lines) == 0:
        return gpd.GeoDataFrame(columns=['geometry'], crs=crs)
    
    gdf = gpd.GeoDataFrame(geometry=lines, crs=crs)
    gdf['length'] = gdf.geometry.length
    
    return gdf


def postprocess_polygons(
    gdf: gpd.GeoDataFrame,
    buffer_distance: float = 0.0,
    remove_holes: bool = True,
    hole_threshold: float = 100.0
) -> gpd.GeoDataFrame:
    """
    Post-traite les polygones vectorisés.
    
    Args:
        gdf: GeoDataFrame
        buffer_distance: Distance de buffer (positif = dilatation, négatif = érosion)
        remove_holes: Supprimer les trous
        hole_threshold: Aire minimale des trous à conserver
    
    Returns:
        GeoDataFrame post-traité
    """
    gdf = gdf.copy()
    
    # Buffer
    if buffer_distance != 0:
        logger.info(f"Application d'un buffer: {buffer_distance}")
        gdf.geometry = gdf.geometry.buffer(buffer_distance)
    
    # Suppression des trous
    if remove_holes:
        logger.info(f"Suppression des trous < {hole_threshold}")
        
        def remove_small_holes(geom):
            if geom.geom_type == 'Polygon':
                if geom.interiors:
                    # Garder seulement les grands trous
                    new_interiors = [
                        interior for interior in geom.interiors
                        if Polygon(interior).area >= hole_threshold
                    ]
                    return Polygon(geom.exterior.coords, new_interiors)
            return geom
        
        gdf.geometry = gdf.geometry.apply(remove_small_holes)
    
    # Re-calculer les attributs
    gdf['area'] = gdf.geometry.area
    gdf['perimeter'] = gdf.geometry.length
    
    return gdf