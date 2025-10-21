from __future__ import annotations
from typing import List, Dict, Any
import numpy as np
import rasterio
from rasterio import features
import geopandas as gpd
from shapely.geometry import shape, mapping

def mask_to_polygons(mask: np.ndarray, transform, crs) -> gpd.GeoDataFrame:
    """Vectorise un masque (H,W) en polygones.

    Retourne un GeoDataFrame (polygons) avec CRS.
    """
    if mask.ndim == 3:
        mask = mask[0]
    polygons = []
    for geom, val in features.shapes(mask.astype("uint8"), transform=transform):
        if int(val) == 0:
            continue
        polygons.append(shape(geom))
    gdf = gpd.GeoDataFrame(geometry=polygons, crs=crs)
    return gdf
