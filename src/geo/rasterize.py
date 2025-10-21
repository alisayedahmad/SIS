from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
import geopandas as gpd
from rasterio import features

def vector_to_mask(gdf: gpd.GeoDataFrame, out_shape: Tuple[int,int], transform, class_map: Dict[str,int], property_key: str) -> np.ndarray:
    """Rasterise des géométries en masque 2D. Multi-classes gérées via `class_map`.

    - gdf: GeoDataFrame avec une colonne attributaire (e.g. 'type').

    - class_map: mapping valeur_attribut -> entier de classe (>=1).

    - property_key: nom de la colonne attributaire.

    Retour: (H,W) uint8.

    """
    shapes = []
    for _, row in gdf.iterrows():
        label = class_map.get(str(row.get(property_key, "")).lower(), 1)
        geom = row.geometry
        if geom is None or geom.is_empty: 
            continue
        shapes.append((geom, label))
    mask = features.rasterize(shapes=shapes, out_shape=out_shape, transform=transform, fill=0, dtype="uint8")
    return mask
