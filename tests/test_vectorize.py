"""Tests pour le vectorisation des masques de la segmentation"""

import numpy as np
import pytest
import rasterio.transform as rt
from rasterio.crs import CRS
from shapely.geometry import shape
from rasterio import features


# On teste directement la logique de vectorisation sans passer par geopandas
# (crash pyarrow sur Windows). On réutilise rasterio.features.shapes directement

def vectorize_simple(mask, transform):
    """Version minimale de mask_to_polygons sans geopandas juste  pour les tests """
    if mask.ndim == 3:
        mask = mask[0]
    polygons = []
    for geom, val in features.shapes(mask.astype("uint8"), transform=transform, connectivity=8):
        if int(val) == 0:
            continue
        poly = shape(geom)
        if poly.is_valid and not poly.is_empty:
            polygons.append(poly)
    return polygons


def make_transform():
    return rt.from_origin(0, 64, 1, 1)


class TestMaskToPolygons:

    def test_simple_blob(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[10:30, 10:30] = 1
        polys = vectorize_simple(mask, make_transform())
        assert len(polys) >= 1

    def test_empty_mask(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        polys = vectorize_simple(mask, make_transform())
        assert len(polys) == 0

    def test_full_mask(self):
        mask = np.ones((64, 64), dtype=np.uint8)
        polys = vectorize_simple(mask, make_transform())
        assert len(polys) >= 1

    def test_polygons_are_valid(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[5:20, 5:20] = 1
        mask[35:55, 35:55] = 1
        polys = vectorize_simple(mask, make_transform())
        for poly in polys:
            assert poly.is_valid
            assert not poly.is_empty

    def test_polygon_covers_blob(self):
        # Le polygone extrait doit couvrir approx. la zone du blob
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[10:30, 10:30] = 1     # bloc 20x20 = 400 pixels
        polys = vectorize_simple(mask, make_transform())
        assert len(polys) >= 1
        total_area = sum(p.area for p in polys)
        assert 300 < total_area < 500    # tolerance raisonnable

    def test_two_separate_blobs(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[5:15, 5:15] = 1
        mask[45:55, 45:55] = 1
        polys = vectorize_simple(mask, make_transform())
        assert len(polys) >= 2

    def test_3d_mask_input(self):
        # (1, H, W) doit aussi fonctionner
        mask = np.zeros((1, 64, 64), dtype=np.uint8)
        mask[0, 10:30, 10:30] = 1
        polys = vectorize_simple(mask, make_transform())
        assert len(polys) >= 1
