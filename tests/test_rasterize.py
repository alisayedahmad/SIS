import numpy as np
import rasterio
from shapely.geometry import Polygon
from src.geo.rasterize import vector_to_mask
def test_rasterize_simple():
    poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    # On bypasse gpd.GeoDataFrame qui crashe sur Windows (bug pandas/pyarrow)
    class MockRow:
        def __init__(self, geom, type_val):
            self.geometry = geom
            self._data = {"type": type_val}

        def __contains__(self, key):
            return key in self._data

        def __getitem__(self, key):
            return self._data[key]

    class MockGDF:
        def __init__(self, rows):
            self._rows = rows

        def __len__(self):
            return len(self._rows)

        def iterrows(self):
            for i, row in enumerate(self._rows):
                yield i, row

    mock_gdf = MockGDF([MockRow(poly, "building")])
    transform = rasterio.transform.from_origin(0, 10, 1, 1)

    mask = vector_to_mask(
        mock_gdf,
        out_shape=(10, 10),
        transform=transform,
        class_map={"building": 1},
        property_key="type",
    )

    assert mask.sum() > 0