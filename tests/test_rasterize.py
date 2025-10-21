import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon
import rasterio
from src.geo.rasterize import vector_to_mask

def test_rasterize_simple():
    poly = Polygon([(0,0),(10,0),(10,10),(0,10)])
    gdf = gpd.GeoDataFrame({'type':['building']}, geometry=[poly], crs='EPSG:4326')
    transform = rasterio.transform.from_origin(0,10,1,1)
    mask = vector_to_mask(gdf, out_shape=(10,10), transform=transform, class_map={'building':1}, property_key='type')
    assert mask.sum() > 0
