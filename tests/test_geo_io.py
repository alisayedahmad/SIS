import numpy as np
import rasterio
from src.geo.io import write_geotiff_like, read_geotiff

def test_geo_io(tmp_path):
    H,W = 64,64
    arr = (np.random.rand(3,H,W)*255).astype('uint8')
    meta = {'driver':'GTiff','dtype':'uint8','count':3,'height':H,'width':W,'crs':'EPSG:4326','transform':rasterio.transform.from_origin(0,0,1,1)}
    f = tmp_path / "a.tif"
    write_geotiff_like(str(f), meta, arr)
    arr2, meta2 = read_geotiff(str(f))
    assert arr2.shape == arr.shape
    assert str(meta2['crs']) == 'EPSG:4326'
