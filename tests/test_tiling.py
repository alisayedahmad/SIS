import numpy as np
from src.geo.tiling import tile_array

def test_tiling():
    arr = np.zeros((1, 1024, 1024), dtype='uint8')
    tiles = tile_array(arr, size=512, overlap=64)
    # should cover the image fully
    assert len(tiles) > 0
    # last tile must be within bounds
    coords,_ = tiles[-1]
    y,x,h,w = coords
    assert y+h <= 1024 and x+w <= 1024
