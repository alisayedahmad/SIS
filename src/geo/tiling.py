from __future__ import annotations
from typing import List, Tuple, Dict, Any, Iterable
import json
import numpy as np

def _gen_slices(H:int, W:int, size:int, overlap:int) -> Iterable[Tuple[int,int,int,int]]:
    step = size - overlap
    ys = list(range(0, max(H - size + 1, 1), step))
    xs = list(range(0, max(W - size + 1, 1), step))
    if ys[-1] != H - size: ys.append(max(H - size, 0))
    if xs[-1] != W - size: xs.append(max(W - size, 0))
    for y in ys:
        for x in xs:
            yield (y, x, size, size)

def tile_array(arr: np.ndarray, size:int=512, overlap:int=64):
    """Découpe un tableau (C,H,W) ou (H,W) en tuiles (size,size).
    Retourne liste [(coords, tile_array)].
    """
    if arr.ndim == 2:
        arr = arr[None, ...]
    _, H, W = arr.shape
    tiles = []
    for (y,x,h,w) in _gen_slices(H,W,size,overlap):
        tiles.append(((y,x,h,w), arr[:, y:y+h, x:x+w]))
    return tiles

def blend_mosaic(prob: np.ndarray, weight: np.ndarray, patch: np.ndarray, coords):
    y,x,h,w = coords
    prob[:, y:y+h, x:x+w] += patch
    weight[:, y:y+h, x:x+w] += 1.0

def save_tiles_manifest(path, items):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)
