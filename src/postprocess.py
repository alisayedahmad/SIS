from __future__ import annotations
import numpy as np
from skimage.morphology import closing, square, skeletonize

def postprocess_prediction(mask: np.ndarray, kind:str):
    """Post-traitements simples: fermeture (bâtiments), squelette (routes)."""
    if mask.ndim==3: mask = mask[0]
    if kind=="buildings":
        return closing(mask.astype(bool), square(3)).astype(np.uint8)[None, ...]
    if kind=="roads":
        sk = skeletonize(mask.astype(bool)).astype(np.uint8)
        return sk[None, ...]
    return mask[None, ...]
