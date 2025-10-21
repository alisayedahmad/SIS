from __future__ import annotations
import numpy as np
import torch
from .geo.tiling import _gen_slices, blend_mosaic

def slide_infer_on_raster(model, img, tile_size=512, overlap=64, batch_size=4, device="cpu"):
    """Inférence sliding-window + blending sur tableau (C,H,W).
    Retourne probas (1,H,W) pour binaire ou (C,H,W) pour multi-classes.
    """
    model.eval()
    if img.ndim==2: img = img[None, ...]
    C,H,W = img.shape
    prob = None
    weight = None
    # buffer patches
    patches, coords_list = [], []
    for coords in _gen_slices(H,W,tile_size,overlap):
        y,x,h,w = coords
        patch = img[:3, y:y+h, x:x+w]
        patches.append(patch)
        coords_list.append(coords)
        if len(patches)==batch_size:
            p = _infer_batch(model, patches, device)
            prob, weight = _accumulate(prob, weight, p, coords_list, H, W)
            patches, coords_list = [], []
    if patches:
        p = _infer_batch(model, patches, device)
        prob, weight = _accumulate(prob, weight, p, coords_list, H, W)
    prob = prob / np.maximum(weight, 1e-6)
    return prob

def _infer_batch(model, patches, device):
    x = np.stack([np.moveaxis(p, 0, -1) for p in patches])
    x = (x - x.mean(axis=(1,2,3), keepdims=True)) / (x.std(axis=(1,2,3), keepdims=True)+1e-6)
    x = torch.from_numpy(x).permute(0,3,1,2).float().to(device)
    with torch.inference_mode():
        logits = model(x)
        if logits.shape[1]==1:
            prob = torch.sigmoid(logits).cpu().numpy().transpose(0,1,2,3)
        else:
            prob = torch.softmax(logits, dim=1).cpu().numpy().transpose(0,1,2,3)
    return prob

def _accumulate(prob, weight, p, coords_list, H, W):
    C = p.shape[1]
    if prob is None:
        prob = np.zeros((C,H,W), dtype=np.float32)
        weight = np.zeros((C,H,W), dtype=np.float32)
    for i, coords in enumerate(coords_list):
        blend_mosaic(prob, weight, p[i], coords)
    return prob, weight
