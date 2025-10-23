from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Callable, List
import logging
from tqdm import tqdm

from .geo.tiling import _gen_slices, blend_mosaic

logger = logging.getLogger(__name__)


class TTAWrapper:
    """Test-Time Augmentation pour améliorer les prédictions."""
    
    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.model = model
        self.device = device
        self.model.eval()
    
    def _hflip(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flip(x, dims=[3])
    
    def _vflip(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flip(x, dims=[2])
    
    def _rot90(self, x: torch.Tensor, k: int = 1) -> torch.Tensor:
        return torch.rot90(x, k=k, dims=[2, 3])
    
    def predict_with_tta(self, x: torch.Tensor, merge_mode: str = 'mean') -> torch.Tensor:
        """Prédiction avec TTA (8 augmentations)."""
        predictions = []
        
        # Original
        with torch.no_grad():
            pred = self.model(x)
            predictions.append(torch.sigmoid(pred) if pred.shape[1] == 1 else torch.softmax(pred, dim=1))
        
        # Horizontal flip
        x_hflip = self._hflip(x)
        with torch.no_grad():
            pred = self.model(x_hflip)
            pred = torch.sigmoid(pred) if pred.shape[1] == 1 else torch.softmax(pred, dim=1)
            predictions.append(self._hflip(pred))
        
        # Vertical flip
        x_vflip = self._vflip(x)
        with torch.no_grad():
            pred = self.model(x_vflip)
            pred = torch.sigmoid(pred) if pred.shape[1] == 1 else torch.softmax(pred, dim=1)
            predictions.append(self._vflip(pred))
        
        # Rotation 90°
        x_rot90 = self._rot90(x, k=1)
        with torch.no_grad():
            pred = self.model(x_rot90)
            pred = torch.sigmoid(pred) if pred.shape[1] == 1 else torch.softmax(pred, dim=1)
            predictions.append(self._rot90(pred, k=-1))
        
        # Rotation 180°
        x_rot180 = self._rot90(x, k=2)
        with torch.no_grad():
            pred = self.model(x_rot180)
            pred = torch.sigmoid(pred) if pred.shape[1] == 1 else torch.softmax(pred, dim=1)
            predictions.append(self._rot90(pred, k=-2))
        
        # Rotation 270°
        x_rot270 = self._rot90(x, k=3)
        with torch.no_grad():
            pred = self.model(x_rot270)
            pred = torch.sigmoid(pred) if pred.shape[1] == 1 else torch.softmax(pred, dim=1)
            predictions.append(self._rot90(pred, k=-3))
        
        # Hflip + Vflip
        x_hv = self._vflip(self._hflip(x))
        with torch.no_grad():
            pred = self.model(x_hv)
            pred = torch.sigmoid(pred) if pred.shape[1] == 1 else torch.softmax(pred, dim=1)
            predictions.append(self._hflip(self._vflip(pred)))
        
        # Hflip + Rot90
        x_hr = self._rot90(self._hflip(x), k=1)
        with torch.no_grad():
            pred = self.model(x_hr)
            pred = torch.sigmoid(pred) if pred.shape[1] == 1 else torch.softmax(pred, dim=1)
            predictions.append(self._hflip(self._rot90(pred, k=-1)))
        
        # Merge
        if merge_mode == 'mean':
            return torch.stack(predictions).mean(dim=0)
        elif merge_mode == 'max':
            return torch.stack(predictions).max(dim=0)[0]
        else:
            raise ValueError(f"Mode de fusion inconnu: {merge_mode}")


def slide_infer_on_raster(
    model: nn.Module,
    img: np.ndarray,
    tile_size: int = 512,
    overlap: int = 64,
    batch_size: int = 4,
    device: str = "cpu",
    use_tta: bool = False,
    normalize: bool = True,
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None,
    progress_bar: bool = True
) -> np.ndarray:
    """
    Inférence sliding-window optimisée avec options avancées.
    
    Args:
        model: Modèle PyTorch
        img: Image (C,H,W) numpy array
        tile_size: Taille des tuiles
        overlap: Chevauchement entre tuiles
        batch_size: Taille du batch pour l'inférence
        device: Device ('cpu' ou 'cuda')
        use_tta: Activer Test-Time Augmentation
        normalize: Normaliser les données
        mean: Moyenne pour normalisation (défaut: ImageNet)
        std: Std pour normalisation (défaut: ImageNet)
        progress_bar: Afficher la barre de progression
    
    Returns:
        Probabilités (C,H,W) ou (1,H,W) pour binaire
    """
    model.eval()
    
    if img.ndim == 2:
        img = img[None, ...]
    
    C, H, W = img.shape
    
    # Normalisation
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]
    
    # TTA wrapper
    if use_tta:
        tta_model = TTAWrapper(model, device)
        logger.info("TTA activé - 8 augmentations seront appliquées")
    
    # Buffers pour accumulation
    prob = None
    weight = None
    
    # Génération des coordonnées
    coords_list = list(_gen_slices(H, W, tile_size, overlap))
    total_tiles = len(coords_list)
    
    logger.info(f"Inférence sur {total_tiles} tuiles ({tile_size}x{tile_size}, overlap={overlap})")
    
    # Buffer de patches
    patches = []
    coords_batch = []
    
    iterator = tqdm(coords_list, desc="Inférence") if progress_bar else coords_list
    
    for coords in iterator:
        y, x, h, w = coords
        patch = img[:3, y:y+h, x:x+w]  # RGB seulement
        
        patches.append(patch)
        coords_batch.append(coords)
        
        # Traiter le batch
        if len(patches) == batch_size:
            p = _infer_batch(
                model, patches, device, normalize, mean, std,
                use_tta, tta_model if use_tta else None
            )
            prob, weight = _accumulate(prob, weight, p, coords_batch, H, W)
            patches, coords_batch = [], []
    
    # Dernier batch partiel
    if patches:
        p = _infer_batch(
            model, patches, device, normalize, mean, std,
            use_tta, tta_model if use_tta else None
        )
        prob, weight = _accumulate(prob, weight, p, coords_batch, H, W)
    
    # Moyenne pondérée finale
    prob = prob / np.maximum(weight, 1e-6)
    
    logger.info(f"Inférence terminée - Forme de sortie: {prob.shape}")
    
    return prob


def _infer_batch(
    model: nn.Module,
    patches: List[np.ndarray],
    device: str,
    normalize: bool,
    mean: List[float],
    std: List[float],
    use_tta: bool,
    tta_model: Optional[TTAWrapper]
) -> np.ndarray:
    """Inférence sur un batch de patches."""
    # Stack et permutation
    x = np.stack([np.moveaxis(p, 0, -1) for p in patches])  # (B,H,W,C)
    
    # Normalisation
    if normalize:
        mean_arr = np.array(mean).reshape(1, 1, 1, 3)
        std_arr = np.array(std).reshape(1, 1, 1, 3)
        x = (x - mean_arr) / std_arr
    else:
        # Normalisation min-max simple
        x = (x - x.mean(axis=(1, 2, 3), keepdims=True)) / (
            x.std(axis=(1, 2, 3), keepdims=True) + 1e-6
        )
    
    # Conversion en tensor
    x = torch.from_numpy(x).permute(0, 3, 1, 2).float().to(device)
    
    # Inférence
    with torch.inference_mode():
        if use_tta and tta_model is not None:
            prob = tta_model.predict_with_tta(x).cpu().numpy()
        else:
            logits = model(x)
            if logits.shape[1] == 1:
                prob = torch.sigmoid(logits).cpu().numpy()
            else:
                prob = torch.softmax(logits, dim=1).cpu().numpy()
    
    return prob  # (B,C,H,W)


def _accumulate(
    prob: Optional[np.ndarray],
    weight: Optional[np.ndarray],
    p: np.ndarray,
    coords_list: List[tuple],
    H: int,
    W: int
) -> tuple:
    """Accumule les prédictions avec blending."""
    C = p.shape[1]
    
    if prob is None:
        prob = np.zeros((C, H, W), dtype=np.float32)
        weight = np.zeros((C, H, W), dtype=np.float32)
    
    for i, coords in enumerate(coords_list):
        blend_mosaic(prob, weight, p[i], coords)
    
    return prob, weight


def predict_large_image(
    model: nn.Module,
    img: np.ndarray,
    tile_size: int = 512,
    overlap: int = 64,
    batch_size: int = 4,
    device: str = "cpu",
    use_tta: bool = False,
    threshold: float = 0.5
) -> np.ndarray:
    """
    Pipeline complet de prédiction pour grande image.
    
    Returns:
        Masque binaire (H,W) uint8
    """
    prob = slide_infer_on_raster(
        model, img, tile_size, overlap, batch_size, device, use_tta
    )
    
    # Pour binaire, prend le canal de probabilité
    if prob.shape[0] == 1:
        pred = (prob[0] >= threshold).astype(np.uint8)
    else:
        # Multi-classes: argmax
        pred = prob.argmax(axis=0).astype(np.uint8)
    
    return pred