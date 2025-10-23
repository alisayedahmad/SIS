from __future__ import annotations
import numpy as np
from skimage.morphology import (
    closing, opening, square, disk,
    remove_small_objects, remove_small_holes,
    skeletonize, binary_dilation
)
from scipy import ndimage
import logging

logger = logging.getLogger(__name__)


def postprocess_prediction(
    mask: np.ndarray,
    kind: str,
    min_size: int = 50,
    closing_size: int = 3,
    opening_size: int = 2
) -> np.ndarray:
    """
    Post-traitement avancé des prédictions.
    
    Args:
        mask: Masque binaire (1,H,W) ou (H,W)
        kind: Type ('buildings', 'roads', 'multi')
        min_size: Taille minimale des objets à conserver (pixels)
        closing_size: Taille de l'élément structurant pour closing
        opening_size: Taille de l'élément structurant pour opening
    
    Returns:
        Masque post-traité (1,H,W)
    """
    if mask.ndim == 3:
        mask = mask[0]
    
    mask_bool = mask.astype(bool)
    
    logger.info(f"Post-traitement ({kind}): pixels avant = {mask_bool.sum()}")
    
    if kind == "buildings":
        # Fermeture pour combler les trous
        selem_close = square(closing_size)
        mask_bool = closing(mask_bool, selem_close)
        
        # Ouverture pour lisser les contours
        selem_open = square(opening_size)
        mask_bool = opening(mask_bool, selem_open)
        
        # Suppression des petits objets
        mask_bool = remove_small_objects(mask_bool, min_size=min_size)
        
        # Remplissage des petits trous
        mask_bool = remove_small_holes(mask_bool, area_threshold=min_size // 2)
        
        logger.info(f"Bâtiments: {np.unique(ndimage.label(mask_bool)[1][-1])} objets détectés")
    
    elif kind == "roads":
        # Squelettisation pour extraire le réseau routier
        mask_skel = skeletonize(mask_bool)
        
        # Dilatation légère pour élargir les routes
        selem_dilate = disk(1)
        mask_bool = binary_dilation(mask_skel, selem_dilate)
        
        # Suppression des petits fragments
        mask_bool = remove_small_objects(mask_bool, min_size=min_size // 2)
        
        logger.info(f"Routes: squelettisation appliquée")
    
    elif kind == "multi":
        # Pour multi-classes, appliquer un nettoyage léger
        mask_bool = remove_small_objects(mask_bool, min_size=min_size // 2)
    
    else:
        logger.warning(f"Type inconnu: {kind}. Aucun post-traitement appliqué.")
    
    result = mask_bool.astype(np.uint8)[None, ...]
    logger.info(f"Post-traitement: pixels après = {result.sum()}")
    
    return result


def apply_crf(
    image: np.ndarray,
    mask: np.ndarray,
    num_iter: int = 5,
    gt_prob: float = 0.7
) -> np.ndarray:
    """
    Applique CRF (Conditional Random Field) pour affiner les contours.
    
    Note: Nécessite pydensecrf (optionnel)
    """
    try:
        import pydensecrf.densecrf as dcrf
        from pydensecrf.utils import unary_from_softmax
        
        H, W = mask.shape
        
        # Préparer les unary potentials
        probs = np.stack([1 - mask, mask])
        U = unary_from_softmax(probs)
        
        # CRF
        d = dcrf.DenseCRF2D(W, H, 2)
        d.setUnaryEnergy(U)
        
        # Pairwise potentials
        d.addPairwiseGaussian(sxy=3, compat=3)
        d.addPairwiseBilateral(
            sxy=80, srgb=13,
            rgbim=image.transpose(1, 2, 0).astype(np.uint8),
            compat=10
        )
        
        # Inférence
        Q = d.inference(num_iter)
        refined = np.argmax(Q, axis=0).reshape(H, W).astype(np.uint8)
        
        logger.info(f"CRF appliqué ({num_iter} itérations)")
        return refined
        
    except ImportError:
        logger.warning("pydensecrf non installé. CRF ignoré.")
        return mask