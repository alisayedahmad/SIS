from __future__ import annotations
from typing import List, Tuple, Dict, Any, Iterable, Optional ,Union
import json
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def _gen_slices(
    H: int,
    W: int,
    size: int,
    overlap: int,
    ensure_coverage: bool = True
) -> Iterable[Tuple[int, int, int, int]]:
    """
    Génère les coordonnées de tuiles avec couverture garantie.
    
    Args:
        H, W: Dimensions de l'image
        size: Taille de tuile
        overlap: Chevauchement
        ensure_coverage: Garantir que toute l'image est couverte
    
    Yields:
        (y, x, h, w) - Coordonnées de tuile
    """
    step = size - overlap
    
    if step <= 0:
        raise ValueError(f"Overlap ({overlap}) doit être < size ({size})")
    
    # Génération des positions Y
    ys = list(range(0, max(H - size + 1, 1), step))
    # Génération des positions X
    xs = list(range(0, max(W - size + 1, 1), step))
    
    # Assurer la couverture complète
    if ensure_coverage:
        if ys[-1] + size < H:
            ys.append(H - size)
        if xs[-1] + size < W:
            xs.append(W - size)
    
    # Enlever les doublons
    ys = sorted(list(set([max(0, y) for y in ys])))
    xs = sorted(list(set([max(0, x) for x in xs])))
    
    logger.debug(f"Grille de tuilage: {len(ys)}×{len(xs)} = {len(ys)*len(xs)} tuiles")
    
    for y in ys:
        for x in xs:
            # La taille effective peut être inférieure à size sur les bords
            h = min(size, H - y)
            w = min(size, W - x)
            yield (y, x, h, w)

def tile_array(
    arr: np.ndarray,
    size: int = 512,
    overlap: int = 64,
    min_coverage: float = 0.0,
    pad_mode: str = 'reflect'
) -> List[Tuple[Tuple[int, int, int, int], np.ndarray]]:
    """
    Découpe un tableau en tuiles avec options avancées.

    Args:
        arr: Tableau (C, H, W) ou (H, W)
        size: Taille de la tuile (en pixels)
        overlap: Chevauchement entre tuiles (en pixels)
        min_coverage: Couverture minimale (0-1, pixels non-nuls) pour inclure une
            tuile. Par défaut 0.0 (aucun filtrage) : la grille de tuiles est
            entièrement déterministe et ne dépend pas du contenu du tableau.
            ATTENTION: si vous tuilez une image et son masque séparément, ne
            passez PAS un min_coverage > 0 sur un seul des deux tableaux, sinon
            les listes de tuiles résultantes ne seront plus alignées coordonnée
            par coordonnée (et peuvent même différer en longueur). Si vous
            voulez filtrer les tuiles "vides", tuilez d'abord avec
            min_coverage=0.0 puis filtrez les paires (image, masque) ensemble
            en utilisant la couverture du masque.
        pad_mode: Mode de padding ('reflect', 'edge', 'constant')

    Returns:
        Liste de tuples (coords, tile_array)
        - coords: (y, x, h, w)
        - tile_array: np.ndarray (C, size, size)
    """
    # Normalisation des dimensions
    if arr.ndim == 2:
        arr = arr[None, ...]  # -> (1, H, W)
    elif arr.ndim != 3:
        raise ValueError(f"Le tableau doit être 2D ou 3D, obtenu: {arr.shape}")

    C, H, W = arr.shape
    logger.info(f"Tuilage: {arr.shape} -> tuiles {size}×{size} (overlap={overlap})")

    tiles = []

    # Itération sur les tuiles générées par _gen_slices
    for coords in _gen_slices(H, W, size, overlap):
        y, x, h, w = coords
        tile = arr[:, y:y+h, x:x+w]

        # Vérifier si la couverture respecte le seuil requis
        coverage = np.count_nonzero(tile) / tile.size
        if coverage < min_coverage:
            logger.debug(f"Tuile ignorée (couverture={coverage:.2%})")
            continue

        # Appliquer du padding si la tuile est partielle
        if h < size or w < size:
            pad_h = size - h
            pad_w = size - w
            tile = np.pad(
                tile,
                ((0, 0), (0, pad_h), (0, pad_w)),
                mode=pad_mode
            )

        tiles.append((coords, tile))

    logger.info(f"Nombre de tuiles générées: {len(tiles)}")
    return tiles


def _triangular_ramp(n: int) -> np.ndarray:
    """Fenêtre triangulaire 1D (0 exclu) qui culmine au centre

    Utilisée pour pondérer davantage le centre d'une tuile que ses bords
    lors du blending, ce qui atténue les artefacts de bloc aux jonctions
    de tuiles qui se chevauchent
    """
    if n <= 1:
        return np.ones(max(n, 1), dtype=np.float32)
    ramp = np.minimum(np.arange(1, n + 1), np.arange(n, 0, -1)).astype(np.float32)
    return ramp / ramp.max()


def blend_mosaic(
    prob: np.ndarray,
    weight: np.ndarray,
    patch: np.ndarray,
    coords: Tuple[int, int, int, int],
    blend_mode: str = 'linear'
) -> None:
    """
    Accumule les prédictions avec blending
    
    Args:
        prob: Tableau de probabilités accumulées (C,H,W)
        weight: Tableau de poids (C,H,W)
        patch: Patch à ajouter (C,h,w)
        coords: (y, x, h, w)
        blend_mode: Mode de blending ('linear', 'gaussian')
    """
    y, x, h, w = coords
    
    # Créer un masque de blending
    if blend_mode == 'linear':
        # Fenêtre triangulaire 2D : poids maximal au centre de la tuile,
        # minimal (mais non nul) sur les bords
        ramp_y = _triangular_ramp(h)
        ramp_x = _triangular_ramp(w)
        mask = np.outer(ramp_y, ramp_x).astype(np.float32)
    elif blend_mode == 'gaussian':
        # Distribution gaussienne centrée
        from scipy.ndimage import gaussian_filter
        mask = np.zeros((h, w), dtype=np.float32)
        mask[h // 2, w // 2] = 1.0
        mask = gaussian_filter(mask, sigma=max(h, w) / 6)
        mask = mask / mask.max()
    else:
        # Pas de blending particulier
        mask = np.ones((h, w), dtype=np.float32)
    
    # Adapter le masque à tous les canaux
    mask = mask[None, :, :]  # (1,h,w)
    
    # Accumuler les probabilités et poids
    prob[:, y:y+h, x:x+w] += patch[:, :h, :w] * mask
    weight[:, y:y+h, x:x+w] += mask


def save_tiles_manifest(
    path: Union[str, Path],
    items: List[Dict[str, Any]],
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Sauvegarde un manifest de tuiles avec métadonnées
    
    Args:
        path: Chemin du fichier JSON
        items: Liste des items (dict avec 'image', 'mask', etc.)
        metadata: Métadonnées additionnelles
    """
    manifest = {
        'version': '1.0',
        'num_tiles': len(items),
        'items': items
    }
    
    if metadata:
        manifest['metadata'] = metadata
    
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Manifest sauvegardé: {path} ({len(items)} tuiles)")


def load_tiles_manifest(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Charge un manifest de tuiles."""
    with open(path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)
    
    items = manifest.get('items', [])
    logger.info(f"Manifest chargé: {len(items)} tuiles")
    
    return items