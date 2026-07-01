"""
Script de visualisation des prédictions.

Génère trois images PNG côte à côte :
  1. Image originale RGB
  2. Masque prédit (blanc = bâtiment)
  3. Superposition (image RGB + masque en rouge transparent)

Usage :
    python -m scripts.visualize_prediction \
        --image data/raw/AerialImageDataset/train/images/austin1.tif \
        --pred predictions/austin1_pred.tif \
        --out predictions/austin1_viz.png
"""

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # pas besoin d'interface graphique

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import rasterio

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def read_rgb(path: Path) -> np.ndarray:
    """
    Lit une image GeoTIFF RGB et la ramène en uint8 [0-255].
    Gère le cas où les valeurs dépassent 255 (images 16-bit).
    """
    with rasterio.open(path) as src:
        arr = src.read()[:3]  # on prend les 3 premières bandes (RGB)

    # Passage en float pour la normalisation
    arr = arr.astype(np.float32)

    # Normalisation par percentile pour un rendu visuel correct,
    # surtout si l'image est en 16-bit
    p2, p98 = np.percentile(arr, 2), np.percentile(arr, 98)
    if p98 > p2:
        arr = (arr - p2) / (p98 - p2)
    arr = np.clip(arr, 0, 1)

    # HWC pour matplotlib
    return np.moveaxis(arr, 0, -1)


def read_mask(path: Path) -> np.ndarray:
    """
    Lit un masque de prédiction GeoTIFF (1 bande, valeurs 0/1).
    Retourne un tableau 2D bool.
    """
    with rasterio.open(path) as src:
        mask = src.read(1)
    return mask > 0


def overlay(rgb: np.ndarray, mask: np.ndarray, color=(1.0, 0.2, 0.2), alpha=0.4) -> np.ndarray:
    """
    Superpose le masque binaire sur l'image RGB avec une couleur semi-transparente.
    """
    result = rgb.copy()
    for c, val in enumerate(color):
        channel = result[:, :, c]
        channel[mask] = channel[mask] * (1 - alpha) + val * alpha
        result[:, :, c] = channel
    return np.clip(result, 0, 1)


# ──────────────────────────────────────────────────────────────────────────────
# Script principal
# ──────────────────────────────────────────────────────────────────────────────

def visualize(
    image_path: Path,
    pred_path: Path,
    out_path: Path,
    crop_size: int = 1024,
) -> None:
    """
    Génère une figure avec 3 panneaux : original | masque | superposition.

    Pour éviter des PNG de 5000×5000 trop lourds, on coupe une zone centrale
    de `crop_size` pixels. Modifiable via --crop-size.
    """
    logger.info(f"Lecture image : {image_path}")
    rgb = read_rgb(image_path)

    logger.info(f"Lecture masque : {pred_path}")
    mask = read_mask(pred_path)

    h, w = rgb.shape[:2]
    logger.info(f"Dimensions : {h}x{w}")

    # Découpe d'une zone centrale pour ne pas générer une image trop lourde
    if crop_size and (h > crop_size or w > crop_size):
        cy, cx = h // 2, w // 2
        half = crop_size // 2
        y0, y1 = max(0, cy - half), min(h, cy + half)
        x0, x1 = max(0, cx - half), min(w, cx + half)
        rgb = rgb[y0:y1, x0:x1]
        mask = mask[y0:y1, x0:x1]
        logger.info(f"Zone crop : [{y0}:{y1}, {x0}:{x1}]")

    # Statistiques de couverture
    coverage = mask.mean() * 100
    n_buildings = _count_objects(mask)
    logger.info(f"Couverture batiments : {coverage:.1f}%")
    logger.info(f"Objets détectés (approximation) : {n_buildings}")

    # Superposition
    overlay_img = overlay(rgb, mask)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=150)
    fig.suptitle(
        f"Prédiction bâtiments — {image_path.name}\n"
        f"Couverture : {coverage:.1f}%  |  Objets : {n_buildings}",
        fontsize=13,
        fontweight="bold",
    )

    # Panneau 1 : image originale
    axes[0].imshow(rgb)
    axes[0].set_title("Image originale", fontsize=11)
    axes[0].axis("off")

    # Panneau 2 : masque prédit
    axes[1].imshow(mask, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Masque prédit", fontsize=11)
    axes[1].axis("off")

    # Panneau 3 : superposition
    axes[2].imshow(overlay_img)
    patch = mpatches.Patch(color=(1.0, 0.2, 0.2), alpha=0.7, label="Bâtiments détectés")
    axes[2].legend(handles=[patch], loc="lower right", fontsize=9)
    axes[2].set_title("Superposition", fontsize=11)
    axes[2].axis("off")

    plt.tight_layout()

    # Sauvegarde
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Visualisation sauvegardée : {out_path}")


def _count_objects(mask: np.ndarray) -> int:
    """
    Compte le nombre approximatif d'objets (composantes connexes)
    sans dépendance à skimage pour garder le script léger.
    """
    try:
        from scipy import ndimage
        labeled, n = ndimage.label(mask)
        return n
    except ImportError:
        return -1  # scipy non disponible, on retourne -1


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        level=logging.INFO,
    )

    parser = argparse.ArgumentParser(
        description="Visualisation des prédictions de segmentation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image", required=True, help="Image GeoTIFF originale")
    parser.add_argument("--pred", required=True, help="Masque de prédiction GeoTIFF")
    parser.add_argument("--out", required=True, help="Fichier PNG de sortie")
    parser.add_argument(
        "--crop-size", type=int, default=1024,
        help="Taille de la zone centrale à visualiser (en pixels). 0 = image entière."
    )
    args = parser.parse_args()

    visualize(
        image_path=Path(args.image),
        pred_path=Path(args.pred),
        out_path=Path(args.out),
        crop_size=args.crop_size,
    )


if __name__ == "__main__":
    main()
