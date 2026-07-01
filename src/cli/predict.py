import argparse
import os
from pathlib import Path
import json
import logging
from typing import Optional

import numpy as np
import rasterio
import torch
from tqdm import tqdm

from ..infer import slide_infer_on_raster, predict_large_image
from ..train_module import SegLightningModule
from ..postprocess import postprocess_prediction
from ..geo.io import read_geotiff, write_geotiff_like
from ..geo.vectorize import mask_to_polygons
from ..utils import setup_logging

logger = logging.getLogger(__name__)


def validate_inputs(checkpoint: str, image: str, out: str) -> None:
    """Valide les chemins d'entrée/sortie."""
    if not Path(checkpoint).exists():
        raise FileNotFoundError(f"Checkpoint introuvable: {checkpoint}")
    
    if not Path(image).exists():
        raise FileNotFoundError(f"Image introuvable: {image}")
    
    # Créer le dossier de sortie si nécessaire
    out_dir = Path(out).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("✓ Validation des entrées réussie")


def load_model(checkpoint: str, device: str) -> SegLightningModule:
    """Charge le modèle depuis un checkpoint."""
    logger.info(f"Chargement du modèle depuis: {checkpoint}")
    
    try:
        model = SegLightningModule.load_from_checkpoint(
            checkpoint,
            map_location=device
        )
        model.eval()
        model.to(device)
        
        logger.info(f"✓ Modèle chargé sur {device}")
        logger.info(f"  - Architecture: {model.hparams.model_name}")
        logger.info(f"  - Classes: {model.hparams.num_classes}")
        
        return model
    
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle: {e}")
        raise


def main(
    checkpoint: str,
    image: str,
    out: str,
    geojson: Optional[str] = None,
    tile_size: int = 512,
    overlap: int = 64,
    batch_size: int = 4,
    threshold: float = 0.5,
    task: str = "buildings",
    use_tta: bool = False,
    postprocess: bool = True,
    device: Optional[str] = None
):
    """
    Pipeline complet de prédiction.
    
    Args:
        checkpoint: Chemin vers le checkpoint Lightning
        image: Chemin vers l'image GeoTIFF d'entrée
        out: Chemin de sortie pour le GeoTIFF prédit
        geojson: Chemin optionnel pour export GeoJSON vectorisé
        tile_size: Taille des tuiles pour l'inférence
        overlap: Chevauchement entre tuiles
        batch_size: Taille du batch
        threshold: Seuil de binarisation (0-1)
        task: Type de tâche ('buildings', 'roads', 'multi')
        use_tta: Activer Test-Time Augmentation
        postprocess: Appliquer les post-traitements
        device: Device ('cpu', 'cuda', ou None pour auto)
    """
    # Setup logging
    setup_logging(log_dir="logs", level=logging.INFO)
    
    logger.info("=" * 80)
    logger.info("DÉBUT DE LA PRÉDICTION")
    logger.info("=" * 80)
    
    # Validation
    validate_inputs(checkpoint, image, out)
    
    # Device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device sélectionné: {device}")
    
    # Chargement de l'image
    logger.info(f"Chargement de l'image: {image}")
    arr, meta = read_geotiff(image)
    logger.info(f"  - Dimensions: {arr.shape}")
    logger.info(f"  - CRS: {meta.get('crs', 'N/A')}")
    logger.info(f"  - Transform: {meta.get('transform', 'N/A')}")
    
    # Chargement du modèle
    model = load_model(checkpoint, device)
    
    # Paramètres d'inférence
    logger.info("Paramètres d'inférence:")
    logger.info(f"  - Taille tuile: {tile_size}")
    logger.info(f"  - Overlap: {overlap}")
    logger.info(f"  - Batch size: {batch_size}")
    logger.info(f"  - TTA: {use_tta}")
    logger.info(f"  - Threshold: {threshold}")
    logger.info(f"  - Task: {task}")
    
    # Inférence
    logger.info("=" * 80)
    logger.info("INFÉRENCE EN COURS...")
    logger.info("=" * 80)
    
    with torch.inference_mode():
        prob = slide_infer_on_raster(
            model=model,
            img=arr,
            tile_size=tile_size,
            overlap=overlap,
            batch_size=batch_size,
            device=device,
            use_tta=use_tta,
            normalize=True,
            progress_bar=True
        )
    
    logger.info("✓ Inférence terminée")
    
    # Binarisation
    if prob.shape[0] == 1:
        pred = (prob[0] >= threshold).astype(np.uint8)
        logger.info(f"Segmentation binaire - Pixels positifs: {pred.sum()} ({pred.sum()/pred.size*100:.2f}%)")
    else:
        pred = prob.argmax(axis=0).astype(np.uint8)
        logger.info(f"Segmentation multi-classes")
        for c in range(prob.shape[0]):
            count = (pred == c).sum()
            logger.info(f"  - Classe {c}: {count} pixels ({count/pred.size*100:.2f}%)")
    
    # Post-traitement
    if postprocess and task in ["buildings", "roads"]:
        logger.info(f"Application du post-traitement ({task})...")
        pred = postprocess_prediction(pred[None, ...], kind=task)[0]
        logger.info("✓ Post-traitement terminé")
    
    # Export GeoTIFF
    logger.info(f"Sauvegarde du GeoTIFF: {out}")
    meta_out = meta.copy()
    meta_out.update(count=1, dtype='uint8')
    write_geotiff_like(out, meta_out, pred[None, ...])
    logger.info("✓ GeoTIFF sauvegardé")
    
    # Export GeoJSON (optionnel)
    if geojson:
        logger.info(f"Vectorisation et export GeoJSON: {geojson}")
        try:
            import json
            from rasterio import features as rio_features
            from shapely.geometry import shape, mapping

            # On bypasse geopandas/pandas entièrement pour éviter le crash
            # pyarrow sur Windows. On utilise rasterio.features.shapes()
            # directement et on écrit le GeoJSON avec le module json standard.
            polygons = []
            for geom, val in rio_features.shapes(
                pred.astype("uint8"),
                transform=meta["transform"],
                connectivity=8,
            ):
                if int(val) == 0:
                    continue
                poly = shape(geom)
                if poly.area < 10 or not poly.is_valid or poly.is_empty:
                    continue
                polygons.append({
                    "type": "Feature",
                    "geometry": mapping(poly),
                    "properties": {
                        "area": float(poly.area),
                        "perimeter": float(poly.length),
                    },
                })

            geojson_dict = {
                "type": "FeatureCollection",
                "features": polygons,
            }

            Path(geojson).parent.mkdir(parents=True, exist_ok=True)
            with open(geojson, "w", encoding="utf-8") as fout:
                json.dump(geojson_dict, fout)

            logger.info(f"  - {len(polygons)} polygones sauvegardés")
            logger.info("GeoJSON sauvegarde OK")

        except Exception as e:
            logger.error(f"Erreur lors de la vectorisation: {e}")
    # Statistiques finales
    logger.info("=" * 80)
    logger.info("RÉSUMÉ")
    logger.info("=" * 80)
    logger.info(f"Image d'entrée: {image}")
    logger.info(f"Checkpoint: {checkpoint}")
    logger.info(f"Sortie GeoTIFF: {out}")
    if geojson:
        logger.info(f"Sortie GeoJSON: {geojson}")
    logger.info(f"Couverture prédite: {pred.sum()/pred.size*100:.2f}%")
    logger.info("=" * 80)
    logger.info("PRÉDICTION TERMINÉE AVEC SUCCÈS")
    logger.info("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prédiction sur grande image avec modèle de segmentation"
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Chemin vers le checkpoint Lightning (.ckpt)"
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Chemin vers l'image GeoTIFF d'entrée"
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Chemin de sortie pour le GeoTIFF prédit"
    )
    parser.add_argument(
        "--geojson",
        type=str,
        default=None,
        help="Chemin optionnel pour export GeoJSON vectorisé"
    )
    parser.add_argument(
        "--tile",
        type=int,
        default=512,
        help="Taille des tuiles pour l'inférence"
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=64,
        help="Chevauchement entre tuiles"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Taille du batch"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Seuil de binarisation (0-1)"
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["buildings", "roads", "multi"],
        default="buildings",
        help="Type de tâche pour post-traitement"
    )
    parser.add_argument(
        "--use-tta",
        action="store_true",
        help="Activer Test-Time Augmentation"
    )
    parser.add_argument(
        "--no-postprocess",
        action="store_true",
        help="Désactiver le post-traitement"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda", "mps"],
        help="Device (None pour auto-détection)"
    )
    
    args = parser.parse_args()
    
    try:
        main(
            checkpoint=args.checkpoint,
            image=args.image,
            out=args.out,
            geojson=args.geojson,
            tile_size=args.tile,
            overlap=args.overlap,
            batch_size=args.batch_size,
            threshold=args.threshold,
            task=args.task,
            use_tta=args.use_tta,
            postprocess=not args.no_postprocess,
            device=args.device
        )
    except Exception as e:
        logger.exception("Erreur fatale durant la prédiction")
        raise e