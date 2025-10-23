"""
Script pour vérifier les tuiles générées après preprocessing.
"""

import argparse
from pathlib import Path
import json
import logging
import numpy as np
import rasterio
from tqdm import tqdm
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


def verify_tiles(tiles_dir: Path, num_samples: int = 5):
    """
    Vérifie les tuiles et génère un rapport.
    
    Args:
        tiles_dir: Dossier contenant les tuiles
        num_samples: Nombre d'échantillons à visualiser
    """
    logger.info("=" * 80)
    logger.info("VÉRIFICATION DES TUILES")
    logger.info("=" * 80)
    
    # Charger le manifest
    manifest_path = tiles_dir / 'manifest.json'
    
    if not manifest_path.exists():
        logger.error(f"Manifest introuvable: {manifest_path}")
        return
    
    with open(manifest_path, 'r') as f:
        data = json.load(f)
    
    items = data.get('items', [])
    metadata = data.get('metadata', {})
    
    logger.info(f"Manifest chargé: {len(items)} tuiles")
    logger.info(f"Métadonnées: {metadata}")
    
    # Vérifications
    issues = []
    
    logger.info("\n1. Vérification de l'existence des fichiers...")
    missing_images = []
    missing_masks = []
    
    for item in tqdm(items, desc="Vérification fichiers"):
        img_path = tiles_dir / item['image']
        mask_path = tiles_dir / item['mask']
        
        if not img_path.exists():
            missing_images.append(str(img_path))
        if not mask_path.exists():
            missing_masks.append(str(mask_path))
    
    if missing_images:
        issues.append(f"{len(missing_images)} images manquantes")
        logger.error(f"✗ {len(missing_images)} images manquantes!")
        logger.error(f"  Exemples: {missing_images[:3]}")
    else:
        logger.info("✓ Toutes les images existent")
    
    if missing_masks:
        issues.append(f"{len(missing_masks)} masques manquants")
        logger.error(f"✗ {len(missing_masks)} masques manquants!")
    else:
        logger.info("✓ Tous les masques existent")
    
    # Vérifier dimensions et valeurs
    logger.info("\n2. Vérification des dimensions et valeurs...")
    
    dim_issues = []
    value_issues = []
    
    sample_items = np.random.choice(items, min(100, len(items)), replace=False)
    
    for item in tqdm(sample_items, desc="Vérification dimensions"):
        img_path = tiles_dir / item['image']
        mask_path = tiles_dir / item['mask']
        
        try:
            with rasterio.open(img_path) as src:
                img = src.read()
                if img.shape[1] > 512 or img.shape[2] > 512:
                    dim_issues.append(f"{img_path.name}: {img.shape}")
            
            with rasterio.open(mask_path) as src:
                mask = src.read()
                if mask.shape[1] > 512 or mask.shape[2] > 512:
                    dim_issues.append(f"{mask_path.name}: {mask.shape}")
                
                # Vérifier valeurs
                unique_vals = np.unique(mask)
                if not all(v in [0, 1, 2] for v in unique_vals):
                    value_issues.append(f"{mask_path.name}: valeurs {unique_vals}")
        
        except Exception as e:
            issues.append(f"Erreur lecture {img_path.name}: {e}")
    
    if dim_issues:
        logger.warning(f"⚠️  {len(dim_issues)} tuiles avec dimensions inhabituelles")
        logger.warning(f"  Exemples: {dim_issues[:3]}")
    else:
        logger.info("✓ Dimensions correctes")
    
    if value_issues:
        logger.warning(f"⚠️  {len(value_issues)} masques avec valeurs inhabituelles")
        logger.warning(f"  Exemples: {value_issues[:3]}")
    else:
        logger.info("✓ Valeurs de masques correctes")
    
    # Statistiques de couverture
    logger.info("\n3. Statistiques de couverture...")
    
    coverages = []
    for item in items:
        if 'coverage' in item:
            coverages.append(item['coverage'])
    
    if coverages:
        logger.info(f"  Moyenne: {np.mean(coverages):.2f}%")
        logger.info(f"  Médiane: {np.median(coverages):.2f}%")
        logger.info(f"  Std: {np.std(coverages):.2f}%")
        logger.info(f"  Min/Max: {np.min(coverages):.2f}% / {np.max(coverages):.2f}%")
        
        # Tuiles vides
        empty_tiles = sum(1 for c in coverages if c < 0.1)
        logger.info(f"  Tuiles quasi-vides (<0.1%): {empty_tiles}")
    
    # Visualisation d'échantillons
    logger.info(f"\n4. Génération de {num_samples} visualisations...")
    
    output_dir = Path('outputs/verification')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    sample_items = np.random.choice(items, min(num_samples, len(items)), replace=False)
    
    for i, item in enumerate(sample_items):
        img_path = tiles_dir / item['image']
        mask_path = tiles_dir / item['mask']
        
        try:
            with rasterio.open(img_path) as src:
                img = src.read()
            with rasterio.open(mask_path) as src:
                mask = src.read()
            
            # Visualisation
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Image RGB
            rgb = np.moveaxis(img[:3], 0, -1)
            rgb = np.clip((rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-6), 0, 1)
            axes[0].imshow(rgb)
            axes[0].set_title('Image RGB')
            axes[0].axis('off')
            
            # Masque
            axes[1].imshow(mask[0], cmap='gray')
            axes[1].set_title('Masque')
            axes[1].axis('off')
            
            # Overlay
            overlay = rgb.copy()
            overlay[mask[0] > 0] = [1, 0, 0]  # Rouge pour les bâtiments
            axes[2].imshow(overlay)
            axes[2].set_title('Overlay')
            axes[2].axis('off')
            
            plt.suptitle(f"Sample {i+1}: {item['source_image']} - Coverage: {item.get('coverage', 0):.2f}%")
            plt.tight_layout()
            
            output_path = output_dir / f'sample_{i+1}.png'
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"  Sauvegardé: {output_path}")
        
        except Exception as e:
            logger.error(f"  Erreur visualisation {img_path.name}: {e}")
    
    # Résumé final
    logger.info("\n" + "=" * 80)
    logger.info("RÉSUMÉ")
    logger.info("=" * 80)
    
    if issues:
        logger.error(f"❌ {len(issues)} problèmes détectés:")
        for issue in issues:
            logger.error(f"  - {issue}")
    else:
        logger.info("✅ Aucun problème détecté!")
    
    logger.info(f"\nVisualisations sauvegardées dans: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Vérifier les tuiles générées"
    )
    parser.add_argument(
        "--tiles-dir",
        type=str,
        required=True,
        help="Dossier contenant les tuiles (ex: data/tiles/spacenet2_paris)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Nombre d'échantillons à visualiser"
    )
    
    args = parser.parse_args()
    
    tiles_dir = Path(args.tiles_dir)
    
    if not tiles_dir.exists():
        logger.error(f"Dossier introuvable: {tiles_dir}")
        return
    
    verify_tiles(tiles_dir, args.num_samples)


if __name__ == "__main__":
    main()