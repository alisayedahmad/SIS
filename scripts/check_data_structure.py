import argparse
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


def check_spacenet2_structure(root: Path) -> dict:
    """
    Vérifie la structure d'un dataset SpaceNet 2
    
    Returns:
        dict avec statistiques
    """
    logger.info("=" * 80)
    logger.info("VÉRIFICATION DE LA STRUCTURE - SPACENET 2")
    logger.info("=" * 80)
    logger.info(f"Racine: {root}")
    
    stats = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'num_images': 0,
        'num_labels': 0,
        'matched_pairs': 0,
        'unmatched_images': [],
        'unmatched_labels': []
    }
    
    # Chercher le dossier d'images
    possible_img_dirs = [
        root / 'RGB-PanSharpen',
        root / 'images',
        root / 'AOI_3_Paris_Train' / 'RGB-PanSharpen'
    ]
    
    img_dir = None
    for path in possible_img_dirs:
        if path.exists():
            img_dir = path
            logger.info(f"✓ Dossier d'images trouvé: {img_dir}")
            break
    
    if img_dir is None:
        stats['valid'] = False
        stats['errors'].append("Dossier d'images introuvable")
        logger.error("✗ Aucun dossier d'images trouvé!")
        logger.error(f"  Cherché dans: {[str(p) for p in possible_img_dirs]}")
        return stats
    
    # Chercher le dossier de labels
    possible_label_dirs = [
        root / 'geojson' / 'buildings',
        root / 'labels',
        img_dir.parent / 'geojson' / 'buildings'
    ]
    
    label_dir = None
    for path in possible_label_dirs:
        if path.exists():
            label_dir = path
            logger.info(f"✓ Dossier de labels trouvé: {label_dir}")
            break
    
    if label_dir is None:
        stats['valid'] = False
        stats['errors'].append("Dossier de labels introuvable")
        logger.error("✗ Aucun dossier de labels trouvé!")
        return stats
    
    # Compter les fichiers
    images = list(img_dir.glob("*.tif"))
    stats['num_images'] = len(images)
    logger.info(f"  Images: {stats['num_images']}")
    
    labels = list(label_dir.glob("*.geojson"))
    stats['num_labels'] = len(labels)
    logger.info(f"  Labels: {stats['num_labels']}")
    
    if stats['num_images'] == 0:
        stats['valid'] = False
        stats['errors'].append("Aucune image .tif trouvée")
        logger.error("✗ Aucune image trouvée!")
        return stats
    
    if stats['num_labels'] == 0:
        stats['valid'] = False
        stats['errors'].append("Aucun label .geojson trouvé")
        logger.error("✗ Aucun label trouvé!")
        return stats
    
    # Vérifier les paires
    logger.info("\nVérification des paires image-label...")
    
    for img_path in images:
        # Chercher le label correspondant
        label_path = label_dir / f"{img_path.stem}.geojson"
        
        if not label_path.exists():
            # Essayer avec variations de noms
            possible_names = [
                f"buildings_{img_path.stem}.geojson",
                f"{img_path.stem.replace('RGB-PanSharpen_', '')}.geojson"
            ]
            
            found = False
            for name in possible_names:
                alt_path = label_dir / name
                if alt_path.exists():
                    label_path = alt_path
                    found = True
                    break
            
            if not found:
                stats['unmatched_images'].append(img_path.name)
                continue
        
        stats['matched_pairs'] += 1
    
    # Labels sans images
    for label_path in labels:
        stem = label_path.stem.replace('buildings_', '')
        img_path = img_dir / f"{stem}.tif"
        
        if not img_path.exists():
            stats['unmatched_labels'].append(label_path.name)
    
    # Résumé
    logger.info("=" * 80)
    logger.info("RÉSUMÉ")
    logger.info("=" * 80)
    logger.info(f"Images totales: {stats['num_images']}")
    logger.info(f"Labels totaux: {stats['num_labels']}")
    logger.info(f"Paires valides: {stats['matched_pairs']}")
    
    if stats['unmatched_images']:
        logger.warning(f"⚠️  Images sans label: {len(stats['unmatched_images'])}")
        logger.warning(f"    Exemples: {stats['unmatched_images'][:3]}")
        stats['warnings'].append(f"{len(stats['unmatched_images'])} images sans label")
    
    if stats['unmatched_labels']:
        logger.warning(f"⚠️  Labels sans image: {len(stats['unmatched_labels'])}")
        stats['warnings'].append(f"{len(stats['unmatched_labels'])} labels sans image")
    
    if stats['matched_pairs'] == 0:
        stats['valid'] = False
        stats['errors'].append("Aucune paire valide trouvée")
        logger.error("✗ AUCUNE PAIRE VALIDE!")
    else:
        logger.info(f"✓ Structure valide: {stats['matched_pairs']} paires prêtes")
    
    # Estimation de l'espace disque nécessaire
    if images:
        sample_size = images[0].stat().st_size
        total_size_mb = (sample_size * stats['matched_pairs']) / (1024 * 1024)
        tiles_per_image = ((2048 // 512) ** 2) * 1.5  # Estimation avec overlap
        estimated_tiles = int(stats['matched_pairs'] * tiles_per_image)
        estimated_space_gb = (total_size_mb * tiles_per_image * 2) / 1024  # x2 pour images + masks
        
        logger.info("\nEstimations:")
        logger.info(f"  Tuiles attendues: ~{estimated_tiles}")
        logger.info(f"  Espace disque nécessaire: ~{estimated_space_gb:.1f} GB")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Vérifier la structure des données avant preprocessing"
    )
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Racine du dataset (ex: data/raw/spacenet2/AOI_3_Paris_Train)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=['spacenet2', 'spacenet3', 'deepglobe'],
        default='spacenet2',
        help="Type de dataset"
    )
    
    args = parser.parse_args()
    
    root = Path(args.root)
    
    if not root.exists():
        logger.error(f"✗ Le chemin n'existe pas: {root}")
        return
    
    if args.dataset == 'spacenet2':
        stats = check_spacenet2_structure(root)
    else:
        logger.error(
            f"✗ Vérification non implémentée pour le dataset '{args.dataset}'. "
            f"Seul 'spacenet2' est actuellement supporté par ce script."
        )
        return
    
    # Afficher les erreurs
    if stats['errors']:
        logger.error("\n❌ ERREURS DÉTECTÉES:")
        for err in stats['errors']:
            logger.error(f"  - {err}")
    
    if stats['warnings']:
        logger.warning("\n⚠️  AVERTISSEMENTS:")
        for warn in stats['warnings']:
            logger.warning(f"  - {warn}")
    
    if stats['valid']:
        logger.info("\n✅ STRUCTURE VALIDE - Prêt pour le preprocessing!")
    else:
        logger.error("\n❌ STRUCTURE INVALIDE - Corrigez les erreurs avant de continuer")


if __name__ == "__main__":
    main()