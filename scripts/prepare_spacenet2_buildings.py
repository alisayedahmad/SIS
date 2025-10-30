"""
Script de préparation pour SpaceNet 2 - Buildings Dataset.
Tuilage et rasterisation des annotations GeoJSON.
"""

import argparse
import os
from pathlib import Path
import json
import numpy as np
import rasterio
from rasterio.windows import Window
import geopandas as gpd
from tqdm import tqdm
import logging
from typing import Dict, List, Tuple, Any,Optional

from src.geo.io import read_geotiff, write_geotiff_like
from src.geo.tiling import tile_array, save_tiles_manifest
from src.geo.rasterize import vector_to_mask
from src.utils import setup_logging, set_seed

logger = logging.getLogger(__name__)


class SpaceNet2Preprocessor:
    """Préprocesseur pour SpaceNet 2 Buildings dataset."""
    
    def __init__(
        self,
        root: Path,
        output: Path,
        tile_size: int = 512,
        overlap: int = 64,
        min_building_area: float = 20.0,
        class_map: Optional[Dict[str, int]] = None,
        property_key: str = "type"
    ):
        self.root = Path(root)
        self.output = Path(output)
        self.tile_size = tile_size
        self.overlap = overlap
        self.min_building_area = min_building_area
        self.class_map = class_map or {"building": 1}
        self.property_key = property_key
        
        # Créer les dossiers de sortie
        self.output_images = self.output / 'images'
        self.output_masks = self.output / 'masks'
        self.output_images.mkdir(parents=True, exist_ok=True)
        self.output_masks.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"SpaceNet2Preprocessor initialisé")
        logger.info(f"  Root: {self.root}")
        logger.info(f"  Output: {self.output}")
        logger.info(f"  Tile size: {self.tile_size}")
        logger.info(f"  Overlap: {self.overlap}")
    
    def find_image_label_pairs(self) -> List[Tuple[Path, Path]]:
        """
        Trouve les paires image-label dans le dataset SpaceNet 2.
        
        Returns:
            Liste de tuples (image_path, label_path)
        """
        # Chercher les images dans différents dossiers possibles
        possible_img_dirs = [
            self.root / 'RGB-PanSharpen',
            self.root / 'images',
            self.root / 'AOI_2_Vegas_Train' / 'RGB-PanSharpen',
            self.root / 'AOI_3_Paris_Train' / 'RGB-PanSharpen',
            self.root / 'AOI_4_Shanghai_Train' / 'RGB-PanSharpen',
            self.root / 'AOI_5_Khartoum_Train' / 'RGB-PanSharpen'
        ]
        
        img_dir = None
        for possible_dir in possible_img_dirs:
            if possible_dir.exists():
                img_dir = possible_dir
                logger.info(f"Dossier d'images trouvé: {img_dir}")
                break
        
        if img_dir is None:
            raise FileNotFoundError(
                f"Aucun dossier d'images trouvé dans {self.root}. "
                f"Vérifiez la structure du dataset SpaceNet 2."
            )
        
        # Chercher les labels
        possible_label_dirs = [
            self.root / 'geojson' / 'buildings',
            self.root / 'labels',
            img_dir.parent / 'geojson' / 'buildings'
        ]
        
        label_dir = None
        for possible_dir in possible_label_dirs:
            if possible_dir.exists():
                label_dir = possible_dir
                logger.info(f"Dossier de labels trouvé: {label_dir}")
                break
        
        if label_dir is None:
            raise FileNotFoundError(
                f"Aucun dossier de labels trouvé. "
                f"Cherché dans: {[str(d) for d in possible_label_dirs]}"
            )
        
        # Trouver les paires
        images = sorted(img_dir.glob("**/*.tif"))
        logger.info(f"Nombre d'images trouvées: {len(images)}")
        
        pairs = []
        missing_labels = []
        
        for img_path in images:
            # Chercher le label correspondant
            label_path = label_dir / f"{img_path.stem}.geojson"
            
            if not label_path.exists():
                # Essayer d'autres formats
                alternatives = list(label_dir.glob(f"{img_path.stem}.*"))
                if alternatives:
                    label_path = alternatives[0]
                else:
                    missing_labels.append(img_path.name)
                    continue
            
            pairs.append((img_path, label_path))
        
        if missing_labels:
            logger.warning(
                f"{len(missing_labels)} images sans labels trouvées. "
                f"Premières: {missing_labels[:5]}"
            )
        
        logger.info(f"Paires valides trouvées: {len(pairs)}")
        
        if len(pairs) == 0:
            raise ValueError("Aucune paire image-label valide trouvée!")
        
        return pairs
    
    def filter_small_buildings(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Filtre les petits bâtiments."""
        if len(gdf) == 0:
            return gdf
        
        gdf = gdf.copy()
        gdf['area'] = gdf.geometry.area
        
        original_count = len(gdf)
        gdf = gdf[gdf['area'] >= self.min_building_area]
        filtered_count = original_count - len(gdf)
        
        if filtered_count > 0:
            logger.debug(
                f"Filtré {filtered_count} bâtiments < {self.min_building_area}m²"
            )
        
        return gdf
    
    def save_pair(
        self,
        img_tile: np.ndarray,
        mask_tile: np.ndarray,
        meta: Dict[str, Any],
        out_img_path: Path,
        out_mask_path: Path
    ) -> None:
        """Sauvegarde une paire image-masque."""
        # Écriture image
        write_geotiff_like(out_img_path, meta, img_tile)
        
        # Écriture masque
        mask_meta = meta.copy()
        mask_meta.update(count=1, dtype='uint8')
        write_geotiff_like(out_mask_path, mask_meta, mask_tile)
    
    def process_pair(
        self,
        image_path: Path,
        label_path: Path
    ) -> List[Dict[str, Any]]:
        """
        Traite une paire image-label.
        
        Args:
            image_path: Chemin de l'image
            label_path: Chemin du label
        
        Returns:
            Liste des items de tuiles créées
        """
        try:
            # Lecture de l'image
            img, img_meta = read_geotiff(image_path)
            
            # Lecture et filtrage des labels
            gdf = gpd.read_file(label_path)
            
            if len(gdf) == 0:
                logger.warning(f"Aucune annotation dans {label_path.name}")
                return []
            
            # Filtrer les petits bâtiments
            gdf = self.filter_small_buildings(gdf)
            
            if len(gdf) == 0:
                logger.warning(f"Toutes les annotations filtrées pour {label_path.name}")
                return []
            
            # Rasterisation
            mask = vector_to_mask(
                gdf,
                out_shape=img.shape[1:],
                transform=img_meta['transform'],
                class_map=self.class_map,
                property_key=self.property_key
            )
            
            # Statistiques
            building_pixels = (mask > 0).sum()
            coverage = building_pixels / mask.size * 100
            
            logger.debug(
                f"{image_path.name}: {len(gdf)} bâtiments, "
                f"couverture {coverage:.2f}%"
            )
            
            # Tuilage
            img_tiles = tile_array(img, size=self.tile_size, overlap=self.overlap)
            mask_tiles = tile_array(
                mask[np.newaxis, ...],
                size=self.tile_size,
                overlap=self.overlap
            )
            
            if len(img_tiles) != len(mask_tiles):
                raise ValueError(
                    f"Nombre de tuiles incohérent: "
                    f"{len(img_tiles)} images vs {len(mask_tiles)} masques"
                )
            
            # Sauvegarder les tuiles
            items = []
            
            for i, ((coords, img_tile), (_, mask_tile)) in enumerate(
                zip(img_tiles, mask_tiles)
            ):
                y, x, h, w = coords
                stem = f"{image_path.stem}_{y:04d}_{x:04d}"
                
                img_out = self.output_images / f"{stem}.tif"
                mask_out = self.output_masks / f"{stem}.tif"
                
                # Métadonnées de la tuile
                tile_meta = img_meta.copy()
                tile_meta.update({
                    'height': h,
                    'width': w,
                    'transform': rasterio.Affine(
                        img_meta['transform'].a, 0,
                        img_meta['transform'].c + x * img_meta['transform'].a,
                        0, img_meta['transform'].e,
                        img_meta['transform'].f + y * img_meta['transform'].e
                    )
                })
                
                # Sauvegarder
                self.save_pair(img_tile, mask_tile, tile_meta, img_out, mask_out)
                
                # Statistiques de la tuile
                tile_coverage = (mask_tile.sum() / mask_tile.size) * 100
                
                items.append({
                    'image': str(img_out.relative_to(self.output)),
                    'mask': str(mask_out.relative_to(self.output)),
                    'source_image': image_path.name,
                    'coords': coords,
                    'coverage': float(tile_coverage),
                    'num_pixels': int(mask_tile.sum())
                })
            
            logger.info(
                f"✓ {image_path.name}: {len(items)} tuiles générées "
                f"(couverture moyenne: {np.mean([it['coverage'] for it in items]):.2f}%)"
            )
            
            return items
        
        except Exception as e:
            logger.error(f"Erreur sur {image_path.name}: {e}")
            raise
    
    def run(self) -> None:
        """Exécute le préprocessing complet."""
        logger.info("=" * 80)
        logger.info("DÉBUT DU PREPROCESSING - SPACENET 2 BUILDINGS")
        logger.info("=" * 80)
        
        # Trouver les paires
        pairs = self.find_image_label_pairs()
        
        # Traiter chaque paire
        all_items = []
        errors = []
        
        for img_path, label_path in tqdm(pairs, desc="Traitement"):
            try:
                items = self.process_pair(img_path, label_path)
                all_items.extend(items)
            except Exception as e:
                logger.error(f"Erreur fatale sur {img_path.name}: {e}")
                errors.append((img_path.name, str(e)))
                continue
        
        # Statistiques globales
        logger.info("=" * 80)
        logger.info("STATISTIQUES")
        logger.info("=" * 80)
        logger.info(f"Paires traitées: {len(pairs)}")
        logger.info(f"Erreurs: {len(errors)}")
        logger.info(f"Tuiles générées: {len(all_items)}")
        
        if all_items:
            coverages = [item['coverage'] for item in all_items]
            logger.info(f"Couverture moyenne: {np.mean(coverages):.2f}%")
            logger.info(f"Couverture médiane: {np.median(coverages):.2f}%")
            logger.info(f"Couverture min/max: {np.min(coverages):.2f}% / {np.max(coverages):.2f}%")
        
        # Sauvegarder le manifest
        manifest_path = self.output / 'manifest.json'
        save_tiles_manifest(
            manifest_path,
            all_items,
            metadata={
                'dataset': 'SpaceNet 2 Buildings',
                'tile_size': self.tile_size,
                'overlap': self.overlap,
                'class_map': self.class_map,
                'property_key': self.property_key,
                'min_building_area': self.min_building_area,
                'num_source_images': len(pairs),
                'num_errors': len(errors),
                'errors': errors[:10]  # Premières erreurs seulement
            }
        )
        
        logger.info("=" * 80)
        logger.info("PREPROCESSING TERMINÉ")
        logger.info("=" * 80)
        logger.info(f"Manifest: {manifest_path}")
        
        if errors:
            logger.warning(f"⚠️  {len(errors)} erreurs détectées. Vérifiez les logs.")


def main():
    parser = argparse.ArgumentParser(
        description="Préprocessing SpaceNet 2 Buildings Dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Racine du dataset SpaceNet 2 (contient RGB-PanSharpen/ et geojson/)"
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Dossier de sortie pour les tuiles"
    )
    parser.add_argument(
        "--size",
        type=int,
        default=512,
        help="Taille des tuiles (pixels)"
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=64,
        help="Chevauchement entre tuiles (pixels)"
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=20.0,
        help="Aire minimale des bâtiments à conserver (m²)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed pour reproductibilité"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Mode verbeux"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_dir="logs", level=log_level)
    
    # Set seed
    set_seed(args.seed)
    
    # Créer le préprocesseur
    preprocessor = SpaceNet2Preprocessor(
        root=Path(args.root),
        output=Path(args.out),
        tile_size=args.size,
        overlap=args.overlap,
        min_building_area=args.min_area
    )
    
    # Exécuter
    try:
        preprocessor.run()
    except Exception as e:
        logger.exception("Erreur fatale durant le preprocessing")
        raise


if __name__ == "__main__":
    main()