"""
Script de préparation pour SpaceNet 3 - Roads Dataset.
Tuilage et rasterisation des réseaux routiers.
"""

import argparse
import os
from pathlib import Path
import json
import numpy as np
import rasterio
from rasterio.windows import Window
from shapely.geometry import shape, mapping, LineString
import geopandas as gpd
from tqdm import tqdm
import logging
from typing import Dict, List, Tuple, Any

from src.geo.io import read_geotiff, write_geotiff_like
from src.geo.tiling import tile_array, save_tiles_manifest
from src.geo.rasterize import vector_to_mask
from src.utils import setup_logging, set_seed

logger = logging.getLogger(__name__)


class SpaceNet3Preprocessor:
    """Préprocesseur pour SpaceNet 3 Roads dataset."""
    
    def __init__(
        self,
        root: Path,
        output: Path,
        tile_size: int = 512,
        overlap: int = 64,
        road_buffer: float = 2.0,
        min_road_length: float = 10.0,
        class_map: Dict[str, int] = None,
        property_key: str = "type"
    ):
        self.root = Path(root)
        self.output = Path(output)
        self.tile_size = tile_size
        self.overlap = overlap
        self.road_buffer = road_buffer
        self.min_road_length = min_road_length
        self.class_map = class_map or {"road": 1}
        self.property_key = property_key
        
        # Créer les dossiers de sortie
        self.output_images = self.output / 'images'
        self.output_masks = self.output / 'masks'
        self.output_images.mkdir(parents=True, exist_ok=True)
        self.output_masks.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"SpaceNet3Preprocessor initialisé")
        logger.info(f"  Root: {self.root}")
        logger.info(f"  Output: {self.output}")
        logger.info(f"  Tile size: {self.tile_size}")
        logger.info(f"  Road buffer: {self.road_buffer}m")
    
    def find_image_label_pairs(self) -> List[Tuple[Path, Path]]:
        """Trouve les paires image-label dans le dataset SpaceNet 3."""
        # Chercher les images
        possible_img_dirs = [
            self.root / 'RGB-PanSharpen',
            self.root / 'images',
            self.root / 'PS-RGB'
        ]
        
        img_dir = None
        for possible_dir in possible_img_dirs:
            if possible_dir.exists():
                img_dir = possible_dir
                logger.info(f"Dossier d'images trouvé: {img_dir}")
                break
        
        if img_dir is None:
            raise FileNotFoundError(
                f"Aucun dossier d'images trouvé dans {self.root}"
            )
        
        # Chercher les labels
        possible_label_dirs = [
            self.root / 'geojson' / 'roads',
            self.root / 'labels',
            img_dir.parent / 'geojson' / 'spacenetroads'
        ]
        
        label_dir = None
        for possible_dir in possible_label_dirs:
            if possible_dir.exists():
                label_dir = possible_dir
                logger.info(f"Dossier de labels trouvé: {label_dir}")
                break
        
        if label_dir is None:
            raise FileNotFoundError(
                f"Aucun dossier de labels trouvé"
            )
        
        # Trouver les paires
        images = sorted(img_dir.glob("**/*.tif"))
        logger.info(f"Nombre d'images trouvées: {len(images)}")
        
        pairs = []
        missing_labels = []
        
        for img_path in images:
            label_path = label_dir / f"{img_path.stem}.geojson"
            
            if not label_path.exists():
                alternatives = list(label_dir.glob(f"{img_path.stem}.*"))
                if alternatives:
                    label_path = alternatives[0]
                else:
                    missing_labels.append(img_path.name)
                    continue
            
            pairs.append((img_path, label_path))
        
        if missing_labels:
            logger.warning(
                f"{len(missing_labels)} images sans labels. "
                f"Premières: {missing_labels[:5]}"
            )
        
        logger.info(f"Paires valides: {len(pairs)}")
        
        if len(pairs) == 0:
            raise ValueError("Aucune paire valide trouvée!")
        
        return pairs
    
    def preprocess_roads(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Prétraite les géométries de routes."""
        if len(gdf) == 0:
            return gdf
        
        gdf = gdf.copy()
        
        # Calculer longueur
        gdf['length'] = gdf.geometry.length
        
        # Filtrer les routes trop courtes
        original_count = len(gdf)
        gdf = gdf[gdf['length'] >= self.min_road_length]
        filtered = original_count - len(gdf)
        
        if filtered > 0:
            logger.debug(f"Filtré {filtered} routes < {self.min_road_length}m")
        
        # Appliquer un buffer pour élargir les routes
        if self.road_buffer > 0:
            gdf.geometry = gdf.geometry.buffer(self.road_buffer)
            logger.debug(f"Buffer de {self.road_buffer}m appliqué")
        
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
        write_geotiff_like(out_img_path, meta, img_tile)
        
        mask_meta = meta.copy()
        mask_meta.update(count=1, dtype='uint8')
        write_geotiff_like(out_mask_path, mask_meta, mask_tile)
    
    def process_pair(
        self,
        image_path: Path,
        label_path: Path
    ) -> List[Dict[str, Any]]:
        """Traite une paire image-label."""
        try:
            # Lecture de l'image
            img, img_meta = read_geotiff(image_path)
            
            # Lecture et prétraitement des routes
            gdf = gpd.read_file(label_path)
            
            if len(gdf) == 0:
                logger.warning(f"Aucune route dans {label_path.name}")
                return []
            
            gdf = self.preprocess_roads(gdf)
            
            if len(gdf) == 0:
                logger.warning(f"Toutes les routes filtrées pour {label_path.name}")
                return []
            
            # Rasterisation
            mask = vector_to_mask(
                gdf,
                out_shape=img.shape[1:],
                transform=img_meta['transform'],
                class_map=self.class_map,
                property_key=self.property_key,
                all_touched=True  # Important pour les routes
            )
            
            # Statistiques
            road_pixels = (mask > 0).sum()
            coverage = road_pixels / mask.size * 100
            total_length = gdf['length'].sum()
            
            logger.debug(
                f"{image_path.name}: {len(gdf)} segments, "
                f"longueur totale {total_length:.0f}m, "
                f"couverture {coverage:.2f}%"
            )
            
            # Tuilage
            img_tiles = tile_array(img, size=self.tile_size, overlap=self.overlap)
            mask_tiles = tile_array(
                mask[np.newaxis, ...],
                size=self.tile_size,
                overlap=self.overlap
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
                
                # Métadonnées
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
                
                self.save_pair(img_tile, mask_tile, tile_meta, img_out, mask_out)
                
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
                f"✓ {image_path.name}: {len(items)} tuiles "
                f"(couverture moyenne: {np.mean([it['coverage'] for it in items]):.2f}%)"
            )
            
            return items
        
        except Exception as e:
            logger.error(f"Erreur sur {image_path.name}: {e}")
            raise
    
    def run(self) -> None:
        """Exécute le préprocessing complet."""
        logger.info("=" * 80)
        logger.info("DÉBUT DU PREPROCESSING - SPACENET 3 ROADS")
        logger.info("=" * 80)
        
        pairs = self.find_image_label_pairs()
        
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
        
        # Statistiques
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
        
        # Manifest
        manifest_path = self.output / 'manifest.json'
        save_tiles_manifest(
            manifest_path,
            all_items,
            metadata={
                'dataset': 'SpaceNet 3 Roads',
                'tile_size': self.tile_size,
                'overlap': self.overlap,
                'class_map': self.class_map,
                'road_buffer': self.road_buffer,
                'min_road_length': self.min_road_length,
                'num_source_images': len(pairs),
                'num_errors': len(errors)
            }
        )
        
        logger.info("=" * 80)
        logger.info("PREPROCESSING TERMINÉ")
        logger.info("=" * 80)
        logger.info(f"Manifest: {manifest_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Préprocessing SpaceNet 3 Roads Dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--road-buffer", type=float, default=2.0, help="Buffer autour des routes (m)")
    parser.add_argument("--min-length", type=float, default=10.0, help="Longueur minimale des routes (m)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    
    args = parser.parse_args()
    
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_dir="logs", level=log_level)
    set_seed(args.seed)
    
    preprocessor = SpaceNet3Preprocessor(
        root=Path(args.root),
        output=Path(args.out),
        tile_size=args.size,
        overlap=args.overlap,
        road_buffer=args.road_buffer,
        min_road_length=args.min_length
    )
    
    try:
        preprocessor.run()
    except Exception as e:
        logger.exception("Erreur fatale")
        raise


if __name__ == "__main__":
    main()