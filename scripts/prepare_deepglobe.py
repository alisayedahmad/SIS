"""
Script de préparation pour DeepGlobe Dataset.
Support multi-classes (buildings, roads, etc.).
"""

import argparse
import os
from pathlib import Path
import json
import numpy as np
import rasterio
from rasterio.windows import Window
from shapely.geometry import shape, mapping
import geopandas as gpd
from tqdm import tqdm
import logging
from typing import Dict, List, Tuple, Any

from src.geo.io import read_geotiff, write_geotiff_like
from src.geo.tiling import tile_array, save_tiles_manifest
from src.geo.rasterize import multiclass_rasterize
from src.utils import setup_logging, set_seed

logger = logging.getLogger(__name__)


class DeepGlobePreprocessor:
    """Préprocesseur pour DeepGlobe dataset (multi-classes)."""
    
    def __init__(
        self,
        root: Path,
        output: Path,
        tile_size: int = 512,
        overlap: int = 64,
        class_map: Dict[str, int] = None,
        property_key: str = "class"
    ):
        self.root = Path(root)
        self.output = Path(output)
        self.tile_size = tile_size
        self.overlap = overlap
        self.class_map = class_map or {"building": 1, "road": 2}
        self.property_key = property_key
        
        # Créer les dossiers
        self.output_images = self.output / 'images'
        self.output_masks = self.output / 'masks'
        self.output_images.mkdir(parents=True, exist_ok=True)
        self.output_masks.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"DeepGlobePreprocessor initialisé")
        logger.info(f"  Classes: {self.class_map}")
    
    def find_image_label_pairs(self) -> List[Tuple[Path, Path]]:
        """Trouve les paires dans DeepGlobe."""
        possible_img_dirs = [
            self.root / 'images',
            self.root / 'train',
            self.root / 'sat'
        ]
        
        img_dir = None
        for possible_dir in possible_img_dirs:
            if possible_dir.exists():
                img_dir = possible_dir
                break
        
        if img_dir is None:
            raise FileNotFoundError(f"Dossier d'images introuvable dans {self.root}")
        
        possible_label_dirs = [
            self.root / 'labels',
            self.root / 'masks',
            self.root / 'mask'
        ]
        
        label_dir = None
        for possible_dir in possible_label_dirs:
            if possible_dir.exists():
                label_dir = possible_dir
                break
        
        if label_dir is None:
            raise FileNotFoundError("Dossier de labels introuvable")
        
        images = sorted(img_dir.glob("**/*.tif")) + sorted(img_dir.glob("**/*.png"))
        logger.info(f"Images trouvées: {len(images)}")
        
        pairs = []
        
        for img_path in images:
            # Chercher plusieurs formats de labels
            for ext in ['.geojson', '.json', '.shp', '.tif', '.png']:
                label_path = label_dir / f"{img_path.stem}{ext}"
                if label_path.exists():
                    pairs.append((img_path, label_path))
                    break
        
        logger.info(f"Paires valides: {len(pairs)}")
        
        if len(pairs) == 0:
            raise ValueError("Aucune paire trouvée!")
        
        return pairs
    
    def save_pair(
        self,
        img_tile: np.ndarray,
        mask_tile: np.ndarray,
        meta: Dict[str, Any],
        out_img_path: Path,
        out_mask_path: Path
    ) -> None:
        """Sauvegarde une paire."""
        write_geotiff_like(out_img_path, meta, img_tile)
        
        mask_meta = meta.copy()
        mask_meta.update(count=1, dtype='uint8')
        write_geotiff_like(out_mask_path, mask_meta, mask_tile)
    
    def process_pair(
        self,
        image_path: Path,
        label_path: Path
    ) -> List[Dict[str, Any]]:
        """Traite une paire."""
        try:
            # Lecture image
            img, img_meta = read_geotiff(image_path)
            
            # Lecture label (vectoriel ou raster)
            if label_path.suffix in ['.geojson', '.json', '.shp']:
                # Vectoriel -> rasteriser
                gdf = gpd.read_file(label_path)
                
                if len(gdf) == 0:
                    logger.warning(f"Label vide: {label_path.name}")
                    return []
                
                # Rasterisation multi-classes
                mask = multiclass_rasterize(
                    gdf,
                    out_shape=img.shape[1:],
                    transform=img_meta['transform'],
                    class_column=self.property_key
                )
            else:
                # Raster
                mask, _ = read_geotiff(label_path)
                if mask.ndim == 3:
                    mask = mask[0]
            
            # Statistiques
            unique, counts = np.unique(mask, return_counts=True)
            logger.debug(f"{image_path.name}: classes {unique}")
            
            # Tuilage
            img_tiles = tile_array(img, size=self.tile_size, overlap=self.overlap)
            mask_tiles = tile_array(
                mask[np.newaxis, ...],
                size=self.tile_size,
                overlap=self.overlap
            )
            
            items = []
            
            for i, ((coords, img_tile), (_, mask_tile)) in enumerate(
                zip(img_tiles, mask_tiles)
            ):
                y, x, h, w = coords
                stem = f"{image_path.stem}_{y:04d}_{x:04d}"
                
                img_out = self.output_images / f"{stem}.tif"
                mask_out = self.output_masks / f"{stem}.tif"
                
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
                
                # Compter pixels par classe
                unique_tile, counts_tile = np.unique(mask_tile, return_counts=True)
                class_distribution = {
                    int(u): int(c) for u, c in zip(unique_tile, counts_tile)
                }
                
                items.append({
                    'image': str(img_out.relative_to(self.output)),
                    'mask': str(mask_out.relative_to(self.output)),
                    'source_image': image_path.name,
                    'coords': coords,
                    'class_distribution': class_distribution
                })
            
            logger.info(f"✓ {image_path.name}: {len(items)} tuiles")
            
            return items
        
        except Exception as e:
            logger.error(f"Erreur sur {image_path.name}: {e}")
            raise
    
    def run(self) -> None:
        """Exécute le préprocessing."""
        logger.info("=" * 80)
        logger.info("DÉBUT DU PREPROCESSING - DEEPGLOBE")
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
        logger.info(f"Paires: {len(pairs)}")
        logger.info(f"Erreurs: {len(errors)}")
        logger.info(f"Tuiles: {len(all_items)}")
        
        # Manifest
        manifest_path = self.output / 'manifest.json'
        save_tiles_manifest(
            manifest_path,
            all_items,
            metadata={
                'dataset': 'DeepGlobe',
                'tile_size': self.tile_size,
                'overlap': self.overlap,
                'class_map': self.class_map,
                'property_key': self.property_key,
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
        description="Préprocessing DeepGlobe Dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    
    args = parser.parse_args()
    
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_dir="logs", level=log_level)
    set_seed(args.seed)
    
    preprocessor = DeepGlobePreprocessor(
        root=Path(args.root),
        output=Path(args.out),
        tile_size=args.size,
        overlap=args.overlap
    )
    
    try:
        preprocessor.run()
    except Exception as e:
        logger.exception("Erreur fatale")
        raise


if __name__ == "__main__":
    main()