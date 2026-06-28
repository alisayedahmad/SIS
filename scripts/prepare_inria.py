"""
Script de préparation pour INRIA Aerial Image Labeling Dataset.

Différences clés vs SpaceNet2:
- Pas de GeoJSON : les masques sont déjà des TIF rasterisés (gt/*.tif)
- Masques 8-bit : 255 = bâtiment, 0 = fond  → on binarise en 0/1
- Images 5000×5000 pixels (une par ville+numéro)
- Pas de CRS garanti (certaines images ont une projection locale)

Structure attendue APRÈS extraction :
    <root>/
    └── AerialImageDataset/
        └── train/
            ├── images/
            │   ├── austin1.tif
            │   ├── austin2.tif
            │   └── ...
            └── gt/
                ├── austin1.tif
                ├── austin2.tif
                └── ...

Usage :
    python -m scripts.prepare_inria \\
        --root /chemin/vers/AerialImageDataset/train \\
        --out data/tiles/inria \\
        --size 512 \\
        --overlap 64
"""

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rasterio

from src.geo.io import read_geotiff, write_geotiff_like
from src.geo.tiling import tile_array, save_tiles_manifest
from src.utils import setup_logging, set_seed

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _find_train_root(root: Path) -> Path:
    """
    Accepte plusieurs conventions de chemin :
      • root pointe directement sur le dossier qui contient images/ et gt/
      • root/AerialImageDataset/train
      • root/train
    """
    candidates = [
        root,
        root / "train",
        root / "AerialImageDataset" / "train",
    ]
    for candidate in candidates:
        if (candidate / "images").exists() and (candidate / "gt").exists():
            return candidate
    raise FileNotFoundError(
        f"Impossible de trouver images/ et gt/ sous {root}.\n"
        "Vérifiez l'extraction : le dossier doit contenir images/ et gt/."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Preprocesseur principal
# ──────────────────────────────────────────────────────────────────────────────

class InriaPreprocessor:
    """
    Convertit le dataset INRIA en tuiles GeoTIFF compatibles avec
    TilesDataModule (manifest.json + images/ + masks/).
    """

    def __init__(
        self,
        root: Path,
        output: Path,
        tile_size: int = 512,
        overlap: int = 64,
        cities: Optional[List[str]] = None,
    ):
        self.train_root = _find_train_root(Path(root))
        self.output = Path(output)
        self.tile_size = tile_size
        self.overlap = overlap
        self.cities = cities  # None = toutes les villes

        self.out_images = self.output / "images"
        self.out_masks = self.output / "masks"
        self.out_images.mkdir(parents=True, exist_ok=True)
        self.out_masks.mkdir(parents=True, exist_ok=True)

        logger.info("InriaPreprocessor initialisé")
        logger.info(f"  Train root : {self.train_root}")
        logger.info(f"  Output     : {self.output}")
        logger.info(f"  Tile size  : {self.tile_size}")
        logger.info(f"  Overlap    : {self.overlap}")
        if self.cities:
            logger.info(f"  Villes     : {self.cities}")
        else:
            logger.info("  Villes     : toutes")

    # ── Découverte des paires ──────────────────────────────────────────────

    def find_pairs(self) -> List[Tuple[Path, Path]]:
        """
        Renvoie la liste de (image_path, mask_path) pour le split train.
        """
        img_dir = self.train_root / "images"
        gt_dir = self.train_root / "gt"

        all_images = sorted(img_dir.glob("*.tif"))
        if not all_images:
            raise FileNotFoundError(
                f"Aucune image .tif trouvée dans {img_dir}"
            )

        pairs: List[Tuple[Path, Path]] = []
        missing: List[str] = []

        for img_path in all_images:
            # Filtre optionnel par ville
            if self.cities:
                city = "".join(c for c in img_path.stem if not c.isdigit())
                if city not in self.cities:
                    continue

            mask_path = gt_dir / img_path.name
            if not mask_path.exists():
                missing.append(img_path.name)
                continue

            pairs.append((img_path, mask_path))

        if missing:
            logger.warning(
                f"{len(missing)} images sans masque gt ignorées : "
                f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
            )

        logger.info(f"Paires valides trouvées : {len(pairs)}")
        if not pairs:
            raise ValueError("Aucune paire image-masque valide !")

        return pairs

    # ── Traitement d'une paire ─────────────────────────────────────────────

    def process_pair(
        self,
        img_path: Path,
        mask_path: Path,
    ) -> List[Dict[str, Any]]:
        """
        Lit une image + son masque INRIA, les découpe en tuiles et les sauvegarde.
        Renvoie la liste des items à ajouter au manifest.
        """
        try:
            # ── Lecture image RGB ──────────────────────────────────────────
            img, img_meta = read_geotiff(img_path)  # (3, H, W) uint8

            # ── Lecture masque (1 bande, valeurs 0 ou 255) ────────────────
            mask_raw, _ = read_geotiff(mask_path)   # (1, H, W) uint8

            # Binarisation : 255 → 1
            mask = (mask_raw > 0).astype(np.uint8)  # (1, H, W), valeurs 0/1

            # Vérification de cohérence spatiale
            if img.shape[1:] != mask.shape[1:]:
                raise ValueError(
                    f"Dimensions incohérentes : image {img.shape[1:]} "
                    f"vs masque {mask.shape[1:]}"
                )

            coverage_pct = mask.mean() * 100
            logger.debug(
                f"{img_path.name} : {img.shape}, "
                f"couverture bâtiments {coverage_pct:.2f}%"
            )

            # ── Tuilage ───────────────────────────────────────────────────
            img_tiles = tile_array(
                img, size=self.tile_size, overlap=self.overlap
            )
            mask_tiles = tile_array(
                mask, size=self.tile_size, overlap=self.overlap
            )

            if len(img_tiles) != len(mask_tiles):
                raise ValueError(
                    f"Tuiles désalignées : {len(img_tiles)} images "
                    f"vs {len(mask_tiles)} masques"
                )

            # ── Sauvegarde des tuiles ─────────────────────────────────────
            items: List[Dict[str, Any]] = []

            for (coords, img_tile), (_, mask_tile) in zip(img_tiles, mask_tiles):
                y, x, h, w = coords
                stem = f"{img_path.stem}_{y:05d}_{x:05d}"

                img_out = self.out_images / f"{stem}.tif"
                mask_out = self.out_masks / f"{stem}.tif"

                # Métadonnées de la tuile (préservation du géoréférencement si présent)
                tile_meta = img_meta.copy()
                if img_meta.get("transform") is not None:
                    import rasterio.transform as rt
                    orig_t = img_meta["transform"]
                    tile_meta["transform"] = rt.Affine(
                        orig_t.a, 0,
                        orig_t.c + x * orig_t.a,
                        0, orig_t.e,
                        orig_t.f + y * orig_t.e,
                    )
                tile_meta.update(height=h, width=w)

                # Image RGB
                write_geotiff_like(img_out, tile_meta, img_tile)

                # Masque (1 bande, uint8)
                mask_meta = tile_meta.copy()
                mask_meta.update(count=1, dtype="uint8")
                write_geotiff_like(mask_out, mask_meta, mask_tile)

                tile_coverage = float(mask_tile.mean() * 100)
                items.append({
                    "image": str(img_out.relative_to(self.output)),
                    "mask": str(mask_out.relative_to(self.output)),
                    "source_image": img_path.name,
                    "coords": list(coords),
                    "coverage": tile_coverage,
                    "num_pixels": int(mask_tile.sum()),
                })

            logger.info(
                f"✓ {img_path.name} → {len(items)} tuiles "
                f"(couverture moy. {np.mean([it['coverage'] for it in items]):.2f}%)"
            )
            return items

        except Exception as e:
            logger.error(f"✗ Erreur sur {img_path.name} : {e}")
            raise

    # ── Point d'entrée ────────────────────────────────────────────────────

    def run(self) -> None:
        logger.info("=" * 70)
        logger.info("DÉBUT DU PREPROCESSING - INRIA AERIAL IMAGE LABELING")
        logger.info("=" * 70)

        pairs = self.find_pairs()

        all_items: List[Dict[str, Any]] = []
        errors: List[Tuple[str, str]] = []

        for img_path, mask_path in pairs:
            try:
                items = self.process_pair(img_path, mask_path)
                all_items.extend(items)
            except Exception as e:
                errors.append((img_path.name, str(e)))
                continue

        # ── Résumé ────────────────────────────────────────────────────────
        logger.info("=" * 70)
        logger.info("RÉSUMÉ")
        logger.info("=" * 70)
        logger.info(f"Paires traitées  : {len(pairs)}")
        logger.info(f"Erreurs          : {len(errors)}")
        logger.info(f"Tuiles générées  : {len(all_items)}")

        if all_items:
            coverages = [it["coverage"] for it in all_items]
            logger.info(f"Couverture moy.  : {np.mean(coverages):.2f}%")
            logger.info(f"Couverture méd.  : {np.median(coverages):.2f}%")
            logger.info(f"Couv. min / max  : {np.min(coverages):.2f}% / {np.max(coverages):.2f}%")

        # ── Manifest ──────────────────────────────────────────────────────
        manifest_path = self.output / "manifest.json"
        save_tiles_manifest(
            manifest_path,
            all_items,
            metadata={
                "dataset": "INRIA Aerial Image Labeling",
                "tile_size": self.tile_size,
                "overlap": self.overlap,
                "cities": self.cities,
                "num_source_images": len(pairs),
                "num_errors": len(errors),
                "errors": [{"file": f, "error": e} for f, e in errors[:10]],
            },
        )

        logger.info("=" * 70)
        logger.info("PREPROCESSING TERMINÉ")
        logger.info("=" * 70)
        logger.info(f"Manifest : {manifest_path}")

        if errors:
            logger.warning(f"⚠  {len(errors)} erreur(s) — consultez les logs.")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Préprocessing INRIA Aerial Image Labeling Dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--root", type=str, required=True,
        help=(
            "Racine du dataset (dossier qui contient images/ et gt/, "
            "ou parent de AerialImageDataset/)"
        ),
    )
    parser.add_argument(
        "--out", type=str, required=True,
        help="Dossier de sortie pour les tuiles",
    )
    parser.add_argument(
        "--size", type=int, default=512,
        help="Taille des tuiles en pixels",
    )
    parser.add_argument(
        "--overlap", type=int, default=64,
        help="Chevauchement entre tuiles en pixels",
    )
    parser.add_argument(
        "--cities", nargs="*", default=None,
        help=(
            "Villes à traiter (ex: --cities austin chicago). "
            "Par défaut : toutes."
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Seed pour la reproductibilité",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Logs détaillés (DEBUG)",
    )

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_dir="logs", level=log_level)
    set_seed(args.seed)

    preprocessor = InriaPreprocessor(
        root=Path(args.root),
        output=Path(args.out),
        tile_size=args.size,
        overlap=args.overlap,
        cities=args.cities,
    )

    try:
        preprocessor.run()
    except Exception:
        logger.exception("Erreur fatale durant le preprocessing")
        raise


if __name__ == "__main__":
    main()
