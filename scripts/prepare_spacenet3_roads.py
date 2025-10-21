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

from src.geo.io import read_geotiff, write_geotiff_like
from src.geo.tiling import tile_array, save_tiles_manifest
from src.geo.rasterize import vector_to_mask

def save_pair(img_tile, mask_tile, meta, out_img_path, out_mask_path):
    os.makedirs(os.path.dirname(out_img_path), exist_ok=True)
    os.makedirs(os.path.dirname(out_mask_path), exist_ok=True)
    with rasterio.open(out_img_path, 'w', **meta) as dst:
        dst.write(img_tile)
    with rasterio.open(out_mask_path, 'w', **meta) as dst:
        dst.write(mask_tile)

def run(root, out, size, overlap, vector_key, class_map):
    root = Path(root); out = Path(out)
    img_dir = next((root / 'images'), root)
    vec_dir = root / 'labels'
    out_img = out / 'images'
    out_mask = out / 'masks'
    out_img.mkdir(parents=True, exist_ok=True)
    out_mask.mkdir(parents=True, exist_ok=True)

    images = list(img_dir.glob("**/*.tif"))
    assert images, f"Aucune image trouvée dans {img_dir}"

    all_items = []
    for img_path in tqdm(images, desc="Préparation"):
        with rasterio.open(img_path) as src:
            img = src.read()
            meta = src.meta.copy()
            meta.update(count=1, dtype="uint8")
            # Cherche vecteur co-localisé
            vec_path = (vec_dir / img_path.with_suffix(".geojson").name)
            if not vec_path.exists():
                # tente .json ou .shp
                alt = list(vec_dir.glob(img_path.stem + ".*"))
                assert alt, f"Aucun label trouvé pour {img_path.name}"
                vec_path = alt[0]

            gdf = gpd.read_file(vec_path)
            mask = vector_to_mask(gdf, out_shape=img.shape[1:], transform=src.transform, class_map=class_map, property_key=vector_key)
            # Tuilage
            tiles = tile_array(img, size=size, overlap=overlap)
            mask_tiles = tile_array(mask[np.newaxis, ...], size=size, overlap=overlap)

            for i,(coords, timg) in enumerate(tiles):
                _, tmask = mask_tiles[i]
                yi, xi, h, w = coords
                stem = f"{img_path.stem}_{yi}_{xi}"
                img_out = out_img / f"{stem}.tif"
                mask_out = out_mask / f"{stem}.tif"

                m = meta.copy()
                m.update(height=h, width=w, transform=rasterio.Affine(src.transform.a, 0, src.transform.c+xi*src.transform.a,
                                                                      0, src.transform.e, src.transform.f+yi*src.transform.e))
                save_pair(timg, tmask, m, img_out, mask_out)

                all_items.append({"image": str(img_out), "mask": str(mask_out)})

    save_tiles_manifest(out / "manifest.json", all_items)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Racine SpaceNet 3 (images & labels)")
    ap.add_argument("--out", required=True, help="Dossier de sortie tiles")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--overlap", type=int, default=64)
    args = ap.parse_args()

    # SpaceNet 3: routes (graph)
    class_map = {"road": 1}
    run(args.root, args.out, args.size, args.overlap, vector_key="type", class_map=class_map)
